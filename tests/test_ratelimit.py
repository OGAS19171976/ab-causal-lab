"""``WindowRateLimiter`` 的单元测试。

时钟注入是这里唯一的技术点：默认的 ``time.monotonic`` 没法在测试里推进，
而"窗口滚过去之后额度恢复"这句声明**必须能被验证** —— 否则它只是注释。
"""

from __future__ import annotations

import pytest

from ablab.platform.ratelimit import WindowRateLimiter


class FakeClock:
    """可控时钟：``advance`` 推进秒数。"""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_allows_up_to_the_limit_then_refuses():
    clock = FakeClock()
    limiter = WindowRateLimiter(3, clock=clock)
    assert [limiter.check("u").allowed for _ in range(3)] == [True] * 3
    refused = limiter.check("u")
    assert not refused.allowed
    assert refused.remaining == 0
    assert refused.limit == 3


def test_remaining_counts_down():
    limiter = WindowRateLimiter(3, clock=FakeClock())
    assert [limiter.check("u").remaining for _ in range(3)] == [2, 1, 0]


def test_window_rollover_restores_the_budget():
    clock = FakeClock()
    limiter = WindowRateLimiter(1, clock=clock)
    assert limiter.check("u").allowed
    assert not limiter.check("u").allowed
    clock.advance(60)  # 下一个窗口
    assert limiter.check("u").allowed


def test_retry_after_points_at_the_next_window():
    clock = FakeClock(start=10.0)
    limiter = WindowRateLimiter(1, clock=clock)
    limiter.check("u")
    # 窗口 [0,60)，现在 t=10 -> 还剩 50 秒；+1 是为了避免 Retry-After: 0
    # 让客户端"立刻重试"，那等于没限
    assert limiter.check("u").retry_after == 51
    clock.advance(49.0)  # t=59
    assert limiter.check("u").retry_after == 2
    clock.advance(1.0)  # t=60：新窗口
    assert limiter.check("u").allowed


def test_window_is_aligned_to_absolute_time():
    """窗口按 ``now // 60`` 对齐，不是"从第一次请求起算 60 秒"。

    两条路都能用，但选对齐的这条是因为它**没有每键状态**（不需要记"这个键的
    窗口什么时候开始"），代价是窗口边界上的两倍突发 —— 那一条写在模块文档里。
    """
    clock = FakeClock(start=30.0)
    limiter = WindowRateLimiter(1, clock=clock)
    assert limiter.check("u").allowed
    clock.advance(29.9)  # t=59.9，仍在 [0,60)
    assert not limiter.check("u").allowed
    clock.advance(0.1)  # t=60.0，进入下一个窗口
    assert limiter.check("u").allowed


def test_keys_are_independent():
    limiter = WindowRateLimiter(1, clock=FakeClock())
    assert limiter.check("a").allowed
    assert not limiter.check("a").allowed
    assert limiter.check("b").allowed, "一个人的额度不该吃掉另一个人的"


def test_weight_can_be_greater_than_one():
    limiter = WindowRateLimiter(5, clock=FakeClock())
    assert limiter.check("u", weight=5).allowed
    assert not limiter.check("u").allowed


def test_non_positive_limit_is_refused():
    with pytest.raises(ValueError):
        WindowRateLimiter(0)


def test_stale_windows_are_pruned():
    """跨窗口的残留会被清掉 —— 否则键只增不减。

    **边界（如实测）**：清理只对"上一个窗口的残留"生效。如果**同一个窗口内**
    就来了几万个不同的键（伪造源 IP 的洪泛），字典在那一分钟内仍然会涨到那个量级 ——
    硬上界需要共享存储里的 LRU，那不在"单进程单机"的部署形态里。
    这条测试钉的是"清理确实发生"，不是"内存有硬上界"。
    """
    clock = FakeClock()
    limiter = WindowRateLimiter(10, clock=clock)
    for i in range(limiter._MAX_KEYS + 50):
        limiter.check(f"old{i}")
    assert len(limiter._hits) > limiter._MAX_KEYS
    clock.advance(60)  # 进入新窗口，老的都成了残留
    limiter.check("fresh")
    limiter.check("another")
    assert len(limiter._hits) < limiter._MAX_KEYS
    assert set(limiter._hits) == {"fresh", "another"}
