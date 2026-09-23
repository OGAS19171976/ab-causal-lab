"""**限速**：固定窗口 + ``429`` + ``Retry-After``。

为什么是固定窗口，而不是令牌桶
------------------------------
令牌桶（可以是漏桶/滑动窗口）更平滑，但它多两个参数（容量与补充速率），
而这两个参数在"单机实验平台、用户是同事"的场景里没有真实的调参依据 ——
调不出来的参数最后只会变成一拍脑袋的默认值。固定窗口只有一个参数
（每分钟多少次），能被解释、也能被验证（``X-RateLimit-Remaining`` 看得到）。

**它的已知缺陷写在明处**：固定窗口在窗口边界上允许**两倍突发**
（前 60 秒的最后一毫秒和后 60 秒的第一毫秒各打满一次）。要挡的是
"跑飞的脚本"与"猜 token"，两倍突发不影响这两件事；要把突发也压平，
才值得上令牌桶。

为什么可以放在内存里
--------------------
单进程 uvicorn（见 `run_platform.py`）+ 单机平台，所以一个进程内的字典就够。
**多进程/多副本部署时这条会退化成"每副本各自限速"** —— 那时需要把窗口
挪到共享存储（Redis 之类）。这一条是边界，不是待办：现在的部署形态就是单进程。

内存上界也只是**跨窗口**的：``_MAX_KEYS`` 触发时清掉的是上一个窗口的残留；
如果同一个窗口内就涌进几万个不同的键（伪造源 IP 的洪泛），字典在这一分钟内
仍然会长到那个量级。硬上界需要共享存储里的 LRU —— 与本段上一条同一个理由。

为什么时钟可以注入
------------------
"限速生效"这句话要用测试证明，而测试里 **sleep 60 秒不可接受**。
``clock`` 默认 ``time.monotonic``（不受系统时间调整影响），
测试传一个自己控制的函数即可确定性地推进时间。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

__all__ = ["RateLimitDecision", "WindowRateLimiter"]


@dataclass(frozen=True)
class RateLimitDecision:
    """一次限速判断的结果。``retry_after`` 只在被拒时有意义（秒）。"""

    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class WindowRateLimiter:
    """按 ``key`` 各自计数的固定窗口限速器（线程安全）。

    ``key`` 由调用方决定（这里用 ``user:<id>`` 或 ``ip:<host>``）——
    限速器不猜身份，身份是 API 层的事。
    """

    #: 内存上界：键多到一定程度就把上一个窗口的残留清掉。
    #: 为什么需要它：键是用户/客户端，正常只有几十个；但如果有人用海量伪造
    #: 源 IP 打过来，字典会**无限增长** —— 那样限速器自己成了内存泄漏。
    _MAX_KEYS = 4096

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        window_seconds: float = 60.0,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute 必须为正数")
        self.per_minute = int(per_minute)
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[int, int]] = {}

    def check(self, key: str, *, weight: int = 1) -> RateLimitDecision:
        now = self._clock()
        window = int(now // self.window_seconds)
        with self._lock:
            stamped, used = self._hits.get(key, (window, 0))
            if stamped != window:
                used = 0
            used += weight
            self._hits[key] = (window, used)
            if len(self._hits) > self._MAX_KEYS:
                self._hits = {
                    k: v for k, v in self._hits.items() if v[0] == window
                }
        remaining = max(self.per_minute - used, 0)
        if used > self.per_minute:
            # +1：``Retry-After: 0`` 会让客户端立刻重试，等于没限
            retry_after = int(self.window_seconds - (now % self.window_seconds)) + 1
            return RateLimitDecision(False, self.per_minute, 0, retry_after)
        return RateLimitDecision(True, self.per_minute, remaining, 0)
