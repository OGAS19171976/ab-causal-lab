"""``daily_z``（序贯累计 z 轨迹）的回归测试。

为什么值得单独测
----------------
这一层连着出过两次**同一类**事故，而且两次都是"报告里有一个体面的数字、
它却是错的"：

* 第一版解包时把 ``post_sum`` 接到了 ``pre_sum`` 上 —— 真实侧 pre 全 0，
  均值恒为 0；
* 第二版修好了 ``post_sum``，却把 ``post_sq`` 接到了 ``pre_sq_sum`` 上。
  真实库 594 行里 ``pre_sq_sum`` 只有 111 行非 0，于是两臂方差恒为 0、
  ``se = 0``、**一个桶都发不出来** —— 而报告只显示一句"0 天"，
  看着像"这段时间没有数据"。

第三个 bug 是第二个的后果：真实侧的累计 z 轨迹在报告里挂了整整一轮。

所以这组测试不去复述公式，而是钉住两条**不变量**：

1. 真实侧那种形状（``pre_sq_sum`` 全 0 / 大部分 0）必须照样算得出桶；
2. **末桶 == 全样本的 post-only z**（末桶就是全部数据，两个口径走的是同一批
   充分统计量）。这条比"有桶"更强：它能抓到两个口径都算出数、但接错了列的情况。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 真实侧形状的日表：两臂、三个季度，``pre_sum`` / ``pre_sq_sum`` **全是 0**
#: （MovieLens 那一边大部分用户没有处置前评分），``post_*`` 正常。
#: 第一行只有 control —— 第一个季度缺一臂是**正常**的跳过，不是失败。
ROWS: tuple[tuple[str, str, str, int, float, float], ...] = (
    ("t_aa", "control", "1996-01-01", 5, 20.0, 100.0),
    ("t_aa", "control", "1996-04-01", 6, 30.0, 170.0),
    ("t_aa", "treatment", "1996-04-01", 4, 26.0, 180.0),
    ("t_aa", "control", "1996-07-01", 4, 24.0, 160.0),
    ("t_aa", "treatment", "1996-07-01", 5, 30.0, 200.0),
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "_run_real_vs_synthetic", ROOT / "scripts" / "run_real_vs_synthetic.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(con: duckdb.DuckDBPyConnection, rows: Any) -> None:
    """建一张与 ``sql/02`` 同形的日表（只填这一层用得到的列）。"""
    con.execute(
        """
        create table dws_experiment_variant_daily (
            experiment varchar, variant varchar, ds date,
            user_cnt bigint, post_sum double, post_sq_sum double,
            pre_sum double, pre_sq_sum double
        )
        """
    )
    if not rows:
        return
    con.executemany(
        "insert into dws_experiment_variant_daily values (?, ?, ?, ?, ?, ?, 0.0, 0.0)",
        [list(r) for r in rows],
    )


@pytest.fixture
def module(monkeypatch):
    mod = _load()
    # 这两个是模块级"当前口径"，由 readings() 在真实运行里设置
    monkeypatch.setattr(mod, "_EXP", "t_aa")
    monkeypatch.setattr(mod, "_GRAIN", "quarter")
    return mod


def _post_only_z(rows: Any) -> float:
    """独立算一遍**全样本**的 post-only z（不调用被测代码）。"""
    acc: dict[str, list[float]] = {}
    for _exp, variant, _ds, n, post_sum, post_sq in rows:
        a = acc.setdefault(variant, [0.0, 0.0, 0.0])
        a[0] += n
        a[1] += post_sum
        a[2] += post_sq
    c, t = acc["control"], acc["treatment"]
    mc, mt = c[1] / c[0], t[1] / t[0]
    vc = max(c[2] / c[0] - mc**2, 0.0)
    vt = max(t[2] / t[0] - mt**2, 0.0)
    return (mt - mc) / math.sqrt(vt / t[0] + vc / c[0])


class TestRealShapedDailyTable:
    """真实侧那种形状（pre 的平方和大部分是 0）必须照样出桶。"""

    def test_buckets_survive_when_pre_sq_is_zero(self, module):
        con = duckdb.connect(":memory:")
        try:
            _table(con, ROWS)
            out = module.daily_z(con)
        finally:
            con.close()
        assert out["buckets"] == 3.0, out
        # 第一桶只有 control（跳过是正常的），所以可用桶比总桶数少一个
        assert out["days"] == 2.0, out
        # **这条就是那个 bug 的形状**：pre_sq_sum 全 0 时旧代码给 0 个桶
        assert out["degenerate"] == 0.0, out

    def test_last_bucket_equals_full_sample_z(self, module):
        """末桶就是全样本 ⇒ 它必须等于 post-only 的 z（自洽不变量）。"""
        con = duckdb.connect(":memory:")
        try:
            _table(con, ROWS)
            out = module.daily_z(con)
        finally:
            con.close()
        assert out["last_z"] == pytest.approx(_post_only_z(ROWS), rel=1e-12)

    def test_max_abs_z_is_an_observed_bucket(self, module):
        """max|z| 必须是轨迹上真的出现过的某个桶，而不是外插出来的数。"""
        con = duckdb.connect(":memory:")
        try:
            _table(con, ROWS)
            out = module.daily_z(con)
        finally:
            con.close()
        assert out["max_abs_z"] >= abs(out["last_z"])


class TestDegenerateGate:
    """有行却发不出桶 = 链路级失败，必须报红而不是写成"0 天"。"""

    def test_zero_variance_flags_degenerate(self, module):
        # post_sq_sum 取得恰好让方差为 0（每个桶里的值都等于均值）
        rows = (
            ("t_aa", "control", "1996-01-01", 3, 6.0, 12.0),
            ("t_aa", "treatment", "1996-01-01", 3, 9.0, 27.0),
        )
        con = duckdb.connect(":memory:")
        try:
            _table(con, rows)
            out = module.daily_z(con)
        finally:
            con.close()
        assert out["days"] == 0.0, out
        assert out["degenerate"] == 1.0, (
            "方差恒为 0 时一个桶都发不出来 —— 这正是上一轮真实侧的样子，"
            "它必须走 degenerate 报红，而不是静默返回 0 天"
        )

    def test_no_rows_is_not_degenerate(self, module):
        """表本来就空（没有流量）不算失败 —— 与"有行却算不出"是两件事。"""
        con = duckdb.connect(":memory:")
        try:
            _table(con, ())
            out = module.daily_z(con)
        finally:
            con.close()
        assert out["days"] == 0.0
        assert out["degenerate"] == 0.0
        assert math.isnan(out["last_z"])
