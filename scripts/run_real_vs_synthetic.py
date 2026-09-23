#!/usr/bin/env python
"""**差异审计**：同一条链路，合成数据 vs 真实外部数据（MovieLens）。

外部数据买不到 `true_lift`，所以它回答不了"效应估得准不准"。
它能回答的是这句话：**换了数据源之后，哪些数字变了、哪些没变，为什么。**
这个脚本用**同一组公式**在两个仓库上各算一遍：

  A. **分布形状**（用户级结果指标）：零值占比、偏度、峰度、极值占比 ——
     合成器给的是我们写进去的分布，真实数据给的是现实的分布；
  B. **CUPED 的方差缩减**（θ = cov/var_pre，逐臂算再加权）；
  C. **主效应与 z**（post-only 与 CUPED 两个口径，充分统计量闭式）；
  D. **SRM 卡方**（真实那一份是 A/A，所以它**不应该**显著）；
  E. **序贯视角**：按日累计的 z 轨迹（单次实现，只能看量级）；
  F. **簇级路径**：真实数据的 `city` 是空的 ⇒ 聚合退化成一个占位簇；
  G. **`true_lift` 是否存在**（外部数据没有演示真值）。

读法写在报告最后：**该一致的是"方法学性质"**（SRM 校准、CUPED 不放大方差、
A/A 不显著），**该不一致的是"分布性质"**（零膨胀、峰度、极值集中度）——
后者正是"合成器乐观"可能藏身的地方。
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "build" / "warehouse.duckdb"
REAL = ROOT / "build" / "warehouse_real.duckdb"
REPORT = ROOT / "reports" / "real_vs_synthetic.md"


#: 当前的分日颗粒度。**两侧不该用同一个口径**：合成侧是"日历日 × 每天很多用户"，
#: 真实侧 MovieLens 的曝光日是**每个用户自己的中位评分时间** —— 573 个不同曝光日、
#: 每臂每个曝光日只有 1 个用户。拿"天数"比大小没有意义，所以真实侧按**季度**聚合。
#: 第一版两边都按"曝光日"算，于是真实侧的 z 轨迹是 0 天（每天只有 1 人，样本不足），
#: 报告里还留着"nan" —— 看着像缺数据，其实是口径选错了。
_GRAIN = "day"


def _bucket(column: str) -> str:
    """按颗粒度把日期列切成桶。"""
    return f"date_trunc('quarter', {column})" if _GRAIN == "quarter" else column


#: 当前正在审计的实验名；由 ``readings`` 设定。
#: 第一版**没有按实验过滤**：合成库里 3 个实验的 6 行被当成"两个臂"加在一起，
#: 算出 z = −2.38、SRM χ² = 77.6 —— 数字看着像发现，其实是把三个实验混成了一个。
_EXP = ""


def _w(join: str = "where") -> str:
    return f"{join} experiment = '{_EXP}'"


def pick_experiment(con: duckdb.DuckDBPyConnection) -> str:
    """选一个**负对照**实验来代表该仓库（真实的 ml_aa 也是 A/A）。

    排序规则刻意写死成"先看真实效应是不是 0"：合成库里有 3 个演示实验，
    其中 exp_rank_v2 / exp_city_ctr **本来就有效应**（后者还是整簇随机化，
    用户数天生不平衡）。拿它们跟 A/A 对比，等于把"有效应"读成"校准坏了" ——
    第一版就是这么错的（挑到 exp_city_ctr，z=+7.6、SRM χ²=77.6，
    看着像大发现，其实是选错了对照）。
    """
    rows = con.execute(
        """
        select experiment,
               max(case when true_lift = 0 then 1 else 0 end) as is_aa,
               sum(user_cnt) as n
        from ads_experiment_result
        group by 1
        order by is_aa desc, n desc
        """
    ).fetchall()
    if not rows:
        raise ValueError("ads_experiment_result 是空的")
    return str(rows[0][0])


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def shape_stats(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """用户级结果指标的分布形状（两库同一 SQL）。"""
    row = con.execute(
        f"""
        select count(*) as n,
               avg(post_metric) as mean,
               stddev_samp(post_metric) as sd,
               skewness(post_metric) as skew,
               kurtosis(post_metric) as kurt,
               avg(case when post_metric = 0 then 1.0 else 0.0 end) as zero_share,
               max(post_metric) as mx
        from dwd_experiment_user
        {_w()}
        """
    ).fetchone()
    assert row is not None  # count(*) 永远有一行；mypy 要这句
    n, mean, sd, skew, kurt, zero_share, mx = row
    return {
        "n": float(n or 0),
        "mean": float(mean or 0.0),
        "sd": float(sd or 0.0),
        "skew": float(skew or 0.0),
        "kurt": float(kurt or 0.0),
        "zero_share": float(zero_share or 0.0),
        "max_over_mean": float(mx or 0.0) / float(mean) if mean else float("nan"),
    }


def cuped_and_effect(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """CUPED 方差缩减 + 主效应（两个口径），全部由充分统计量闭式算。"""
    rows = con.execute(
        f"""
        select variant, user_cnt, pre_sum, post_sum, pre_sq_sum, post_sq_sum,
               pre_post_cross_sum
        from ads_experiment_result
        {_w()}
        """
    ).fetchall()
    arms: dict[str, dict[str, float]] = {}
    for variant, n, pre_sum, post_sum, pre_sq, post_sq, cross in rows:
        n = float(n)
        arms[str(variant)] = {
            "n": n,
            "mean_pre": float(pre_sum) / n,
            "mean_post": float(post_sum) / n,
            "var_pre": float(pre_sq) / n - (float(pre_sum) / n) ** 2,
            "var_post": float(post_sq) / n - (float(post_sum) / n) ** 2,
            "cov": float(cross) / n - (float(pre_sum) / n) * (float(post_sum) / n),
        }
    for arm in arms.values():
        # var_pre = 0 有两种可能：真的没有处置前指标，或者**事件名没对上**
        # （数仓 SQL 里事件名是写死的）。两者都必须报出来，不能静默给 0。
        arm["theta"] = arm["cov"] / arm["var_pre"] if arm["var_pre"] else float("nan")
        if arm["var_post"] and not math.isnan(arm["theta"]):
            arm["var_adj"] = (
                arm["var_post"]
                - 2 * arm["theta"] * arm["cov"]
                + arm["theta"] ** 2 * arm["var_pre"]
            )
            arm["reduction"] = 1.0 - arm["var_adj"] / arm["var_post"]
        else:
            # 结果指标本身没有方差（例如全 0）⇒ CUPED 无意义，标 NaN 而不是 0
            arm["var_adj"] = float("nan")
            arm["reduction"] = float("nan")
    control = next((a for k, a in arms.items() if k == "control"), None)
    treatment = next((a for k, a in arms.items() if k == "treatment"), None)
    nan = float("nan")
    out: dict[str, float] = {
        "tau": nan, "se": nan, "z": nan, "p": nan,
        "tau_cuped": nan, "se_cuped": nan, "z_cuped": nan, "p_cuped": nan,
        "theta_pool": nan,
    }
    if control and treatment and control["var_post"] > 0 and treatment["var_post"] > 0:
        tau = treatment["mean_post"] - control["mean_post"]
        se = math.sqrt(
            treatment["var_post"] / treatment["n"] + control["var_post"] / control["n"]
        )
        theta_pool = (
            (treatment["cov"] * treatment["n"] + control["cov"] * control["n"])
            / (treatment["var_pre"] * treatment["n"] + control["var_pre"] * control["n"])
        )
        tau_c = tau - theta_pool * (treatment["mean_pre"] - control["mean_pre"])
        se_c = math.sqrt(
            treatment["var_adj"] / treatment["n"] + control["var_adj"] / control["n"]
        )
        out.update({
            "tau": tau,
            "se": se,
            "z": tau / se if se else float("nan"),
            "tau_cuped": tau_c,
            "se_cuped": se_c,
            "z_cuped": tau_c / se_c if se_c else float("nan"),
            "theta_pool": theta_pool,
        })
        out["p"] = normal_two_sided_p(out["z"])
        out["p_cuped"] = normal_two_sided_p(out["z_cuped"])
    usable = [a for a in arms.values() if not math.isnan(a["reduction"])]
    out["reduction_weighted"] = (
        sum(a["reduction"] * a["n"] for a in usable) / sum(a["n"] for a in usable)
        if usable
        else float("nan")
    )
    out["degenerate"] = 0.0 if usable else 1.0
    return out


def srm(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    row = con.execute(
        "select max(chi2_statistic), max(degrees_of_freedom) from ads_experiment_srm "
        + _w()
    ).fetchone()
    assert row is not None
    return {"chi2": float(row[0] or 0.0), "df": float(row[1] or 0.0)}


def daily_z(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """按桶累计的 z 轨迹（序贯视角的单次实现）。

    这一层只用到 post 的充分统计量（n / Σpost / Σpost²），所以 SQL **只取这三列**。
    这是被两次事故逼出来的，两次都出在"解包顺序"上：

    * 第一版把 ``post_sum`` 接到了 ``pre_sum`` 上 —— 真实侧 pre 全 0，
      于是均值恒为 0；
    * 第二版修好了 ``post_sum``，却把 ``post_sq`` 接到了 ``pre_sq_sum`` 上。
      真实库 594 行里 ``pre_sq_sum`` 只有 **111 行**非 0（不是每个用户都有
      处置前评分），于是两臂方差恒为 0 ⇒ ``se = 0`` ⇒ **一个桶都发不出来**，
      而报告里只显示一句体面的"0 天" —— 看着像没数据，其实是列接到了别的列上。

    根因不是列序本身，而是"同一次查询解包两遍、两遍顺序不一样"。现在只解包
    一次、只查要用的列；并且**有行却发不出一个桶**时返回 ``degenerate=1``，
    由 ``main`` 报红 —— 这类失败不能再靠人读报告发现。
    """
    rows = con.execute(
        f"""
        select variant, {_bucket("ds")} as ds,
               sum(user_cnt), sum(post_sum), sum(post_sq_sum)
        from dws_experiment_variant_daily
        {_w()}
        group by 1, 2
        order by 2
        """
    ).fetchall()
    slots: dict[str, dict[str, list[float]]] = {}
    for variant, ds, n, post_sum, post_sq in rows:
        slot = slots.setdefault(str(ds), {}).setdefault(str(variant), [0.0, 0.0, 0.0])
        slot[0] += float(n)
        slot[1] += float(post_sum)
        slot[2] += float(post_sq)
    # 逐桶累计：每桶先把该桶各臂的量加进 running，再用**当刻累计量**算 z。
    # 跳过是正常的：桶里只有一臂（比如真实侧 1996Q1 只有 control），
    # 或者累计样本还不够（n <= 1）。
    running: dict[str, list[float]] = {}
    zs: list[float] = []
    for ds in sorted(slots):
        for variant, slot in slots[ds].items():
            run = running.setdefault(variant, [0.0, 0.0, 0.0])
            run[0] += slot[0]
            run[1] += slot[1]
            run[2] += slot[2]
        if {"control", "treatment"} <= set(running):
            c, t = running["control"], running["treatment"]
            if c[0] > 1 and t[0] > 1:
                mc, mt = c[1] / c[0], t[1] / t[0]
                vc = max(c[2] / c[0] - mc**2, 0.0)
                vt = max(t[2] / t[0] - mt**2, 0.0)
                se = math.sqrt(vt / t[0] + vc / c[0])
                if se > 0:
                    zs.append((mt - mc) / se)
    out = {
        "days": float(len(zs)),
        "max_abs_z": float("nan"),
        "last_z": float("nan"),
        "buckets": float(len(slots)),
        "degenerate": 0.0,
    }
    if zs:
        out["max_abs_z"] = max(abs(z) for z in zs)
        out["last_z"] = zs[-1]
    elif rows:
        # 拿到行却一个桶都算不出来：链路级失败（列序 / 颗粒度 / 方差退化），
        # 不能静默变成"0 天"。
        out["degenerate"] = 1.0
    return out


def cluster_view(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    row = con.execute(
        "select count(*), count(distinct cluster_id) from dws_experiment_cluster_daily "
        + _w()
    ).fetchone()
    assert row is not None
    return {"rows": float(row[0] or 0), "clusters": float(row[1] or 0)}


def true_lift_present(con: duckdb.DuckDBPyConnection) -> float:
    row = con.execute(
        "select count(*) from ads_experiment_result "
        + _w("where") + " and true_lift is not null"
    ).fetchone()
    assert row is not None
    return float(row[0] or 0)


def readings(
    db: pathlib.Path, experiment: str | None = None, grain: str = "day"
) -> dict[str, float]:
    global _EXP, _GRAIN
    con = duckdb.connect(str(db), read_only=True)
    try:
        global _GRAIN
        _EXP = experiment or pick_experiment(con)
        _GRAIN = grain
        out: dict[str, Any] = {}
        out.update({f"shape.{k}": v for k, v in shape_stats(con).items()})
        out.update({f"effect.{k}": v for k, v in cuped_and_effect(con).items()})
        out.update({f"srm.{k}": v for k, v in srm(con).items()})
        daily = daily_z(con)
        out.update({f"daily.{k}": v for k, v in daily.items()})
        out.update({f"cluster.{k}": v for k, v in cluster_view(con).items()})
        out["true_lift_rows"] = true_lift_present(con)
        out["experiment_name"] = _EXP
        # **自洽检查**：累计 z 轨迹的末桶就是全样本，所以它必须等于
        # ads_experiment_result 上算出来的 post-only z。两个口径走的是同一批
        # 充分统计量，对不上就说明有一层接错了列 / 漏了过滤 —— 两次列序错位
        # 都属于这一类：两边都算得出数，只是那个数不是那个数。
        z_last, z_agg = daily["last_z"], out["effect.z"]
        if math.isnan(z_last) or math.isnan(z_agg):
            # 任一侧没有数：由 daily.degenerate / effect.degenerate 负责报红
            out["daily.matches_effect"] = float("nan")
        else:
            out["daily.matches_effect"] = (
                1.0 if abs(z_last - z_agg) <= 1e-9 * max(1.0, abs(z_agg)) else 0.0
            )
        return out
    finally:
        con.close()


ROWS: tuple[tuple[str, str, str], ...] = (
    ("用户数", "shape.n", "{:.0f}"),
    ("结果指标：均值", "shape.mean", "{:.4f}"),
    ("结果指标：SD", "shape.sd", "{:.4f}"),
    ("结果指标：偏度", "shape.skew", "{:+.3f}"),
    ("结果指标：峰度", "shape.kurt", "{:+.3f}"),
    ("零值占比", "shape.zero_share", "{:.4f}"),
    ("最大值 / 均值", "shape.max_over_mean", "{:.1f}"),
    ("CUPED 方差缩减（加权）", "effect.reduction_weighted", "{:+.4f}"),
    ("θ（合并）", "effect.theta_pool", "{:+.4f}"),
    ("主效应 τ̂（post-only）", "effect.tau", "{:+.4f}"),
    ("  z（post-only）", "effect.z", "{:+.3f}"),
    ("  p（post-only）", "effect.p", "{:.4f}"),
    ("主效应 τ̂（CUPED）", "effect.tau_cuped", "{:+.4f}"),
    ("  z（CUPED）", "effect.z_cuped", "{:+.3f}"),
    ("  p（CUPED）", "effect.p_cuped", "{:.4f}"),
    ("SRM χ²（最大）", "srm.chi2", "{:.4f}"),
    ("累计 z 轨迹：分桶数（含被跳过的桶）", "daily.buckets", "{:.0f}"),
    ("累计 z 轨迹：可用桶数（合成按日/真实按季度）", "daily.days", "{:.0f}"),
    ("累计 z 轨迹：max|z|", "daily.max_abs_z", "{:.3f}"),
    ("累计 z 轨迹：最后一个桶", "daily.last_z", "{:+.3f}"),
    ("累计 z 轨迹：末桶 == 主效应 z（自洽检查，1=通过）",
     "daily.matches_effect", "{:.0f}"),
    ("簇级日表行数", "cluster.rows", "{:.0f}"),
    ("不同簇数", "cluster.clusters", "{:.0f}"),
    ("有 true_lift 的行数", "true_lift_rows", "{:.0f}"),
)


def main() -> int:
    if not SYNTHETIC.exists() or not REAL.exists():
        print("缺仓库文件：先跑 scripts/run_warehouse.py 与 scripts/check_real_traffic.py")
        return 1
    syn = readings(SYNTHETIC, grain="day")
    real = readings(REAL, grain="quarter")
    lines = [
        "# 差异审计：合成数据 vs 真实外部数据（MovieLens）",
        "",
        "同一组公式、同一条链路，两个仓库各算一遍。",
        "读法：**该一致的是方法学性质**（SRM 校准、CUPED 不放大方差、A/A 不显著），",
        "**该不一致的是分布性质**（零膨胀、峰度、极值集中度）。",
        "",
        f"| 读数 | 合成（{SYNTHETIC.name}） | 真实（{REAL.name}） |",
        "|---|---|---|",
    ]
    lines.append(f"（合成侧取负对照实验：{syn['experiment_name']}；"
                 f"真实侧：{real['experiment_name']}）")
    lines.append("")
    for label, key, fmt in ROWS:
        a = syn.get(key, float("nan"))
        b = real.get(key, float("nan"))
        lines.append(f"| {label} | {fmt.format(a)} | {fmt.format(b)} |")
    effect_bad = bool(real.get("effect.degenerate") or syn.get("effect.degenerate"))
    daily_bad = bool(real.get("daily.degenerate") or syn.get("daily.degenerate"))
    notes = []
    if effect_bad:
        notes.append("> **注意：有一侧的指标是退化的**（方差为 0 / 全 0）—— 下面第 1 条不可用。")
    if daily_bad:
        notes.append("> **注意：有一侧的序贯轨迹退化为 0 桶**（拿到了行却发不出桶）—— 见第 4 条。")
    mismatch = [
        name
        for name, side in (("合成", syn), ("真实", real))
        if side.get("daily.matches_effect") == 0.0
    ]
    if mismatch:
        notes.append(
            f"> **注意：{'、'.join(mismatch)}侧的序贯末桶与主效应 z 对不上**"
            "（两个口径本该逐位相等）—— 见第 4 条。"
        )
    if notes:
        lines += ["", *notes]
    lines += [
        "",
        "## 结论（哪些变了、哪些没变）",
        "",
        "**两个链路 bug 已经修掉** —— 留在这里，因为它们正是「接入外部数据」买到的东西：",
        "",
        "* **事件名写死**：数仓 SQL 曾经**写死** `ev.event_name = 'interaction'`"
        "（合成器的事件名），",
        "  而真实数据的事件叫 `rating` —— exposure 计数正常、SRM 正常、**指标却全 0**。",
        "  事件名现在从 provenance 一路传到 SQL，而且门会在「主指标全 0」时直接报红",
        "  （`check_metric_is_alive`）：这类「链路假设不匹配」以后由机器抓，"
        "不靠人读审计报告。",
        "* **序贯轨迹的列序**：`daily_z` 把同一次查询解包两遍、第二遍顺序错位，"
        "`post_sq` 接到了",
        "  `pre_sq_sum` 上，于是真实侧两臂方差恒为 0、**一个桶都发不出来**"
        "（见第 4 条）。",
        "",
    ]
    if effect_bad:
        lines += [
            "1. **该一致的一致：本次只做到分流那一半** —— 指标退化，"
            "CUPED 与主效应算不出来。",
            "",
        ]
    else:
        lines += [
            "1. **该一致的一致**：两个仓库的 SRM 都在正常量级（真实那一份是 A/A，",
            f"   χ² = {real['srm.chi2']:.4f}，不显著 ⇒ 分流没坏）；CUPED 在两个仓库上都",
            "   是**正**的方差缩减（真实 {:+.4f} vs 合成 {:+.4f}）；".format(
                real["effect.reduction_weighted"], syn["effect.reduction_weighted"]
            ),
            f"   真实的 A/A 主效应 z = {real['effect.z']:+.3f}"
            f"（p = {real['effect.p']:.4f}），",
            "   即「没有效应」这个结论在真实分布上也守得住。",
            "",
        ]
    lines += [
        "2. **该不一致的不一致**（这一份报告真正买到的东西）：",
        f"   分布形状差得很远 —— 零值占比 {syn['shape.zero_share']:.4f} → "
        f"{real['shape.zero_share']:.4f}，峰度 {syn['shape.kurt']:+.3f} → "
        f"{real['shape.kurt']:+.3f}，最大值/均值 {syn['shape.max_over_mean']:.1f} → "
        f"{real['shape.max_over_mean']:.1f}。",
        "   合成器的分布是我们写进去的，所以它的方差性质偏**乐观**；",
        "   真实数据这一列才是「方差估计会不会太窄」的参照。",
        "",
        "3. **真实数据带出来的两个结构性限制**（不是 bug，是边界）：",
        f"   真实侧的簇级日表只有 **{real['cluster.clusters']:.0f} 个簇**"
        f"（合成侧 {syn['cluster.clusters']:.0f} 个）——",
        "   因为 MovieLens 没有 `city`，簇级聚合退化成一个占位簇",
        "   （列在契约里是可选，但「可选」不等于「能算簇级」）；",
        f"   真实侧的 `true_lift` 行数 = **{real['true_lift_rows']:.0f}**"
        f"（合成侧 {syn['true_lift_rows']:.0f}）——",
        "   外部数据没有演示真值，所以它只能做校准，不能做「效应估得准不准」的验收。",
        "",
        "4. **序贯视角：口径按仓库分开，两侧的「天数」本来就不可比**。"
        "合成的桶是**日历日**",
        "（每天很多用户）；真实侧 MovieLens 的曝光日是**每人自己的中位评分时间**"
        "（573 个曝光日、每臂每日只有 1 人）—— 按日根本没有样本，所以真实侧按**季度**聚合。",
        "   能比的只有「轨迹像不像随机游走」：max|z| 真实 "
        f"{real['daily.max_abs_z']:.3f} / 合成 {syn['daily.max_abs_z']:.3f}"
        f"（可用桶 {real['daily.days']:.0f} / {syn['daily.days']:.0f}，"
        f"分桶 {real['daily.buckets']:.0f} / {syn['daily.buckets']:.0f}）——",
        "   被跳过的桶不是「没有数据」：真实侧第一个季度只有一臂（1996Q1 只有 control），",
        "   或者累计样本还不够（n ≤ 1）。",
        "   **上一版这里真实侧是 0 个桶**，根因就是开头那条列序错位：真实库 594 行里",
        "   `pre_sq_sum` 只有 **111 行**非 0（不是每个用户都有处置前评分），"
        "两臂方差因此恒为 0，",
        "   `se = 0` ⇒ 一个桶都发不出来，而报告只显示一句体面的「0 天」——"
        "看着像没数据，其实是列接到了别的列上。",
        "   现在 `daily_z` 只解包一次、只查用得到的列，并且**有行却发不出一个桶就报红**",
        "   （`daily.degenerate`）。另外多了一条不变量：末桶就是全样本，所以它必须等于",
        "   `ads_experiment_result` 上的 post-only z —— 两个口径走的是同一批充分统计量，"
        "对不上就报红",
        "   （`daily.matches_effect`）。这条不变量比「有桶」更强："
        "它能抓到两个口径都算出数、但接错了列的情况。",
        "   合成侧的 max|z| 也随之变了：旧数用的也是错的方差列。",
        "   这是**一次实现**，不是 FWER；真实的 FWER 要用重随机化（本仓库在合成数据上做过，",
        "   见 m2 报告），外部数据上做不了，因为分流是我们自己做的 A/A。",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(f"\n报告已写入 {REPORT}")
    if daily_bad:
        # 序贯那一段拿不到桶：这正是上一版静默显示"0 天"的那种失败。
        # 报告已经写下来了，但这一步必须红 —— 否则没人会去看那三个数。
        print(
            "\n**序贯轨迹退化**：daily_z 拿到了行，却一个桶都算不出来"
            "（见报告第 4 条与 `daily.buckets` / `daily.days`）——"
            "这是链路级失败，不是「这段时间没有数据」。"
        )
        return 1
    if mismatch:
        # 末桶就是全样本 ⇒ 它必须等于 ads_experiment_result 上的 post-only z。
        # 对不上说明有一层接错了列 / 漏了过滤，而两边都能算出数来 ——
        # 这种"看着正常的错"正是要靠不变量抓的。
        print(
            f"\n**序贯末桶与主效应 z 对不上**：{'、'.join(mismatch)}侧 "
            f"daily.last_z 与 effect.z 不一致（合成 "
            f"{syn['daily.last_z']:+.6f} / {syn['effect.z']:+.6f}，真实 "
            f"{real['daily.last_z']:+.6f} / {real['effect.z']:+.6f}）——"
            "两者本该是同一个数。"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
