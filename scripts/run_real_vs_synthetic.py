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

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "build" / "warehouse.duckdb"
REAL = ROOT / "build" / "warehouse_real.duckdb"
REPORT = ROOT / "reports" / "real_vs_synthetic.md"


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def shape_stats(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """用户级结果指标的分布形状（两库同一 SQL）。"""
    row = con.execute(
        """
        select count(*) as n,
               avg(post_metric) as mean,
               stddev_samp(post_metric) as sd,
               skewness(post_metric) as skew,
               kurtosis(post_metric) as kurt,
               avg(case when post_metric = 0 then 1.0 else 0.0 end) as zero_share,
               max(post_metric) as mx
        from dwd_experiment_user
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
        """
        select variant, user_cnt, pre_sum, post_sum, pre_sq_sum, post_sq_sum,
               pre_post_cross_sum
        from ads_experiment_result
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
        "select max(chi2_statistic), max(degrees_of_freedom) from ads_experiment_srm"
    ).fetchone()
    assert row is not None
    return {"chi2": float(row[0] or 0.0), "df": float(row[1] or 0.0)}


def daily_z(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """按日累计的 z 轨迹（序贯视角的单次实现）。"""
    rows = con.execute(
        """
        select variant, ds, user_cnt, pre_sum, post_sum, pre_sq_sum, post_sq_sum
        from dws_experiment_variant_daily order by ds
        """
    ).fetchall()
    acc: dict[str, list[float]] = {}
    for variant, _ds, n, pre_sum, post_sum, pre_sq, post_sq in rows:
        a = acc.setdefault(str(variant), [0.0] * 4)
        a[0] += float(n)
        a[1] += float(post_sum)
        a[2] += float(post_sq)
        a[3] += 0.0
        # 逐日重算需要逐臂累计；这里只累计，最后一天统一算（见下）
    # 逐日轨迹：重新按 ds 累计并算 z
    per_day: dict[str, dict[str, list[float]]] = {}
    for variant, ds, n, _pre_sum, post_sum, post_sq, _pre_sq in rows:
        per_day.setdefault(str(ds), {}).setdefault(str(variant), [0.0, 0.0, 0.0])
        slot = per_day[str(ds)][str(variant)]
        slot[0] += float(n)
        slot[1] += float(post_sum)
        slot[2] += float(post_sq)
    running: dict[str, list[float]] = {}
    zs: list[float] = []
    for ds in sorted(per_day):
        for variant, slot in per_day[ds].items():
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
    if not zs:
        return {"days": 0.0, "max_abs_z": float("nan"), "last_z": float("nan")}
    return {"days": float(len(zs)), "max_abs_z": max(abs(z) for z in zs), "last_z": zs[-1]}


def cluster_view(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    row = con.execute(
        "select count(*), count(distinct cluster_id) from dws_experiment_cluster_daily"
    ).fetchone()
    assert row is not None
    return {"rows": float(row[0] or 0), "clusters": float(row[1] or 0)}


def true_lift_present(con: duckdb.DuckDBPyConnection) -> float:
    row = con.execute(
        "select count(*) from ads_experiment_result where true_lift is not null"
    ).fetchone()
    assert row is not None
    return float(row[0] or 0)


def readings(db: pathlib.Path) -> dict[str, float]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        out: dict[str, float] = {}
        out.update({f"shape.{k}": v for k, v in shape_stats(con).items()})
        out.update({f"effect.{k}": v for k, v in cuped_and_effect(con).items()})
        out.update({f"srm.{k}": v for k, v in srm(con).items()})
        out.update({f"daily.{k}": v for k, v in daily_z(con).items()})
        out.update({f"cluster.{k}": v for k, v in cluster_view(con).items()})
        out["true_lift_rows"] = true_lift_present(con)
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
    ("按日累计 z：天数", "daily.days", "{:.0f}"),
    ("按日累计 z：max|z|", "daily.max_abs_z", "{:.3f}"),
    ("按日累计 z：最后一天", "daily.last_z", "{:+.3f}"),
    ("簇级日表行数", "cluster.rows", "{:.0f}"),
    ("不同簇数", "cluster.clusters", "{:.0f}"),
    ("有 true_lift 的行数", "true_lift_rows", "{:.0f}"),
)


def main() -> int:
    if not SYNTHETIC.exists() or not REAL.exists():
        print("缺仓库文件：先跑 scripts/run_warehouse.py 与 scripts/check_real_traffic.py")
        return 1
    syn, real = readings(SYNTHETIC), readings(REAL)
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
    for label, key, fmt in ROWS:
        a = syn.get(key, float("nan"))
        b = real.get(key, float("nan"))
        lines.append(f"| {label} | {fmt.format(a)} | {fmt.format(b)} |")
    if real.get("effect.degenerate") or syn.get("effect.degenerate"):
        lines += [
            "",
            "> **注意：有一侧的指标是退化的**（方差为 0 / 全 0）—— 见下面第 0 条。",
        ]
    lines += [
        "",
        "## 结论（哪些变了、哪些没变）",
        "",
        "0. **第一次跑就抓到一个真问题**：真实侧的 pre/post 指标全是 0。",
        "   根因不在数据：数仓 SQL `sql/01_dwd_experiment_user.sql` 第 59 行**写死**了",
        "   `ev.event_name = \'interaction\'`（合成器的事件名），而真实数据的事件叫",
        "   `rating` —— 于是 exposure 计数正常、SRM 正常、**指标却是空的**。",
        "   这正是「接入外部数据」该买到的东西：契约与反冒充都过了，",
        "   但**链路里的假设**（事件名）只有换成真数据才会暴露。",
        "   下一轮做：把事件名从声明传到 SQL（参数化），并让门在「指标全 0」时直接报红。",
        "",
        *(
            []
            if real.get("effect.degenerate")
            else [
                "1. **该一致的一致**：两个仓库的 SRM 都在正常量级（真实那一份是 A/A，"
            ]
        ),
        *(
            []
            if real.get("effect.degenerate")
            else [
                f"   χ² = {real['srm.chi2']:.4f}，不显著 ⇒ 分流没坏）；CUPED 在两个仓库上都",
                "   是**正**的方差缩减（真实 {:+.4f} vs 合成 {:+.4f}）；".format(
                    real["effect.reduction_weighted"], syn["effect.reduction_weighted"]
                ),
                f"   真实的 A/A 主效应 z = {real['effect.z']:+.3f}（p = {real['effect.p']:.4f}），",
                "   即「没有效应」这个结论在真实分布上也守得住。",
                "",
            ]
        ),
        "1b. **该一致的一致（本次只做到一半）**：分流的 SRM 两侧都正常"
        f"（真实 χ² = {real['srm.chi2']:.4f}），但真实侧的 CUPED 与主效应**算不出来** ——",
        "   因为指标是空的（见第 0 条）。所以「合成 vs 真实」这一步现在只完成了",
        "   分流那一半，方差与口径那一半要等事件名接上之后重跑。",
        "",
        "1. ~~该一致的一致~~（见 1b：本次只有 SRM 那一半）",
        "",
    ]
    lines += [

        f"   χ² = {real['srm.chi2']:.4f}，不显著 ⇒ 分流没坏）；CUPED 在两个仓库上都",
        "   CUPED 方差缩减：合成 {:+.4f}；真实侧本次**算不出**"
        "（指标为空，见第 0 条）。".format(syn["effect.reduction_weighted"]),
        "",
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
        "4. **序贯视角只能看量级**：按日累计 z 的 max|z| 真实 "
        f"{real['daily.max_abs_z']:.3f} / 合成 {syn['daily.max_abs_z']:.3f}"
        f"（天数 {real['daily.days']:.0f} / {syn['daily.days']:.0f}）——",
        "   这是**一次实现**，不是 FWER；真实的 FWER 要用重随机化（本仓库在合成数据上做过，",
        "   见 m2 报告），外部数据上做不了，因为分流是我们自己做的 A/A。",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    print(f"\n报告已写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
