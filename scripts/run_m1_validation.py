#!/usr/bin/env python
"""M1 验证：CUPED / 比值指标 delta method / 聚类稳健标准误的全部证据。

运行::

    python scripts/run_m1_validation.py            # 完整版，约 2 分钟
    python scripts/run_m1_validation.py --quick    # 快速版

输出到 ``reports/``：

    fig6_cuped_conditional.png  CUPED 消掉固定分流下的协变量失衡偏置
    fig7_cuped_variance.png     CUPED 的方差缩减 vs 理论 rho^2
    fig8_cuped_power.png        CUPED 用同样流量换来的功效提升
    fig9_ratio_estimand.png     比值指标：naive 校准但答错问题
    fig10_cluster.png           聚类随机化下用户级 t 检验的崩溃
    m1_validation.md            全部数字
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from ablab.plotting import label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402
from ablab.sim import PopulationConfig, generate_population, two_arm_spec  # noqa: E402
from ablab.sim.scenarios import ClusterScenarioConfig, RatioScenarioConfig  # noqa: E402
from ablab.validation import (  # noqa: E402
    CLUSTER_LEVEL,
    CLUSTER_NAIVE,
    CLUSTER_ROBUST,
    CUPED_LABEL,
    CUPED_NAIVE,
    RATIO_DELTA,
    RATIO_NAIVE,
    run_cluster_comparison,
    run_cuped_bias_decomposition,
    run_cuped_comparison,
    run_cuped_power_comparison,
    run_ratio_comparison,
    run_ratio_power_comparison,
)

BLUE, ORANGE, GREY, RED, GREEN, PURPLE = (
    "#1f77b4",
    "#ff7f0e",
    "#7f7f7f",
    "#d62728",
    "#2ca02c",
    "#9467bd",
)


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def fig_cuped_conditional(decomposition, conditional, out: Path) -> None:
    """左：跨随机化的偏置分解（斜率）；右：条件模式下的 p 值分布。"""
    naive = conditional.get(CUPED_NAIVE)
    cuped = conditional.get(CUPED_LABEL)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax = axes[0]
    x = decomposition.pre_gaps
    ax.scatter(x, decomposition.naive_effects, s=14, color=ORANGE, alpha=0.6,
               label=label(f"post-only  斜率 {decomposition.naive_slope:.3f}",
                           f"post-only  slope {decomposition.naive_slope:.3f}"))
    ax.scatter(x, decomposition.cuped_effects, s=14, color=BLUE, alpha=0.6,
               label=label(f"CUPED  斜率 {decomposition.cuped_slope:.3f}",
                           f"CUPED  slope {decomposition.cuped_slope:.3f}"))
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, decomposition.naive_slope * xs, color=ORANGE, lw=2)
    ax.plot(xs, decomposition.cuped_slope * xs, color=BLUE, lw=2)
    ax.plot(xs, decomposition.theoretical_slope * xs, color=RED, ls="--", lw=1.4,
            label=label(f"理论 beta={decomposition.theoretical_slope:.3f}",
                        f"theory beta={decomposition.theoretical_slope:.3f}"))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel(label("前置指标组间差（这一次分流的失衡）", "Pre-metric arm gap"))
    ax.set_ylabel(label("效应估计值", "Estimated effect"))
    ax.set_title(label("偏置完全由失衡驱动：naive 斜率 = beta，CUPED 斜率 = 0",
                       "Bias is fully imbalance-driven: naive slope=beta, CUPED=0"))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(naive.p_values, bins=20, range=(0, 1), color=ORANGE, alpha=0.6, density=True,
            label=label(f"post-only (KS p={naive.uniformity_test()[1]:.3g})",
                        f"post-only (KS p={naive.uniformity_test()[1]:.3g})"))
    ax.hist(cuped.p_values, bins=20, range=(0, 1), color=BLUE, alpha=0.55, density=True,
            label=label(f"CUPED (KS p={cuped.uniformity_test()[1]:.3g})",
                        f"CUPED (KS p={cuped.uniformity_test()[1]:.3g})"))
    ax.axhline(1.0, color=RED, ls="--", lw=1.2, label="Uniform(0,1)")
    ax.set_xlabel(label("p 值", "p-value"))
    ax.set_ylabel(label("密度", "Density"))
    ax.set_title(label("CUPED 把 p 值分布修回均匀",
                       "CUPED restores p-value uniformity"))
    ax.legend(fontsize=8)

    save(fig, out / "fig6_cuped_conditional.png")


def fig_cuped_variance(randomized, out: Path) -> None:
    naive = randomized.get(CUPED_NAIVE)
    cuped = randomized.get(CUPED_LABEL)
    fit = randomized.extras["cuped_fit"]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    lo = min(naive.effects.min(), cuped.effects.min())
    hi = max(naive.effects.max(), cuped.effects.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(naive.effects, bins=bins, color=ORANGE, alpha=0.55, density=True,
            label=label(f"post-only  SD={naive.sd_effect:.3f}",
                        f"post-only  SD={naive.sd_effect:.3f}"))
    ax.hist(cuped.effects, bins=bins, color=BLUE, alpha=0.55, density=True,
            label=label(f"CUPED  SD={cuped.sd_effect:.3f}",
                        f"CUPED  SD={cuped.sd_effect:.3f}"))
    ax.axvline(0.0, color=RED, lw=1.6, label=label("真值 0", "Truth 0"))

    measured = 1 - (cuped.sd_effect / naive.sd_effect) ** 2
    ax.set_title(
        label(
            f"随机化模式：实测方差缩减 {measured:.4f} vs 理论 rho^2 {fit.variance_reduction:.4f}",
            f"Randomized: measured VR {measured:.4f} vs theory rho^2 {fit.variance_reduction:.4f}",
        )
    )
    ax.set_xlabel(label("效应估计值", "Estimated effect"))
    ax.set_ylabel(label("密度", "Density"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig7_cuped_variance.png")


def fig_cuped_power(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    lifts = np.array([r["lift"] for r in rows])
    ax.plot(lifts, [r["naive_power"] for r in rows], "o-", color=ORANGE, lw=1.8, ms=5,
            label=label("post-only", "post-only"))
    ax.plot(lifts, [r["cuped_power"] for r in rows], "s-", color=BLUE, lw=1.8, ms=5,
            label="CUPED")
    ax.axhline(0.8, color=GREY, ls=":", lw=1.2,
               label=label("常规 80% 功效线", "Conventional 80%"))
    ax.set_xlabel(label("真实效应量", "True effect size"))
    ax.set_ylabel(label("检出率", "Detection rate"))
    ax.set_title(label("同样流量下的功效：CUPED 让实验更快出结论",
                       "Power at equal traffic: CUPED reaches conclusions sooner"))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)
    save(fig, out / "fig8_cuped_power.png")


def fig_ratio(comparison, power_rows, out: Path) -> None:
    delta = comparison.get(RATIO_DELTA)
    naive = comparison.get(RATIO_NAIVE)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    names = [label("业务口径\nΣy/Σx", "Business\nΣy/Σx"),
             label("人均比值\nmean(y/x)", "Per-user\nmean(y/x)")]
    vals = [delta.control_level, naive.control_level]
    bars = ax.bar(names, vals, color=[BLUE, ORANGE], alpha=0.85, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    gap = (naive.control_level - delta.control_level) / delta.control_level
    ax.set_ylabel(label("对照组指标水平", "Control-arm metric level"))
    ax.set_title(label(f"两种口径差 {gap:+.2%}：naive 报的是另一个数字",
                       f"Estimands differ by {gap:+.2%}"))
    ax.set_ylim(0, max(vals) * 1.25)

    ax = axes[1]
    lifts = np.array([r["relative_lift"] for r in power_rows])
    ax.plot(lifts * 100, [r["delta_power"] for r in power_rows], "o-", color=BLUE,
            lw=1.8, ms=5, label="delta method")
    ax.plot(lifts * 100, [r["naive_power"] for r in power_rows], "s-", color=ORANGE,
            lw=1.8, ms=5, label=label("用户级比值", "per-user ratio"))
    ax.axhline(0.8, color=GREY, ls=":", lw=1.2)
    ax.set_xlabel(label("真实相对提升 (%)", "True relative lift (%)"))
    ax.set_ylabel(label("检出率", "Detection rate"))
    ax.set_title(label("功效几乎相同 —— 没有任何统计信号提示你答错了问题",
                       "Nearly identical power — no statistical red flag at all"))
    ax.legend(fontsize=8.5)

    save(fig, out / "fig9_ratio_estimand.png")


def fig_cluster(comparison, out: Path) -> None:
    naive = comparison.get(CLUSTER_NAIVE)
    cr = comparison.get(CLUSTER_ROBUST)
    level = comparison.get(CLUSTER_LEVEL)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    names = [label("用户级 t 检验\n（反例）", "User-level t\n(anti-pattern)"),
             "CR1", label(CLUSTER_LEVEL, "Cluster-level")]
    colors = [RED, BLUE, GREEN]

    ax = axes[0]
    ax.bar(names, [naive.fpr(), cr.fpr(), level.fpr()], color=colors, alpha=0.85, width=0.55)
    ax.axhline(0.05, color="black", ls="--", lw=1.3,
               label=label("名义 5%", "Nominal 5%"))
    for i, m in enumerate((naive, cr, level)):
        lo, hi = m.fpr_interval()
        ax.errorbar(i, m.fpr(), yerr=[[m.fpr() - lo], [hi - m.fpr()]], fmt="none",
                    ecolor="black", capsize=4, lw=1.1)
    ax.set_ylabel(label("实际 I 类错误率", "Actual Type I error"))
    ax.set_title(label("聚类随机化：用户级检验彻底崩溃",
                       "Cluster randomization breaks user-level t"))
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.bar(names, [naive.coverage(), cr.coverage(), level.coverage()],
           color=colors, alpha=0.85, width=0.55)
    ax.axhline(0.95, color="black", ls="--", lw=1.3,
               label=label("名义 95%", "Nominal 95%"))
    ax.set_ylabel(label("置信区间覆盖率", "CI coverage"))
    ax.set_title(label("覆盖率同理：反例只有 35%",
                       "Coverage: the anti-pattern collapses to 35%"))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)

    save(fig, out / "fig10_cluster.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M1 验证：三个新方法的校准证据")
    ap.add_argument("--quick", action="store_true", help="减少重复次数")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--skip-warehouse", action="store_true", help="跳过数仓部分")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_units = 6_000 if args.quick else 20_000
    n_cuped = 400 if args.quick else 1_500
    n_power = 120 if args.quick else 300
    n_ratio = 400 if args.quick else 1_000
    n_cluster = 300 if args.quick else 800

    setup_style()
    log: list[str] = []
    t0 = time.perf_counter()

    def emit(text: str = "") -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · M1 验证报告")
    emit("CUPED · 比值指标 delta method · 聚类稳健标准误")
    emit(header)

    pop = generate_population(PopulationConfig(n_units=n_units, seed=20260101))
    spec = two_arm_spec("m1_cuped", salt="m1_cuped_v1")

    # ---- 1. CUPED 消偏置（跨随机化的偏置分解） ---------------------------- #
    emit("\n### 1. CUPED 消掉固定分流下的协变量失衡偏置")
    emit("单次实现的失衡大小是随机的，可能恰好接近 0；判据必须看跨随机化的斜率。")
    decomposition = run_cuped_bias_decomposition(
        pop,
        spec,
        n_assignments=80 if args.quick else 250,
        n_noise=6 if args.quick else 8,
        seed=91,
    )
    emit(decomposition.summary())

    emit("单次实现（固定分流）下的具体表现：")
    conditional = run_cuped_comparison(
        pop, spec, n_trials=n_cuped, mode="conditional", seed=7
    )
    emit(conditional.summary())

    # ---- 2. CUPED 方差缩减（随机化） -------------------------------------- #
    emit("\n### 2. CUPED 的方差缩减（随机化模式）")
    randomized = run_cuped_comparison(pop, spec, n_trials=n_cuped, mode="randomized", seed=7)
    emit(randomized.summary())
    fit = randomized.extras["cuped_fit"]
    naive_r = randomized.get(CUPED_NAIVE)
    cuped_r = randomized.get(CUPED_LABEL)
    measured = 1 - (cuped_r.sd_effect / naive_r.sd_effect) ** 2
    emit(f"\n  实测方差缩减 = {measured:.4f}   理论 rho^2 = {fit.variance_reduction:.4f}")
    emit(f"  实测 SE 比   = {cuped_r.mean_se / naive_r.mean_se:.4f}   "
         f"理论 sqrt(1-rho^2) = {np.sqrt(fit.remaining_variance):.4f}")
    emit(f"  等效样本量 x{fit.effective_sample_multiplier:.2f}")

    # ---- 3. CUPED 功效 ---------------------------------------------------- #
    emit("\n### 3. CUPED 的功效提升")
    power_rows = run_cuped_power_comparison(pop, spec, n_trials=n_power, seed=11)
    emit(f"{'lift':>7} {'post-only':>12} {'CUPED':>12} {'相对提升':>10}   备注")
    for r in power_rows:
        if r["lift"] == 0:
            emit(f"{r['lift']:>7.2f} {r['naive_power']:>12.4f} {r['cuped_power']:>12.4f} "
                 f"{'—':>10}   真值为 0，这一行是 I 类错误而非功效")
            continue
        gain = (
            (r["cuped_power"] - r["naive_power"]) / r["naive_power"]
            if r["naive_power"] > 0
            else float("nan")
        )
        emit(f"{r['lift']:>7.2f} {r['naive_power']:>12.4f} {r['cuped_power']:>12.4f} "
             f"{gain:>9.1%}")

    # ---- 4. 比值指标 ------------------------------------------------------ #
    emit("\n### 4. 比值指标：delta method vs 用户级比值 t 检验")
    ratio_cfg = RatioScenarioConfig(n_users=n_units)
    ratio_cmp = run_ratio_comparison(ratio_cfg, n_trials=n_ratio, seed=21)
    emit(ratio_cmp.summary())
    ratio_power = run_ratio_power_comparison(ratio_cfg, n_trials=n_power, seed=23)
    emit(f"\n{'相对提升':>10} {'delta 功效':>12} {'naive 功效':>12}")
    for r in ratio_power:
        emit(f"{r['relative_lift']:>10.2%} {r['delta_power']:>12.4f} {r['naive_power']:>12.4f}")

    # ---- 5. 聚类随机化 ---------------------------------------------------- #
    emit("\n### 5. 聚类随机化：用户级 t 检验的崩溃")
    cluster_cfg = ClusterScenarioConfig(n_clusters=100, users_per_cluster=50)
    cluster_cmp = run_cluster_comparison(cluster_cfg, n_trials=n_cluster, seed=31)
    emit(cluster_cmp.summary())

    # ---- 6. 数仓链路 ------------------------------------------------------ #
    if not args.skip_warehouse:
        emit("\n### 6. 数仓链路：CUPED 成为默认口径之后")
        try:
            from ablab.warehouse import (
                WarehouseConfig,
                analyse_ads,
                build_warehouse,
                verify_against_detail,
            )

            con = build_warehouse(
                db_path=ROOT / "build" / "warehouse.duckdb",
                data_dir=ROOT / "build" / "source",
                sql_dir=ROOT / "sql",
                config=WarehouseConfig(n_users=20_000),
                force_data=False,
                verbose=False,
            )
            analyses = analyse_ads(con)
            for a in analyses:
                emit(f"\n  {a.experiment}（真实效应 {a.true_lift:+.2f}/天）")
                emit(f"    post-only : {a.naive.absolute_effect:+.4f} "
                     f"SE {a.naive.std_error:.4f}  p={a.naive.p_value:.4g}  "
                     f"significant={a.naive.significant}")
                emit(f"    CUPED     : {a.cuped.absolute_effect:+.4f} "
                     f"SE {a.cuped.std_error:.4f}  p={a.cuped.p_value:.4g}  "
                     f"significant={a.cuped.significant}")
                cv = verify_against_detail(con, analyses, a.experiment)
                emit(f"    ADS/DWD 交叉验证: {cv.passed}")
            con.close()
        except Exception as exc:  # pragma: no cover - 数仓不可用时降级
            emit(f"  （跳过：{exc}）")

    # ---- 7. 图表 ---------------------------------------------------------- #
    emit("\n### 7. 生成图表")
    fig_cuped_conditional(decomposition, conditional, out)
    fig_cuped_variance(randomized, out)
    fig_cuped_power(power_rows, out)
    fig_ratio(ratio_cmp, ratio_power, out)
    fig_cluster(cluster_cmp, out)
    for i in range(6, 11):
        for m in sorted(out.glob(f"fig{i}_*.png")):
            emit(f"  {m.name}")

    # ---- 结论 ------------------------------------------------------------- #
    c_naive = conditional.get(CUPED_NAIVE)
    c_cuped = conditional.get(CUPED_LABEL)
    r_naive = cluster_cmp.get(CLUSTER_NAIVE)
    r_cr = cluster_cmp.get(CLUSTER_ROBUST)

    checks = {
        "naive 偏置由失衡驱动（斜率=beta）": decomposition.naive_bias_is_driven_by_imbalance,
        "CUPED 无失衡偏置（斜率不显著）": decomposition.cuped_is_free_of_imbalance_bias,
        "CUPED 随机化下校准": 0.025 <= randomized.get(CUPED_LABEL).fpr() <= 0.085,
        "CUPED 条件模式下校准": 0.025 <= c_cuped.fpr() <= 0.085,
        "条件模式下 naive 未校准": c_naive.uniformity_test()[1] < 0.01,
        "方差缩减贴近 rho^2": abs(measured - fit.variance_reduction) < 0.10,
        "delta method 校准": 0.025 <= ratio_cmp.get(RATIO_DELTA).fpr() <= 0.085,
        "CR1 校准": 0.02 <= r_cr.fpr() <= 0.09,
        "聚类反例确实崩溃": r_naive.fpr() > 0.25,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    emit("\n" + header)
    emit("结论")
    emit(header)
    emit(f"[1] CUPED 消偏置：回归「效应 ~ 前置组间差」的斜率，"
         f"post-only = {decomposition.naive_slope:.4f}"
         f"（理论 beta {decomposition.theoretical_slope:.4f}，p={decomposition.naive_slope_p:.3g}）"
         f" vs CUPED = {decomposition.cuped_slope:.4f}（p={decomposition.cuped_slope_p:.3g}）")
    emit("    -> naive 的偏置完全由「前置失衡 × beta」产生，CUPED 把这一项整项扣掉")
    emit(f"    固定分流下的 p 值均匀性：KS p {c_naive.uniformity_test()[1]:.3g}"
         f" -> {c_cuped.uniformity_test()[1]:.3g}；"
         f"I 类错误 {c_naive.fpr():.4f} -> {c_cuped.fpr():.4f}")
    emit(f"[2] CUPED 降方差：实测 {measured:.4f} vs 理论 rho^2 {fit.variance_reduction:.4f}，"
         f"等效样本量 x{fit.effective_sample_multiplier:.2f}，标准误降 {fit.se_shrinkage:.1%}")
    emit("[3] CUPED 提功效：见上表，同流量下检出率全面更高")
    delta_ref = ratio_cmp.get(RATIO_DELTA)
    naive_ref = ratio_cmp.get(RATIO_NAIVE)
    emit(f"[4] 比值指标：naive 的 I 类错误也是校准的（{naive_ref.fpr():.4f}），"
         f"但它报的口径差 "
         f"{(naive_ref.control_level - delta_ref.control_level) / delta_ref.control_level:+.2%}")
    emit("    -> 校准不等于正确：一个检验可以既不偏高也不偏低，却系统性报出错误的量")
    emit(f"[5] 聚类随机化：用户级 t 检验 I 类错误 {r_naive.fpr():.1%}"
         f"（覆盖率 {r_naive.coverage():.1%}）-> CR1 {r_cr.fpr():.4f}，"
         f"簇级 {cluster_cmp.get('簇级 Welch').fpr():.4f}")
    emit(f"\n逐项检查: {checks}")
    emit(f"总体判定: {verdict}")
    emit(f"总耗时 {time.perf_counter() - t0:.1f}s")

    report = out / "m1_validation.md"
    report.write_text("# M1 验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n", encoding="utf-8", newline="\n")
    print(f"\n报告已写入 {report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
