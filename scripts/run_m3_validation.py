#!/usr/bin/env python
"""M3 验证：观察数据因果推断的全部证据。

运行::

    python scripts/run_m3_validation.py            # 完整版，约 3 分钟
    python scripts/run_m3_validation.py --quick    # 快速版

M3 的验证台和前三个阶段的重心不同：随机化没了，所以不再问"我的数算得对不对"，
而是问"**假设被打破时我会错得多离谱，而且我会不会察觉**"。

输出到 ``reports/``：

    fig16_twfe_negative_weights.png  TWFE 的隐式权重与真实效应反向
    fig17_staggered_estimators.png   TWFE vs CS：符号翻转率 100% vs 0%
    fig18_pretrend_blindspot.png     平行趋势检验的盲区
    fig19_scm_placebo.png            合成控制的安慰剂推断
    fig20_sensitivity.png            翻转点分析
    m3_validation.md                 全部数字
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from ablab.causal import (  # noqa: E402
    StaggeredPanelConfig,
    callaway_santanna,
    generate_scm_scenario,
    generate_staggered_panel,
    placebo_inference,
    pretrend_test,
    sun_abraham,
    synthetic_control,
    trend_sensitivity,
    twfe,
    twfe_decomposition,
)
from ablab.plotting import bin_edges, label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402
from ablab.validation import (  # noqa: E402
    run_aggregation_variance_audit,
    run_pretrend_audit,
    run_scm_audit,
    run_sensitivity_audit,
    run_staggered_estimator_comparison,
)

BLUE, ORANGE, GREY, RED, GREEN, PURPLE = (
    "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd",
)

#: M3 的主角配置：交错处置 + 队列间效应异质。
#: 数据里**每一个单元每一期的真实效应都是正的**，所以负号一定是估计量的问题。
HEADLINE = StaggeredPanelConfig(
    n_units=800,
    n_periods=7,
    cohorts=(2, 4),
    cohort_weights=(0.5, 0.5),
    never_treated_share=0.04,
    effects=(1.0, 2.0, 3.0, 4.0),
    cohort_effect_multiplier=(1.0, 0.25),
    noise_sd=0.5,
)


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def fig_negative_weights(panel, truth, decomposition, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    tw = twfe(panel).absolute_effect
    cs = callaway_santanna(panel).overall.absolute_effect
    vals = [truth.overall_att, tw, cs]
    names = [label("真值", "Truth"), "TWFE", label("Callaway-\nSant'Anna", "Callaway-\nSant'Anna")]
    colors = [GREEN, RED, BLUE]
    bars = ax.bar(names, vals, color=colors, alpha=0.85, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(label("ATT 估计", "Estimated ATT"))
    ax.set_title(label("效应处处为正，TWFE 给出负号",
                       "All effects positive; TWFE goes negative"))

    ax = axes[1]
    post = decomposition.post_cells
    eff = decomposition.true_effect[post]
    w = decomposition.weight[post] * 1e4
    neg = w < 0
    ax.scatter(eff[~neg], w[~neg], s=6, color=BLUE, alpha=0.35,
               label=label(f"正权重 ({(~neg).mean():.0%})", f"positive weight ({(~neg).mean():.0%})"))
    ax.scatter(eff[neg], w[neg], s=6, color=RED, alpha=0.45,
               label=label(f"负权重 ({neg.mean():.0%})", f"negative weight ({neg.mean():.0%})"))
    ax.axhline(0, color="black", lw=1.0)
    edges = np.unique(eff)
    mids, means = [], []
    for e in edges:
        m = eff == e
        mids.append(e)
        means.append(w[m].mean())
    ax.plot(mids, means, "o-", color=ORANGE, lw=2, ms=6,
            label=label("各效应水平的平均权重", "mean weight by effect level"))
    ax.set_xlabel(label("真实效应", "True effect"))
    ax.set_ylabel(label("TWFE 隐式权重 (×10⁻⁴)", "Implicit TWFE weight"))
    ax.set_title(
        label(f"权重与真实效应相关 {decomposition.weight_effect_correlation():+.3f}",
              f"corr(weight, effect) = {decomposition.weight_effect_correlation():+.3f}")
    )
    ax.legend(fontsize=8)

    save(fig, out / "fig16_twfe_negative_weights.png")


def fig_staggered_estimators(comparison, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    lo = min(comparison.twfe_estimates.min(), comparison.truths.min()) - 0.3
    hi = max(comparison.twfe_estimates.max(), comparison.truths.max()) + 0.3
    bins = bin_edges(np.linspace(lo, hi, 40))

    ax.hist(comparison.twfe_estimates, bins=bins, color=RED, alpha=0.65,
            label=label(f"TWFE（符号翻转 {comparison.twfe_sign_flip_rate:.0%}）",
                        f"TWFE (sign flip {comparison.twfe_sign_flip_rate:.0%})"))
    ax.hist(comparison.cs_estimates, bins=bins, color=BLUE, alpha=0.65,
            label=label(f"CS（偏置 {comparison.cs_bias:+.3f}）",
                        f"CS (bias {comparison.cs_bias:+.3f})"))
    ax.axvline(comparison.truths.mean(), color=GREEN, lw=2.2,
               label=label(f"真值 {comparison.truths.mean():.3f}",
                           f"Truth {comparison.truths.mean():.3f}"))
    ax.axvline(0.0, color="black", ls=":", lw=1.4)
    ax.set_xlabel(label("ATT 估计", "Estimated ATT"))
    ax.set_ylabel(label("频次", "Count"))
    ax.set_title(label(f"同一批数据：TWFE 偏置 {comparison.twfe_bias:+.2f}，"
                       f"CS 偏置 {comparison.cs_bias:+.3f}",
                       f"TWFE bias {comparison.twfe_bias:+.2f} vs CS {comparison.cs_bias:+.3f}"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig17_staggered_estimators.png")


def fig_pretrend_blindspot(audit, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = [
        label("平行趋势成立\n(size)", "Parallel trends\n(size)"),
        label("队列专属趋势\n(处置前可见)", "Cohort trends\n(visible pre)"),
        label("处置后才分岔\n(处置前不可见)", "Diverges post\n(invisible pre)"),
    ]
    vals = [audit.size, audit.power_trend_violation, audit.power_post_divergence]
    colors = [GREY, GREEN, RED]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
    ax.axhline(audit.alpha, color="black", ls="--", lw=1.4,
               label=label(f"名义 {audit.alpha}", f"Nominal {audit.alpha}"))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10)
    ax.set_ylabel(label("平行趋势检验的拒绝率", "Pre-trend test rejection rate"))
    ax.set_ylim(0, 1.08)
    ax.set_title(label("检验对「处置后才分岔」几乎没有功效 —— 假设破了它却照常通过",
                       "The test is blind to post-treatment divergence"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig18_pretrend_blindspot.png")


def fig_scm_placebo(data, main, placebo, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    T = data.outcome.shape[1]
    periods = np.arange(1, T + 1)
    ax.plot(periods, data.outcome[0], "o-", color=RED, ms=3.5, lw=1.8,
            label=label("处置单元", "Treated"))
    ax.plot(periods, main.synthetic_pre.tolist() + main.synthetic_post.tolist(), "s--",
            color=BLUE, ms=3.5, lw=1.8, label=label("合成对照", "Synthetic control"))
    ax.axvline(data.n_pre + 0.5, color=GREY, ls=":", lw=1.5,
               label=label("处置时点", "Treatment"))
    ax.set_xlabel(label("期数", "Period"))
    ax.set_ylabel(label("结果变量", "Outcome"))
    ax.set_title(label(f"合成控制：ATT = {main.att:+.3f}",
                       f"Synthetic control: ATT = {main.att:+.3f}"))
    ax.legend(fontsize=8)

    ax = axes[1]
    ratios = np.sort(placebo.placebo_ratios)
    ax.hist(ratios, bins=18, color=GREY, alpha=0.8,
            label=label("安慰剂单元", "Placebo units"))
    ax.axvline(placebo.treated_ratio, color=RED, lw=2.4,
               label=label(f"真实单元 {placebo.treated_ratio:.2f}", f"Treated {placebo.treated_ratio:.2f}"))
    ax.set_xlabel(label("处置后/处置前 RMSE 比值", "Post/Pre RMSE ratio"))
    ax.set_ylabel(label("频次", "Count"))
    ax.set_title(label(f"排名 p 值 = {placebo.p_value:.3f}（排名 {placebo.rank}/{placebo.n_placebos + 1}）",
                       f"Rank p-value = {placebo.p_value:.3f}"))
    ax.legend(fontsize=8)

    save(fig, out / "fig19_scm_placebo.png")


def fig_sensitivity(audit, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    vals = [audit.actual_violation, audit.median_breakdown]
    names = [label("实际注入的违背", "Actual violation"),
             label("翻转点中位数 δ*", "Median breakdown δ*")]
    bars = ax.bar(names, vals, color=[ORANGE, BLUE], alpha=0.85, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center", va="bottom",
                fontsize=10)
    ax.set_ylabel(label("每期趋势差", "Per-period trend gap"))
    ax.set_title(
        label(f"要推翻结论需要 {audit.median_breakdown / audit.actual_violation:.1f} 倍于实际违背的偏离",
              f"Breakdown is {audit.median_breakdown / audit.actual_violation:.1f}x the actual violation")
    )
    ax.text(
        0.5, 0.92,
        label(f"结论扛住的比例 = {audit.robust_share:.0%}",
              f"Conclusion holds in {audit.robust_share:.0%} of trials"),
        transform=ax.transAxes, ha="center", fontsize=10, color=GREY,
    )
    save(fig, out / "fig20_sensitivity.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M3 验证：观察数据因果推断")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_est = 40 if args.quick else 200
    n_pre = 80 if args.quick else 300
    n_scm = 30 if args.quick else 100
    n_sens = 40 if args.quick else 150
    # 聚合方差审计比别的贵（每次仿真都要算全部 ATT(g,t) 及其影响函数），
    # 但它是这一节唯一的证据来源，所以不放进 --quick 里省掉。
    n_aggvar = 40 if args.quick else 200

    setup_style()
    log: list[str] = []
    t0 = time.perf_counter()

    def emit(text: str = "") -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · M3 验证报告")
    emit("DiD（含交错处置）· 合成控制 · 平行趋势敏感性")
    emit(header)
    emit("")
    emit("M0–M2 的结论靠随机化站住；M3 没有这个靠山。")
    emit("所以验证台的问题从「我的数算得对不对」变成")
    emit("「**假设被打破时我会错得多离谱，而且我会不会察觉**」。")

    # ---- 1. 主角：交错处置下的 TWFE ---------------------------------------- #
    emit("\n### 1. 交错处置：效应处处为正，TWFE 却给出负号")
    panel, truth = generate_staggered_panel(HEADLINE)
    emit(panel.summary())
    emit(truth.summary())
    dec = twfe_decomposition(panel, truth)
    emit("")
    emit(dec.summary())

    post = dec.post_cells
    eff = dec.known_true_effect[post]
    w = dec.weight[post]
    emit("")
    emit("按真实效应看平均隐式权重：")
    emit(f"{'真实效应':>10} {'单元-期数':>10} {'平均权重(×1e4)':>16}")
    for e in np.unique(eff):
        m = eff == e
        emit(f"{e:>10.2f} {int(m.sum()):>10} {w[m].mean() * 1e4:>16.3f}")

    tw = twfe(panel)
    cs = callaway_santanna(panel)
    emit("")
    emit(f"TWFE = {tw.absolute_effect:+.4f}   "
         f"CS = {cs.overall.absolute_effect:+.4f}   真值 = {truth.overall_att:+.4f}")
    emit(pretrend_test(panel).message)

    # ---- 2. 批量对照 ------------------------------------------------------- #
    emit(f"\n### 2. 批量仿真：TWFE vs Callaway-Sant'Anna（{n_est} 次）")
    comparison = run_staggered_estimator_comparison(n_trials=n_est, seed=0)
    emit(comparison.summary())

    # ---- 2.5 交互加权（Sun-Abraham 的聚合）+ 聚合方差 ----------------------- #
    emit("\n### 2.5 交互加权聚合，以及**聚合方差**用哪种算法")
    sa = sun_abraham(panel)
    emit(sa.summary())
    emit("")
    emit("  与 CS 的关系：在饱和设定下，IW 聚合与 CS 的事件研究**点估计相同**"
         "（因为分格估计用的是同一套 2×2）。")
    emit(f"    实测：CS 整体 ATT = {cs.overall.absolute_effect:+.4f}，"
         f"IW = {sa.overall.absolute_effect:+.4f}")
    emit("  真正的差别在**方差的聚合方式**上，下面这组 H0 仿真把它量出来：")
    emit("")
    agg_var = run_aggregation_variance_audit(n_trials=n_aggvar, seed=0)
    emit(agg_var.summary())
    emit("")
    emit("  这条差别的意义：整体 ATT 是若干 ATT(g,t) 的加权和，而它们共用对照单元、")
    emit("  相邻队列还共用基准期 —— 相关性非负。按独立合成算 SE 会低估它，")
    emit("  于是「名义 5% 的检验」在 H0 下拒绝得远多于 5%。")
    emit("  **这个 bug 曾经真的在库里**：CS 的整体 SE 就是这么算的，")
    emit("  而 `_att_influence` 的文档里早就写着「独立合成会严重低估方差」——")
    emit("  那条教训当时只用在了 lead 的联合检验上，聚合这一步漏掉了。")
    emit("  现在两条路径都改用影响函数合成，并把「旧算法会是多少」作为诊断一并报出。")

    # ---- 3. 平行趋势检验的盲区 --------------------------------------------- #
    emit(f"\n### 3. 平行趋势检验：能发现什么、发现不了什么（{n_pre} 次/场景）")
    pretrend = run_pretrend_audit(n_trials=n_pre, seed=0)
    emit(pretrend.summary())

    # ---- 4. 合成控制 ------------------------------------------------------- #
    emit(f"\n### 4. 合成控制与安慰剂推断（{n_scm} 次/场景）")
    scm_audit = run_scm_audit(n_trials=n_scm, seed=0)
    emit(scm_audit.summary())
    emit("")
    demo = generate_scm_scenario(effect=3.0, seed=7)
    demo_main = synthetic_control(demo)
    demo_placebo = placebo_inference(demo)
    emit("单个示例：")
    emit(demo_main.summary())
    emit(demo_placebo.summary())

    # ---- 5. 敏感性分析 ----------------------------------------------------- #
    emit(f"\n### 5. 平行趋势敏感性：翻转点分析（{n_sens} 次）")
    sens_audit = run_sensitivity_audit(n_trials=n_sens, seed=0)
    emit(sens_audit.summary())
    emit("")
    emit("单次示例：")
    emit(trend_sensitivity(callaway_santanna(panel), panel).summary())

    # ---- 6. 图表 ----------------------------------------------------------- #
    emit("\n### 6. 生成图表")
    fig_negative_weights(panel, truth, dec, out)
    fig_staggered_estimators(comparison, out)
    fig_pretrend_blindspot(pretrend, out)
    fig_scm_placebo(demo, demo_main, demo_placebo, out)
    fig_sensitivity(sens_audit, out)
    for i in range(16, 21):
        for m in sorted(out.glob(f"fig{i}_*.png")):
            emit(f"  {m.name}")

    # ---- 结论 -------------------------------------------------------------- #
    checks = {
        "数据里效应处处为正": bool(dec.known_true_effect[dec.post_cells].min() > 0),
        "TWFE 出现符号翻转": comparison.twfe_sign_flip_rate > 0.5,
        "TWFE 权重与效应负相关": dec.weight_effect_correlation() < -0.3,
        "CS 基本无偏": abs(comparison.cs_bias) < 0.05,
        "CS 符号翻转率极低": comparison.cs_sign_flip_rate < 0.02,
        "平行趋势检验 size 合理": pretrend.size < 0.10,
        "检验对可见违背有功效": pretrend.power_trend_violation > 0.8,
        "检验对不可见违背无功效": pretrend.blind_spot,
        "合成控制安慰剂假阳性率接近名义": abs(scm_audit.false_positive_rate - 0.05) < 0.06,
        "敏感性分析给出有限翻转点": np.isfinite(sens_audit.median_breakdown),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    emit("\n" + header)
    emit("结论")
    emit(header)
    emit(f"[1] 交错处置下 TWFE：偏置 {comparison.twfe_bias:+.4f}，"
         f"**符号翻转率 {comparison.twfe_sign_flip_rate:.0%}**")
    emit(f"    机制：{dec.negative_post_weight_share:.0%} 的处置后单元-期拿到**负权重**，"
         f"它们承载了 {dec.negative_weight_effect_share:.0%} 的真实效应，"
         f"corr(权重, 效应) = {dec.weight_effect_correlation():+.3f}")
    emit(f"[2] Callaway-Sant'Anna：偏置 {comparison.cs_bias:+.4f}，"
         f"符号翻转率 {comparison.cs_sign_flip_rate:.0%}")
    emit(f"[3] 平行趋势检验：size {pretrend.size:.4f}，"
         f"对可见违背功效 {pretrend.power_trend_violation:.3f}，"
         f"对**处置后才分岔**功效 {pretrend.power_post_divergence:.3f}")
    emit("    -> 假设已经破了，检验却照常通过。这是 M1「校准不等于正确」的姊妹命题：")
    emit("       **检验通过不等于假设成立**。")
    emit(f"[4] 合成控制安慰剂：无效应时假阳性率 {scm_audit.false_positive_rate:.4f}"
         f"（名义 0.05），有效应时检出率 {scm_audit.power:.4f}")
    emit(f"[5] 敏感性分析：实际违背 {sens_audit.actual_violation:+.3f}/期，"
         f"翻转点中位数 {sens_audit.median_breakdown:+.3f}/期，"
         f"结论扛住 {sens_audit.robust_share:.0%}")
    emit(f"\n逐项检查: {checks}")
    emit(f"总体判定: {verdict}")
    emit(f"总耗时 {time.perf_counter() - t0:.1f}s")

    report = out / "m3_validation.md"
    report.write_text("# M3 验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n", encoding="utf-8", newline="\n")
    print(f"\n报告已写入 {report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
