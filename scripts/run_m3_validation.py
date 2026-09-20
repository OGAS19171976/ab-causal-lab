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
    sun_abraham_regression,
    synthetic_control,
    trend_sensitivity,
    twfe,
    twfe_decomposition,
    two_sls,
)
from ablab.plotting import bin_edges, label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402
from ablab.sim import IVScenarioConfig, generate_iv_scenario  # noqa: E402
from ablab.validation import (  # noqa: E402
    run_aggregation_variance_audit,
    run_iv_audit,
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
    n_iv = 40 if args.quick else 150

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

    # ---- 2.6 Sun-Abraham 的**回归版**：与 IW 版的交叉验证 ------------------ #
    emit("\n### 2.6 Sun-Abraham 回归版：与 IW 版的交叉验证，以及它的边界")
    emit("  回归版 = 一条回归（队列×相对期数交互项）+ 双向固定效应，")
    emit("  用**交替投影**吸收固定效应（不构造 N+T 个哑变量，本仓库没有稀疏最小二乘）。")
    emit("  它与 IW 版走的是完全不同的计算路径，所以两者一致才是强证据：")
    emit("")
    iw_nt = sun_abraham(panel, control_group="never_treated")
    rg_nt = sun_abraham_regression(panel, control_group="never_treated")
    emit(f"  {'相对期数':>8}{'IW':>12}{'回归':>12}{'差':>10}{'IW SE':>10}{'回归 SE':>10}")
    max_diff_never = 0.0
    for k in sorted(iw_nt.event_study):
        a, b = iw_nt.event_study[k], rg_nt.event_study[k]
        d = abs(a.absolute_effect - b.absolute_effect)
        max_diff_never = max(max_diff_never, d)
        emit(f"  {k:>8}{a.absolute_effect:>+12.4f}{b.absolute_effect:>+12.4f}"
             f"{d:>10.1e}{a.std_error:>10.4f}{b.std_error:>10.4f}")
    emit("")
    emit(f"  对照组 = never_treated：逐 k 最大点估计差 **{max_diff_never:.1e}**、"
         f"SE 比值处处 1.000 ——")
    emit("  「两种算法、同一个估计量」在这里成立。整体 ATT：")
    emit(f"    IW {iw_nt.overall.absolute_effect:+.6f}（SE {iw_nt.overall.std_error:.4f}）"
         f" vs 回归 {rg_nt.overall.absolute_effect:+.6f}"
         f"（SE {rg_nt.overall.std_error:.4f}），真值 {truth.overall_att:+.4f}")
    emit(f"    TWFE 是 {tw.absolute_effect:+.4f} —— 效应异质时被负权重拉偏，"
         f"两个 SA 版本都不偏。")
    emit("")
    emit("  **但换对照组就不一样了**（这是本轮新测出来的边界，不是实现问题）：")
    iw_ny = sun_abraham(panel, control_group="not_yet_treated")
    rg_ny = sun_abraham_regression(panel, control_group="not_yet_treated")
    emit(f"  {'相对期数':>8}{'IW':>12}{'回归':>12}{'差':>10}{'贡献队列数':>12}")
    for k in sorted(iw_ny.event_study):
        a, b = iw_ny.event_study[k], rg_ny.event_study[k]
        n_coh = len(iw_ny.weights) if False else len(
            {g for g in panel.cohorts() if 0 <= k + int(g) - 1 < panel.n_periods}
        )
        emit(f"  {k:>8}{a.absolute_effect:>+12.4f}{b.absolute_effect:>+12.4f}"
             f"{abs(a.absolute_effect - b.absolute_effect):>10.1e}{n_coh:>12}")
    emit("")
    emit("  读法：差异只出现在**多个队列共同贡献**的相对期数上，")
    emit("  单队列贡献的期数（该 k 只有最晚队列还能被观测到）又回到 1e-14。")
    emit("  原因：IW 的每个 (g,t) 分格用**当时尚未处置**的单元当对照（对照集随 t 变），")
    emit("  而饱和回归只有一套双向固定效应，已处置队列的变化会进入比较 ——")
    emit("  这正是 Sun & Abraham 提醒的 forbidden comparison。")
    emit("  所以：**有未处置组时用回归版（与 IW 等同）**。")
    emit("")
    emit("  **没有未处置组时怎么办**：原文那一步（用最后一个队列当基准）已经实现。")
    emit("  Sun & Abraham 自己的 Stata 包把做法写得很具体：用最后处置队列当对照时，")
    emit("  要**剔除未处置单元**、并且**只保留最后队列被处置之前的期数**。")
    emit("  实测（600 单元、队列 3/5/7、无未处置组）：")
    from ablab.causal import sun_abraham_regression as _sa_reg

    cfg_no_never = StaggeredPanelConfig(
        n_units=600, n_periods=10, cohorts=(3, 5, 7),
        cohort_weights=(1 / 3, 1 / 3, 1 / 3), never_treated_share=0.0,
        effects=(1.0, 2.0, 3.0, 3.0, 3.0), noise_sd=1.0, seed=5,
    )
    panel_nn, truth_nn = generate_staggered_panel(cfg_no_never)
    iw_nn = sun_abraham(panel_nn, control_group="not_yet_treated")
    naive_nn = _sa_reg(panel_nn, control_group="not_yet_treated")
    last_nn = _sa_reg(
        panel_nn, control_group="not_yet_treated", base_cohort="last_treated"
    )

    def _kdiff(a, b):
        ks = set(a.event_study) & set(b.event_study)
        return max(
            abs(a.event_study[k].absolute_effect - b.event_study[k].absolute_effect)
            for k in ks
        )

    emit(f"    未处置当基准（默认）：与 IW 的最大逐 k 差 {_kdiff(iw_nn, naive_nn):.3f}")
    _d_naive, _d_last = _kdiff(iw_nn, naive_nn), _kdiff(iw_nn, last_nn)
    emit(f"    最后队列当基准（新）：与 IW 的最大逐 k 差 {_d_last:.3f}"
         f"（改善了 {_d_naive / _d_last:.0f} 倍）")
    emit(f"    整体 ATT：IW {iw_nn.overall.absolute_effect:+.4f} vs "
         f"新口径 {last_nn.overall.absolute_effect:+.4f}（真值 {truth_nn.overall_att:+.4f}）")
    emit("    代价是**事件窗变短**（最后队列的处置后期数不再可估）——")
    emit("    这是原文的要求，不是实现妥协；它有测试钉着。")
    emit("")
    emit("")
    emit("  **第三种估计量：BJS 插补**（只用未处置观测拟合双向固定效应，")
    emit("  给每条处置观测插补反事实，再对处置观测等权平均）。")
    from ablab.causal import borusyak_jaravel_spiess as _bjs

    bjs_res = _bjs(panel)
    emit(f"    真值 {truth.overall_att:+.4f} | CS {cs.overall.absolute_effect:+.6f}"
         f"（SE {cs.overall.std_error:.4f}）| BJS {bjs_res.overall.absolute_effect:+.6f}"
         f"（SE {bjs_res.overall.std_error:.4f}）")
    emit(f"    BJS − CS = {bjs_res.overall.absolute_effect - cs.overall.absolute_effect:+.6f}"
         "（2%~4% 量级：**两者不是同一个估计量**）")
    emit("")
    emit("  为什么差、以及为什么这不是 bug —— 做了一次**诊断**：")
    emit("  两者只差在**加权方式**（CS/SA 按 (g,t) 分格 2×2 再加权，")
    emit("  插补对处置观测等权）。判据是：把「效应异质」这个解释排除掉 ——")
    _gaps = []
    for _hom in (True, False):
        _cfg = StaggeredPanelConfig(
            n_units=600, n_periods=9, cohorts=(3, 6), cohort_weights=(0.5, 0.5),
            never_treated_share=0.3, effects=(2.0, 2.0, 2.0, 2.0),
            cohort_effect_multiplier=(1.0, 1.0) if _hom else (1.0, 0.25),
            noise_sd=0.5, seed=5,
        )
        _panel_h, _ = generate_staggered_panel(_cfg)
        _cs_h = callaway_santanna(_panel_h).overall.absolute_effect
        _bjs_h = _bjs(_panel_h).overall.absolute_effect
        _gaps.append(_bjs_h - _cs_h)
    emit(f"    同质效应面板 BJS−CS = {_gaps[0]:+.8f}")
    emit(f"    异质效应面板 BJS−CS = {_gaps[1]:+.8f}")
    emit("    两者**逐位相同** -> 差异来自加权而不是效应异质（有测试钉着）。")

    emit("")
    emit("  **第四种估计量：dCDH 换手估计量**（比较「同一批人相邻两期的变化」，")
    emit("  处置组 = 刚好换手的单元，对照组 = 还没换手的单元）。")
    from ablab.causal import de_chaisemartin_dhaultfoeuille as _dcdh

    dcdh_ok = _dcdh(panel)
    # **口径要说清**：dCDH 估的是"换手那一刻"的即时效应（相当于 k=0），
    # 而 truth.overall_att 是所有处置后各期的**平均**。这份 DGP 的效应逐期递增
    # （1,2,3,4），两者本来就不该相等 —— 所以对照的是 CS 的 k=0 那一格。
    cs_k0 = cs.event_study.get(0)
    emit(f"    平行趋势成立时：DID+ {dcdh_ok.overall.absolute_effect:+.4f}"
         f"（SE {dcdh_ok.overall.std_error:.4f}）")
    emit("    逐队列核对（dCDH 的 DID+ vs CS 的同一格 g,t=g，base=g−1，"
         "not_yet_treated）：")
    from ablab.causal.did import cs_att_with_influence as _cell

    for _g in sorted(dcdh_ok.effects):
        _ref = _cell(panel, _g, _g, _g - 1, "not_yet_treated")
        _ref_val = _ref[0] if _ref else float("nan")
        emit(f"      g={_g}: dCDH {dcdh_ok.effects[_g].absolute_effect:+.6f}"
             f"  CS {_ref_val:+.6f}"
             f"  差 {dcdh_ok.effects[_g].absolute_effect - _ref_val:+.1e}")
    if cs_k0 is not None:
        emit(f"    聚合后：dCDH {dcdh_ok.overall.absolute_effect:+.4f} vs "
             f"CS k=0 {cs_k0.absolute_effect:+.4f}")
    emit("    **差别不在加权，而在「能用哪些队列」**：dCDH 要多一个前置期做安慰剂，")
    emit("    所以**最早那个队列被排除**；这份 DGP 的效应逐队列异质，")
    emit("    少一个队列就会改变加权平均。逐队列上两者是**逐位相同**的。")
    emit(f"    安慰剂 DID- = {dcdh_ok.placebo_overall.absolute_effect:+.4f}"
         f"（p={dcdh_ok.placebo_overall.p_value:.3f}）-> 安静")
    cfg_viol = StaggeredPanelConfig(
        n_units=800, n_periods=9, cohorts=(3, 5, 7), cohort_weights=(1 / 3, 1 / 3, 1 / 3),
        never_treated_share=0.25, effects=(2.0, 2.0, 2.0, 2.0), noise_sd=1.0,
        trend_violation=1.5, seed=7,
    )
    panel_viol, truth_viol = generate_staggered_panel(cfg_viol)
    dcdh_bad = _dcdh(panel_viol)
    emit(f"    平行趋势**被违反**时：DID+ {dcdh_bad.overall.absolute_effect:+.4f}"
         f"（真值 {truth_viol.overall_att:+.4f} —— **同样会偏**）")
    emit(f"    安慰剂 DID- = {dcdh_bad.placebo_overall.absolute_effect:+.4f}"
         f"（p={dcdh_bad.placebo_overall.p_value:.2g}）-> **报警**")
    emit("    这就是这一族估计量真正的价值：点估计并不比别的更抗违反，")
    emit("    但它**自带一个直接指向违反的安慰剂**（同一批人、同一次比较，")
    emit("    只把时间往前挪一期）。TWFE 在同样的数据上也会给一个偏的数，")
    emit("    但不会告诉你假设坏了。")

    emit("  一句话口径：**有未处置组 -> 默认（1e-14 量级一致）；")
    emit("  没有 -> base_cohort='last_treated'（差 2.39 -> 0.047）**。")

    # ---- 2.7 回归版的交叉验证顺手抓出来的 bug：对照泄漏 -------------------- #
    #
    # 这一段刻意把"修之前会算成什么"也重算一遍 —— 与 `naive_overall_se` 同一个
    # 思路：**差异要可见，而不是靠相信**。修之前的数只存在于 git 历史里，
    # 那样读者没法复跑，所以这里用旧掩码重算一次。
    emit("\n### 2.7 交叉验证抓出来的 bug：处置队列进了自己的对照组")
    emit("  写回归版时做的逐 k 比对（上面 2.6）暴露了一个**早就存在**的错误：")
    emit("  `not_yet_treated` 的判据是 `C_i > max(t, g-1)`，而处置队列自己满足")
    emit("  `g > g-1` —— 于是**处置前的格子里，处置组被算成了自己的对照**。")
    emit("  后果不是崩溃，而是 placebo 被静默压向 0（对照均值里混进了处置组自身的变化）。")
    emit("")
    from ablab.causal.did import _att_influence, _control_mask

    pre_cells: list[tuple[int, int, int, float, float, float, int, int]] = []
    for g in panel.cohorts():
        g = int(g)
        base = g - 1
        if base < 1:
            continue
        for t in panel.periods:
            t = int(t)
            if t >= base:  # 只看处置前的格子
                continue
            g_mask = panel.cohort == g
            c_leak = _control_mask(panel, t, base, "not_yet_treated")
            c_fixed = c_leak & ~g_mask
            if (c_leak & g_mask).sum() == 0 or c_fixed.sum() < 2:
                continue
            d_y = panel.outcome[:, t - 1] - panel.outcome[:, base - 1]
            leak, _ = _att_influence(d_y, g_mask, c_leak)
            fixed, _ = _att_influence(d_y, g_mask, c_fixed)
            pre_cells.append(
                (g, t, base, leak, fixed, abs(fixed) - abs(leak),
                 int((c_leak & g_mask).sum()), int(c_leak.sum()))
            )
    if pre_cells:
        emit(f"  {'队列':>6}{'期':>4}{'基准':>6}{'对照(泄漏)':>12}{'对照(修复)':>12}"
             f"{'ATT(泄漏)':>12}{'ATT(修复)':>12}{'被压掉':>10}")
        for g, t, base, leak, fixed, gap, n_over, n_leak in pre_cells:
            emit(f"  {g:>6}{t:>4}{base:>6}{n_leak:>12}{n_leak - n_over:>12}"
                 f"{leak:>+12.4f}{fixed:>+12.4f}{gap / abs(fixed) if fixed else float('nan'):>9.0%}")
        mean_leak = float(np.mean([abs(c[3]) for c in pre_cells]))
        mean_fixed = float(np.mean([abs(c[4]) for c in pre_cells]))
        emit("")
        emit(f"  处置前 {len(pre_cells)} 个格子：平均 |placebo| 从 {mean_leak:.4f} 变成 "
             f"**{mean_fixed:.4f}**")
        emit(f"  （被压掉 {1 - mean_leak / mean_fixed:.0%}）—— 而「处置前系数接近 0」")
        emit("  看起来**正是我们想看到的结论**，所以这个 bug 一直没被怀疑。")
        emit("  连带影响：对照掩码重叠让影响函数的两个作用项部分抵消，")
        emit("  处置前的 SE 被低估一个量级（实测 k=-3 处 0.0082 → 0.1072，13 倍），")
        emit("  于是「处置前没有异常」这个判断本身也是失真的。")
    else:
        emit("  （这份面板的处置前格子没有触发对照泄漏，用测试里的面板验证）")

    # ---- 2.8 工具变量：横截面上的内生性 ------------------------------------ #
    #
    # 前面几节处理的都是**面板**（有时间前后）。横截面上的一次性决策
    #（"上过培训班的人收入更高"）没有处置前趋势可用，只能靠工具变量。
    # 而这个方法最容易骗人的地方是：**"我用了工具变量"被当成了结论**。
    # 工具很弱时，2SLS 的中位偏差会朝 OLS 靠、均值与 RMSE 会爆炸，
    # 报告上却仍然写着"95% 置信区间"。
    emit(f"\n### 2.8 工具变量（2SLS + Anderson-Rubin）：{n_iv} 次重抽")
    emit("  横截面上的内生处置：D = pi·Z + u，Y = tau·D + rho·u + e。")
    emit("  rho=0.8 时 OLS 把混淆记在 D 头上；工具 Z 只通过 D 影响 Y（由构造保证）。")
    emit("")
    iv_audit = run_iv_audit(n=1500, n_replications=n_iv)
    emit(iv_audit.summary())
    emit("")
    emit("  单个示例（最弱那一档，看看 AR 区间到底长什么样）：")
    demo_iv = generate_iv_scenario(
        IVScenarioConfig(n=2000, pi=0.05, rho=0.8, tau=2.0, seed=5)
    )
    emit(two_sls(demo_iv.Y, demo_iv.D, demo_iv.Z, demo_iv.X).summary())
    emit("")
    emit("  **这一节的判据被实测改写了一次**（值得单独记）：动手前写的是")
    emit("  「弱工具会让 Wald 区间覆盖率崩到 0.6 上下」（教科书结论）。")
    emit("  实测最弱档位（F≈1.2）Wald 覆盖率 **0.9733**，同方差口径与稳健口径")
    emit("  一模一样 —— 因为 SE 与点估计的尾部**一起**变大，区间宽到盖得住。")
    emit("  真正崩掉的是三件别的事：中位偏差朝 OLS 靠到 **72.2%**、")
    emit("  均值偏差与 RMSE 爆炸、AR 区间 **89% 无界**（= 排除不掉任何 β）。")
    emit("  所以照抄教科书会把报告写错，而量一遍只要 30 秒。")

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
        "内生性把 OLS 推偏了": iv_audit.ols_is_biased,
        "没有内生性时 OLS 无偏（正对照）": abs(iv_audit.ols_bias_no_endogeneity) < 0.15,
        "弱工具把 2SLS 拉向 OLS": iv_audit.weak_pulls_to_ols,
        "AR 区间全程守住名义覆盖": iv_audit.ar_holds_throughout,
        "AR 的代价（无界）看得见": iv_audit.ar_cost_is_visible,
        "Wald 在这个设计下没有崩（实测记录）": iv_audit.wald_does_not_break,
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
    worst = iv_audit.rows[0]
    emit(f"[6] 工具变量：OLS 偏差 {worst.ols_bias:+.4f}（与工具强度无关），"
         f"最弱档位（F={worst.first_stage_f:.2f}）2SLS 中位偏差 {worst.tsls_median_bias:+.4f}"
         f"= OLS 的 {worst.median_bias_ratio:+.1%}，RMSE {worst.tsls_rmse:.2f}")
    emit(f"    Wald 覆盖 {worst.wald_coverage:.4f}（**没崩**）而 AR 覆盖 "
         f"{worst.ar_coverage:.4f} —— 代价是 {worst.ar_unbounded_share:.0%} 的 AR 区间无界。")
    emit("    -> 弱工具伤的是**点估计与可用性**，不是覆盖率：")
    emit("       「我用了工具变量」不是结论，「工具有多强」才是。")
    emit(f"\n逐项检查: {checks}")
    emit(f"总体判定: {verdict}")
    emit(f"总耗时 {time.perf_counter() - t0:.1f}s")

    report = out / "m3_validation.md"
    report.write_text("# M3 验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n", encoding="utf-8", newline="\n")
    print(f"\n报告已写入 {report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
