#!/usr/bin/env python
"""M2 验证：序贯检验 / always-valid / 贝叶斯决策的全部证据。

运行::

    python scripts/run_m2_validation.py            # 完整版，约 2 分钟
    python scripts/run_m2_validation.py --quick    # 快速版

输出到 ``reports/``：

    fig11_sequential_boundaries.png  三种消耗函数的边界形状与"窥视的代价"
    fig12_sequential_fwer.png        FWER 随查看次数：naive / 群序贯 / mSPRT
    fig13_sequential_rules.png       五种停止规则的 I 类错误与功效对比
    fig14_always_valid_path.png      一条真实路径上的名义 p vs always-valid p
    fig15_prior_sensitivity.png      先验尺度 tau 如何决定贝叶斯阈值的错误率
    m2_validation.md                 全部数字
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy import stats  # noqa: E402

from ablab.plotting import label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402
from ablab.sequential import (  # noqa: E402
    build_design,
    msprt_p_value,
    repeated_ci,
)
from ablab.sim import PopulationConfig, generate_population  # noqa: E402
from ablab.sim.sequential import (  # noqa: E402
    default_information_fractions,
    simulate_canonical_sequences,
    simulate_experiment_sequence,
)
from ablab.validation import (  # noqa: E402
    run_monitoring_intensity,
    run_stopping_rule_comparison,
    run_tau_sensitivity,
    verify_adjusted_p_value,
    verify_boundary_accuracy,
)

BLUE, ORANGE, GREY, RED, GREEN, PURPLE = (
    "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd",
)


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def fig_boundaries(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax = axes[0]
    colors = {"obf": BLUE, "pocock": ORANGE, "linear": GREEN}
    for name, color in colors.items():
        d = build_design(alpha=0.05, n_looks=5, spending=name)
        ax.plot(d.information_fractions, d.boundaries, "o-", color=color, lw=1.8,
                ms=5, label=d.spending_name)
    ax.axhline(1.959964, color=RED, ls="--", lw=1.3,
               label=label("固定时点 1.96", "Fixed-horizon 1.96"))
    ax.set_xlabel(label("信息量比例 t", "Information fraction t"))
    ax.set_ylabel(label("边界 b（|z| 超过即拒绝）", "Boundary b"))
    ax.set_title(label("三种消耗函数下的群序贯边界（K=5）",
                       "Group-sequential boundaries by spending function"))
    ax.legend(fontsize=8.5)

    ax = axes[1]
    names, nominal, adjusted = [], [], []
    for name in ("obf", "pocock"):
        d = build_design(alpha=0.05, n_looks=5, spending=name)
        names.append(d.spending_name)
        nominal.append(d.final_nominal_p)
        adjusted.append(d.alpha)
    x = np.arange(len(names))
    ax.bar(x - 0.18, nominal, 0.34, color=BLUE, alpha=0.85,
           label=label("末次边界对应的名义 p", "Nominal p at final boundary"))
    ax.bar(x + 0.18, adjusted, 0.34, color=RED, alpha=0.85,
           label=label("整体 I 类错误 0.05", "Overall alpha 0.05"))
    for i, v in enumerate(nominal):
        ax.text(i - 0.18, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(label("p 值", "p-value"))
    ax.set_title(label("窥视的代价：名义 p 必须比 0.05 更严",
                       "The cost of peeking: nominal p must beat 0.05"))
    ax.legend(fontsize=8)

    save(fig, out / "fig11_sequential_boundaries.png")


def fig_monitoring(points, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    looks = np.array([p.n_looks for p in points])
    ax.plot(looks, [p.naive_fwer for p in points], "o-", color=RED, lw=1.8, ms=5,
            label=label("naive：每次都判 |z|>=1.96", "naive peeking"))
    ax.plot(looks, [p.sequential_fwer for p in points], "s-", color=BLUE, lw=1.8, ms=5,
            label=label("群序贯（OBF）", "Group sequential (OBF)"))
    ax.plot(looks, [p.msprt_fwer for p in points], "^-", color=GREEN, lw=1.8, ms=6,
            label="mSPRT (always-valid)")
    ax.axhline(0.05, color="black", ls="--", lw=1.3,
               label=label("名义 5%", "Nominal 5%"))
    ax.set_xscale("log")
    ax.set_xlabel(label("实验期间查看次数", "Number of interim looks"))
    ax.set_ylabel(label("实际 I 类错误率", "Actual Type I error"))
    ax.set_title(label("群序贯每种密度都精确；mSPRT 随监控变密才逼近名义值",
                       "Group sequential is exact; mSPRT approaches alpha only when dense"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig12_sequential_fwer.png")


def fig_rules(null_rules, alt_rules, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    x = np.arange(len(null_rules))
    short = [r.label.split("：")[0] for r in null_rules]
    colors = [RED, GREY, BLUE, GREEN, PURPLE]

    ax = axes[0]
    ax.bar(x, [r.rate for r in null_rules], color=colors, alpha=0.85, width=0.6)
    ax.axhline(0.05, color="black", ls="--", lw=1.3,
               label=label("名义 5%", "Nominal 5%"))
    for i, r in enumerate(null_rules):
        ax.text(i, r.rate, f"{r.rate:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(label("实际 I 类错误率（H0）", "Type I error (H0)"))
    ax.set_title(label("原假设下的实际错误率", "Actual error rate under H0"))
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.bar(x, [r.rate for r in alt_rules], color=colors, alpha=0.85, width=0.6)
    for i, r in enumerate(alt_rules):
        ax.text(i, r.rate, f"{r.rate:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(label("功效（H1，真效应 2 SE）", "Power (H1)"))
    ax.set_title(label("功效：mSPRT 的保守直接反映在这里",
                       "Power: mSPRT's conservatism shows up here"))

    save(fig, out / "fig13_sequential_rules.png")


def fig_always_valid_path(sequences, tau, alpha, out: Path) -> None:
    """三条示例路径：名义 p 与 always-valid p 的对比。"""
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    t = sequences.information_fractions

    for i in range(sequences.z_statistics.shape[0]):
        z = sequences.z_statistics[i]
        nominal = 2 * stats.norm.sf(np.abs(z))
        av = msprt_p_value(sequences.estimates[i], sequences.standard_errors, tau)
        style = "-" if i == 0 else "--"
        ax.plot(t, nominal, style, color=GREY, lw=1.1, alpha=0.75,
                label=label("名义 p（普通 t 检验）", "Nominal p") if i == 0 else None)
        ax.plot(t, av, style, color=BLUE, lw=1.1, alpha=0.75,
                label=label("always-valid p（mSPRT）", "Always-valid p") if i == 0 else None)

    ax.axhline(alpha, color=RED, ls=":", lw=1.5,
               label=label(f"alpha = {alpha}", f"alpha = {alpha}"))
    ax.set_yscale("log")
    ax.set_xlabel(label("信息量比例 t", "Information fraction t"))
    ax.set_ylabel(label("p 值（对数轴）", "p-value (log scale)"))
    ax.set_title(label("名义 p 会探到 0.05 以下又弹回去；always-valid p 不会",
                       "Nominal p dips below alpha and recovers; always-valid never does"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig14_always_valid_path.png")


def fig_prior_sensitivity(points, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    mult = np.array([p.tau_over_se for p in points])

    ax = axes[0]
    ax.plot(mult, [p.msprt_fwer for p in points], "^-", color=GREEN, lw=1.8, ms=6,
            label="mSPRT")
    ax.plot(mult, [p.bayes_fwer for p in points], "s-", color=PURPLE, lw=1.8, ms=5,
            label=label("贝叶斯 P(delta>0)>=0.95", "Bayesian P(delta>0)>=0.95"))
    ax.axhline(0.05, color=RED, ls="--", lw=1.3,
               label=label("名义 5%", "Nominal 5%"))
    ax.set_xscale("log")
    ax.set_xlabel(label("先验尺度 tau（相对全样本标准误）", "Prior scale tau / final SE"))
    ax.set_ylabel(label("实际 I 类错误率", "Actual Type I error"))
    ax.set_title(label("先验一换，贝叶斯的错误率能差三个数量级",
                       "The Bayesian rule's error rate spans 3 orders of magnitude"))
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(mult, [p.msprt_power for p in points], "^-", color=GREEN, lw=1.8, ms=6,
            label="mSPRT")
    ax.plot(mult, [p.bayes_power for p in points], "s-", color=PURPLE, lw=1.8, ms=5,
            label=label("贝叶斯阈值", "Bayesian threshold"))
    ax.axhline(0.8, color=GREY, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(label("先验尺度 tau（相对全样本标准误）", "Prior scale tau / final SE"))
    ax.set_ylabel(label("功效（真效应 2 SE）", "Power (true effect 2 SE)"))
    ax.set_title(label("同一个 tau 同时决定错误率与功效",
                       "The same tau sets both error rate and power"))
    ax.legend(fontsize=8.5)

    save(fig, out / "fig15_prior_sensitivity.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M2 验证：序贯检验与贝叶斯决策")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_trials = 8_000 if args.quick else 40_000
    n_accuracy = 50_000 if args.quick else 200_000
    alpha = 0.05
    se_final = 0.42
    effect_alt = 2.0 * se_final

    setup_style()
    log: list[str] = []
    t0 = time.perf_counter()

    def emit(text: str = "") -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · M2 验证报告")
    emit("群序贯边界 · always-valid p 值 · 贝叶斯决策")
    emit(header)

    # ---- 1. 边界与调整 p 值 ------------------------------------------------ #
    emit("\n### 1. 群序贯边界（Armitage-McPherson 递归）")
    for name in ("obf", "pocock", "linear"):
        d = build_design(alpha=alpha, n_looks=5, spending=name)
        emit("")
        emit(d.summary())

    emit("\n### 2. 边界精度：蒙特卡洛 vs 消耗函数")
    accuracy = verify_boundary_accuracy(n_trials=n_accuracy)
    emit(accuracy.summary())

    emit("\n### 3. 调整 p 值：放在边界上必须正好等于 alpha")
    ok, rows = verify_adjusted_p_value()
    for k, b, p in rows:
        emit(f"  第 {k} 次边界 b={b:.4f} -> p_adj={p:.8f}")
    emit(f"  全部落在 alpha 附近（容差 1e-3）: {ok}")

    # ---- 4. 停止规则对比 --------------------------------------------------- #
    emit("\n### 4. 五种停止规则（H0：真实效应为 0）")
    null_rules, design = run_stopping_rule_comparison(
        n_trials=n_trials, n_looks=5, alpha=alpha, se_final=se_final,
        effect=0.0, seed=0,
    )
    for r in null_rules:
        emit(r.summary())

    emit(f"\n### 5. 五种停止规则（H1：真实效应 = {effect_alt:.3f} = 2 个标准误）")
    alt_rules, _ = run_stopping_rule_comparison(
        n_trials=n_trials, n_looks=5, alpha=alpha, se_final=se_final,
        effect=effect_alt, seed=1,
    )
    for r in alt_rules:
        emit(r.summary())

    # ---- 6. 监控密度 ------------------------------------------------------- #
    emit("\n### 6. 查看次数越多，各方法的实际错误率如何变化")
    monitoring = run_monitoring_intensity(n_trials=n_trials, seed=0)
    emit(f"{'查看次数':>10} {'naive':>10} {'群序贯':>10} {'mSPRT':>10}")
    for mp in monitoring:
        emit(f"{mp.n_looks:>10} {mp.naive_fwer:>10.4f} {mp.sequential_fwer:>10.4f} "
             f"{mp.msprt_fwer:>10.4f}")

    # ---- 7. 先验敏感性 ----------------------------------------------------- #
    emit("\n### 7. 先验尺度 tau 的作用")
    tau_points = run_tau_sensitivity(n_trials=n_trials, seed=0)
    emit(f"{'tau/SE':>8} {'mSPRT FWER':>12} {'mSPRT 功效':>11} "
         f"{'Bayes FWER':>12} {'Bayes 功效':>11}")
    # 循环变量叫 tp 而不是 p：这个函数上面用 `p` 表示 p 值（float），
    # 复用同一个名字会让"p 到底是个数还是一个 TauPoint"变得要往回翻。
    # （这也是类型检查顺手抓到的一处可读性问题。）
    for tp in tau_points:
        emit(f"{tp.tau_over_se:>8.2f} {tp.msprt_fwer:>12.4f} {tp.msprt_power:>11.4f} "
             f"{tp.bayes_fwer:>12.4f} {tp.bayes_power:>11.4f}")

    # ---- 8. 用户级端到端 --------------------------------------------------- #
    emit("\n### 8. 用户级端到端：真实分流 + 真实抽样")
    pop = generate_population(PopulationConfig(n_units=40_000, seed=20260101))
    seq = simulate_experiment_sequence(
        pop, n_looks=5, true_lift=0.0, seed=42
    )
    emit(seq.summary())
    seq_design = build_design(alpha=alpha, n_looks=5, spending="obf")
    emit(f"  朴素窥视是否误报: {seq.naive_significant()}")
    emit(f"  群序贯是否拒绝  : {seq_design.reject(seq.z_statistics)}")
    av = msprt_p_value(seq.estimates, seq.standard_errors, 2 * se_final)
    emit(f"  mSPRT 最小 p     : {av.min():.6f}  -> 是否拒绝: {bool(av.min() <= alpha)}")
    lo, hi = repeated_ci(seq.estimates[-1], seq.standard_errors[-1], seq_design.final_boundary)
    emit(f"  末次重复置信区间 : [{lo:+.4f}, {hi:+.4f}]（边界 {seq_design.final_boundary:.4f}）")

    # ---- 9. 图表 ----------------------------------------------------------- #
    emit("\n### 9. 生成图表")
    illustrative = simulate_canonical_sequences(
        n_trials=3, information_fractions=default_information_fractions(5),
        se_final=se_final, effect=0.0, seed=12345,
    )
    fig_boundaries(out)
    fig_monitoring(monitoring, out)
    fig_rules(null_rules, alt_rules, out)
    fig_always_valid_path(illustrative, 2 * se_final, alpha, out)
    fig_prior_sensitivity(tau_points, out)
    for i in range(11, 16):
        for m in sorted(out.glob(f"fig{i}_*.png")):
            emit(f"  {m.name}")

    # ---- 结论 -------------------------------------------------------------- #
    by_label = {r.label.split("：")[0]: r for r in null_rules}
    alt_by_label = {r.label.split("：")[0]: r for r in alt_rules}
    bayes_max = max(p.bayes_fwer for p in tau_points)
    msprt_max = max(p.msprt_fwer for p in tau_points)

    checks = {
        "边界精度通过": accuracy.passed,
        "调整 p 值与边界一致": ok,
        "naive 窥视确实膨胀": by_label["naive"].rate > 0.10,
        "群序贯校准": abs(by_label["sequential"].rate - alpha) < 0.01,
        "mSPRT 保证成立（不超 alpha）": msprt_max <= alpha,
        "贝叶斯阈值并非自动校准": abs(bayes_max - alpha) > 0.02,
        "群序贯功效优于 mSPRT": alt_by_label["sequential"].rate > alt_by_label["always_valid"].rate,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    emit("\n" + header)
    emit("结论")
    emit(header)
    emit(f"[1] 群序贯：H0 下 I 类错误 {by_label['sequential'].rate:.4f}"
         f"（名义 {alpha}），五种查看密度下都稳定，功效 {alt_by_label['sequential'].rate:.4f}")
    emit("[2] naive 窥视：5 次查看就涨到 "
         f"{by_label['naive'].rate:.4f}，500 次查看涨到 "
         f"{[m.naive_fwer for m in monitoring if m.n_looks == 500][0]:.4f}")
    emit(f"[3] mSPRT：保证成立但保守。5 次查看 FWER 仅 "
         f"{[m.msprt_fwer for m in monitoring if m.n_looks == 5][0]:.4f}，"
         f"500 次才到 {[m.msprt_fwer for m in monitoring if m.n_looks == 500][0]:.4f}")
    emit("    -> 它买的是「任意停止规则都有效」，看得越密越接近把 alpha 用满")
    emit(f"[4] 贝叶斯阈值：P(delta>0)>=0.95 的错误率随先验从 "
         f"{min(p.bayes_fwer for p in tau_points):.4f} 到 {bayes_max:.4f}")
    emit("    -> 后验任何时候都自洽，但「反复看到阈值就停」这个决策规则不是自动校准的")
    emit(f"\n逐项检查: {checks}")
    emit(f"总体判定: {verdict}")
    emit(f"总耗时 {time.perf_counter() - t0:.1f}s")

    report = out / "m2_validation.md"
    report.write_text("# M2 验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n", encoding="utf-8", newline="\n")
    print(f"\n报告已写入 {report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
