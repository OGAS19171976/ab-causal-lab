#!/usr/bin/env python
"""M0 验证：产出"框架是对的"的全部证据（图表 + 结论报告）。

运行::

    python scripts/run_m0_validation.py            # 完整版，约 1 分钟
    python scripts/run_m0_validation.py --quick    # 快速版，约 10 秒

输出到 ``reports/``：

    fig1_fpr_convergence.png    经验 I 类错误率收敛到 5%
    fig2_pvalue_uniformity.png  原假设下 p 值是否均匀分布
    fig3_effect_distribution.png 效应估计的抽样分布（随机化 vs 条件）
    fig4_power_curve.png        经验功效 vs 解析功效
    fig5_peeking.png            窥视如何毁掉 I 类错误率
    m0_validation.md            全部数字
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
from ablab.validation import (  # noqa: E402
    run_aa_trials,
    run_assignment_audit,
    run_peeking_simulation,
    run_power_trials,
    wilson_interval,
)

BLUE, ORANGE, GREY, RED, GREEN = "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c"


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def fig_fpr_convergence(rand, cond, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(1, rand.n_trials + 1)

    ax.plot(x, rand.cumulative_fpr(), color=BLUE, lw=1.8,
            label=label("随机化模式（每次重新分流）", "Randomized (re-assign each trial)"))
    ax.plot(x, cond.cumulative_fpr(), color=ORANGE, lw=1.8,
            label=label("条件模式（分流固定）", "Conditional (fixed assignment)"))

    # 随机化模式的 Wilson 置信带
    checkpoints = np.unique(np.linspace(20, rand.n_trials, 120).astype(int))
    p = rand.cumulative_fpr()[checkpoints - 1]
    band = np.array([wilson_interval(int(round(pi * k)), k) for pi, k in zip(p, checkpoints)])
    ax.fill_between(checkpoints, band[:, 0], band[:, 1], color=BLUE, alpha=0.15,
                    label=label("随机化模式的 95% Wilson 带", "95% Wilson band"))

    ax.axhline(0.05, color=RED, ls="--", lw=1.3,
               label=label("名义水平 5%", "Nominal 5%"))
    ax.set_xlabel(label("A/A 重复次数", "A/A repetitions"))
    ax.set_ylabel(label("累计经验 I 类错误率", "Cumulative empirical Type I error"))
    ax.set_title(label("A/A 仿真：I 类错误率是否收敛到名义 5%",
                       "A/A simulation: does the Type I error converge to 5%?"))
    ax.set_ylim(0, 0.14)
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, out / "fig1_fpr_convergence.png")


def fig_pvalue_uniformity(rand, cond, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, res, color, title in (
        (axes[0], rand, BLUE, label("随机化模式", "Randomized")),
        (axes[1], cond, ORANGE, label("条件模式", "Conditional")),
    ):
        ax.hist(res.p_values, bins=20, range=(0, 1), color=color, alpha=0.8,
                edgecolor="white", density=True)
        ax.axhline(1.0, color=RED, ls="--", lw=1.2)
        d, p = res.uniformity_test()
        ax.set_title(f"{title}\nKS D={d:.4f}, p={p:.3g}")
        ax.set_xlabel(label("p 值", "p-value"))
    axes[0].set_ylabel(label("密度", "Density"))
    fig.suptitle(
        label("原假设下 p 值应服从 Uniform(0,1) —— 虚线为理论值",
              "Under H0 the p-values must be Uniform(0,1)"),
        y=1.03, fontsize=11,
    )
    save(fig, out / "fig2_pvalue_uniformity.png")


def fig_effect_distribution(rand, cond, pop_cfg, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    theory_full = np.sqrt(2 * pop_cfg.post_sd**2 / rand.mean_n_per_arm)
    eps_sd = pop_cfg.post_sd * np.sqrt(1 - pop_cfg.corr_pre_post**2)
    theory_noise = np.sqrt(2 * eps_sd**2 / cond.mean_n_per_arm)

    bins = np.linspace(-1.3, 1.3, 60)
    ax.hist(rand.effects, bins=bins, density=True, color=BLUE, alpha=0.65,
            label=label(f"随机化模式  sd={rand.sd_effect:.3f}",
                        f"Randomized  sd={rand.sd_effect:.3f}"))
    ax.hist(cond.effects, bins=bins, density=True, color=ORANGE, alpha=0.65,
            label=label(f"条件模式  sd={cond.sd_effect:.3f}",
                        f"Conditional  sd={cond.sd_effect:.3f}"))

    for x0, c, ls, txt in (
        (theory_full, BLUE, "--", label(f"理论 SE={theory_full:.3f}", f"Theory SE={theory_full:.3f}")),
        (-theory_full, BLUE, "--", None),
        (theory_noise, ORANGE, "--", label(f"理论噪声={theory_noise:.3f}", f"Theory noise={theory_noise:.3f}")),
        (-theory_noise, ORANGE, "--", None),
    ):
        ax.axvline(x0, color=c, ls=ls, lw=1.4, label=txt)

    ax.axvline(0.0, color=RED, lw=1.6, label=label("真值 0", "Truth = 0"))
    ax.set_xlabel(label("效应估计值", "Estimated effect"))
    ax.set_ylabel(label("密度", "Density"))
    ax.set_title(label(
        "效应估计的抽样分布：条件模式波动更小，但整体偏移（协变量失衡）",
        "Sampling distribution: conditional is tighter but shifted (imbalance)",
    ))
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, out / "fig3_effect_distribution.png")


def fig_power_curve(results, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    lifts = np.array([r.true_lift for r in results])
    emp = np.array([r.empirical_power for r in results])
    ana = np.array([r.analytic_power for r in results])
    lo, hi = np.array([r.power_interval for r in results]).T

    ax.plot(lifts, ana, color=RED, lw=1.8, ls="--",
            label=label("解析功效（公式）", "Analytic power"))
    ax.errorbar(lifts, emp, yerr=[emp - lo, hi - emp], fmt="o", color=BLUE,
                capsize=3, ms=5, lw=1.2,
                label=label("经验功效（仿真，95% Wilson）", "Empirical power (95% Wilson)"))

    ax.axhline(0.8, color=GREY, ls=":", lw=1.2,
               label=label("常规 80% 功效线", "Conventional 80%"))
    ax.set_xlabel(label("真实效应量 (absolute lift)", "True effect size"))
    ax.set_ylabel(label("功效", "Power"))
    ax.set_title(label("功效校准：仿真能否复现解析公式",
                       "Power calibration: simulation vs closed form"))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)
    save(fig, out / "fig4_power_curve.png")


def fig_peeking(peeks, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    looks = [p.n_looks for p in peeks]
    w = 0.26
    pos = np.arange(len(looks))

    series = (
        ("naive", RED, label("朴素窥视（每次判 p<0.05）", "Naive peeking")),
        ("fixed_horizon", GREEN, label("只看最后一次", "Fixed horizon only")),
        ("calibrated", BLUE, label("标定常数边界", "Calibrated boundary")),
    )
    for i, (key, color, lab) in enumerate(series):
        vals = [getattr(p, f"{key}_fpr") for p in peeks]
        ax.bar(pos + (i - 1) * w, vals, w, color=color, alpha=0.85, label=lab)

    ax.axhline(0.05, color="black", ls="--", lw=1.3,
               label=label("名义 5%", "Nominal 5%"))
    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in looks])
    ax.set_xlabel(label("实验期间查看数据的次数", "Number of interim looks"))
    ax.set_ylabel(label("实际 I 类错误率", "Actual Type I error rate"))
    ax.set_title(label("窥视问题：看得越勤，假阳性越高",
                       "Peeking: more looks, more false positives"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig5_peeking.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M0 验证：产出仿真证据")
    ap.add_argument("--quick", action="store_true", help="减少重复次数，快速跑一遍")
    ap.add_argument("--out", default=str(ROOT / "reports"), help="报告输出目录")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_aa = 400 if args.quick else 2_000
    n_power = 150 if args.quick else 400
    n_peek = 400 if args.quick else 2_000
    n_units = 5_000 if args.quick else 20_000
    look_counts = (2, 10) if args.quick else (1, 2, 5, 10, 25)

    setup_style()
    log: list[str] = []
    t_start = time.perf_counter()

    def emit(text: str) -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · M0 验证报告")
    emit(f"队列规模 {n_units:,} 用户；A/A 重复 {n_aa:,} 次；"
         f"功效重复 {n_power} 次/点；窥视重复 {n_peek} 次/点")
    emit(header)

    pop_cfg = PopulationConfig(n_units=n_units, seed=20260101)
    pop = generate_population(pop_cfg)
    spec = two_arm_spec("aa_test", salt="aa_test_v1")

    # ---- 1. 分流层审计 ---------------------------------------------------- #
    emit("\n### 1. 分流层正确性")
    audit = run_assignment_audit(
        pop,
        spec,
        n_salts_uniformity=60 if args.quick else 200,
        n_salts_srm=120 if args.quick else 500,
        n_layer_pairs=60 if args.quick else 200,
    )
    emit(audit.summary())

    # ---- 2. A/A 校准 ------------------------------------------------------ #
    emit("\n### 2. 推断层校准（A/A 仿真）")
    emit("随机化模式：每次重新分流 —— 这才是 5% 所对应的重复抽样框架")
    rand = run_aa_trials(pop, spec, n_trials=n_aa, mode="randomized", seed=7)
    emit(rand.summary())

    emit("条件模式：分流固定，只重抽噪声 —— 不是校准检验")
    cond = run_aa_trials(pop, spec, n_trials=n_aa, mode="conditional", seed=7)
    emit(cond.summary())

    # ---- 3. 功效校准 ------------------------------------------------------ #
    emit("\n### 3. 功效校准")
    lifts = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0])
    powers = [
        run_power_trials(pop, spec, true_lift=float(v), n_trials=n_power, seed=100 + i)
        for i, v in enumerate(lifts)
    ]
    emit(f"{'lift':>7} {'经验功效':>10} {'解析功效':>10} {'偏差':>9} {'95% CI':>20}")
    for r in powers:
        lo, hi = r.power_interval
        emit(f"{r.true_lift:>7.2f} {r.empirical_power:>10.4f} {r.analytic_power:>10.4f} "
             f"{r.gap:>+9.4f}   [{lo:.3f}, {hi:.3f}]")
    max_gap = max(abs(r.gap) for r in powers)
    emit(f"最大偏差 = {max_gap:.4f}")

    # ---- 4. 窥视问题 ------------------------------------------------------ #
    emit("\n### 4. 窥视问题（Peeking）")
    peeks = [
        run_peeking_simulation(pop, n_looks=k, n_trials=n_peek, seed=200 + i)
        for i, k in enumerate(look_counts)
    ]
    emit(f"{'查看次数':>8} {'朴素窥视':>10} {'只看最后':>10} {'标定边界':>10} {'边界值':>9}")
    for p in peeks:
        emit(f"{p.n_looks:>8} {p.naive_fpr:>10.4f} {p.fixed_horizon_fpr:>10.4f} "
             f"{p.calibrated_fpr:>10.4f} {p.boundary:>9.3f}")

    # ---- 5. 图表 ---------------------------------------------------------- #
    emit("\n### 5. 生成图表")
    fig_fpr_convergence(rand, cond, out)
    fig_pvalue_uniformity(rand, cond, out)
    fig_effect_distribution(rand, cond, pop_cfg, out)
    fig_power_curve(powers, out)
    fig_peeking(peeks, out)
    for name in sorted(p.name for p in out.glob("fig*.png")):
        emit(f"  {name}")

    # ---- 结论 ------------------------------------------------------------- #
    fpr_lo, fpr_hi = rand.fpr_interval()
    cov_lo, cov_hi = rand.coverage_interval()
    verdict = (
        "PASS"
        if (fpr_lo <= 0.05 <= fpr_hi and cov_lo <= 0.95 <= cov_hi and rand.uniformity_test()[1] > 0.05)
        else "FAIL"
    )

    emit("\n" + header)
    emit("结论")
    emit(header)
    emit(f"[1] 分流层：哈希均匀、分层正交、放量不换组 —— {audit.summary().splitlines()[-1]}")
    emit(f"[2] 推断层：随机化模式 I 类错误 {rand.empirical_fpr():.4f} "
         f"(95% CI [{fpr_lo:.4f}, {fpr_hi:.4f}])，覆盖率 {rand.coverage():.4f} "
         f"(95% CI [{cov_lo:.4f}, {cov_hi:.4f}])，p 值均匀性 p={rand.uniformity_test()[1]:.3g}")
    emit(f"[3] 条件模式（固定分流）：I 类错误 {cond.empirical_fpr():.4f}"
         f"{'（偏保守）' if cond.empirical_fpr() < 0.05 else '（偏激进）'}，"
         f"效应估计均值 {cond.mean_effect:+.4f} ≠ 0")
    emit("    t 检验的 5% 是**对随机化取期望**的边际保证；固定住一次分流后，")
    emit("    实际错误率由这一次实现的协变量失衡决定，可高可低且无从知晓。")
    emit("    -> 这是 M1 引入 CUPED 的动机：前置协变量校正同时消掉偏置与虚高方差")
    emit(f"[4] 功效：经验与解析最大偏差 {max_gap:.4f}")
    emit(f"[5] 窥视：看 {look_counts[-1]} 次时朴素窥视假阳性 "
         f"{peeks[-1].naive_fpr:.1%}，是名义水平的 {peeks[-1].naive_fpr / 0.05:.1f} 倍")
    emit(f"总体判定: {verdict}")
    emit(f"总耗时 {time.perf_counter() - t_start:.1f}s")

    report = out / "m0_validation.md"
    report.write_text(
        "# M0 验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\n报告已写入 {report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
