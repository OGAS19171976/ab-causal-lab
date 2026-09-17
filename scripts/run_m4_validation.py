#!/usr/bin/env python
"""M4 验证：异质效应的全部证据。

运行::

    python scripts/run_m4_validation.py            # 完整版，约 2 分钟
    python scripts/run_m4_validation.py --quick    # 快速版

M4 的验证台比前三个阶段难：DML 的 theta 有真值可比，
但 CATE 没有唯一正确答案 —— 答案取决于 DGP 的函数形式。
所以这里的做法是**四种形式各跑一遍**，把"没有单一赢家"当成结论报出来。

输出到 ``reports/``：

    fig21_dml_bias.png           DML 消掉 naive plug-in 的偏置
    fig22_cate_forms.png         四种 CATE 形式下：森林 vs 常数基线
    fig23_ranking_vs_level.png   排序指标与水平指标给出相反结论
    fig24_uplift_curve.png       Qini 曲线：森林 / 常数 / 完美
    m4_validation.md             全部数字
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
    CausalForest,
    ForestConfig,
    HTEConfig,
    constant_prediction,
    generate_hte_data,
    uplift_curve,
)
from ablab.plotting import label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402
from ablab.validation import (  # noqa: E402
    run_cate_form_comparison,
    run_dml_audit,
    run_uplift_metric_audit,
)

BLUE, ORANGE, GREY, RED, GREEN, PURPLE = (
    "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd",
)

FOREST_CFG = ForestConfig(n_trees=60, max_depth=5, min_leaf=20, seed=0)


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def fig_dml_bias(audit, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ns = [p.n for p in audit.points]

    ax = axes[0]
    ax.plot(ns, [p.naive_bias for p in audit.points], "o-", color=RED, lw=1.9, ms=6,
            label=label("naive 线性回归", "naive OLS"))
    ax.plot(ns, [p.dml_bias for p in audit.points], "s-", color=BLUE, lw=1.9, ms=6,
            label="DML (5 折)")
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(label("样本量 n", "Sample size n"))
    ax.set_ylabel(label("偏置", "Bias"))
    ax.set_title(label("naive 的偏置不随 n 消失", "The naive bias does not vanish with n"))
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(ns, [p.naive_coverage for p in audit.points], "o-", color=RED, lw=1.9, ms=6,
            label=label("naive 线性回归", "naive OLS"))
    ax.plot(ns, [p.dml_coverage for p in audit.points], "s-", color=BLUE, lw=1.9, ms=6,
            label="DML (5 折)")
    ax.axhline(0.95, color="black", ls="--", lw=1.4,
               label=label("名义 95%", "Nominal 95%"))
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(label("样本量 n", "Sample size n"))
    ax.set_ylabel(label("95% 置信区间覆盖率", "95% CI coverage"))
    ax.set_title(label("更多数据只让有偏的区间更自信地错",
                       "More data makes a biased interval confidently wrong"))
    ax.legend(fontsize=8.5)

    save(fig, out / "fig21_dml_bias.png")


def fig_cate_forms(comparison, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    forms = [r.cate_form for r in comparison.results]
    x = np.arange(len(forms))

    ax = axes[0]
    ratios = [r.mse_ratio for r in comparison.results]
    plot_ratios = [min(v, 3.0) if np.isfinite(v) else 3.0 for v in ratios]
    colors = [RED if (not np.isfinite(v) or v > 1) else GREEN for v in ratios]
    ax.bar(x, plot_ratios, color=colors, alpha=0.85, width=0.55)
    ax.axhline(1.0, color="black", ls="--", lw=1.4,
               label=label("与常数基线打平", "Break-even vs constant"))
    for i, v in enumerate(ratios):
        txt = "∞" if not np.isfinite(v) else f"{v:.2f}"
        ax.text(i, plot_ratios[i], txt, ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(forms, fontsize=9)
    ax.set_ylabel(label("森林 MSE / 常数基线 MSE", "Forest MSE / constant MSE"))
    ax.set_title(label("比 1 大 = 弹性模型还不如报一个平均数",
                       "Above 1 = flexible model loses to a constant"))
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, [r.qini_forest for r in comparison.results], width,
           color=BLUE, alpha=0.85, label=label("森林", "Forest"))
    ax.bar(x + width / 2, [r.qini_constant for r in comparison.results], width,
           color=GREY, alpha=0.85, label=label("常数 ATE", "Constant ATE"))
    for i, r in enumerate(comparison.results):
        ax.plot([i - width, i + width], [r.qini_perfect] * 2, color=GREEN, lw=2.2)
    ax.plot([], [], color=GREEN, lw=2.2, label=label("完美 CATE", "Oracle CATE"))
    ax.set_xticks(x)
    ax.set_xticklabels(forms, fontsize=9)
    ax.set_ylabel(label("Qini 系数（排序能力）", "Qini coefficient"))
    ax.set_title(label("但排序上森林远胜常数",
                       "Yet the forest ranks far better"))
    ax.legend(fontsize=8)

    save(fig, out / "fig22_cate_forms.png")


def fig_ranking_vs_level(audit, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    names = [label("常数 ATE", "Constant ATE"), label("森林", "Forest"),
             label("完美", "Oracle")]
    vals = [audit.qini_constant, audit.qini_out_sample, audit.qini_perfect]
    ax.bar(names, vals, color=[GREY, BLUE, GREEN], alpha=0.85, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel(label("Qini 系数（留出集）", "Qini (holdout)"))
    ax.set_title(label("排序：森林 >> 常数", "Ranking: forest >> constant"))

    ax = axes[1]
    vals = [audit.mse_constant, audit.mse_forest]
    names = [label("常数 ATE", "Constant ATE"), label("森林", "Forest")]
    ax.bar(names, vals, color=[GREY, BLUE], alpha=0.85, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel(label("CATE 的 MSE（留出集，越小越好）", "CATE MSE (holdout)"))
    ax.set_title(label("水平：森林 < 常数", "Level: forest < constant"))

    fig.suptitle(
        label("Qini/AUUC 只衡量排序，不衡量水平 —— 两者可以给出相反结论",
              "Qini/AUUC measures ranking only — the two can disagree"),
        y=1.02, fontsize=11,
    )
    save(fig, out / "fig23_ranking_vs_level.png")


def fig_uplift_curve(data, pred, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    n = data.n
    rng = np.random.default_rng(0)
    te = rng.permutation(n)[n // 2 :]

    for name, score, color, style in (
        (label("完美 CATE", "Oracle CATE"), data.tau[te], GREEN, "-"),
        (label("因果森林", "Causal forest"), pred, BLUE, "-"),
        (label("常数 ATE", "Constant ATE"), constant_prediction(data.tau[te]), GREY, "--"),
    ):
        curve = uplift_curve(data.Y[te], data.D[te], score)
        ax.plot(curve.fractions, curve.qini, style, color=color, lw=2,
                label=f"{name}（Qini 系数 {curve.qini_coefficient:.0f}）")

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel(label("按预测提升幅度排序后取前 x 比例", "Top fraction by predicted uplift"))
    ax.set_ylabel(label("累积增量（Qini）", "Cumulative gain (Qini)"))
    ax.set_title(label("Qini 曲线：排序能力的可视化", "Qini curve: ranking ability"))
    ax.legend(fontsize=8.5)
    save(fig, out / "fig24_uplift_curve.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M4 验证：异质处理效应")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_dml_trials = 6 if args.quick else 15
    n_hte = 1500 if args.quick else 3000
    n_uplift = 2500 if args.quick else 5000
    sizes = (1000, 2500) if args.quick else (1000, 4000)

    setup_style()
    log: list[str] = []
    t0 = time.perf_counter()

    def emit(text: str = "") -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · M4 验证报告")
    emit("DML · 因果森林 · Uplift 与 Qini")
    emit(header)
    emit("")
    emit("M4 的验证台比前三个阶段难：DML 的 theta 有真值可比，")
    emit("但 CATE 没有唯一正确答案 —— 答案取决于 DGP 的函数形式。")
    emit("所以下面**四种形式各跑一遍**，「没有单一赢家」本身就是结论。")

    # ---- 1. DML ------------------------------------------------------------ #
    emit(f"\n### 1. DML 消掉 naive plug-in 的偏置（每个样本量 {n_dml_trials} 次仿真）")
    dml = run_dml_audit(n_trials=n_dml_trials, sizes=sizes, seed=0)
    emit(dml.summary())

    # ---- 2. 四种 CATE 形式 -------------------------------------------------- #
    emit(f"\n### 2. 因果森林 vs 常数基线（n={n_hte}，四种 CATE 形式）")
    comparison = run_cate_form_comparison(n=n_hte, forest_config=FOREST_CONFIG(), seed=0)
    emit(comparison.summary())

    # ---- 3. 排序 vs 水平 ---------------------------------------------------- #
    emit(f"\n### 3. 排序指标与水平指标（n={n_uplift}）")
    uplift = run_uplift_metric_audit(n=n_uplift, forest_config=FOREST_CONFIG(), seed=0)
    emit(uplift.summary())

    # ---- 4. 图表 ------------------------------------------------------------ #
    emit("\n### 4. 生成图表")
    data = generate_hte_data(HTEConfig(n=n_uplift, cate_form="nonlinear", seed=0))
    rng = np.random.default_rng(7)
    perm = rng.permutation(data.n)
    tr = perm[: data.n // 2]
    te = perm[data.n // 2 :]
    forest = CausalForest(FOREST_CONFIG()).fit(data.X[tr], data.D[tr], data.Y[tr])
    pred = forest.predict(data.X[te])

    fig_dml_bias(dml, out)
    fig_cate_forms(comparison, out)
    fig_ranking_vs_level(uplift, out)
    fig_uplift_curve(data, pred, out)
    for i in range(21, 25):
        for m in sorted(out.glob(f"fig{i}_*.png")):
            emit(f"  {m.name}")

    # ---- 结论 --------------------------------------------------------------- #
    best = comparison.best_form
    checks = {
        "naive plug-in 有可见偏置": abs(dml.large_sample.naive_bias) > 0.05,
        "DML 偏置远小于 naive": abs(dml.large_sample.dml_bias)
        < abs(dml.large_sample.naive_bias) / 3,
        "DML 覆盖率守住在名义附近": dml.dml_holds,
        "naive 覆盖率随 n 崩塌": dml.coverage_collapses,
        "四种 CATE 形式都跑过": len(comparison.results) == 4,
        "存在排序能力（秩相关为正）": all(
            (not np.isfinite(r.rank_correlation)) or r.rank_correlation > 0
            for r in comparison.results
        ),
        "排序与水平结论冲突": uplift.verdicts_conflict,
        "样本内 Qini 虚高": uplift.in_sample_optimism > 0,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    big = dml.large_sample
    small = dml.points[0]
    emit("\n" + header)
    emit("结论")
    emit(header)
    emit(f"[1] DML 消偏置（n={big.n}）：naive 偏置 {big.naive_bias:+.4f}，"
         f"95% CI 覆盖率只有 {big.naive_coverage:.0%}；"
         f"DML 偏置 {big.dml_bias:+.4f}，覆盖率 {big.dml_coverage:.0%}")
    emit(f"    naive 的偏置**不随 n 消失**：覆盖率从 n={small.n} 的 "
         f"{small.naive_coverage:.0%} 崩到 n={big.n} 的 {big.naive_coverage:.0%}")
    emit("    -> 更多数据只会让有偏的区间更自信地错")
    emit(f"    诚实的边界：n={small.n} 时 DML 自己也偏（{small.dml_bias:+.3f}）——"
         " nuisance 太弱时不满足 n^{-1/4} 收敛条件")
    emit(f"[2] 四种 CATE 形式下，弹性模型的 MSE 相对常数基线分别是 "
         + " / ".join(
             ("∞" if not np.isfinite(r.mse_ratio) else f"{r.mse_ratio:.2f}")
             for r in comparison.results
         ))
    winners = [r.cate_form for r in comparison.results if r.forest_beats_constant]
    losers = [r.cate_form for r in comparison.results if not r.forest_beats_constant]
    if winners:
        emit(f"    弹性模型只在 {', '.join(winners)} 下赢过常数基线；"
             f"在 {', '.join(losers)} 下都输")
        emit("    -> 它只在自身归纳偏置恰好匹配 DGP 时才赢 —— "
             "而你在真实数据里并不知道 DGP 是哪一种")
    else:
        emit(f"    最好的形式是 {best}，但**全部大于 1** ——"
             " 比「对所有人报同一个 ATE」还差")
    emit(f"[3] 秩相关 {comparison.results[-1].rank_correlation:+.3f} 说明森林**确实抓到了排序**，"
         "只是水平的方差太大")
    emit(f"[4] Qini：留出集森林 {uplift.qini_out_sample:.1f} vs 常数 {uplift.qini_constant:.1f}"
         f"（完美 {uplift.qini_perfect:.1f}）；样本内虚高 {uplift.in_sample_optimism:+.0%}")
    emit("    -> **排序指标与水平指标给出相反结论**。")
    emit("       「我的模型 AUUC 更高」不等于「我的 CATE 估得更准」。")
    emit(f"\n逐项检查: {checks}")
    emit(f"总体判定: {verdict}")
    emit(f"总耗时 {time.perf_counter() - t0:.1f}s")

    report = out / "m4_validation.md"
    report.write_text("# M4 验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n", encoding="utf-8")
    print(f"\n报告已写入 {report}")
    return 0 if verdict == "PASS" else 1


def FOREST_CONFIG() -> ForestConfig:
    """每个脚本运行都拿一份新的配置（避免共享可变状态）。"""
    return ForestConfig(n_trees=60, max_depth=5, min_leaf=20, seed=0)


if __name__ == "__main__":
    raise SystemExit(main())
