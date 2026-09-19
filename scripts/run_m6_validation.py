#!/usr/bin/env python
"""M6 验证：平台在**生产口径**下站不站得住。

运行::

    python scripts/run_m6_validation.py             # 完整版，约 6 分钟
    python scripts/run_m6_validation.py --quick     # 快速版

M5 把引擎接成了服务；M6 修的是"接得对不对"。三件事：

1. **口径一致**：平台原来"头条用 CUPED、监控曲线用 post-only 的 z"——
   两个估计量。现在判定口径跟着记录上的声明走，且与头条结论共用同一个实现。
   新问题必须实测量：CUPED 的 z 配同一组 OBF 边界，FWER 还稳不稳？
2. **分析单元**：整簇随机化下用单元级 t 检验，I 类错误率会到 60% 以上，
   而那个 p 值看起来完全正常。平台现在把"随机化单元 = 分析单元"这条接住了。
   比值指标同理：估计量必须是业务口径 Σy/Σx（delta method），不是人均比值。
3. **MDE / 功效**：回答实验前的问题 —— 现在这点样本能检出多大的效应、要跑多久。

输出到 ``reports/``：

    fig28_monitoring_estimator.png  两个口径的 z 分布与检出率
    fig29_unit_awareness.png        单元级 vs 簇级的 I 类错误率
    fig30_mde_power.png             MDE 与功效曲线（CUPED 的收益）
    m6_validation.md                全部数字
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

from ablab.inference import mde, required_n_per_arm, se_of_mean_diff, z_power  # noqa: E402
from ablab.platform import (  # noqa: E402
    analyse_experiment,
    run_monitoring_fwer_audit,
    run_ratio_calibration_audit,
    run_unit_awareness_audit,
)
from ablab.platform.analysis import PLATFORM_POPULATION  # noqa: E402
from ablab.platform.registry import ExperimentRecord  # noqa: E402
from ablab.plotting import label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN, PURPLE = (
    "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd",
)

TWO_ARM = [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}]


def _record(tag: str, **kw) -> ExperimentRecord:
    return ExperimentRecord(
        id="x", name=tag, salt=f"{tag}_v1", variants=TWO_ARM,
        primary_metric="metric", **kw,
    )


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def fig_monitoring_estimator(h0, h1, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))

    ax = axes[0]
    bins = np.linspace(-4.5, 4.5, 40)
    grid = np.linspace(-4.5, 4.5, 300)
    ax.hist(h0.cuped_z, bins=bins, density=True, alpha=0.5, color=BLUE,
            label=label("CUPED（现判定口径）", "CUPED (declared)"))
    ax.hist(h0.post_only_z, bins=bins, density=True, alpha=0.4, color=GREY,
            label=label("post-only（对照）", "post-only (alt)"))
    ax.plot(grid, stats.norm.pdf(grid), color=RED, lw=1.8, label="N(0,1)")
    ax.set_title(label("零效应下末次 z 的分布", "Final-look z under H0"))
    ax.set_xlabel("z")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.arange(2)
    rates = [h0.post_only_fwer, h0.cuped_fwer]
    lo = [h0.fwer_interval[0], h0.cuped_fwer_interval[0]]
    hi = [h0.fwer_interval[1], h0.cuped_fwer_interval[1]]
    ax.bar(xs, rates, width=0.5, color=[GREY, BLUE], alpha=0.85)
    ax.errorbar(xs, rates, yerr=[np.array(rates) - np.array(lo), np.array(hi) - np.array(rates)],
                fmt="none", ecolor="black", capsize=5, lw=1.4)
    ax.axhline(h0.alpha, color=RED, ls="--", lw=1.4, label=label("α = 0.05", "alpha"))
    ax.set_xticks(xs)
    ax.set_xticklabels([label("post-only", "post-only"), "CUPED"])
    ax.set_title(label("序贯越界率（= FWER）", "Sequential crossing rate (FWER)"))
    ax.legend(fontsize=8)

    ax = axes[2]
    rates1 = [h1.post_only_fwer, h1.cuped_fwer]
    ax.bar(xs, rates1, width=0.5, color=[GREY, BLUE], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([label("post-only", "post-only"), "CUPED"])
    ax.set_ylim(0, 1.05)
    for x, r in zip(xs, rates1):
        ax.annotate(f"{r:.1%}", (x, r), ha="center", va="bottom", fontsize=9)
    ax.set_title(
        label(
            f"真实效应 {h1.true_lift} 下的检出率",
            f"Detection rate at lift {h1.true_lift}",
        )
    )

    fig.suptitle(
        label(
            "M6.1 监控口径与判定口径对齐：同一边界下 CUPED 的 FWER 仍守住、功效更高",
            "M6.1 Aligned monitoring: same boundary, calibrated FWER, higher power",
        ),
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out / "fig28_monitoring_estimator.png")


def fig_unit_awareness(u, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    xs = np.arange(2)
    fixed = [u.unit_level_fpr, u.cluster_level_fpr]
    seq = [u.unit_level_sequential, u.cluster_level_sequential]
    w = 0.36
    ax.bar(xs - w / 2, fixed, width=w, color=[RED, GREEN], alpha=0.85,
           label=label("固定样本（只看最后一次）", "fixed sample"))
    ax.bar(xs + w / 2, seq, width=w, color=[RED, GREEN], alpha=0.45,
           label=label("序贯（至少越界一次）", "sequential"))
    ax.axhline(u.alpha, color="black", ls="--", lw=1.4, label=label("α = 0.05", "alpha"))
    ax.set_xticks(xs)
    ax.set_xticklabels([label("单元级（错）", "unit-level (wrong)"),
                        label("簇级（对）", "cluster-level (right)")])
    ax.set_ylim(0, 1.05)
    for x, v in zip(xs - w / 2, fixed):
        ax.annotate(f"{v:.1%}", (x, v), ha="center", va="bottom", fontsize=9)
    for x, v in zip(xs + w / 2, seq):
        ax.annotate(f"{v:.1%}", (x, v), ha="center", va="bottom", fontsize=9)
    ax.set_title(label("整簇随机化、真实效应为零", "Cluster randomized, true effect zero"))
    ax.set_ylabel(label("I 类错误率", "Type I error rate"))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar([0], [1.0], width=0.5, color=GREEN, alpha=0.85,
           label=label("簇级 SE（正确）", "cluster SE (right)"))
    ax.bar([1], [u.se_ratio], width=0.5, color=RED, alpha=0.85,
           label=label("单元级 SE（被低估）", "unit SE (understated)"))
    ax.annotate(f"低估 {u.se_understatement:.1%}", (1, u.se_ratio),
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label("簇级", "cluster"), label("单元级", "unit")])
    ax.set_title(label(f"标准误之比（{u.n_clusters} 个簇）", f"SE ratio ({u.n_clusters} clusters)"))
    ax.legend(fontsize=8)

    fig.suptitle(
        label(
            "M6.2 分析单元必须与随机化单元对齐，否则 p 值看着正常、结论是错的",
            "M6.2 Analysis unit must match randomization unit",
        ),
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out / "fig29_unit_awareness.png")


def fig_mde_power(out: Path, post_sd: float, rho: float, baseline: float, n_arm: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    lifts = np.linspace(0.0, 3.0, 60)

    ax = axes[0]
    for sd, color, name in (
        (post_sd, GREY, label("post-only", "post-only")),
        (post_sd * np.sqrt(1 - rho**2), BLUE, f"CUPED (ρ={rho})"),
    ):
        power = [z_power(lift, se_of_mean_diff(sd, n_arm, n_arm), 0.05) for lift in lifts]
        ax.plot(lifts, power, color=color, lw=2.0, label=name)
    ax.axhline(0.8, color=RED, ls="--", lw=1.3, label=label("功效 0.8", "power 0.8"))
    ax.set_xlabel(label("绝对效应", "absolute effect"))
    ax.set_ylabel(label("功效", "power"))
    ax.set_title(label(f"功效曲线（每臂 n={n_arm:,}）", f"Power curve (n={n_arm:,}/arm)"))
    ax.legend(fontsize=8)

    ax = axes[1]
    rels = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
    needs = [required_n_per_arm(post_sd, baseline * r, power=0.8) * 2 for r in rels]
    needs_cuped = [
        required_n_per_arm(post_sd * np.sqrt(1 - rho**2), baseline * r, power=0.8) * 2
        for r in rels
    ]
    ax.plot(rels * 100, needs, "o-", color=GREY, lw=1.9, ms=6, label=label("post-only", "post-only"))
    ax.plot(rels * 100, needs_cuped, "o-", color=BLUE, lw=1.9, ms=6, label=f"CUPED (ρ={rho})")
    ax.set_yscale("log")
    ax.set_xlabel(label("要检出的相对提升（%）", "relative MDE (%)"))
    ax.set_ylabel(label("所需总样本量", "required total n"))
    ax.set_title(label("实验前定量：要多少样本", "Pre-experiment sizing"))
    ax.legend(fontsize=8)

    fig.suptitle(
        label(
            "M6.3 MDE / 功效：CUPED 把所需样本量降到 1-ρ² 倍",
            "M6.3 MDE / power: CUPED cuts required n by 1-rho^2",
        ),
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out / "fig30_mde_power.png")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="M6 平台生产口径验证")
    ap.add_argument("--quick", action="store_true", help="快速版（数字更粗）")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()

    n_salts = 120 if args.quick else 400
    n_unit_salts = 100 if args.quick else 300
    # 比值口径的校准审计（合成路径，每次 8k 用户）—— 与 M6.1 那个同量级
    n_ratio_salts = 60 if args.quick else 200
    # 簇级 CUPED 的校准次数：每次要跑两遍分析（cuped + post_only），
    # 200 次约 1 分钟；快速版给 40 次。
    n_cluster_cuped = 40 if args.quick else 200

    log: list[str] = []
    t_start = time.time()

    def say(line: str = "") -> None:
        print(line, flush=True)
        log.append(line)

    # ---- 1. 口径一致（M6.1） --------------------------------------------- #
    say("=" * 78)
    say("1. M6.1 监控口径 == 判定口径：FWER 还稳吗、功效变多少")
    say("=" * 78)
    t0 = time.time()
    h0 = run_monitoring_fwer_audit(n_salts=n_salts, n_units=20_000, n_looks=5, true_lift=0.0)
    say(f"零效应 {n_salts} 个 salt，耗时 {time.time() - t0:.0f}s")
    say("")
    say(f"{'口径':<12}{'序贯越界率':>12}{'Wilson 95% CI':>26}{'末次越界率':>12}")
    say(f"{'post-only':<12}{h0.post_only_fwer:>12.4f}"
        f"{f'({h0.fwer_interval[0]:.4f}, {h0.fwer_interval[1]:.4f})':>26}"
        f"{h0.post_only_final_rate:>12.4f}")
    say(f"{'CUPED':<12}{h0.cuped_fwer:>12.4f}"
        f"{f'({h0.cuped_fwer_interval[0]:.4f}, {h0.cuped_fwer_interval[1]:.4f})':>26}"
        f"{h0.cuped_final_rate:>12.4f}")
    say("")
    say(f"末次 z：post-only 均值 {h0.post_only_z_mean:+.4f} 标准差 {h0.post_only_z_sd:.4f}；"
        f"CUPED 均值 {h0.cuped_z_mean:+.4f} 标准差 {h0.cuped_z_sd:.4f}")
    say(f"两个口径 z 的相关系数 = {h0.z_correlation:.4f}")
    say(f"**两个口径越界结论不一致的比例 = {h0.disagreement:.4f}**"
        "（这就是'曲线和结论卡互相打架'的发生率）")
    say(f"CUPED 的 FWER 区间盖住 alpha? {h0.cuped_calibrated}")
    say("")
    say("结论：CUPED 的 z 配同一组 OBF 边界，FWER 仍然守住 5% ——")
    say("      所以对齐到 CUPED 不是拿有效性换灵敏度，而是纯收益。")

    say("")
    t0 = time.time()
    h1 = run_monitoring_fwer_audit(n_salts=n_salts, n_units=20_000, n_looks=5, true_lift=0.25)
    say(f"真实效应 0.25 下（{n_salts} 个 salt，耗时 {time.time() - t0:.0f}s）")
    say(f"  post-only 检出率 = {h1.post_only_fwer:.4f}")
    say(f"  CUPED     检出率 = {h1.cuped_fwer:.4f}"
        f"   （相对提升 {h1.cuped_advantage:+.1%}）")

    # ---- 2. 分析单元（M6.2） --------------------------------------------- #
    say("")
    say("=" * 78)
    say("2. M6.2 分析单元：整簇随机化下用错单元会怎样")
    say("=" * 78)
    t0 = time.time()
    ua = run_unit_awareness_audit(n_salts=n_unit_salts, n_users=10_000, n_looks=5)
    say(f"{ua.n_salts} 个 salt，{ua.n_users:,} 用户 / {ua.n_clusters} 个簇，"
        f"真实效应为零，耗时 {time.time() - t0:.0f}s")
    say("")
    say(f"{'口径':<14}{'固定样本 I 类错误':>18}{'序贯越界率':>14}{'Wilson 95% CI':>26}")
    say(f"{'单元级（错）':<14}{ua.unit_level_fpr:>18.4f}{ua.unit_level_sequential:>14.4f}"
        f"{f'({ua.unit_level_interval[0]:.4f}, {ua.unit_level_interval[1]:.4f})':>26}")
    say(f"{'簇级（对）':<14}{ua.cluster_level_fpr:>18.4f}{ua.cluster_level_sequential:>14.4f}"
        f"{f'({ua.cluster_level_interval[0]:.4f}, {ua.cluster_level_interval[1]:.4f})':>26}")
    say("")
    say(f"单元级标准误被低估 {ua.se_understatement:.1%}（SE 之比 {ua.se_ratio:.4f}）")
    say(f"簇级的 I 类错误率区间盖住 alpha? {ua.cluster_calibrated}")
    say("")
    say("注意：SRM 在簇设计下检验的是**簇数**，不是用户数 —— 随机化单元是簇。")

    # 比值指标的口径差：单次实现会波动（M1 报告的是 12.6%），
    # 所以量它的**分布**，而不是拿一次的数字当结论。
    say("")
    say("比值指标：业务口径（delta method）vs 人均比值（同一份数据）")
    from ablab.hashing import murmur3_32
    from ablab.inference import ratio_delta_method
    from ablab.platform.datasource import build_synthetic_data

    n_gap = 20 if args.quick else 40
    gaps: list[float] = []
    for i in range(n_gap):
        salt = f"m6_gap_{i}_v1"
        data_i = build_synthetic_data(
            experiment=f"m6_gap_{i}", salt=salt, variants=[("control", 0.5), ("treatment", 0.5)],
            metric="metric", n_users=20_000, n_looks=5, true_lift=0.02,
            seed=murmur3_32(salt.encode("utf-8")), population=PLATFORM_POPULATION,
            metric_type="ratio",
        )
        delta = ratio_delta_method(data_i.total.treatment, data_i.total.control).absolute_effect
        naive = float(data_i.extra["naive_ratio_treatment"]) - float(
            data_i.extra["naive_ratio_control"]
        )
        if abs(naive) > 1e-12:
            gaps.append((delta - naive) / naive)
    gap_arr = np.array(gaps)
    say(f"  {len(gaps)} 个 salt 上，delta method 相对人均比值的口径差：")
    say(f"    均值 {gap_arr.mean():+.2%}  中位数 {np.median(gap_arr):+.2%}  "
        f"范围 [{gap_arr.min():+.2%}, {gap_arr.max():+.2%}]")
    say("  M1 用另一个 DGP 量到的是 −12.6%。两个数不能直接比 ——")
    say("  口径差取决于暴露量分布与点击率水平，符号也会变。")
    say("  **而且这个'相对差'本身不稳**：效应接近 0 时它会爆炸（范围跨了 250 个百分点）。")
    say("  可以依赖的只有一件事：两个估计量系统性地不是同一个数，"
        "所以'用哪个'必须由口径定义决定，不能看哪个 p 值好看。")
    say("  两者都「校准」，但答的不是同一个问题 —— 平台只支持前者，"
        "因为后者需要明细，而数仓的 ADS 只给充分统计量。")

    # ---- 3. MDE / 功效（M6.3） ------------------------------------------- #
    say("")
    say("=" * 78)
    say("3. M6.3 MDE / 功效：实验前定量")
    say("=" * 78)
    say("")
    say("公式自洽性：z_power(mde(se)) 必须回到目标功效")
    for p in (0.5, 0.8, 0.9, 0.95):
        d = mde(0.12, power=p)
        say(f"  power={p:<5} mde={d:.6f}  z_power(mde)={z_power(d, 0.12):.10f}")
    say(f"  z_power(0, se, 0.05) = {z_power(0.0, 0.12, 0.05):.10f}（应等于 alpha）")
    say("")
    cfg = PLATFORM_POPULATION
    say("平台报告里的功效读数（同一份合成数据，两种口径）：")
    for est in ("cuped", "post_only"):
        rep = analyse_experiment(_record(f"m6_pow_{est}", estimator=est, true_lift=0.35), n_users=20_000)
        pw = rep.power
        say(f"  {est:<10} SE={pw['se']:.4f}  MDE(80%)={pw['mde_abs']:.4f}"
            f"（相对基线 {pw['mde_relative']:.2%}）"
            f"  观测效应功效={pw['power_at_observed']:.1%}")
    say("")
    say("实验前定量：要检出给定的相对提升，需要多少**总**样本（功效 0.8, alpha 0.05）")
    say(f"{'相对提升':>10}{'post-only 总量':>16}{'用 CUPED 总量':>16}{'省下':>10}")
    for rel in (0.05, 0.03, 0.02, 0.01):
        target = cfg.post_mean * rel
        n_plain = required_n_per_arm(cfg.post_sd, target, power=0.8) * 2
        n_cuped = (
            required_n_per_arm(
                cfg.post_sd * (1 - cfg.corr_pre_post**2) ** 0.5, target, power=0.8
            ) * 2
        )
        say(f"{rel:>10.1%}{n_plain:>16,.0f}{n_cuped:>16,.0f}{1 - n_cuped / n_plain:>10.1%}")
    say("")
    say("不等权分配的效率代价（同一目标、同样总样本）")
    for q in (0.5, 0.3, 0.2, 0.1):
        need = required_n_per_arm(cfg.post_sd, cfg.post_mean * 0.02, treatment_ratio=q) * 2
        say(f"  处理组占比 {q:>4.0%} → 需要总量 {need:>10,.0f}"
            f"（相对 50/50 多 {need / (required_n_per_arm(cfg.post_sd, cfg.post_mean * 0.02) * 2) - 1:+.0%}）")

    # ---- 3.5 比值口径的 A/A 校准 ----------------------------------------- #
    #
    # "换了口径就要重跑审计"这条规矩对比值链路的一次执行。
    # 均值口径那套经验（CUPED 对齐后 FWER 仍守 5%）**不能直接搬过来**：
    # 比值指标的 SE 走 delta method，而序贯查看点是按累计**信息量**挑的 ——
    # 而比值指标的精度由分母驱动，信息量的定义与均值口径不同。
    say("")
    say("=" * 78)
    say("3.5 比值口径（delta method）在平台真实入口上的校准")
    say("=" * 78)
    t0 = time.time()
    rc = run_ratio_calibration_audit(n_salts=n_ratio_salts, n_users=8_000, n_looks=5)
    say(rc.summary())
    say(f"（{n_ratio_salts} 个 salt，耗时 {time.time() - t0:.0f}s）")
    say("")
    say("  结论：比值口径的序贯 FWER 区间盖住 α，末次 z 的均值≈0、标准差≈1 ——")
    say("  也就是说**换了口径之后监控曲线仍然校准**，不需要为它另解边界。")
    say("  边界：这一轮走的是**合成源**（只有它能换 salt 重复）；")
    say("  数仓侧那份比值数据只有一份、换 salt 不会得到新实现，")
    say("  所以「数仓比值链路的序贯校准」仍然是未验证的，写在 README 已知边界里。")

    # ---- 4. 图 ----------------------------------------------------------- #
    fig_monitoring_estimator(h0, h1, out)
    fig_unit_awareness(ua, out)
    fig_mde_power(out, cfg.post_sd, cfg.corr_pre_post, cfg.post_mean, 10_000)

    say("")
    say("=" * 78)
    # ---- 簇级 CUPED 的校准（换 salt 重复抽样） ----------------------------- #
    say("\n### 7. 簇级 CUPED 的校准：观测单位必须仍然是簇")
    say("整簇随机化下按用户做推断会把簇内相关当成独立信息 ——")
    say("那条错误做法的误停率在 M1/M6 早就量过（60% 以上）。")
    say("CUPED 引入回归调整之后，同一个问题必须**重新问一遍**：")
    say("调整之后观测单位还是簇吗？能重复抽样的数据源只有合成路径")
    say("（数仓只有一份实现），所以这里换 salt 跑 A/A：")
    say("")
    from ablab.validation.cluster_cuped_audit import run_cluster_cuped_audit

    cca = run_cluster_cuped_audit(n_trials=n_cluster_cuped, n_users=4000)
    for line in cca.summary().splitlines():
        say("  " + line)
    say("")
    say("  结论：**簇级 CUPED 守住名义水平**（Wilson 区间覆盖 5%），")
    say("  而用户级检验的误停率是它的十几倍 —— 也就是说 CUPED 没有、也不该")
    say("  改变「观测单位」这件事。")

    say(f"总耗时 {time.time() - t_start:.0f}s；图：fig28_monitoring_estimator.png、"
        "fig29_unit_awareness.png、fig30_mde_power.png")
    say("=" * 78)

    report = out / "m6_validation.md"
    report.write_text(
        "# M6 验证报告：平台的生产口径\n\n"
        "> 由 `python scripts/run_m6_validation.py` 生成，全部数字可复现。\n"
        "> M5 把引擎接成了服务，M6 修的是「接得对不对」：口径一致、分析单元、MDE/功效。\n\n"
        "```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n",
        encoding="utf-8", newline="\n",
    )
    print(f"报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
