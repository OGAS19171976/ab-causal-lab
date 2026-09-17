"""序贯检验审计：把每一种停止规则放进同一批仿真里，量它真实的表现。

三种正解，三种不同的取舍
------------------------
``sequential``（群序贯）
    查看次数与消耗函数**事先声明**，边界由此反解。
    换来的是**精确用满 alpha**：实测 FWER 0.0504，功效也最高。
    代价：事后加一次计划外的查看会破坏保证。
``always_valid``（mSPRT）
    不需要事先声明看几次，对**任意**停止规则都有效。
    换来的是**保守**：它保证的是"连续监控"这种最坏情况，
    只看 5 次时用不上那么多，实测 FWER 仅 0.008（看 500 次时才到 0.029）。
``bayesian``（后验阈值）
    后验在任何时刻都自洽 —— 但那句话说的是**概率陈述**，
    不是"反复看到阈值就停"这个**决策规则**的长期误判率。

必须量出来的事实
----------------
**贝叶斯阈值的频率派错误率完全由先验尺度 tau 决定**，
从 0%（紧先验，几乎不可能拒绝）一路涨到 13%（扩散先验，严重偏激进）。
常用的 ``P(delta>0) >= 0.95`` 本身**不是**一个校准的判据 ——
它偏保守还是偏激进，取决于你随手选的 tau。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..sequential import (
    SequentialDesign,
    adjusted_p_value,
    build_design,
    msprt_p_value,
    probability_better,
)
from ..sequential.bayesian import NormalPrior
from ..sim.sequential import default_information_fractions, simulate_canonical_sequences
from .aa import wilson_interval

__all__ = [
    "StopRuleResult",
    "TauPoint",
    "MonitoringPoint",
    "BoundaryAccuracy",
    "run_stopping_rule_comparison",
    "run_tau_sensitivity",
    "run_monitoring_intensity",
    "verify_boundary_accuracy",
    "verify_adjusted_p_value",
]

#: mSPRT 的默认先验尺度：以全样本标准误为单位。见 run_tau_sensitivity。
DEFAULT_TAU_MULTIPLE = 2.0


@dataclass
class StopRuleResult:
    """一种停止规则在一批重复实验上的表现。"""

    label: str
    alpha: float
    n_trials: int
    rejections: int
    looks_used: np.ndarray
    true_effect: float
    is_null: bool

    @property
    def rate(self) -> float:
        """原假设下是 I 类错误率，备择假设下是功效。"""
        return self.rejections / self.n_trials

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.rejections, self.n_trials)

    @property
    def mean_looks(self) -> float:
        return float(np.mean(self.looks_used))

    @property
    def look_saving(self) -> float:
        n = int(self.looks_used.max())
        return 1.0 - self.mean_looks / n

    def summary(self) -> str:
        lo, hi = self.interval
        kind = "I 类错误" if self.is_null else "功效"
        return (
            f"  {self.label}\n"
            f"    {kind} = {self.rate:.4f}  [{lo:.4f}, {hi:.4f}]\n"
            f"    平均查看次数 = {self.mean_looks:.2f}（省下 {self.look_saving:.1%}）"
        )


def _looks_used(mask: np.ndarray) -> np.ndarray:
    """布尔矩阵 -> 每次实验停在第几次；从未命中则记为总次数。"""
    n_looks = mask.shape[1]
    any_hit = mask.any(axis=1)
    first = mask.argmax(axis=1) + 1
    return np.where(any_hit, first, n_looks).astype(int)


def run_stopping_rule_comparison(
    *,
    n_trials: int = 20_000,
    n_looks: int = 5,
    alpha: float = 0.05,
    se_final: float = 0.42,
    effect: float = 0.0,
    spending: str = "obf",
    tau: float | None = None,
    bayesian_threshold: float = 0.95,
    information_fractions: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[list[StopRuleResult], SequentialDesign]:
    """在同一批仿真路径上对比全部停止规则。

    ``effect`` 为 0 时量的是 I 类错误率；非 0 时量的是功效。
    ``tau`` 默认取 ``DEFAULT_TAU_MULTIPLE * se_final``（见模块文档：
    mSPRT 与贝叶斯的功效对 tau 都很敏感，这里只取一个代表值，
    完整的敏感性分析在 ``run_tau_sensitivity``）。
    """
    t = (
        default_information_fractions(n_looks)
        if information_fractions is None
        else np.asarray(information_fractions, dtype=float)
    )
    tau = DEFAULT_TAU_MULTIPLE * se_final if tau is None else float(tau)
    sim = simulate_canonical_sequences(
        n_trials=n_trials,
        information_fractions=t,
        se_final=se_final,
        effect=effect,
        seed=seed,
    )

    design = build_design(alpha=alpha, spending=spending, information_fractions=t)
    z = sim.z_statistics
    is_null = effect == 0.0
    nominal_crit = float(stats.norm.isf(alpha / 2))

    # 只看最后一次：命中只能出现在最后一列
    fixed_mask = np.zeros_like(z, dtype=bool)
    fixed_mask[:, -1] = np.abs(z[:, -1]) >= nominal_crit

    prior = NormalPrior(sd=tau)
    bayes_mask = np.zeros_like(z, dtype=bool)
    for k in range(z.shape[1]):
        p = probability_better(sim.estimates[:, k], se_final / np.sqrt(t[k]), prior)
        # 双边判据：处理明显更好或明显更差都停
        bayes_mask[:, k] = (p >= bayesian_threshold) | (
            (1.0 - p) >= bayesian_threshold
        )

    masks: list[tuple[str, np.ndarray]] = [
        ("naive：每次 |z|>=1.96 就停", np.abs(z) >= nominal_crit),
        ("fixed：只看最后一次", fixed_mask),
        (f"sequential：群序贯 {design.spending_name}", design.crossed(z)),
        (
            f"always_valid：mSPRT (tau={tau:.3f})",
            msprt_p_value(sim.estimates, sim.standard_errors, tau) <= alpha,
        ),
        (
            f"bayesian：P(delta>0)>={bayesian_threshold} (tau={tau:.3f})",
            bayes_mask,
        ),
    ]

    results = [
        StopRuleResult(
            label=label,
            alpha=alpha,
            n_trials=n_trials,
            rejections=int(mask.any(axis=1).sum()),
            looks_used=_looks_used(mask),
            true_effect=float(effect),
            is_null=is_null,
        )
        for label, mask in masks
    ]
    return results, design


# --------------------------------------------------------------------------- #
# 先验尺度敏感性
# --------------------------------------------------------------------------- #
@dataclass
class TauPoint:
    """一个 ``tau`` 取值下两种方法的 I 类错误与功效。"""

    tau: float
    tau_over_se: float
    msprt_fwer: float
    msprt_power: float
    bayes_fwer: float
    bayes_power: float

    def summary(self) -> str:
        return (
            f"  tau={self.tau:>6.3f} (x{self.tau_over_se:>5.2f} SE)  "
            f"mSPRT FWER={self.msprt_fwer:.4f} 功效={self.msprt_power:.4f}  |  "
            f"Bayes FWER={self.bayes_fwer:.4f} 功效={self.bayes_power:.4f}"
        )


def run_tau_sensitivity(
    *,
    n_trials: int = 40_000,
    n_looks: int = 5,
    alpha: float = 0.05,
    se_final: float = 0.42,
    effect_scale: float = 2.0,
    taus: np.ndarray | None = None,
    bayesian_threshold: float = 0.95,
    seed: int = 0,
) -> list[TauPoint]:
    """扫描先验尺度 ``tau``，量两种方法的 FWER 与功效。

    ``effect_scale`` 是备择假设的效应量（以 ``se_final`` 为单位）。
    """
    t = default_information_fractions(n_looks)
    if taus is None:
        taus = se_final * np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]
        )

    null = simulate_canonical_sequences(
        n_trials=n_trials, information_fractions=t, se_final=se_final,
        effect=0.0, seed=seed,
    )
    alt = simulate_canonical_sequences(
        n_trials=n_trials, information_fractions=t, se_final=se_final,
        effect=effect_scale * se_final, seed=seed + 1,
    )

    points: list[TauPoint] = []
    for tau in taus:
        prior = NormalPrior(sd=float(tau))

        m_null = msprt_p_value(null.estimates, null.standard_errors, float(tau)) <= alpha
        m_alt = msprt_p_value(alt.estimates, alt.standard_errors, float(tau)) <= alpha

        b_null_p = probability_better(null.estimates, null.standard_errors, prior)
        b_alt_p = probability_better(alt.estimates, alt.standard_errors, prior)
        # 双边判据，与 run_stopping_rule_comparison 保持一致
        b_null = (b_null_p >= bayesian_threshold) | (
            (1.0 - b_null_p) >= bayesian_threshold
        )
        b_alt = (b_alt_p >= bayesian_threshold) | (
            (1.0 - b_alt_p) >= bayesian_threshold
        )

        points.append(
            TauPoint(
                tau=float(tau),
                tau_over_se=float(tau) / se_final,
                msprt_fwer=float(m_null.any(axis=1).mean()),
                msprt_power=float(m_alt.any(axis=1).mean()),
                bayes_fwer=float(b_null.any(axis=1).mean()),
                bayes_power=float(b_alt.any(axis=1).mean()),
            )
        )
    return points


# --------------------------------------------------------------------------- #
# 监控密度：mSPRT 的代价从哪里来
# --------------------------------------------------------------------------- #
@dataclass
class MonitoringPoint:
    """固定查看次数下的 FWER（H0）。"""

    n_looks: int
    msprt_fwer: float
    sequential_fwer: float
    naive_fwer: float

    def summary(self) -> str:
        return (
            f"  查看 {self.n_looks:>4} 次：naive={self.naive_fwer:.4f}  "
            f"群序贯={self.sequential_fwer:.4f}  mSPRT={self.msprt_fwer:.4f}"
        )


def run_monitoring_intensity(
    *,
    look_counts: tuple[int, ...] = (2, 5, 20, 100, 500),
    n_trials: int = 40_000,
    alpha: float = 0.05,
    se_final: float = 0.42,
    tau_multiple: float = DEFAULT_TAU_MULTIPLE,
    seed: int = 0,
) -> list[MonitoringPoint]:
    """FWER 如何随查看次数变化。

    mSPRT 的保证覆盖"连续监控"，所以查看越密它越接近把 alpha 用满；
    群序贯在每种密度下都精确用满 alpha；naive 则随密度膨胀。
    这条曲线解释了 mSPRT 的保守**不是实现问题，是它买的东西更贵**。

    查看次数很大时自动降低重复次数，避免 ``(n_trials, n_looks)`` 矩阵撑爆内存。
    """
    points: list[MonitoringPoint] = []
    for n_looks in look_counts:
        # 总元素数控制在 4e6 以内，内存与精度之间取平衡
        trials = max(2_000, min(n_trials, int(4e6 / n_looks)))
        t = default_information_fractions(n_looks)
        sim = simulate_canonical_sequences(
            n_trials=trials, information_fractions=t, se_final=se_final,
            effect=0.0, seed=seed,
        )
        design = build_design(alpha=alpha, spending="obf", information_fractions=t)
        crit = float(stats.norm.isf(alpha / 2))
        tau = tau_multiple * se_final

        points.append(
            MonitoringPoint(
                n_looks=int(n_looks),
                msprt_fwer=float(
                    (msprt_p_value(sim.estimates, sim.standard_errors, tau) <= alpha)
                    .any(axis=1)
                    .mean()
                ),
                sequential_fwer=float(design.crossed(sim.z_statistics).any(axis=1).mean()),
                naive_fwer=float((np.abs(sim.z_statistics) >= crit).any(axis=1).mean()),
            )
        )
    return points


# --------------------------------------------------------------------------- #
# 边界与调整 p 值
# --------------------------------------------------------------------------- #
@dataclass
class BoundaryAccuracy:
    """蒙特卡洛验证边界是否真的把累计越界概率控制在消耗函数上。"""

    spending_name: str
    n_trials: int
    information_fractions: np.ndarray
    simulated_cumulative: np.ndarray
    target_cumulative: np.ndarray
    standard_errors: np.ndarray

    @property
    def max_deviation(self) -> float:
        return float(np.max(np.abs(self.simulated_cumulative - self.target_cumulative)))

    @property
    def max_sigma(self) -> float:
        """最大偏差相当于几个蒙特卡洛标准误。"""
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.sqrt(
                self.target_cumulative * (1 - self.target_cumulative) / self.n_trials
            )
            ratio = np.abs(
                self.simulated_cumulative - self.target_cumulative
            ) / np.where(se > 0, se, np.inf)
        return float(np.nanmax(ratio))

    @property
    def passed(self) -> bool:
        return self.max_sigma < 5.0

    def summary(self) -> str:
        lines = [
            f"边界精度验证：{self.spending_name}（{self.n_trials:,} 次模拟）",
            f"{'查看':>4} {'目标累计':>12} {'仿真累计':>12} {'偏差':>12} {'标准误':>10}",
        ]
        for k in range(self.information_fractions.size):
            lines.append(
                f"{k + 1:>4} {self.target_cumulative[k]:>12.6f} "
                f"{self.simulated_cumulative[k]:>12.6f} "
                f"{self.simulated_cumulative[k] - self.target_cumulative[k]:>+12.6f} "
                f"{self.standard_errors[k]:>10.6f}"
            )
        lines.append(
            f"  最大偏差 {self.max_deviation:.2e} = {self.max_sigma:.2f} 个蒙特卡洛标准误"
            f" -> {'PASS' if self.passed else 'FAIL'}"
        )
        return "\n".join(lines)


def verify_boundary_accuracy(
    *,
    spending: str = "obf",
    n_looks: int = 5,
    alpha: float = 0.05,
    n_trials: int = 200_000,
    seed: int = 1,
) -> BoundaryAccuracy:
    """用蒙特卡洛验证边界：累计越界概率是否等于消耗函数。

    算的必须是**至少越界一次**的概率。把各次查看的边际越界概率相加是错的 ——
    它们高度相关，相加会重复计数，从而把 5% 误算成 7.6%
    （这个错误真的发生过，见 M2 的调试记录）。
    """
    design = build_design(alpha=alpha, n_looks=n_looks, spending=spending)
    t = design.information_fractions
    sim = simulate_canonical_sequences(
        n_trials=n_trials, information_fractions=t, se_final=1.0, effect=0.0, seed=seed
    )
    crossed = np.abs(sim.z_statistics) >= design.boundaries
    cumulative = np.array(
        [crossed[:, : k + 1].any(axis=1).mean() for k in range(t.size)]
    )
    target = design.cumulative_spend
    return BoundaryAccuracy(
        spending_name=design.spending_name,
        n_trials=n_trials,
        information_fractions=t,
        simulated_cumulative=cumulative,
        target_cumulative=target,
        standard_errors=np.sqrt(target * (1 - target) / n_trials),
    )


def verify_adjusted_p_value(
    *,
    n_looks: int = 5,
    alpha: float = 0.05,
    spending: str = "obf",
    atol: float = 1e-3,
) -> tuple[bool, list[tuple[int, float, float]]]:
    """验证调整 p 值：把观测值**恰好放在第 k 次边界上**，``p_adj`` 必须等于 alpha。

    这是一个精确的恒等式而不是统计检验：若路径 ``z`` 只在第 k 次等于边界
    ``b_k(alpha)``、其余各次为 0，那么能拒绝它的最小 alpha 就正好是 alpha
    （更小的 alpha 会把 ``b_k`` 抬高，而其余各次为 0 又不可能更早越界）。

    用边界点做检验而不是随机抽样，是因为它对每个 k 都精确、且**快得多** ——
    每个 ``p_adj`` 需要对 alpha 做几十次二分搜索，上千次抽样会慢到不可用。

    ``atol`` 取 1e-3 是因为这里量的是数值求解精度而不是算法正确性：
    网格离散 + 80 步对数二分搜索，相对误差在 1e-4 量级。

    返回 ``(是否全部通过, [(k, 边界值, 得到的 p_adj), ...])``。
    """
    design = build_design(alpha=alpha, n_looks=n_looks, spending=spending)
    t = design.information_fractions

    rows: list[tuple[int, float, float]] = []
    ok = True
    for k in range(n_looks):
        z = np.zeros(n_looks)
        z[k] = design.boundaries[k]
        p = adjusted_p_value(z, information_fractions=t, spending=spending)
        rows.append((k + 1, float(design.boundaries[k]), float(p)))
        if abs(p - alpha) > atol:
            ok = False
    return ok, rows
