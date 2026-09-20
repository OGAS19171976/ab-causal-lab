"""M1 方法校准台：把每个新方法放进 A/A 仿真里，看它到底校准不校准。

M0 证明了"朴素 post-only 检验在随机化意义下是校准的"。
M1 加的每一个方法都要回答同一个问题：**它自己的 I 类错误率是不是 5%？**
回答不了这个问题的实现，无论代码多漂亮都不该进主干。

三组对比，每组都是"naive 做法 vs 正确做法"：

============================  ==========================  ==========================
场景                          naive 做法                   正确做法
============================  ==========================  ==========================
实验前指标可用                 post-only t 检验             CUPED
指标是比值（CTR）              用户级 r_i 的 t 检验          delta method（线性化）
处理在簇级别分配               用户级 t 检验                CR1 聚类稳健 / 簇级检验
============================  ==========================  ==========================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..assignment import ExperimentSpec, Randomizer
from ..inference import (
    AggregateStats,
    CupedFit,
    cluster_level_ttest,
    cluster_robust_ttest,
    cuped_ttest,
    naive_unit_ratio_ttest,
    ratio_delta_method,
    welch_ttest,
)
from ..sim import Population, simulate_outcomes_treated, two_arm_spec
from ..sim.scenarios import (
    ClusterSample,
    ClusterScenarioConfig,
    RatioSample,
    RatioScenarioConfig,
    generate_cluster_scenario,
    generate_ratio_scenario,
)
from .aa import fresh_spec, make_key_batcher, wilson_interval

__all__ = [
    "MethodTrials",
    "MethodComparison",
    "BiasDecomposition",
    "CUPED_NAIVE",
    "CUPED_LABEL",
    "RATIO_DELTA",
    "RATIO_NAIVE",
    "CLUSTER_NAIVE",
    "CLUSTER_ROBUST",
    "CLUSTER_LEVEL",
    "run_cuped_comparison",
    "run_cuped_bias_decomposition",
    "run_cuped_power_comparison",
    "run_ratio_comparison",
    "run_ratio_power_comparison",
    "run_cluster_comparison",
]

# 方法名在这里定义一次，脚本和测试都从这里取 —— 避免字符串写错却只在运行时才发现
CUPED_NAIVE = "post-only t 检验"
CUPED_LABEL = "CUPED + Welch"
RATIO_DELTA = "delta method（业务口径 Σy/Σx）"
RATIO_NAIVE = "用户级比值 t 检验（人均比值）"
CLUSTER_NAIVE = "用户级 t 检验（反例）"
CLUSTER_ROBUST = "CR1 聚类稳健"
CLUSTER_LEVEL = "簇级 Welch"


# --------------------------------------------------------------------------- #
# 一种方法在 N 次重复实验上的表现
# --------------------------------------------------------------------------- #
@dataclass
class MethodTrials:
    """一种分析方法在 ``n_trials`` 次重复实验上的表现。"""

    label: str
    alpha: float
    true_effect: float
    p_values: np.ndarray
    effects: np.ndarray
    std_errors: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    #: 对照组的点估计水平。naive 与正确做法估计不同口径时，差距体现在这里。
    control_levels: np.ndarray | None = None

    @property
    def n_trials(self) -> int:
        return int(self.p_values.size)

    @property
    def control_level(self) -> float:
        """对照组点估计的平均水平（也就是这个方法的"口径"）。"""
        if self.control_levels is None:
            return float("nan")
        return float(np.mean(self.control_levels))

    # -- 校准 -------------------------------------------------------------- #
    def fpr(self, alpha: float | None = None) -> float:
        a = self.alpha if alpha is None else alpha
        return float(np.mean(self.p_values < a))

    def fpr_interval(self, alpha: float | None = None) -> tuple[float, float]:
        a = self.alpha if alpha is None else alpha
        return wilson_interval(int(np.sum(self.p_values < a)), self.n_trials)

    def coverage(self) -> float:
        inside = (self.ci_low <= self.true_effect) & (self.ci_high >= self.true_effect)
        return float(np.mean(inside))

    def coverage_interval(self) -> tuple[float, float]:
        inside = (self.ci_low <= self.true_effect) & (self.ci_high >= self.true_effect)
        return wilson_interval(int(inside.sum()), self.n_trials)

    def uniformity_test(self) -> tuple[float, float]:
        res = stats.kstest(self.p_values, "uniform")
        return float(res.statistic), float(res.pvalue)

    # -- 估计量的抽样分布 -------------------------------------------------- #
    @property
    def mean_effect(self) -> float:
        return float(np.mean(self.effects))

    @property
    def sd_effect(self) -> float:
        return float(np.std(self.effects, ddof=1))

    @property
    def mean_se(self) -> float:
        """标准误估计的平均值。应与 ``sd_effect`` 吻合。"""
        return float(np.mean(self.std_errors))

    @property
    def bias(self) -> float:
        return self.mean_effect - self.true_effect

    def cumulative_fpr(self, alpha: float | None = None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        return np.cumsum(self.p_values < a) / np.arange(1, self.n_trials + 1)

    def summary(self, name: str | None = None) -> str:
        lo, hi = self.fpr_interval()
        clo, chi = self.coverage_interval()
        _d, ksp = self.uniformity_test()
        head = name or self.label
        return (
            f"  {head}\n"
            f"    I 类错误 = {self.fpr():.4f} [{lo:.4f}, {hi:.4f}]   "
            f"覆盖率 = {self.coverage():.4f} [{clo:.4f}, {chi:.4f}]\n"
            f"    效应均值 = {self.mean_effect:+.4f}（真值 {self.true_effect:+.4f}，"
            f"偏置 {self.bias:+.4f}）\n"
            f"    效应 SD = {self.sd_effect:.4f}   平均 SE = {self.mean_se:.4f}\n"
            f"    p 值均匀性 KS p = {ksp:.4g}"
        )


@dataclass
class MethodComparison:
    """一组方法在同一批重复实验上的横向对比。"""

    title: str
    mode: str
    methods: tuple[MethodTrials, ...]
    notes: tuple[str, ...] = ()
    #: 方法特有的附加产物（例如 CUPED 的拟合结果）
    extras: dict = field(default_factory=dict)

    def get(self, label: str) -> MethodTrials:
        for m in self.methods:
            if m.label == label:
                return m
        raise KeyError(f"没有名为 {label!r} 的方法；可选 {[m.label for m in self.methods]}")

    def summary(self) -> str:
        lines = [f"{self.title}  [{self.mode}]", f"  重复 {self.methods[0].n_trials:,} 次"]
        for m in self.methods:
            lines.append(m.summary())
            if m.control_levels is not None:
                lines.append(f"    对照口径水平 = {m.control_level:.6f}")
        for note in self.notes:
            lines.append(f"  * {note}")
        return "\n".join(lines)


def _collect(
    label: str,
    estimates,
    *,
    true_effect: float,
    alpha: float,
    with_levels: bool = False,
) -> MethodTrials:
    return MethodTrials(
        label=label,
        alpha=alpha,
        true_effect=true_effect,
        p_values=np.array([e.p_value for e in estimates]),
        effects=np.array([e.absolute_effect for e in estimates]),
        std_errors=np.array([e.std_error for e in estimates]),
        ci_low=np.array([e.ci_low for e in estimates]),
        ci_high=np.array([e.ci_high for e in estimates]),
        control_levels=(
            np.array([e.mean_control for e in estimates]) if with_levels else None
        ),
    )


# --------------------------------------------------------------------------- #
# 一、CUPED
# --------------------------------------------------------------------------- #
def run_cuped_comparison(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    n_trials: int = 1_000,
    mode: str = "randomized",
    alpha: float = 0.05,
    true_lift: float = 0.0,
    seed: int = 7,
) -> MethodComparison:
    """同一批重复实验上对比 post-only 与 CUPED。

    ``mode="randomized"`` 才能测出**方差缩减**（它是对随机化取期望的性质）；
    ``mode="conditional"`` 用来测**偏置消除**（固定分流下的协变量失衡）。

    CUPED 的拟合结果（theta / rho / 方差缩减）放在 ``result.extras["cuped_fit"]``。
    """
    if mode not in ("randomized", "conditional"):
        raise ValueError(f"未知 mode: {mode!r}")

    spec = spec or two_arm_spec("cuped_exp", salt="cuped_exp_v1")
    ids = population.ids()
    batcher = make_key_batcher(ids)
    rz = Randomizer()
    rng = np.random.default_rng(seed)
    n_variants = len(spec.variants)
    treat_code, ctrl_code = n_variants - 1, 0

    fixed_codes = rz.assign_codes(ids, spec, batcher) if mode == "conditional" else None

    naive, cuped = [], []
    fit: CupedFit | None = None

    for i in range(n_trials):
        codes = (
            fixed_codes
            if fixed_codes is not None
            else rz.assign_codes(ids, fresh_spec(spec, f"c{i}"), batcher)
        )
        treated = codes == treat_code
        control = codes == ctrl_code

        post = simulate_outcomes_treated(
            population, treated, true_lift=true_lift, seed=int(rng.integers(1 << 31))
        )
        pre = population.pre_metric

        naive.append(welch_ttest(post[treated], post[control], alpha=alpha))
        estimate, fit = cuped_ttest(
            pre[treated], post[treated], pre[control], post[control], alpha=alpha
        )
        cuped.append(estimate)

    assert fit is not None
    return MethodComparison(
        title="CUPED vs post-only",
        mode=mode,
        methods=(
            _collect(CUPED_NAIVE, naive, true_effect=true_lift, alpha=alpha),
            _collect(CUPED_LABEL, cuped, true_effect=true_lift, alpha=alpha),
        ),
        extras={"cuped_fit": fit},
    )


def run_cuped_power_comparison(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    lifts: np.ndarray | None = None,
    n_trials: int = 300,
    alpha: float = 0.05,
    seed: int = 11,
):
    """在效应量网格上对比 post-only 与 CUPED 的**检出率（功效）**。

    返回 ``[(lift, naive_power, cuped_power, naive_se, cuped_se), ...]``。
    CUPED 的方差更小 → 同样样本量下功效更高，
    等价于"用更少的流量得出同样结论"。
    """
    lifts = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5]) if lifts is None else np.asarray(lifts)
    rows = []
    for j, lift in enumerate(lifts):
        comparison = run_cuped_comparison(
            population,
            spec,
            n_trials=n_trials,
            mode="randomized",
            alpha=alpha,
            true_lift=float(lift),
            seed=seed + j,
        )
        naive = comparison.get(CUPED_NAIVE)
        cuped = comparison.get(CUPED_LABEL)
        rows.append(
            {
                "lift": float(lift),
                "naive_power": naive.fpr(),
                "cuped_power": cuped.fpr(),
                "naive_se": naive.mean_se,
                "cuped_se": cuped.mean_se,
            }
        )
    return rows


def run_cuped_bias_decomposition(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    n_assignments: int = 200,
    n_noise: int = 8,
    alpha: float = 0.05,
    seed: int = 91,
) -> "BiasDecomposition":
    """跨随机化的偏置分解：证明 naive 的偏置**完全由前置失衡驱动**，CUPED 不是。

    单次实现的失衡大小是随机的，可能恰好接近 0 —— 那一次就没法演示"消偏置"。
    稳健的做法是对 ``n_assignments`` 次**独立分流**各跑 ``n_noise`` 次噪声重复，
    得到 (前置组间差, 效应估计) 的散点，再回归：

    * naive 的效应对前置差距的**斜率应等于 beta**（偏置 = beta × 失衡），
      且统计上极显著；
    * CUPED 的斜率应**统计上不显著于 0** —— 偏置被整项扣掉。

    斜率是比"均值偏置"稳健得多的判据：它不依赖某一次实现失衡的大小。
    """

    spec = spec or two_arm_spec("bias_decomp", salt="bias_decomp_v1")
    ids = population.ids()
    batcher = make_key_batcher(ids)
    rz = Randomizer()
    rng = np.random.default_rng(seed)
    n_variants = len(spec.variants)
    pre = population.pre_metric

    pre_gaps = np.empty(n_assignments)
    naive_effects = np.empty(n_assignments)
    cuped_effects = np.empty(n_assignments)

    for k in range(n_assignments):
        codes = rz.assign_codes(ids, fresh_spec(spec, f"bd{k}"), batcher)
        treated = codes == (n_variants - 1)
        control = codes == 0

        pre_gaps[k] = pre[treated].mean() - pre[control].mean()

        naive_k = np.empty(n_noise)
        cuped_k = np.empty(n_noise)
        for j in range(n_noise):
            post = simulate_outcomes_treated(
                population, treated, true_lift=0.0, seed=int(rng.integers(1 << 31))
            )
            naive_k[j] = post[treated].mean() - post[control].mean()
            est, _fit = cuped_ttest(
                pre[treated], post[treated], pre[control], post[control], alpha=alpha
            )
            cuped_k[j] = est.absolute_effect

        naive_effects[k] = naive_k.mean()
        cuped_effects[k] = cuped_k.mean()

    def _regress(y: np.ndarray) -> tuple[float, float, float]:
        res = stats.linregress(pre_gaps, y)
        return float(res.slope), float(res.stderr), float(res.pvalue)

    n_slope, n_se, n_p = _regress(naive_effects)
    c_slope, c_se, c_p = _regress(cuped_effects)

    # 理论斜率 beta = Corr(pre, post) * sd(post) / sd(pre)
    var_pre = float(np.var(pre, ddof=1))
    beta = population.config.corr_pre_post * population.config.post_sd / population.config.pre_sd

    return BiasDecomposition(
        pre_gaps=pre_gaps,
        naive_effects=naive_effects,
        cuped_effects=cuped_effects,
        naive_slope=n_slope,
        naive_slope_se=n_se,
        naive_slope_p=n_p,
        cuped_slope=c_slope,
        cuped_slope_se=c_se,
        cuped_slope_p=c_p,
        theoretical_slope=float(beta),
        var_pre=var_pre,
        n_assignments=n_assignments,
        n_noise=n_noise,
    )


@dataclass
class BiasDecomposition:
    """跨随机化的偏置分解结果。"""

    pre_gaps: np.ndarray
    naive_effects: np.ndarray
    cuped_effects: np.ndarray

    naive_slope: float
    naive_slope_se: float
    naive_slope_p: float
    cuped_slope: float
    cuped_slope_se: float
    cuped_slope_p: float

    theoretical_slope: float
    var_pre: float
    n_assignments: int
    n_noise: int

    @property
    def naive_bias_is_driven_by_imbalance(self) -> bool:
        """naive 的斜率必须显著、且与理论 beta 吻合。"""
        return (
            self.naive_slope_p < 1e-6
            and abs(self.naive_slope - self.theoretical_slope) < 3 * self.naive_slope_se
        )

    @property
    def cuped_is_free_of_imbalance_bias(self) -> bool:
        """CUPED 的斜率必须统计上不显著于 0。"""
        return self.cuped_slope_p > 0.01

    @property
    def passed(self) -> bool:
        return self.naive_bias_is_driven_by_imbalance and self.cuped_is_free_of_imbalance_bias

    def summary(self) -> str:
        return (
            f"CUPED 偏置分解 [{self.n_assignments} 次独立分流 × {self.n_noise} 次噪声重复]\n"
            f"  回归：效应 ~ 前置组间差\n"
            f"    post-only 斜率 = {self.naive_slope:.4f} "
            f"(SE {self.naive_slope_se:.4f}, p={self.naive_slope_p:.3g})\n"
            f"      理论 beta = {self.theoretical_slope:.4f}"
            f"  -> {'吻合' if self.naive_bias_is_driven_by_imbalance else '不吻合 <<<'}\n"
            f"    CUPED 斜率     = {self.cuped_slope:.4f} "
            f"(SE {self.cuped_slope_se:.4f}, p={self.cuped_slope_p:.3g})\n"
            f"      -> {'与 0 无显著差异（偏置被整项扣掉）' if self.cuped_is_free_of_imbalance_bias else '仍显著 <<<'}\n"
            f"  含义：naive 的偏置完全来自「前置失衡 × beta」，CUPED 把这一项扣掉了\n"
        )


# --------------------------------------------------------------------------- #
# 二、比值指标
# --------------------------------------------------------------------------- #
def run_ratio_comparison(
    config: RatioScenarioConfig | None = None,
    *,
    n_trials: int = 1_000,
    alpha: float = 0.05,
    relative_lift: float = 0.0,
    seed: int = 21,
) -> MethodComparison:
    """对比"用户级比值 t 检验"与"delta method"。

    **每次重复重新抽一批用户**（超总体框架），而不是固定样本只换分流。

    这个选择不是随意的：delta method 是"单元从超总体独立同分布抽样"的
    渐近结果，它的 5% 只在这个重复抽样框架下成立。M0 已经教过同一条原则 ——
    t 检验的 5% 是对随机化取期望的边际保证，所以校准检验必须重新随机化；
    这里同理，必须重新抽样。固定样本只换分流会让 delta method 偏保守，
    那不是方法有问题，是检验方法的时候用错了框架。
    """
    cfg = config or RatioScenarioConfig()
    rng = np.random.default_rng(seed)

    delta_estimates, naive_estimates = [], []
    sample: RatioSample | None = None

    for i in range(n_trials):
        trial_seed = int(rng.integers(1 << 31))
        sample = generate_ratio_scenario(
            cfg,
            relative_lift=relative_lift,
            salt=f"ratio_aa_{i}",
            seed=trial_seed,
        )
        t, c = sample.treated, ~sample.treated
        delta_estimates.append(
            ratio_delta_method(
                AggregateStats.from_arrays(sample.clicks[t], sample.views[t]),
                AggregateStats.from_arrays(sample.clicks[c], sample.views[c]),
                alpha=alpha,
            )
        )
        naive_estimates.append(
            naive_unit_ratio_ttest(
                sample.clicks[t], sample.views[t], sample.clicks[c], sample.views[c],
                alpha=alpha,
            ).estimate
        )

    assert sample is not None

    # 两个方法估计的是**不同的量**，所以真值也必须各给各的，
    # 否则覆盖率检验会拿错参照系。
    pooled_control = sample.pooled_ratio(treated=False)
    unit_control = sample.mean_unit_ratio(treated=False)
    gap = sample.estimand_gap(treated=False)

    delta = _collect(
        RATIO_DELTA,
        delta_estimates,
        true_effect=pooled_control * relative_lift,
        alpha=alpha,
        with_levels=True,
    )
    naive = _collect(
        RATIO_NAIVE,
        naive_estimates,
        true_effect=unit_control * relative_lift,
        alpha=alpha,
        with_levels=True,
    )

    se_ratio = naive.mean_se / delta.mean_se
    if se_ratio > 1.05:
        precision_note = f"naive 的标准误是 delta method 的 {se_ratio:.2f} 倍（更不精确）"
    elif se_ratio < 0.95:
        precision_note = (
            f"naive 的标准误反而更小（{se_ratio:.2f} 倍）—— "
            "但它在估另一个量，更精确地答错问题并不能救回来"
        )
    else:
        precision_note = f"两者精确度接近（SE 之比 {se_ratio:.2f}）"

    return MethodComparison(
        title="比值指标：delta method vs 用户级比值 t 检验",
        mode="randomized（每次重抽一批用户）",
        methods=(delta, naive),
        notes=(
            f"曝光量分布：均值 {sample.views.mean():.2f}，"
            f"中位数 {np.median(sample.views):.0f}，"
            f"P95 {np.percentile(sample.views, 95):.0f}（重尾）",
            f"两种口径在对照组上：业务口径 Σy/Σx = {pooled_control:.6f}，"
            f"人均比值 mean(y_i/x_i) = {unit_control:.6f}",
            f"**口径差距 = {gap:+.6f}（相对 {gap / pooled_control:+.2%}）**"
            " —— naive 做法给出的是另一个业务数字",
            precision_note,
            "注意：naive 的 I 类错误也是校准的。"
            "**校准不等于正确** —— 一个检验可以既不偏高也不偏低，"
            "却系统性地报出错误的量。这是本节最值得记住的一句。",
        ),
        # 单次实现的口径差（上面 notes 里印的那个数）与「重复若干次之后
        # control_level 的平均之比」**不是同一个量**：前者是一次抽样里
        # Σy/Σx 与 mean(y_i/x_i) 之差，后者是各次重复的均值之差。
        # 两个都放进 extras，让调用方能把它们分别标清楚 ——
        # 否则同一份报告里会出现两个都叫"口径差"的数（M1 报告曾经如此：
        # notes 里 -12.61%、结论行 -12.77%，读者无从判断哪个是哪个）。
        extras={
            "pooled_control": pooled_control,
            "unit_control": unit_control,
            "estimand_gap": gap,
            "single_realization_gap_relative": gap / pooled_control,
        },
    )


def run_ratio_power_comparison(
    config: RatioScenarioConfig | None = None,
    *,
    relative_lifts: np.ndarray | None = None,
    n_trials: int = 300,
    alpha: float = 0.05,
    seed: int = 23,
):
    """在相对提升网格上对比 delta method 与 naive 的检出率。

    返回 ``[(relative_lift, delta_power, naive_power), ...]``。
    """
    cfg = config or RatioScenarioConfig()
    lifts = (
        np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
        if relative_lifts is None
        else np.asarray(relative_lifts)
    )
    rows = []
    for j, lift in enumerate(lifts):
        comparison = run_ratio_comparison(
            cfg, n_trials=n_trials, alpha=alpha, relative_lift=float(lift), seed=seed + j
        )
        rows.append(
            {
                "relative_lift": float(lift),
                "delta_power": comparison.get(RATIO_DELTA).fpr(),
                "naive_power": comparison.get(RATIO_NAIVE).fpr(),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 三、聚类随机化
# --------------------------------------------------------------------------- #
def run_cluster_comparison(
    config: ClusterScenarioConfig | None = None,
    *,
    n_trials: int = 500,
    alpha: float = 0.05,
    true_lift: float = 0.0,
    seed: int = 31,
) -> MethodComparison:
    """对比"用户级 t 检验"与两种正确处理（CR1 / 簇级）。

    **每次重复重新抽一批簇**（簇级随机效应与用户级噪声都重抽），
    这是聚类稳健标准误所声称的重复抽样框架：簇本身也是从超总体抽来的。
    """
    cfg = config or ClusterScenarioConfig()
    rng = np.random.default_rng(seed)

    naive, robust, cluster_level = [], [], []
    sample: ClusterSample | None = None

    for i in range(n_trials):
        sample = generate_cluster_scenario(
            cfg, true_lift=true_lift, salt=f"cluster_aa_{i}", seed=int(rng.integers(1 << 31))
        )
        naive.append(welch_ttest(sample.outcome[sample.treated], sample.outcome[~sample.treated], alpha=alpha))
        robust.append(
            cluster_robust_ttest(sample.cluster_id, sample.treated, sample.outcome, alpha=alpha)
        )
        cluster_level.append(
            cluster_level_ttest(sample.cluster_id, sample.treated, sample.outcome, alpha=alpha)
        )

    assert sample is not None
    icc, m0 = (
        robust[0].diagnostics[0].detail["icc"],
        cfg.users_per_cluster,
    )
    deff = 1.0 + (m0 - 1.0) * icc
    return MethodComparison(
        title="聚类随机化：用户级 t 检验 vs 聚类稳健 vs 簇级",
        mode="randomized",
        methods=(
            _collect(CLUSTER_NAIVE, naive, true_effect=true_lift, alpha=alpha),
            _collect(CLUSTER_ROBUST, robust, true_effect=true_lift, alpha=alpha),
            _collect(CLUSTER_LEVEL, cluster_level, true_effect=true_lift, alpha=alpha),
        ),
        notes=(
            f"簇数 {sample.n_clusters}，每簇 {cfg.users_per_cluster} 用户，"
            f"簇级 SD={cfg.cluster_sd}, 用户级 SD={cfg.user_sd}",
            f"估计 ICC ≈ {icc:.4f}  ->  设计效应 deff ≈ {deff:.2f}，"
            f"朴素标准误被低估约 {np.sqrt(deff):.2f} 倍",
        ),
    )
