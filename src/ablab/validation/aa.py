"""仿真台：A/A 校准、功效、效应估计的抽样分布。

这是整个项目的地基。任何"我的 A/B 框架是对的"的主张，
都必须由这里产出的数字支撑 —— 而不是由"跑通了一个真实数据集"支撑。

两种重复抽样模式，回答两个**不同**的问题，别混用：

``randomized``（默认，标准 A/A 校准检验）
    每次重复都**重新分流**（换 salt 再哈希一遍）并重抽结果噪声。
    这才是 t 检验那 5% 的 I 类错误率所对应的重复抽样框架 ——
    频率派保证是"对随机化取期望"的，所以校准检验也必须重新随机化。

``conditional``（分流固定，只重抽噪声）
    **不是校准检验。** 频率派那 5% 是"**对随机化取期望**"的**边际**保证；
    一旦把某一次实现的分流固定住，各组前置协变量的差异就被冻结成一个
    **固定偏置**，而 t 检验的标准误仍然把前置协变量的方差当成随机波动。

    结果是：这一次实验的实际错误率可以远高于、也可以远低于 5%，
    方向完全取决于这次分流碰巧失衡到什么程度（偏置 ≈ beta × 前置组间差），
    而你永远无从知道落在哪一边。实测 20k 用户下偏保守、5k 用户下偏激进，
    两种都可能出现。

    这正是"CUPED 为什么值得做"最直接的论证：用实验前指标把那部分方差扣掉，
    偏置项和虚高的方差同时消失，无论这次分流偏成什么样都救得回来。M1 接上这条线。

性能：分流用 ``Randomizer.assign_codes`` 的整数编码 + ``KeyBatcher``
复用 unit_id 字节，20k 用户单次重新分流约 3ms。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import stats

from ..assignment import ExperimentSpec, Randomizer
from ..hashing import KeyBatcher
from ..inference import se_of_mean_diff, welch_ttest_from_stats, z_power
from ..sim.generator import Population, simulate_outcomes_treated, two_arm_spec

__all__ = [
    "AAResult",
    "PowerResult",
    "run_aa_trials",
    "run_power_trials",
    "power_curve",
    "wilson_interval",
    "fresh_spec",
    "make_key_batcher",
]

Mode = Literal["randomized", "conditional"]

_MODES = ("randomized", "conditional")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """比例的 Wilson 区间。

    比正态近似更可靠：I 类错误这类比例接近 0.05，
    而我们希望"经验 FPR 是否等于 5%"这句话本身带不确定性时，
    Wilson 在小样本和边界附近表现良好。
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


def _welch(t_vals: np.ndarray, c_vals: np.ndarray, alpha: float):
    """仿真循环里调用共享的统计实现，保证与正式分析路径逐位一致。

    这里刻意**不**另写一份快速公式：一旦仿真台用的公式和线上用的不是同一份，
    仿真验证的意义就归零了。实测这点对象构造开销在千次量级完全可以忽略。
    """
    return welch_ttest_from_stats(
        n_treatment=int(t_vals.size),
        mean_treatment=float(t_vals.mean()),
        var_treatment=float(t_vals.var(ddof=1)),
        n_control=int(c_vals.size),
        mean_control=float(c_vals.mean()),
        var_control=float(c_vals.var(ddof=1)),
        alpha=alpha,
    )


def fresh_spec(spec: ExperimentSpec, suffix: str) -> ExperimentSpec:
    """派生一个换了 salt 的等价实验定义，用于重新分流。"""
    return replace(spec, salt=f"{spec.salt_}#{suffix}")


def make_key_batcher(unit_ids: list[str]) -> KeyBatcher | None:
    """等宽 unit_id 时建一个可复用编码缓冲，加速上千次重新分流。"""
    try:
        return KeyBatcher(unit_ids)
    except ValueError:
        return None


@dataclass
class _Arms:
    """一次试验的分组掩码。"""

    treated: np.ndarray
    control: np.ndarray

    @property
    def n_treated(self) -> int:
        return int(self.treated.sum())

    @property
    def n_control(self) -> int:
        return int(self.control.sum())

    def split(self, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return post[self.treated], post[self.control]


def _arms_from_codes(codes: np.ndarray, n_variants: int) -> _Arms:
    return _Arms(treated=codes == (n_variants - 1), control=codes == 0)


# --------------------------------------------------------------------------- #
# A/A 仿真
# --------------------------------------------------------------------------- #
@dataclass
class AAResult:
    """一次 A/A 仿真的全部产出。"""

    p_values: np.ndarray
    effects: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    covers_truth: np.ndarray
    alpha: float
    n_units: int
    mean_n_per_arm: float
    mode: str
    true_effect: float = 0.0

    def __len__(self) -> int:
        return int(self.p_values.size)

    @property
    def n_trials(self) -> int:
        return int(self.p_values.size)

    # -- 核心指标 ---------------------------------------------------------- #
    def empirical_fpr(self, alpha: float | None = None) -> float:
        a = self.alpha if alpha is None else alpha
        return float(np.mean(self.p_values < a))

    def fpr_interval(self, alpha: float | None = None) -> tuple[float, float]:
        a = self.alpha if alpha is None else alpha
        return wilson_interval(int(np.sum(self.p_values < a)), self.n_trials)

    def coverage(self) -> float:
        """置信区间覆盖率：真值应落在区间内的比例。"""
        return float(np.mean(self.covers_truth))

    def coverage_interval(self) -> tuple[float, float]:
        return wilson_interval(int(np.sum(self.covers_truth)), self.n_trials)

    def cumulative_fpr(self, alpha: float | None = None) -> np.ndarray:
        """前 k 次重复的经验 FPR，用于画"收敛到 5%"的曲线。"""
        a = self.alpha if alpha is None else alpha
        return np.cumsum(self.p_values < a) / np.arange(1, self.n_trials + 1)

    def cumulative_coverage(self) -> np.ndarray:
        return np.cumsum(self.covers_truth) / np.arange(1, self.n_trials + 1)

    def uniformity_test(self) -> tuple[float, float]:
        """原假设下 p 值应服从 Uniform(0,1)，返回 KS 统计量与 p 值。

        这是比"I 类错误约等于 5%"更强的证据：它检验的是**整个分布**，
        任何形状的偏离（保守、激进、U 型）都会被抓住。
        """
        res = stats.kstest(self.p_values, "uniform")
        return float(res.statistic), float(res.pvalue)

    # -- 效应估计的抽样分布（另一条独立证据） ------------------------------ #
    @property
    def mean_effect(self) -> float:
        """效应估计的均值。真值为 0，所以应约等于 0。"""
        return float(np.mean(self.effects))

    @property
    def sd_effect(self) -> float:
        """效应估计的标准差，应与理论 SE 一致。"""
        return float(np.std(self.effects, ddof=1))

    def summary(self) -> str:
        fpr = self.empirical_fpr()
        lo, hi = self.fpr_interval()
        cov = self.coverage()
        clo, chi = self.coverage_interval()
        ks_stat, ks_p = self.uniformity_test()
        tag = (
            "对随机化取期望的标准校准检验"
            if self.mode == "randomized"
            else "固定一次分流：误差率取决于这次实现的不平衡，方向不可预知"
        )
        lines = [
            f"A/A 仿真 [{self.mode}] {tag}",
            f"  n_trials={self.n_trials:,}  n_units={self.n_units:,}  "
            f"每组均值 {self.mean_n_per_arm:,.0f}",
            f"  经验 I 类错误 = {fpr:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   名义 {self.alpha}",
            f"  区间覆盖率   = {cov:.4f}   95% CI [{clo:.4f}, {chi:.4f}]   名义 {1 - self.alpha:.2f}",
            f"  效应估计均值 = {self.mean_effect:+.4f}   标准差 = {self.sd_effect:.4f}",
            f"  p 值均匀性   = KS D={ks_stat:.4f}, p={ks_p:.4g}  "
            f"({'无法拒绝均匀分布 -> 检验校准正常' if ks_p > 0.05 else '拒绝均匀分布 -> 检验未校准'})",
        ]
        if self.mode == "conditional":
            lines.append(
                "  注：固定分流下误差率偏离名义值的方向由这一次实现的协变量失衡决定，"
                "不能据此判断检验本身有偏 —— 必须看 randomized 模式。"
            )
        return "\n".join(lines) + "\n"


def run_aa_trials(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    n_trials: int = 2_000,
    alpha: float = 0.05,
    mode: Mode = "randomized",
    seed: int = 7,
) -> AAResult:
    """跑 ``n_trials`` 次 A/A 实验（真效应为 0），统计假阳性行为。

    ``mode="randomized"``（默认）每次重新分流，是标准校准检验；
    ``mode="conditional"`` 固定一次分流，用于观察前置协变量失衡的影响。
    """
    if n_trials < 1:
        raise ValueError("n_trials 必须为正")
    if mode not in _MODES:
        raise ValueError(f"未知 mode: {mode!r}，可选 {_MODES}")

    spec = spec or two_arm_spec("aa_test", salt="aa_test_v1")
    n_variants = len(spec.variants)
    ids = population.ids()
    batcher = make_key_batcher(ids)
    rz = Randomizer()
    rng = np.random.default_rng(seed)

    fixed_codes = rz.assign_codes(ids, spec, batcher) if mode == "conditional" else None

    p_values = np.empty(n_trials)
    effects = np.empty(n_trials)
    ci_low = np.empty(n_trials)
    ci_high = np.empty(n_trials)
    n_arm_sum = 0
    n_valid = 0

    for i in range(n_trials):
        codes = (
            fixed_codes
            if fixed_codes is not None
            else rz.assign_codes(ids, fresh_spec(spec, f"r{i}"), batcher)
        )
        arms = _arms_from_codes(codes, n_variants)
        post = simulate_outcomes_treated(
            population, arms.treated, true_lift=0.0, seed=int(rng.integers(1 << 31))
        )
        t_vals, c_vals = arms.split(post)

        if t_vals.size < 2 or c_vals.size < 2:  # pragma: no cover - 防御
            p_values[i] = np.nan
            effects[i] = ci_low[i] = ci_high[i] = np.nan
            continue

        n_arm_sum += t_vals.size + c_vals.size
        n_valid += 1
        est = _welch(t_vals, c_vals, alpha)
        p_values[i] = est.p_value
        effects[i] = est.absolute_effect
        ci_low[i] = est.ci_low
        ci_high[i] = est.ci_high

    valid = np.isfinite(p_values)
    if not valid.any():  # pragma: no cover - 防御
        raise RuntimeError("所有重复都失败，请检查样本量设置")

    covers = (ci_low <= 0.0) & (ci_high >= 0.0)

    return AAResult(
        p_values=p_values[valid],
        effects=effects[valid],
        ci_low=ci_low[valid],
        ci_high=ci_high[valid],
        covers_truth=covers[valid],
        alpha=alpha,
        n_units=len(population),
        mean_n_per_arm=n_arm_sum / (2 * n_valid),
        mode=mode,
    )


# --------------------------------------------------------------------------- #
# 功效仿真
# --------------------------------------------------------------------------- #
@dataclass
class PowerResult:
    """给定真实效应量下的功效仿真结果。"""

    true_lift: float
    alpha: float
    n_trials: int
    detections: int
    empirical_power: float
    analytic_power: float
    mean_n_per_arm: float
    post_sd: float
    mode: str

    @property
    def power_interval(self) -> tuple[float, float]:
        return wilson_interval(self.detections, self.n_trials)

    @property
    def gap(self) -> float:
        return self.empirical_power - self.analytic_power

    def summary(self) -> str:
        lo, hi = self.power_interval
        return (
            f"功效仿真 lift={self.true_lift:+.3f} [{self.mode}] "
            f"(n/arm≈{self.mean_n_per_arm:,.0f}, sd={self.post_sd:.2f})\n"
            f"  经验功效 = {self.empirical_power:.4f}  95% CI [{lo:.4f}, {hi:.4f}]\n"
            f"  解析功效 = {self.analytic_power:.4f}   偏差 {self.gap:+.4f}\n"
        )


def run_power_trials(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    true_lift: float,
    n_trials: int = 500,
    alpha: float = 0.05,
    mode: Mode = "randomized",
    seed: int = 11,
) -> PowerResult:
    """在给定效应量下跑 A/B，统计检出率，并与解析功效对比。"""
    if mode not in _MODES:
        raise ValueError(f"未知 mode: {mode!r}，可选 {_MODES}")

    spec = spec or two_arm_spec("power_test", salt="power_test_v1")
    n_variants = len(spec.variants)
    ids = population.ids()
    batcher = make_key_batcher(ids)
    rz = Randomizer()
    rng = np.random.default_rng(seed)

    fixed_codes = rz.assign_codes(ids, spec, batcher) if mode == "conditional" else None

    detections = 0
    n_arm_sum = 0
    n_valid = 0

    for i in range(n_trials):
        codes = (
            fixed_codes
            if fixed_codes is not None
            else rz.assign_codes(ids, fresh_spec(spec, f"p{i}"), batcher)
        )
        arms = _arms_from_codes(codes, n_variants)
        post = simulate_outcomes_treated(
            population, arms.treated, true_lift=true_lift, seed=int(rng.integers(1 << 31))
        )
        t_vals, c_vals = arms.split(post)
        if t_vals.size < 2 or c_vals.size < 2:  # pragma: no cover - 防御
            continue

        n_arm_sum += t_vals.size + c_vals.size
        n_valid += 1
        detections += int(_welch(t_vals, c_vals, alpha).p_value < alpha)

    mean_n = n_arm_sum / (2 * max(n_valid, 1))

    # 解析功效：两样本 z 近似（Welch 自由度在 n=1e4 量级时 t ≈ z）。
    # 公式本体在 inference.power 里，这里只是换了个输入形式 —— 免得两处各写一遍。
    post_sd = population.config.post_sd
    se = se_of_mean_diff(post_sd, mean_n, mean_n)
    analytic = z_power(true_lift, se, alpha)

    return PowerResult(
        true_lift=true_lift,
        alpha=alpha,
        n_trials=n_trials,
        detections=detections,
        empirical_power=detections / n_trials,
        analytic_power=analytic,
        mean_n_per_arm=mean_n,
        post_sd=post_sd,
        mode=mode,
    )


def power_curve(
    population: Population,
    lifts: np.ndarray,
    spec: ExperimentSpec | None = None,
    *,
    n_trials: int = 300,
    alpha: float = 0.05,
    mode: Mode = "randomized",
    seed: int = 13,
) -> list[PowerResult]:
    """在效应量网格上跑一遍功效仿真，用于画"经验 vs 解析"对照曲线。"""
    return [
        run_power_trials(
            population,
            spec,
            true_lift=float(lift),
            n_trials=n_trials,
            alpha=alpha,
            mode=mode,
            seed=seed + i,
        )
        for i, lift in enumerate(lifts)
    ]
