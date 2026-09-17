"""序贯实验的仿真数据。

两种粒度，回答两个不同的问题：

``simulate_canonical_sequences``
    按**正则联合分布**直接抽检验统计量路径（布朗运动表示）。
    这是群序贯边界递推所假设的模型本身，几十万次只要几秒。
    它验证的是：**边界递推的数值实现对不对**。
``simulate_experiment_sequence``
    用户级仿真：真实分流、真实抽样、按前缀增长算运行统计量。
    它验证的是：**整套假设在真实实验数据上成不成立**。

两个都要有。只做前者是"拿假设验假设"，只做后者则几十万次跑不动。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assignment import ExperimentSpec, Randomizer
from ..hashing import KeyBatcher
from ..sim.generator import Population, simulate_outcomes_treated, two_arm_spec

__all__ = [
    "LookSequence",
    "CanonicalSequences",
    "default_information_fractions",
    "simulate_canonical_sequences",
    "simulate_experiment_sequence",
]


def default_information_fractions(n_looks: int) -> np.ndarray:
    """等距信息量：``1/K, 2/K, ..., 1``。"""
    if n_looks < 1:
        raise ValueError("n_looks 必须为正")
    return np.arange(1, n_looks + 1) / n_looks


@dataclass(frozen=True)
class CanonicalSequences:
    """正则联合分布下的一批检验路径。"""

    information_fractions: np.ndarray
    z_statistics: np.ndarray  # (n_trials, n_looks)
    estimates: np.ndarray  # (n_trials, n_looks)
    #: ``(n_looks,)``：每次查看的标准误。保持一维靠广播使用，
    #: 查看次数上千时能省下成百 MB —— 全部下游函数都支持广播。
    standard_errors: np.ndarray
    true_effect: float
    se_final: float
    effect_scale: float  # drift = effect / se_final

    @property
    def n_trials(self) -> int:
        return int(self.z_statistics.shape[0])

    @property
    def n_looks(self) -> int:
        return int(self.z_statistics.shape[1])


def simulate_canonical_sequences(
    *,
    n_trials: int,
    information_fractions: np.ndarray | None = None,
    se_final: float,
    effect: float = 0.0,
    n_looks: int | None = None,
    seed: int = 0,
) -> CanonicalSequences:
    """按正则联合分布抽 ``Z`` 路径。

    构造：``S(t_k)`` 是带漂移的布朗运动，``Z_k = S(t_k)/sqrt(t_k)``。于是

        E[Z_k] = theta * sqrt(t_k),   Corr(Z_i, Z_j) = sqrt(t_i/t_j)

    其中 ``theta = effect / se_final``。这正是群序贯检验所假设的分布。
    """
    if information_fractions is None:
        if n_looks is None:
            raise ValueError("information_fractions 与 n_looks 至少要给一个")
        t = default_information_fractions(n_looks)
    else:
        t = np.asarray(information_fractions, dtype=float)
    if se_final <= 0:
        raise ValueError("se_final 必须为正")

    rng = np.random.default_rng(seed)
    theta = effect / se_final
    steps = np.diff(np.r_[0.0, t])

    # 带漂移的独立增量：E[inc_j] = theta * step_j, Var = step_j
    increments = rng.normal(theta * steps, np.sqrt(steps), size=(n_trials, t.size))
    S = np.cumsum(increments, axis=1)
    z = S / np.sqrt(t)

    se = se_final / np.sqrt(t)
    return CanonicalSequences(
        information_fractions=t,
        z_statistics=z,
        estimates=z * se,
        standard_errors=se,
        true_effect=float(effect),
        se_final=float(se_final),
        effect_scale=float(theta),
    )


@dataclass(frozen=True)
class LookSequence:
    """一条真实实验的检验路径。"""

    information_fractions: np.ndarray
    per_arm: np.ndarray
    estimates: np.ndarray
    standard_errors: np.ndarray
    z_statistics: np.ndarray
    true_effect: float

    @property
    def n_looks(self) -> int:
        return int(self.z_statistics.size)

    def naive_significant(self, alpha: float = 0.05) -> bool:
        """朴素做法：任何一次 ``|z| >= 1.96`` 就宣布显著。"""
        return bool(np.any(np.abs(self.z_statistics) >= 1.959964))

    def first_naive_crossing(self, alpha: float = 0.05) -> int | None:
        hits = np.flatnonzero(np.abs(self.z_statistics) >= 1.959964)
        return int(hits[0]) + 1 if hits.size else None

    def summary(self) -> str:
        lines = ["真实实验检验路径（每臂样本量 / 效应 / 标准误 / z）"]
        for k in range(self.n_looks):
            lines.append(
                f"  查看 {k + 1}: n/arm={self.per_arm[k]:>7,}  "
                f"效应={self.estimates[k]:+.4f}  SE={self.standard_errors[k]:.4f}  "
                f"z={self.z_statistics[k]:+.3f}"
            )
        return "\n".join(lines)


def _running_stats(values: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在嵌套前缀上的运行均值与**样本方差**。"""
    cs = np.cumsum(values)
    cs2 = np.cumsum(values * values)
    n = sizes.astype(float)
    mean = cs[sizes - 1] / n
    var = (cs2[sizes - 1] - n * mean * mean) / (n - 1)
    return mean, var


def simulate_experiment_sequence(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    n_looks: int = 5,
    true_lift: float = 0.0,
    seed: int | None = None,
    randomizer: Randomizer | None = None,
    batcher: KeyBatcher | None = None,
) -> LookSequence:
    """用户级仿真：真实分流 + 真实抽样，按信息量增长算运行统计量。

    用户进入实验的顺序是**随机的**，所以对每个臂做一次随机排列，
    再取嵌套前缀 —— 这与"逐个招募受试者"是等价的。
    """
    if n_looks < 2:
        raise ValueError("序贯检验至少要 2 次查看")

    spec = spec or two_arm_spec("sequential_exp", salt="sequential_exp_v1")
    ids = population.ids()
    rz = randomizer or Randomizer()
    batcher = batcher or KeyBatcher(ids)

    codes = rz.assign_codes(ids, spec, batcher)
    n_variants = len(spec.variants)
    treated = codes == n_variants - 1
    control = codes == 0

    post = simulate_outcomes_treated(
        population, treated, true_lift=true_lift, seed=seed
    )

    rng = np.random.default_rng(seed)
    t_order = rng.permutation(np.flatnonzero(treated))
    c_order = rng.permutation(np.flatnonzero(control))

    n_per_arm = int(min(t_order.size, c_order.size))
    if n_per_arm < n_looks * 2:
        raise ValueError(
            f"每臂只有 {n_per_arm} 个单元，撑不起 {n_looks} 次查看（每次至少 2 个）"
        )

    sizes = (np.arange(1, n_looks + 1) * (n_per_arm / n_looks)).astype(int)
    sizes = np.maximum.accumulate(np.minimum(sizes, n_per_arm))

    mean_t, var_t = _running_stats(post[t_order[: n_per_arm]], sizes)
    mean_c, var_c = _running_stats(post[c_order[: n_per_arm]], sizes)

    estimates = mean_t - mean_c
    se = np.sqrt(var_t / sizes + var_c / sizes)

    return LookSequence(
        information_fractions=sizes / n_per_arm,
        per_arm=sizes,
        estimates=estimates,
        standard_errors=se,
        z_statistics=estimates / se,
        true_effect=float(true_lift),
    )
