"""合成数据生成器：给仿真台提供 **已知 ground truth** 的数据。

为什么必须有这个模块
--------------------
"框架算出来的 p 值对不对"这件事，在真实数据上**永远无法验证** ——
因为你不知道真实效应是多少。只有自己造数据、自己设定真值，
才能回答下面这些唯一重要的问题：

    * 零效应时，I 类错误率是不是 5%？
    * 95% 置信区间真的覆盖真值 95% 次吗？
    * 给定效应量和样本量，功效是不是和我算的解析值一致？
    * p 值在原假设下是不是均匀分布？

真实业务数据做不到这一点，所以"有没有仿真验证台"是区分
"会调统计库"和"懂统计"的第一道分水岭。

数据生成模型
------------
用户 ``i`` 有一个实验前的基线指标 ``X_i``，实验后指标为

    Y_i = post_mean + beta * (X_i - pre_mean) + eps_i + tau_i * T_i

其中 ``beta = rho * post_sd / pre_sd``，``eps ~ N(0, post_sd^2 (1-rho^2))``。
这样构造的好处：``Corr(X, Y) = rho`` **精确可控**，而
CUPED 校正后的**残余方差**恰好是 ``1 - rho^2`` 倍，也就是**方差缩减 = rho^2**
（rho=0.7 时约去掉 49% 的方差）—— 于是 M1 做 CUPED 时，
实测缩减比例可以和理论值直接对比，不用靠感觉。

注意别把两个量搞混：
* **方差缩减**（去掉的比例）= ``rho^2``
* **残余方差**（剩下的比例）= ``1 - rho^2``
* 对应到标准误，降幅是 ``1 - sqrt(1 - rho^2)``，rho=0.7 时约 39%

``tau_i = true_lift + lift_heterogeneity * segment_i`` 提供异质效应，
供 M4 的 CATE / uplift 模块使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assignment import ExperimentSpec, Randomizer, Variant

__all__ = [
    "PopulationConfig",
    "Population",
    "ExperimentSample",
    "generate_population",
    "two_arm_spec",
    "assign_variants",
    "assign_codes",
    "simulate_outcomes",
    "simulate_outcomes_treated",
    "make_experiment_sample",
]


@dataclass(frozen=True)
class PopulationConfig:
    """合成人群的超参数。"""

    n_units: int = 20_000
    seed: int = 20260101
    pre_mean: float = 100.0
    pre_sd: float = 30.0
    post_mean: float = 105.0
    post_sd: float = 30.0
    corr_pre_post: float = 0.70  # rho：直接决定 CUPED 的理论方差缩减 1-rho^2
    segment_sd: float = 1.0

    def __post_init__(self) -> None:
        if self.n_units < 10:
            raise ValueError("n_units 太小，至少 10")
        if self.pre_sd <= 0 or self.post_sd <= 0:
            raise ValueError("标准差必须为正")
        if not -1 < self.corr_pre_post < 1:
            raise ValueError("corr_pre_post 必须在 (-1, 1) 内")

    @property
    def cuped_variance_reduction(self) -> float:
        """CUPED 去掉的方差比例 = rho^2（不是 1-rho^2，别搞反）。"""
        return self.corr_pre_post**2

    @property
    def cuped_remaining_variance(self) -> float:
        """CUPED 校正后剩下的方差比例 = 1 - rho^2。"""
        return 1.0 - self.corr_pre_post**2


@dataclass(frozen=True)
class Population:
    """一批合成用户。"""

    user_id: np.ndarray  # (n,) str
    pre_metric: np.ndarray  # (n,) float，实验前基线
    segment: np.ndarray  # (n,) float，潜在异质性维度
    config: PopulationConfig

    def __len__(self) -> int:
        return int(self.user_id.size)

    def ids(self) -> list[str]:
        return [str(u) for u in self.user_id]


@dataclass(frozen=True)
class ExperimentSample:
    """一次完整实验观测：分组 + 实验前指标 + 实验后指标。"""

    user_id: np.ndarray
    variant: np.ndarray  # (n,) 分支名
    pre_metric: np.ndarray
    post_metric: np.ndarray
    true_lift: float
    treatment: str
    control: str

    def __len__(self) -> int:
        return int(self.user_id.size)

    def arm(self, name: str) -> ExperimentSample:
        mask = self.variant == name
        return ExperimentSample(
            user_id=self.user_id[mask],
            variant=self.variant[mask],
            pre_metric=self.pre_metric[mask],
            post_metric=self.post_metric[mask],
            true_lift=self.true_lift,
            treatment=self.treatment,
            control=self.control,
        )

    @property
    def observed_effect(self) -> float:
        """这次抽样的表观效应（含噪声）。"""
        t = self.post_metric[self.variant == self.treatment]
        c = self.post_metric[self.variant == self.control]
        return float(t.mean() - c.mean())


def generate_population(config: PopulationConfig | None = None, **overrides) -> Population:
    """按 ``PopulationConfig`` 生成合成人群。"""
    cfg = config or PopulationConfig(**overrides)
    rng = np.random.default_rng(cfg.seed)

    pre = rng.normal(cfg.pre_mean, cfg.pre_sd, cfg.n_units)
    segment = rng.normal(0.0, 1.0, cfg.n_units)
    user_id = np.array([f"u{i:07d}" for i in range(cfg.n_units)], dtype=object)

    return Population(user_id=user_id, pre_metric=pre, segment=segment, config=cfg)


def two_arm_spec(
    name: str = "m0_demo",
    *,
    salt: str | None = None,
    treatment_weight: float = 0.5,
    traffic_ratio: float = 1.0,
    layer: str | None = None,
    control: str = "control",
    treatment: str = "treatment",
) -> ExperimentSpec:
    """构造一个标准两臂（1:1 或指定比例）实验定义。"""
    return ExperimentSpec(
        name=name,
        variants=(
            Variant(control, 1.0 - treatment_weight),
            Variant(treatment, treatment_weight),
        ),
        salt=salt,
        traffic_ratio=traffic_ratio,
        layer=layer,
    )


def assign_variants(
    population: Population,
    spec: ExperimentSpec,
    randomizer: Randomizer | None = None,
    batcher=None,
) -> np.ndarray:
    """对人群分流，返回分支名数组；未进入实验的单元为 ``""``。"""
    rz = randomizer or Randomizer()
    assigned = rz.assign_many(population.ids(), spec, batcher)
    return np.array([a if a is not None else "" for a in assigned], dtype=object)


def assign_codes(
    population: Population,
    spec: ExperimentSpec,
    randomizer: Randomizer | None = None,
    batcher=None,
) -> np.ndarray:
    """对人群分流，返回**整数分支下标**；未进入实验的单元为 ``-1``。

    仿真循环的推荐入口：整数数组可以直接做布尔掩码，省掉对象数组的比较开销。
    """
    rz = randomizer or Randomizer()
    return rz.assign_codes(population.ids(), spec, batcher)


def simulate_outcomes_treated(
    population: Population,
    is_treated: np.ndarray,
    *,
    true_lift: float = 0.0,
    lift_heterogeneity: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """按潜在结果模型生成实验后指标；处理指示用布尔数组传入。

    ``seed`` 决定噪声；**同一 population + 同一分组 + 不同 seed**
    就是一次合法的 A/A 重复抽样。
    """
    cfg = population.config
    rng = np.random.default_rng(seed)

    beta = cfg.corr_pre_post * cfg.post_sd / cfg.pre_sd
    eps_sd = cfg.post_sd * np.sqrt(1.0 - cfg.corr_pre_post**2)

    baseline = (
        cfg.post_mean
        + beta * (population.pre_metric - cfg.pre_mean)
        + rng.normal(0.0, eps_sd, population.pre_metric.size)
    )

    tau = true_lift + lift_heterogeneity * population.segment
    return baseline + tau * is_treated.astype(float)


def simulate_outcomes(
    population: Population,
    variant: np.ndarray,
    *,
    true_lift: float = 0.0,
    lift_heterogeneity: float = 0.0,
    treatment: str = "treatment",
    seed: int | None = None,
) -> np.ndarray:
    """``simulate_outcomes_treated`` 的分支名便捷入口（可读性优先）。"""
    return simulate_outcomes_treated(
        population,
        np.asarray(variant) == treatment,
        true_lift=true_lift,
        lift_heterogeneity=lift_heterogeneity,
        seed=seed,
    )


def make_experiment_sample(
    population: Population,
    spec: ExperimentSpec,
    *,
    true_lift: float = 0.0,
    lift_heterogeneity: float = 0.0,
    randomizer: Randomizer | None = None,
    seed: int | None = None,
) -> ExperimentSample:
    """一步生成"分流 + 观测"的完整实验数据。"""
    variant = assign_variants(population, spec, randomizer)
    post = simulate_outcomes(
        population,
        variant,
        true_lift=true_lift,
        lift_heterogeneity=lift_heterogeneity,
        treatment=spec.variants[-1].name,
        seed=seed,
    )
    return ExperimentSample(
        user_id=population.user_id,
        variant=variant,
        pre_metric=population.pre_metric,
        post_metric=post,
        true_lift=true_lift,
        treatment=spec.variants[-1].name,
        control=spec.variants[0].name,
    )
