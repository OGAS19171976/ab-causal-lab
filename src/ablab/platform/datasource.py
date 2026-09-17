"""数据源层：把"分析引擎的输入"从"某个具体的数据来源"里摘出来。

为什么需要这一层
----------------
M5 的第一版里，``analyse_experiment`` 直接就地生成合成数据并把三件事
（造数据、算指标、判健康）揉在一个函数里。这只在"数据源只有一个"时成立。
接数仓时你会立刻撞上两个问题：

1. **真实数仓给不出明细。** 数仓的分层设计（M0）刻意只让 ADS 输出
   **充分统计量**（n 与五个 SUM）—— 这正是"SQL 负责口径、Python 负责检验"
   这条边界的代价与收益。于是分析引擎不能再假设自己拿得到逐用户的数组。

2. **序贯监控的"查看"在两种数据源下含义不同。**
   合成数据里"查看"是**按用户进入顺序**取嵌套前缀（可以做到精确等距）；
   数仓里"查看"是**按天累计**（DWS 天然按 ds 汇总）。
   两者都能表达成"一串 LookData"，但只有把抽象定在充分统计量这一层，二者才能共用同一段监控代码。

所以 ``ExperimentData`` 的粒度定为：一个实验 = 一串 **LookData**，
每个 LookData 是两臂的 ``AggregateStats``。最后一次查看必须等于全量。

一条必须守住的不变量
--------------------
``looks[-1]`` 必须与"全量分析"用到的统计量**逐位相同**。
否则页面上会出现"监控最后一点"和"主结论"对不上的诡异现象 ——
而那是纯粹的实现缺陷，不是统计现象。有测试守着这条（``test_last_look_equals_total``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..inference import AggregateStats
from ..sequential import build_design

__all__ = [
    "LookData",
    "ExperimentData",
    "build_synthetic_data",
    "build_warehouse_data",
    "list_warehouse_experiments",
]


@dataclass(frozen=True)
class LookData:
    """一次"查看"：两臂的充分统计量 + 它对应的信息比例。

    ``post_only()`` 与 ``cuped()`` 都**委托给推断层的同一个实现**，
    而不是在这里重写一遍公式。理由不只是"少写代码"：
    平台上"监控曲线"和"头条结论"必须给出同一个数，
    只有共用实现才能让这条恒等式成立（有测试守着）。
    """

    label: str
    information_fraction: float
    treatment: AggregateStats
    control: AggregateStats
    #: 簇级充分统计量（每个簇一个）。``analysis_unit == "cluster"`` 时必需。
    #:
    #: 为什么给**每簇一组**统计量就够了：簇级检验要的是"簇均值的均值与方差"，
    #: 而它只需要每簇的 (n, Σy)。换句话说，CR1 在"整簇随机化"这个设计下
    #: 并不需要明细 —— 这也是数仓只落可加量的又一个回报。
    cluster_treatment: tuple[AggregateStats, ...] | None = None
    cluster_control: tuple[AggregateStats, ...] | None = None

    @property
    def n_per_arm(self) -> int:
        return int(min(self.treatment.n, self.control.n))

    @property
    def effect(self) -> float:
        return self.treatment.mean_y - self.control.mean_y

    @property
    def std_error(self) -> float:
        return float(
            np.sqrt(self.treatment.var_y / self.treatment.n + self.control.var_y / self.control.n)
        )

    @property
    def z(self) -> float:
        se = self.std_error
        return float(self.effect / se) if se > 0 else float("nan")

    # -- 各口径，全部委托给推断层 ------------------------------------------ #
    def post_only(self):
        """post-only（Welch 检验），返回 ``Estimate``。"""
        from ..inference import welch_ttest_from_stats

        return welch_ttest_from_stats(
            n_treatment=self.treatment.n,
            mean_treatment=self.treatment.mean_y,
            var_treatment=self.treatment.var_y,
            n_control=self.control.n,
            mean_control=self.control.mean_y,
            var_control=self.control.var_y,
        )

    def cuped(self):
        """CUPED，返回 ``(Estimate, CupedFit)``。

        **θ̂ 用本次查看自己的合并样本估计**，而不是预先钉死一个值。这样做的两个好处：

        * 没有"信息前视" —— 第 1 次查看只用到第 1 次查看之前进入的用户
          （若用全样本估出的 θ 去算早期 z，早期 z 就偷看了尚未进入实验的数据）；
        * 最后一次查看的 θ̂ 恰好等于全样本 θ̂，于是**监控右端点 == 头条结论**逐位成立。

        代价是各次查看的 z 不再是"累计和的线性泛函"，序贯边界所依赖的
        典型联合分布（corr = √(t_j/t_k)）只是近似成立 ——
        这个近似到底有多准，不靠推演，靠仿真量（见 ``run_monitoring_fwer_audit``）。
        """
        from ..inference import cuped_estimate

        return cuped_estimate(self.treatment, self.control)

    def ratio_delta(self):
        """比值指标的 delta method，返回 ``Estimate``。

        ``x`` 是分母、``y`` 是分子，估计量是**业务口径**的 ``Σy/Σx``。
        这与"人均比值 ``mean(y_i/x_i)``"是两个不同的量（M1 实测口径差 12.6%），
        而后者需要明细、前者不需要 —— 这也是数仓路径只支持前者的原因。
        """
        from ..inference import ratio_delta_method

        return ratio_delta_method(self.treatment, self.control)

    def estimate(self, method: str):
        """按声明的口径取估计。``method`` ∈ {cuped, post_only}。"""
        if method == "cuped":
            return self.cuped()[0]
        if method == "post_only":
            return self.post_only()
        raise ValueError(f"未知估计口径 {method!r}，应为 'cuped' 或 'post_only'")

    # -- 簇级 -------------------------------------------------------------- #
    @property
    def has_clusters(self) -> bool:
        return self.cluster_treatment is not None and self.cluster_control is not None

    def clusters_consistent(self, tol: float = 1e-6) -> bool:
        """簇级统计量合并回去，必须等于臂级统计量。

        守这条是为了防一种很难发现的错：簇级数据和臂级数据来自两次口径不同的读取，
        于是"簇级检验"和"汇报的总样本量"对不上。
        """
        if not self.has_clusters:
            return False
        merged_t = self.cluster_treatment[0].merge(*self.cluster_treatment[1:])
        merged_c = self.cluster_control[0].merge(*self.cluster_control[1:])
        return (
            _close(merged_t.n, self.treatment.n)
            and _close(merged_t.sum_y, self.treatment.sum_y, tol * max(1.0, abs(self.treatment.sum_y)))
            and _close(merged_c.n, self.control.n)
            and _close(merged_c.sum_y, self.control.sum_y, tol * max(1.0, abs(self.control.sum_y)))
        )

    def cluster_level(self):
        """簇级检验，返回 ``Estimate``。

        做法：先把每簇压成簇均值，再把**簇**当成观测单位做 Welch 检验。
        没有自己写公式 —— 委托给 ``welch_ttest_from_stats``，
        这样"用户级 t 检验"和"簇级 t 检验"在代码里是同一个检验，
        只在"输入的单位是什么"上有区别。这正是 M1 想说明的那件事。
        """
        from ..inference import welch_ttest_from_stats

        if not self.has_clusters:
            raise ValueError("这次查看没有簇级统计量，无法做簇级检验")

        def arm(clusters: tuple[AggregateStats, ...]) -> AggregateStats:
            means = np.array([c.mean_y for c in clusters], dtype=float)
            return AggregateStats.from_outcomes(means)

        t_clusters, c_clusters = arm(self.cluster_treatment), arm(self.cluster_control)
        return welch_ttest_from_stats(
            n_treatment=t_clusters.n,
            mean_treatment=t_clusters.mean_y,
            var_treatment=t_clusters.var_y,
            n_control=c_clusters.n,
            mean_control=c_clusters.mean_y,
            var_control=c_clusters.var_y,
        )

    @property
    def n_clusters(self) -> tuple[int, int]:
        if not self.has_clusters:
            return (0, 0)
        return (len(self.cluster_treatment), len(self.cluster_control))


@dataclass(frozen=True)
class ExperimentData:
    """分析引擎的输入：与数据源无关。

    ``counts`` / ``all_weights`` 覆盖**全部**分支（用于整体 SRM）；
    ``treatment`` / ``control`` / ``design_weights`` 只描述被对比的两臂
    （平台约定：最后一臂 vs 第一臂）。
    """

    experiment: str
    metric: str
    source: str
    treated: str
    control: str
    counts: dict[str, int]
    all_weights: dict[str, float]
    design_weights: dict[str, float]
    looks: tuple[LookData, ...]
    #: 平台按哪个口径做**判定**（序贯边界与"是否显著"都跟着它）。
    #: 必须与头条结论同一个估计量 —— 否则页面上会出现"曲线越界了但结论说不显著"。
    primary_estimator: str = "cuped"
    #: **分析单元**：``unit``（随机化单元就是分析单元）或 ``cluster``（整簇随机化）。
    #: 声明为 cluster 时，簇级充分统计量必需 —— 否则用单元级 t 检验会把
    #: I 类错误率抬到 60% 以上（M1 实测 64.5%），而那个 p 值看起来完全正常。
    analysis_unit: str = "unit"
    #: **指标类型**：``mean``（人均指标）或 ``ratio``（比值指标 Σy/Σx）。
    #: 比值指标必须用 delta method，否则答的是另一个问题（M1 实测口径差 12.6%）。
    metric_type: str = "mean"
    #: 仅合成/演示数据有真值；真实数仓没有
    true_lift: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> LookData:
        """全量那一次查看 —— 主结论用的就是它。"""
        return self.looks[-1]

    @property
    def n_looks(self) -> int:
        return len(self.looks)

    def validate(self) -> None:
        """守住几条会让报告自相矛盾的不变量。"""
        if self.primary_estimator not in ("cuped", "post_only"):
            raise ValueError(
                f"primary_estimator 必须是 'cuped' 或 'post_only'，收到 {self.primary_estimator!r}"
            )
        if self.analysis_unit not in ("unit", "cluster"):
            raise ValueError(f"analysis_unit 必须是 'unit' 或 'cluster'，收到 {self.analysis_unit!r}")
        if self.metric_type not in ("mean", "ratio"):
            raise ValueError(f"metric_type 必须是 'mean' 或 'ratio'，收到 {self.metric_type!r}")
        if self.analysis_unit == "cluster":
            if not self.total.has_clusters:
                raise ValueError("analysis_unit='cluster' 但这次查看没有簇级统计量")
            if not self.total.clusters_consistent():
                raise ValueError("簇级统计量合并回去与臂级统计量不一致（两次读取口径不同）")
        if self.analysis_unit == "cluster" and self.primary_estimator == "cuped":
            raise ValueError(
                "整簇随机化下 CUPED 需要**簇级**的前置指标；当前数据源没有提供，"
                "请把 estimator 设为 'post_only'"
            )
        if len(self.looks) < 2:
            raise ValueError(f"至少需要 2 次查看才能谈序贯监控，收到 {len(self.looks)}")
        fractions = [lk.information_fraction for lk in self.looks]
        if any(b <= a for a, b in zip(fractions, fractions[1:])):
            # 必须**严格**递增：BoundarySolver 的卷积网格吃非递增输入会给出无意义边界
            raise ValueError(f"信息比例必须严格递增，收到 {fractions}")
        if abs(fractions[-1] - 1.0) > 1e-9:
            raise ValueError(f"最后一次查看的信息比例必须是 1.0，收到 {fractions[-1]}")
        for lk in self.looks:
            if lk.treatment.n < 2 or lk.control.n < 2:
                raise ValueError(
                    f"查看 {lk.label!r} 某臂样本不足 2（{lk.treatment.n}/{lk.control.n}），"
                    "方差无法估计"
                )
        if {self.treated, self.control} != set(self.design_weights):
            raise ValueError(
                f"design_weights 必须恰好覆盖被对比的两臂 {self.treated}/{self.control}，"
                f"收到 {sorted(self.design_weights)}"
            )
        if self.treated not in self.counts or self.control not in self.counts:
            raise ValueError("counts 里缺少被对比的那两臂")


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _weight_map(raw: dict[str, float]) -> dict[str, float]:
    total = float(sum(raw.values()))
    if total <= 0:
        raise ValueError(f"权重之和必须为正，收到 {raw}")
    return {k: float(v) / total for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# 合成数据源（演示用）
# --------------------------------------------------------------------------- #
def build_synthetic_data(
    *,
    experiment: str,
    salt: str,
    variants: list[tuple[str, float]],
    metric: str,
    n_users: int,
    n_looks: int,
    true_lift: float = 0.0,
    traffic_ratio: float = 1.0,
    seed: int,
    population: Any,
    primary_estimator: str = "cuped",
    analysis_unit: str = "unit",
    metric_type: str = "mean",
) -> ExperimentData:
    """按 salt 确定性生成合成数据，并给出按**随机进入顺序**的嵌套查看。

    每次查看在**每一臂内部**按比例取前缀（``t_k·n_臂``），而不是"两臂都取
    ``min(n_t, n_c)·t_k``"。区别在分支不等权时（如 90/10）才显现：
    后者永远取不到大臂的全部样本，于是最后一次查看 ≠ 全量分析 —— 那会让
    监控曲线的右端点与主结论对不上。

    三条分支各有各的 DGP，且都复用 ``sim/scenarios.py`` 里 M1 验证过的那一份：

    * ``mean``    —— 平台自带的业务量纲人群（``population``）
    * ``ratio``   —— 比值场景：每人有曝光数（分母）与点击数（分子）
    * ``cluster`` —— 整簇随机化场景：城市级随机效应，簇内相关
    """
    if analysis_unit == "cluster":
        return _synthetic_cluster(
            experiment=experiment,
            salt=salt,
            metric=metric,
            n_users=n_users,
            n_looks=n_looks,
            true_lift=true_lift,
            seed=seed,
        )
    if metric_type == "ratio":
        return _synthetic_ratio(
            experiment=experiment,
            salt=salt,
            metric=metric,
            n_users=n_users,
            n_looks=n_looks,
            true_lift=true_lift,
            seed=seed,
            primary_estimator=primary_estimator,
        )

    from dataclasses import replace

    from ..assignment import ExperimentSpec, Randomizer, Variant
    from ..hashing import KeyBatcher
    from ..sim.generator import generate_population, simulate_outcomes_treated

    spec = ExperimentSpec(
        name=experiment,
        variants=tuple(Variant(n, w) for n, w in variants),
        salt=salt,
        traffic_ratio=traffic_ratio,
    )
    cfg = replace(population, n_units=n_users, seed=seed)
    pop = generate_population(cfg)
    ids = pop.ids()
    codes = Randomizer().assign_codes(ids, spec, KeyBatcher(ids))

    treated_name, control_name = spec.variants[-1].name, spec.variants[0].name
    treated = codes == len(spec.variants) - 1
    control = codes == 0
    if treated.sum() < 2 or control.sum() < 2:
        raise ValueError("分流后某一组样本不足，无法分析")

    post = simulate_outcomes_treated(pop, treated, true_lift=true_lift, seed=seed + 1)
    pre = pop.pre_metric

    counts = {v.name: int((codes == i).sum()) for i, v in enumerate(spec.variants)}
    all_weights = _weight_map({v.name: v.weight for v in spec.variants})
    design_weights = _weight_map(
        {control_name: spec.variants[0].weight, treated_name: spec.variants[-1].weight}
    )

    looks = _nested_prefix_looks(
        post=post,
        pre=pre,
        treated=treated,
        control=control,
        n_looks=n_looks,
        seed=seed + 2,
        labels=lambda k: f"look {k + 1}",
    )
    data = ExperimentData(
        experiment=experiment,
        metric=metric,
        source="synthetic",
        treated=treated_name,
        control=control_name,
        counts=counts,
        all_weights=all_weights,
        design_weights=design_weights,
        looks=looks,
        primary_estimator=primary_estimator,
        analysis_unit="unit",
        metric_type="mean",
        true_lift=float(true_lift),
        extra={"salt": salt, "n_users": int(n_users)},
    )
    data.validate()
    return data


def _synthetic_ratio(
    *,
    experiment: str,
    salt: str,
    metric: str,
    n_users: int,
    n_looks: int,
    true_lift: float,
    seed: int,
    primary_estimator: str,
) -> ExperimentData:
    """比值指标场景：x 是曝光数（分母），y 是点击数（分子）。

    这里 ``true_lift`` 是**相对**提升（比值指标上业务只关心相对量），
    与人均指标路径的绝对效应不同 —— 报告里会把这一点写出来。

    naive 口径（人均比值 ``mean(y_i/x_i)``）**需要明细**，所以只在这一侧算出来、
    放进 ``extra`` 供报告做口径对照；数仓路径给不出它（那是 M1 的另一个发现：
    能算的业务口径只有一个，而 naive 那个恰好是错的）。
    """
    from ..sim.scenarios import RatioScenarioConfig, generate_ratio_scenario

    sample = generate_ratio_scenario(
        RatioScenarioConfig(n_users=n_users),
        relative_lift=true_lift,
        salt=salt,
        seed=seed,
    )
    treated, control = sample.treated, ~sample.treated
    views, clicks = sample.views, sample.clicks

    t_stats = AggregateStats.from_arrays(clicks[treated], views[treated])
    c_stats = AggregateStats.from_arrays(clicks[control], views[control])

    looks = _nested_prefix_looks(
        post=clicks, pre=views, treated=treated, control=control,
        n_looks=n_looks, seed=seed + 2, labels=lambda k: f"look {k + 1}",
    )
    counts = {"control": int(control.sum()), "treatment": int(treated.sum())}
    data = ExperimentData(
        experiment=experiment,
        metric=metric,
        source="synthetic",
        treated="treatment",
        control="control",
        counts=counts,
        all_weights={"control": 0.5, "treatment": 0.5},
        design_weights={"control": 0.5, "treatment": 0.5},
        looks=looks,
        primary_estimator=primary_estimator,
        analysis_unit="unit",
        metric_type="ratio",
        true_lift=float(true_lift),
        extra={
            "salt": salt,
            "n_users": int(n_users),
            # naive 口径：人均比值。它是**另一个问题的答案**，只用来展示口径差。
            "naive_ratio_treatment": float((clicks[treated] / views[treated]).mean()),
            "naive_ratio_control": float((clicks[control] / views[control]).mean()),
            "true_lift_is_relative": True,
        },
    )
    data.validate()
    return data


def _synthetic_cluster(
    *,
    experiment: str,
    salt: str,
    metric: str,
    n_users: int,
    n_looks: int,
    true_lift: float,
    seed: int,
) -> ExperimentData:
    """整簇随机化场景：处理在**簇**级别分配，用户嵌在簇内。

    簇级随机效应让同簇用户的结果相关（ICC > 0），此时用户级 t 检验的
    I 类错误率会到 60% 以上（M1 实测 64.5%），而它给出的 p 值看起来完全正常。

    查看序列按**簇**的进入顺序取前缀 —— 与结论同一个分析单元，
    不然又会回到 M6.1 那个"曲线和结论不是一个东西"的坑。
    """
    from ..sim.scenarios import ClusterScenarioConfig, generate_cluster_scenario

    users_per_cluster = 100
    n_clusters = max(20, int(n_users) // users_per_cluster)
    sample = generate_cluster_scenario(
        ClusterScenarioConfig(
            n_clusters=n_clusters, users_per_cluster=users_per_cluster
        ),
        true_lift=true_lift,
        salt=salt,
        seed=seed,
    )

    # 每簇一组充分统计量。x 全为 0 —— 这个 DGP 没有前置指标，
    # 所以整簇路径只能用 post-only 口径（有簇级前置指标时才能上 CUPED）。
    cluster_codes = sample.cluster_id
    unique_clusters = sorted(set(cluster_codes.tolist()))
    first_index = {g: int(np.flatnonzero(cluster_codes == g)[0]) for g in unique_clusters}
    treated_clusters = [g for g in unique_clusters if bool(sample.treated[first_index[g]])]
    control_clusters = [g for g in unique_clusters if not bool(sample.treated[first_index[g]])]
    by_cluster = {g: sample.outcome[cluster_codes == g] for g in unique_clusters}
    per_cluster_t = [AggregateStats.from_outcomes(by_cluster[g]) for g in treated_clusters]
    per_cluster_c = [AggregateStats.from_outcomes(by_cluster[g]) for g in control_clusters]

    rng = np.random.default_rng(seed + 2)
    order_t = rng.permutation(len(per_cluster_t))
    order_c = rng.permutation(len(per_cluster_c))

    def prefix_stats(clusters, order, k: int) -> tuple[AggregateStats, tuple[AggregateStats, ...]]:
        picked = tuple(clusters[i] for i in order[:k])
        return picked[0].merge(*picked[1:]), picked

    looks: list[LookData] = []
    for i in range(1, n_looks + 1):
        t = i / n_looks
        k_t = max(2, int(round(t * len(per_cluster_t))))
        k_c = max(2, int(round(t * len(per_cluster_c))))
        if i == n_looks:
            k_t, k_c = len(per_cluster_t), len(per_cluster_c)
        merged_t, picked_t = prefix_stats(per_cluster_t, order_t, k_t)
        merged_c, picked_c = prefix_stats(per_cluster_c, order_c, k_c)
        looks.append(
            LookData(
                label=f"cluster-look {i}",
                information_fraction=1.0 if i == n_looks else t,
                treatment=merged_t,
                control=merged_c,
                cluster_treatment=picked_t,
                cluster_control=picked_c,
            )
        )

    data = ExperimentData(
        experiment=experiment,
        metric=metric,
        source="synthetic",
        treated="treatment",
        control="control",
        # **SRM 的检验对象必须是随机化单元**：这里是簇，不是用户。
        # 拿用户数去查 SRM 会问错问题（用户数不等只说明簇大小不等，与分流无关）。
        counts={"control": len(per_cluster_c), "treatment": len(per_cluster_t)},
        all_weights={"control": 0.5, "treatment": 0.5},
        design_weights={"control": 0.5, "treatment": 0.5},
        looks=tuple(looks),
        # 整簇路径没有簇级前置指标 → 只能用 post-only
        primary_estimator="post_only",
        analysis_unit="cluster",
        metric_type="mean",
        true_lift=float(true_lift),
        extra={
            "salt": salt,
            "n_users": int(sample.outcome.size),
            "n_clusters": n_clusters,
            "analysis_unit_is_cluster": True,
        },
    )
    data.validate()
    return data


def _nested_prefix_looks(
    *,
    post: np.ndarray,
    pre: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    n_looks: int,
    seed: int,
    labels,
) -> tuple[LookData, ...]:
    """按随机进入顺序构造嵌套前缀的充分统计量。

    实现要点：只走一趟 ``cumsum``。因为 ``AggregateStats`` 的六个量都是可加的，
    任何前缀的统计量都能由前缀和直接读出 —— 这正是可加性在仿真侧的用处
    （等价于 DWS 按天汇总后在上层做窗口 SUM）。
    """
    if n_looks < 2:
        raise ValueError("n_looks 至少为 2")
    n_small = int(min(np.count_nonzero(treated), np.count_nonzero(control)))
    # 第一次查看落在 t=1/K，小臂至少要凑够 2 个样本才谈得上方差 ——
    # 否则序贯曲线的前几个点纯粹是噪声，而它们恰恰最容易撞到早期边界。
    if n_small < 2 * n_looks:
        raise ValueError(
            f"每组可用样本只有 {n_small} 个，不足以构造 {n_looks} 次序贯查看"
            f"（该流量比例/分支权重下至少需要 {2 * n_looks} 个）；请提高 n_users。"
        )

    rng = np.random.default_rng(seed)
    order_t = rng.permutation(np.flatnonzero(treated))
    order_c = rng.permutation(np.flatnonzero(control))

    def prefix_sums(order: np.ndarray) -> dict[str, np.ndarray]:
        y = post[order]
        x = pre[order]
        return {
            "sum_x": np.concatenate(([0.0], np.cumsum(x))),
            "sum_y": np.concatenate(([0.0], np.cumsum(y))),
            "sum_xx": np.concatenate(([0.0], np.cumsum(x * x))),
            "sum_yy": np.concatenate(([0.0], np.cumsum(y * y))),
            "sum_xy": np.concatenate(([0.0], np.cumsum(x * y))),
        }

    sums_t, sums_c = prefix_sums(order_t), prefix_sums(order_c)

    def stats_at(sums: dict[str, np.ndarray], k: int) -> AggregateStats:
        return AggregateStats.from_sums(
            n=k,
            sum_x=float(sums["sum_x"][k]),
            sum_y=float(sums["sum_y"][k]),
            sum_xx=float(sums["sum_xx"][k]),
            sum_yy=float(sums["sum_yy"][k]),
            sum_xy=float(sums["sum_xy"][k]),
        )

    looks: list[LookData] = []
    for i in range(1, n_looks + 1):
        t = i / n_looks
        k_t = max(2, int(round(t * order_t.size)))
        k_c = max(2, int(round(t * order_c.size)))
        looks.append(
            LookData(
                label=labels(i - 1),
                information_fraction=t,
                treatment=stats_at(sums_t, k_t),
                control=stats_at(sums_c, k_c),
            )
        )
    # 最后一次必须是全量（round 可能把它落在 n-1 上）
    last = looks[-1]
    looks[-1] = LookData(
        label=last.label,
        information_fraction=1.0,
        treatment=stats_at(sums_t, order_t.size),
        control=stats_at(sums_c, order_c.size),
    )
    return tuple(looks)


# --------------------------------------------------------------------------- #
# 数仓数据源
# --------------------------------------------------------------------------- #
_WAREHOUSE_QUERY = """
SELECT experiment, variant, user_cnt, design_weight, layer, hypothesis, true_lift,
       pre_sum, post_sum, pre_sq_sum, post_sq_sum, pre_post_cross_sum
FROM ads_experiment_result
{where}
ORDER BY experiment, variant
"""


def list_warehouse_experiments(con) -> list[dict[str, Any]]:
    """列出数仓里可绑定的实验（供平台的下拉框用）。"""
    rows = con.execute(
        """
        SELECT experiment,
               any_value(layer)      AS layer,
               any_value(hypothesis) AS hypothesis,
               any_value(true_lift)  AS true_lift,
               COUNT(*)              AS n_variants,
               SUM(user_cnt)         AS n_users
        FROM ads_experiment_result
        GROUP BY experiment
        ORDER BY experiment
        """
    ).df()
    return [
        {
            "experiment": str(r["experiment"]),
            "layer": str(r["layer"]),
            "hypothesis": str(r["hypothesis"]),
            "true_lift": float(r["true_lift"]),
            "n_variants": int(r["n_variants"]),
            "n_users": int(r["n_users"]),
        }
        for _, r in rows.iterrows()
    ]


def build_warehouse_data(
    con,
    experiment: str,
    *,
    metric: str = "post_metric_14d",
    n_looks: int = 5,
    primary_estimator: str = "cuped",
) -> ExperimentData:
    """从 ADS + DWS 读取一个实验，返回与合成源同构的 ``ExperimentData``。

    两条读取路径各有分工：

    * **ADS**（实验×分支粒度）给主结论要的充分统计量；
    * **DWS**（实验×分支×日粒度）给序贯监控要的**累计**查看。

    监控的信息比例用的是**实际累计样本量之比**，不是日历天数之比 ——
    每天进入实验的人数并不相等，用天数比会高估早期信息量，
    进而让早期边界偏松。``build_design`` 恰好支持传入自定义信息比例。
    """
    ads = con.execute(_WAREHOUSE_QUERY.format(where="WHERE experiment = ?"), [experiment]).df()
    if ads.empty:
        raise ValueError(f"数仓 ADS 里找不到实验 {experiment!r}")

    variants = {str(r["variant"]): r for _, r in ads.iterrows()}
    if len(variants) < 2:
        raise ValueError(f"实验 {experiment!r} 只有一个分支，无法对比")

    control_name = "control" if "control" in variants else sorted(variants)[0]
    treated_name = "treatment" if "treatment" in variants else sorted(variants)[-1]
    c_row, t_row = variants[control_name], variants[treated_name]

    def stats_of(row) -> AggregateStats:
        return AggregateStats.from_sums(
            n=int(row["user_cnt"]),
            sum_x=float(row["pre_sum"]),
            sum_y=float(row["post_sum"]),
            sum_xx=float(row["pre_sq_sum"]),
            sum_yy=float(row["post_sq_sum"]),
            sum_xy=float(row["pre_post_cross_sum"]),
        )

    counts = {name: int(row["user_cnt"]) for name, row in variants.items()}
    all_weights = _weight_map(
        {name: float(row["design_weight"]) for name, row in variants.items()}
    )
    design_weights = _weight_map(
        {control_name: float(c_row["design_weight"]), treated_name: float(t_row["design_weight"])}
    )

    looks = _warehouse_looks(
        con, experiment, control_name, treated_name, n_looks=n_looks
    )
    data = ExperimentData(
        experiment=experiment,
        metric=metric,
        source="warehouse",
        treated=treated_name,
        control=control_name,
        counts=counts,
        all_weights=all_weights,
        design_weights=design_weights,
        looks=looks,
        primary_estimator=primary_estimator,
        true_lift=float(t_row["true_lift"]),
        extra={
            "layer": str(t_row["layer"]),
            "hypothesis": str(t_row["hypothesis"]),
        },
    )
    data.validate()

    # 交叉验证：DWS 累计到最后必须等于 ADS 汇总（同一条 SQL 链的两端）。
    # 不是在验 SQL 对不对（那是 verify_against_detail 的事），
    # 而是在验"监控右端点 == 主结论口径"这条不变量在这个数据源上也成立。
    total = data.total
    for name, row, lk in (
        (treated_name, t_row, total.treatment),
        (control_name, c_row, total.control),
    ):
        if lk.n != counts[name]:
            raise ValueError(
                f"{name} 的 DWS 累计样本量 {lk.n} 与 ADS 的 {counts[name]} 不一致 —— "
                "两次读取走了不同口径"
            )
    return data


def _warehouse_looks(
    con,
    experiment: str,
    control_name: str,
    treated_name: str,
    *,
    n_looks: int,
) -> tuple[LookData, ...]:
    """按天累计构造查看序列，信息比例取自实际累计样本量。"""
    daily = con.execute(
        """
        SELECT variant, ds,
               user_cnt, pre_sum, post_sum, pre_sq_sum, post_sq_sum, pre_post_cross_sum
        FROM dws_experiment_variant_daily
        WHERE experiment = ? AND variant IN (?, ?)
        ORDER BY variant, ds
        """,
        [experiment, control_name, treated_name],
    ).df()
    if daily.empty:
        raise ValueError(f"DWS 里找不到实验 {experiment!r} 的日汇总")

    totals = {
        str(r["variant"]): int(r["user_cnt"])
        for _, r in con.execute(
            "SELECT variant, user_cnt FROM ads_experiment_result WHERE experiment = ?",
            [experiment],
        ).df().iterrows()
    }

    cumulative: dict[str, list[dict[str, float]]] = {}
    for variant in (control_name, treated_name):
        rows = daily[daily["variant"] == variant]
        if rows.empty:
            raise ValueError(f"DWS 里 {experiment!r}/{variant!r} 没有日汇总")
        acc = {k: 0.0 for k in ("user_cnt", "pre_sum", "post_sum", "pre_sq_sum", "post_sq_sum", "pre_post_cross_sum")}
        series: list[dict[str, float]] = []
        for _, r in rows.iterrows():
            for k in acc:
                acc[k] += float(r[k])
            series.append(dict(acc, ds=str(r["ds"])))
        cumulative[variant] = series

    # 目标信息比例 = 等距，但**落到实际某一天的累计值**上
    n_t, n_c = totals[treated_name], totals[control_name]
    n_days = len(cumulative[control_name])
    # 信息量用两臂合计样本量做代理（等权时与单臂等价）
    info = np.array(
        [
            (cumulative[control_name][d]["user_cnt"] + cumulative[treated_name][d]["user_cnt"])
            / (n_t + n_c)
            for d in range(n_days)
        ]
    )
    picks: list[int] = []
    for i in range(1, n_looks + 1):
        picks.append(int(np.argmin(np.abs(info - i / n_looks))))
    picks.append(n_days - 1)  # 最后一次必须是全量
    # 去重并保持升序；顺便保证信息比例**严格**递增
    # （等累计样本量的两天会让 BoundarySolver 拿到非递增的网格）
    ordered: list[int] = []
    for d in sorted(set(picks)):
        if not ordered or info[d] > info[ordered[-1]]:
            ordered.append(d)
        elif d == n_days - 1:
            ordered[-1] = d
    if len(ordered) < 2:
        raise ValueError(f"实验 {experiment!r} 的可选查看点不足 2 个（只有 {n_days} 天）")

    looks: list[LookData] = []
    for pos, d in enumerate(ordered):
        ct = cumulative[treated_name][d]
        cc = cumulative[control_name][d]
        frac = 1.0 if pos == len(ordered) - 1 else float(info[d])
        looks.append(
            LookData(
                # DuckDB 的 DATE 经 pandas 会变成 Timestamp，只取日期部分
                label=f"{str(ct['ds'])[:10]}（累计）",
                information_fraction=frac,
                treatment=AggregateStats.from_sums(
                    n=int(ct["user_cnt"]), sum_x=ct["pre_sum"], sum_y=ct["post_sum"],
                    sum_xx=ct["pre_sq_sum"], sum_yy=ct["post_sq_sum"], sum_xy=ct["pre_post_cross_sum"],
                ),
                control=AggregateStats.from_sums(
                    n=int(cc["user_cnt"]), sum_x=cc["pre_sum"], sum_y=cc["post_sum"],
                    sum_xx=cc["pre_sq_sum"], sum_yy=cc["post_sq_sum"], sum_xy=cc["pre_post_cross_sum"],
                ),
            )
        )
    return tuple(looks)


def design_for(data: ExperimentData, alpha: float):
    """按数据源的**实际**信息比例构建设计。

    合成源的信息比例是精确等距，数仓源是实际累计比例 ——
    ``build_design`` 两者都支持，所以监控层不需要知道自己在哪个数据源上。
    若一律按等距算，数仓路径会高估早期信息量、让早期边界偏松。
    """
    return build_design(
        alpha=alpha,
        n_looks=data.n_looks,
        spending="obf",
        information_fractions=[lk.information_fraction for lk in data.looks],
    )
