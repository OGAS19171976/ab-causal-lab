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

    @property
    def clusters(self) -> tuple[tuple[AggregateStats, ...], tuple[AggregateStats, ...]]:
        """``(处理组各簇, 对照组各簇)``；没有簇级统计量就报错。

        为什么要有这个属性，而不是让调用处各自 ``if x is None: raise``：
        那两个字段是 Optional（只有 ``analysis_unit == "cluster"`` 的读取才填），
        而**"有簇"这件事必须同时成立**（两臂都不能缺）。把它收在一个地方，
        调用处就不必各自复述一遍"检查两个字段"，也就不会有人只检查了一个。
        """
        if self.cluster_treatment is None or self.cluster_control is None:
            raise ValueError("这次查看没有簇级统计量（只有整簇随机化的分析单元才有）")
        return self.cluster_treatment, self.cluster_control

    def clusters_consistent(self, tol: float = 1e-6) -> bool:
        """簇级统计量合并回去，必须等于臂级统计量。

        守这条是为了防一种很难发现的错：簇级数据和臂级数据来自两次口径不同的读取，
        于是"簇级检验"和"汇报的总样本量"对不上。
        """
        if not self.has_clusters:
            return False
        cluster_treatment, cluster_control = self.clusters
        merged_t = cluster_treatment[0].merge(*cluster_treatment[1:])
        merged_c = cluster_control[0].merge(*cluster_control[1:])
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

        treatment_clusters, control_clusters = self.clusters
        t_clusters, c_clusters = arm(treatment_clusters), arm(control_clusters)
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
        treatment_clusters, control_clusters = self.clusters
        return (len(treatment_clusters), len(control_clusters))


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
    #: **已声明的护栏指标名**（记录上声明的，不是分析出来的）。
    #:
    #: 之所以让它进"分析输入"而不是留在记录里：报告必须能说一句
    #: "你声明了 N 个护栏，而**本平台不分析它们**"。数据模型只有一个主指标，
    #: 护栏需要另建一张指标表；在那之前，"存了字段、界面显示了、引擎没读过"
    #: 会让用户以为护栏被看着 —— 那是静默的不作为，比缺功能危险。
    guardrails: tuple[str, ...] = ()
    #: 护栏的**声明**（方向 + 容忍度）。名字在 ``guardrails`` 里、规格在这里 ——
    #: 只有名字没有规格时，护栏分析会判 ``unknown``（"没声明"不等于"通过"）。
    guardrail_specs: tuple[Any, ...] = ()
    #: 护栏的实测数据：``{护栏名: {臂名: AggregateStats}}``。
    #: 与主指标同一套可加充分统计量，所以护栏走同一套推断（Welch）。
    #: 数仓路径暂时给不出它（没有护栏表），于是那些护栏会被判 ``unknown`` ——
    #: 这是有意的：**缺数据时假装通过**是这一整块最危险的事。
    guardrail_series: dict[str, dict[str, Any]] = field(default_factory=dict)
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
        if len(self.looks) < 1:
            raise ValueError("至少需要一次查看")
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
            if self.analysis_unit == "cluster":
                g_t, g_c = lk.n_clusters
                if g_t < 2 or g_c < 2:
                    # 簇级方差的自由度是簇数减 2；每臂 1 个簇时它**无定义**，
                    # 这不是"噪声大"，是算不出来 —— 必须在这里拦住。
                    raise ValueError(
                        f"查看 {lk.label!r} 的簇数不足（{g_t}/{g_c}），"
                        "簇级检验每臂至少需要 2 个簇"
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
    guardrail_specs: tuple[Any, ...] = (),
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
        guardrail_specs=tuple(guardrail_specs),
        guardrail_series=_synthetic_guardrails(
            specs=guardrail_specs,
            treated=treated,
            control=control,
            treated_name=treated_name,
            control_name=control_name,
            seed=seed + 3,
        ),
        extra={"salt": salt, "n_users": int(n_users)},
    )
    data.validate()
    return data


def _synthetic_guardrails(
    *,
    specs: tuple[Any, ...],
    treated: "np.ndarray",
    control: "np.ndarray",
    treated_name: str,
    control_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """按声明合成护栏数据。

    每个护栏一族取值：基准均值取 1.0（量纲无所谓，判定看的是**相对伤害**），
    噪声相对标准差 5%。``demo_harm`` 是**演示专用**的真实伤害
    （与主指标的 ``true_lift`` 同一个性质）：把它设成 0.12，这条护栏在处置组
    就会真的劣化 12%，于是"护栏触发 -> 建议停实验"这条链路能被真跑出来。

    方向的处理：``demo_harm`` 永远表示**伤害**，所以
    ``higher_is_better`` 的护栏把伤害注入成"变低"。
    """
    from ..inference.aggregates import AggregateStats

    if not specs:
        return {}
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name = getattr(spec, "name", str(spec))
        direction = getattr(spec, "direction", "") or "lower_is_better"
        harm = float(getattr(spec, "demo_harm", 0.0) or 0.0)
        sign = 1.0 if direction == "lower_is_better" else -1.0
        control_values = rng.normal(1.0, 0.05, int(control.sum()))
        treated_values = rng.normal(1.0 + sign * harm, 0.05, int(treated.sum()))
        # 键必须是**真实变体名**：分析层是按 data.treated / data.control 查的，
        # 写死成 "treated"/"control" 会让所有护栏都判成"没有数据"（实测踩过）。
        out[name] = {
            treated_name: AggregateStats.from_arrays(treated_values),
            control_name: AggregateStats.from_arrays(control_values),
        }
    return out


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
    counts = {"control": int(control.sum()), "treatment": int(treated.sum())}

    looks = _nested_prefix_looks(
        post=clicks, pre=views, treated=treated, control=control,
        n_looks=n_looks, seed=seed + 2, labels=lambda k: f"look {k + 1}",
    )
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
    analysis_unit: str = "unit",
    metric_type: str = "mean",
) -> ExperimentData:
    """从数仓读取一个实验，返回与合成源同构的 ``ExperimentData``。

    两条读取路径各有分工：

    * **ADS**（实验×分支粒度）给主结论要的充分统计量；
    * **DWS**（实验×分支×日粒度）给序贯监控要的**累计**查看。

    ``analysis_unit="cluster"`` 时改读**簇粒度** DWS（``05_...``）：
    拿到每组簇的充分统计量，结论与监控都换成簇级口径。
    那张表与按天的 DWS 出自同一张 DWD、同一组 SUM，只是分组键多了一个 cluster_id ——
    所以"支持簇级分析"是一次 GROUP BY，而不是一条新链路。

    监控的信息比例用的是**实际累计样本量之比**，不是日历天数之比 ——
    每天进入实验的人数并不相等，用天数比会高估早期信息量，
    进而让早期边界偏松。``build_design`` 恰好支持传入自定义信息比例。
    """
    if analysis_unit == "cluster":
        return _warehouse_cluster_data(
            con,
            experiment,
            metric=metric,
            n_looks=n_looks,
            primary_estimator=primary_estimator,
        )

    if metric_type == "ratio":
        # 比值指标走**另一条 ADS 链路**（07）：那一层落的是
        # sum_y / sum_x / sum_xx / sum_yy / sum_xy，与 03 的六个可加量语义不同。
        # 从这里分派而不是在下面那张表的查询里加分支 ——
        # 两条链路的列名与含义都不同，混在一段 SQL 里迟早会串。
        return _warehouse_ratio_data(
            con,
            experiment,
            metric=metric,
            n_looks=n_looks,
        )

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
        # 护栏走 09 路 ADS（08 -> 09 的长表链路）。取不到时返回空字典 ——
        # 于是护栏会被判 unknown（"无法判断"），**不是通过**：
        # 缺数据时假装通过正是这一块最容易犯的错。
        guardrail_series=_warehouse_guardrail_data(
            con, experiment, control_name, treated_name
        ),
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


def _warehouse_guardrail_data(
    con: Any,
    experiment: str,
    control: str,
    treated: str,
) -> dict[str, dict[str, Any]]:
    """从 09 路 ADS 读护栏的充分统计量。

    返回 ``{护栏名: {臂名: AggregateStats}}``，形状与合成路径完全一致 ——
    所以护栏的判定规则（方向折算、置信下界、Bonferroni）只写了一份，
    两条数据源走同一段代码。

    **读不到就返回空字典**：宁可在报告里写"无法判断"，也不要因为
    "数仓里没有这张表"而让护栏看起来通过。表不存在（老库）与没有该实验的
    护栏行，都归到这一条。
    """
    from ..inference.aggregates import AggregateStats

    try:
        rows = con.execute(
            """
            SELECT guardrail, variant, user_cnt, value_sum, value_sq_sum
            FROM ads_experiment_guardrail_result
            WHERE experiment = ?
            """,
            [experiment],
        ).fetchall()
    except Exception:
        # 老库没有 09 路表：这不是错误，而是"这份数据源还没有护栏"
        return {}

    series: dict[str, dict[str, Any]] = {}
    for guardrail, variant, n, value_sum, value_sq_sum in rows:
        name = str(guardrail)
        if str(variant) not in (control, treated):
            continue
        series.setdefault(name, {})[str(variant)] = AggregateStats(
            n=int(n),
            sum_y=float(value_sum or 0.0),
            sum_yy=float(value_sq_sum or 0.0),
        )
    return series


def _warehouse_ratio_data(
    con,
    experiment: str,
    *,
    metric: str,
    n_looks: int,
) -> ExperimentData:
    """比值指标口径的数仓读取：读 ``06/07`` 两张**比值链路**的表。

    与簇级那条（``_warehouse_cluster_data``）同一个套路，差别只在读哪张表、
    以及充分统计量的含义：这里 ``x`` 是**分母**（互动次数），``y`` 是分子（互动值之和），
    估计量是 ``Σy/Σx`` —— 不是"人均比值的均值" ``mean(y_i/x_i)``（M1 实测口径差 12.6%）。

    信息比例用**累计分母**（Σx）之比：比值指标的精度由分母驱动，
    用天数比会高估早期信息量（与均值路径用样本量之比是同一个道理）。
    """
    ads = con.execute(_WAREHOUSE_QUERY.format(where="WHERE experiment = ?"), [experiment]).df()
    if ads.empty:
        raise ValueError(f"数仓 ADS 里找不到实验 {experiment!r}")
    variants = {str(r["variant"]): r for _, r in ads.iterrows()}
    if len(variants) < 2:
        raise ValueError(f"实验 {experiment!r} 只有一个分支，无法对比")
    control_name, treated_name = sorted(variants)[0], sorted(variants)[-1]

    daily = con.execute(
        """
        SELECT variant, ds, user_cnt, sum_y, sum_x, sum_yy, sum_xx, sum_xy
        FROM dws_experiment_ratio_daily WHERE experiment = ? ORDER BY variant, ds
        """,
        [experiment],
    ).df()
    if daily.empty:
        raise ValueError(
            f"比值链路里找不到实验 {experiment!r} 的日汇总 —— "
            "确认这一轮建仓跑过 06/07，且该实验在 DWD 里有数据"
        )

    cols = ("user_cnt", "sum_y", "sum_x", "sum_yy", "sum_xx", "sum_xy")
    cumulative: dict[str, list[dict[str, float]]] = {}
    labels: dict[str, list[str]] = {}
    for variant in (control_name, treated_name):
        rows = daily[daily["variant"] == variant]
        if rows.empty:
            raise ValueError(f"比值链路里 {experiment!r}/{variant!r} 没有日汇总")
        acc = {k: 0.0 for k in cols}
        series: list[dict[str, float]] = []
        tags: list[str] = []
        for _, r in rows.iterrows():
            for col in cols:
                acc[col] += float(r[col])
            series.append(dict(acc))
            tags.append(str(r["ds"]))
        cumulative[variant] = series
        labels[variant] = tags

    n_days = len(cumulative[control_name])
    if len(cumulative[treated_name]) != n_days:
        raise ValueError("两个分支的日期数不同，比值累计无法对齐")

    # 信息量 = 累计分母（两臂合计）；严格递增，最后一次必须是全量
    info: list[float] = [
        float(cumulative[control_name][d]["sum_x"] + cumulative[treated_name][d]["sum_x"])
        for d in range(n_days)
    ]
    total = info[-1]
    if total <= 0:
        raise ValueError(f"实验 {experiment!r} 的分母合计为 0，比值指标无定义")
    info = [v / total for v in info]

    picks: list[int] = []
    for k in range(1, n_looks):
        target = k / n_looks
        picks.append(min(range(n_days), key=lambda d: abs(info[d] - target)))
    picks.append(n_days - 1)
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
                label=f"{labels[treated_name][d][:10]}（累计）",
                information_fraction=frac,
                treatment=AggregateStats.from_sums(
                    n=int(ct["user_cnt"]), sum_x=ct["sum_x"], sum_y=ct["sum_y"],
                    sum_xx=ct["sum_xx"], sum_yy=ct["sum_yy"], sum_xy=ct["sum_xy"],
                ),
                control=AggregateStats.from_sums(
                    n=int(cc["user_cnt"]), sum_x=cc["sum_x"], sum_y=cc["sum_y"],
                    sum_xx=cc["sum_xx"], sum_yy=cc["sum_yy"], sum_xy=cc["sum_xy"],
                ),
            )
        )

    data = ExperimentData(
        experiment=experiment,
        metric=metric,
        source="warehouse",
        treated=treated_name,
        control=control_name,
        counts={name: int(row["user_cnt"]) for name, row in variants.items()},
        all_weights=_weight_map(
            {name: float(row["design_weight"]) for name, row in variants.items()}
        ),
        design_weights=_weight_map(
            {
                control_name: float(variants[control_name]["design_weight"]),
                treated_name: float(variants[treated_name]["design_weight"]),
            }
        ),
        looks=tuple(looks),
        # 比值指标只能用 delta method；CUPED 需要前置协变量，而比值链路里没有。
        primary_estimator="post_only",
        analysis_unit="unit",
        metric_type="ratio",
    )
    data.validate()
    return data


def _warehouse_cluster_data(
    con,
    experiment: str,
    *,
    metric: str,
    n_looks: int,
    primary_estimator: str,
) -> ExperimentData:
    """从 ADS（臂级总数）+ 簇粒度 DWS（每组簇的统计量）构造簇级 ``ExperimentData``。

    两条读取互为校验：**簇级统计量逐簇合并回去，必须等于 ADS 的人数与求和**。
    这条不变量在 ``ExperimentData.validate()`` 里强制检查 —— 它挡的是一种很难发现的错：
    簇级与臂级来自两次口径不同的读取，于是"簇级检验"和"报告里的总样本量"对不上。
    """
    ads = con.execute(_WAREHOUSE_QUERY.format(where="WHERE experiment = ?"), [experiment]).df()
    if ads.empty:
        raise ValueError(f"数仓 ADS 里找不到实验 {experiment!r}")
    variants = {str(r["variant"]): r for _, r in ads.iterrows()}
    if len(variants) < 2:
        raise ValueError(f"实验 {experiment!r} 只有一个分支，无法对比")
    control_name = "control" if "control" in variants else sorted(variants)[0]
    treated_name = "treatment" if "treatment" in variants else sorted(variants)[-1]

    cluster_counts = _cluster_counts(con, experiment, control_name, treated_name)
    # 先过闸门：簇必须是真正的随机化单元
    _verify_clusters_are_randomized(con, experiment, control_name, treated_name)
    looks, look_note = _warehouse_cluster_looks(
        con, experiment, control_name, treated_name, n_looks=n_looks
    )

    c_row, t_row = variants[control_name], variants[treated_name]
    last_treatment, last_control = looks[-1].clusters
    # 兜底校验：簇粒度 DWS 里数出来的簇数必须与 looks 的一致
    if cluster_counts != {treated_name: len(last_treatment),
                          control_name: len(last_control)}:
        raise ValueError(
            f"实验 {experiment!r} 的簇数对不上：DWS 直接数 {cluster_counts}，"
            f"按日累计得到 {len(last_treatment)}/"
            f"{len(last_control)}"
        )

    data = ExperimentData(
        experiment=experiment,
        metric=metric,
        source="warehouse",
        treated=treated_name,
        control=control_name,
        # **SRM 的检验对象是随机化单元**。簇设计下就是簇数。
        counts=cluster_counts,
        all_weights=_weight_map(
            {name: float(row["design_weight"]) for name, row in variants.items()}
        ),
        design_weights=_weight_map(
            {control_name: float(c_row["design_weight"]),
             treated_name: float(t_row["design_weight"])}
        ),
        looks=looks,
        # 簇级 CUPED 需要**簇级**前置指标。ADS 里只有用户级 pre_sum，
        # 而簇级 DWS 里也有 —— 但当前簇 DGP 没有前置期，所以先只支持 post-only。
        primary_estimator="post_only",
        analysis_unit="cluster",
        metric_type="mean",
        true_lift=float(t_row["true_lift"]),
        extra={
            "layer": str(t_row["layer"]),
            "hypothesis": str(t_row["hypothesis"]),
            "n_users": int(sum(int(r["user_cnt"]) for r in variants.values())),
            "n_clusters": sum(cluster_counts.values()),
            "cluster_key": "city（来自 ods_user_profile）",
            "look_note": look_note,
        },
    )
    data.validate()
    return data


def _verify_clusters_are_randomized(
    con, experiment: str, control_name: str, treated_name: str
) -> None:
    """确认每个簇**整簇**落在同一个分支里 —— 否则拒绝做簇级分析。

    为什么这是一道必须有的闸门
    --------------------------
    簇级检验假设"处理在簇级别分配"。如果实际上是人级随机化（同一个城市里既有
    处理组也有对照组用户），那么"城市"根本不是簇，簇级检验虽然**算得出一个数**，
    却答的是另一个问题：它把簇内那些本来可以互相抵消的信息丢掉了。

    实测踩过：数仓那份数据是人级随机化的，用城市当簇去做簇级分析给出了
    ``效应 +27.27、SE 3.38、p=5.5e-5`` —— 一个**看起来完全正常**的结论。
    而 M1 写的独立实现 ``cluster_level_ttest`` 直接拒绝了这个输入
    （"有 5 个簇内部同时存在处理与对照单元，这不是聚类随机化"）。

    这正是本项目反复强调的那类错误：**算得出来 ≠ 该算**。
    所以这里做同样的检查 —— 而且是让独立实现当裁判，而不是我自己再写一套判据。
    """
    mixed = con.execute(
        """
        SELECT COUNT(*) AS n_mixed FROM (
            SELECT cluster_id
            FROM dws_experiment_cluster_daily
            WHERE experiment = ? AND variant IN (?, ?)
            GROUP BY cluster_id
            HAVING COUNT(DISTINCT variant) > 1
        )
        """,
        [experiment, control_name, treated_name],
    ).df().iloc[0]["n_mixed"]
    if int(mixed) > 0:
        raise ValueError(
            f"实验 {experiment!r} 有 {int(mixed)} 个簇内部同时存在处理与对照单元 —— "
            "这不是整簇随机化，簇级检验不适用（它会把簇内本来能抵消的信息丢掉）。"
            "请把该实验的 analysis_unit 设为 'unit'；"
            "若确实是整簇随机化，检查簇键是否与分流时用的键一致"
            "（簇键决定一切：键错了，「簇」就只是个分组标签）"
        )


def _cluster_counts(con, experiment: str, control_name: str, treated_name: str) -> dict[str, int]:
    """每个分支有多少个簇 —— SRM 要检验的就是这个数。"""
    rows = con.execute(
        """
        SELECT variant, COUNT(DISTINCT cluster_id) AS n_clusters
        FROM dws_experiment_cluster_daily
        WHERE experiment = ? AND variant IN (?, ?)
        GROUP BY variant
        """,
        [experiment, control_name, treated_name],
    ).df()
    if rows.empty:
        raise ValueError(
            f"数仓里找不到实验 {experiment!r} 的簇粒度汇总"
            "（需要先执行 sql/05_dws_experiment_cluster_daily.sql 建表）"
        )
    return {str(r["variant"]): int(r["n_clusters"]) for _, r in rows.iterrows()}


def _warehouse_cluster_looks(
    con,
    experiment: str,
    control_name: str,
    treated_name: str,
    *,
    n_looks: int,
) -> tuple[tuple[LookData, ...], str]:
    """按日累计构造**簇级**查看序列，返回 ``(查看序列, 选点说明)``。

    三个必须说清的口径决定：

    **① 快照按「日」建，不是按「行」建。**
    第一版在遍历 (簇, 日) 行时逐行记快照，于是同一天被记了多次（一天里每个簇一条），
    查看标签出现"同一个日期重复 4 次"。正确做法是把一天的所有行累加完之后再记一次。

    **② 信息比例用累计簇数之比，不是累计用户数之比。**
    簇级估计量的方差 ≈ Var(簇均值)/G，信息量随**簇数**增长；
    用用户数会高估早期信息量（早期的簇少，但每个簇里的人可能已经不少）。
    这条与单元级路径刻意不同。

    **③ 查看点必须满足"每臂至少 2 个簇"。**
    簇级方差的自由度是簇数减 2，每臂 1 个簇时它无定义 —— 这不是"噪声大"，是算不出来。
    所以候选查看点先按这条筛一遍；筛完不够 ``n_looks`` 个就**明确报错**，
    而不是悄悄把次数减下来（改了次数就等于改了 alpha 消耗计划，那是个统计决定，
    不能由数据可用性替你决定）。
    """
    daily = con.execute(
        """
        SELECT variant, cluster_id, ds, user_cnt,
               pre_sum, post_sum, pre_sq_sum, post_sq_sum, pre_post_cross_sum
        FROM dws_experiment_cluster_daily
        WHERE experiment = ? AND variant IN (?, ?)
        ORDER BY variant, ds, cluster_id
        """,
        [experiment, control_name, treated_name],
    ).df()
    if daily.empty:
        raise ValueError(f"DWS 里找不到实验 {experiment!r} 的簇粒度汇总")

    sums_keys = ("user_cnt", "pre_sum", "post_sum", "pre_sq_sum", "post_sq_sum",
                 "pre_post_cross_sum")

    # 每个分支：每天一条快照「截至该日的 簇 -> 累计可加量」
    per_variant: dict[str, list[tuple[str, dict[str, dict[str, float]]]]] = {}
    for variant in (control_name, treated_name):
        rows = daily[daily["variant"] == variant]
        if rows.empty:
            raise ValueError(f"簇粒度 DWS 里 {experiment!r}/{variant!r} 没有数据")
        acc: dict[str, dict[str, float]] = {}
        snapshots: list[tuple[str, dict[str, dict[str, float]]]] = []
        for ds, day_rows in rows.groupby("ds", sort=True):
            for _, r in day_rows.iterrows():
                bucket = acc.setdefault(str(r["cluster_id"]), dict.fromkeys(sums_keys, 0.0))
                for key in sums_keys:
                    bucket[key] += float(r[key])
            # 深拷贝：否则后续累加会污染历史快照
            snapshots.append((str(ds)[:10], {c: dict(v) for c, v in acc.items()}))
        per_variant[variant] = snapshots

    # 两个分支的日期集合必须一致（同一天进来的用户分到两臂）
    dates = [d for d, _ in per_variant[control_name]]
    if dates != [d for d, _ in per_variant[treated_name]]:
        raise ValueError(f"实验 {experiment!r} 两个分支的日期序列不一致，簇级累计无法对齐")

    info_total = sum(len(per_variant[v][-1][1]) for v in (control_name, treated_name))

    def clusters_at(i: int, variant: str) -> int:
        return len(per_variant[variant][i][1])

    def fraction_at(i: int) -> float:
        return (
            clusters_at(i, control_name) + clusters_at(i, treated_name)
        ) / info_total

    # ③ 候选点：每臂至少 2 个簇
    feasible = [
        i for i in range(len(dates))
        if clusters_at(i, control_name) >= 2 and clusters_at(i, treated_name) >= 2
    ]
    if not feasible:
        g_c = clusters_at(len(dates) - 1, control_name)
        g_t = clusters_at(len(dates) - 1, treated_name)
        raise ValueError(
            f"实验 {experiment!r} 只有 {g_t} / {g_c} 个簇（处理组/对照组），"
            "达不到「每臂至少 2 个簇」——簇级方差的自由度是簇数减 2，两边都不够时它无定义。"
            "请增加簇数（簇键来自 ods_user_profile.city）"
        )

    # 按**信息比例**去重：同一个比例只保留**最后**那一天（数据最全，且保证末次是全量）。
    #
    # 这一步会暴露一件真实的事：如果所有簇在第一天就全部到位（本仓库的
    # 5 个城市就是如此），那么"累计簇数之比"全程都是 1.0 —— 序贯监控在簇级
    # **退化成固定样本分析**。这不是缺陷，而是数据结构的必然：
    # 簇级监控只在"簇分批进入"（城市/门店分批上线）时才有意义。
    distinct: dict[float, int] = {}
    for i in feasible:
        distinct[round(fraction_at(i), 12)] = i
    points = sorted(distinct.items())

    if len(points) == 1:
        ordered = [points[0][1]]
        look_note = (
            f"簇在第一天就全部到位（{clusters_at(len(dates) - 1, treated_name)}/"
            f"{clusters_at(len(dates) - 1, control_name)}），累计簇数全程为 1.0，"
            "序贯监控退化为**固定样本**分析（单次查看，边界即 1.96）"
        )
    else:
        k = min(n_looks, len(points))
        if k == 1:
            idxs = [len(points) - 1]
        else:
            step = (len(points) - 1) / (k - 1)
            idxs = sorted({round(j * step) for j in range(k)})
        ordered = [points[j][1] for j in idxs]
        look_note = (
            f"可用查看点 {len(points)} 个，按请求取 {len(ordered)} 个"
            + ("" if len(ordered) == n_looks else f"（请求 {n_looks} 次，受可用点限制降低）")
        )
        if len(ordered) < 2:
            raise ValueError(f"实验 {experiment!r} 的可选查看点不足 2 个")

    def arm_stats(snapshot: dict[str, dict[str, float]]) -> tuple[AggregateStats, ...]:
        return tuple(
            AggregateStats.from_sums(
                n=int(v["user_cnt"]), sum_x=v["pre_sum"], sum_y=v["post_sum"],
                sum_xx=v["pre_sq_sum"], sum_yy=v["post_sq_sum"],
                sum_xy=v["pre_post_cross_sum"],
            )
            for _, v in sorted(snapshot.items())
        )

    looks: list[LookData] = []
    for pos, i in enumerate(ordered):
        ct = arm_stats(per_variant[treated_name][i][1])
        cc = arm_stats(per_variant[control_name][i][1])
        looks.append(
            LookData(
                label=f"{dates[i]}（累计）",
                information_fraction=1.0 if pos == len(ordered) - 1 else fraction_at(i),
                treatment=ct[0].merge(*ct[1:]),
                control=cc[0].merge(*cc[1:]),
                cluster_treatment=ct,
                cluster_control=cc,
            )
        )
    # 把选点情况带出去，由上层写进报告（不静默）
    return tuple(looks), look_note


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
    #: 与 cumulative 逐日对齐的日期标签。**为什么不塞进那个 dict**：那样它就成了
    #: ``dict[str, float | str]``，而下游要做 ``float(...)`` 的地方全得再判一次类型。
    #: 数值归数值、标签归标签，读的人也不用先想"这个键是数还是串"。
    day_labels: dict[str, list[str]] = {}
    for variant in (control_name, treated_name):
        rows = daily[daily["variant"] == variant]
        if rows.empty:
            raise ValueError(f"DWS 里 {experiment!r}/{variant!r} 没有日汇总")
        acc = {k: 0.0 for k in ("user_cnt", "pre_sum", "post_sum", "pre_sq_sum", "post_sq_sum", "pre_post_cross_sum")}
        series: list[dict[str, float]] = []
        labels: list[str] = []
        for _, r in rows.iterrows():
            for k in acc:
                acc[k] += float(r[k])
            series.append(dict(acc))
            labels.append(str(r["ds"]))
        cumulative[variant] = series
        day_labels[variant] = labels

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
                label=f"{day_labels[treated_name][d][:10]}（累计）",
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
