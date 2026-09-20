"""分析编排：把一个实验跑完整套引擎，产出一份**可展示的体检报告**。

这一层的价值不在算法（算法全在 M0–M4），而在于**顺序与口径**：

    SRM 体检 -> 主指标（CUPED）-> 朴素口径对照 -> 效应分解 -> 序贯监控 -> 健康判定

顺序是有讲究的：**SRM 不过就不该看指标**。一个分组比例失衡的实验，
后面的 p 值再漂亮也是废的 —— 但如果没有一层强制这个顺序，
分析师很容易直接跳到"效应是多少"。

健康判定也刻意做成**三档**而不是"通过/不通过"：
``pass`` / ``warn`` / ``fail``，因为大多数真实问题是"可疑"而不是"确定错了"。

与数据源的解耦
--------------
核心是 ``analyse_data(data)``：它只吃 ``ExperimentData``（一串充分统计量），
不关心数据来自合成器还是数仓。两个薄封装各自负责造数据：

* ``analyse_experiment``                —— 合成数据（演示）
* ``analyse_experiment_from_warehouse`` —— ADS + DWS（真实链路）

这么分的实际收益：序贯监控那段代码只写了一遍。
第一版里它直接操作逐用户的数组（按随机进入顺序取前缀），
而数仓给不出明细 —— 把抽象定在充分统计量这一层之后，两条路径才共用同一段实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..hashing import murmur3_32
from ..inference import (
    CupedFit,
    Diagnostic,
    Estimate,
    cuped_estimate,
    mde,
    srm_check,
    welch_ttest_from_stats,
    z_power,
)
from ..sequential import SequentialDesign, choose_tau, msprt_p_value
from ..sim.generator import PopulationConfig
from .datasource import (
    ExperimentData,
    LookData,
    build_synthetic_data,
    build_warehouse_data,
    design_for,
)
from .registry import ExperimentRecord

__all__ = [
    "ExperimentReport",
    "CheckItem",
    "PLATFORM_POPULATION",
    "analyse_data",
    "analyse_experiment",
    "analyse_experiment_from_warehouse",
    "run_aa_validation",
]


#: 平台演示用的**业务量纲**人群。
#:
#: M0–M4 的验证台用 ``pre_mean=100, post_sd=30`` 的抽象量纲 —— 那对校验分布性质更合适。
#: 但平台上的实验要"看起来像真的"：一个 ``true_lift=0.35`` 的排序模型改版，
#: 若指标标准差是 30，效应量只有 0.01σ，**无论多少样本都检不出来**，演示会全线不显著。
#: 所以这里换成"人均互动次数"这种量纲（均值约 19、标准差约 7.5）；
#: 相关系数仍然是 0.7，M1 关于 CUPED 的结论原样成立。
PLATFORM_POPULATION = PopulationConfig(
    pre_mean=18.0,
    pre_sd=7.0,
    post_mean=19.5,
    post_sd=7.5,
    corr_pre_post=0.70,
)


@dataclass
class CheckItem:
    """体检项的统一展示结构。"""

    name: str
    status: str
    message: str
    statistic: float | None = None
    p_value: float | None = None

    @classmethod
    def from_diagnostic(cls, diag: Diagnostic) -> "CheckItem":
        return cls(
            name=diag.name,
            status=diag.status,
            message=diag.message,
            statistic=diag.statistic,
            p_value=diag.p_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }


@dataclass
class ExperimentReport:
    """一个实验的完整体检报告。"""

    experiment_id: str
    experiment_name: str
    status: str
    #: **实际进入分析的单元数**（被对比的那两臂之和），不是"请求了多少人"。
    #: 数仓路径下也没有"请求量"这个概念，所以统一按分析量记。
    n_users: int
    alpha: float
    health: str
    checks: list[CheckItem] = field(default_factory=list)

    naive: Estimate | None = None
    cuped: Estimate | None = None
    cuped_fit: CupedFit | None = None
    sequential: SequentialDesign | None = None
    monitoring: list[dict[str, Any]] = field(default_factory=list)
    srm: Diagnostic | None = None

    #: naive 效应里由**前置协变量失衡**贡献的那部分（= naive − CUPED = θ̂·ΔX̄）。
    imbalance_component: float | None = None
    #: 扣掉失衡后剩下的部分（= CUPED 效应）：真实效应 + 结果噪声。
    residual_component: float | None = None

    #: ``synthetic``（平台自造）或 ``warehouse``（ADS + DWS）
    source: str = "synthetic"
    #: 合成数据路径下的**请求人群规模**；数仓路径为 None
    population_size: int | None = None
    #: 判定口径（= 头条结论的估计量）
    estimator: str = "cuped"
    #: 分析单元与指标类型（决定了上面这些字段里哪些有意义）
    analysis_unit: str = "unit"
    metric_type: str = "mean"
    #: **分析单元个数**。单元设计下等于 ``n_users``；簇设计下是簇数，
    #: 而 SRM 与簇级检验用的正是这个数 —— 不区分就会让人拿用户数去读 SRM。
    n_analysis_units: int = 0
    #: **头条结论**与**对照口径**。任何数据源、任何分析单元、任何指标类型，
    #: 报告里"那个结论"永远在 ``primary`` 里 —— 前端与调用方不需要知道
    #: 它背后是 CUPED、簇级检验还是比值 delta。
    primary: Estimate | None = None
    alt: Estimate | None = None
    primary_estimator_name: str = "cuped"
    alt_estimator_name: str | None = None
    #: 功效读数：MDE、当前数据能检出的最小效应、相对对照口径的 MDE 收缩
    power: dict[str, Any] = field(default_factory=dict)

    def _estimate_dict(self, est: Estimate | None) -> dict[str, Any] | None:
        if est is None:
            return None
        return {
            "method": est.method,
            "absolute_effect": est.absolute_effect,
            "relative_effect": est.relative_effect,
            "std_error": est.std_error,
            "ci_low": est.ci_low,
            "ci_high": est.ci_high,
            "p_value": est.p_value,
            "significant": est.significant,
            "n_treatment": est.n_treatment,
            "n_control": est.n_control,
            "mean_treatment": est.mean_treatment,
            "mean_control": est.mean_control,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "source": self.source,
            "n_users": self.n_users,
            "population_size": self.population_size,
            "estimator": self.estimator,
            "analysis_unit": self.analysis_unit,
            "metric_type": self.metric_type,
            "n_analysis_units": self.n_analysis_units,
            "power": self.power,
            "alpha": self.alpha,
            "health": self.health,
            "checks": [c.to_dict() for c in self.checks],
            "imbalance_component": self.imbalance_component,
            "residual_component": self.residual_component,
            "naive": self._estimate_dict(self.naive),
            "cuped": self._estimate_dict(self.cuped),
            "primary": self._estimate_dict(self.primary),
            "alt": self._estimate_dict(self.alt),
            "primary_estimator_name": self.primary_estimator_name,
            "alt_estimator_name": self.alt_estimator_name,
            "cuped_fit": None
            if self.cuped_fit is None
            else {
                "theta": self.cuped_fit.theta,
                "correlation": self.cuped_fit.correlation,
                "variance_reduction": self.cuped_fit.variance_reduction,
                "remaining_variance": self.cuped_fit.remaining_variance,
                "se_shrinkage": self.cuped_fit.se_shrinkage,
                "effective_sample_multiplier": self.cuped_fit.effective_sample_multiplier,
            },
            "sequential": None
            if self.sequential is None
            else {
                "spending_name": self.sequential.spending_name,
                "boundaries": self.sequential.boundaries.tolist(),
                "information_fractions": self.sequential.information_fractions.tolist(),
                "final_boundary": self.sequential.final_boundary,
                "final_nominal_p": self.sequential.final_nominal_p,
                "reliable": self.sequential.reliable,
            },
            "monitoring": self.monitoring,
        }


def _headline_path(
    data: ExperimentData, lk: LookData
) -> tuple[Estimate, str, Estimate | None, str | None]:
    """决定这一次查看"按哪个口径报"和"拿谁当对照"。

    三套规则，全按记录上的两个声明走（``analysis_unit`` / ``metric_type``）：

    ==================  ====================  ==========================
    声明                 头条口径               对照口径
    ==================  ====================  ==========================
    unit + mean          cuped / post_only     另一个
    cluster              cluster_level          unit_level（**错的那个**）
    ratio                ratio_delta            （无，见下）
    ==================  ====================  ==========================

    比值指标的对照是"人均比值"，它**需要明细**才能算标准误，所以不放进监控表，
    而是在体检项里单独报口径差的**点估计** —— 放一个算不出 SE 的 z 更不诚实。
    """
    if data.analysis_unit == "cluster":
        # **按声明走**：这一支原先写死成"簇级 post-only + 单元级对照"，
        # 于是声明 estimator='cuped' 的簇级实验会被**静默忽略** ——
        # 报告里既没有 CUPED 那一项，也没人知道声明的口径没生效。
        # 现在：声明什么就用什么（簇级的两种口径都在），
        # 对照仍然放"单元级"：那是 M1 里 FPR 64.5% 的那个错误做法，
        # 摆在旁边正好说明为什么分析单元必须与随机化单元对齐。
        if data.primary_estimator == "cuped":
            return (
                lk.cluster_cuped()[0], "cluster_cuped",
                lk.cluster_level(), "cluster_level",
            )
        return lk.cluster_level(), "cluster_level", lk.post_only(), "unit_level"
    if data.metric_type == "ratio":
        return lk.ratio_delta(), "ratio_delta", None, None
    primary = data.primary_estimator
    other = "post_only" if primary == "cuped" else "cuped"
    return lk.estimate(primary), primary, lk.estimate(other), other


def _monitoring_from(data: ExperimentData, design: SequentialDesign) -> list[dict[str, Any]]:
    """把一串查看变成监控表 —— **与数据源无关，且两个口径都报**。

    平台的头条结论是 CUPED，所以**判定也必须用 CUPED**。
    第一版这里用的是 post-only 的 z，于是页面上会出现"曲线越界了，但结论卡说 p=0.06"
    这种自相矛盾 —— 那不是统计现象，是两个口径没对齐（实测每 15 个实验就有 1 个）。

    但只报一个口径又会丢掉信息：两者的差本身有价值。所以主口径与对照口径都报，
    ``crossed`` 与 always-valid p 按头条口径判定。
    """
    primary_name: str | None = None
    alt_name: str | None = None
    effects = np.empty(data.n_looks)
    ses = np.empty(data.n_looks)
    z_path = np.empty(data.n_looks)
    alt_z = np.full(data.n_looks, np.nan)
    alt_effects = np.full(data.n_looks, np.nan)
    alt_ses = np.full(data.n_looks, np.nan)

    for k, lk in enumerate(data.looks):
        primary_est, primary_name, alt_est, alt_name = _headline_path(data, lk)
        e, se = primary_est.absolute_effect, primary_est.std_error
        if not np.isfinite(se) or se <= 0:
            raise ValueError(f"第 {k + 1} 次查看的标准误无法估计（某臂方差为 0 或样本量不足）")
        effects[k], ses[k] = e, se
        z_path[k] = e / se
        if alt_est is not None:
            ase = alt_est.std_error
            if np.isfinite(ase) and ase > 0:
                alt_effects[k], alt_ses[k], alt_z[k] = (
                    alt_est.absolute_effect, ase, alt_est.absolute_effect / ase,
                )

    # tau 从"取 2 倍标准误"改成**算出来的规则**（阈值最小化，见
    # ``sequential.always_valid.choose_tau``）：alpha=0.05 时它给出 2.87×SE。
    # 依据是实测（reports/m2_validation.md 第 7 节）：同样的 FWER 下，
    # 功效从 0.1613 抬到 0.1814 —— 而 2×SE 恰好落在最优点附近但偏低，
    # 所以这不是"旧的是错的"，是"旧的没人量过"。
    # **tau 是设计期常数**：这里用末次查看的标准误（由事先定好的信息分数决定），
    # 不是当前观测到的 SE —— 后者是数据依赖的选择，会让 always-valid 保证作废。
    msp = msprt_p_value(
        effects, ses, tau=choose_tau(std_error=float(ses[-1]), alpha=design.alpha)
    )
    # mSPRT 逐个 look 的 p 值本身都是有效的，但"任何时候看都有效"这个卖点
    # 对应的量是它们的**运行最小值** —— 也就是"截止到目前为止最有利的那个 p"。
    # 只报逐次值会让人以为可以永远等到最后一个 look 再挑一个最小的看。
    msp_running_min = np.minimum.accumulate(msp)

    rows: list[dict[str, Any]] = []
    for k, lk in enumerate(data.looks):
        z = float(z_path[k])
        boundary = float(design.boundaries[k])
        n_t, n_c = lk.n_clusters if lk.has_clusters else (0, 0)
        rows.append(
            {
                "look": k + 1,
                "label": lk.label,
                "information_fraction": float(lk.information_fraction),
                # 每次查看实际用了多少样本（每组）。多臂实验里这个数只算被对比的两臂，
                # 不是"总样本" —— 报出来是为了让口径可见。
                "n_per_arm": lk.n_per_arm,
                "n_treatment": int(lk.treatment.n),
                "n_control": int(lk.control.n),
                # 簇设计下"样本量"是簇数，两个都要标出来
                "n_clusters_treatment": n_t,
                "n_clusters_control": n_c,
                "n_clusters_per_arm": min(n_t, n_c) if lk.has_clusters else 0,
                # 判定口径（= 头条结论的估计量）
                "estimator": primary_name,
                "z": z,
                "boundary": boundary,
                "crossed": bool(abs(z) >= boundary),
                "effect": float(effects[k]),
                "std_error": float(ses[k]),
                "always_valid_p": float(msp[k]),
                "always_valid_p_running_min": float(msp_running_min[k]),
                # 对照口径：两者的差就是"口径选错了会差多少"
                "alt_estimator": alt_name,
                "alt_z": None if not np.isfinite(alt_z[k]) else float(alt_z[k]),
                "alt_effect": None if not np.isfinite(alt_effects[k]) else float(alt_effects[k]),
                "alt_std_error": None if not np.isfinite(alt_ses[k]) else float(alt_ses[k]),
            }
        )
    return rows


def analyse_data(data: ExperimentData, *, alpha: float = 0.05) -> ExperimentReport:
    """跑完整套引擎 —— 只吃充分统计量，不关心数据来自哪里。"""
    data.validate()
    total = data.total

    # ---- 1. SRM 体检（先做，不通过就不该看指标） -------------------------- #
    # 这一条用的是**全臂**计数与设计权重，检验的是整体分流。
    srm = srm_check(data.counts, data.all_weights)

    # 送进估计量的设计权重只保留**被对比的两臂**（已在 datasource 层归一）。
    weights = data.design_weights

    # ---- 2. 头条结论：按记录上的两个声明分派 ------------------------------- #
    # 单元 + 人均指标时，post-only 与 CUPED **两个都算出来**：
    # 前者是"不做任何协变量校正"的对照，报告里两个都要有。
    # 簇 / 比值路径下这两个名字没有意义（把簇级数塞进叫 cuped 的字段就是撒谎），
    # 所以它们置空，由下面 headline_path 给出的 primary / alt 承担。
    unit_mean = data.analysis_unit == "unit" and data.metric_type == "mean"
    naive: Estimate | None = None
    cuped: Estimate | None = None
    fit: CupedFit | None = None
    if unit_mean:
        naive = welch_ttest_from_stats(
            n_treatment=total.treatment.n,
            mean_treatment=total.treatment.mean_y,
            var_treatment=total.treatment.var_y,
            n_control=total.control.n,
            mean_control=total.control.mean_y,
            var_control=total.control.var_y,
            metric=data.metric,
            variant=data.treated,
            control_name=data.control,
            alpha=alpha,
            expected_weights=weights,
        )
        cuped, fit = cuped_estimate(
            total.treatment,
            total.control,
            metric=data.metric,
            variant=data.treated,
            control_name=data.control,
            alpha=alpha,
            expected_weights=weights,
        )

    # 头条与对照口径。任何路径都走这一个入口 ——
    # 保证"报告里的结论"和"监控曲线的判定"永远是同一个估计量。
    primary, primary_name, alt, alt_name = _headline_path(data, total)
    if unit_mean:
        # 单元 + 人均路径下，把带名称/诊断的版本换回来（_headline_path 走的是无标签版本）
        # 这条路只有均值指标 + 用户级才会进来，所以 naive 与 CUPED 一定都在；
        # 不在就明说 —— 类型上的 Optional 是真实的（比值/簇级路径确实没有 CUPED）。
        if naive is None or cuped is None:
            raise ValueError("均值指标的用户级分析必须同时给出 post-only 与 CUPED 两个口径")
        primary = cuped if primary_name == "cuped" else naive
        alt = naive if primary_name == "cuped" else cuped
    if primary_name == "cluster_cuped":
        # 簇级 CUPED 也要落到 `cuped` / `cuped_fit` 两个字段上。
        # 否则报告里"用了簇级 CUPED"只能从 primary_estimator_name 看出来，
        # 而 to_dict()/API 的 cuped 字段是 None —— 读起来像"没跑"，
        # 正是"声明了却看不出来"的那类毛病。
        cuped, fit = total.cluster_cuped()

    # ---- 3. 序贯监控路径 --------------------------------------------------- #
    design = design_for(data, alpha)
    monitoring = _monitoring_from(data, design)

    # ---- 4. 体检汇总 ------------------------------------------------------- #
    # SRM 排在第一位，而且**只出现一次**：``cuped_estimate`` 自己也会挂一条 SRM，
    # 直接拼接会重复。按名字去重，保留第一次出现的那条。
    checks: list[CheckItem] = [CheckItem.from_diagnostic(srm)]
    seen = {"SRM"}
    for source_est in (cuped, primary):
        for diag in () if source_est is None else source_est.diagnostics:
            if diag.name in seen:
                continue
            seen.add(diag.name)
            checks.append(CheckItem.from_diagnostic(diag))

    # ---- 5. 效应分解 ------------------------------------------------------- #
    # naive 与 CUPED 的**点估计之差恰好等于 θ̂·ΔX̄**（前置协变量的组间差乘上回归系数），
    # 也就是"这个效应里有多少只是处置前就不平衡"。把它显式报出来，
    # 是因为一个显著结果最容易被误读的地方就在这里 —— 尤其当真实效应为零时，
    # naive 的显著性可能**全部**来自失衡，而 CUPED 会把它扣掉。
    # 只有"单元 + 人均"路径才有这一对口径；簇/比值路径没有协变量校正这一步。
    imbalance: float | None = None
    if naive is not None and cuped is not None:
        imbalance = naive.absolute_effect - cuped.absolute_effect
        checks.append(
            CheckItem(
                name="效应分解",
                status="info",
                message=(
                    f"naive 效应 {naive.absolute_effect:+.4f} = 前置协变量失衡贡献 {imbalance:+.4f}"
                    f"（CUPED 扣掉） + 校正后残余 {cuped.absolute_effect:+.4f}（真实效应 + 结果噪声）"
                    + (
                        f"；失衡占 naive 的 {imbalance / naive.absolute_effect:.1%}"
                        if abs(naive.absolute_effect) > 1e-12
                        else ""
                    )
                ),
                statistic=imbalance,
            )
        )

    crossed_at = next((m["look"] for m in monitoring if m["crossed"]), None)
    # 头条口径与对照口径是否给出**不同**的越界结论 —— 这正是"口径不一致"的可见后果，
    # 所以把它算出来明说，而不是让读者自己去比两个 z。
    def _crossed_by(key: str) -> bool:
        return any(
            m[key] is not None and abs(m[key]) >= m["boundary"] for m in monitoring
        )

    primary_crossed = any(m["crossed"] for m in monitoring)
    alt_crossed = _crossed_by("alt_z") if alt_name is not None else None
    warn = ""
    if alt_crossed is not None and alt_crossed != primary_crossed:
        warn = (
            f"；**两个口径的越界结论不同**（{alt_name} "
            f"{'越界' if alt_crossed else '未越界'} vs {primary_name} "
            f"{'越界' if primary_crossed else '未越界'}），判定按 {primary_name}"
        )
    unit_tag = "簇（随机化单元）" if data.analysis_unit == "cluster" else "用户"
    look_note = data.extra.get("look_note")
    checks.append(
        CheckItem(
            name="序贯监控",
            status="info",
            message=(
                f"判定口径 = {primary_name}（与头条结论同一个估计量，分析单元 = {unit_tag}）；"
                f"{data.n_looks} 次查看中"
                f"{'第 ' + str(crossed_at) + ' 次触及边界' if crossed_at else '始终未触及边界'}；"
                f"末次边界 {design.final_boundary:.4f}（名义 p={design.final_nominal_p:.4f}）；"
                f"always-valid p（运行最小）="
                f"{float(monitoring[-1]['always_valid_p_running_min']):.4f}。"
                "OBF 边界与 always-valid p 不一致是正常的：前者按**计划的查看次数**换来，"
                "后者对**任意**查看次数都有效，代价是更保守"
                + (f"。查看点：{look_note}" if look_note else "")
                + warn
            ),
            statistic=design.final_boundary,
        )
    )

    # ---- 5b. 分析单元 / 指标类型 ------------------------------------------- #
    if data.analysis_unit == "cluster":
        g_t, g_c = total.n_clusters
        unit_level = total.post_only()
        checks.append(
            CheckItem(
                name="分析单元",
                status="info",
                message=(
                    f"整簇随机化：{g_t} / {g_c} 个簇（SRM 检验的是**簇数**，不是用户数）。"
                    f"簇级检验给出 {total.cluster_level().absolute_effect:+.4f}"
                    f"（SE {total.cluster_level().std_error:.4f}）；"
                    f"若误用单元级检验则是 {unit_level.absolute_effect:+.4f}"
                    f"（SE {unit_level.std_error:.4f}）—— SE 被低估 "
                    f"{1 - unit_level.std_error / total.cluster_level().std_error:.1%}，"
                    "而那个 p 值看起来完全正常（M1 实测这类误用的 I 类错误率 64.5%）"
                ),
                statistic=float(total.cluster_level().std_error),
            )
        )
    if data.metric_type == "ratio":
        naive_t = data.extra.get("naive_ratio_treatment")
        naive_c = data.extra.get("naive_ratio_control")
        gap = ""
        if naive_t is not None and naive_c is not None:
            naive_effect = float(naive_t) - float(naive_c)
            delta_effect = total.ratio_delta().absolute_effect
            if abs(naive_effect) > 1e-12:
                gap = (
                    f"；人均比值口径给出 {naive_effect:+.6f}，与业务口径相差 "
                    f"{(delta_effect - naive_effect) / naive_effect:+.1%} —— "
                    "两者都「校准」，但答的不是同一个问题"
                )
        checks.append(
            CheckItem(
                name="指标类型",
                status="info",
                message=(
                    f"比值指标：估计量是**业务口径** Σy/Σx = "
                    f"{total.ratio_delta().mean_treatment:.6f} vs "
                    f"{total.ratio_delta().mean_control:.6f}，"
                    f"效应 {total.ratio_delta().absolute_effect:+.6f}"
                    f"（SE {total.ratio_delta().std_error:.6f}，delta method）" + gap
                ),
                statistic=float(total.ratio_delta().absolute_effect),
            )
        )

    # ---- 6. 功效读数：这份数据能检出多大的效应 ------------------------------ #
    # 用**头条口径**的 SE 算 MDE。这一段的业务含义比"方差缩减 49%"更直接：
    # CUPED 把 SE 降到 √(1-ρ²) 倍，MDE 就跟着降到同样的倍数 ——
    # "以前要跑两倍的量才能看出来的效应，现在这个量就能看出来"。
    primary_est = primary
    mde_abs = mde(float(primary_est.std_error), alpha=alpha, power=0.8)
    # 只有"单元 + 人均 + CUPED"这一条路径上，"MDE 相对 post-only 收缩"才是有意义的说法：
    # 簇级口径的 MDE 本来就会**大于**（错用的）单元级口径，那叫"多花的代价"，不叫收益。
    mde_shrinkage: float | None = None
    if unit_mean and naive is not None and cuped is not None:
        mde_post = mde(float(naive.std_error), alpha=alpha, power=0.8)
        mde_shrinkage = 1.0 - mde_abs / mde_post if mde_post > 0 else None
    baseline = abs(float(total.control.mean_y))
    power_block: dict[str, Any] = {
        "alpha": alpha,
        "target_power": 0.8,
        "estimator": primary_name,
        "se": float(primary_est.std_error),
        "mde_abs": mde_abs,
        "mde_relative": mde_abs / baseline if baseline > 0 else None,
        "power_at_observed": z_power(primary_est.absolute_effect, primary_est.std_error, alpha),
        "mde_shrinkage_vs_post_only": mde_shrinkage,
        "baseline_mean": baseline,
    }
    checks.append(
        CheckItem(
            name="功效 / MDE",
            status="info",
            message=(
                f"以 {primary_name} 口径、alpha={alpha}、功效 0.8 计，"
                f"这份数据能检出的最小效应 = {mde_abs:.4f}"
                + (f"（相对基线 {mde_abs / baseline:.2%}）" if baseline > 0 else "")
                + f"；当前观测效应的功效 = {power_block['power_at_observed']:.1%}。"
                + (
                    f"MDE 比 post-only 口径小 {mde_shrinkage:.1%}"
                    "（这就是 CUPED 在业务上的样子：同样的样本量能看出更小的改动）"
                    if mde_shrinkage is not None
                    else "（当前分析单元/指标类型下，该对照口径不适用）"
                )
            ),
            statistic=mde_abs,
        )
    )
    # ---- 护栏指标：**真的判定它们**（这一块原来是"声明了但没人看"） ---------- #
    #
    # 历史：``guardrails`` 字段一直存在、界面上也显示了，而引擎从头到尾没读过它 ——
    # 于是用户合理地以为护栏被看着。当时（正确地）选择"明说没分析"而不是假装分析。
    # 现在有了数据模型（每个护栏一条按臂的充分统计量）与**事先声明的方向与容忍度**，
    # 于是可以真的判定：``fail`` 意味着"有把握地越过容忍度"，进 health，
    # 并在报告里给出**建议停止实验** —— Kohavi 那本书里护栏触发是停实验的理由，
    # 不是参考信息。
    #
    # 三种"未知"必须分开说，因为它们的补救办法完全不同：
    #   1. 声明了名字但没声明方向/容忍度 -> 让人去补声明；
    #   2. 声明了规格但没有数据（数仓路径还没有护栏表）-> 让人去接数据；
    #   3. 什么都没声明 -> 不出现这一项。
    # 三者的共同点是：**都不是"通过"**。
    declared_specs = list(getattr(data, "guardrail_specs", ()) or ())
    declared_names = list(data.guardrails or ())
    if declared_specs or declared_names:
        from .guardrails import GuardrailSpec, analyse_guardrails

        have = {s.name for s in declared_specs}
        bare = [GuardrailSpec(name=n) for n in declared_names if n not in have]
        guard = analyse_guardrails(
            [*declared_specs, *bare],
            dict(getattr(data, "guardrail_series", {}) or {}),
            treated=data.treated,
            control=data.control,
            alpha=alpha,
        )
        checks.append(
            CheckItem(
                name="护栏指标",
                status=guard.check_status,
                message=guard.summary() + "  " + _guardrail_explainer(guard),
                statistic=float(guard.n_declared),
            )
        )

    statuses = {c.status for c in checks}
    health = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")

    return ExperimentReport(
        experiment_id=data.extra.get("experiment_id", ""),
        experiment_name=data.experiment,
        status=data.extra.get("status", ""),
        n_users=int(total.treatment.n + total.control.n),
        alpha=alpha,
        health=health,
        checks=checks,
        naive=naive,
        cuped=cuped,
        cuped_fit=fit,
        primary=primary,
        alt=alt,
        primary_estimator_name=primary_name,
        alt_estimator_name=alt_name,
        sequential=design,
        monitoring=monitoring,
        srm=srm,
        imbalance_component=imbalance,
        residual_component=cuped.absolute_effect if cuped is not None else None,
        source=data.source,
        population_size=data.extra.get("n_users"),
        estimator=primary_name,
        analysis_unit=data.analysis_unit,
        metric_type=data.metric_type,
        n_analysis_units=(
            sum(total.n_clusters) if data.analysis_unit == "cluster"
            else int(total.treatment.n + total.control.n)
        ),
        power=power_block,
    )


def analyse_experiment(
    record: ExperimentRecord,
    *,
    n_users: int = 20_000,
    alpha: float = 0.05,
    n_looks: int = 5,
    seed: int | None = None,
) -> ExperimentReport:
    """合成数据路径：按 salt 确定性生成数据，然后跑 ``analyse_data``。

    多臂实验的约定：**对比"最后一臂 vs 第一臂"**（``variants[-1]`` vs ``variants[0]``），
    中间臂不参与。
    """
    spec = record.to_spec()
    if len(spec.variants) < 2:
        raise ValueError("至少需要两个分支才能做对比分析")
    # 派生种子的两个讲究：
    #   1. 用项目自己的 murmur3，而不是内置 ``hash()`` —— 后者对字符串哈希
    #      **每个进程都加盐**，会让"同一个实验重启后数字全变"，演示不再可复现。
    #   2. 挂在 **salt** 上而不是 ``record.id``：id 是每次入库新生成的 uuid4，
    #      删库重建（``--reset``）就会换 id，于是演示数字又变了。
    #      salt 才是这个项目里"决定每个用户分组"的不可变量，数据生成跟着它走才对。
    base_seed = seed if seed is not None else murmur3_32(record.salt.encode("utf-8"))

    data = build_synthetic_data(
        experiment=record.name,
        salt=record.salt,
        variants=[(v.name, v.weight) for v in spec.variants],
        metric=record.primary_metric,
        n_users=n_users,
        n_looks=n_looks,
        true_lift=record.true_lift,
        traffic_ratio=record.traffic_ratio,
        seed=base_seed,
        population=PLATFORM_POPULATION,
        primary_estimator=record.estimator,
        analysis_unit=record.analysis_unit,
        metric_type=record.metric_type,
        guardrail_specs=tuple(record.guardrail_specs),
    )
    data = _with_record_metadata(data, record)
    return analyse_data(data, alpha=alpha)


def analyse_experiment_from_warehouse(
    record: ExperimentRecord,
    con,
    *,
    alpha: float = 0.05,
    n_looks: int = 5,
) -> ExperimentReport:
    """数仓路径：从 ADS + DWS 读数，然后跑**同一个** ``analyse_data``。

    这一步是"数仓层与推断层的接口面"真正被用起来的地方：
    平台不碰明细，只读充分统计量；而 CUPED 的原料
    ``pre_post_cross_sum`` 正是 M1 为了让这条路径成立而加进 DWS 的那一列。

    **记录上的声明必须原样传下去。** 第一版漏了 ``analysis_unit``，
    于是勾了"整簇随机化"的实验被静默按单元级分析 —— 声明写了、没人理。
    这类"静默丢弃声明"是本项目反复在防的错，而它恰好又是最容易漏的一处，
    因为函数签名本身不会报错。
    """
    if not record.warehouse_experiment:
        raise ValueError(f"实验 {record.name!r} 没有绑定数仓实验")
    if record.metric_type not in ("mean", "ratio"):
        raise ValueError(f"数仓路径不支持 metric_type={record.metric_type!r}")
    data = build_warehouse_data(
        con,
        record.warehouse_experiment,
        metric=record.primary_metric,
        n_looks=n_looks,
        primary_estimator=record.estimator,
        analysis_unit=record.analysis_unit,
        # **必须把口径传下去**：这条链路曾经因为漏传 analysis_unit 而把
        # "整簇随机化"的实验静默按单元级分析（M6 的 01 节）。
        # metric_type 是同一个坑的第二次入口 —— 漏了它，比值实验会被
        # 当成人均指标读，而两条链路的列名与含义都不同，数字会**静默错**。
        metric_type=record.metric_type,
    )
    data = _with_record_metadata(data, record)
    return analyse_data(data, alpha=alpha)


def _guardrail_explainer(guard) -> str:
    """把"为什么是未知"写到读者能照做 —— 三种未知的补救办法完全不同。"""
    if guard.verdict == "stop":
        return (
            "**这是停实验的理由**：护栏触发不是参考信息。"
            "注意判定用的是伤害的**置信下界**超过事先声明的容忍度，"
            "而且多条护栏做了 Bonferroni 校正（alpha/K）—— "
            "宁可少报几条，也不要因为多看几个指标就误停一次实验。"
        )
    if guard.verdict == "watch":
        return "点估计超了但证据不足，继续观察；攒够样本再看一次。"
    if guard.verdict == "unknown":
        return (
            "**这不是通过**。补救办法：只有名字的护栏要补 direction + max_harm；"
            "有规格但没数据的（例如数仓路径还没有护栏表）要先接数据。"
            "缺数据时判通过会让人以为护栏被看着 —— 那正是这一块原来的毛病。"
        )
    return (
        f"全部护栏在容忍度内（{guard.n_analysed} 条可判定，"
        f"校正后 alpha={guard.outcomes[0].alpha_adjusted if guard.outcomes else float('nan'):.4f}）。"
    )


def _with_record_metadata(data: ExperimentData, record: ExperimentRecord) -> ExperimentData:
    """把注册表里的展示字段带进数据对象（报告的标题栏要用）。

    **护栏也在这里挂上去**，而且是刻意放在这一个函数里：它是"记录 → 数据对象"的
    唯一通道，所以合成路径与数仓路径都会带上护栏，不可能只挂一条。
    """
    from dataclasses import replace as _replace

    return _replace(
        data,
        guardrails=tuple(record.guardrails or ()),
        guardrail_specs=tuple(record.guardrail_specs or ()),
        extra={
            **data.extra,
            "experiment_id": record.id,
            "status": record.status,
            "salt": record.salt,
        },
    )


def run_aa_validation(
    *,
    n_trials: int = 400,
    n_units: int = 8_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """在线跑一次 A/A 验证 —— 让"框架是校准的"这个主张可以被点出来。

    刻意用**小规模**：这是给人现场点的，不是替代 ``run_m0_validation.py``
    的完整验证。返回值里带上样本量，避免把一次快速检查误读成完整结论。
    """
    from ..sim.generator import PopulationConfig as _PopCfg
    from ..sim.generator import generate_population, two_arm_spec
    from ..validation import run_aa_trials

    pop = generate_population(_PopCfg(n_units=n_units, seed=seed))
    result = run_aa_trials(
        pop,
        two_arm_spec("aa_live", salt="aa_live_v1"),
        n_trials=n_trials,
        alpha=alpha,
        mode="randomized",
        seed=seed + 1,
    )
    fpr_lo, fpr_hi = result.fpr_interval()
    cov_lo, cov_hi = result.coverage_interval()
    ks_d, ks_p = result.uniformity_test()

    return {
        "n_trials": result.n_trials,
        "n_units": result.n_units,
        "alpha": alpha,
        "empirical_fpr": result.empirical_fpr(),
        "fpr_interval": [fpr_lo, fpr_hi],
        "coverage": result.coverage(),
        "coverage_interval": [cov_lo, cov_hi],
        "ks_statistic": ks_d,
        "ks_p_value": ks_p,
        "calibrated": bool(fpr_lo <= alpha <= fpr_hi and ks_p > 0.05),
        "note": (
            "这是现场快速检查，样本量有限；完整验证见 reports/m0_validation.md"
        ),
    }
