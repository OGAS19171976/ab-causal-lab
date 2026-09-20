"""因果推断审计：**假设被打破时，我会错得多离谱，而且我会不会察觉？**

M0–M2 的验证台问的是"我的数算得对不对"（校准）。
M3 没有随机化撑腰，问题变成三个：

1. **估计量对不对**：交错处置下 TWFE 与 CS 各自偏多少、会不会连符号都错
2. **伪证检验有没有用**：处置前的平行趋势检验能发现哪类违背、**发现不了哪类**
3. **推断可不可信**：合成控制的安慰剂"p 值"在无效应时的假阳性率是多少

第 2 条是 M3 最重要的一课，也是它和 M1"校准不等于正确"的呼应：
**一个检验可以通过，同时它想守的那个假设已经破了。**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..causal.did import (
    callaway_santanna,
    pretrend_test,
    twfe,
    twfe_decomposition,
)
from ..causal.panel import StaggeredPanelConfig, generate_staggered_panel
from ..causal.sensitivity import (
    rambachan_roth_smoothness,
    trend_sensitivity,
)
from ..causal.synthetic import (
    SCMConfig,
    generate_scm_scenario,
    leave_one_out,
    placebo_inference,
    time_placebo,
)
from .aa import wilson_interval

__all__ = [
    "AggregationVarianceAudit",
    "EstimatorComparison",
    "PretrendAudit",
    "SCMAudit",
    "SensitivityAudit",
    "run_aggregation_variance_audit",
    "run_staggered_estimator_comparison",
    "run_pretrend_audit",
    "run_scm_audit",
    "run_sensitivity_audit",
]


# --------------------------------------------------------------------------- #
# 一、交错处置下 TWFE 与 CS 的对照
# --------------------------------------------------------------------------- #
@dataclass
class EstimatorComparison:
    """两个估计量在一批仿真面板上的表现（真值已知）。"""

    n_trials: int
    truths: np.ndarray
    twfe_estimates: np.ndarray
    cs_estimates: np.ndarray
    twfe_negative_weight_share: float
    twfe_weight_effect_corr: float

    @property
    def twfe_bias(self) -> float:
        return float(np.mean(self.twfe_estimates - self.truths))

    @property
    def cs_bias(self) -> float:
        return float(np.mean(self.cs_estimates - self.truths))

    @property
    def twfe_rmse(self) -> float:
        return float(np.sqrt(np.mean((self.twfe_estimates - self.truths) ** 2)))

    @property
    def cs_rmse(self) -> float:
        return float(np.sqrt(np.mean((self.cs_estimates - self.truths) ** 2)))

    @property
    def twfe_sign_flip_rate(self) -> float:
        """真值为正、TWFE 却估成负的比例 —— 最刺眼的失效方式。"""
        return float(np.mean((self.truths > 0) & (self.twfe_estimates < 0)))

    @property
    def cs_sign_flip_rate(self) -> float:
        return float(np.mean((self.truths > 0) & (self.cs_estimates < 0)))

    def summary(self) -> str:
        return (
            f"交错处置：TWFE vs Callaway-Sant'Anna（{self.n_trials} 次仿真，真值已知）\n"
            f"  真值均值 = {self.truths.mean():+.4f}\n"
            f"  TWFE: 偏置 {self.twfe_bias:+.4f}   RMSE {self.twfe_rmse:.4f}   "
            f"符号翻转率 {self.twfe_sign_flip_rate:.1%}\n"
            f"  CS  : 偏置 {self.cs_bias:+.4f}   RMSE {self.cs_rmse:.4f}   "
            f"符号翻转率 {self.cs_sign_flip_rate:.1%}\n"
            f"  TWFE 隐式权重：处置后负权重占比 {self.twfe_negative_weight_share:.1%}，"
            f"权重与真实效应相关 {self.twfe_weight_effect_corr:+.3f}"
        )


def run_staggered_estimator_comparison(
    config: StaggeredPanelConfig | None = None,
    *,
    n_trials: int = 200,
    seed: int = 0,
) -> EstimatorComparison:
    """反复生成面板，比较两个估计量相对真值的偏置。

    数据里**每一个单元的每一期真实效应都是正的**，
    所以"估计为负"一定是估计量的问题，不是数据的问题。
    """
    cfg = config or StaggeredPanelConfig(
        n_units=800,
        n_periods=7,
        cohorts=(2, 4),
        cohort_weights=(0.5, 0.5),
        never_treated_share=0.04,
        effects=(1.0, 2.0, 3.0, 4.0),
        cohort_effect_multiplier=(1.0, 0.25),
        noise_sd=0.5,
    )

    truths, tw, cs = [], [], []
    neg_share, corr = [], []

    for i in range(n_trials):
        trial_cfg = StaggeredPanelConfig(**{**cfg.__dict__, "seed": seed + i})
        panel, truth = generate_staggered_panel(trial_cfg)
        truths.append(truth.overall_att)
        tw.append(twfe(panel).absolute_effect)
        cs.append(callaway_santanna(panel).overall.absolute_effect)
        if i == 0:
            dec = twfe_decomposition(panel, truth)
            neg_share.append(dec.negative_post_weight_share)
            corr.append(dec.weight_effect_correlation())

    return EstimatorComparison(
        n_trials=n_trials,
        truths=np.asarray(truths),
        twfe_estimates=np.asarray(tw),
        cs_estimates=np.asarray(cs),
        twfe_negative_weight_share=float(np.mean(neg_share)) if neg_share else float("nan"),
        twfe_weight_effect_corr=float(np.mean(corr)) if corr else float("nan"),
    )


# --------------------------------------------------------------------------- #
# 一之二、聚合方差：独立合成会让 size 膨胀多少
# --------------------------------------------------------------------------- #
@dataclass
class AggregationVarianceAudit:
    """在 H0（真实效应处处为零）下，两种聚合方差算法给出的 I 类错误率。

    为什么值得单独审：整体 ATT 是若干 ``ATT(g,t)`` 的加权和，而它们
    **共用同一批对照单元、相邻队列还共用基准期** —— 相关性非负。
    把标准误按 ``sqrt(Σw²se²)`` 独立合成，会**低估** SE，
    于是"5% 的检验"实际上拒绝得更多。这件事在 ``_att_influence`` 的文档里
    早就写过（"实测把 size 从 5% 抬到 11%"），但当时只用在了 lead 的联合检验上。

    这里把差别量出来：同一批仿真、同一个点估计，只换方差算法。

    **两次排查的结果差别很大，如实记在这里**：

    * **整体 ATT**：独立合成低估约 44%，H0 下越界率 0.22（应为 0.05）、
      覆盖率 0.78（应为 0.95）。这是真 bug —— 它横跨 7 个相对期数，
      而这些格子共用对照单元、相邻队列还共用基准期。
    * **事件研究（逐个 k）**：同样的写法，但低估只有 **0~2%**。
      原因不是"写法对了"，而是**结构不同**：单个 k 内往往只有 1~2 个格子
      （处置前的 k 常常只有一个队列），彼此只共用对照组。
      也就是说：**同一个错误的后果取决于相关性结构，不能按"公式看起来一样"外推。**
      这一处仍然改了 —— 它更正确，而且让两条聚合路径共用同一个 `_se_from_influence`。

    另外注意 ``pretrend_*_reject_rate`` **不该**被读成"处置前的 size"：
    它统计的是"**至少有一个**处置前系数显著"，而各 k 之间高度相关、
    又没做多重比较校正，所以它天然高于 α（实测约 0.12）。
    逐个系数的 5% 是没问题的；要判断"整条处置前路径是否异常"，
    该用 ``pretrend_test`` 那个**联合**检验，而不是数有几个星号。
    """

    n_trials: int
    n_units: int
    alpha: float
    #: 用**影响函数**合成（正确）：越界率应当 ≈ alpha
    influence_reject_rate: float
    #: 用**独立合成**（旧写法）：越界率会明显高于 alpha
    naive_reject_rate: float
    mean_se_influence: float
    mean_se_naive: float
    #: 独立合成把 SE 低估的相对幅度（按每个面板取比值再平均）
    mean_se_understatement: float
    #: 两种算法下 95% 区间覆盖 0 的比例（H0 下应当 ≈ 95%）
    influence_coverage: float
    naive_coverage: float
    #: **处置前（placebo）**的越界率：取每个面板里所有 k<0 的系数，
    #: 只要有一个在名义 5% 下"显著"就算一次越界。
    #:
    #: 这个数比整体 ATT 那个更要紧：处置前的系数正是"平行趋势看起来成立吗"
    #: 的唯一依据。方差被低估 → 处置前的显著变多 → **平行趋势会被误判为不成立**
    #: （或者反过来，使用者以为自己检验过了）。
    pretrend_naive_reject_rate: float
    pretrend_influence_reject_rate: float

    def summary(self) -> str:
        return "\n".join(
            [
                f"H0 下两种聚合方差（{self.n_trials} 次仿真，每次 {self.n_units:,} 单元）",
                f"  影响函数合成：越界率 {self.influence_reject_rate:.4f}"
                f"（α={self.alpha}），覆盖率 {self.influence_coverage:.4f}",
                f"  独立合成　　：越界率 {self.naive_reject_rate:.4f}，"
                f"覆盖率 {self.naive_coverage:.4f}",
                f"  平均 SE：{self.mean_se_influence:.4f} vs {self.mean_se_naive:.4f}"
                f"（低估 {self.mean_se_understatement:.1%}）",
                f"  处置前（placebo）越界率：{self.pretrend_influence_reject_rate:.4f}"
                f" vs {self.pretrend_naive_reject_rate:.4f}"
                "（两者通常相同 —— 同一 k 内往往只有一个队列，聚合本身没什么可差的；"
                "而这个数**受多重比较影响**，不等于逐系数的 size，"
                "要判断整条处置前路径请用 pretrend_test 的联合检验）",
            ]
        )


def run_aggregation_variance_audit(
    config: StaggeredPanelConfig | None = None,
    *,
    n_trials: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> AggregationVarianceAudit:
    """H0 下比较两种聚合方差的越界率与覆盖率。"""
    from ..causal.did import callaway_santanna

    cfg = config or StaggeredPanelConfig(
        n_units=600,
        n_periods=7,
        cohorts=(2, 4),
        cohort_weights=(0.5, 0.5),
        never_treated_share=0.2,
        effects=(0.0,),  # 真实效应处处为零
        noise_sd=1.0,
    )

    inf_reject = naive_reject = inf_cov = naive_cov = 0
    pre_naive = pre_inf = 0
    ses_inf: list[float] = []
    ses_naive: list[float] = []
    for i in range(n_trials):
        trial_cfg = StaggeredPanelConfig(**{**cfg.__dict__, "seed": seed + i})
        panel, _truth = generate_staggered_panel(trial_cfg)
        cs = callaway_santanna(panel, alpha=alpha)
        effect = cs.overall.absolute_effect
        se = cs.overall.std_error
        se_naive = cs.naive_overall_se
        ses_inf.append(se)
        ses_naive.append(se_naive)

        z = 1.959963984540054  # 正态 97.5% 分位；大样本下与 t 分位几乎一致
        inf_reject += int(abs(effect) > z * se)
        naive_reject += int(abs(effect) > z * se_naive)
        inf_cov += int(abs(effect) <= z * se)
        naive_cov += int(abs(effect) <= z * se_naive)

        # 处置前的系数：同一批点估计，只换 SE 的算法。
        # naive 版的 SE 用 sqrt(Σ w² se²) 重算 —— 那是修之前的写法。
        for k, est in cs.event_study.items():
            if k >= 0:
                continue
            pre_inf += int(abs(est.absolute_effect) > z * est.std_error)
            rows = [
                e for (g, t), e in cs.group_time.items() if t - g == k
            ]
            if rows:
                w = np.array([e.n_treatment for e in rows], dtype=float)
                w = w / w.sum()
                se_naive_k = float(
                    np.sqrt(sum(wi**2 * e.std_error**2 for wi, e in zip(w, rows)))
                )
                pre_naive += int(abs(est.absolute_effect) > z * se_naive_k)

    se_inf = float(np.mean(ses_inf))
    se_naive = float(np.mean(ses_naive))
    return AggregationVarianceAudit(
        n_trials=n_trials,
        n_units=cfg.n_units,
        alpha=alpha,
        influence_reject_rate=inf_reject / n_trials,
        naive_reject_rate=naive_reject / n_trials,
        mean_se_influence=se_inf,
        mean_se_naive=se_naive,
        mean_se_understatement=float(
            1.0 - np.mean([n / i for n, i in zip(ses_naive, ses_inf)])
        ),
        influence_coverage=inf_cov / n_trials,
        naive_coverage=naive_cov / n_trials,
        pretrend_naive_reject_rate=pre_naive / n_trials,
        pretrend_influence_reject_rate=pre_inf / n_trials,
    )


# --------------------------------------------------------------------------- #
# 二、平行趋势检验：能发现什么、发现不了什么
# --------------------------------------------------------------------------- #
@dataclass
class PretrendAudit:
    """平行趋势伪证检验的 size 与三类 power。"""

    n_trials: int
    alpha: float
    size: float
    power_trend_violation: float
    power_post_divergence: float
    twfe_bias_trend_violation: float
    twfe_bias_post_divergence: float

    @property
    def size_interval(self) -> tuple[float, float]:
        return wilson_interval(int(round(self.size * self.n_trials)), self.n_trials)

    @property
    def blind_spot(self) -> bool:
        """检验对"处置后才分岔"是否基本无功效（拒绝率接近 size）。"""
        return self.power_post_divergence < self.size + 0.10

    def summary(self) -> str:
        lo, hi = self.size_interval
        lines = [
            f"平行趋势检验的表现（{self.n_trials} 次仿真，名义 alpha={self.alpha}）",
            f"  size（平行趋势成立时误报）        = {self.size:.4f} [{lo:.4f}, {hi:.4f}]",
            f"  power（队列专属趋势，处置前可见）  = {self.power_trend_violation:.4f}",
            f"  power（**处置后才分岔**，处置前不可见）= {self.power_post_divergence:.4f}",
            "",
            f"  有队列专属趋势时 TWFE 偏置 = {self.twfe_bias_trend_violation:+.4f}",
            f"  处置后才分岔时 TWFE 偏置   = {self.twfe_bias_post_divergence:+.4f}",
        ]
        if self.blind_spot:
            lines.append(
                "  -> 检验对「处置后才分岔」几乎没有功效：假设已经破了，检验却照常通过。"
            )
        return "\n".join(lines)


def run_pretrend_audit(
    config: StaggeredPanelConfig | None = None,
    *,
    n_trials: int = 300,
    alpha: float = 0.05,
    trend_violation: float = 0.35,
    post_divergence: float = 0.6,
    seed: int = 0,
) -> PretrendAudit:
    """量平行趋势检验的 size、对可见违背的功效、以及对不可见违背的功效。"""
    base = config or StaggeredPanelConfig(
        n_units=600,
        n_periods=8,
        cohorts=(3, 5, 7),
        cohort_weights=(0.25, 0.25, 0.25),
        never_treated_share=0.25,
        effects=(1.0, 1.5, 2.0, 2.5),
        noise_sd=0.6,
    )

    def run(**patch) -> tuple[float, float]:
        rejects = 0
        biases = []
        for i in range(n_trials):
            cfg = StaggeredPanelConfig(**{**base.__dict__, **patch, "seed": seed + i})
            panel, truth = generate_staggered_panel(cfg)
            diag = pretrend_test(panel)
            rejects += int(diag.status == "warn")
            biases.append(twfe(panel).absolute_effect - truth.overall_att)
        return rejects / n_trials, float(np.mean(biases))

    size, _ = run()
    power_visible, bias_visible = run(trend_violation=trend_violation)
    power_invisible, bias_invisible = run(post_divergence=post_divergence)

    return PretrendAudit(
        n_trials=n_trials,
        alpha=alpha,
        size=size,
        power_trend_violation=power_visible,
        power_post_divergence=power_invisible,
        twfe_bias_trend_violation=bias_visible,
        twfe_bias_post_divergence=bias_invisible,
    )


# --------------------------------------------------------------------------- #
# 三、合成控制：
# --------------------------------------------------------------------------- #
@dataclass
class SCMAudit:
    """合成控制 + 安慰剂推断的假阳性率与功效。"""

    n_trials: int
    alpha: float
    false_positive_rate: float
    power: float
    rmse_ratio_quantiles: np.ndarray
    att_bias: float

    def summary(self) -> str:
        lo, hi = wilson_interval(
            int(round(self.false_positive_rate * self.n_trials)), self.n_trials
        )
        return (
            f"合成控制安慰剂推断（{self.n_trials} 次仿真，名义 alpha={self.alpha}）\n"
            f"  无效应时「p < alpha」的比例 = {self.false_positive_rate:.4f} "
            f"[{lo:.4f}, {hi:.4f}]\n"
            f"  有效应时的检出率 = {self.power:.4f}\n"
            f"  ATT 偏置 = {self.att_bias:+.4f}\n"
            f"  处置后/处置前 RMSE 比值分位数 = "
            + " / ".join(f"{q:.2f}" for q in self.rmse_ratio_quantiles)
        )


@dataclass
class SCMPlaceboAudit:
    """空间安慰剂 / 时间安慰剂 / 留一法：三条检查各自的角色。

    **三条检查回答三个不同的问题**，混在一起读就会得出错误结论：

    * **空间安慰剂**（``placebo_inference``）：别的**单元**会不会也这样 ——
      它给排名 p 值，是这三条里唯一能"当检验用"的；
    * **时间安慰剂**（``time_placebo``）：别的**时段**会不会也这样。
      它**只看处置前窗口**，所以与有没有真实效应**无关** ——
      这一节特意把"同一个种子下 H0 与 H1 的读数逐位相同"当成不变量量出来。
      它的用途是诊断"这条合成对照在留出的处置前窗口上站不站得住"，
      不是检验效应；
    * **留一法**（``leave_one_out``）：结论会不会被某一个捐赠单元撑着。
    """

    n_trials: int
    alpha: float
    effect: float
    #: 空间安慰剂：H0 下的误报率与 H1 下的功效
    space_fpr: float
    space_power: float
    #: 时间安慰剂"不干净"的比例（H0 / H1）—— 两者**应当接近**（它与效应无关）
    time_not_clean_h0: float
    time_not_clean_h1: float
    #: 同种子下 H0 与 H1 的时间安慰剂读数**逐位相同**的比例（不变量的实测）
    time_effect_blind_share: float
    #: 留一法：H1 下符号稳定的比例与中位移动幅度
    loo_sign_stable_h1: float
    loo_median_shift_h1: float

    @property
    def space_is_calibrated(self) -> bool:
        """空间安慰剂的 H0 误报率不超过名义值 +0.05（小样本，容差放宽）。"""
        return self.space_fpr <= self.alpha + 0.05

    @property
    def space_has_power(self) -> bool:
        return self.space_power >= 0.8

    @property
    def time_placebo_is_effect_blind(self) -> bool:
        """时间安慰剂在 H0/H1 下逐位相同 —— 它是处置前窗口的诊断，不是效应检验。"""
        return self.time_effect_blind_share >= 0.999

    @property
    def loo_is_stable(self) -> bool:
        return self.loo_sign_stable_h1 >= 0.95 and self.loo_median_shift_h1 <= 0.15

    def summary(self) -> str:
        return "\n".join(
            [
                f"SCM 的三条安慰剂检查（各 {self.n_trials} 次，alpha={self.alpha}，"
                f"真实效应 {self.effect:g}）",
                f"  空间安慰剂：H0 误报率 {self.space_fpr:.4f}，H1 功效 {self.space_power:.4f}",
                f"  时间安慰剂：不干净的比例 H0 {self.time_not_clean_h0:.4f} / "
                f"H1 {self.time_not_clean_h1:.4f}",
                f"    **同种子下 H0/H1 读数逐位相同**的比例 "
                f"{self.time_effect_blind_share:.4f} —— 它只看处置前窗口，与效应无关",
                f"  留一法：H1 下符号稳定 {self.loo_sign_stable_h1:.4f}，"
                f"中位移动 {self.loo_median_shift_h1:.1%}",
                "  读法：三条回答三个不同的问题（别的单元 / 别的时段 / 是否靠某一个捐赠单元）；",
                "        只有空间安慰剂能当检验用，另外两条是**诊断**。",
            ]
        )


def run_scm_placebo_audit(
    config: SCMConfig | None = None,
    *,
    n_trials: int = 150,
    alpha: float = 0.05,
    effect: float = 3.0,
    seed: int = 0,
) -> SCMPlaceboAudit:
    """跑三条检查：空间安慰剂（检验）、时间安慰剂（诊断）、留一法（稳健性）。"""
    cfg = config or SCMConfig(n_units=25, n_pre=16, n_post=8, noise_sd=0.4)

    space_hits_null = space_hits_alt = 0
    time_not_clean_h0 = time_not_clean_h1 = 0
    identical = 0
    loo_stable = 0
    loo_shifts: list[float] = []

    for i in range(n_trials):
        data_null = generate_scm_scenario(cfg, effect=0.0, seed=seed + i)
        data_alt = generate_scm_scenario(cfg, effect=effect, seed=seed + i)

        space_hits_null += int(placebo_inference(data_null).p_value < alpha)
        space_hits_alt += int(placebo_inference(data_alt).p_value < alpha)

        tp_null = time_placebo(data_null)
        tp_alt = time_placebo(data_alt)
        time_not_clean_h0 += int(not tp_null.clean)
        time_not_clean_h1 += int(not tp_alt.clean)
        # 不变量：时间安慰剂只用处置前窗口 ⇒ 同一个种子下两个效应档必须逐位相同
        identical += int(
            tp_null.placebo_att == tp_alt.placebo_att
            and tp_null.n_pre_used == tp_alt.n_pre_used
        )

        loo = leave_one_out(data_alt)
        loo_stable += int(loo.sign_is_stable)
        loo_shifts.append(loo.max_shift_share)

    return SCMPlaceboAudit(
        n_trials=n_trials,
        alpha=alpha,
        effect=effect,
        space_fpr=space_hits_null / n_trials,
        space_power=space_hits_alt / n_trials,
        time_not_clean_h0=time_not_clean_h0 / n_trials,
        time_not_clean_h1=time_not_clean_h1 / n_trials,
        time_effect_blind_share=identical / n_trials,
        loo_sign_stable_h1=loo_stable / n_trials,
        loo_median_shift_h1=float(np.median(loo_shifts)),
    )


def run_scm_audit(
    config: SCMConfig | None = None,
    *,
    n_trials: int = 200,
    alpha: float = 0.05,
    effect: float = 3.0,
    seed: int = 0,
) -> SCMAudit:
    """在无效应与有效应两种场景下各跑一批，量安慰剂 p 值的表现。"""
    cfg = config or SCMConfig(n_units=25, n_pre=15, n_post=8, noise_sd=0.4)

    def run(eff: float) -> tuple[float, np.ndarray, float]:
        hits = 0
        ratios = []
        biases = []
        for i in range(n_trials):
            data = generate_scm_scenario(cfg, effect=eff, seed=seed + i)
            res = placebo_inference(data)
            hits += int(res.p_value < alpha)
            ratios.append(res.treated_ratio)
            biases.append(res.treated_att - eff)
        return hits / n_trials, np.asarray(ratios), float(np.mean(biases))

    fpr, ratios_null, _ = run(0.0)
    power, _, bias = run(effect)

    return SCMAudit(
        n_trials=n_trials,
        alpha=alpha,
        false_positive_rate=fpr,
        power=power,
        rmse_ratio_quantiles=np.percentile(ratios_null, [50, 90, 95]),
        att_bias=bias,
    )


# --------------------------------------------------------------------------- #
# 四、敏感性分析
# --------------------------------------------------------------------------- #
@dataclass
class SensitivityAudit:
    """翻转点分析：违背多大时结论翻盘。"""

    n_trials: int
    median_breakdown: float
    median_breakdown_in_sd: float
    actual_violation: float
    robust_share: float
    median_robustness_ratio: float
    finite_ratio_share: float

    def summary(self) -> str:
        return (
            f"平行趋势敏感性（{self.n_trials} 次仿真，数据里真的注入了违背）\n"
            f"  实际注入的违背 = {self.actual_violation:+.4f}/期\n"
            f"  翻转点中位数 delta* = {self.median_breakdown:+.4f}/期"
            f"（{self.median_breakdown_in_sd:+.3f} 个结果变量 SD/期）\n"
            f"  结论真的扛住的（delta* > 实际违背）比例 = {self.robust_share:.1%}\n"
            f"  —— 这才是敏感性分析该回答的问题：**要翻盘得违背到什么程度**\n"
            f"\n"
            f"  附：与「处置前趋势」的比值中位数 = {self.median_robustness_ratio:.2f}"
            f"（{self.finite_ratio_share:.0%} 的样本该比值有限）\n"
            f"  这个比值在处置前趋势平坦时会**爆炸**，本身就不可用 ——"
            " 处置前看不到趋势，不代表处置后不会分岔。"
        )


@dataclass
class RambachanRothAudit:
    """三档限制 × 三种 DGP：结论的稳健性各不一样。

    这一节要说的**不是**"哪个限制更好"，而是"**换个限制，结论就换个说法**" ——
    所以必须把三档并排放在同一批数据上，并标出它们在哪个格子上给出不同裁决。
    """

    n_trials: int
    #: ``(regime, restriction) -> {M: 结论存活的份额}``
    survives: dict[tuple[str, str], dict[float, float]]
    #: 各档翻转点的中位数
    median_breakdown: dict[tuple[str, str], float]

    @property
    def restrictions_disagree(self) -> bool:
        """存在某个 (DGP, M)：三档限制给出的裁决**差得很远**（份额相差 ≥0.5）。

        第一版写成"某个份额落在 (0.05, 0.95) 之间" —— 那测的是"有没有不确定性"，
        而这一节要主张的是"**换个限制就换个说法**"：实测每个格子都是 0.00 或 1.00
        （裁决非常干脆），分歧恰恰体现在"同一格里三档不一样"。
        判据要对着要主张的东西写。
        """
        regimes = {regime for regime, _ in self.survives}
        ms = {m for table in self.survives.values() for m in table}
        for regime in regimes:
            for m in ms:
                shares = [
                    self.survives[(regime, r)][m]
                    for r in ("linear", "relative_magnitude", "smoothness")
                    if (regime, r) in self.survives
                ]
                if max(shares) - min(shares) >= 0.5:
                    return True
        return False

    def _row(self, regime: str, restriction: str, m: float) -> float:
        return self.survives[(regime, restriction)][m]

    @property
    def clean_regime_is_robust(self) -> bool:
        """平行趋势成立时，相对幅度那一档在小 M 下应当撑住。"""
        return self._row("平行趋势成立", "relative_magnitude", 2.0) > 0.8

    @property
    def violated_pretrend_is_fragile(self) -> bool:
        """处置前趋势被违反时，相对幅度那一档**很脆**（分数尺子已经被污染）。"""
        return (
            self.median_breakdown[("处置前趋势被违反", "relative_magnitude")] < 1.0
        )

    @property
    def smoothness_flags_the_post_only_blind_spot(self) -> bool:
        """只有处置后分岔时：线性/相对幅度还在撑，平滑已经开始报警。

        这正是平行趋势检验的**盲区**（事前检验对它零功效），
        而平滑限制是这三档里唯一能碰到它的 —— 因为它约束的是"拐弯"，
        而"处置后突然分岔"就是一个拐弯。
        """
        return (
            self._row("只有处置后分岔", "smoothness", 1.0) < 0.5
            and self._row("只有处置后分岔", "relative_magnitude", 1.0) > 0.8
        )

    def summary(self) -> str:
        ms = sorted({m for table in self.survives.values() for m in table})
        lines = [
            f"Rambachan-Roth 三档敏感性（各 {self.n_trials} 次）",
            f"  {'DGP':<22}{'限制':<20}" + "".join(f"{('M=' + str(m)):>9}" for m in ms),
        ]
        for regime in dict.fromkeys(r for r, _ in self.survives):
            for restriction in ("linear", "relative_magnitude", "smoothness"):
                row = self.survives.get((regime, restriction))
                if row is None:
                    continue
                cells = "".join(f"{row[m]:>9.2f}" for m in ms)
                lines.append(f"  {regime:<22}{restriction:<20}{cells}")
        lines.append("  每格 = 结论符号仍然站得住的份额（1.00 = 全部撑住）")
        lines.append("  读法：**换个限制，结论就换个说法** —— 所以报告必须写清用的是哪一档。")
        return "\n".join(lines)


def run_rambachan_roth_audit(
    *,
    n_trials: int = 30,
    ms: tuple[float, ...] = (0.5, 1.0, 2.0),
    seed: int = 0,
) -> RambachanRothAudit:
    """三种 DGP × 三档限制，量"结论还站得住"的份额。"""
    # 三种 DGP 各自覆盖 ``StaggeredPanelConfig`` 的**不同**字段，所以这里的
    # 值只能是"展开成关键字参数的一包东西"；用 ``Any`` 而不是 ``float`` ——
    # 后者会让 mypy 认为它想喂给**每一个**字段（实测报 3 个 arg-type）。
    regimes: dict[str, dict[str, Any]] = {
        "平行趋势成立": {},
        "处置前趋势被违反": {"trend_violation": 0.6},
        "只有处置后分岔": {"post_divergence": 0.8},
    }
    survives: dict[tuple[str, str], dict[float, float]] = {}
    medians: dict[tuple[str, str], float] = {}

    for regime, kwargs in regimes.items():
        counts = {
            (regime, r): {m: 0 for m in ms}
            for r in ("linear", "relative_magnitude", "smoothness")
        }
        breakdowns: dict[str, list[float]] = {
            r: [] for r in ("linear", "relative_magnitude", "smoothness")
        }
        for i in range(n_trials):
            panel, _truth = generate_staggered_panel(
                StaggeredPanelConfig(**kwargs), seed=seed + i
            )
            res = callaway_santanna(panel)
            rr = rambachan_roth_smoothness(res, panel)
            for r in breakdowns:
                breakdowns[r].append(rr.breakdown(r))
                for m in ms:
                    counts[(regime, r)][m] += int(rr.survives(r, m))
        for r in breakdowns:
            survives[(regime, r)] = {
                m: counts[(regime, r)][m] / n_trials for m in ms
            }
            medians[(regime, r)] = float(np.median(breakdowns[r]))

    return RambachanRothAudit(
        n_trials=n_trials, survives=survives, median_breakdown=medians
    )


def run_sensitivity_audit(
    config: StaggeredPanelConfig | None = None,
    *,
    n_trials: int = 150,
    post_divergence: float = 0.6,
    seed: int = 0,
) -> SensitivityAudit:
    """在**存在真实违背**的数据上跑敏感性分析，看它能不能提示出来。

    这里刻意注入 ``post_divergence``（处置后才分岔）——
    平行趋势检验看不见它，但敏感性分析应该把它反映成较低的翻转点。
    """
    base = config or StaggeredPanelConfig(
        n_units=600,
        n_periods=9,
        cohorts=(3, 5, 7),
        cohort_weights=(0.25, 0.25, 0.25),
        never_treated_share=0.25,
        effects=(1.5, 2.0, 2.5, 3.0),
        noise_sd=0.6,
    )

    breakdowns, ratios, robust = [], [], []
    panel = None
    for i in range(n_trials):
        cfg = StaggeredPanelConfig(
            **{**base.__dict__, "post_divergence": post_divergence, "seed": seed + i}
        )
        panel, _ = generate_staggered_panel(cfg)
        res = callaway_santanna(panel)
        sens = trend_sensitivity(res, panel)
        breakdowns.append(sens.breakdown_delta)
        if np.isfinite(sens.robustness_ratio):
            ratios.append(sens.robustness_ratio)
        robust.append(int(abs(sens.breakdown_delta) > abs(post_divergence)))

    assert panel is not None
    scale = float(panel.outcome[panel.treated_units].std(ddof=1))
    return SensitivityAudit(
        n_trials=n_trials,
        median_breakdown=float(np.median(breakdowns)),
        median_breakdown_in_sd=float(np.median(breakdowns)) / scale if scale else float("nan"),
        actual_violation=post_divergence,
        robust_share=float(np.mean(robust)),
        median_robustness_ratio=float(np.median(ratios)) if ratios else float("inf"),
        finite_ratio_share=len(ratios) / n_trials,
    )
