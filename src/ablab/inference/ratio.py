"""比值指标的 delta method（线性化）。

指标形如 CTR = 总点击 / 总曝光、人均订单额 = 总金额 / 总订单数。
随机化单元是**用户**，但指标是**比值**，分子分母都是用户级聚合。

naive 做法错在哪
----------------
"对每个用户算 r_i = y_i/x_i，再对 r_i 做 t 检验"有两个独立的问题：

1. **口径变了**
   ``mean(y_i / x_i) != Σy / Σx``。前者是"人均比值"，
   把只曝光 1 次的长尾用户和曝光 1000 次的头部用户赋予**相同权重**；
   后者才是业务口径的 CTR。两者在效应方向和幅度上可以不同，
   所以 naive 检验很可能在回答一个没人问的问题。

2. **方差算错（更隐蔽）**
   r_i 的方差主要由曝光量的倒数决定（曝光 1 次的用户 r_i ∈ {0,1}），
   分布重尾，t 检验的正态近似失效。

正确做法
--------
    R = Ȳ / X̄
    Var(R) ≈ Var(y_i - R·x_i) / (n · X̄²)

分子必须展开成：

    Var(y_i - R·x_i) = Var(y) - 2R·Cov(x, y) + R²·Var(x)

**协方差项是 naive 做法丢掉的那一项** —— 分母大（曝光多）的用户往往分子也大，
忽略这一点会把方差估得离谱。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aggregates import AggregateStats
from .result import Diagnostic, Estimate, Status
from .welch import welch_inference_from_components

__all__ = ["ratio_delta_method", "naive_unit_ratio_ttest", "NaiveRatioResult"]


def _ratio_diagnostic(treatment: AggregateStats, control: AggregateStats) -> Diagnostic:
    pooled = treatment.merge(control)
    corr = pooled.corr_xy
    if not np.isfinite(corr):
        return Diagnostic(
            name="比值指标",
            status="info",
            message="分子或分母无波动，无法评估 delta method 的适用性",
        )
    status: Status = "pass"
    note = ""
    if abs(corr) < 0.05:
        # 分子分母几乎不相关时 delta method 仍然正确，但说明分母的调整作用有限
        note = "；分子与分母几乎不相关，注意确认指标口径是否合理"
        status = "warn"
    return Diagnostic(
        name="比值指标",
        status=status,
        message=(
            f"对照组分母均值 X̄={control.mean_x:,.4f}，"
            f"合并比值 R={pooled.ratio:.6f}，corr(分子,分母)={corr:.4f}{note}"
        ),
        statistic=float(corr),
    )


def ratio_delta_method(
    treatment: AggregateStats,
    control: AggregateStats,
    *,
    metric: str = "ratio_metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
) -> Estimate:
    """比值指标的 delta method 检验。

    ``AggregateStats`` 里 **x 是分母**（曝光 / 会话 / 订单数），
    **y 是分子**（点击 / 时长 / 金额）。
    """
    if treatment.mean_x <= 0 or control.mean_x <= 0:
        raise ValueError("分母均值必须为正，检查是否把分子分母传反了")
    if treatment.n < 2 or control.n < 2:
        raise ValueError(f"每组至少需要 2 个样本，收到 {treatment.n} / {control.n}")

    r_t = treatment.ratio
    r_c = control.ratio
    effect = r_t - r_c

    var_t = treatment.ratio_variance()
    var_c = control.ratio_variance()
    if not np.isfinite(var_t) or not np.isfinite(var_c):
        raise ValueError("比值方差无法估计，检查分母方差是否为 0")

    inference = welch_inference_from_components(
        effect,
        var_treatment=float(var_t),
        df_treatment=float(treatment.n - 1),
        var_control=float(var_c),
        df_control=float(control.n - 1),
        alpha=alpha,
    )
    ci_low, ci_high = inference.interval(effect)

    return Estimate(
        metric=metric,
        variant=variant,
        control=control_name,
        method="ratio delta method",
        absolute_effect=float(effect),
        relative_effect=float(effect / r_c) if r_c else float("nan"),
        std_error=inference.se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=int(treatment.n),
        n_control=int(control.n),
        mean_treatment=float(r_t),
        mean_control=float(r_c),
        alpha=alpha,
        diagnostics=(_ratio_diagnostic(treatment, control),),
    )


@dataclass(frozen=True)
class NaiveRatioResult:
    """反例做法的结果，附带它**实际**估计的口径，便于和正确口径对比。"""

    estimate: Estimate
    treatment_mean_unit_ratio: float
    control_mean_unit_ratio: float
    n_units_dropped: int

    def gap_vs(self, other: Estimate, *, arm: str = "control") -> float:
        """naive 口径与另一个口径（通常是 delta method）的差距。

        注意**不能**拿 ``control_mean_unit_ratio - estimate.mean_control`` 来算 ——
        那两个是同一个量（都来自对 r_i 取平均），相减恒等于 0。
        真正要比的是"人均比值"与"合并比值 Σy/Σx"这两个不同口径。
        """
        reference = other.mean_control if arm == "control" else other.mean_treatment
        observed = (
            self.control_mean_unit_ratio if arm == "control" else self.treatment_mean_unit_ratio
        )
        return observed - reference

    def summary(self) -> str:
        return (
            f"[反例] 用户级比值 t 检验\n"
            f"  它估计的人均比值: {self.treatment_mean_unit_ratio:.6f} / "
            f"{self.control_mean_unit_ratio:.6f}\n"
            f"  效应 {self.estimate.absolute_effect:+.6f}  "
            f"SE {self.estimate.std_error:.6f}  p={self.estimate.p_value:.4g}\n"
            f"  因分母为 0 被丢弃的单元: {self.n_units_dropped}\n"
        )


def naive_unit_ratio_ttest(
    treatment_numerator,
    treatment_denominator,
    control_numerator,
    control_denominator,
    *,
    metric: str = "ratio_metric",
    alpha: float = 0.05,
) -> NaiveRatioResult:
    """**反例**：对每个用户的比值 r_i = y_i/x_i 直接做 Welch t 检验。

    保留这个函数只用于仿真演示"naive 做法到底错在哪"，**不要在生产里用**。
    它估计的是人均比值 ``mean(y_i/x_i)``，而不是业务口径的 ``Σy/Σx``。
    """
    from .tests import welch_ttest as _welch

    dropped = [0]  # 两个臂共用，用列表避开 nonlocal 的样板

    def _ratios(num, den):
        num = np.asarray(num, dtype=float).ravel()
        den = np.asarray(den, dtype=float).ravel()
        if num.size != den.size:
            raise ValueError("分子分母长度不一致")
        keep = np.isfinite(num) & np.isfinite(den) & (den != 0)
        # 分母为 0 的用户比值无定义，只能丢弃 —— 这本身也是一处口径损失
        dropped[0] += int((~keep).sum())
        return num[keep] / den[keep]

    r_t = _ratios(treatment_numerator, treatment_denominator)
    r_c = _ratios(control_numerator, control_denominator)

    est = _welch(r_t, r_c, metric=metric, alpha=alpha)
    return NaiveRatioResult(
        estimate=est,
        treatment_mean_unit_ratio=float(r_t.mean()),
        control_mean_unit_ratio=float(r_c.mean()),
        n_units_dropped=int(dropped[0]),
    )
