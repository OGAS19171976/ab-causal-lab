"""CUPED：用实验前指标做方差缩减，同时消掉协变量失衡造成的偏置。

    Y_adj = Y - θ·(X - X̄),   θ = Cov(X, Y) / Var(X)

X 是**实验前**指标（与分流独立），Y 是实验后指标。

为什么这一步同时解决两个问题
----------------------------
记 β 为 Y 对 X 的回归系数。固定住某一次分流后：

    post_gap = β·pre_gap + η        （η 零均值，见 M0 的 conditional 模式）

* **消偏置**：CUPED 减去 θ·pre_gap，θ̂ → β 时偏置项被整项扣掉，
  剩下的是真正零均值的 η。M0 里那个"负对照实验跑出 p=0.004"的假象就来自 β·pre_gap。
* **降方差**：Var(η) = (1-ρ²)·Var(post_gap)。实测 ρ≈0.795 时
  **方差缩减 = ρ² ≈ 63%**，**残余方差 = 1-ρ² ≈ 37%**，
  等价于**有效样本量放大 1/(1-ρ²) ≈ 2.7 倍**。

三个量必须分清楚（本项目用三个独立命名隔开，并有测试守着不被写反）：

======================  ==================  ==========
量                       公式                ρ=0.795
======================  ==================  ==========
方差缩减（去掉的）        ρ²                  0.632
残余方差（剩下的）        1-ρ²                0.368
标准误降幅               1-√(1-ρ²)           0.394
有效样本量倍数            1/(1-ρ²)            2.72
======================  ==================  ==========

θ 用哪一臂估计
--------------
``pooled``（默认）
    两臂合并估计。方差最小、最常用。若处理会改变 (X,Y) 的联合分布，
    θ̂ 会是两臂的加权平均，略有污染。
``control``
    只用对照组。更"干净"（完全不看处理组），代价是效率略低。
    在担心处理影响协方差结构时用它。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from .aggregates import AggregateStats
from .result import Diagnostic, Estimate
from .srm import srm_check
from .welch import welch_inference

__all__ = ["CupedFit", "ThetaSource", "fit_cuped", "cuped_estimate", "cuped_ttest"]

ThetaSource = Literal["pooled", "control"]

#: |rho| 低于这个值时 CUPED 收益有限，给一条 warn
_WEAK_CORRELATION = 0.10

#: 前置协变量组间差超过这个显著性水平就提示失衡（比常规 0.05 严，
#: 因为每个实验都会跑这个检查，且我们只想抓"系统性问题"）
_IMBALANCE_ALPHA = 1e-3


@dataclass(frozen=True)
class CupedFit:
    """CUPED 的 θ 估计及其质量指标。"""

    theta: float
    theta_source: str
    n_used: int
    covariate_mean: float
    covariate_var: float
    covariance: float
    correlation: float

    # -- 四个必须分清的收益指标 -------------------------------------------- #
    @property
    def variance_reduction(self) -> float:
        """去掉的方差比例 = rho^2。"""
        return self.correlation**2

    @property
    def remaining_variance(self) -> float:
        """校正后剩下的方差比例 = 1 - rho^2。"""
        return 1.0 - self.correlation**2

    @property
    def se_shrinkage(self) -> float:
        """标准误的降幅 = 1 - sqrt(1 - rho^2)。**不等于**方差缩减。"""
        return 1.0 - math.sqrt(self.remaining_variance)

    @property
    def effective_sample_multiplier(self) -> float:
        """等效样本量倍数 = 1 / (1 - rho^2)。"""
        return 1.0 / self.remaining_variance

    def summary(self) -> str:
        return (
            f"CUPED 拟合 (theta 来源={self.theta_source}, n={self.n_used:,})\n"
            f"  rho(pre, post) = {self.correlation:.4f}\n"
            f"  theta = {self.theta:.4f}\n"
            f"  方差缩减   = rho^2     = {self.variance_reduction:.4f}\n"
            f"  残余方差   = 1 - rho^2 = {self.remaining_variance:.4f}\n"
            f"  标准误降幅 = 1-sqrt(...) = {self.se_shrinkage:.4f}\n"
            f"  等效样本量 x{self.effective_sample_multiplier:.2f}\n"
        )


def fit_cuped(
    pooled: AggregateStats,
    control: AggregateStats | None = None,
    *,
    theta_source: ThetaSource = "pooled",
) -> CupedFit:
    """估计 CUPED 的 θ。

    Parameters
    ----------
    pooled:
        两臂**合并**后的统计量（用于算 ρ 与默认的 θ）。
    control:
        对照组统计量，``theta_source="control"`` 时必需。
    """
    if theta_source not in ("pooled", "control"):
        raise ValueError(f"未知 theta_source: {theta_source!r}")
    if theta_source == "control":
        if control is None:
            raise ValueError("theta_source='control' 需要传入 control 统计量")
        source = control
    else:
        source = pooled

    var_x = source.var_x
    if not np.isfinite(var_x) or var_x <= 0:
        raise ValueError(
            "前置协变量没有方差，CUPED 无法估计 theta"
            "（检查前置指标是否全为常数，或窗口是否取在了曝光之后）"
        )

    return CupedFit(
        theta=float(source.cov_xy / var_x),
        theta_source=theta_source,
        n_used=source.n,
        covariate_mean=float(pooled.mean_x),
        covariate_var=float(var_x),
        covariance=float(source.cov_xy),
        correlation=float(pooled.corr_xy),
    )


def _balance_diagnostic(treatment: AggregateStats, control: AggregateStats) -> Diagnostic:
    """前置协变量的组间平衡性检验。

    这是实验可信度的体检项：前置指标在**分流前**测量，两臂不应有系统性差异。
    （单次实现的随机不平衡是正常的，频繁出现才说明分流或埋点有问题。）
    """
    var_t = treatment.var_x / treatment.n
    var_c = control.var_x / control.n
    se = math.sqrt(var_t + var_c)
    gap = treatment.mean_x - control.mean_x

    if se == 0 or not np.isfinite(se):
        return Diagnostic(
            name="协变量平衡",
            status="info",
            message="前置协变量无波动，平衡性检验不可用",
        )

    z = gap / se
    p = float(2 * stats.norm.sf(abs(z)))
    status = "warn" if p < _IMBALANCE_ALPHA else "pass"
    msg = (
        f"前置指标组间差 {gap:+.4f}（{z:+.2f} 个标准误），p={p:.4g}"
        + ("；**失衡**——CUPED 收益最大的情形，但也提示分流链路值得复查" if status == "warn" else "")
    )
    return Diagnostic(
        name="协变量平衡",
        status=status,
        message=msg,
        statistic=float(z),
        p_value=p,
    )


def cuped_estimate(
    treatment: AggregateStats,
    control: AggregateStats,
    *,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
    theta_source: ThetaSource = "pooled",
    expected_weights: dict[str, float] | None = None,
    srm_alpha: float = 1e-3,
) -> tuple[Estimate, CupedFit]:
    """从汇总统计量做 CUPED 校正 + Welch 检验，返回 (估计, 拟合)。

    ``treatment``/``control`` 里的 ``x`` 是**实验前**指标，``y`` 是实验后指标。
    """
    pooled = treatment.merge(control)
    fit = fit_cuped(pooled, control, theta_source=theta_source)
    theta = fit.theta

    # 中心化常数对组间差没有影响（两臂各减同一个 pooled.mean_x），
    # 只影响"校正后均值"的绝对水平，因此这里统一用合并均值。
    center = pooled.mean_x
    adj_mean_t = treatment.mean_y - theta * (treatment.mean_x - center)
    adj_mean_c = control.mean_y - theta * (control.mean_x - center)
    effect = adj_mean_t - adj_mean_c

    # 校正后每一臂内的方差：Var(y - theta*x) = Var(y) - 2*theta*Cov + theta^2*Var(x)
    var_adj_t = treatment.var_y - 2 * theta * treatment.cov_xy + theta**2 * treatment.var_x
    var_adj_c = control.var_y - 2 * theta * control.cov_xy + theta**2 * control.var_x
    # 数值保护：理论上非负，浮点误差可能给出极小负数
    var_adj_t = max(float(var_adj_t), 0.0)
    var_adj_c = max(float(var_adj_c), 0.0)

    inference = welch_inference(
        effect,
        n_treatment=treatment.n,
        var_treatment=var_adj_t,
        n_control=control.n,
        var_control=var_adj_c,
        alpha=alpha,
    )
    ci_low, ci_high = inference.interval(effect)

    diagnostics: list[Diagnostic] = [_balance_diagnostic(treatment, control)]

    if abs(fit.correlation) < _WEAK_CORRELATION:
        diagnostics.append(
            Diagnostic(
                name="CUPED 收益",
                status="warn",
                message=(
                    f"|rho|={abs(fit.correlation):.4f} 太低，方差缩减仅 "
                    f"{fit.variance_reduction:.2%}；换一个与结果更相关的前置指标，"
                    "或考虑 CUPAC（用模型预测值当协变量）"
                ),
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                name="CUPED 收益",
                status="pass",
                message=(
                    f"方差缩减 {fit.variance_reduction:.2%}（残余 {fit.remaining_variance:.2%}，"
                    f"标准误降 {fit.se_shrinkage:.2%}），等效样本量 x"
                    f"{fit.effective_sample_multiplier:.2f}"
                ),
                statistic=float(fit.correlation),
            )
        )

    if expected_weights is not None:
        diagnostics.append(
            srm_check(
                {control_name: control.n, variant: treatment.n},
                expected_weights,
                alpha=srm_alpha,
            )
        )

    relative = effect / adj_mean_c if adj_mean_c != 0 else float("nan")

    return (
        Estimate(
            metric=metric,
            variant=variant,
            control=control_name,
            method="CUPED + Welch t-test",
            absolute_effect=float(effect),
            relative_effect=float(relative),
            std_error=inference.se,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            p_value=inference.p_value,
            n_treatment=int(treatment.n),
            n_control=int(control.n),
            mean_treatment=float(adj_mean_t),
            mean_control=float(adj_mean_c),
            alpha=alpha,
            diagnostics=tuple(diagnostics),
        ),
        fit,
    )


def cuped_ttest(
    treatment_pre,
    treatment_post,
    control_pre,
    control_post,
    **kwargs,
) -> tuple[Estimate, CupedFit]:
    """逐单元明细入口：``(前, 后)`` 两臂各一对数组。"""
    return cuped_estimate(
        AggregateStats.from_arrays(treatment_post, treatment_pre),
        AggregateStats.from_arrays(control_post, control_pre),
        **kwargs,
    )
