"""t 检验推断的**唯一实现**。

M0 时 `welch_ttest`（明细入口）和仿真台各自算过一遍，靠测试保证一致。
M1 加了 CUPED、比值指标、聚类稳健之后会有六个入口共用同一套推断，
再各写一遍迟早会漂移 —— 所以抽成这里一份。

三层结构，上层都用下层：

``t_inference``            给定 (效应, 标准误, 自由度) —— 最底层
``welch_inference``        给定两臂的逐单元方差与样本量，算 Welch 自由度
``welch_inference_from_components``
                           给定两臂**估计量自身**的方差与自由度
                           （比值指标的 delta method、聚类稳健标准误走这条）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "WelchInference",
    "t_inference",
    "welch_inference",
    "welch_inference_from_components",
]


@dataclass(frozen=True)
class WelchInference:
    """推断结果。置信区间由调用方按 ``effect ± ci_half_width`` 得到。"""

    se: float
    degrees_of_freedom: float
    p_value: float
    ci_half_width: float
    degenerate: bool

    def interval(self, effect: float) -> tuple[float, float]:
        return (effect - self.ci_half_width, effect + self.ci_half_width)


def t_inference(
    effect: float,
    *,
    se: float,
    degrees_of_freedom: float,
    alpha: float = 0.05,
) -> WelchInference:
    """最底层：给定效应、标准误、自由度，得到 p 值与置信区间。"""
    if se < 0 or not np.isfinite(se):
        raise ValueError(f"标准误必须是有限非负数，收到 {se}")
    if degrees_of_freedom <= 0 or not np.isfinite(degrees_of_freedom):
        raise ValueError(f"自由度必须为正，收到 {degrees_of_freedom}")

    if se == 0.0:
        # 没有波动：p 值无定义，退化为看效应是否恰好为 0
        return WelchInference(
            se=0.0,
            degrees_of_freedom=float(degrees_of_freedom),
            p_value=1.0 if effect == 0 else 0.0,
            ci_half_width=0.0,
            degenerate=True,
        )

    t_stat = effect / se
    p_value = float(2 * stats.t.sf(abs(t_stat), degrees_of_freedom))
    crit = float(stats.t.ppf(1 - alpha / 2, degrees_of_freedom))

    return WelchInference(
        se=float(se),
        degrees_of_freedom=float(degrees_of_freedom),
        p_value=p_value,
        ci_half_width=crit * se,
        degenerate=False,
    )


def welch_inference_from_components(
    effect: float,
    *,
    var_treatment: float,
    df_treatment: float,
    var_control: float,
    df_control: float,
    alpha: float = 0.05,
) -> WelchInference:
    """两臂各自给出**估计量方差**与其自由度，算 Welch 自由度后做推断。

    用在方差不是简单 ``s²/n`` 的场合：比值指标（delta method）与聚类稳健标准误。
    """
    if var_treatment < 0 or var_control < 0:
        raise ValueError(f"方差不能为负：{var_treatment} / {var_control}")

    se = float(np.sqrt(var_treatment + var_control))
    if se == 0.0:
        return t_inference(
            effect,
            se=0.0,
            degrees_of_freedom=max(df_treatment + df_control, 1.0),
            alpha=alpha,
        )

    if df_treatment <= 0 or df_control <= 0:
        raise ValueError(f"自由度必须为正：{df_treatment} / {df_control}")

    denom = var_treatment**2 / df_treatment + var_control**2 / df_control
    df = (var_treatment + var_control) ** 2 / denom if denom > 0 else df_treatment + df_control

    return t_inference(effect, se=se, degrees_of_freedom=float(df), alpha=alpha)


def welch_inference(
    effect: float,
    *,
    n_treatment: int,
    var_treatment: float,
    n_control: int,
    var_control: float,
    alpha: float = 0.05,
) -> WelchInference:
    """标准两样本 Welch：给定两臂的**逐单元样本方差**与样本量。

    用 Welch 而不是合并方差版本：线上指标在实验组与对照组的方差经常不同
    （尤其当处理会改变用户行为的波动性时），合并方差会低估标准误、抬高假阳性。
    """
    if n_treatment < 2 or n_control < 2:
        raise ValueError(f"每组至少需要 2 个样本，收到 {n_treatment} / {n_control}")
    if var_treatment < 0 or var_control < 0:
        raise ValueError(f"方差不能为负：{var_treatment} / {var_control}")

    return welch_inference_from_components(
        effect,
        var_treatment=var_treatment / n_treatment,
        df_treatment=float(n_treatment - 1),
        var_control=var_control / n_control,
        df_control=float(n_control - 1),
        alpha=alpha,
    )
