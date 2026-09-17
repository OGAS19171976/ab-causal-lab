"""序贯检验：alpha 消耗、群序贯边界、always-valid p 值、贝叶斯决策。

M0 用仿真标定了一个常数边界来对抗窥视；M2 把它换成有解析保证的做法。

``spending``       alpha 消耗函数（OBF / Pocock / Kim-DeMets）
``boundaries``     Armitage-McPherson 递归解边界 + 重复置信区间 + 调整 p 值
``always_valid``   mSPRT：任何时刻都有效的 p 值
``bayesian``       正态-正态后验、P(δ>0)、期望损失决策
"""

from .always_valid import (
    AlwaysValidResult,
    always_valid_path,
    msprt_p_value,
    msprt_statistic,
)
from .bayesian import (
    BayesianDecision,
    NormalPrior,
    Posterior,
    decide,
    expected_loss,
    posterior,
    posterior_mean_sd,
    probability_better,
)
from .boundaries import (
    BoundarySolver,
    SequentialDesign,
    adjusted_p_value,
    build_design,
    repeated_ci,
)
from .spending import (
    SPENDING_FUNCTIONS,
    SpendingFunction,
    get_spending,
    kim_demets,
    linear,
    obrien_fleming,
    pocock,
)

__all__ = [
    "SPENDING_FUNCTIONS",
    "SpendingFunction",
    "get_spending",
    "kim_demets",
    "linear",
    "obrien_fleming",
    "pocock",
    "BoundarySolver",
    "SequentialDesign",
    "adjusted_p_value",
    "build_design",
    "repeated_ci",
    "AlwaysValidResult",
    "always_valid_path",
    "msprt_p_value",
    "msprt_statistic",
    "BayesianDecision",
    "NormalPrior",
    "Posterior",
    "decide",
    "expected_loss",
    "posterior",
    "posterior_mean_sd",
    "probability_better",
]
