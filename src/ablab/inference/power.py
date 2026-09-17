"""功效与最小可检出效应（MDE）：把"我该跑多久"变成可回答的问题。

为什么这一层值得单独存在
------------------------
平台此前只回答"实验跑完了，效应是多少"。而实验平台被问得最多的其实是
**跑之前**的问题：这个改动要多少样本才能看出来？现在这点样本能检出多大的效应？

好消息是这三个问题共用同一个解析式，而且 M0 已经在仿真里校准过它
（7 个效应量上，解析功效与经验功效最大偏差 0.037）：

    z 统计量 ~ N(λ, 1)，  λ = |效应| / SE
    功效 = P(|Z| > z_{1-α/2}) = Φ(λ - z) + Φ(-λ - z)

符号约定
--------
* ``se`` 是**效应估计的标准误**（不是单个观测的标准差）
* ``mde`` 是在给定 α 与功效下，能被检出的最小 |效应|
* 三者关系：``mde(se) = (z_{1-α/2} + z_{power}) · se``

CUPED 的收益在这里第一次变成业务语言
------------------------------------
CUPED 把 SE 降到 ``√(1-ρ²)`` 倍，于是 **MDE 也降到同样的倍数**。
ρ=0.7 时 MDE 变成原来的 0.71 倍 —— 也就是"以前要跑 2 倍的量才能看出来的效应，
现在这个量就能看出来"。这比"方差缩减 49%"更容易被业务方理解。
"""

from __future__ import annotations

import math

from scipy import stats

__all__ = [
    "z_power",
    "mde",
    "se_of_mean_diff",
    "required_n_per_arm",
]


def z_power(effect: float, se: float, alpha: float = 0.05) -> float:
    """双侧 z 检验的解析功效。

    效应为 0 时返回 α —— 这正是"零效应下还是会以 α 的概率显著"的意思，
    所以它同时是一个自洽性检查：``z_power(0, se, alpha) == alpha``。
    """
    if se <= 0:
        raise ValueError(f"se 必须为正，收到 {se}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha}")
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lam = abs(effect) / se
    return float(stats.norm.sf(z_crit - lam) + stats.norm.cdf(-z_crit - lam))


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """给定 SE，要达到 ``power`` 功效所需的最小可检出 |效应|。

    这是 ``z_power`` 的反函数：``z_power(mde(se), se) == power``（有测试守着）。
    """
    if se <= 0:
        raise ValueError(f"se 必须为正，收到 {se}")
    if not 0 < power < 1:
        raise ValueError(f"power 必须在 (0,1)，收到 {power}")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * se)


def se_of_mean_diff(sd: float, n_treatment: float, n_control: float) -> float:
    """两组均值差的标准误。"""
    if n_treatment <= 0 or n_control <= 0:
        raise ValueError("两组样本量必须为正")
    return float(sd * math.sqrt(1.0 / n_treatment + 1.0 / n_control))


def required_n_per_arm(
    sd: float,
    target_mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    treatment_ratio: float = 0.5,
) -> float:
    """要检出 ``target_mde``，需要多少总样本（返回**每臂**的不动点解）。

    ``treatment_ratio`` 是分给处理组的比例。不等权分配会降低效率：
    q=0.5 时 ``1/q+1/(1-q) = 4``；q=0.1 时是 11.1 —— 同样的总量，
    SE 要差 ``√(11.1/4) = 1.67`` 倍。这正是"为什么别做 90/10"的定量答案。

    返回的是**总样本量的每臂平均值**（总量 / 2），便于和 n_users 对齐；
    调用方拿到 ``n_total = 2 * 返回值`` 即可。
    """
    if sd <= 0:
        raise ValueError(f"sd 必须为正，收到 {sd}")
    if target_mde <= 0:
        raise ValueError(f"target_mde 必须为正，收到 {target_mde}")
    if not 0 < treatment_ratio < 1:
        raise ValueError(f"treatment_ratio 必须在 (0,1)，收到 {treatment_ratio}")
    # se=1 时的 MDE 恰好就是 z_{1-α/2} + z_{power}，直接复用它，免得两处各写一遍
    z_sum = mde(1.0, alpha, power)
    target_se = target_mde / z_sum
    # 解 sd²·(1/(q·n) + 1/((1-q)·n)) = target_se²  对 n
    factor = 1.0 / treatment_ratio + 1.0 / (1.0 - treatment_ratio)
    n_total = sd**2 * factor / target_se**2
    return float(n_total / 2.0)
