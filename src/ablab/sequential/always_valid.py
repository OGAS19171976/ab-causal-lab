"""mSPRT：任何时刻都有效的 p 值（always-valid p-value）。

为什么还需要这个
----------------
群序贯检验要求**事先**声明看几次、在什么信息量上看。实际看的次数与声明不符，
保证就作废。而现实中"我今天点进去看一眼"是常态。

always-valid p 值不需要这个前提：

    P_{H0}(存在某个 n 使得 p_n <= alpha) <= alpha

对**任意**停止规则成立 —— 包括"看到显著就停"这种由数据决定的规则。
代价是它比固定时点检验保守（同样样本量下功效略低）。

做法（Johari, Pekelis & Walsh 2015, *Always Valid Inference*）
-------------------------------------------------------------
对效应量设正态混合先验 ``delta ~ N(0, tau^2)``，构造混合似然比

    Lambda_n = phi(delta_hat_n; 0, V_n + tau^2) / phi(delta_hat_n; 0, V_n)

其中 ``V_n = SE_n^2``。由 Ville 不等式，``p_n = min(1, 1/Lambda_n)`` 是 always-valid 的。

展开成可直接计算的形式：

    Lambda_n = sqrt(V_n / (V_n + tau^2))
               * exp( delta_hat_n^2 * tau^2 / (2 * V_n * (V_n + tau^2)) )

``tau`` 怎么选
--------------
``tau`` 是"你预期效应有多大"的先验尺度：越大则对大效应越敏感、越早能停，
越小越保守。实践上取你关心的最小可检测效应（MDE）量级。

**本模块不提供魔法默认值**，``tau`` 必须显式给出。
随便挑一个 tau 也能得到"有效"的 p 值（有效性不依赖 tau 的取值），
但功效会因此天差地别 —— 这种参数不该藏起来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "AlwaysValidResult",
    "msprt_statistic",
    "msprt_p_value",
    "always_valid_path",
    "choose_tau",
    "optimal_tau",
    "rejection_threshold",
]


def msprt_statistic(
    estimate: float | np.ndarray,
    std_error: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """混合似然比 ``Lambda_n``（越大越倾向有效应）。

    ``tau`` 既可以是标量（固定先验，**这是唯一有 always-valid 保证的用法**），
    也可以是逐时点的数组 —— 后者只为"让数据选先验"那个反例服务
    （见 ``validation/tau_rule_audit.py`` 的实测：它会把 I 类错误抬上去）。
    """
    est = np.asarray(estimate, dtype=float)
    se = np.asarray(std_error, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    if np.any(tau_arr <= 0):
        raise ValueError(f"tau 必须为正，收到 {tau!r}")
    if np.any(se <= 0):
        raise ValueError("标准误必须为正")

    V = se**2
    tau2 = tau_arr**2
    with np.errstate(over="ignore"):
        lam = np.sqrt(V / (V + tau2)) * np.exp(est**2 * tau2 / (2.0 * V * (V + tau2)))
    return np.minimum(lam, np.inf)


def msprt_p_value(
    estimate: float | np.ndarray,
    std_error: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """always-valid p 值 ``p_n = min(1, 1/Lambda_n)``。"""
    lam = msprt_statistic(estimate, std_error, tau)
    return np.minimum(1.0, 1.0 / lam)


def rejection_threshold(std_error: float, tau: float, alpha: float = 0.05) -> float:
    """在给定标准误下，``|delta_hat|`` 要多大才能让 ``p <= alpha``。

    反解 ``Lambda = 1/alpha``：

        delta_hat^2 = 2 V (V + tau^2) / tau^2 * [ ln(1/alpha) + 0.5 ln((V + tau^2)/V) ]

    这个阈值随 ``V`` 缩小而缩小（样本越多越容易拒绝），
    但**不会**像固定时点检验那样按 ``1/sqrt(n)`` 下降 —— 保守就体现在这里。
    """
    V = std_error**2
    tau2 = tau**2
    inner = np.log(1.0 / alpha) + 0.5 * np.log((V + tau2) / V)
    return float(np.sqrt(2.0 * V * (V + tau2) / tau2 * inner))


@dataclass(frozen=True)
class AlwaysValidResult:
    """一条 always-valid 监控路径的结果。"""

    standard_errors: np.ndarray
    estimates: np.ndarray
    p_values: np.ndarray
    alpha: float
    tau: float

    @property
    def n_looks(self) -> int:
        return int(self.p_values.size)

    @property
    def min_p(self) -> float:
        return float(np.min(self.p_values))

    @property
    def ever_rejected(self) -> bool:
        return bool(self.min_p <= self.alpha)

    @property
    def first_rejection_look(self) -> int | None:
        hits = np.flatnonzero(self.p_values <= self.alpha)
        return int(hits[0]) + 1 if hits.size else None

    @property
    def thresholds(self) -> np.ndarray:
        """每个时点上触发拒绝所需的 ``|delta_hat|``。"""
        return np.array(
            [rejection_threshold(se, self.tau, self.alpha) for se in self.standard_errors]
        )

    def summary(self) -> str:
        look = self.first_rejection_look
        return (
            f"always-valid 监控（mSPRT, tau={self.tau:.4f}, alpha={self.alpha}）\n"
            f"  查看 {self.n_looks} 次，最小 p = {self.min_p:.6g}\n"
            f"  是否曾拒绝 = {self.ever_rejected}"
            + (f"（第 {look} 次）" if look else "")
            + f"\n  末次阈值 |delta| >= {self.thresholds[-1]:.4f}，"
            f"末次标准误 {self.standard_errors[-1]:.4f}"
        )


def always_valid_path(
    estimates: Sequence[float] | np.ndarray,
    standard_errors: Sequence[float] | np.ndarray,
    *,
    tau: float,
    alpha: float = 0.05,
) -> AlwaysValidResult:
    """把一条随时间变化的 ``(估计, 标准误)`` 序列变成 always-valid 监控路径。"""
    est = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    if est.size != se.size:
        raise ValueError("estimates 与 standard_errors 长度必须一致")
    if est.size == 0:
        raise ValueError("序列不能为空")

    return AlwaysValidResult(
        standard_errors=se,
        estimates=est,
        p_values=msprt_p_value(est, se, tau),
        alpha=float(alpha),
        tau=float(tau),
    )


# --------------------------------------------------------------------------- #
# tau 怎么选：一条能算出来的规则，而不是"取 2 倍标准误"
# --------------------------------------------------------------------------- #
def optimal_tau(std_error: float, alpha: float = 0.05) -> float:
    """**使拒绝阈值最小**的 ``tau``（固定 ``std_error`` 与 ``alpha``）。

    为什么这是"最优"：在给定时点，功效 = ``P(|theta_hat| > 阈值(tau))``，
    而 ``theta_hat ~ N(delta, V)`` —— 所以**对任何 delta，阈值越小功效越高**。
    于是"选 tau"这件事可以化成一个一维最小化，不需要先猜 delta：

        f(tau) = (V + tau^2)/tau^2 * [ ln(1/alpha) + 0.5*ln((V + tau^2)/V) ]

    令 ``u = tau^2/V``，``c = ln(1/alpha)``：``f = (1+u)/u*(c + 0.5*ln(1+u))``。
    它在 ``u`` 上只有一个极小点（alpha=0.05 时 u* ≈ 8~9，即
    ``tau* ≈ 3*SE``）—— 这也解释了线上那句 "tau 取 2 倍标准误" 为什么**不算错**：
    它恰好落在最优点附近，只是没人量过它离最优点有多远。

    返回的是**设计期**的量：``std_error`` 应当是**末次查看的预期标准误**
    （不是当前观测到的 SE —— 那是数据依赖的选择，见 ``choose_tau`` 的警告）。
    """
    if std_error <= 0:
        raise ValueError("std_error 必须为正")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须在 (0, 1)")

    def objective(log_tau: float) -> float:
        return rejection_threshold(std_error, float(np.exp(log_tau)), alpha)

    # 在 log 尺度上做黄金分割：tau 的合理范围跨几个量级，线性搜索会漏掉
    lo, hi = np.log(std_error * 0.1), np.log(std_error * 100.0)
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    for _ in range(200):
        if objective(c) < objective(d):
            b, d = d, c
            c = b - inv_phi * (b - a)
        else:
            a, c = c, d
            d = a + inv_phi * (b - a)
        if b - a < 1e-12:
            break
    return float(np.exp((a + b) / 2.0))


def choose_tau(
    *,
    std_error: float,
    target_effect: float | None = None,
    alpha: float = 0.05,
    rule: str = "threshold",
) -> float:
    """选 ``tau``。两条规则，**都必须在看数据之前定下来**。

    * ``rule="threshold"``（默认）：使拒绝阈值最小 —— 见 ``optimal_tau``。
      它不需要事先知道效应量，是"我不知道效应多大"时的稳妥选择；
    * ``rule="match"``：``tau = |target_effect|``（先验与你想检出的效应匹配）。
      知道目标效应时它更贴题，代价是效应猜小了会显著掉功效（有实测）。

    **``tau`` 必须是设计期的常数**。always-valid 的保证来自
    "先验在事前固定"：如果每次查看都拿**当前**观测到的 SE 去重算 tau，
    这种数据依赖会让 I 类错误膨胀（本仓库有一条实测钉着这件事，
    见 ``reports/m2_validation.md``）。所以这里的 ``std_error`` 参数
    指的是**末次查看的预期标准误**，而不是"现在这批数据的 SE"。
    """
    if rule == "threshold":
        return optimal_tau(std_error, alpha=alpha)
    if rule == "match":
        if target_effect is None:
            raise ValueError('rule="match" 需要给出 target_effect')
        if target_effect <= 0:
            raise ValueError("target_effect 必须为正")
        return float(target_effect)
    raise ValueError(f'rule 只能是 "threshold" / "match"，收到 {rule!r}')
