"""贝叶斯决策：后验、P(delta > 0)、期望损失。

正态-正态共轭
-------------
先验 ``delta ~ N(mu0, tau0^2)``，似然 ``delta_hat | delta ~ N(delta, V)``。
后验仍是正态：

    1/tau_n^2 = 1/tau0^2 + 1/V
    mu_n      = tau_n^2 * (mu0/tau0^2 + delta_hat/V)

两个决策量
----------
``probability_better``
    ``P(delta > 0 | data) = Phi(mu_n / tau_n)``
``expected_loss``
    选错一边的**期望损失**，量纲与业务指标一致（"如果决定错了，平均要亏多少"），
    比"提升概率 95%"更容易和业务方对齐。

    ``E[max(-delta, 0)] = tau_n * phi(mu_n/tau_n) - mu_n * Phi(-mu_n/tau_n)``

必须讲清楚的一点
----------------
**贝叶斯后验在任何时刻都是自洽的，但"盯着 P(delta>0) 的阈值做决策"并不自动
控制频率派的 I 类错误。**

这两句话不矛盾：后验说的是"给定先验和数据，delta 有多大概率为正"，
这是一个合法的概率陈述，不需要为窥视做校正；
但"反复查看、一旦超过阈值就停"这个**决策规则**的长期误判率是另一回事 ——
它仍然是频率派对象，仍然会被可选停止放大。

本模块两种都提供，并在 ``validation.sequential_audit`` 里用仿真把差距量出来。
需要频率派保证就老实用群序贯或 always-valid；需要可解释的决策阈值就用期望损失。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "NormalPrior",
    "Posterior",
    "BayesianDecision",
    "posterior",
    "posterior_mean_sd",
    "probability_better",
    "expected_loss",
    "decide",
]


@dataclass(frozen=True)
class NormalPrior:
    """效应量的正态先验。"""

    sd: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        if self.sd <= 0:
            raise ValueError(f"先验标准差必须为正，收到 {self.sd}")

    def summary(self) -> str:
        return f"N(mean={self.mean:.4f}, sd={self.sd:.4f})"


@dataclass(frozen=True)
class Posterior:
    """给定数据后的效应量后验。"""

    mean: float
    sd: float
    prior: NormalPrior
    estimate: float
    std_error: float

    @property
    def precision(self) -> float:
        return 1.0 / self.sd**2

    @property
    def prior_weight(self) -> float:
        """先验在后验精度里占的比重 —— 数据越多它越接近 0。"""
        return (1.0 / self.prior.sd**2) / self.precision

    def probability_better(self, threshold: float = 0.0) -> float:
        """``P(delta > threshold | data)``。"""
        return float(stats.norm.sf((threshold - self.mean) / self.sd))

    def expected_loss(self, side: str = "treatment") -> float:
        """选 ``side`` 的期望损失（另一侧更好的程度）。

        ``side="treatment"`` 时是 ``E[max(-delta, 0)]``；
        ``side="control"`` 时是 ``E[max(delta, 0)]``。
        """
        return expected_loss(self.mean, self.sd, side=side)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        if not 0 < level < 1:
            raise ValueError("level 必须在 (0,1)")
        z = float(stats.norm.ppf(1 - (1 - level) / 2))
        return (self.mean - z * self.sd, self.mean + z * self.sd)

    def summary(self) -> str:
        lo, hi = self.credible_interval(0.95)
        return (
            f"后验: delta ~ N({self.mean:+.4f}, {self.sd:.4f}^2)\n"
            f"  数据: 估计 {self.estimate:+.4f} ± {self.std_error:.4f}\n"
            f"  先验权重 {self.prior_weight:.2%}（先验 {self.prior.summary()}）\n"
            f"  P(delta > 0) = {self.probability_better():.4f}\n"
            f"  选处理的期望损失 = {self.expected_loss('treatment'):.6f}\n"
            f"  选对照的期望损失 = {self.expected_loss('control'):.6f}\n"
            f"  95% 可信区间 [{lo:+.4f}, {hi:+.4f}]"
        )


def posterior_mean_sd(
    estimate,
    std_error,
    prior: NormalPrior,
):
    """后验均值与标准差（**向量化**：``estimate``/``std_error`` 可以是数组）。

    仿真审计一次要算几十万个后验，逐个构造 ``Posterior`` 对象太慢，
    所以把共轭更新的核心单独暴露出来。
    """
    est = np.asarray(estimate, dtype=float)
    se = np.asarray(std_error, dtype=float)
    if np.any(se <= 0):
        raise ValueError("标准误必须为正")

    data_precision = 1.0 / se**2
    prior_precision = 1.0 / prior.sd**2
    total = data_precision + prior_precision

    mean = (prior.mean * prior_precision + est * data_precision) / total
    return mean, np.sqrt(1.0 / total)


def posterior(
    estimate: float,
    std_error: float,
    prior: NormalPrior,
) -> Posterior:
    """由估计值与其标准误得到后验（标量入口，返回完整对象）。"""
    if std_error <= 0:
        raise ValueError(f"标准误必须为正，收到 {std_error}")

    mean, sd = posterior_mean_sd(estimate, std_error, prior)
    return Posterior(
        mean=float(mean),
        sd=float(sd),
        prior=prior,
        estimate=float(estimate),
        std_error=float(std_error),
    )


def probability_better(
    estimate,
    std_error,
    prior: NormalPrior,
    threshold: float = 0.0,
):
    """``P(delta > threshold | data)``，**向量化**入口。"""
    mean, sd = posterior_mean_sd(estimate, std_error, prior)
    return stats.norm.sf((threshold - mean) / sd)


def expected_loss(mu, sd, *, side: str = "treatment"):
    """正态分布下选错一边的期望损失。

    ``side="treatment"``：``E[max(-delta, 0)] = sd*phi(mu/sd) - mu*Phi(-mu/sd)``
    ``side="control"``  ：``E[max( delta, 0)] = sd*phi(mu/sd) + mu*Phi( mu/sd)``

    同样支持数组输入。
    """
    mu_arr = np.asarray(mu, dtype=float)
    sd_arr = np.asarray(sd, dtype=float)
    if np.any(sd_arr <= 0):
        raise ValueError("标准差必须为正")
    ratio = mu_arr / sd_arr
    density = stats.norm.pdf(ratio)

    if side == "treatment":
        out = sd_arr * density - mu_arr * stats.norm.cdf(-ratio)
    elif side == "control":
        out = sd_arr * density + mu_arr * stats.norm.cdf(ratio)
    else:
        raise ValueError(f"side 只能是 'treatment' 或 'control'，收到 {side!r}")

    return out if out.ndim else float(out)


@dataclass(frozen=True)
class BayesianDecision:
    """按给定判据给出的决策。"""

    posterior: Posterior
    loss_threshold: float | None
    probability_threshold: float | None

    def _probability_triggered(self) -> bool:
        """**双边**判据：任一方向的把握超过阈值就停。

        只写 ``P(delta>0) >= 阈值`` 是单边的 —— 那样"处理明显更差"时永远不会停，
        规则就不对称了。这里两个方向都判。
        """
        if self.probability_threshold is None:
            return False
        p = self.posterior.probability_better()
        return p >= self.probability_threshold or (1.0 - p) >= self.probability_threshold

    def _loss_triggered(self) -> bool:
        if self.loss_threshold is None:
            return False
        return min(
            self.posterior.expected_loss("treatment"),
            self.posterior.expected_loss("control"),
        ) <= self.loss_threshold

    @property
    def should_stop(self) -> bool:
        return self._probability_triggered() or self._loss_triggered()

    @property
    def action(self) -> str:
        """``"ship_treatment"`` / ``"keep_control"`` / ``"continue"``。"""
        if not self.should_stop:
            return "continue"
        loss_t = self.posterior.expected_loss("treatment")
        loss_c = self.posterior.expected_loss("control")
        return "ship_treatment" if loss_t < loss_c else "keep_control"

    def summary(self) -> str:
        lines = [
            f"贝叶斯决策: {self.action}",
            f"  P(delta>0) = {self.posterior.probability_better():.4f}"
            + (
                f"（阈值 {self.probability_threshold}）"
                if self.probability_threshold is not None
                else "（未设阈值）"
            ),
            f"  最小期望损失 = "
            f"{min(self.posterior.expected_loss('treatment'), self.posterior.expected_loss('control')):.6f}"
            + (
                f"（阈值 {self.loss_threshold}）"
                if self.loss_threshold is not None
                else "（未设阈值）"
            ),
        ]
        return "\n".join(lines)


def decide(
    post: Posterior,
    *,
    loss_threshold: float | None = None,
    probability_threshold: float | None = None,
) -> BayesianDecision:
    """按期望损失和/或提升概率阈值给出决策（两个判据是"或"的关系）。"""
    if loss_threshold is None and probability_threshold is None:
        raise ValueError("至少要给一个判据：loss_threshold 或 probability_threshold")
    return BayesianDecision(
        posterior=post,
        loss_threshold=loss_threshold,
        probability_threshold=probability_threshold,
    )
