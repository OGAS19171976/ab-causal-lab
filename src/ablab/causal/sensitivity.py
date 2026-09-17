"""平行趋势的敏感性分析：**多少违背会让结论翻盘**。

为什么需要它
------------
平行趋势不可检验。处置前看起来平行，只能说明"处置前看起来平行" ——
对"处置后才分岔"（政策往往因为某次冲击才落地，于是处置组本来就要变）**毫无功效**。

既然假设无法证实，退而求其次的问法是：

> **要推翻这个结论，平行趋势得被违背到什么程度？**

实现（线性违背的翻转点）
------------------------
假设处置组的反事实轨迹每期比对照组快 ``delta``。在事件研究尺度上，
相对期数 ``k`` 的系数就被污染了 ``delta * (k + 1)``（基准期是 ``k = -1``）。于是

    ATT(delta) = Σ w_k [beta_k - delta*(k+1)] / Σ w_k

令其为零得到**翻转点**：

    delta* = Σ w_k beta_k / Σ w_k (k+1)

同样地把这个公式套在处置前的 leads 上，得到**处置前趋势暗示的违背幅度**
``delta_pre``。两者一比就是最有用的那句话：

    |delta*| / |delta_pre| > 1  ->  结论能扛住"和处置前一样大"的违背

这是 Rambachan & Roth (2023) "honest DiD" 的**线性版本**，比他们的
平滑/相对幅度约束弱，但胜在**可解释、可直接算**，而且结论方向一致：
他们也在说"别只看点估计，报出它能承受多少违背"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .did import CSResult
from .panel import Panel

__all__ = ["TrendSensitivity", "trend_sensitivity"]


@dataclass(frozen=True)
class TrendSensitivity:
    """线性趋势违背下的翻转点分析。"""

    att: float
    breakdown_delta: float
    pretrend_delta: float
    scale: float
    n_post_coefs: int
    n_pre_coefs: int

    @property
    def robustness_ratio(self) -> float:
        """``|delta*| / |delta_pre|``。

        **这个比值在很多场景下不可用**：处置前趋势平坦时 ``delta_pre ~ 0``，
        比值会爆炸成几百甚至无穷，给出"极其稳健"的假象。
        而处置前平坦恰恰不排除处置后分岔 —— 那才是最需要警惕的情形。
        所以主判据是 ``breakdown_delta`` 本身（配合 ``breakdown_in_scale``），
        这个比值只在处置前确实有可测趋势时才参考。
        """
        if not np.isfinite(self.pretrend_delta) or self.pretrend_delta == 0:
            return float("inf")
        return abs(self.breakdown_delta) / abs(self.pretrend_delta)

    @property
    def breakdown_in_scale(self) -> float:
        """翻转点折合成"每期多少个结果变量标准差" —— 尺度无关，便于跨指标比较。"""
        return self.breakdown_delta / self.scale if self.scale else float("nan")

    def summary(self) -> str:
        ratio = self.robustness_ratio
        lines = [
            f"平行趋势敏感性（基于 {self.n_post_coefs} 个处置后、"
            f"{self.n_pre_coefs} 个处置前系数）",
            f"  ATT = {self.att:+.4f}",
            f"  **翻转点 delta* = {self.breakdown_delta:+.4f}/期**"
            f"（结果变量 SD = {self.scale:.4f}，即 {self.breakdown_in_scale:+.3f} SD/期）",
            "  含义：处置组的反事实轨迹每期比对照组多偏离 delta*，结论就归零。",
        ]
        if not np.isfinite(ratio):
            lines.append("  处置前没有可测趋势，稳健性比率无定义（见下）")
        elif ratio > 10:
            lines.append(
                f"  与处置前趋势之比 = {ratio:.1f} —— **这个数不可用**："
                "处置前趋势近乎平坦，比值被放大；"
                "而处置前平坦并不排除处置后分岔。请以 delta* 本身为准。"
            )
        else:
            lines.append(f"  与处置前趋势之比 = {ratio:.2f}（> 1 表示能扛住同量级违背）")
        return "\n".join(lines)


def _weighted_delta(coefs: dict[int, float], weights: dict[int, float]) -> float:
    """``Σ w_k beta_k / Σ w_k (k+1)``。"""
    num = 0.0
    den = 0.0
    for k, beta in coefs.items():
        w = weights.get(k, 0.0)
        num += w * beta
        den += w * (k + 1)
    if den == 0:
        return float("nan")
    return num / den


def trend_sensitivity(res: CSResult, panel: Panel | None = None) -> TrendSensitivity:
    """由 CS 的事件研究算翻转点与处置前趋势。"""
    es = res.event_study
    weights = {k: float(e.n_treatment) for k, e in es.items()}
    coefs = {k: float(e.absolute_effect) for k, e in es.items()}

    post = {k: v for k, v in coefs.items() if k >= 0}
    pre = {k: v for k, v in coefs.items() if k < 0}
    if not post:
        raise ValueError("没有任何处置后系数，无法做敏感性分析")

    breakdown = _weighted_delta(post, weights)
    pretrend = _weighted_delta(pre, weights) if pre else float("nan")

    if panel is not None:
        treated = panel.treated_units
        scale = float(panel.outcome[treated].std(ddof=1))
    else:
        scale = float("nan")

    return TrendSensitivity(
        att=float(res.overall.absolute_effect),
        breakdown_delta=float(breakdown),
        pretrend_delta=float(pretrend),
        scale=scale,
        n_post_coefs=len(post),
        n_pre_coefs=len(pre),
    )
