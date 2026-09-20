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

__all__ = [
    "RESTRICTIONS",
    "RambachanRothSensitivity",
    "TrendSensitivity",
    "rambachan_roth_smoothness",
    "trend_sensitivity",
]


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


# --------------------------------------------------------------------------- #
# Rambachan-Roth 的另外两种限制：相对幅度与平滑
# --------------------------------------------------------------------------- #
#: 三种限制的名字。**它们的单位不一样**，所以数字不可直接比大小 ——
#: 能比的是"同一个 M 下结论还在不在"。
RESTRICTIONS: tuple[str, ...] = ("linear", "relative_magnitude", "smoothness")


@dataclass(frozen=True)
class RambachanRothSensitivity:
    """平行趋势的三档敏感性：线性违背、相对幅度、二阶差分（平滑）。

    为什么不能只有一档
    ------------------
    原来只有"线性违背"（``TrendSensitivity``）：假设处置后的偏离沿一条直线增长。
    这个假设**很强**，而且它与"处置前趋势"的刻度绑在一起 ——
    处置前越平坦，它给出的翻转点看起来越稳健，恰好把最危险的情形说成最安全。

    Rambachan & Roth (2023) 的贡献是把"允许多大的违背"写成**可解释的约束**：

    * ``relative_magnitude``：处置后每期的偏离 ≤ ``M ×`` **处置前最大的那期偏离**。
      它把"处置前趋势"当作刻度尺 —— 处置前越干净，这把尺子越严（这是对的）；
    * ``smoothness``：偏离序列的**二阶差分** ≤ ``M``。
      它允许"线性趋势继续走"，只禁止突然拐弯。第 ``h`` 期的偏离上界是
      ``M·(h+1)(h+2)/2``（对最后两期处置前取值做线性外推之后）——
      所以它随期数**平方增长**，远期的结论本来就该更脆。

    翻转点（breakdown）的含义是：**M 大到多少，ATT 的识别集才会包含 0**。

    说清近似：这里给的是**识别集**（把标准误当已知），不是 RR 原文那种
    同时处理抽样不确定性的"诚实置信集"（那要解线性规划）。所以这些数是
    "结论对违背的敏感度"，不是"置信区间"。**本仓库没有做线性规划版本** ——
    这一点写在 README 的已知边界里，不假装做了。
    """

    att: float
    #: 各档限制下的翻转点（单位各不相同，见类文档）
    breakdown_linear: float
    breakdown_relative_magnitude: float
    breakdown_smoothness: float
    #: 处置前最大的单期偏离（相对幅度那一档的刻度尺）
    max_pre_violation: float
    #: 未归一化的权重和（``Σ w_k (k+1)(k+2)/2`` 之类），供复核
    post_weight_sum: float
    n_post_coefs: int
    n_pre_coefs: int
    pre_coefs: dict[int, float]
    post_coefs: dict[int, float]
    weights: dict[int, float]

    def breakdown(self, restriction: str) -> float:
        if restriction == "linear":
            return self.breakdown_linear
        if restriction == "relative_magnitude":
            return self.breakdown_relative_magnitude
        if restriction == "smoothness":
            return self.breakdown_smoothness
        raise ValueError(f"未知限制：{restriction!r}（可选 {RESTRICTIONS}）")

    def identified_set(self, restriction: str, m: float) -> tuple[float, float]:
        """在强度 ``M`` 下 ATT 的识别集（把标准误当已知）。"""
        if m < 0:
            raise ValueError("M 不能为负")
        if restriction == "linear":
            half = m * self.post_weight_sum
        elif restriction == "relative_magnitude":
            half = m * self.max_pre_violation
        elif restriction == "smoothness":
            # Σ_k w_k·(k+1)(k+2)/2 ÷ Σ_k w_k：加权平均的"平方增长"系数
            total_w = sum(self.weights.get(k, 0.0) for k in self.post_coefs) or 1.0
            grow = (
                sum(
                    self.weights.get(k, 0.0) * (k + 1) * (k + 2) / 2.0
                    for k in self.post_coefs
                )
                / total_w
            )
            half = m * grow
        else:
            raise ValueError(f"未知限制：{restriction!r}（可选 {RESTRICTIONS}）")
        return (self.att - half, self.att + half)

    def survives(self, restriction: str, m: float) -> bool:
        """在这个限制与强度下，结论的**符号**还站得住吗（0 不在识别集里）。"""
        lo, hi = self.identified_set(restriction, m)
        return bool(lo > 0 or hi < 0)

    def summary(self) -> str:
        lines = [
            f"Rambachan-Roth 三档敏感性（{self.n_post_coefs} 个处置后、"
            f"{self.n_pre_coefs} 个处置前系数）",
            f"  ATT = {self.att:+.4f}",
            f"  ① 线性违背：每期多偏离 {self.breakdown_linear:+.4f} 就归零"
            "（假设最强）",
            f"  ② 相对幅度：处置后偏离达到处置前最大偏离"
            f"（{self.max_pre_violation:.4f}）的 {self.breakdown_relative_magnitude:.2f} 倍才归零",
            f"  ③ 平滑（二阶差分）：每期二阶差分到 {self.breakdown_smoothness:.3f} 才归零"
            "（随期数平方增长，远期更脆）",
            "  注意：三档的**单位不同**，数字不能直接比大小；能比的是"
            "「同一个 M 下结论还在不在」。",
        ]
        if not np.isfinite(self.breakdown_relative_magnitude):
            lines.append(
                "  处置前没有任何可测偏离 ⇒ 相对幅度那一档退化成点识别："
                "**一点点处置后偏离就能推翻结论**（这不是稳健，是尺子为零）。"
            )
        return "\n".join(lines)


def rambachan_roth_smoothness(
    res: CSResult, panel: Panel | None = None
) -> RambachanRothSensitivity:
    """由 CS 的事件研究算三档翻转点。"""
    es = res.event_study
    weights = {k: float(e.n_treatment) for k, e in es.items()}
    coefs = {k: float(e.absolute_effect) for k, e in es.items()}
    post = {k: v for k, v in coefs.items() if k >= 0}
    pre = {k: v for k, v in coefs.items() if k < 0}
    if not post:
        raise ValueError("没有任何处置后系数，无法做敏感性分析")

    att = float(res.overall.absolute_effect)
    # ① 线性：加权平均后每期多偏离多少 ⇒ 与 TrendSensitivity 同一口径
    linear = _weighted_delta(post, weights)

    # ② 相对幅度：刻度尺 = 处置前最大单期偏离
    max_pre = max((abs(v) for v in pre.values()), default=0.0)
    rm = abs(att) / max_pre if max_pre > 0 else float("inf")

    # ③ 平滑：第 k 期的偏离上界 M·(k+1)(k+2)/2（对最后两期处置前取值线性外推之后）
    total_w = sum(weights.get(k, 0.0) for k in post) or float(len(post))
    grow = (
        sum(weights.get(k, 0.0) * (k + 1) * (k + 2) / 2.0 for k in post) / total_w
    )
    smooth = abs(att) / grow if grow > 0 else float("inf")

    # linear 那一档的识别集半宽 = M·Σ w_k (k+1)/Σ w_k
    lin_grow = (
        sum(weights.get(k, 0.0) * (k + 1) for k in post) / total_w
        if total_w
        else float("nan")
    )

    return RambachanRothSensitivity(
        att=att,
        breakdown_linear=float(linear),
        breakdown_relative_magnitude=float(rm),
        breakdown_smoothness=float(smooth),
        max_pre_violation=float(max_pre),
        post_weight_sum=float(lin_grow),
        n_post_coefs=len(post),
        n_pre_coefs=len(pre),
        pre_coefs={int(k): float(v) for k, v in pre.items()},
        post_coefs={int(k): float(v) for k, v in post.items()},
        weights={int(k): float(v) for k, v in weights.items()},
    )
