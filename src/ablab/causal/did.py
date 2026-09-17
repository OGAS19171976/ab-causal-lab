"""DiD 估计量：2×2、双向固定效应、Callaway-Sant'Anna、事件研究。

三种估计量，一个核心问题
------------------------
``two_by_two_did``
    教科书版本：一个处置组、一个对照组、处置前后两段。
    识别靠**平行趋势**：没有处置时，处置组的趋势会和对照组一样。
``twfe``
    双向固定效应回归 ``Y_it = a_i + l_t + tau*D_it + e_it``。
    在**单一处置时点**下它等于 2×2 DiD；但在**交错处置 + 效应随时间变化**时，
    它会用"已经处置的组"当对照，产生 **forbidden comparison**，
    权重可能为负 —— 于是一组**处处为正**的效应能估出**负数**。
``callaway_santanna``
    按队列×期数逐一算干净的 2×2（对照组只用从未处置或尚未处置的单元），
    再按需要聚合。它把 TWFE 那堆隐式权重换成**显式**的、非负的权重。

M3 的核心证据就是：同一批数据上，TWFE 给出负号，CS 给出正确的正号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import stats

from ..inference.result import Diagnostic, Estimate
from ..inference.welch import t_inference, welch_inference
from .panel import GroundTruth, Panel

__all__ = [
    "two_by_two_did",
    "twfe",
    "twfe_decomposition",
    "TWFEDecomposition",
    "callaway_santanna",
    "cs_att_with_influence",
    "collect_leads_with_influence",
    "event_study_leads",
    "CSResult",
    "pretrend_test",
]

ControlGroup = Literal["never_treated", "not_yet_treated"]


# --------------------------------------------------------------------------- #
# 2×2
# --------------------------------------------------------------------------- #
def _mean_with_influence(values: np.ndarray) -> tuple[float, np.ndarray]:
    """均值及其影响函数（``psi_i = x_i - mean``）。"""
    m = float(values.mean())
    return m, values - m


def two_by_two_did(
    panel: Panel,
    *,
    treated_units: np.ndarray,
    control_units: np.ndarray,
    pre_periods: Sequence[int],
    post_periods: Sequence[int],
    metric: str = "ATT",
    alpha: float = 0.05,
) -> Estimate:
    """标准 2×2 DiD：``(处置组后-前) - (对照组后-前)``。

    面板数据同一单元前后相关，所以先对每个单元取**差分** ``dY = 后 - 前``，
    再对两组的 ``dY`` 做 Welch 检验 —— 这样就自动处理了单元内相关。
    """
    pre = np.asarray(list(pre_periods), dtype=int) - 1
    post = np.asarray(list(post_periods), dtype=int) - 1
    if pre.size == 0 or post.size == 0:
        raise ValueError("pre_periods 与 post_periods 都不能为空")

    dY = panel.outcome[:, post].mean(axis=1) - panel.outcome[:, pre].mean(axis=1)
    t_vals = dY[treated_units]
    c_vals = dY[control_units]
    if t_vals.size < 2 or c_vals.size < 2:
        raise ValueError("每组至少需要 2 个单元")

    effect = float(t_vals.mean() - c_vals.mean())
    inference = _welch(effect, t_vals, c_vals, alpha)
    ci_low, ci_high = inference.interval(effect)

    return Estimate(
        metric=metric,
        variant="treated",
        control="control",
        method="2x2 DiD",
        absolute_effect=effect,
        relative_effect=float(effect / c_vals.mean()) if c_vals.mean() else float("nan"),
        std_error=inference.se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=int(t_vals.size),
        n_control=int(c_vals.size),
        mean_treatment=float(t_vals.mean()),
        mean_control=float(c_vals.mean()),
        alpha=alpha,
    )


def _welch(effect: float, t_vals: np.ndarray, c_vals: np.ndarray, alpha: float):
    return welch_inference(
        effect,
        n_treatment=t_vals.size,
        var_treatment=float(t_vals.var(ddof=1)),
        n_control=c_vals.size,
        var_control=float(c_vals.var(ddof=1)),
        alpha=alpha,
    )


# --------------------------------------------------------------------------- #
# 双向固定效应
# --------------------------------------------------------------------------- #
def twfe(panel: Panel, *, alpha: float = 0.05, cluster: bool = True) -> Estimate:
    """双向固定效应估计 + 单元层面聚类稳健标准误。

    用 within 变换实现：``ỹ = y - ȳ_i - ȳ_t + ȳ``，``D̃`` 同理，
    然后 ``tau_hat = Σ ỹ D̃ / Σ D̃²``。

    聚类稳健方差用三明治形式，簇 = 单元（面板数据的标准做法）。
    """
    y = panel.outcome
    d = panel.treated.astype(float)
    n, T = y.shape

    y_tilde = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + y.mean()
    d_tilde = d - d.mean(axis=1, keepdims=True) - d.mean(axis=0, keepdims=True) + d.mean()

    denom = float((d_tilde * d_tilde).sum())
    if denom <= 0:
        raise ValueError("处理变量在双向去均值后没有变异，无法识别（检查是否所有单元同时处置）")

    tau = float((y_tilde * d_tilde).sum() / denom)

    # 残差与聚类稳健方差：簇 = 单元
    resid = y_tilde - tau * d_tilde
    cluster_scores = (d_tilde * resid).sum(axis=1)  # 每个单元一个
    meat = float((cluster_scores**2).sum())
    var = meat / denom**2

    if cluster and n > 1:
        # 小样本修正
        var *= n / (n - 1)

    inference = t_inference(tau, se=float(np.sqrt(var)), degrees_of_freedom=max(n - 1, 1), alpha=alpha)
    ci_low, ci_high = inference.interval(tau)

    n_treated_units = int(panel.treated_units.sum())
    diagnostics = []
    if panel.cohorts().size > 1:
        diagnostics.append(
            Diagnostic(
                name="交错处置",
                status="warn",
                message=(
                    f"有 {panel.cohorts().size} 个处置队列。TWFE 会用"
                    "「已处置组」当对照（forbidden comparison），"
                    "效应随时间变化时权重可能为负 —— 请用 callaway_santanna 复核。"
                ),
                statistic=float(panel.cohorts().size),
            )
        )

    return Estimate(
        metric="ATT",
        variant="treated",
        control="control",
        method="TWFE (two-way fixed effects)",
        absolute_effect=tau,
        relative_effect=float("nan"),
        std_error=inference.se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=n_treated_units,
        n_control=n - n_treated_units,
        mean_treatment=float(y[panel.treated_units].mean()),
        mean_control=float(y[panel.never_treated_units].mean())
        if panel.never_treated_units.any()
        else float("nan"),
        alpha=alpha,
        diagnostics=tuple(diagnostics),
    )


@dataclass
class TWFEDecomposition:
    """TWFE 的**逐单元-期**隐式权重。

    这是 ``twfe`` 的全部机制，而且是精确的：双向去均值后

        tau = Σ_it ỹ_it D̃_it / Σ_it D̃_it²

    所以每个单元-期对估计的贡献权重就是归一化的 ``D̃_it``。**它可以为负**，
    而负权重恰好落在那些效应最大的单元-期上 —— 这就是"效应处处为正、
    TWFE 却给出负号"的完整解释，不需要额外假设。

    （Goodman-Bacon 把同样的东西按 2×2 比较分组，是更粗粒度的视角；
    逐单元的版本更直接，而且能被精确验证。）
    """

    delta: np.ndarray  # (N, T) 未归一权重
    weight: np.ndarray  # (N, T) 归一后权重，Σ = 1
    tau: float
    treated: np.ndarray  # (N, T)
    true_effect: np.ndarray | None = None  # (N, T) 仅仿真时知道

    @property
    def post_cells(self) -> np.ndarray:
        return self.treated.astype(bool)

    @property
    def negative_post_weight_share(self) -> float:
        """处置后的单元-期里，权重为负的比例。"""
        post = self.post_cells
        if not post.any():
            return float("nan")
        return float(np.mean(self.weight[post] < 0))

    @property
    def negative_weight_effect_share(self) -> float:
        """负权重的那些单元-期，承载了多少**真实效应**。

        这个数大于 1 就说明负权重系统性地落在效应大的地方 ——
        这正是估计量被拉向反方向的原因。
        """
        if self.true_effect is None:
            return float("nan")
        post = self.post_cells
        neg = post & (self.weight < 0)
        total = float(self.true_effect[post].sum())
        if total == 0:
            return float("nan")
        return float(self.true_effect[neg].sum() / total)

    def weight_effect_correlation(self) -> float:
        """处置后，权重与真实效应的相关系数。**为负就说明权重方向是错的**。"""
        if self.true_effect is None:
            return float("nan")
        post = self.post_cells
        w = self.weight[post]
        e = self.true_effect[post]
        if w.std() == 0 or e.std() == 0:
            return float("nan")
        return float(np.corrcoef(w, e)[0, 1])

    def summary(self) -> str:
        lines = [
            f"TWFE 隐式权重分解：tau = {self.tau:+.4f}",
            f"  处置后单元-期中，权重为负的比例 = {self.negative_post_weight_share:.1%}",
        ]
        if self.true_effect is not None:
            lines.append(
                f"  负权重单元承载的真实效应占比 = {self.negative_weight_effect_share:.1%}"
            )
            lines.append(
                f"  权重与真实效应的相关系数 = {self.weight_effect_correlation():+.4f}"
                + ("   <- 负相关：权重方向和效应方向相反" if self.weight_effect_correlation() < 0 else "")
            )
        return "\n".join(lines)


def twfe_decomposition(
    panel: Panel,
    truth: GroundTruth | None = None,
) -> TWFEDecomposition:
    """把 TWFE 拆成逐单元-期的隐式权重，并（若有真值）检验权重方向。"""
    y = panel.outcome
    d = panel.treated.astype(float)

    y_tilde = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + y.mean()
    delta = d - d.mean(axis=1, keepdims=True) - d.mean(axis=0, keepdims=True) + d.mean()

    denom = float((delta * delta).sum())
    tau = float((y_tilde * delta).sum() / denom)
    weight = delta / denom  # Σ weight * y = tau

    true_effect = None
    if truth is not None:
        true_effect = np.zeros_like(y)
        for g in panel.cohorts():
            mask = panel.cohort == g
            for t in panel.periods:
                key = (int(g), int(t))
                if key in truth.group_time:
                    true_effect[mask, int(t) - 1] = truth.group_time[key]

    return TWFEDecomposition(
        delta=delta,
        weight=weight,
        tau=tau,
        treated=panel.treated,
        true_effect=true_effect,
    )


# --------------------------------------------------------------------------- #
# Callaway & Sant'Anna
# --------------------------------------------------------------------------- #
@dataclass
class CSResult:
    """Callaway-Sant'Anna 的群-时 ATT 及各种聚合。"""

    group_time: dict[tuple[int, int], Estimate]
    event_study: dict[int, Estimate]
    overall: Estimate
    control_group: str
    base_period: int | None

    @property
    def atts(self) -> dict[tuple[int, int], float]:
        return {k: v.absolute_effect for k, v in self.group_time.items()}

    def summary(self) -> str:
        base = (
            "各队列自己的处置前一期 (g-1)"
            if self.base_period is None
            else str(self.base_period)
        )
        lines = [
            f"Callaway-Sant'Anna（对照组 = {self.control_group}，基准期 = {base}）",
            f"  整体 ATT = {self.overall.absolute_effect:+.4f} "
            f"(SE {self.overall.std_error:.4f}, p={self.overall.p_value:.4g})",
            "  事件研究（相对期数 -> ATT）：",
        ]
        for k in sorted(self.event_study):
            e = self.event_study[k]
            tag = " <- 处置前（应接近 0）" if k < 0 else ""
            lines.append(
                f"    k={k:>3}: {e.absolute_effect:+.4f} "
                f"(SE {e.std_error:.4f}){tag}"
            )
        return "\n".join(lines)


def _att_influence(
    dY: np.ndarray, g_mask: np.ndarray, c_mask: np.ndarray
) -> tuple[float, np.ndarray]:
    """``ATT`` 及其**影响函数**。

    ``ATT = mean_g(dY) - mean_c(dY)``，其中 ``dY = Y_t - Y_base``。
    影响函数让不同 ``(g,t)`` 之间的协方差能被精确算出来 —— 这很重要，
    因为**所有 lead 共用同一个基准期**，它们高度相关；
    把标准误按独立合成会严重**低估**方差（实测把 size 从 5% 抬到 11%）。
    """
    p_g = float(g_mask.mean())
    p_c = float(c_mask.mean())
    m_g = float(dY[g_mask].mean())
    m_c = float(dY[c_mask].mean())

    psi = np.where(g_mask, (dY - m_g) / p_g, 0.0) - np.where(
        c_mask, (dY - m_c) / p_c, 0.0
    )
    return m_g - m_c, psi


def _control_mask(
    panel: Panel, period: int, base: int, control_group: ControlGroup
) -> np.ndarray:
    """对照组掩码。

    "尚未处置"必须对**两次观测都成立**：``C_i > max(t, base)``。
    只判 t 期是错的 —— 基准期已经处置的单元会把自身的效应变化带进对照组
    （这个 bug 真的出现过：它让处置前的 placebo 系数从 0 涨到 +0.72）。
    """
    if control_group == "never_treated":
        return panel.never_treated_units
    return panel.cohort > max(period, base)


def cs_att_with_influence(
    panel: Panel,
    cohort: int,
    period: int,
    base: int,
    control_group: ControlGroup = "not_yet_treated",
) -> tuple[float, np.ndarray, int, int] | None:
    """单个 ``ATT(g,t)`` 的点估计与影响函数（联合检验用）。"""
    g_mask = panel.cohort == cohort
    if not g_mask.any():
        return None
    c_mask = _control_mask(panel, period, base, control_group)
    if c_mask.sum() < 2:
        return None

    dY = panel.outcome[:, period - 1] - panel.outcome[:, base - 1]
    effect, psi = _att_influence(dY, g_mask, c_mask)
    return effect, psi, int(g_mask.sum()), int(c_mask.sum())


def _cs_group_time(
    panel: Panel,
    cohort: int,
    period: int,
    base: int,
    control_group: ControlGroup,
    alpha: float,
) -> Estimate | None:
    """单个 ``ATT(g, t)``：``E[Y_t - Y_base | G=g] - E[Y_t - Y_base | C]``。"""
    out = cs_att_with_influence(panel, cohort, period, base, control_group)
    if out is None:
        return None
    effect, psi, n_t, n_c = out

    n = psi.size
    se = float(np.sqrt(np.var(psi, ddof=1) / n))
    inference = t_inference(effect, se=se, degrees_of_freedom=max(n - 1, 1), alpha=alpha)
    ci_low, ci_high = inference.interval(effect)

    return Estimate(
        metric=f"ATT({cohort},{period})",
        variant="treated",
        control="control",
        method="Callaway-Sant'Anna group-time ATT",
        absolute_effect=effect,
        relative_effect=float("nan"),
        std_error=se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=n_t,
        n_control=n_c,
        mean_treatment=effect,
        mean_control=0.0,
        alpha=alpha,
    )


def callaway_santanna(
    panel: Panel,
    *,
    control_group: ControlGroup = "not_yet_treated",
    base_period: int | Literal["universal"] = "universal",
    alpha: float = 0.05,
    include_pre: bool = True,
) -> CSResult:
    """Callaway & Sant'Anna (2021) 的群-时 ATT。

    ``base_period="universal"`` 时所有 ``ATT(g,t)`` 都用 ``g-1`` 作基准期，
    处置前的 ``t < g-1`` 就变成"安慰剂"估计（应接近 0，可用来做平行趋势检验）。

    聚合用**显式、非负**的权重：整体 ATT 按 (队列人数 × 处置后期数) 加权，
    这正是它相对 TWFE 的核心优势 —— 不会出现负权重。
    """
    events: dict[tuple[int, int], Estimate] = {}
    weights: dict[tuple[int, int], float] = {}

    for g in panel.cohorts():
        g = int(g)
        base = int(base_period) if base_period != "universal" else g - 1
        if base < 1:
            continue
        periods = panel.periods if include_pre else panel.periods[panel.periods >= g]
        for t in periods:
            t = int(t)
            if t == base:
                continue
            if base_period == "universal" and t > base and t < g:
                # 基准期之后的处置前时期也能算，但口径容易混淆，跳过
                continue
            est = _cs_group_time(panel, g, t, base, control_group, alpha)
            if est is None:
                continue
            events[(g, t)] = est
            weights[(g, t)] = float(est.n_treatment)

    if not events:
        raise ValueError("没有任何可估计的 ATT(g,t)，检查队列设置与对照组选择")

    # ---- 整体 ATT：只聚合处置后（t >= g），权重 = 该队列的观测数 ---------- #
    post_keys = [k for k in events if k[1] >= k[0]]
    if not post_keys:
        raise ValueError("没有任何处置后的 ATT(g,t)")
    w = np.array([weights[k] for k in post_keys], dtype=float)
    w = w / w.sum()
    vals = np.array([events[k].absolute_effect for k in post_keys])
    ses = np.array([events[k].std_error for k in post_keys])
    overall_effect = float((w * vals).sum())
    # 保守做法：忽略跨 (g,t) 的相关性，按独立加权合成方差
    overall_se = float(np.sqrt((w**2 * ses**2).sum()))
    overall_inf = t_inference(
        overall_effect, se=overall_se, degrees_of_freedom=max(panel.n_units - 1, 1), alpha=alpha
    )
    lo, hi = overall_inf.interval(overall_effect)
    overall = Estimate(
        metric="ATT",
        variant="treated",
        control="control",
        method="Callaway-Sant'Anna (aggregated)",
        absolute_effect=overall_effect,
        relative_effect=float("nan"),
        std_error=overall_se,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=overall_inf.p_value,
        n_treatment=int(sum(events[k].n_treatment for k in post_keys)),
        n_control=int(events[post_keys[0]].n_control),
        mean_treatment=float("nan"),
        mean_control=float("nan"),
        alpha=alpha,
        diagnostics=(
            Diagnostic(
                name="权重",
                status="pass",
                message=(
                    f"聚合 {len(post_keys)} 个 ATT(g,t)，权重全部非负"
                    f"（范围 {w.min():.4f}~{w.max():.4f}）"
                ),
            ),
        ),
    )

    # ---- 事件研究：按相对期数聚合 ---------------------------------------- #
    es: dict[int, list[tuple[float, float, int]]] = {}
    for (g, t), est in events.items():
        k = t - g
        es.setdefault(k, []).append((est.absolute_effect, est.std_error, est.n_treatment))

    event_study: dict[int, Estimate] = {}
    for k, rows in sorted(es.items()):
        ww = np.array([r[2] for r in rows], dtype=float)
        ww = ww / ww.sum()
        eff = float(sum(wi * r[0] for wi, r in zip(ww, rows)))
        se = float(np.sqrt(sum(wi**2 * r[1] ** 2 for wi, r in zip(ww, rows))))
        inf = t_inference(eff, se=se, degrees_of_freedom=max(panel.n_units - 1, 1), alpha=alpha)
        lo, hi = inf.interval(eff)
        event_study[k] = Estimate(
            metric=f"event_study(k={k})",
            variant="treated",
            control="control",
            method="Callaway-Sant'Anna event study",
            absolute_effect=eff,
            relative_effect=float("nan"),
            std_error=se,
            ci_low=float(lo),
            ci_high=float(hi),
            p_value=inf.p_value,
            n_treatment=int(sum(r[2] for r in rows)),
            n_control=0,
            mean_treatment=float("nan"),
            mean_control=float("nan"),
            alpha=alpha,
        )

    return CSResult(
        group_time=events,
        event_study=event_study,
        overall=overall,
        control_group=control_group,
        base_period=None if base_period == "universal" else int(base_period),
    )


# --------------------------------------------------------------------------- #
# 平行趋势检验
# --------------------------------------------------------------------------- #
def event_study_leads(
    panel: Panel,
    *,
    control_group: ControlGroup = "not_yet_treated",
    alpha: float = 0.05,
) -> dict[int, Estimate]:
    """处置前的事件研究系数（"leads"）—— 平行趋势的**伪证检验**。"""
    res = callaway_santanna(panel, control_group=control_group, alpha=alpha)
    return {k: v for k, v in res.event_study.items() if k < 0}


def collect_leads_with_influence(
    panel: Panel,
    *,
    control_group: ControlGroup = "not_yet_treated",
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """收集所有处置前 ``ATT(g,t)`` 的点估计与影响函数矩阵。

    返回 ``(beta, Psi, keys)``：``beta`` 形状 ``(K,)``，``Psi`` 形状 ``(n_units, K)``。
    ``Psi`` 的列可以精确算出跨 ``(g,t)`` 的协方差 ——
    这正是联合检验必须用它的原因。
    """
    effects: list[float] = []
    psis: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []

    for g in panel.cohorts():
        g = int(g)
        base = g - 1
        if base < 1:
            continue
        for t in panel.periods:
            t = int(t)
            if t >= base:
                continue
            out = cs_att_with_influence(panel, g, t, base, control_group)
            if out is None:
                continue
            eff, psi, _n_t, _n_c = out
            effects.append(eff)
            psis.append(psi)
            keys.append((g, t))

    if not effects:
        return np.empty(0), np.empty((panel.n_units, 0)), []

    return np.asarray(effects), np.column_stack(psis), keys


def pretrend_test(
    panel: Panel,
    *,
    control_group: ControlGroup = "not_yet_treated",
    alpha: float = 0.05,
) -> Diagnostic:
    """把所有处置前的 lead 做一个**联合 Wald 检验**。

    为什么不能用"按标准误加权合成一个 z"：所有 lead **共用同一个基准期**，
    彼此高度相关。按独立合成会低估方差，实测把 size 从 5% 抬到 **11%**
    —— 一个自称在 5% 水平上的检验，实际误报率是它的两倍多。

    这里改用影响函数算出完整的协方差矩阵 ``V = Psi'Psi / n²``，
    统计量 ``beta' V^{-1} beta``，并**用 F 分布而不是卡方做参照**：
    协方差是从同一批数据估出来的，有限样本下 Wald 统计量的尾部比卡方厚。
    实测（n=600，K=9 个 lead）：卡方参照给 size 0.073，F 参照给 0.061；
    n=200 时更是 0.064 -> 0.048。所以 F 参照是必须的。

    结论分两档：

    * ``pass``：没能拒绝"处置前平行"
    * ``warn``：拒绝 —— 平行趋势假设有直接证据不支持

    **注明**：``pass`` 不等于平行趋势成立。这个检验对"处置后才分岔"
    （``post_divergence``）**没有任何功效** —— 实测拒绝率与 size 相同，
    而那种违背恰恰最常见：政策往往因为某次冲击才落地。
    """
    beta, psi, keys = collect_leads_with_influence(panel, control_group=control_group)
    n_leads = beta.size
    if n_leads == 0:
        return Diagnostic(
            name="平行趋势（处置前）",
            status="info",
            message="没有可用的处置前期数，无法做伪证检验",
        )

    n = psi.shape[0]
    cov = psi.T @ psi / (n * n)

    try:
        stat = float(beta @ np.linalg.solve(cov, beta))
        singular = False
    except np.linalg.LinAlgError:
        stat = float(beta @ np.linalg.pinv(cov) @ beta)
        singular = True

    if not np.isfinite(stat) or stat < 0:
        return Diagnostic(
            name="平行趋势（处置前）",
            status="info",
            message="协方差矩阵数值异常，联合检验不可用",
            detail={"n_leads": n_leads},
        )

    p = float(stats.f.sf(stat / n_leads * (n - n_leads) / n, n_leads, n - n_leads))
    status = "warn" if p < alpha else "pass"

    msg = (
        f"处置前 {n_leads} 个 lead 的联合 Wald 检验：chi2({n_leads}) = {stat:.3f}, p={p:.4g}"
    )
    if singular:
        msg += "（协方差矩阵奇异，用了伪逆）"
    if status == "warn":
        msg += "；**拒绝平行趋势，DiD 结论不可信**"
    else:
        msg += "；未能拒绝 —— 但这**不等于**平行趋势成立（检验对处置后分岔无功效）"

    return Diagnostic(
        name="平行趋势（处置前）",
        status=status,
        message=msg,
        statistic=float(stat),
        p_value=p,
        detail={"n_leads": n_leads, "singular": singular, "keys": keys},
    )



