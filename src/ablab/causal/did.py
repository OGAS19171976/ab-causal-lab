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

from ..inference.result import Diagnostic, Estimate, Status
from ..inference.welch import t_inference, welch_inference
from .panel import GroundTruth, Panel

__all__ = [
    "two_by_two_did",
    "twfe",
    "twfe_decomposition",
    "TWFEDecomposition",
    "callaway_santanna",
    "cs_att_with_influence",
    "sun_abraham",
    "SAEventStudy",
    "SAResult",
    "collect_leads_with_influence",
    "event_study_leads",
    "CSResult",
    "pretrend_test",
]

ControlGroup = Literal["never_treated", "not_yet_treated"]

#: ``(队列, 绝对期数)`` 的键
EstimatorKey = tuple[int, int]


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
    def known_true_effect(self) -> np.ndarray:
        """``true_effect`` 的"必须存在"版本 —— 只有仿真数据才知道真值。

        真值缺失时，旧代码会在 ``true_effect[post]`` 上抛
        ``TypeError: 'NoneType' object is not subscriptable``；
        这里给一句能读懂的报错，而不是让调用处猜。
        """
        if self.true_effect is None:
            raise ValueError("这份分解没有真实效应（只有仿真数据才有），无法按真值分组")
        return self.true_effect

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
    #: 若按"忽略跨 (g,t) 相关、独立合成"算，整体 ATT 的 SE 会是多少。
    #: **一定小于等于影响函数算出来的那个**（相关性非负），所以它是反保守的；
    #: 留在这里是为了能被报告出来并对比 —— 实测低估约 46%。
    naive_overall_se: float = float("nan")

    @property
    def se_understatement(self) -> float:
        """独立合成把整体 ATT 的 SE 低估了多少（相对）。"""
        se = self.overall.std_error
        return float(1.0 - self.naive_overall_se / se) if se > 0 else float("nan")

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
    """单个 ``ATT(g,t)`` 的点估计与影响函数（联合检验用）。

    **处置队列必须从对照里剔除**（``& ~g_mask``）。在处置前的格子上
    （``t < g-1``）这一定会发生：``not_yet_treated`` 的判据是
    ``C_i > max(t, g-1)``，而处置队列自己满足 ``g > g-1``，于是它被算成了
    "尚未处置"。后果不是崩溃，而是**静默的把 placebo 压向 0** ——
    对照均值里混进了处置组自身的变化（实测 g=4、t=1 时对照 125 个单元里有
    75 个就是处置组）。这个 bug 一直藏着，因为"处置前系数接近 0"
    看起来正是我们想看到的结论 —— 这类"顺眼"的错误最难发现。
    """
    g_mask = panel.cohort == cohort
    if not g_mask.any():
        return None
    c_mask = _control_mask(panel, period, base, control_group) & ~g_mask
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
) -> tuple[Estimate, np.ndarray] | None:
    """单个 ``ATT(g, t)`` 及其**影响函数**。

    连影响函数一起返回，是因为聚合必须用它：各 ``ATT(g,t)`` 之间相关
    （共用对照单元、相邻队列还共用基准期），按独立量合成会低估方差。
    """
    out = cs_att_with_influence(panel, cohort, period, base, control_group)
    if out is None:
        return None
    effect, psi, n_t, n_c = out

    n = psi.size
    se = float(np.sqrt(np.var(psi, ddof=1) / n))
    inf = t_inference(effect, se=se, degrees_of_freedom=max(n - 1, 1), alpha=alpha)
    ci_low, ci_high = inf.interval(effect)

    estimate = Estimate(
        metric=f"ATT({cohort},{period})",
        variant="treated",
        control="control",
        method="Callaway-Sant'Anna group-time ATT",
        absolute_effect=effect,
        relative_effect=float("nan"),
        std_error=se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inf.p_value,
        n_treatment=n_t,
        n_control=n_c,
        mean_treatment=effect,
        mean_control=0.0,
        alpha=alpha,
    )
    return estimate, psi


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
    #: 每个 ATT(g,t) 的影响函数，**聚合时必须用它**（见下面的方差说明）
    psi_map: dict[tuple[int, int], np.ndarray] = {}

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
            out = _cs_group_time(panel, g, t, base, control_group, alpha)
            if out is None:
                continue
            est, psi = out
            events[(g, t)] = est
            weights[(g, t)] = float(est.n_treatment)
            psi_map[(g, t)] = psi

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

    # **方差要从影响函数算**：各个 ATT(g,t) 高度相关（共用同一批对照单元、
    # 相邻队列还共用基准期），把它们当独立量合成会**低估**标准误。
    # 这里曾经就是这么写的（sqrt(Σw²se²)），而且这个文件的 `_att_influence`
    # 文档里早就写过"独立合成会严重低估方差（实测把 size 从 5% 抬到 11%）" ——
    # 但那个教训当时只用在了 lead 的联合检验上，聚合这一步漏掉了。
    #
    # 旧算法仍然算出来（naive_overall_se）并放进诊断，好让差异可见：
    # 实测在这份面板上它把整体 ATT 的 SE 低估约 46%。
    naive_overall_se = float(np.sqrt((w**2 * ses**2).sum()))
    psi_overall = np.zeros(panel.n_units)
    for weight, key in zip(w, post_keys):
        psi_overall = psi_overall + weight * psi_map[key]
    overall_se = float(np.sqrt(np.var(psi_overall, ddof=1) / panel.n_units))
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
            Diagnostic(
                name="聚合方差",
                status="pass",
                message=(
                    f"SE 由影响函数合成（协方差自动进入）：{overall_se:.4f}；"
                    f"若按独立合成只有 {naive_overall_se:.4f}"
                    f"（低估 {1 - naive_overall_se / overall_se:.1%}）"
                    "—— 各 ATT(g,t) 共用对照单元与基准期，相关性非负。"
                ),
                statistic=overall_se,
            ),
        ),
    )

    # ---- 事件研究：按相对期数聚合 ---------------------------------------- #
    #
    # 这里的 SE **必须**用影响函数合成，理由与整体 ATT 完全相同：
    # 同一 k 上的各 ATT(g,t) 共用对照单元，不同 k 之间还共用基准期。
    # 第一版这里是 sqrt(Σ w² se²)（独立合成）—— 整体 ATT 那处修好之后
    # 这一处漏了一阵子，是靠"把仓库里所有方差合成点扫一遍"才发现的。
    # **同一个教训要应用到所有相关处，而不是只修被报告出来的那一处。**
    es: dict[int, list[tuple[EstimatorKey, int]]] = {}
    for (g, t) in events:
        es.setdefault(t - g, []).append(((g, t), int(events[(g, t)].n_treatment)))

    event_study: dict[int, Estimate] = {}
    for k, rows in sorted(es.items()):
        ww = np.array([n for _key, n in rows], dtype=float)
        ww = ww / ww.sum()
        eff = float(sum(wi * events[key].absolute_effect for wi, (key, _n) in zip(ww, rows)))
        psi = np.zeros(panel.n_units)
        for wi, (key, _n) in zip(ww, rows):
            psi = psi + wi * psi_map[key]
        se, inf = _se_from_influence(eff, psi, alpha)
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
            n_treatment=int(sum(n for _key, n in rows)),
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
        naive_overall_se=naive_overall_se,
    )


# --------------------------------------------------------------------------- #
# Sun & Abraham (2021) 交互加权估计量
# --------------------------------------------------------------------------- #
def _se_from_influence(effect: float, psi: np.ndarray, alpha: float):
    """由影响函数得到 ``(se, Inference)``。

    ``se = sqrt(Var(psi)/n)`` —— 与 CS 的单格算法一致，
    关键是**多格合成时可以直接把影响函数相加**，于是协方差自动进来。

    注意第一个参数是**点估计本身**：``t_inference`` 要用它算 t 统计量，
    第一版这里传的是 ``0.0``（只想要 se），于是整体 ATT 的 p 值恒等于 1 ——
    一个"看起来只是显示问题"的错误，其实是在报告一个**没有任何证据支持**的结论。
    """
    n = psi.size
    se = float(np.sqrt(np.var(psi, ddof=1) / n))
    return se, t_inference(effect, se=se, degrees_of_freedom=max(n - 1, 1), alpha=alpha)


@dataclass(frozen=True)
class SAEventStudy:
    """Sun-Abraham 在某个相对期数 ``k`` 上的估计。"""

    k: int
    effect: float
    #: 影响函数（长度 = 单元数）。**聚合时用它，而不是各自的标准误** —— 见 ``sun_abraham``。
    influence: np.ndarray
    #: 各队列在该 ``k`` 上的权重（队列份额，非负、和为 1）
    weights: dict[int, float]
    n_treated: int


@dataclass
class SAResult:
    """Sun & Abraham (2021) 交互加权（IW）估计量的结果。"""

    event_study: dict[int, Estimate]
    overall: Estimate
    control_group: str
    #: 若按"忽略跨期相关、独立合成"算，整体 ATT 的标准误会是多少。
    #:
    #: **一定小于等于影响函数算出来的那个**（相关性是非负的），
    #: 所以它是**反保守**的。留在这里是为了能被报告出来并和正确的那个对比 ——
    #: 一个被算出来却没人看的数字，比不写还糟。
    naive_overall_se: float
    weights: dict[int, float]

    @property
    def se_understatement(self) -> float:
        """独立合成把标准误低估了多少（相对）。"""
        se = self.overall.std_error
        return float(1.0 - self.naive_overall_se / se) if se > 0 else float("nan")

    def summary(self) -> str:
        lines = [
            f"Sun & Abraham 交互加权（对照组 = {self.control_group}）",
            f"  整体 ATT = {self.overall.absolute_effect:+.4f} "
            f"(SE {self.overall.std_error:.4f}, p={self.overall.p_value:.4g})",
            f"  同一组估计、按独立合成的 SE = {self.naive_overall_se:.4f}"
            f"（低估 {self.se_understatement:.1%}）",
            "  事件研究（相对期数 -> ATT，权重 = 该期已处置队列的份额）：",
        ]
        for k in sorted(self.event_study):
            e = self.event_study[k]
            tag = " <- 处置前（应接近 0）" if k < 0 else ""
            lines.append(f"    k={k:>3}: {e.absolute_effect:+.4f} (SE {e.std_error:.4f}){tag}")
        return "\n".join(lines)


def sun_abraham(
    panel: Panel,
    *,
    control_group: ControlGroup = "not_yet_treated",
    min_k: int | None = None,
    max_k: int | None = None,
    alpha: float = 0.05,
) -> SAResult:
    """Sun & Abraham (2021) 的**交互加权（IW）聚合**。

    **先说清楚这个实现是什么、不是什么**（这句话必须写在这里，而不是留在 README 里）：

    * 它是 **IW 聚合**：把"队列 × 相对期数"的分格效应按队列份额
      ``w_{g,k} = P(G=g | 已处置)`` 加权，权重**非负**、和为 1，所以不会出现
      TWFE 那种"负权重把估计拉反"的问题。
    * 分格效应用的是**各队列自己的 2×2**（基准期 ``g-1``，对照组按
      ``control_group`` 选）—— 与 ``callaway_santanna`` 同一套分格估计。
    * 因此，在**饱和设定**下它与 CS 的事件研究**点估计完全相同**（实测到小数点后四位）。
      Sun-Abraham 的原始形式是"一条回归 + 队列×相对期数交互项 + 双向固定效应"，
      那需要吸收 N+T 个固定效应；本仓库没有稀疏最小二乘，
      **回归版没有实现** —— 这条边界写在 README 的已知边界里。
    * 那么这一版**新增的价值**在哪：在**聚合的方差**上。见下。

    整体 ATT 是若干高度相关量的加权和（它们**共用同一个基准期 g-1**，
    还用同一批对照单元）。``callaway_santanna`` 的聚合按独立合成算，
    ``sqrt(Σ w² se²)`` —— 忽略非负协方差，**反保守**。
    这个文件里 ``_att_influence`` 的文档早就指出了这件事（"实测把 size 从 5% 抬到 11%"），
    但那个教训当时只用在了 lead 的联合检验上，聚合那一步没用。

    这里把影响函数**直接加**起来：``ψ = Σ_k w_k ψ_k``，协方差自动进来，
    ``se = sqrt(Var(ψ)/n)``。两种算法都返回，好让差异**可见**而不是靠相信 ——
    实测在这份面板上，独立合成把整体 ATT 的 SE 低估了 **45.6%**。
    """
    cells: dict[int, dict[int, tuple[float, np.ndarray, int]]] = {}
    for g in panel.cohorts():
        g = int(g)
        base = g - 1
        if base < 1:
            continue
        for t in panel.periods:
            t = int(t)
            if t == base:
                continue
            k = t - g
            if min_k is not None and k < min_k:
                continue
            if max_k is not None and k > max_k:
                continue
            out = cs_att_with_influence(panel, g, t, base, control_group)
            if out is None:
                continue
            effect, psi, n_t, _n_c = out
            cells.setdefault(k, {})[g] = (effect, psi, n_t)

    if not cells:
        raise ValueError("没有任何可估计的相对期数，检查队列设置与对照组选择")

    study: dict[int, SAEventStudy] = {}
    for k, by_cohort in sorted(cells.items()):
        total = float(sum(n for _e, _p, n in by_cohort.values()))
        if total <= 0:
            continue
        weights = {g: n / total for g, (_e, _p, n) in by_cohort.items()}
        effect = float(sum(weights[g] * by_cohort[g][0] for g in by_cohort))
        psi = np.zeros(panel.n_units)
        for g, (e, p, _n) in by_cohort.items():
            psi = psi + weights[g] * p  # 点估计与影响函数都用同一组权重
        study[k] = SAEventStudy(
            k=k, effect=effect, influence=psi, weights=weights,
            n_treated=int(total),
        )

    return _aggregate_sa(
        panel, study, control_group=control_group, alpha=alpha,
        method="Sun & Abraham (interaction-weighted, aggregated)",
    )


def _aggregate_sa(
    panel: Panel,
    study: dict[int, SAEventStudy],
    *,
    control_group: ControlGroup,
    alpha: float,
    method: str,
) -> SAResult:
    """把"逐队列 × 逐相对期数"的分格结果聚合成事件研究与整体 ATT。

    **两个版本（IW 与回归）共用这一份**：聚合规则只有一条 ——
    每个 ``k`` 上按队列份额加权、影响函数同步加权；整体 ATT 只聚合 ``k >= 0``
    且按该相对期数上的处置单元数加权。分成两份实现的话，
    "两个版本点估计应当相同"这件事就会变成"两份代码碰巧一样"。

    ``method`` 只进 ``Estimate.method`` 这个标签，不参与计算。
    """
    # ---- 事件研究（逐 k，SE 来自该 k 的影响函数） ------------------------- #
    event_study: dict[int, Estimate] = {}
    for k, sa in study.items():
        se, inf = _se_from_influence(sa.effect, sa.influence, alpha)
        lo, hi = inf.interval(sa.effect)
        event_study[k] = Estimate(
            metric=f"event_study(k={k})",
            variant="treated",
            control="control",
            method=method,
            absolute_effect=sa.effect,
            relative_effect=float("nan"),
            std_error=se,
            ci_low=float(lo),
            ci_high=float(hi),
            p_value=inf.p_value,
            n_treatment=sa.n_treated,
            n_control=0,
            mean_treatment=float("nan"),
            mean_control=float("nan"),
            alpha=alpha,
            diagnostics=(
                Diagnostic(
                    name="聚合权重",
                    status="pass",
                    message=(
                        f"k={k} 由 {len(sa.weights)} 个队列加权，权重全部非负"
                        f"（{min(sa.weights.values()):.3f}~{max(sa.weights.values()):.3f}），"
                        f"合计 {sum(sa.weights.values()):.3f}"
                    ),
                ),
            ),
        )

    # ---- 整体 ATT：只聚合处置后（k >= 0），权重 = 该相对期数上的处置单元数 -- #
    post = [k for k in study if k >= 0]
    if not post:
        raise ValueError("没有任何处置后的相对期数")
    w_raw = np.array([study[k].n_treated for k in post], dtype=float)
    k_weights = {k: float(v / w_raw.sum()) for k, v in zip(post, w_raw)}
    overall_effect = float(sum(k_weights[k] * study[k].effect for k in post))

    # 正确做法：影响函数相加（协方差自动进来）
    psi_overall = np.zeros(panel.n_units)
    for k in post:
        psi_overall = psi_overall + k_weights[k] * study[k].influence
    overall_se, inf = _se_from_influence(overall_effect, psi_overall, alpha)
    lo, hi = inf.interval(overall_effect)

    # 对照做法：忽略跨期相关，独立合成 —— 为的是把差异**显示出来**
    naive_overall_se = float(
        np.sqrt(sum(k_weights[k] ** 2 * event_study[k].std_error**2 for k in post))
    )

    overall = Estimate(
        metric="ATT",
        variant="treated",
        control="control",
        method=method,
        absolute_effect=overall_effect,
        relative_effect=float("nan"),
        std_error=overall_se,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=inf.p_value,
        n_treatment=int(sum(study[k].n_treated for k in post)),
        n_control=int(panel.never_treated_units.sum())
        if control_group == "never_treated"
        else int(np.sum(panel.cohort > max(panel.periods))),
        mean_treatment=float("nan"),
        mean_control=float("nan"),
        alpha=alpha,
        diagnostics=(
            Diagnostic(
                name="聚合方差",
                status="pass",
                message=(
                    "整体 ATT 的影响函数 = Σ_k w_k·ψ_k（协方差自动进入）；"
                    f"SE={overall_se:.4f}。若按独立合成会得到 {naive_overall_se:.4f}"
                    f"（低估 {1 - naive_overall_se / overall_se:.1%}）"
                    "—— 各相对期数共用基准期与对照，相关性非负，独立合成是反保守的。"
                ),
                statistic=overall_se,
            ),
        ),
    )

    return SAResult(
        event_study=event_study,
        overall=overall,
        control_group=control_group,
        naive_overall_se=naive_overall_se,
        weights=k_weights,
    )


def _absorb_two_way(
    values: np.ndarray,
    unit_idx: np.ndarray,
    time_idx: np.ndarray,
    n_units: int,
    n_times: int,
    *,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> np.ndarray:
    """用**交替投影**吸收双向固定效应，返回残差。

    为什么不用"加哑变量 + 最小二乘"：N+T 个哑变量会让设计矩阵变成
    ``(N·T) × (N+T+K)``，面板一大就直接爆掉（本仓库没有稀疏最小二乘，
    这也是回归版一直没做的原因）。交替投影每次迭代只要两次分组均值，
    复杂度与面板大小成线性。

    平衡面板下它收敛很快；非平衡面板同样可用（分组均值按各组的实际观测数算）。
    """
    resid = values.astype(float, copy=True)
    resid -= resid.mean()
    for _ in range(max_iter):
        cnt_u = np.bincount(unit_idx, minlength=n_units).astype(float)
        mu = np.bincount(unit_idx, weights=resid, minlength=n_units) / cnt_u
        resid -= mu[unit_idx]
        cnt_t = np.bincount(time_idx, minlength=n_times).astype(float)
        mt = np.bincount(time_idx, weights=resid, minlength=n_times) / cnt_t
        resid -= mt[time_idx]
        if np.max(np.abs(mu)) < tol and np.max(np.abs(mt)) < tol:
            break
    return resid


def sun_abraham_regression(
    panel: Panel,
    *,
    control_group: ControlGroup = "not_yet_treated",
    min_k: int | None = None,
    max_k: int | None = None,
    alpha: float = 0.05,
) -> SAResult:
    """Sun & Abraham (2021) 的**回归版**：一条回归 + 队列×相对期数交互项 + 双向固定效应。

    估计方程（省略基准期 ``k = -1``）：:

        Y_it = α_i + λ_t + Σ_g Σ_{k ≠ -1} δ_{g,k} · 1{G_i = g, t - g = k} + ε_it

    ``δ_{g,k}`` 就是"队列 g 在相对期数 k"的效应，聚合规则与 IW 版**完全相同**
    （共用 ``_aggregate_sa``）。

    **与 IW 版的关系（实测，别把结论说过头）**：

    * 用 ``control_group="never_treated"`` 时，两者**逐位相同**：点估计差
      1e-14 量级、SE 比值 1.000、聚合权重完全一致。此时"两种算法、同一个
      估计量"成立，这也是本仓库能给出的最强交叉验证（两条计算路径完全不同：
      逐 2×2 加权 vs 吸收双向固定效应后的一条回归）。
    * 用 ``control_group="not_yet_treated"`` 时**两者不同**，而且不是噪音：
      实测在 cohorts=(3,5,7)、n_periods=10 的面板上，k<=3（多队列共同贡献）
      的点估计差最大 0.17（该格 IW 的 SE 是 0.086），而 k>=4（只有单队列贡献）
      的差又回到 1e-14。原因：IW 的每个 ``(g,t)`` 分格用**当时尚未处置**的
      单元当对照（对照集随 t 变），而饱和回归只有一套双向固定效应，
      已处置队列的变化会进入比较 —— 这正是 Sun & Abraham 提醒的
      "forbidden comparison"。

    所以这一版的定位是**可核对的回归形式**：有未处置组时它给出与 IW 等同的结果，
    没有未处置组时应当用 IW 版（或者像原文那样把最后一个队列当基准，
    本仓库**没有实现**那一步，理由写在 README 的已知边界里）。

    实现要点：

    * 双向固定效用**交替投影**吸收（``_absorb_two_way``），不构造 N+T 个哑变量；
    * 设计矩阵一起被吸收（Frisch-Waugh-Lovell），系数由吸收后的回归给出；
    * 标准误按**单元聚类**：``ψ_i = (X̃'X̃)^{-1} Σ_t X̃_it·û_it``，
      与 IW 版的影响函数是同一个东西，所以聚合时可以直接相加；
    * 空分格（该队列在该相对期数没有观测）不进入设计矩阵 —— 否则会得到一个
      恒等于 0 的伪系数，还会把权重算错。
    """
    n_units, n_periods = panel.n_units, panel.n_periods
    periods = np.asarray(panel.periods)
    unit_idx = np.repeat(np.arange(n_units), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_units)
    y_long = panel.outcome.reshape(-1)

    # ---- 设计矩阵：队列 × 相对期数（去掉基准期 k=-1） --------------------- #
    cols: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    n_by_key: dict[tuple[int, int], int] = {}
    for g in panel.cohorts():
        g = int(g)
        if g - 1 < 1:  # 基准期不存在（队列在面板第一期中就被处置）
            continue
        for t in periods:
            t = int(t)
            k = t - g
            if k == -1:  # 基准期：系数被归一化为 0
                continue
            if min_k is not None and k < min_k:
                continue
            if max_k is not None and k > max_k:
                continue
            mask = (panel.cohort == g)[unit_idx] & (periods[time_idx] == t)
            n_g = int(mask.sum())
            if n_g == 0:
                continue  # 空分格：会给一个恒为 0 的伪系数
            cols.append(mask.astype(float))
            keys.append((g, k))
            n_by_key[(g, k)] = n_g

    if not cols:
        raise ValueError("没有任何可估计的分格，检查队列设置与对照组选择")

    design = np.column_stack(cols)
    design_abs = np.column_stack(
        [
            _absorb_two_way(design[:, j], unit_idx, time_idx, n_units, n_periods)
            for j in range(design.shape[1])
        ]
    )
    y_abs = _absorb_two_way(y_long, unit_idx, time_idx, n_units, n_periods)

    xtx_inv = np.linalg.pinv(design_abs.T @ design_abs)
    coef = xtx_inv @ (design_abs.T @ y_abs)
    resid = y_abs - design_abs @ coef

    # ---- 影响函数（按单元聚类）：ψ_i = n·(X̃'X̃)^{-1} Σ_t X̃_it·û_it ------- #
    #
    # 为什么要乘 n：`_se_from_influence` 用的是**均值型**约定
    # （``se = sqrt(Var(ψ)/n)``，对应"ψ 是单个单元对估计量的贡献"）。
    # 回归系数的聚类稳健方差是 ``Σ_i ψ_i²``（ψ 未乘 n 时），
    # 两者差一个 n：不乘 n 的话 SE 会被低估约 n 倍 —— 实测差了 400 倍
    # （0.0001 vs 0.0431），这正是"看起来很小很漂亮"的那类错误。
    # 乘 n 之后 ``sqrt(Var(n·ψ)/n) = sqrt(n/(n-1)·Σψ²)``，与聚类稳健方差一致
    # （均值恰好为 0：正规方程给出 X̃'û = 0）。
    weighted = design_abs * resid[:, None]
    meat = np.zeros((n_units, design.shape[1]))
    for j in range(design.shape[1]):
        meat[:, j] = np.bincount(unit_idx, weights=weighted[:, j], minlength=n_units)
    psi_matrix = (meat @ xtx_inv) * n_units  # (n_units, n_coef)

    # ---- 按 k 聚合（与 IW 版同一套规则：队列份额加权、影响函数同步加权） --- #
    by_k: dict[int, list[int]] = {}
    for j, (_g, k) in enumerate(keys):
        by_k.setdefault(k, []).append(j)

    study: dict[int, SAEventStudy] = {}
    for k, idxs in sorted(by_k.items()):
        total = float(sum(n_by_key[keys[j]] for j in idxs))
        weights = {keys[j][0]: n_by_key[keys[j]] / total for j in idxs}
        effect = float(sum(weights[keys[j][0]] * coef[j] for j in idxs))
        psi = np.zeros(n_units)
        for j in idxs:
            psi = psi + weights[keys[j][0]] * psi_matrix[:, j]
        study[k] = SAEventStudy(
            k=k, effect=effect, influence=psi, weights=weights,
            n_treated=int(total),
        )

    return _aggregate_sa(
        panel, study, control_group=control_group, alpha=alpha,
        method="Sun & Abraham (regression, aggregated)",
    )
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
    status: Status = "warn" if p < alpha else "pass"

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



