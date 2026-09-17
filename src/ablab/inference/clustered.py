"""聚类稳健标准误：当**处理是在簇级别分配**时，用户级 t 检验会严重高估显著性。

问题出在哪
----------
把城市、门店、market 整群随机分成实验/对照，用户嵌在簇里。
同一座城市的用户共享城市级冲击（当地促销、天气、竞品动作），
他们的结果高度相关 —— 这就是组内相关 ICC。

用户级 t 检验假设所有用户独立，于是把"有效样本量"当成用户数 n，
而真实的独立单元数只有簇数 G。样本量被高估，标准误被低估：

    方差膨胀因子（design effect） deff = 1 + (m̄ - 1)·ICC
    真实标准误 ≈ 朴素标准误 × sqrt(deff)

G=200 个簇、每簇 100 人、ICC=0.1 时 deff = 1 + 99×0.1 ≈ 10.9，
真实标准误是朴素标准误的 3.3 倍 —— 一个 t=3.3 的"显著结果"其实什么都不是。

两种正确做法
------------
``cluster_robust_ttest``（CR1 三明治）
    保留用户级估计量（用户加权口径），但把方差换成聚类稳健版本。
    自由度用 G-2，这是保守且标准的取法。
``cluster_level_ttest``（先聚合再检验）
    把每个簇压成一个数再比较。最简单、最稳健，估计的是**簇均值**口径
    （每座城市等权）。当簇大小差异很大时，它与用户加权口径不同。

两个口径都对，但要**事先**想清楚哪个是业务要问的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .result import Diagnostic, Estimate
from .welch import t_inference, welch_inference

__all__ = [
    "ClusterDiagnostics",
    "cluster_robust_ttest",
    "cluster_level_ttest",
    "estimate_icc",
]


@dataclass(frozen=True)
class ClusterDiagnostics:
    """聚类结构的关键指标，决定朴素 t 检验错得有多离谱。"""

    n_units: int
    n_clusters: int
    n_treatment_clusters: int
    n_control_clusters: int
    mean_cluster_size: float
    icc: float

    @property
    def design_effect(self) -> float:
        """方差膨胀因子 = 1 + (m̄ - 1)·ICC。"""
        return 1.0 + (self.mean_cluster_size - 1.0) * max(self.icc, 0.0)

    @property
    def se_inflation(self) -> float:
        """朴素标准误被低估的倍数 = sqrt(deff)。"""
        return float(np.sqrt(self.design_effect))

    @property
    def effective_sample_size(self) -> float:
        """有效独立单元数 ≈ n / deff。"""
        return self.n_units / self.design_effect if self.design_effect > 0 else float("nan")

    def summary(self) -> str:
        return (
            f"聚类结构: {self.n_clusters} 个簇（处理 {self.n_treatment_clusters} / "
            f"对照 {self.n_control_clusters}），共 {self.n_units:,} 个单元，"
            f"平均簇大小 {self.mean_cluster_size:.1f}\n"
            f"  ICC = {self.icc:.4f}  ->  设计效应 {self.design_effect:.2f}，"
            f"朴素标准误被低估 {self.se_inflation:.2f} 倍\n"
            f"  有效独立单元数 ≈ {self.effective_sample_size:,.0f}"
        )


def _encode_clusters(cluster_ids: Sequence) -> np.ndarray:
    cid = np.asarray(cluster_ids)
    if cid.ndim != 1:
        raise ValueError("cluster_ids 必须是一维")
    _, codes = np.unique(cid, return_inverse=True)
    return codes.astype(np.int64)


def estimate_icc(cluster_ids: Sequence, outcome) -> tuple[float, float]:
    """单因素随机效应的 ANOVA 估计量，返回 ``(ICC, 平均簇大小 m0)``。

    ``ICC = (MSB - MSW) / (MSB + (m0 - 1)·MSW)``，其中 m0 是不平衡簇的调整均值
    ``m0 = (n - Σn_g²/n) / (G - 1)``。
    """
    codes = _encode_clusters(cluster_ids)
    y = np.asarray(outcome, dtype=float).ravel()
    if y.size != codes.size:
        raise ValueError("cluster_ids 与 outcome 长度不一致")

    G = int(codes.max()) + 1
    n = y.size
    if G < 2 or n <= G:
        return float("nan"), float(n / max(G, 1))

    sizes = np.bincount(codes, minlength=G).astype(float)
    sums = np.bincount(codes, weights=y, minlength=G)
    sq_sums = np.bincount(codes, weights=y * y, minlength=G)

    grand = y.mean()
    ss_between = float((sizes * (sums / sizes - grand) ** 2).sum())
    ss_within = float((sq_sums - sums**2 / sizes).sum())
    df_b, df_w = G - 1, n - G
    if df_w <= 0:
        return float("nan"), float(n / G)

    ms_b = ss_between / df_b
    ms_w = ss_within / df_w

    m0 = (n - float((sizes**2).sum()) / n) / (G - 1)
    denom = ms_b + (m0 - 1.0) * ms_w
    icc = (ms_b - ms_w) / denom if denom > 0 else 0.0
    # 负的 ICC 在估计量里是可能的（真值非负），截断到 0
    return float(max(icc, 0.0)), float(m0)


def _cluster_diagnostics(
    codes: np.ndarray, treated: np.ndarray, y: np.ndarray
) -> ClusterDiagnostics:
    G = int(codes.max()) + 1
    cluster_treated = np.bincount(codes, weights=treated.astype(float), minlength=G) > 0
    icc, m0 = estimate_icc(codes, y)
    return ClusterDiagnostics(
        n_units=int(y.size),
        n_clusters=G,
        n_treatment_clusters=int(cluster_treated.sum()),
        n_control_clusters=int(G - cluster_treated.sum()),
        mean_cluster_size=float(m0),
        icc=icc,
    )


def _check_cluster_assignment(codes: np.ndarray, treated: np.ndarray) -> None:
    """确认处理确实是在簇级别分配的。

    如果同一个簇里既有处理又有对照，那这不是聚类随机化，
    而是"用户级随机 + 聚类相关"—— 此时用户级 t 检验其实没问题，
    用聚类稳健只会白白损失效率。这个检查防止误用。
    """
    G = int(codes.max()) + 1
    t_sum = np.bincount(codes, weights=treated.astype(float), minlength=G)
    sizes = np.bincount(codes, minlength=G)
    mixed = (t_sum > 0) & (t_sum < sizes)
    if mixed.any():
        raise ValueError(
            f"有 {int(mixed.sum())} 个簇内部同时存在处理与对照单元，"
            "这不是聚类随机化。若分流其实是用户级的，直接用 welch_ttest 即可，"
            "不需要聚类稳健标准误。"
        )


def cluster_robust_ttest(
    cluster_ids: Sequence,
    treated: Sequence[bool],
    outcome,
    *,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
) -> Estimate:
    """聚类稳健（CR1 三明治）的均值差检验，估计量仍是**用户加权**口径。

    对模型 ``y_i = a + tau*T_i + e_i``，取 CR1 夹心方差：

        V = (X'X)^{-1} [Σ_g X_g' u_g u_g' X_g] (X'X)^{-1}

    只有 ``V[1,1]``（即 tau 的方差）是我们要的。
    """
    codes = _encode_clusters(cluster_ids)
    t = np.asarray(treated, dtype=bool).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    if not (y.size == t.size == codes.size):
        raise ValueError("cluster_ids / treated / outcome 长度必须一致")
    if y.size < 4:
        raise ValueError("样本量过小")

    _check_cluster_assignment(codes, t)

    n = y.size
    n1 = int(t.sum())
    n0 = n - n1
    if n1 < 2 or n0 < 2:
        raise ValueError(f"两臂样本量不足：{n1} / {n0}")

    mean_t, mean_c = float(y[t].mean()), float(y[~t].mean())
    effect = mean_t - mean_c

    resid = y - np.where(t, mean_t, mean_c)
    G = int(codes.max()) + 1

    # 三明治的"肉"：按簇汇总 Σu 与 Σ T·u
    s_g = np.bincount(codes, weights=resid, minlength=G)
    r_g = np.bincount(codes, weights=t.astype(float) * resid, minlength=G)

    meat11 = float((s_g * s_g).sum())
    meat12 = float((s_g * r_g).sum())
    meat22 = float((r_g * r_g).sum())

    # (X'X)^{-1} 的右上/右下元素；X = [1, T]
    b12 = -1.0 / n0
    b22 = n / (n1 * n0)
    var = b12 * b12 * meat11 + 2 * b12 * b22 * meat12 + b22 * b22 * meat22

    # CR1 小样本修正
    if G > 1 and n > 2:
        var *= (G / (G - 1)) * ((n - 1) / (n - 2))
    var = max(float(var), 0.0)

    diag = _cluster_diagnostics(codes, t, y)
    df = max(G - 2, 1)
    inference = t_inference(effect, se=float(np.sqrt(var)), degrees_of_freedom=df, alpha=alpha)
    ci_low, ci_high = inference.interval(effect)

    return Estimate(
        metric=metric,
        variant=variant,
        control=control_name,
        method="cluster-robust (CR1) mean difference",
        absolute_effect=effect,
        relative_effect=float(effect / mean_c) if mean_c else float("nan"),
        std_error=inference.se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=n1,
        n_control=n0,
        mean_treatment=mean_t,
        mean_control=mean_c,
        alpha=alpha,
        diagnostics=(
            Diagnostic(
                name="聚类稳健",
                status="pass" if diag.design_effect < 1.5 else "warn",
                message=diag.summary(),
                statistic=diag.design_effect,
                detail={
                    "n_clusters": diag.n_clusters,
                    "icc": diag.icc,
                    "design_effect": diag.design_effect,
                    "se_inflation": diag.se_inflation,
                    "degrees_of_freedom": df,
                },
            ),
        ),
    )


def cluster_level_ttest(
    cluster_ids: Sequence,
    treated: Sequence[bool],
    outcome,
    *,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
) -> Estimate:
    """先聚合到簇再检验，估计的是**簇均值口径**（每簇等权）。

    最稳健的做法：把每个簇压成一个数，问题就退化成普通的 Welch t 检验，
    自由度是簇数减 2，天然处理了组内相关。
    """
    codes = _encode_clusters(cluster_ids)
    t = np.asarray(treated, dtype=bool).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    if not (y.size == t.size == codes.size):
        raise ValueError("cluster_ids / treated / outcome 长度必须一致")

    _check_cluster_assignment(codes, t)

    G = int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=G)
    sums = np.bincount(codes, weights=y, minlength=G)
    cluster_mean = sums / sizes
    cluster_treated = np.bincount(codes, weights=t.astype(float), minlength=G) > 0

    n_t_c = int(cluster_treated.sum())
    n_c_c = G - n_t_c
    if n_t_c < 2 or n_c_c < 2:
        raise ValueError(f"每臂至少需要 2 个簇，收到 {n_t_c} / {n_c_c}")

    t_means = cluster_mean[cluster_treated]
    c_means = cluster_mean[~cluster_treated]
    mean_t = float(t_means.mean())
    mean_c = float(c_means.mean())
    effect = mean_t - mean_c

    inference = welch_inference(
        effect,
        n_treatment=n_t_c,
        var_treatment=float(t_means.var(ddof=1)),
        n_control=n_c_c,
        var_control=float(c_means.var(ddof=1)),
        alpha=alpha,
    )
    ci_low, ci_high = inference.interval(effect)
    diag = _cluster_diagnostics(codes, t, y)

    return Estimate(
        metric=metric,
        variant=variant,
        control=control_name,
        method="cluster-level Welch t-test",
        absolute_effect=effect,
        relative_effect=float(effect / mean_c) if mean_c else float("nan"),
        std_error=inference.se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=n_t_c,
        n_control=n_c_c,
        mean_treatment=mean_t,
        mean_control=mean_c,
        alpha=alpha,
        diagnostics=(
            Diagnostic(
                name="簇级口径",
                status="info",
                message=(
                    f"每簇等权（处理 {n_t_c} / 对照 {n_c_c} 个簇），"
                    "与用户加权口径不同；簇大小差异大时两者结论可能不一致。\n    "
                    + diag.summary().replace("\n  ", "\n    ")
                ),
                statistic=diag.icc,
            ),
        ),
    )
