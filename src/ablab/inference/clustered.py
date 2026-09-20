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

簇数很少时，上面两条都还不够
------------------------------
CR1 是**渐近**的：它要 G 大。G = 4~10 个城市时（真实业务里很常见），
CR1 的 t 统计量分布与 t(G-2) 差得很远，**过度拒绝**是常态。
标准解法是 **wild cluster bootstrap**（Cameron-Gelbach-Miller 2008；
MacKinnon-Webb 2018）：在**施加零假设**的模型上重抽残差，
每个簇整体乘一个随机权重 ``v_g``（簇内共享，所以保留组内相关），
再看重抽出来的 t 统计量有多少超过观测到的。

两个实现细节决定它灵不灵：

* **权重取 Rademacher 还是 Webb**。Rademacher 只有 ``±1``，G 个簇最多
  ``2^G`` 种抽法 —— G=6 时只有 64 种，p 值的最小分辨率就是 1/64，
  而且尾部很粗。Webb 的 6 点权重把可用抽法变成 ``6^G``，
  这是 G < 12 时的推荐做法（MacKinnon & Webb 2018）；
* **零假设要不要施加**。施加（WCR）用受限残差，size 更好；
  不施加（WCU）用无约束残差并把分布平移到估计值上，功效略高但 size 更松。
  两条都实现，因为"哪个更好"取决于你更怕哪一类错误。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 这几个入口收的是"数组或序列" —— 调用方既会传 ``list[float]``（测试里好写），
#: 也会传 ``np.ndarray``（数仓/仿真的路径）。第一版注解写成 ``Sequence``，
#: 于是所有传 ndarray 的地方都被 mypy 报 **错误**：那不是"类型不匹配"，
#: 而是**注解写窄了**（ndarray 在运行时完全可用）。``ArrayLike`` 是 numpy 自己的
#: 惯例，它同时覆盖两者 —— 把注释从"谎"改成"真"。
from numpy.typing import ArrayLike

from .result import Diagnostic, Estimate
from .welch import t_inference, welch_inference

__all__ = [
    "ClusterDiagnostics",
    "WildBootstrapResult",
    "cluster_robust_ttest",
    "cluster_level_ttest",
    "estimate_icc",
    "wild_cluster_bootstrap",
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


def _encode_clusters(cluster_ids: ArrayLike) -> np.ndarray:
    cid = np.asarray(cluster_ids)
    if cid.ndim != 1:
        raise ValueError("cluster_ids 必须是一维")
    _, codes = np.unique(cid, return_inverse=True)
    return codes.astype(np.int64)


def estimate_icc(cluster_ids: ArrayLike, outcome: ArrayLike) -> tuple[float, float]:
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
    cluster_ids: ArrayLike,
    treated: ArrayLike,
    outcome: ArrayLike,
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
    cluster_ids: ArrayLike,
    treated: ArrayLike,
    outcome: ArrayLike,
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


# --------------------------------------------------------------------------- #
# 簇数很少时的推断：wild cluster bootstrap
# --------------------------------------------------------------------------- #
#: Webb 的 6 点权重：均值 0、方差 1，取值比 Rademacher 的 ±1 更丰富。
#: 出处：MacKinnon & Webb (2018) 的六点集合 {±sqrt(3/2), ±1, ±sqrt(1/2)}。
WEBB_WEIGHTS: tuple[float, ...] = (
    -np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5),
)


@dataclass(frozen=True)
class WildBootstrapResult:
    """wild cluster bootstrap-t 的结果，以及必须并排读的 CR1 对照。

    为什么一定要把 CR1 的 p 值一起报：这个方法的全部价值就是"CR1 在少簇时
    过度拒绝"，只报 bootstrap 的 p 值，读者无法判断它到底修掉了什么。
    """

    effect: float
    se_cr1: float
    t_stat: float
    #: bootstrap p 值（分辨率 = 1/(B+1)）
    p_value: float
    #: 同一份数据上 CR1 t 检验的 p 值（自由度 G-2）
    p_value_cr1: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    weights: str
    null: str
    n_clusters: int
    n_clusters_treated: int
    n_clusters_control: int
    n_obs: int

    @property
    def p_resolution(self) -> float:
        """p 值能取到的最小非零值 —— Webb 与 Rademacher 的差别就在这里。"""
        if self.weights == "rademacher":
            return float(2 ** (-self.n_clusters))
        return 1.0 / (self.n_bootstrap + 1)

    def summary(self) -> str:
        return "\n".join(
            [
                f"wild cluster bootstrap-t（{self.weights} 权重，{self.null}，"
                f"B={self.n_bootstrap}）",
                f"  效应 {self.effect:+.4f}，CR1 SE {self.se_cr1:.4f}，t = {self.t_stat:+.3f}",
                f"  bootstrap p = {self.p_value:.4f}（分辨率 {self.p_resolution:.4f}）"
                f"；同一份数据的 CR1 p = {self.p_value_cr1:.4f}",
                f"  簇数：合计 {self.n_clusters}"
                f"（处置 {self.n_clusters_treated} / 对照 {self.n_clusters_control}），"
                f"观测 {self.n_obs}",
                f"  bootstrap-t 区间 [{self.ci_low:+.4f}, {self.ci_high:+.4f}]",
            ]
        )


def _wild_weights(
    rng: np.random.Generator, n_clusters: int, n_bootstrap: int, kind: str
) -> np.ndarray:
    """``(B, G)`` 的簇级权重矩阵。"""
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, n_clusters))
    if kind == "webb":
        return rng.choice(np.asarray(WEBB_WEIGHTS), size=(n_bootstrap, n_clusters))
    raise ValueError(f"weights 只能是 rademacher / webb，收到 {kind!r}")


def wild_cluster_bootstrap(
    cluster_ids: ArrayLike,
    treated: ArrayLike,
    outcome: ArrayLike,
    *,
    n_bootstrap: int = 999,
    weights: str = "webb",
    null: str = "imposed",
    alpha: float = 0.05,
    seed: int = 0,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
) -> WildBootstrapResult:
    """簇数很少时的均值差检验（wild cluster bootstrap-t）。

    做法（施加零假设那一支，WCR）：

    1. 受限模型 ``y = a + e``（即 τ = 0）→ 残差 ``ẽ``；
    2. 每次重抽：``y*_i = a + v_{g(i)}·ẽ_i``，``v_g`` 是簇级随机权重
       （簇内**共享同一个** v，所以组内相关被完整保留）；
    3. 重算 ``τ̂*`` 与它的 CR1 标准误 → ``t* = τ̂* / se*``；
    4. ``p = (1 + #{|t*| ≥ |t_obs|}) / (B + 1)``（加 1 是为了让 p 永远不为 0，
       这是标准的有限 B 修正）；区间用 bootstrap 的 ``|t*|`` 分位数。

    **实现上只用到簇级充分统计量**：因为 ``y*_i = a + v_g ẽ_i`` 里的随机性
    完全在簇这一层，所有需要的量（各簇的 ``Σy*``、``ΣT·y*``、``Σu_g``）
    都能写成 ``(B, G)`` 矩阵上的运算，与用户数无关。
    第一版按"每个 bootstrap 样本重算一遍全样本"写，200 次重抽 × 999 次
    重抽要跑十几分钟 —— 与 Anderson-Rubin 那次是同一个教训：
    **能算和能跑完是两件事**，而这里的杠杆是"找到随机性所在的层级"。
    """
    codes = _encode_clusters(cluster_ids)
    t = np.asarray(treated, dtype=bool).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    if not (y.size == t.size == codes.size):
        raise ValueError("cluster_ids / treated / outcome 长度必须一致")
    if n_bootstrap < 99:
        raise ValueError("n_bootstrap 至少 99（否则 p 值的分辨率没有意义）")
    if null not in ("imposed", "unrestricted"):
        raise ValueError(f"null 只能是 imposed / unrestricted，收到 {null!r}")
    _check_cluster_assignment(codes, t)

    n = y.size
    n1 = int(t.sum())
    n0 = n - n1
    if n1 < 2 or n0 < 2:
        raise ValueError(f"两臂样本量不足：{n1} / {n0}")

    G = int(codes.max()) + 1
    if G < 4:
        raise ValueError(
            f"只有 {G} 个簇：wild bootstrap 也救不了（每臂不足 2 个簇），"
            "这不是实现问题，是识别问题"
        )
    sizes = np.bincount(codes, minlength=G).astype(float)
    treated_share = np.bincount(codes, weights=t.astype(float), minlength=G)
    is_treated = treated_share > (sizes / 2.0)
    n_treated_per_cluster = np.where(is_treated, sizes, 0.0)

    # ---- 观测到的统计量（用户加权口径，与 CR1 那支一致） ------------------ #
    mean_t = float(y[t].mean())
    mean_c = float(y[~t].mean())
    effect = mean_t - mean_c

    cr1 = cluster_robust_ttest(
        cluster_ids, t, y, metric=metric, variant=variant, control_name=control_name,
        alpha=alpha,
    )
    se_cr1 = float(cr1.std_error)
    t_obs = effect / se_cr1 if se_cr1 > 0 else float("nan")

    # ---- 两条重抽路径 ------------------------------------------------------ #
    #
    # * imposed（WCR）：在 τ=0 的受限模型上重抽 ⇒ bootstrap 分布以 0 为中心，
    #   p = P(|t*| ≥ |t_obs|)；
    # * unrestricted（WCU）：在无约束拟合上重抽，统计量取 (τ̂* − τ̂)/se*，
    #   即模拟「τ̂ − τ」的零分布。
    rng = np.random.default_rng(seed)
    V = _wild_weights(rng, G, n_bootstrap, weights)

    if null == "imposed":
        a_base = float(y.mean())
        tau_base = 0.0
        base = y - a_base - tau_base * t
    else:
        a_base = float(y[~t].mean())
        tau_base = effect
        base = y - a_base - tau_base * t

    base_g = np.bincount(codes, weights=base, minlength=G)
    # 每个簇的「拟合值之和」：处置簇是 (a+τ)·n_g，对照簇是 a·n_g
    fitted_g = a_base * sizes + tau_base * n_treated_per_cluster
    # y*_i = 拟合值_i + v_g·残差_i ⇒ 簇级 Σy* = fitted_g + v_g·base_g
    S_g = fitted_g[None, :] + V * base_g[None, :]
    # 处置簇里 T≡1 ⇒ Σ_{i∈g} T_i y*_i = S_g；对照簇里 ≡0
    TY_g = np.where(is_treated[None, :], S_g, 0.0)

    sum_y = S_g.sum(axis=1)
    sum_ty = TY_g.sum(axis=1)
    n_arr = float(n)
    n1_arr = float(n1)

    # OLS（含截距）的 τ̂*：τ = Σ(T − n1/n)·y* ÷ [n0·n1/n]
    denom = n0 * n1 / n
    tau_raw = (sum_ty - (n1_arr / n_arr) * sum_y) / denom
    tau_star = tau_raw - tau_base  # imposed 时 tau_base=0；unrestricted 时平移到 τ̂ 上

    # 残差 u_i = y*_i − a* − τ*T_i：簇级求和只需 S_g、n_g、处置簇的 n_g
    a_ols = sum_y / n_arr - tau_star * (n1_arr / n_arr)
    u_g = (
        S_g
        - a_ols[:, None] * sizes[None, :]
        - tau_star[:, None] * n_treated_per_cluster[None, :]
    )
    # 对照簇里 T≡0 ⇒ ΣT·u = 0；处置簇里 T≡1 ⇒ 就等于 u_g
    tu_g = np.where(is_treated[None, :], u_g, 0.0)

    meat11 = (u_g * u_g).sum(axis=1)
    meat12 = (u_g * tu_g).sum(axis=1)
    meat22 = (tu_g * tu_g).sum(axis=1)

    b12 = -1.0 / n0
    b22 = n / (n1 * n0)
    var_star = b12 * b12 * meat11 + 2 * b12 * b22 * meat12 + b22 * b22 * meat22
    if G > 1 and n > 2:
        var_star = var_star * (G / (G - 1)) * ((n - 1) / (n - 2))
    var_star = np.maximum(var_star, 0.0)
    se_star = np.sqrt(var_star)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(se_star > 0, tau_star / se_star, 0.0)

    # 统计量的中心：imposed 时分布已经以 0 为中心；unrestricted 时上面平移过，
    # 两种情况下比较的都是 |t*| 与 |t_obs|。
    hits = int((np.abs(t_star) >= abs(t_obs)).sum())
    p_value = (1.0 + hits) / (n_bootstrap + 1.0)

    # bootstrap-t 区间：用 |t*| 的分位数对称展开（少簇下的标准做法）
    crit = float(np.quantile(np.abs(t_star), 1.0 - alpha))
    ci_low, ci_high = effect - crit * se_cr1, effect + crit * se_cr1

    return WildBootstrapResult(
        effect=effect,
        se_cr1=se_cr1,
        t_stat=float(t_obs),
        p_value=float(p_value),
        p_value_cr1=float(cr1.p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_bootstrap=n_bootstrap,
        weights=weights,
        null=null,
        n_clusters=G,
        n_clusters_treated=int(is_treated.sum()),
        n_clusters_control=int((~is_treated).sum()),
        n_obs=n,
    )
