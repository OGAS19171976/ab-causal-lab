"""簇数很少时的推断：CR1 到底有多过度拒绝，wild bootstrap 修回来多少。

为什么单独一节
--------------
`cluster_robust_ttest`（CR1）是**渐近**的：它要簇数 G 大。
真实业务里 G 常常只有 4~20（几个城市、几十家门店），而这一档上
CR1 的 t 统计量分布与 t(G-2) 差得很远，**过度拒绝**是常态 ——
报告里那个"p < 0.05"看起来完全正常，错的是参考分布。

**先记一条被实测改写的先验**：动手前我写下的判据是"簇数少 → CR1 过度拒绝"。
实测只对了一半 —— **簇数少本身不太出事，簇大小不平衡才是主因**：

* 均衡簇（每簇一样大）下 G=4~8 时 CR1 的 size 只有 0.055~0.075，
  而 G≥20 时反而**偏保守**（t(G-2) 的厚尾盖住了方差的低估）；
* 簇大小 CV=1（对数正态，真实业务里城市/门店规模差别就这么大）之后，
  CR1 在 G=6 时到 **0.1500**、G=20 时仍有 **0.0967** —— 接近名义值的两倍。

所以这一节按 ``(簇数 × 簇大小不平衡)`` 两维扫，量三件事：

* ``size_cr1``：CR1 的真实拒绝率；
* ``size_wild_webb`` / ``size_wild_rademacher``：两种权重下的拒绝率。
  Rademacher 只有 ``±1``，G 个簇最多 ``2^G`` 种抽法 —— G=4 时只有 16 种，
  尾部分位数是锯齿状的，实测它自己就**过度拒绝**（0.1050）；Webb 的 6 点权重
  把它变成 ``6^G``，这是 G < 12 的推荐做法；
* ``power_*``：修 size 的代价。bootstrap 分布比 t 分布厚，功效会掉一点 ——
  掉多少必须报出来，否则"更保守"就成了万能借口。

DGP 与 ``sim.scenarios.generate_cluster_scenario`` 同构（簇级随机效应 + 用户噪声 +
整簇分流），但**故意不受它 ``n_clusters >= 8`` 的限制**：这一节要看的恰恰是
G=4~6 会发生什么。那条限制是 M1 留下的（每臂不足 4 个簇时连簇级方差都估不出来），
在这里不适用 —— 我们要量的是"明知很糟，糟到什么程度"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..inference import cluster_robust_ttest, wild_cluster_bootstrap

__all__ = ["WildBootstrapAudit", "WildBootstrapRow", "run_wild_bootstrap_audit"]


def _draw_cluster_experiment(
    *,
    n_clusters: int,
    users_per_cluster: int,
    cluster_sd: float,
    user_sd: float,
    lift: float,
    seed: int,
    size_cv: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整簇 50/50 分流 + 簇级随机效应 + 用户噪声。

    ``size_cv`` 是**簇大小的变异系数**：0 表示每簇一样大，
    1 表示对数正态（真实业务里城市/门店规模的量级差别）。
    它才是 CR1 过度拒绝的主因 —— 见模块文档那条被改写的先验。
    """
    rng = np.random.default_rng(seed)
    cluster_effect = rng.normal(0.0, cluster_sd, n_clusters)
    treated_cluster = np.zeros(n_clusters, dtype=bool)
    treated_cluster[rng.permutation(n_clusters)[: n_clusters // 2]] = True

    if size_cv > 0:
        sigma2 = np.log1p(size_cv**2)
        raw = rng.lognormal(
            mean=np.log(users_per_cluster) - sigma2 / 2, sigma=np.sqrt(sigma2),
            size=n_clusters,
        )
        sizes = np.maximum(raw.astype(int), 5)
    else:
        sizes = np.full(n_clusters, users_per_cluster, dtype=int)

    codes = np.repeat(np.arange(n_clusters), sizes)
    treated = np.repeat(treated_cluster, sizes)
    noise = rng.normal(0.0, user_sd, codes.size)
    outcome = 50.0 + cluster_effect[codes] + lift * treated_cluster[codes] + noise
    return codes, treated, outcome


@dataclass(frozen=True)
class WildBootstrapRow:
    """一个簇数档位下的读数。"""

    n_clusters: int
    #: 簇大小的变异系数（0 = 每簇一样大）
    size_cv: float
    size_cr1: float
    size_wild_webb: float
    size_wild_rademacher: float
    power_cr1: float
    power_wild_webb: float
    #: Rademacher 权重下 p 值的分辨率（= 2^{-G}）
    rademacher_min_p: float

    def summary(self) -> str:
        tag = "均衡" if self.size_cv == 0 else f"CV={self.size_cv:g}"
        return (
            f"  G={self.n_clusters:<3}{tag:<7} size: CR1 {self.size_cr1:.4f} | "
            f"webb {self.size_wild_webb:.4f} | rademacher {self.size_wild_rademacher:.4f}"
            f"（最小 p {self.rademacher_min_p:.4f}）  ||  "
            f"功效: CR1 {self.power_cr1:.4f} | webb {self.power_wild_webb:.4f}"
        )


@dataclass(frozen=True)
class WildBootstrapAudit:
    """簇数从 4 到 40：CR1 的 size 崩到什么程度，wild bootstrap 修回多少，代价是多少。"""

    n_trials: int
    users_per_cluster: int
    cluster_sd: float
    user_sd: float
    alpha: float
    n_bootstrap: int
    lift: float
    rows: tuple[WildBootstrapRow, ...]

    @property
    def cr1_oversized_when_unbalanced(self) -> bool:
        """**不平衡**簇大小下 CR1 明显超过名义值（实测主因是这个，不是簇数）。"""
        bad = [r for r in self.rows if r.size_cv > 0]
        return bool(bad) and bad[0].size_cr1 > 1.5 * self.alpha

    @property
    def balanced_small_g_is_not_the_problem(self) -> bool:
        """**与先验相反的那一条**：均衡簇下最少的几档 CR1 并没有严重超名义。

        实测：均衡簇 G=4~8 时 CR1 的 size 只有 0.055~0.075；
        而 G≥20 时它反而偏保守 —— t(G-2) 的厚尾盖住了方差的低估。
        把它写成断言，是为了防止后来人（包括我自己）在没量过的前提下
        把"少簇"当成过度拒绝的充分条件。
        """
        balanced = [r for r in self.rows if r.size_cv == 0]
        return all(r.size_cr1 <= 1.5 * self.alpha for r in balanced)

    @property
    def webb_reduces_size_when_unbalanced(self) -> bool:
        """**Webb 在每一个不平衡档位上都把 size 压下来**（不是"压到名义"）。"""
        bad = [r for r in self.rows if r.size_cv > 0]
        return bool(bad) and all(r.size_wild_webb < r.size_cr1 for r in bad)

    @property
    def webb_closer_to_nominal_when_unbalanced(self) -> bool:
        """不平衡档位上 Webb 离名义值比 CR1 更近（含"矫枉过正"的那些档）。"""
        bad = [r for r in self.rows if r.size_cv > 0]
        return bool(bad) and all(
            abs(r.size_wild_webb - self.alpha) < abs(r.size_cr1 - self.alpha) for r in bad
        )

    @property
    def worst_power_cost(self) -> float:
        """修 size 的**最大**功效代价（百分点）。

        **不能只说"代价很小"**：实测在"每臂 2 个簇 + 簇大小差一个量级"那一档，
        wild bootstrap 会**矫枉过正**——size 掉到名义值以下、功效只剩个位数百分比。
        那种情形下正确的做法不是换检验，而是承认这批数据说不清
        （或者去做簇级配对/协变量调整）。这个数就是给读者看的。
        """
        if not self.rows:
            return float("nan")
        return float(max(r.power_cr1 - r.power_wild_webb for r in self.rows))

    @property
    def rademacher_fails_at_minimal_g(self) -> bool:
        """G 最小那一档上 Rademacher 自己就过度拒绝（分辨率只有 2^-G）。"""
        smallest = min(r.n_clusters for r in self.rows)
        rows = [r for r in self.rows if r.n_clusters == smallest]
        return any(r.size_wild_rademacher > 1.5 * self.alpha for r in rows)

    @property
    def power_cost_is_small(self) -> bool:
        """修 size 的代价：功效掉得不超过 15 个百分点。"""
        return all(r.power_cr1 - r.power_wild_webb <= 0.15 for r in self.rows)

    def summary(self) -> str:
        lines = [
            f"wild cluster bootstrap 的 size/功效（{self.n_trials} 次/档，"
            f"每簇约 {self.users_per_cluster} 人，簇间 sd {self.cluster_sd:g} / "
            f"簇内 sd {self.user_sd:g}，α={self.alpha}，B={self.n_bootstrap}，"
            f"真实效应 {self.lift:g}）",
        ]
        lines += [r.summary() for r in self.rows]
        lines += [
            "  读法（都是量出来的；第一条推翻了我的先验）：",
            "    · **簇数少本身不太出事**：均衡簇下 CR1 的 size 在各档都接近名义值；",
            "      真正的主因是**簇大小不平衡** —— 同一批簇数下 CV=1 会让 CR1 的 size",
            "      翻到名义值的 2~3 倍，而且**不随 G 增大而消失**；",
            "    · **Webb 把 size 压下来**（不平衡档位每一档都比 CR1 低），但**不一定压到名义**；",
            "    · **Rademacher 在簇数最少时自己会过度拒绝**：只有 2^G 种抽法、",
            "      尾部分位数是锯齿状的 —— 这就是 G<12 推荐 Webb 的原因；",
            "    · **代价要一起看**：每臂只有 2 个簇且大小差一个量级时，Webb 会矫枉过正"
            "（size 掉到名义值以下、功效只剩个位数）。",
        ]
        return "\n".join(lines)


def run_wild_bootstrap_audit(
    *,
    n_trials: int = 250,
    n_clusters_grid: tuple[int, ...] = (4, 6, 8, 12, 20),
    size_cv_grid: tuple[float, ...] = (0.0, 1.0),
    users_per_cluster: int = 100,
    cluster_sd: float = 8.0,
    user_sd: float = 10.0,
    lift: float = 6.0,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: int = 0,
) -> WildBootstrapAudit:
    """扫「簇数 × 簇大小不平衡」，逐档量 size（真实效应 0）与功效（真实效应 ``lift``）。"""
    rows: list[WildBootstrapRow] = []
    grid = [(g, cv) for g in n_clusters_grid for cv in size_cv_grid]
    for k, (g, cv) in enumerate(grid):
        size_cr1 = size_webb = size_rad = 0
        power_cr1 = power_webb = 0
        for i in range(n_trials):
            # --- H0：真实效应为 0，量 size ---
            codes, treated, y = _draw_cluster_experiment(
                n_clusters=g, users_per_cluster=users_per_cluster,
                cluster_sd=cluster_sd, user_sd=user_sd, lift=0.0,
                seed=seed + 1_000 * k + i, size_cv=cv,
            )
            cr1 = cluster_robust_ttest(codes, treated, y, alpha=alpha)
            webb = wild_cluster_bootstrap(
                codes, treated, y, n_bootstrap=n_bootstrap, weights="webb",
                alpha=alpha, seed=seed + 10_000 + i,
            )
            rad = wild_cluster_bootstrap(
                codes, treated, y, n_bootstrap=n_bootstrap, weights="rademacher",
                alpha=alpha, seed=seed + 20_000 + i,
            )
            size_cr1 += int(cr1.p_value < alpha)
            size_webb += int(webb.p_value < alpha)
            size_rad += int(rad.p_value < alpha)

            # --- H1：真实效应 = lift，量功效 ---
            codes, treated, y = _draw_cluster_experiment(
                n_clusters=g, users_per_cluster=users_per_cluster,
                cluster_sd=cluster_sd, user_sd=user_sd, lift=lift,
                seed=seed + 500_000 + 1_000 * k + i, size_cv=cv,
            )
            power_cr1 += int(cluster_robust_ttest(codes, treated, y, alpha=alpha).p_value < alpha)
            power_webb += int(
                wild_cluster_bootstrap(
                    codes, treated, y, n_bootstrap=n_bootstrap, weights="webb",
                    alpha=alpha, seed=seed + 30_000 + i,
                ).p_value < alpha
            )

        rows.append(
            WildBootstrapRow(
                n_clusters=g,
                size_cv=cv,
                size_cr1=size_cr1 / n_trials,
                size_wild_webb=size_webb / n_trials,
                size_wild_rademacher=size_rad / n_trials,
                power_cr1=power_cr1 / n_trials,
                power_wild_webb=power_webb / n_trials,
                rademacher_min_p=float(2.0 ** (-g)),
            )
        )

    return WildBootstrapAudit(
        n_trials=n_trials,
        users_per_cluster=users_per_cluster,
        cluster_sd=cluster_sd,
        user_sd=user_sd,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        lift=lift,
        rows=tuple(rows),
    )
