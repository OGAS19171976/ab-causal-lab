"""簇级 CUPED 的**校准**：整簇随机化下，三种口径在 A/A 下的误停率。

为什么单独一个审计
------------------
簇级 CUPED 打开之后，"方差确实降了"只是一半证据 —— 另一半是
**它的名义水平守不守得住**。整簇随机化这条路最著名的坑就是：
按**用户**做推断会把簇内相关当成独立信息，I 类错误率被抬到 60% 以上
（M1/M6 实测 69.3%），而那个 p 值看起来完全正常。
CUPED 引入了回归调整，于是同一个问题要重新问一遍：
调整之后，**观测单位还是簇吗**？

能重复抽样的数据源只有合成路径（数仓只有一份实现），所以这个审计走合成路径：
换 salt 就是换一次实验实现，于是可以量出 A/A 下的误停率。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterCupedAudit:
    """整簇随机化下三种口径的运行特征（A/A，真实效应为 0）。"""

    n_trials: int
    n_users: int
    alpha: float
    #: 按**用户**做 t 检验的误停率 —— 这是错误做法，用来当参照
    unit_level_fpr: float
    #: 簇级 Welch（post-only）的误停率
    cluster_level_fpr: float
    #: **簇级 CUPED** 的误停率 —— 这一轮真正要量的东西
    cluster_cuped_fpr: float
    #: 簇级 CUPED 的平均方差缩减与簇间前后相关
    mean_variance_reduction: float
    mean_cluster_correlation: float
    mean_clusters: float
    #: Wilson 区间（误停率的抽样误差），便于判断"离 5% 远不远"
    cluster_cuped_ci: tuple[float, float] = (float("nan"), float("nan"))
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lo, hi = self.cluster_cuped_ci
        return "\n".join(
            [
                f"簇级 CUPED 校准（{self.n_trials} 次 A/A，n={self.n_users}，"
                f"平均簇数 {self.mean_clusters:.1f}，alpha={self.alpha}）",
                f"  用户级 t 检验（**错误做法**）：误停率 {self.unit_level_fpr:.4f}",
                f"  簇级 Welch（post-only）  ：误停率 {self.cluster_level_fpr:.4f}",
                f"  **簇级 CUPED**            ：误停率 {self.cluster_cuped_fpr:.4f}"
                f"  Wilson [{lo:.4f}, {hi:.4f}]",
                f"  簇级 CUPED 的平均方差缩减 {self.mean_variance_reduction:.2%}"
                f"（簇间前后相关 {self.mean_cluster_correlation:.3f}）",
                "  读法：三行里的第一行是「别这么干」的参照；第二、三行都必须守住名义水平 ——"
                "CUPED 只是把协变量调整进来，**观测单位仍然是簇**。",
            ]
        )


def _wilson(hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (centre - half, centre + half)


def run_cluster_cuped_audit(
    *,
    n_trials: int = 200,
    n_users: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ClusterCupedAudit:
    """跑 A/A：真实效应为 0，看三种口径各自拒绝多少次。

    走产品入口 ``analyse_experiment``（不是自己拼统计量），
    所以量到的是**产品行为**：声明 ``analysis_unit="cluster"`` +
    ``estimator="cuped"`` 时，平台到底按什么单位做推断。
    """
    from ..platform.analysis import analyse_experiment
    from ..platform.registry import ExperimentRecord

    variants = [
        {"name": "control", "weight": 0.5},
        {"name": "treatment", "weight": 0.5},
    ]
    hits = {"unit": 0, "cluster": 0, "cuped": 0}
    reductions: list[float] = []
    correlations: list[float] = []
    clusters: list[float] = []

    for i in range(n_trials):
        salt = f"cluster_cuped_aa_{seed}_{i}"
        cuped_rec = ExperimentRecord(
            name="cluster_aa",
            variants=list(variants),
            salt=salt,
            primary_metric="post_metric_14d",
            analysis_unit="cluster",
            estimator="cuped",
        )
        rep = analyse_experiment(cuped_rec, n_users=n_users, alpha=alpha)
        if rep.primary is not None and rep.primary.p_value < alpha:
            hits["cuped"] += 1
        if rep.alt is not None and rep.alt.p_value < alpha:
            hits["cluster"] += 1
        if rep.cuped_fit is not None:
            reductions.append(rep.cuped_fit.variance_reduction)
            correlations.append(rep.cuped_fit.correlation)

        # 用户级错误做法：post-only 的对照口径就是"误按用户级做检验"
        post_rec = ExperimentRecord(
            name="cluster_aa",
            variants=list(variants),
            salt=salt,
            primary_metric="post_metric_14d",
            analysis_unit="cluster",
            estimator="post_only",
        )
        rep_post = analyse_experiment(post_rec, n_users=n_users, alpha=alpha)
        if rep_post.alt is not None and rep_post.alt.p_value < alpha:
            hits["unit"] += 1
        if rep_post.primary is not None:
            clusters.append(float(rep_post.primary.n_treatment + rep_post.primary.n_control))

    mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    return ClusterCupedAudit(
        n_trials=n_trials,
        n_users=n_users,
        alpha=alpha,
        unit_level_fpr=hits["unit"] / n_trials,
        cluster_level_fpr=hits["cluster"] / n_trials,
        cluster_cuped_fpr=hits["cuped"] / n_trials,
        mean_variance_reduction=mean(reductions),
        mean_cluster_correlation=mean(correlations),
        mean_clusters=mean(clusters),
        cluster_cuped_ci=_wilson(hits["cuped"], n_trials),
        notes=[
            "整簇随机化下按用户做推断是错的（误停率远超名义水平），"
            "而簇级 CUPED 只是把协变量调整进来，观测单位必须仍然是簇。",
        ],
    )


__all__ = ["ClusterCupedAudit", "run_cluster_cuped_audit"]
