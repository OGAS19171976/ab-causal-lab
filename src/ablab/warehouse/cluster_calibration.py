"""数仓路径上的**簇级** A/A 校准：整簇随机化下的误停率与 z 分布。

补的是哪条边界
--------------
M6 的簇级 CUPED 校准一直只有**合成路径**那一份（换 salt 跑 200 次，
误停率 0.0400），而数仓路径上写的是"没有量过 —— 数仓只有一份实现、换不了 salt"。
这一轮造了 ``n`` 个**整簇随机化**的 A/A 复制实验（``cluster_replicate_experiments``），
那句话就不再成立了：数仓路径上也能换 salt 重复。

为什么必须单独量一遍
--------------------
簇级推断走的是**另一条数据链路**：ADS 给臂级总数、簇粒度 DWS 给"每组簇的统计量"，
两者由 ``ExperimentData.validate()`` 的不变量钉在一起。合成路径上用对了统计量，
不能推出数仓那条读取路径也对 —— 这正是本仓库反复强调的
"换了数据源，校准主张不自动成立"。

与单元级那批复制实验同一个设计逻辑：``true_lift=0`` 保证共享结果序列不被污染，
各自一层、各自 salt 保证分流互相独立，于是尖锐零假设逐字成立、
重复就是随机化分布的 i.i.d. 抽样。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..platform.analysis import analyse_experiment_from_warehouse
from ..platform.registry import ExperimentRecord
from ..validation.aa import wilson_interval
from .generate import CLUSTER_REPLICATE_PREFIX

__all__ = [
    "ClusterCalibration",
    "cluster_replicate_name",
    "cluster_replicate_record",
    "run_cluster_replicate_calibration",
]


def cluster_replicate_name(i: int, prefix: str = CLUSTER_REPLICATE_PREFIX) -> str:
    """第 ``i`` 个整簇复制实验的名字。"""
    return f"{prefix}{i:03d}"


def cluster_replicate_record(
    i: int, *, estimator: str = "cuped", prefix: str = CLUSTER_REPLICATE_PREFIX
) -> ExperimentRecord:
    """包成平台记录：数仓路径 + **簇级**分析单元 + 指定的估计量。"""
    name = cluster_replicate_name(i, prefix)
    return ExperimentRecord(
        id=f"clu_{estimator}_{i:03d}",
        name=name,
        salt=f"{name}_v1",
        variants=[
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ],
        primary_metric="post_metric_14d",
        warehouse_experiment=name,
        estimator=estimator,
        analysis_unit="cluster",
    )


@dataclass
class ClusterCalibration:
    """数仓路径上整簇 A/A 的校准读数。

    三个数字回答三件事：

    * ``fpr``：末次查看的误停率（名义 alpha）与 Wilson 区间；
    * ``z_sd`` / ``z_mean``：末次 z 的分布（应为 1 与 0）—— SE 是否诚实的直接证据；
    * ``n_clusters_min``：每个复制实验实际拿到的**簇数**（少簇时簇级检验会更保守，
      所以这个数必须报出来，否则读不出"为什么误停率可能低于名义值"）。
    """

    estimator: str
    n_replicates: int
    alpha: float
    fpr: float
    fpr_interval: tuple[float, float]
    z_mean: float
    z_sd: float
    effect_mean: float
    se_mean: float
    n_clusters_mean: float
    n_clusters_min: int
    n_users_mean: float
    #: 每个复制实验的末次 z（供复核）
    final_z: list[float] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        """Wilson 区间盖住名义 alpha。"""
        return self.fpr_interval[0] <= self.alpha <= self.fpr_interval[1]

    @property
    def z_is_honest(self) -> bool:
        """z 的均值在 ±0.3、标准差在 [0.7, 1.4] 之内（少簇时本来就偏保守）。"""
        return abs(self.z_mean) <= 0.3 and 0.7 <= self.z_sd <= 1.4

    def summary(self) -> str:
        return "\n".join(
            [
                f"数仓路径上的簇级 A/A 校准（{self.n_replicates} 个整簇复制实验，"
                f"估计量 {self.estimator}，名义 alpha={self.alpha}）",
                f"  误停率 {self.fpr:.4f}（Wilson [{self.fpr_interval[0]:.4f}, "
                f"{self.fpr_interval[1]:.4f}]）",
                f"  末次 z：均值 {self.z_mean:+.4f}，sd {self.z_sd:.4f}",
                f"  点估计：均值 {self.effect_mean:+.4f}，平均 SE {self.se_mean:.4f}",
                f"  每个复制实验：平均 {self.n_clusters_mean:.1f} 个簇"
                f"（最少 {self.n_clusters_min}），平均 {self.n_users_mean:.0f} 个用户",
                "  读法：簇级推断在数仓路径上也守住名义水平；z 的 sd 若明显小于 1，"
                "说明少簇 + t(G-2) 让它偏保守（那是**该有**的，不是 bug）。",
            ]
        )


def run_cluster_replicate_calibration(
    con,
    *,
    n_replicates: int = 40,
    estimator: str = "cuped",
    alpha: float = 0.05,
    n_looks: int = 5,
    prefix: str = CLUSTER_REPLICATE_PREFIX,
) -> ClusterCalibration:
    """在数仓上跑 ``n_replicates`` 个整簇 A/A，量误停率与 z 的分布。"""
    if n_replicates < 5:
        raise ValueError("复制实验至少要 5 个，否则误停率量不出来")
    hits = 0
    z_values: list[float] = []
    effects: list[float] = []
    ses: list[float] = []
    clusters: list[int] = []
    users: list[int] = []

    for i in range(n_replicates):
        rep = analyse_experiment_from_warehouse(
            cluster_replicate_record(i, estimator=estimator, prefix=prefix),
            con,
            n_looks=n_looks,
            alpha=alpha,
        )
        primary = rep.primary
        rows = rep.monitoring
        if primary is None or not rows:  # pragma: no cover - 数仓路径必然给出
            raise RuntimeError(f"{cluster_replicate_name(i, prefix)} 没有主结论")
        est = float(primary.absolute_effect)
        se = float(primary.std_error)
        effects.append(est)
        ses.append(se)
        z_values.append(est / se if se > 0 else float("nan"))
        hits += int(float(primary.p_value) < alpha)
        n_t = int(rows[-1].get("n_clusters_treatment") or 0)
        n_c = int(rows[-1].get("n_clusters_control") or 0)
        clusters.append(n_t + n_c)
        users.append(int(rows[-1]["n_treatment"]) + int(rows[-1]["n_control"]))

    z_arr = np.asarray(z_values, dtype=float)
    eff = np.asarray(effects, dtype=float)
    se_arr = np.asarray(ses, dtype=float)
    return ClusterCalibration(
        estimator=estimator,
        n_replicates=n_replicates,
        alpha=alpha,
        fpr=hits / n_replicates,
        fpr_interval=wilson_interval(hits, n_replicates),
        z_mean=float(np.nanmean(z_arr)),
        z_sd=float(np.nanstd(z_arr, ddof=1)),
        effect_mean=float(eff.mean()),
        se_mean=float(se_arr.mean()),
        n_clusters_mean=float(np.mean(clusters)),
        n_clusters_min=int(np.min(clusters)),
        n_users_mean=float(np.mean(users)),
        final_z=[float(z) for z in z_arr],
    )


def cluster_pass_through_share(con, prefix: str = CLUSTER_REPLICATE_PREFIX) -> pd.DataFrame:
    """每个复制实验的簇数/人数（给报告用），顺带暴露"少簇"这种结构问题。"""
    frame = con.execute(
        """
        SELECT experiment,
               COUNT(DISTINCT city) AS n_clusters,
               COUNT(DISTINCT user_id) AS n_users
        FROM dwd_experiment_user
        WHERE experiment LIKE ?
        GROUP BY experiment ORDER BY experiment
        """,
        [f"{prefix}%"],
    ).df()
    return frame
