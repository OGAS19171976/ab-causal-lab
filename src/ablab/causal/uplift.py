"""Uplift 排序指标：Qini 曲线与 AUUC，以及它们的**边界**。

定义（本仓库采用的口径）
------------------------
把样本按预测的提升幅度**从大到小**排序。在前 ``phi`` 比例的样本里记
``Y_t(phi)`` 为处理组结果之和、``n_t(phi)`` 为处理组人数，对照组同理，则

    Qini(phi) = Y_t(phi) - Y_c(phi) * n_t(phi) / n_c(phi)

即"只投前 phi 比例的人，相比不做任何定向，多拿到了多少"。
``AUUC`` 就是这条曲线下的面积。

**这个指标有多个互不兼容的定义。** 有的按分位数离散化、有的用累计人数而不是比例、
有的把随机基线也算进去。跨论文比较 Qini 之前必须先确认口径 ——
这也是本模块把定义写在文档里而不是只写"Qini"的原因。

必须记住的边界
--------------
**Qini / AUUC 只衡量排序，完全不衡量水平。**

一个把真实效应放大 1000 倍的模型，排序完全正确，Qini 拉满，
但它的 CATE 估计毫无用处；反过来，一个只会输出常数 ATE 的模型
排序能力为零，MSE 却可能相当不错。

这两件事在实践中经常被混为一谈 —— "我的模型 AUUC 更高"被当成
"CATE 估得更准"。本模块提供 ``scaled_perfect`` 与 ``constant`` 两个
刻意构造的对照，把这个混淆直接摆在台面上。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "UpliftCurve",
    "uplift_curve",
    "auuc",
    "qini_coefficient",
    "rank_correlation",
    "scaled_perfect",
    "constant_prediction",
]


@dataclass
class UpliftCurve:
    """一条 uplift 曲线及其汇总指标。"""

    fractions: np.ndarray
    qini: np.ndarray
    auuc: float
    auuc_random: float
    qini_coefficient: float

    @property
    def max_qini(self) -> float:
        return float(self.qini.max())

    @property
    def final_qini(self) -> float:
        return float(self.qini[-1])

    def summary(self) -> str:
        return (
            f"Uplift 曲线：AUUC = {self.auuc:.4f}（随机基线 {self.auuc_random:.4f}）\n"
            f"  Qini 系数 = {self.qini_coefficient:.4f}   最高点 {self.max_qini:.4f}"
        )


def uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    *,
    n_points: int = 100,
) -> UpliftCurve:
    """按 ``score`` 降序计算 Qini 曲线与 AUUC。"""
    y = np.asarray(y, dtype=float)
    d = np.asarray(treatment, dtype=float)
    s = np.asarray(score, dtype=float)
    n = y.size
    if not (d.size == n and s.size == n):
        raise ValueError("y / treatment / score 的样本量必须一致")
    if n == 0:
        raise ValueError("样本为空")

    order = np.argsort(-s, kind="mergesort")
    y_s, d_s = y[order], d[order]

    cum_yt = np.cumsum(y_s * d_s)
    cum_yc = np.cumsum(y_s * (1.0 - d_s))
    cum_nt = np.cumsum(d_s)
    cum_nc = np.cumsum(1.0 - d_s)

    with np.errstate(divide="ignore", invalid="ignore"):
        qini_full = cum_yt - cum_yc * np.where(cum_nc > 0, cum_nt / np.maximum(cum_nc, 1e-12), 0.0)
    qini_full = np.nan_to_num(qini_full, nan=0.0)

    idx = np.unique(np.linspace(1, n, min(n_points, n)).astype(int)) - 1
    fractions = (idx + 1) / n
    qini = qini_full[idx]

    # 随机基线：从 (0,0) 直线连到终点
    auuc = float(np.trapezoid(qini, fractions))
    auuc_random = float(np.trapezoid(
        np.linspace(0.0, qini[-1], fractions.size), fractions
    ))

    return UpliftCurve(
        fractions=fractions,
        qini=qini,
        auuc=auuc,
        auuc_random=auuc_random,
        qini_coefficient=auuc - auuc_random,
    )


def auuc(y: np.ndarray, treatment: np.ndarray, score: np.ndarray) -> float:
    """便捷入口：只取 AUUC。"""
    return uplift_curve(y, treatment, score).auuc


def qini_coefficient(y: np.ndarray, treatment: np.ndarray, score: np.ndarray) -> float:
    """AUUC 相对随机基线的超出部分 —— 排序能力的度量。"""
    return uplift_curve(y, treatment, score).qini_coefficient


def rank_correlation(true_tau: np.ndarray, predicted: np.ndarray) -> float:
    """真实 CATE 与预测 CATE 的 Spearman 秩相关。

    只看排序，不看水平 —— 和 Qini 想测的是同一件事，
    但因为它用的是**真值**，所以可以在仿真里做裁判。

    任一侧是常数时秩相关无定义，直接返回 ``nan``
    （而不是让 scipy 抛 ConstantInputWarning）。
    """
    a = np.asarray(true_tau, dtype=float)
    b = np.asarray(predicted, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


# --------------------------------------------------------------------------- #
# 两个刻意构造的对照
# --------------------------------------------------------------------------- #
def scaled_perfect(true_tau: np.ndarray, *, factor: float = 1000.0) -> np.ndarray:
    """**排序完美、水平荒谬**：把真实 CATE 放大 ``factor`` 倍。

    它应当拿到最高的 Qini，同时 MSE 大得离谱。
    """
    return factor * np.asarray(true_tau, dtype=float)


def constant_prediction(true_tau: np.ndarray) -> np.ndarray:
    """**排序能力为零、水平还行**：对所有人都预测 ATE。

    它应当拿到 Qini 系数 ≈ 0，但 MSE 相当不错。
    """
    t = np.asarray(true_tau, dtype=float)
    return np.full_like(t, float(t.mean()))
