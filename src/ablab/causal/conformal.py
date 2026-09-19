"""个体效应的**保形预测区间**（Lei & Candès 2021，arXiv:2006.06138）。

它解决的是什么、不解决什么（这一段最要紧）
------------------------------------------
本仓库早先量到：以 τ̂ 为心的**解析区间**覆盖率只有 0.13~0.44，而且
Chernozhukov 等证明了高维/非参下 **τ(x) 的自适应置信集不存在**。
那两件事说的是**条件均值函数** τ(x) = E[Y(1)−Y(0) | X=x] 的置信集。

保形走的是另一个对象：**个体效应本身** τ_i = Y_i(1) − Y_i(0)（一个随机变量）的
**预测区间**，覆盖是**边际**的：``P(τ_i ∈ Ĉ(X_i)) ≥ 1−α``。
两者不矛盾 —— 前者在 x 处要一致覆盖整条函数，后者是"在整个总体上平均，
有多少比例的个体的效应被区间盖住"。**所以这个区间不能用来对单个 x 下结论**，
它保证的是"如果你按这套规则给每个人一个区间，至少 95% 的人被盖住"。

机制（与原文的差异也写在这里）
------------------------------
原文 Algorithm 1 用加权 split-CQR（分位数回归）。本仓库没有分位数回归，
所以这里用**均值 + 绝对残差分数**这一档：分数 ``V_i = |Y_i − μ̂_{D_i}(X_i)|``，
区间长度由校准集上 V 的加权分位数给出。差异的后果：
区间对**异方差**不敏感（同一个长度给所有人），covariate shift 由权重处理。
两件事都如实写进报告，不假装是原文的完整实现。

权重：在随机化实验里 p(X) 已知，但 P(X|T=1) 与 P(X|T=0) 一般不同，
所以用控制组去预测处置组的反事实时，校样样本要按
``w(x) = P(T=0)/P(T=1) · p(x)/(1−p(x))`` 加权（反之亦然）——
这正是原文里 w₀(x) 的形式。**倾向得分已知**是本仓库仿真的前提（HTEConfig 提供它）。

覆盖的性质：原文证明，完全随机化/分层随机化 + 已知倾向得分时，
有限样本覆盖**不需要任何额外假设**（只要样本 i.i.d.）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalITE:
    """保形区间 + 它自己的诊断。"""

    #: 每个单元的区间（长度相同 —— 均值型分数只给一个常数半宽）
    lower: np.ndarray
    upper: np.ndarray
    half_width: float
    #: 校准集里两臂各自的样本量（决定分位数的有限样本修正）
    n_cal_treated: int
    n_cal_control: int
    #: 权重是否真的用上了（倾向得分非常数时为 True）
    weighted: bool

    def coverage(self, tau_true: np.ndarray) -> float:
        """边际覆盖率（真值只有仿真里才有）。"""
        inside = (self.lower <= tau_true) & (tau_true <= self.upper)
        return float(np.mean(inside))

    def mean_width(self) -> float:
        return float(np.mean(self.upper - self.lower))

    def summary(self, tau_true: np.ndarray | None = None) -> str:
        lines = [
            f"保形个体效应区间（半宽 {self.half_width:.4f}，"
            f"校准集 {self.n_cal_treated}+{self.n_cal_control}，"
            f"加权={'是' if self.weighted else '否'}）",
        ]
        if tau_true is not None:
            lines.append(f"  边际覆盖率 {self.coverage(tau_true):.4f}"
                         f"（名义 {1 - 0.05:.2f}），平均宽度 {self.mean_width():.4f}")
        return "\n".join(lines)


def _fit_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """带截距的最小二乘（与审计里那个 ``_fit_predict`` 同一个口径）。"""
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return beta


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ beta


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, level: float
) -> float:
    """加权分位数：把权归一化后取累积权重的 ``level`` 分位。

    权重全相等时它退化成普通的经验分位数（那种情形下与 split conformal 一致）。
    """
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    if w.sum() <= 0:
        return float(np.quantile(values, level))
    cum = np.cumsum(w) / w.sum()
    idx = int(np.searchsorted(cum, level, side="left"))
    return float(v[min(idx, v.size - 1)])


def conformal_ite_intervals(
    *,
    x: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    propensity: np.ndarray,
    alpha: float = 0.05,
    seed: int = 0,
) -> ConformalITE:
    """给**样本内**每个单元一个个体效应的保形区间。

    ``propensity`` 是**已知**的倾向得分（本仓库的仿真提供它；真实随机化实验里
    它就是设计给出的分流概率）。

    步骤：

    1. 劈成训练/校准两半（互不重叠 —— 用训练样本拟合、校准样本定分位数）；
    2. 训练集上分别拟合两臂的结局模型 μ̂₁、μ̂₀；
    3. 校准集上算绝对残差分数 ``V_i = |Y_i − μ̂_{D_i}(X_i)|``，
       并算权重（控制组样本用于预测处置组反事实时按 ``p/(1−p)`` 加权，反之亦然）；
    4. 处置单元缺 Y(0)：``τ_i = Y_i − Y_i(0)``，所以区间由 Y(0) 的区间镜像得到；
       控制单元对称处理；
    5. 分位数取 ``(1−α)(1+1/n_cal)`` 这一档 —— 这是有限样本覆盖的标准修正。
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(propensity, dtype=float)
    if not (x.shape[0] == d.size == y.size == p.size):
        raise ValueError("x / d / y / propensity 长度必须一致")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    half = x.shape[0] // 2
    train, cal = perm[:half], perm[half:]

    t_train, c_train = train[d[train] == 1], train[d[train] == 0]
    if t_train.size < 5 or c_train.size < 5:
        raise ValueError("训练集某一臂样本太少，无法拟合结局模型")
    beta1 = _fit_mean(x[t_train], y[t_train])
    beta0 = _fit_mean(x[c_train], y[c_train])

    t_cal, c_cal = cal[d[cal] == 1], cal[d[cal] == 0]
    if t_cal.size < 5 or c_cal.size < 5:
        raise ValueError("校准集某一臂样本太少，无法定分位数")

    # 校准分数：用**本臂**的模型
    v_t = np.abs(y[t_cal] - _predict(beta1, x[t_cal]))
    v_c = np.abs(y[c_cal] - _predict(beta0, x[c_cal]))

    # 权重：控制组样本的协变量分布与处置组不同，用 p/(1−p) 校正
    # （形式上与原文的 w₀(x) 一致，常数因子不影响分位数）
    w_c = p[c_cal] / np.maximum(1.0 - p[c_cal], 1e-12)
    w_t = (1.0 - p[t_cal]) / np.maximum(p[t_cal], 1e-12)
    weighted = bool(np.ptp(p) > 1e-6)

    # 校准集里用的是"该组自己的 p(x)"，所以权重需要归一化到同一尺度
    level_c = (1.0 - alpha) * (1.0 + 1.0 / c_cal.size)
    level_t = (1.0 - alpha) * (1.0 + 1.0 / t_cal.size)
    q_c = _weighted_quantile(v_c, w_c, min(level_c, 1.0))
    q_t = _weighted_quantile(v_t, w_t, min(level_t, 1.0))

    mu1 = _predict(beta1, x)
    mu0 = _predict(beta0, x)

    lower = np.empty(x.shape[0])
    upper = np.empty(x.shape[0])
    treated = d == 1
    # 处置单元：τ = Y(1) − Y(0)，观测到 Y(1)=y，缺的是 Y(0) → 用控制组的分位数
    lower[treated] = (y[treated] - mu0[treated]) - q_c
    upper[treated] = (y[treated] - mu0[treated]) + q_c
    # 控制单元：对称，缺的是 Y(1) → 用处置组的分位数
    lower[~treated] = (mu1[~treated] - y[~treated]) - q_t
    upper[~treated] = (mu1[~treated] - y[~treated]) + q_t

    half_width = float((q_c * treated.sum() + q_t * (~treated).sum()) / x.shape[0])
    return ConformalITE(
        lower=lower,
        upper=upper,
        half_width=half_width,
        n_cal_treated=int(t_cal.size),
        n_cal_control=int(c_cal.size),
        weighted=weighted,
    )


__all__ = ["ConformalITE", "conformal_ite_intervals"]
