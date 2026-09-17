"""窥视问题（Peeking）仿真：为什么"随时看 p 值"会毁掉实验。

这是实验平台面试的**第一高频问题**，也是最容易讲清楚的统计现象：

    固定样本量的 t 检验，其 5% 的 I 类错误率是**在只看一次**的前提下
    成立的。如果每积累一点数据就看一次、看到显著就停，
    等价于做了很多次检验并取"至少一次显著"—— 错误率会单调上升。
    看 10 次时，实际假阳性率能到 20% 以上。

本模块用仿真把这个曲线画出来，并对比三种做法：

``fixed_horizon``
    只在最后一次看 —— 名义 5%，实际 5%，正确。
``naive_peeking``
    每次看都用 p < 0.05 判定 —— 错误率随观察次数膨胀。
``calibrated_boundary``
    保持"每次可看"，但把判定阈值收紧到一个常数边界，
    使得**整个序列**至少误判一次的概率仍为 5%。
    这个边界由仿真在原假设下标定（Pocock 型常数边界的思想）。

M2 会在此基础上实现正式的序贯检验：O'Brien-Fleming 型 alpha spending、
以及 always-valid p 值（mSPRT），让"随时可停"建立在解析保证上，
而不是仿真标定上。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..sim.generator import Population

__all__ = ["PeekResult", "calibrate_constant_boundary", "run_peeking_simulation"]

#: 单次观察的最小单臂样本量。
#: z 统计量用的是正态近似临界值 1.96；样本量太小时 t 分布与正态的差距
#: 会把"未校准"混进窥视效应里，让结论不再干净。
_MIN_LOOK_SIZE = 50


@dataclass
class PeekResult:
    """一组窥视策略在相同数据上的表现对比。"""

    n_looks: int
    n_trials: int
    alpha: float
    naive_fpr: float
    fixed_horizon_fpr: float
    calibrated_fpr: float
    boundary: float
    df: int

    def summary(self) -> str:
        return (
            f"窥视仿真 (n_looks={self.n_looks}, n_trials={self.n_trials:,}, "
            f"名义 alpha={self.alpha})\n"
            f"  只看最后一次 (固定时点)     FPR = {self.fixed_horizon_fpr:.4f}  <- 正确\n"
            f"  每次都判 p<0.05 (朴素窥视)  FPR = {self.naive_fpr:.4f}  "
            f"<- 膨胀 {self.naive_fpr / self.alpha:.1f} 倍\n"
            f"  常数边界 |z|>{self.boundary:.3f}  FPR = {self.calibrated_fpr:.4f}  "
            f"<- 标定后可控\n"
        )

    def to_row(self) -> dict[str, float]:
        return {
            "n_looks": self.n_looks,
            "naive": self.naive_fpr,
            "fixed_horizon": self.fixed_horizon_fpr,
            "calibrated": self.calibrated_fpr,
            "boundary": self.boundary,
        }


def _look_sizes(n_per_arm: int, n_looks: int) -> np.ndarray:
    """把单臂样本切成严格递增的嵌套前缀长度。

    第 k 次"看数据"用的是前 ``sizes[k]`` 个观测 —— 嵌套前缀正是
    窥视问题的根源：相邻两次的 z 统计量高度相关，不是独立检验。

    ``n_looks=1`` 必须退化成"用满全样本看一次"，也就是普通的固定时点检验；
    早期版本用 ``linspace(2, n, 1)`` 只会取到 2 个样本，
    于是单次查看的假阳性率被算成 21.8% —— 那不是窥视效应，是样本量算错了。
    """
    if n_looks < 1:
        raise ValueError("n_looks 必须为正")

    # 第 k 次观察累计 n*k/K 个样本，最后一次正好用满全样本
    sizes = (np.arange(1, n_looks + 1) * (n_per_arm / n_looks)).astype(int)
    sizes = np.minimum(sizes, n_per_arm)
    sizes = np.maximum.accumulate(sizes)

    if sizes.size != n_looks or sizes[0] < _MIN_LOOK_SIZE:
        raise ValueError(
            f"单臂样本量 {n_per_arm} 配 {n_looks} 次观察，最早一次只有 "
            f"{sizes[0]} 个样本（要求 >= {_MIN_LOOK_SIZE}）；"
            "请减少观察次数或增大样本量"
        )
    return sizes


def _running_z_stats(y: np.ndarray, look_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给定单臂样本序列，返回在每个观察时点上的 (均值, 样本方差)。

    用累积和实现"嵌套前缀样本"：第 k 次观察用的是前 ``look_sizes[k]`` 个观测。
    嵌套正是窥视问题的根源 —— 相邻两次的 z 统计量高度相关。
    """
    cs = np.cumsum(y)
    cs2 = np.cumsum(y * y)
    n = look_sizes.astype(float)
    mean = cs[look_sizes - 1] / n
    var = (cs2[look_sizes - 1] - n * mean * mean) / (n - 1)
    return mean, var


def _simulate_z_sequence(
    n_trials: int,
    n_per_arm: int,
    n_looks: int,
    *,
    true_lift: float,
    post_sd: float,
    seed: int,
) -> np.ndarray:
    """返回形状 (n_trials, n_looks) 的 z 统计量序列。"""
    rng = np.random.default_rng(seed)
    look_sizes = _look_sizes(n_per_arm, n_looks)

    z = np.empty((n_trials, n_looks))
    for i in range(n_trials):
        t_obs = rng.normal(true_lift, post_sd, n_per_arm)
        c_obs = rng.normal(0.0, post_sd, n_per_arm)
        mt, vt = _running_z_stats(t_obs, look_sizes)
        mc, vc = _running_z_stats(c_obs, look_sizes)
        nt = look_sizes.astype(float)
        se = np.sqrt(vt / nt + vc / nt)
        z[i] = (mt - mc) / se
    return z


def calibrate_constant_boundary(
    z_under_null: np.ndarray, alpha: float = 0.05
) -> float:
    """在原假设的 z 序列上标定常数边界：使 max|z| 超过它的概率为 alpha。"""
    max_abs = np.max(np.abs(z_under_null), axis=1)
    return float(np.quantile(max_abs, 1.0 - alpha))


def run_peeking_simulation(
    population: Population,
    *,
    n_looks: int = 10,
    n_trials: int = 4_000,
    alpha: float = 0.05,
    n_per_arm: int | None = None,
    seed: int = 101,
) -> PeekResult:
    """在 H0（真实效应为 0）下对比三种窥视策略的实际 I 类错误率。"""
    cfg = population.config
    n_per_arm = n_per_arm or len(population) // 2
    post_sd = cfg.post_sd

    # 用前一半重复标定边界，后一半评估，避免"用同一批数据既定阈值又验阈值"
    half = n_trials // 2
    z_cal = _simulate_z_sequence(
        half, n_per_arm, n_looks, true_lift=0.0, post_sd=post_sd, seed=seed
    )
    z_eval = _simulate_z_sequence(
        n_trials - half, n_per_arm, n_looks, true_lift=0.0, post_sd=post_sd, seed=seed + 1
    )

    boundary = calibrate_constant_boundary(z_cal, alpha)

    max_abs_eval = np.max(np.abs(z_eval), axis=1)
    naive = float(np.mean(max_abs_eval > stats.norm.ppf(1 - alpha / 2)))
    fixed = float(np.mean(np.abs(z_eval[:, -1]) > stats.norm.ppf(1 - alpha / 2)))
    calibrated = float(np.mean(max_abs_eval > boundary))

    return PeekResult(
        n_looks=n_looks,
        n_trials=int(z_eval.shape[0]),
        alpha=alpha,
        naive_fpr=naive,
        fixed_horizon_fpr=fixed,
        calibrated_fpr=calibrated,
        boundary=boundary,
        df=n_per_arm * 2 - 2,
    )
