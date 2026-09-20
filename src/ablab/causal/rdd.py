"""断点回归（RDD）：断点两侧的**局部随机化**，以及它的两个经典陷阱。

为什么它与 IV 是两套东西
------------------------
IV 买的是"工具与误差无关"，RDD 买的是"**在断点附近，谁落在哪一侧近似随机**"。
后者的可信度完全取决于两件事，而它们都不是估计量能自己回答的：

  1. **带宽**：断点附近的函数形式未知，离断点越远线性近似越差（偏差），
     越近样本越少（方差）。所以带宽选择是这一节的主战场，
     而且它的好坏只能用**覆盖率**来判，不能用公式好不好看来判。
  2. **操纵**：如果单元能选择自己落在断点哪一侧（成绩刚好过线、指标刚好达标），
     "局部随机化"就不成立了 —— 而且这时**数据本身会留下痕迹**：
     断点两侧的密度不连续。所以配一个操纵检验。

三个实现
--------
* ``sharp_rdd``：核加权局部线性回归（均匀/三角/Epanechnikov），HC1 稳健方差。
  两侧用的是**不相交的样本**，所以两个截距的协方差真的是 0 —— 这个仓库在别处
  反复栽在"把相关的方差按独立量合成"，这里是少数几处独立性真的成立的地方
  （值得写下来，因为它不是一个可以推广的结论）。
* ``mse_optimal_bandwidth``：IK 式 plug-in 带宽。核矩 ``Γ``、``Λ``、``V``
  用**数值积分**算，偏差常数与方差常数由它们推出：

      bias(τ̂) = β₂ · B,      B = e₀' Γ⁻¹ Λ e₀
      Var(τ̂)  ∝ Vc / (n h),  Vc = e₀' Γ⁻¹ V Γ⁻¹ e₀
      h_MSE   = [ (σ²₊/f₊ + σ²₋/f₋) · Vc / (m''² B²) ]^{1/5} · n^{-1/5}

  推导写在代码里，常数**不抄论文** —— 抄错了没有任何东西会报错。
* ``cct_robust_ci``：偏差校正 + 只保留线性项的"稳健"方差（CCT 2014 的思路）。
  朴素局部线性的区间在 MSE 最优带宽下会欠覆盖：那个带宽是给点估计的 MSE 用的，
  此时偏差已经与方差同量级。校正之后区间不会因为减偏差而变宽太多。

再加一个模糊断点（``fuzzy_rdd``）：断点只改变**接受处置的概率**时的 Wald 比。
它顺带把 ITT 与 LATE 的区别量出来 —— 前者是"有意稀释"过的效应，
"断点显著"与"处置有效"不是同一句话。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

__all__ = [
    "KERNELS",
    "FuzzyRDDResult",
    "ManipulationTest",
    "RDDResult",
    "bandwidth_sensitivity",
    "cct_robust_ci",
    "fuzzy_rdd",
    "kernel_weights",
    "manipulation_test",
    "mse_optimal_bandwidth",
    "sharp_rdd",
]

#: 核函数：输入标准化距离 ``u = (x - c) / h``，返回权重（支撑在 |u| ≤ 1）。
KERNELS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "uniform": lambda u: np.where(np.abs(u) <= 1.0, 0.5, 0.0),
    "triangular": lambda u: np.where(np.abs(u) <= 1.0, 1.0 - np.abs(u), 0.0),
    "epanechnikov": lambda u: np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0),
}


def kernel_weights(u: np.ndarray, kernel: str = "triangular") -> np.ndarray:
    if kernel not in KERNELS:
        raise ValueError(f"未知核 {kernel!r}，可选 {sorted(KERNELS)}")
    return KERNELS[kernel](np.asarray(u, dtype=float))


# --------------------------------------------------------------------------- #
# 核矩（数值积分）与两个常数
# --------------------------------------------------------------------------- #
def _kernel_moments(kernel: str, *, order: int = 1, n_grid: int = 8001) -> dict:
    """核矩矩阵（在 ``[0, 1]`` 上积分 —— 这是个**边界问题**，不是内部问题）。

    这一点是这个模块第一版写错的地方，值得写在代码里而不是留在注释外：
    断点的**每一侧**都只有单边数据，所以局部线性拟合是"边界上的局部线性"，
    核矩必须在 ``[0, 1]`` 上积，而不是在对称的 ``[-1, 1]`` 上积。
    实测差多少：三角核的方差常数 ``Vc`` 在两种口径下是 **4.8 : 0.667**（7.2 倍），
    也就是 SE 差 2.7 倍 —— 第一版按内部问题算，理论 SE 比实测小了近 3 倍。

    * ``Gamma[a, b] = ∫₀¹ K(u) u^(a+b) du``     —— 设计矩阵的极限
    * ``Lambda[a, b] = ∫₀¹ K(u) u^(a+b+2) du``  —— 偏差项的极限
    * ``V[a, b] = ∫₀¹ K(u)² u^(a+b) du``        —— 方差项的极限
    """
    u = np.linspace(0.0, 1.0, n_grid)
    k = kernel_weights(u, kernel)
    idx = np.arange(order + 1)

    def moment(power: int, weight: np.ndarray) -> np.ndarray:
        vals = [np.trapezoid(weight * u ** (a + b + power), u) for a in idx for b in idx]
        return np.asarray(vals).reshape(order + 1, order + 1)

    return {"Gamma": moment(0, k), "Lambda": moment(2, k), "V": moment(0, k**2)}


def local_linear_constants(kernel: str = "triangular") -> tuple[float, float]:
    """局部线性截距的偏差常数 ``B`` 与方差常数 ``Vc``（由核矩推出）。

    偏差：``E[τ̂] − τ = (h²/2)·(m''₊(c) − m''₋(c))·B``，而局部二次拟合里
    ``m''(c) = 2β₂/h²``，所以 ``bias = (β₂₊ − β₂₋) · B``（代码里就是这么用的）。
    方差：``Var(τ̂) = Vc · (σ²₊/f₊ + σ²₋/f₋) / (n h)``。

    三角核下的解析值（可以手算核对）：``B = −0.1``，``Vc = 4.8``。
    """
    mom = _kernel_moments(kernel, order=1)
    g_inv = np.linalg.pinv(mom["Gamma"])
    e0 = np.array([1.0, 0.0])
    bias_const = float(e0 @ g_inv @ mom["Lambda"] @ e0)
    var_const = float(e0 @ g_inv @ mom["V"] @ g_inv @ e0)
    return bias_const, var_const


# --------------------------------------------------------------------------- #
# 局部多项式拟合
# --------------------------------------------------------------------------- #
def _local_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cutoff: float,
    bandwidth: float,
    kernel: str,
    order: int,
) -> dict:
    """一侧的核加权局部多项式：截距、稳健方差、影响函数、有效样本量。"""
    u = (x - cutoff) / bandwidth
    w = kernel_weights(u, kernel)
    keep = w > 0
    if keep.sum() < order + 2:
        raise ValueError(f"带宽 {bandwidth:g} 内的样本不足（{int(keep.sum())} 个）")
    u, w, y = u[keep], w[keep], y[keep]
    # 设计矩阵用标准化距离 1, u, u², ...：截距就是断点处的函数值（不是近似）
    M = np.column_stack([u**j for j in range(order + 1)])
    MtWM = M.T @ (M * w[:, None])
    MtWM_inv = np.linalg.pinv(MtWM)
    beta = MtWM_inv @ (M.T @ (w * y))
    resid = y - M @ beta
    # HC1 三明治：中间那层用 w²e²（权重已经在估计里用过一次）
    meat = M.T @ (M * (w**2 * resid**2)[:, None])
    cov = MtWM_inv @ meat @ MtWM_inv
    psi = MtWM_inv @ (M.T * w)  # (order+1, n)：β = ψ y
    return {
        "intercept": float(beta[0]),
        "coefs": beta,
        "var": float(cov[0, 0]),
        "psi": psi,
        "resid": resid,
        "weights": w,
        "n_eff": float(w.sum() ** 2 / (w**2).sum()),
        "n": int(keep.sum()),
    }


# --------------------------------------------------------------------------- #
# 尖锐断点
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RDDResult:
    """一次断点估计的全部读数。"""

    tau: float
    se: float
    alpha: float
    bandwidth: float
    kernel: str
    order: int
    n_left: int
    n_right: int
    n_eff_left: float
    n_eff_right: float
    bias_corrected: bool = False
    bias_estimate: float = 0.0

    @property
    def z(self) -> float:
        return self.tau / self.se if self.se > 0 else float("nan")

    @property
    def p_value(self) -> float:
        return float(2.0 * stats.norm.sf(abs(self.z)))

    @property
    def ci(self) -> tuple[float, float]:
        crit = stats.norm.ppf(1.0 - self.alpha / 2.0)
        return (self.tau - crit * self.se, self.tau + crit * self.se)

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci
        return bool(lo <= truth <= hi)

    def summary(self) -> str:
        lo, hi = self.ci
        tag = "偏差校正 + 稳健方差" if self.bias_corrected else "朴素局部线性"
        return (
            f"断点估计（{tag}，{self.kernel} 核，阶数 {self.order}，"
            f"h={self.bandwidth:.4f}）\n"
            f"  τ̂ = {self.tau:+.4f}（SE {self.se:.4f}，"
            f"95% 区间 [{lo:+.4f}, {hi:+.4f}]，p = {self.p_value:.4g}）\n"
            f"  有效样本：左 {self.n_eff_left:.1f} / 右 {self.n_eff_right:.1f}"
            f"（原始 {self.n_left} / {self.n_right}）"
        )


def sharp_rdd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
    order: int = 1,
    alpha: float = 0.05,
) -> RDDResult:
    """尖锐断点：处置在断点处从 0 跳到 1，估计断点处的跳跃。

    ``bandwidth=None`` 时用 ``mse_optimal_bandwidth`` 的 plug-in 结果。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x / y 的样本量必须一致")
    if bandwidth is None:
        bandwidth = mse_optimal_bandwidth(x, y, cutoff=cutoff, kernel=kernel)
    band = float(bandwidth)
    if not np.isfinite(band) or band <= 0:
        raise ValueError("带宽必须是正的有限数")
    left, right = x < cutoff, x >= cutoff
    fit_l = _local_fit(x[left], y[left], cutoff=cutoff, bandwidth=band,
                       kernel=kernel, order=order)
    fit_r = _local_fit(x[right], y[right], cutoff=cutoff, bandwidth=band,
                       kernel=kernel, order=order)
    tau = fit_r["intercept"] - fit_l["intercept"]
    # 两侧样本不相交 ⇒ 协方差为 0（这里是真的，不是假设）
    var = fit_l["var"] + fit_r["var"]
    return RDDResult(
        tau=float(tau),
        se=float(np.sqrt(var)),
        alpha=alpha,
        bandwidth=band,
        kernel=kernel,
        order=order,
        n_left=fit_l["n"],
        n_right=fit_r["n"],
        n_eff_left=fit_l["n_eff"],
        n_eff_right=fit_r["n_eff"],
    )


# --------------------------------------------------------------------------- #
# 带宽选择（IK 式 plug-in）
# --------------------------------------------------------------------------- #
def mse_optimal_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cutoff: float = 0.0,
    kernel: str = "triangular",
    pilot_scale: float = 1.0,
) -> float:
    """IK 式 plug-in 带宽：先估曲率与条件方差，再代进 MSE 最优的 1/5 次方。

    诚实说明：这是 IK 的**同族**公式，不是逐式复刻（IK 对两侧的样本份额与
    密度的处理更细）。它好不好用，由审计里的覆盖率说话 —— 见 m3 报告 2.9 节。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 50:
        raise ValueError("样本量太小，估不出带宽")
    sd_x = float(np.std(x))
    if sd_x <= 0:
        raise ValueError("断点变量没有变异")
    h_pilot = pilot_scale * 1.84 * sd_x * n ** (-1.0 / 5.0)

    bias_const, var_const = local_linear_constants(kernel)
    curves, sigmas, densities = [], [], []
    for mask in (x < cutoff, x >= cutoff):
        xs, ys = x[mask], y[mask]
        if xs.size < 30:
            raise ValueError("断点一侧的样本太少")
        u = (xs - cutoff) / h_pilot
        w = kernel_weights(u, kernel)
        keep = w > 0
        if keep.sum() < 20:
            raise ValueError("pilot 带宽内样本太少")
        M = np.column_stack([np.ones(int(keep.sum())), u[keep], u[keep] ** 2])
        ww = w[keep]
        coef = np.linalg.pinv(M.T @ (M * ww[:, None])) @ (M.T @ (ww * ys[keep]))
        # m''(c) = 2β₂ / h²（x 的单位）
        curves.append(2.0 * float(coef[2]) / h_pilot**2)
        resid = ys[keep] - M @ coef
        dof = max(float(ww.sum()) - 3.0, 1.0)
        sigmas.append(float(np.sum(ww * resid**2) / dof))
        densities.append(float(np.sum(ww)) / (xs.size * h_pilot))

    curvature = curves[1] - curves[0]
    fallback = 1.84 * sd_x * n ** (-1.0 / 5.0)
    if abs(curvature) < 1e-12:
        # 曲率估不出来（两侧几乎是同一条直线）时退回一个温和的规则：
        # 这时 MSE 最优带宽在理论上趋于无穷，但"趋于无穷"不是能用的答案。
        return float(fallback)
    numer = sigmas[0] / max(densities[0], 1e-12) + sigmas[1] / max(densities[1], 1e-12)
    h = (numer * var_const / (curvature**2 * bias_const**2)) ** 0.2 * n ** (-0.2)
    return float(np.clip(h, 0.05 * sd_x, 2.0 * sd_x))


# --------------------------------------------------------------------------- #
# 偏差校正 + 稳健方差（CCT 思路）
# --------------------------------------------------------------------------- #
def cct_robust_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
) -> RDDResult:
    """偏差校正的断点估计 + 只保留线性项的稳健方差。

    偏差用两侧局部二次拟合的 β₂ 之差估：``bias = (β₂₊ − β₂₋) · B``。

    为什么"两侧之差"才是对的：断点两侧各做一次局部线性，各自的边界偏差
    在 τ̂ 里相减 —— 曲率相同的那部分**自己抵消掉了**，只剩曲率之差。
    这也解释了为什么这个审计的 DGP 要让两侧曲率不同：
    曲率一样时 RDD 的局部线性估计几乎没有一阶偏差，"带宽太宽会出事"
    这句话在那个 DGP 上根本量不出来。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if bandwidth is None:
        bandwidth = mse_optimal_bandwidth(x, y, cutoff=cutoff, kernel=kernel)
    band = float(bandwidth)
    bias_const, _ = local_linear_constants(kernel)
    left, right = x < cutoff, x >= cutoff
    lin_l = _local_fit(x[left], y[left], cutoff=cutoff, bandwidth=band,
                       kernel=kernel, order=1)
    lin_r = _local_fit(x[right], y[right], cutoff=cutoff, bandwidth=band,
                       kernel=kernel, order=1)
    quad_l = _local_fit(x[left], y[left], cutoff=cutoff, bandwidth=band,
                        kernel=kernel, order=2)
    quad_r = _local_fit(x[right], y[right], cutoff=cutoff, bandwidth=band,
                        kernel=kernel, order=2)
    bias = bias_const * (float(quad_r["coefs"][2]) - float(quad_l["coefs"][2]))
    tau_naive = lin_r["intercept"] - lin_l["intercept"]
    var = lin_l["var"] + lin_r["var"]
    return RDDResult(
        tau=float(tau_naive - bias),
        se=float(np.sqrt(var)),
        alpha=alpha,
        bandwidth=band,
        kernel=kernel,
        order=1,
        n_left=lin_l["n"],
        n_right=lin_r["n"],
        n_eff_left=lin_l["n_eff"],
        n_eff_right=lin_r["n_eff"],
        bias_corrected=True,
        bias_estimate=float(bias),
    )


def bandwidth_sensitivity(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cutoff: float = 0.0,
    kernel: str = "triangular",
    factors: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> list[tuple[float, RDDResult]]:
    """同一份数据在若干个带宽上的读数 —— 带宽依赖性是这一节的常规体检。"""
    h0 = mse_optimal_bandwidth(x, y, cutoff=cutoff, kernel=kernel)
    return [
        (h0 * f, sharp_rdd(x, y, cutoff=cutoff, bandwidth=h0 * f, kernel=kernel))
        for f in factors
    ]


# --------------------------------------------------------------------------- #
# 模糊断点
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FuzzyRDDResult:
    """模糊断点：断点改变的是**接受处置的概率**。"""

    tau_itt: float  # 断点对 Y 的跳跃（意向处置效应）
    jump_d: float  # 断点对 D 的跳跃（第一阶段）
    tau_late: float  # Wald 比 = ITT / 第一阶段
    se_late: float
    alpha: float
    bandwidth: float
    kernel: str
    #: 第一阶段跳跃的 t 值（工具强度的一个读数）；默认放在最后，便于旧调用
    t_first_stage: float = float("nan")

    @property
    def p_value(self) -> float:
        z = self.tau_late / self.se_late if self.se_late > 0 else float("nan")
        return float(2.0 * stats.norm.sf(abs(z)))

    @property
    def ci(self) -> tuple[float, float]:
        crit = stats.norm.ppf(1.0 - self.alpha / 2.0)
        return (self.tau_late - crit * self.se_late, self.tau_late + crit * self.se_late)

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci
        return bool(lo <= truth <= hi)

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"模糊断点（{self.kernel} 核，h={self.bandwidth:.4f}）\n"
            f"  ITT（断点对 Y 的跳跃）= {self.tau_itt:+.4f}\n"
            f"  第一阶段（断点对 D 的跳跃）= {self.jump_d:+.4f}"
            f"（t = {self.t_first_stage:+.2f}，合规份额的一个读数）\n"
            f"  LATE = ITT / 第一阶段 = {self.tau_late:+.4f}"
            f"（SE {self.se_late:.4f}，95% 区间 [{lo:+.4f}, {hi:+.4f}]，"
            f"p = {self.p_value:.4g}）"
        )


def fuzzy_rdd(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
) -> FuzzyRDDResult:
    """模糊断点 = 局部 Wald 比（= 局部 2SLS，工具是 ``1{x ≥ c}``）。

    SE 用 delta 方法，而且**要把两个跳跃的协方差算进去**：Y 与 D 来自同一批
    单元，同侧的两个局部线性截距天然相关（两侧之间才是真正独立的）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    if not (x.size == y.size == d.size):
        raise ValueError("x / y / d 的样本量必须一致")
    if bandwidth is None:
        bandwidth = mse_optimal_bandwidth(x, y, cutoff=cutoff, kernel=kernel)
    band = float(bandwidth)
    sides = (x < cutoff, x >= cutoff)
    fits: dict[str, list[dict]] = {"y": [], "d": []}
    for tag, vec in (("y", y), ("d", d)):
        for mask in sides:
            fits[tag].append(
                _local_fit(x[mask], vec[mask], cutoff=cutoff, bandwidth=band,
                           kernel=kernel, order=1)
            )
    itt = fits["y"][1]["intercept"] - fits["y"][0]["intercept"]
    jump_d = fits["d"][1]["intercept"] - fits["d"][0]["intercept"]
    se_jump_d = float(np.sqrt(sum(f["var"] for f in fits["d"])))
    # 判据是**统计上**的，不是"恰好等于 0"：随机 D 也会给出一个 ±0.02 的跳跃。
    # 第一阶段不显著时 Wald 比的分母就是噪声，比值会炸到任意大 —— 必须拒绝，
    # 否则报告里会出现一个 SE 巨大或荒谬的 LATE。
    # 门槛取 5% 显著性（而不是更严的 3 个 SE）：这只是**最低限度的可用性检查**，
    # 不是"工具够不够强"的判据 —— 弱第一阶段的代价是 SE 变大，由区间如实反映。
    if abs(jump_d) < max(1e-8, 1.96 * se_jump_d):
        raise ValueError(
            f"第一阶段跳跃不显著（{jump_d:+.4f}，SE {se_jump_d:.4f}）："
            "断点没有可信地改变处置概率，Wald 比没有意义"
        )

    var_a = sum(f["var"] for f in fits["y"])
    var_b = sum(f["var"] for f in fits["d"])
    cov_ab = 0.0
    for side in (0, 1):
        fy, fd = fits["y"][side], fits["d"][side]
        # 同一侧的协方差：ψ_Y diag(e_Y e_D) ψ_D'（ψ 里已经含权重，不能再乘 w²）
        cov_ab += float(np.sum(fy["psi"][0, :] * fy["resid"] * fd["resid"] * fd["psi"][0, :]))
    tau = itt / jump_d
    # delta 方法：τ = a / b
    var_tau = (var_a - 2.0 * tau * cov_ab + tau**2 * var_b) / jump_d**2
    return FuzzyRDDResult(
        tau_itt=float(itt),
        jump_d=float(jump_d),
        tau_late=float(tau),
        se_late=float(np.sqrt(max(var_tau, 0.0))),
        t_first_stage=float(jump_d / se_jump_d) if se_jump_d > 0 else float("nan"),
        alpha=alpha,
        bandwidth=band,
        kernel=kernel,
    )


# --------------------------------------------------------------------------- #
# 操纵检验（McCrary 思路）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManipulationTest:
    """断点两侧的密度是否连续 —— 单元能不能选边站的痕迹。"""

    log_jump: float
    se: float
    bandwidth: float
    n_bins: int
    alpha: float = 0.05

    @property
    def z(self) -> float:
        return self.log_jump / self.se if self.se > 0 else float("nan")

    @property
    def p_value(self) -> float:
        return float(2.0 * stats.norm.sf(abs(self.z)))

    @property
    def rejects(self) -> bool:
        return bool(self.p_value < self.alpha)

    def summary(self) -> str:
        verdict = "拒绝「密度连续」" if self.rejects else "不能拒绝「密度连续」"
        return (
            f"操纵检验（h={self.bandwidth:.3f}，{self.n_bins} 个箱）\n"
            f"  断点两侧的对数密度跳跃 = {self.log_jump:+.4f}（SE {self.se:.4f}，"
            f"z = {self.z:+.2f}，p = {self.p_value:.4g}）\n"
            f"  判定：{verdict}"
            "（拒绝 = 有人能选边站，断点设计不可信）"
        )


def manipulation_test(
    x: np.ndarray,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    n_bins: int = 20,
    alpha: float = 0.05,
) -> ManipulationTest:
    """McCrary 式的密度跳跃检验（分箱 + 局部线性，Poisson 权重）。

    把断点附近的样本分成等宽箱，对**箱内计数**在两侧各做一次局部线性拟合，
    看断点处的跳跃。``Var(log count) ≈ 1/count``，所以权重取 count，
    而协方差就是 ``(M'WM)^{-1}``（Poisson 近似，McCrary 原文的路子，
    不是精确似然）。用残差三明治估方差会**低估**它 —— 这一点是实测出来的
    （误报率 0.1300 → 0.0600），不是推导出来的。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if bandwidth is None:
        bandwidth = float(1.84 * np.std(x) * n ** (-1.0 / 5.0))
    band = float(bandwidth)
    edges = np.linspace(cutoff - band, cutoff + band, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    counts = np.maximum(counts, 1)  # 空箱给极小权重，不改变拟合量级
    log_c = np.log(counts.astype(float))
    w = counts.astype(float)
    u = (mids - cutoff) / band
    keep = np.abs(u) <= 1.0

    def fit(side_mask: np.ndarray) -> tuple[float, float]:
        m = side_mask & keep
        if m.sum() < 3:
            raise ValueError("断点一侧的箱数太少，无法做操纵检验")
        M = np.column_stack([np.ones(int(m.sum())), u[m]])
        W = w[m]
        MtWM_inv = np.linalg.pinv(M.T @ (M * W[:, None]))
        coef = MtWM_inv @ (M.T @ (W * log_c[m]))
        # 方差用 **Poisson 理论** 而不是残差三明治：log 计数的方差 ≈ 1/count，
        # 权重取 count 时协方差恰好是 (M'WM)^{-1}。第一版按残差算，
        # 实测误报率 0.1300（名义 0.05）—— 分箱后的残差比 Poisson 噪声小，
        # 方差被系统性低估。换成理论方差后同一份数据是 0.0600。
        return float(coef[0]), float(MtWM_inv[0, 0])

    left, var_l = fit(mids < cutoff)
    right, var_r = fit(mids >= cutoff)
    return ManipulationTest(
        log_jump=float(right - left),
        se=float(np.sqrt(var_l + var_r)),
        bandwidth=band,
        n_bins=int(keep.sum()),
        alpha=alpha,
    )
