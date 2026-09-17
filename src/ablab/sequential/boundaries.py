"""群序贯边界：Armitage-McPherson 递归。

模型
----
在信息量比例 ``t_1 < ... < t_K`` 处各看一次数据。用布朗运动表示，
标准化检验统计量满足

    Z_k = B(t_k) / sqrt(t_k),     Corr(Z_i, Z_j) = sqrt(t_i / t_j)  (i < j)

于是 ``Z_k`` 边际服从 ``N(0,1)``，而且**相邻两次高度相关** ——
这正是"多看几次就会假阳性"的根源。

递归
----
设 ``f_k(z)`` 为"到第 k 次为止从未越界、且第 k 次的统计量等于 z"的**子密度**。则

    f_1(z) = phi(z)
    f_k(z) = ∫ f_{k-1}(u) * (1/s) * phi((z - rho*u)/s) du,
             rho = sqrt(t_{k-1}/t_k),  s = sqrt(1 - t_{k-1}/t_k)

``f_k`` 的总质量是 ``1 - alpha*(t_{k-1})``。第 k 次越界的概率就是
``1 - ∫_{-b_k}^{b_k} f_k``，令它等于本次该消耗的 alpha，反解出 ``b_k``。

这是 Armitage-McPherson (1969) / Jennison-Turnbull 的标准做法。
数值上用等距网格 + 梯形法做积分，边界用 ``np.interp`` 在单调的
"内部质量"数组上反查 —— 比逐次跑根搜索快两个数量级。

**核矩阵只与观察时点有关，与 alpha 无关**，所以预计算并缓存。
这一点很关键：调整 p 值要对 alpha 做几十次二分搜索，否则会慢到不可用。

关于边界值与文献对不上
----------------------
本模块实现的是 **Lan-DeMets 消耗函数**版本。它与 1969 年原始 O'Brien-Fleming
边界**不是同一个东西**：原始 OBF 的边界形状是 ``b_k = c / sqrt(t_k)``
（K=5、alpha=0.05 时约为 4.56, 3.23, 2.63, 2.28, 2.04），
而 Lan-DeMets 版本按 ``2(1 - Phi(z_{alpha/2}/sqrt(t)))`` 消耗误差，
数值上略有差异（早期边界更松、末期更紧）。

两者都正确，**Lan-DeMets 更通用**：它允许不等距的观察时点、
允许事后临时加一次查看（只要重新算消耗），这在工程实践里是必需的。
Pocock 的两种定义差别很小（本模块结果 ≈ 2.41 常数，与文献一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from .spending import SpendingFunction, get_spending

__all__ = [
    "BoundarySolver",
    "SequentialDesign",
    "build_design",
    "adjusted_p_value",
    "repeated_ci",
]

#: 缓存核矩阵的内存上限（字节）。观察次数很多时改为就地重建，**不降网格**。
_MAX_KERNEL_BYTES = 150 * 1024 * 1024

#: 网格细化自检的容差：末次边界在两种网格下的相对偏差超过它就判定不可靠
_REFINEMENT_TOL = 0.01


class BoundarySolver:
    """按给定观察时点解群序贯边界（alpha 可在多次求解间变化）。"""

    def __init__(
        self,
        information_fractions: Sequence[float],
        *,
        n_grid: int = 801,
        grid_max: float = 8.0,
    ) -> None:
        t = np.asarray(information_fractions, dtype=float)
        if t.ndim != 1 or t.size < 1:
            raise ValueError("information_fractions 必须是一维非空数组")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"信息量比例必须严格递增，收到 {t}")
        if t[-1] > 1.0 + 1e-12 or t[0] <= 0:
            raise ValueError(f"信息量比例必须落在 (0, 1] 内，收到 {t}")

        self.t = t
        self.n_looks = int(t.size)

        if n_grid % 2 == 0:
            n_grid += 1  # 取奇数，保证 0 落在网格上
        if n_grid < 101:
            raise ValueError(f"网格太粗（n_grid={n_grid}），无法保证精度")

        self.n_grid = int(n_grid)
        self.grid_max = float(grid_max)
        self.z = np.linspace(-grid_max, grid_max, self.n_grid)
        self.h = float(self.z[1] - self.z[0])

        # 梯形法权重
        w = np.full(self.n_grid, self.h)
        w[0] = w[-1] = self.h / 2.0
        self.w = w

        # 核矩阵是否常驻内存。观察次数很多时改成就地重建 ——
        # **不要为此降低网格精度**：网格一粗，几百步卷积的误差会累积到
        # 边界完全失真（K=500 时 FWER 会从 5% 涨到 26%，实测过）。
        n_kernels = max(self.n_looks - 1, 0)
        self.kernels_cached = n_kernels * self.n_grid**2 * 8 <= _MAX_KERNEL_BYTES
        self._kernels = (
            [self._build_kernel(k) for k in range(1, self.n_looks)]
            if self.kernels_cached
            else None
        )

    def _kernel(self, k: int) -> np.ndarray:
        if self._kernels is not None:
            return self._kernels[k - 1]
        return self._build_kernel(k)

    def _build_kernel(self, k: int) -> np.ndarray:
        """从第 k-1 次到第 k 次的转移核，形状 (n_grid, n_grid)。"""
        rho = np.sqrt(self.t[k - 1] / self.t[k])
        s = np.sqrt(1.0 - self.t[k - 1] / self.t[k])
        # kernel[i, j] = (1/s) * phi((z_j - rho * z_i) / s)
        return stats.norm.pdf((self.z[None, :] - rho * self.z[:, None]) / s) / s

    # -- 求解 -------------------------------------------------------------- #
    def solve(self, spending: np.ndarray) -> np.ndarray:
        """给定各次查看后**累计**消耗的 alpha，返回对称边界 ``b_k``。"""
        spend = np.asarray(spending, dtype=float)
        if spend.shape != self.t.shape:
            raise ValueError(
                f"spending 长度 {spend.shape} 与观察次数 {self.t.shape} 不一致"
            )
        if np.any(np.diff(spend) < -1e-12):
            raise ValueError("累计消耗的 alpha 必须单调不减")
        if spend[0] < 0 or spend[-1] > 1.0 + 1e-12:
            raise ValueError(f"累计消耗的 alpha 必须落在 [0, 1]，收到 {spend}")

        bounds = np.empty(self.n_looks)
        density = stats.norm.pdf(self.z)

        # 第一次查看有解析解
        b0 = float(stats.norm.isf(spend[0] / 2.0))
        bounds[0] = b0
        density = np.where(np.abs(self.z) <= b0, density, 0.0)

        for k in range(1, self.n_looks):
            density = (self.w * density) @ self._kernel(k)
            # 需要 ∫_{-b}^{b} f_k = 1 - alpha*(t_k)
            inner = self._inner_mass(density)
            target = 1.0 - spend[k]
            b = float(np.interp(target, inner, self.z))
            bounds[k] = b
            density = np.where(np.abs(self.z) <= b, density, 0.0)

        return bounds

    def is_reliable(self, bounds: np.ndarray) -> bool:
        """自检：末次边界是否落到了网格边缘。

        只检查**末次**边界。早期边界超出网格（甚至为 ``inf``）是**正确**的 ——
        OBF 在观察次数多时前期消耗的 alpha 小到浮点下溢，
        对应的边界本就该是"这次不可能拒绝"。把它当成错误会误报。

        而末次边界一旦撞到边缘，说明目标质量 ``1 - alpha`` 在网格范围内根本没达到，
        后面的结果不可信。观察次数极多时会出现（均匀网格 + 几百步卷积的固有局限）。
        """
        bounds = np.asarray(bounds, dtype=float)
        last = float(bounds[-1])
        if not np.isfinite(last):
            return False
        return last < self.grid_max - 2 * self.h

    def reliability_note(self, bounds: np.ndarray) -> str:
        if self.is_reliable(bounds):
            return "ok"
        return (
            "不可靠：数值边界撞到了网格边缘。均匀网格 + 几百步卷积的误差会累积，"
            "此时应减少观察次数，或改用 mSPRT（它本来就为连续监控设计）"
        )

    def _inner_mass(self, density: np.ndarray) -> np.ndarray:
        """``∫_{-z_j}^{z_j} f``：网格关于 0 对称，用镜像累加即可。"""
        cum = np.concatenate(
            [[0.0], np.cumsum((density[:-1] + density[1:]) * 0.5 * self.h)]
        )
        return cum - cum[::-1]

    def exit_probabilities(self, bounds: np.ndarray) -> np.ndarray:
        """给定边界，返回各次查看的**单次**越界概率（用于自检）。

        注意返回的是**增量** ``P(第 k 次首次越界)``，不是累计，也不是
        "第 k 次的边际越界概率"。三者的区别正是窥视问题的全部内容：
        把边际概率加起来会重复计数，从而高估真实的 I 类错误。
        """
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != self.t.shape:
            raise ValueError("bounds 长度与观察次数不一致")

        out = np.empty(self.n_looks)
        density = stats.norm.pdf(self.z)
        for k in range(self.n_looks):
            if k > 0:
                density = (self.w * density) @ self._kernel(k)
            mass_before = float(density @ self.w)  # = 1 - spend[k-1]
            inner = self._inner_mass(density)
            kept = float(np.interp(bounds[k], self.z, inner))
            out[k] = mass_before - kept
            density = np.where(np.abs(self.z) <= bounds[k], density, 0.0)
        return out


# --------------------------------------------------------------------------- #
# 设计
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SequentialDesign:
    """一个完整的群序贯设计。"""

    alpha: float
    spending_name: str
    information_fractions: np.ndarray
    boundaries: np.ndarray
    cumulative_spend: np.ndarray
    #: 数值求解是否可靠（边界没撞到网格边缘，且网格细化后结果一致）
    reliable: bool = True
    #: 一半网格重解时的末次边界相对偏差
    refinement_error: float = float("nan")

    @property
    def n_looks(self) -> int:
        return int(self.boundaries.size)

    @property
    def final_boundary(self) -> float:
        return float(self.boundaries[-1])

    @property
    def final_nominal_p(self) -> float:
        """末次边界对应的**名义** p 值 —— 与 0.05 比会低估显著性门槛。"""
        return float(2 * stats.norm.sf(self.final_boundary))

    def boundary_at(self, look: int) -> float:
        """第 ``look`` 次查看的边界（``look`` 从 1 开始计数）。"""
        if not 1 <= look <= self.n_looks:
            raise IndexError(f"look 必须在 1~{self.n_looks}")
        return float(self.boundaries[look - 1])

    def crossed(self, z_statistics: Sequence[float]) -> np.ndarray:
        """判断是否越界，返回布尔数组。

        支持两种输入：

        * 一维 ``(n_looks,)`` —— 单条实验路径，返回 ``(n_looks,)``
        * 二维 ``(n_trials, n_looks)`` —— 批量仿真，返回同形状的布尔矩阵
        """
        z = np.asarray(z_statistics, dtype=float)
        if z.ndim == 1:
            if z.size != self.n_looks:
                raise ValueError(f"需要 {self.n_looks} 个 z 值，收到 {z.size}")
            return np.abs(z) >= self.boundaries
        if z.ndim == 2:
            if z.shape[1] != self.n_looks:
                raise ValueError(
                    f"最后一维应为 {self.n_looks}，收到形状 {z.shape}"
                )
            return np.abs(z) >= self.boundaries[None, :]
        raise ValueError(f"z_statistics 必须是一维或二维，收到 {z.ndim} 维")

    def reject(self, z_statistics: Sequence[float]) -> bool:
        """整个序贯检验是否在任一时刻拒绝原假设。"""
        return bool(self.crossed(z_statistics).any())

    def first_crossing_look(self, z_statistics: Sequence[float]) -> int | None:
        """首次越界发生在第几次查看；从未越界返回 ``None``。"""
        hits = np.flatnonzero(self.crossed(z_statistics))
        return int(hits[0]) + 1 if hits.size else None

    def summary(self) -> str:
        lines = [
            f"群序贯设计：{self.spending_name}，alpha={self.alpha}，"
            f"{self.n_looks} 次查看",
            f"{'查看':>4} {'信息量t':>9} {'边界b':>9} {'名义p':>10} {'累计消耗':>10}",
        ]
        for k in range(self.n_looks):
            nominal = float(2 * stats.norm.sf(self.boundaries[k]))
            lines.append(
                f"{k + 1:>4} {self.information_fractions[k]:>9.3f} "
                f"{self.boundaries[k]:>9.4f} {nominal:>10.6f} "
                f"{self.cumulative_spend[k]:>10.6f}"
            )
        lines.append(
            f"  末次边界 {self.final_boundary:.4f}（名义 p={self.final_nominal_p:.4f}），"
            f"比固定时点的 1.96 严 —— 这就是为「看过多次」付出的代价"
        )
        if not self.reliable:
            lines.append(
                "  **数值不可靠**：边界撞到网格边缘。均匀网格 + 几百步卷积的误差会累积，"
                "请减少观察次数，或改用 mSPRT。"
            )
        return "\n".join(lines)


def build_design(
    *,
    alpha: float = 0.05,
    n_looks: int = 5,
    spending: str | SpendingFunction = "obf",
    information_fractions: Sequence[float] | None = None,
    n_grid: int | None = None,
    solver: BoundarySolver | None = None,
    check_refinement: bool = True,
) -> SequentialDesign:
    """构建一个群序贯设计。

    ``information_fractions`` 默认是等距的 ``1/K, 2/K, ..., 1``。
    真实场景里它应该由**信息量**决定（累计样本量之比，或 1/方差之比），
    而不是日历时间之比。
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha}")
    if n_looks < 1:
        raise ValueError(f"n_looks 必须为正，收到 {n_looks}")

    if information_fractions is None:
        t = np.arange(1, n_looks + 1) / n_looks
        n_looks = int(t.size)
    else:
        t = np.asarray(information_fractions, dtype=float)
        n_looks = int(t.size)

    fn = get_spending(spending)
    if solver is None:
        solver = BoundarySolver(t) if n_grid is None else BoundarySolver(t, n_grid=n_grid)
    spend = fn(t, alpha)
    bounds = solver.solve(spend)

    reliable = solver.is_reliable(bounds)
    refinement_error = float("nan")
    if reliable and check_refinement:
        # 网格细化自检：用一半网格重解一次。两种网格给出的末次边界应当一致；
        # 不一致说明离散误差已经主导结果（观察次数很多时会发生）。
        # 只查网格边缘会漏掉这种失效 —— 它表现为边界**偏小**而不是撞边缘。
        coarse = BoundarySolver(t, n_grid=max(101, solver.n_grid // 2))
        coarse_bounds = coarse.solve(spend)
        denom = max(abs(float(bounds[-1])), 1e-12)
        refinement_error = abs(float(coarse_bounds[-1]) - float(bounds[-1])) / denom
        reliable = refinement_error < _REFINEMENT_TOL

    return SequentialDesign(
        alpha=float(alpha),
        spending_name=fn.name,
        information_fractions=t,
        boundaries=bounds,
        cumulative_spend=spend,
        reliable=reliable,
        refinement_error=refinement_error,
    )


# --------------------------------------------------------------------------- #
# 分析工具
# --------------------------------------------------------------------------- #
def repeated_ci(
    estimate: float,
    std_error: float,
    boundary: float,
) -> tuple[float, float]:
    """重复置信区间：``estimate ± boundary * std_error``。

    覆盖率是 ``1 - alpha*(t_k)``，**不是** ``1 - alpha`` ——
    早期查看的区间会宽得离谱（OBF 第一次几乎无穷宽），
    这是"任何时刻都有效"必须付出的代价。别把早期区间当成正式结论。
    """
    if std_error < 0:
        raise ValueError("标准误不能为负")
    half = boundary * std_error
    return (estimate - half, estimate + half)


def adjusted_p_value(
    z_statistics: Sequence[float],
    *,
    information_fractions: Sequence[float] | None = None,
    spending: str | SpendingFunction = "obf",
    solver: BoundarySolver | None = None,
    tol: float = 1e-10,
) -> float:
    """群序贯的调整 p 值。

    定义为**最小的 alpha，使得该序贯设计会在当前或之前的某次查看上拒绝**。
    于是它可以直接和 0.05 比较，而名义 p 值不行。

    注意：这个值依赖于**事先声明的观察次数与消耗函数**。
    如果你实际上看了 10 次却按 5 次的设计算，它就无效了。
    """
    z = np.asarray(z_statistics, dtype=float)
    t = (
        np.asarray(information_fractions, dtype=float)
        if information_fractions is not None
        else np.arange(1, z.size + 1) / z.size
    )
    if t.size != z.size:
        raise ValueError("information_fractions 与 z_statistics 长度不一致")

    fn = get_spending(spending)
    solver = solver or BoundarySolver(t)

    def crosses(a: float) -> bool:
        return bool(np.any(np.abs(z) >= solver.solve(fn(t, a))))

    lo, hi = 1e-13, 1.0 - 1e-9
    if crosses(lo):
        return float(lo)
    if not crosses(hi):
        return 1.0

    # alpha 跨十几个数量级，用对数二分
    for _ in range(80):
        if hi - lo < tol:
            break
        mid = float(np.sqrt(lo * hi))
        if crosses(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)
