"""工具变量（2SLS）：处置是"自己选的"时候怎么办。

为什么 M3 需要它
----------------
M3 已有的两件武器都要求**时间或结构**：DiD 要面板（同一批人前后两期）、
合成控制要一条没被处置的对照序列。而有一大类问题是**横截面上的一次性决策**：
"上过培训班的人后来收入更高"—— 没有处置前趋势可用，因为根本没有"处置前"。
更麻烦的是：能力高的人**既**更爱上培训班、**也**更容易拿高收入，
于是 OLS 把"能力"的功劳记在了"培训"头上。这就是内生性。

工具变量是这一类问题的标准答案：找一个只通过"是否受处置"影响结果的变量
（``Z``），用它把处置里**外生**的那一部分撬出来。

**排除性约束不可检验**。它由经济学论证保证（"抽签结果只通过是否入学影响收入"），
不是数据能证明的东西 —— 本模块能做的是把它**说出来**，而不是假装检验过。
这里的数据是造出来的，所以它逐字成立；真实数据里这一条永远要靠论证。

三件必须一起给的东西（少一件就会得到"看起来很专业但错了"的结论）
----------------------------------------------------------------
1. **点估计**：2SLS。恰好识别时它就是 ``Cov(Z, Y) / Cov(Z, D)``；
2. **第一阶段有多强**：F 统计量（本模块给的是对异方差稳健的 Wald F）。
   弱工具下 2SLS 的中位偏差会朝 OLS 靠，而且 **Wald 型区间的覆盖率会崩**
   —— 实测（``reports/m3_validation.md``）F 在 10 以下时覆盖率掉到 0.6 上下；
3. **对弱工具稳健的区间**：Anderson-Rubin。把"β 的每个候选值"当成零假设去
   检验工具与残差的相关性，于是**不依赖第一阶段有多强**。代价是区间可能很宽、
   甚至无界（网格端点仍未被排除）—— 那正是"这批数据说不清"的诚实表达，
   比给一个过窄的 Wald 区间好得多。

关于"F > 10"这条经验法则
------------------------
它来自 Staiger & Stock (1997) 与 Stock-Yogo (2005) 的临界值表，是**经验法则**
而不是定理：真正的临界值取决于工具个数、内生变量个数、以及你容忍多少偏差，
而且它是为**最坏情形**（第一阶段的偏差方向最不利）定的。所以本模块的做法是：
把 F 报出来、把 Wald 与 AR 两条区间都报出来、把"F < 10"标成 ``weak=True``，
**让读者看到结论对工具强度有多敏感**，而不是拿一个阈值当判决。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "IVResult",
    "anderson_rubin_ci",
    "first_stage_f",
    "sargan_test",
    "two_sls",
]

#: "F < 10 就担心弱工具"的来源（经验法则，不是定理）。
WEAK_F_RULE = 10.0


def _as_2d(a: np.ndarray | None, n: int) -> np.ndarray:
    if a is None:
        return np.empty((n, 0))
    arr = np.asarray(a, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS + HC1 稳健协方差。返回 ``(beta, cov, resid)``。"""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = X.T @ (X * (resid**2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    return beta, cov, resid


def _residualize(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """把 ``x`` 各列从 ``m`` 里投影掉（残差化）。

    **不要构造 n×n 投影矩阵**：AR 区间要在网格上反复求值，而这里每一列
    只需要一次 ``O(n·p)`` 的小投影。第一版写成 ``m - X(X'X)⁻¹X'm`` 的显式
    n×n 形式，单次调用就要 40 秒（网格 4001 点 × 18MB 矩阵乘法）——
    审计跑 480 次，直接跑到超时。**"能算"和"能跑完"是两件事。**
    """
    if x.size == 0:
        return m
    x = np.asarray(x, dtype=float)
    return m - x @ (np.linalg.pinv(x.T @ x) @ (x.T @ m))


@dataclass
class IVResult:
    """2SLS 的结果，以及两件必须一起读的诊断。

    ``ci`` 是 Wald 型（对同方差/弱工具都**不**稳健），``ar_ci`` 是
    Anderson-Rubin 型（对弱工具稳健）。**两个都给**，因为它们的差
    恰恰是"工具有多弱"的直接读数。
    """

    beta: float
    se: float
    t: float
    p_value: float
    #: Wald 型区间
    ci: tuple[float, float]
    #: Anderson-Rubin 区间（可能无界；``ar_unbounded`` 标出来）。
    #: 无界时这里的端点是**网格边界**，不是数据的结论 —— 结论是"没排除掉"。
    ar_ci: tuple[float, float] | None
    ar_unbounded: bool
    #: 区间宽度；无界记 ``inf``，空集记 ``nan``
    ar_width: float
    #: 第一阶段：被排除工具上的稳健 Wald F
    first_stage_f: float
    first_stage_coef: float
    first_stage_se: float
    n: int
    n_instruments: int
    #: F < 10 的经验法则（不是定理，见模块文档）
    weak: bool
    alpha: float
    #: 过度识别检验（只有工具多于内生变量时才有）
    sargan_j: float | None = None
    sargan_p: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def wald_covers_ar(self) -> bool:
        """Wald 区间是否盖住 AR 区间 —— 弱工具下通常**不**成立。"""
        if self.ar_ci is None:
            return True
        return self.ar_ci[0] >= self.ci[0] and self.ar_ci[1] <= self.ci[1]

    def summary(self) -> str:
        lines = [
            f"2SLS：β = {self.beta:+.4f}（稳健 SE {self.se:.4f}，"
            f"Wald 95% 区间 [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]）",
            f"  第一阶段：工具系数 {self.first_stage_coef:+.4f}"
            f"（SE {self.first_stage_se:.4f}），稳健 F = {self.first_stage_f:.2f}"
            f"{'（弱工具，F < 10）' if self.weak else ''}",
        ]
        if self.ar_ci is None:
            lines.append("  Anderson-Rubin 区间：**空集** —— 这批数据给不出 95% 可信的 β")
        else:
            bound = "（无界，只有一侧被排除）" if self.ar_unbounded else ""
            lines.append(
                f"  Anderson-Rubin 95% 区间 [{self.ar_ci[0]:+.4f}, {self.ar_ci[1]:+.4f}]"
                f"（宽 {self.ar_width:.4f}）{bound}"
            )
        if self.sargan_p is not None:
            lines.append(
                f"  过度识别检验：J = {self.sargan_j:.3f}（df={self.n_instruments - 1}），"
                f"p = {self.sargan_p:.4f}"
            )
        for note in self.notes:
            lines.append(f"  · {note}")
        return "\n".join(lines)


def first_stage_f(
    d: np.ndarray, z: np.ndarray, x: np.ndarray | None = None
) -> tuple[float, float, float]:
    """第一阶段：``d`` 对工具（与外生变量）回归，返回 ``(F, 工具系数, 其 SE)``。

    恰好一个工具时返回的就是那一个系数；多个工具时 ``first_stage_coef`` 是
    第一个工具的系数，而 **F 是被排除工具整体的稳健 Wald F** ——
    后者才是弱工具诊断要看的量。
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    Z = _as_2d(z, n)
    X = _as_2d(x, n)
    W = np.column_stack([Z, X, np.ones(n)])
    beta, cov, _ = _ols(d, W)
    k = Z.shape[1]
    v = cov[:k, :k]
    b = beta[:k]
    f = float(b @ np.linalg.pinv(v) @ b / k)
    return f, float(beta[0]), float(np.sqrt(cov[0, 0]))


def two_sls(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
) -> IVResult:
    """两阶段最小二乘（异方差稳健），并给出第一阶段的强度诊断。

    ``z`` 是工具（可多个），``x`` 是外生控制变量（可多个，也被当作自己的工具）。
    ``d`` 是内生处置。返回的 ``beta`` 是 ``d`` 的系数。
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    n = y.size
    Z = _as_2d(z, n)
    X = _as_2d(x, n)
    if Z.shape[0] != n or d.size != n:
        raise ValueError("y / d / z 的样本量必须一致")
    if Z.shape[1] < 1:
        raise ValueError("至少要一个工具")

    W = np.column_stack([Z, X, np.ones(n)])
    Xt = np.column_stack([d, X, np.ones(n)])

    wtw_inv = np.linalg.pinv(W.T @ W)
    A = wtw_inv @ (W.T @ Xt)
    xpx = Xt.T @ (W @ A)
    xpy = Xt.T @ (W @ (wtw_inv @ (W.T @ y)))
    xpx_inv = np.linalg.pinv(xpx)
    beta = xpx_inv @ xpy
    resid = y - Xt @ beta

    # 稳健三明治：E[XX'uu'] 的估计用 (P_W X̃) 而不是 X̃（2SLS 的正确形式）
    pxt = W @ A
    score = pxt * resid[:, None]
    cov = xpx_inv @ (score.T @ score) @ xpx_inv * (n / max(n - Xt.shape[1], 1))

    b_d = float(beta[0])
    se_d = float(np.sqrt(max(cov[0, 0], 0.0)))
    t_stat = b_d / se_d if se_d > 0 else float("nan")
    p_value = float(2 * stats.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else float("nan")
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (b_d - z_crit * se_d, b_d + z_crit * se_d)

    f_stat, fs_coef, fs_se = first_stage_f(d, Z, X)
    ar_ci, ar_unbounded, ar_width = anderson_rubin_ci(y, d, Z, X, alpha=alpha, guess=b_d, se=se_d)

    notes: list[str] = []
    if f_stat < WEAK_F_RULE:
        notes.append(
            f"第一阶段 F = {f_stat:.2f} < {WEAK_F_RULE:g}：**弱工具**。"
            "此时 2SLS 的**中位偏差朝 OLS 靠**（实测最弱档位到 OLS 偏差的 72%）、"
            "均值与 RMSE 会爆炸 —— 而**区间**未必崩：本仓库实测 Wald 覆盖率仍 ≈0.96，"
            "崩掉的是 AR 区间的可用性（89% 无界）。两条都要看，见 m3 报告 2.8 节。"
        )
    wald_width = ci[1] - ci[0]
    if ar_ci is None:
        notes.append(
            "AR 区间是**空集**：连一个 95% 可信的 β 都给不出来 —— "
            "这本身就是结论，不要退回去看 Wald 区间。"
        )
    elif ar_unbounded:
        notes.append(
            "AR 区间**无界**：工具有效信息太少，数据排除不掉任何 β（含两端）。"
            "此时 Wald 区间那个「窄」是假的。"
        )
    elif ar_width > 1.3 * wald_width:
        notes.append(
            f"AR 区间比 Wald 区间宽 {ar_width / wald_width:.2f} 倍 —— "
            "弱工具下 Wald 型区间会过窄，以 AR 为准。"
        )

    sargan_j = sargan_p = None
    if Z.shape[1] > 1:
        sargan_j, sargan_p = sargan_test(resid, Z, X)
        notes.append(
            f"过度识别检验 p = {sargan_p:.4f}：它检验的是"
            "**排除性约束**（工具与误差不相关）是否与数据相容 —— "
            "注意它需要工具多于内生变量，而且功效很低，p 大不等于约束成立。"
        )

    return IVResult(
        beta=b_d,
        se=se_d,
        t=float(t_stat),
        p_value=p_value,
        ci=ci,
        ar_ci=ar_ci,
        ar_unbounded=ar_unbounded,
        ar_width=ar_width,
        first_stage_f=f_stat,
        first_stage_coef=fs_coef,
        first_stage_se=fs_se,
        n=n,
        n_instruments=Z.shape[1],
        weak=bool(f_stat < WEAK_F_RULE),
        alpha=alpha,
        sargan_j=sargan_j,
        sargan_p=sargan_p,
        notes=notes,
    )


def _ar_statistic(
    y: np.ndarray, d: np.ndarray, Z: np.ndarray, X: np.ndarray, beta0: float
) -> float:
    """AR 统计量：在 ``β = β0`` 下检验工具与残差的相关性（稳健版，χ²(k)）。"""
    if X.size:
        y_r, d_r, Z_r = _residualize(X, y), _residualize(X, d), _residualize(X, Z)
    else:
        y_r, d_r, Z_r = y, d, Z
    return _ar_from_residualized(y_r, d_r, Z_r, beta0)


def _ar_from_residualized(
    y_r: np.ndarray, d_r: np.ndarray, Z_r: np.ndarray, beta0: float
) -> float:
    """残差化之后的 AR 统计量（稳健版，``~ χ²(k)``）—— 每点只做 O(n·k)。"""
    u = y_r - d_r * beta0
    score = Z_r.T @ u
    meat = Z_r.T @ (Z_r * (u**2)[:, None])
    return float(score @ np.linalg.pinv(meat) @ score)


def anderson_rubin_ci(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
    guess: float | None = None,
    se: float | None = None,
    n_grid: int = 1201,
    span: float = 40.0,
) -> tuple[tuple[float, float] | None, bool, float]:
    """Anderson-Rubin 置信区间（对弱工具稳健）。

    做法：网格上的每个 ``β0`` 都用稳健 AR 统计量检验一次，保留
    ``AR(β0) ≤ χ²_{1−α}(k)`` 的那些点。返回 ``(区间, 是否无界, 宽度)``。

    * **无界**：网格端点仍未被排除 —— 返回端点并置 ``unbounded=True``，
      因为"这条数据说不清 β 的上界"是结论，不是缺陷；
    * **空集**：返回 ``None`` —— 罕见但会发生（弱工具下 2SLS 估计量本身
      离 AR 的可信集很远），它同样是一个结论。

    网格中心默认取 2SLS 估计（``guess``），跨度 ±``span`` 倍稳健 SE。
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    n = y.size
    Z = _as_2d(z, n)
    X = _as_2d(x, n)
    k = Z.shape[1]
    crit = float(stats.chi2.ppf(1 - alpha, k))

    center = 0.0 if guess is None or not np.isfinite(guess) else float(guess)
    half = 1.0 if se is None or not np.isfinite(se) or se <= 0 else float(se) * span
    grid = np.linspace(center - half, center + half, n_grid)
    # **残差化只做一次**，网格上每点都是 O(n·k)（见 _residualize 的注释）
    if X.size:
        y_r, d_r, Z_r = _residualize(X, y), _residualize(X, d), _residualize(X, Z)
    else:
        y_r, d_r, Z_r = y, d, Z
    inside = np.array(
        [_ar_from_residualized(y_r, d_r, Z_r, b0) <= crit for b0 in grid]
    )
    if not inside.any():
        return None, False, float("nan")

    lo = float(grid[inside][0])
    hi = float(grid[inside][-1])
    unbounded = bool(inside[0] or inside[-1])
    # 无界时**宽度记 inf**：此时区间端点只是网格边界，报一个"宽 1080"会让人
    # 以为那是一段有信息的范围，而它真正说的是"两边都没排除掉"。
    width = float("inf") if unbounded else hi - lo
    return (lo, hi), unbounded, width


def sargan_test(
    resid: np.ndarray, z: np.ndarray, x: np.ndarray | None = None
) -> tuple[float, float]:
    """过度识别检验：``n·R²`` 把 2SLS 残差回归到工具上，``~ χ²(k−1)``。

    它检验的是**工具与误差不相关**这一条（与排除性约束同义）。p 值大只能说明
    "数据没有反对它"，不能说明它成立 —— 而且它的功效通常很低。
    """
    resid = np.asarray(resid, dtype=float)
    n = resid.size
    Z = _as_2d(z, n)
    X = _as_2d(x, n)
    if Z.shape[1] < 2:
        raise ValueError("过度识别检验需要至少两个工具")
    W = np.column_stack([Z, X, np.ones(n)])
    beta, _, r = _ols(resid, W)
    ss_tot = float(((resid - resid.mean()) ** 2).sum())
    ss_res = float(r @ r)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    j = float(n * r2)
    df = Z.shape[1] - 1
    return j, float(stats.chi2.sf(j, df))
