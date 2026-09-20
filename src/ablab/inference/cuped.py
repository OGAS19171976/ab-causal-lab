"""CUPED：用实验前指标做方差缩减，同时消掉协变量失衡造成的偏置。

    Y_adj = Y - θ·(X - X̄),   θ = Cov(X, Y) / Var(X)

X 是**实验前**指标（与分流独立），Y 是实验后指标。

为什么这一步同时解决两个问题
----------------------------
记 β 为 Y 对 X 的回归系数。固定住某一次分流后：

    post_gap = β·pre_gap + η        （η 零均值，见 M0 的 conditional 模式）

* **消偏置**：CUPED 减去 θ·pre_gap，θ̂ → β 时偏置项被整项扣掉，
  剩下的是真正零均值的 η。M0 里那个"负对照实验跑出 p=0.004"的假象就来自 β·pre_gap。
* **降方差**：Var(η) = (1-ρ²)·Var(post_gap)。实测 ρ≈0.795 时
  **方差缩减 = ρ² ≈ 63%**，**残余方差 = 1-ρ² ≈ 37%**，
  等价于**有效样本量放大 1/(1-ρ²) ≈ 2.7 倍**。

三个量必须分清楚（本项目用三个独立命名隔开，并有测试守着不被写反）：

======================  ==================  ==========
量                       公式                ρ=0.795
======================  ==================  ==========
方差缩减（去掉的）        ρ²                  0.632
残余方差（剩下的）        1-ρ²                0.368
标准误降幅               1-√(1-ρ²)           0.394
有效样本量倍数            1/(1-ρ²)            2.72
======================  ==================  ==========

θ 用哪一臂估计
--------------
``pooled``（默认）
    两臂合并估计。方差最小、最常用。若处理会改变 (X,Y) 的联合分布，
    θ̂ 会是两臂的加权平均，略有污染。
``control``
    只用对照组。更"干净"（完全不看处理组），代价是效率略低。
    在担心处理影响协方差结构时用它。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from .aggregates import AggregateStats
from .result import Diagnostic, Estimate, Status
from .srm import srm_check
from .welch import welch_inference

__all__ = [
    "CupedFit",
    "MultivariateCupedFit",
    "ThetaSource",
    "cuped_estimate",
    "cuped_ttest",
    "fit_cuped",
    "fit_multivariate_cuped",
    "multivariate_theta",
]

ThetaSource = Literal["pooled", "control"]

#: |rho| 低于这个值时 CUPED 收益有限，给一条 warn
_WEAK_CORRELATION = 0.10

#: 前置协变量组间差超过这个显著性水平就提示失衡（比常规 0.05 严，
#: 因为每个实验都会跑这个检查，且我们只想抓"系统性问题"）
_IMBALANCE_ALPHA = 1e-3


@dataclass(frozen=True)
class CupedFit:
    """CUPED 的 θ 估计及其质量指标。"""

    theta: float
    theta_source: str
    n_used: int
    covariate_mean: float
    covariate_var: float
    covariance: float
    correlation: float

    # -- 四个必须分清的收益指标 -------------------------------------------- #
    @property
    def variance_reduction(self) -> float:
        """去掉的方差比例 = rho^2。"""
        return self.correlation**2

    @property
    def remaining_variance(self) -> float:
        """校正后剩下的方差比例 = 1 - rho^2。"""
        return 1.0 - self.correlation**2

    @property
    def se_shrinkage(self) -> float:
        """标准误的降幅 = 1 - sqrt(1 - rho^2)。**不等于**方差缩减。"""
        return 1.0 - math.sqrt(self.remaining_variance)

    @property
    def effective_sample_multiplier(self) -> float:
        """等效样本量倍数 = 1 / (1 - rho^2)。"""
        return 1.0 / self.remaining_variance

    def summary(self) -> str:
        return (
            f"CUPED 拟合 (theta 来源={self.theta_source}, n={self.n_used:,})\n"
            f"  rho(pre, post) = {self.correlation:.4f}\n"
            f"  theta = {self.theta:.4f}\n"
            f"  方差缩减   = rho^2     = {self.variance_reduction:.4f}\n"
            f"  残余方差   = 1 - rho^2 = {self.remaining_variance:.4f}\n"
            f"  标准误降幅 = 1-sqrt(...) = {self.se_shrinkage:.4f}\n"
            f"  等效样本量 x{self.effective_sample_multiplier:.2f}\n"
        )


def fit_cuped(
    pooled: AggregateStats,
    control: AggregateStats | None = None,
    *,
    theta_source: ThetaSource = "pooled",
) -> CupedFit:
    """估计 CUPED 的 θ。

    Parameters
    ----------
    pooled:
        两臂**合并**后的统计量（用于算 ρ 与默认的 θ）。
    control:
        对照组统计量，``theta_source="control"`` 时必需。
    """
    if theta_source not in ("pooled", "control"):
        raise ValueError(f"未知 theta_source: {theta_source!r}")
    if theta_source == "control":
        if control is None:
            raise ValueError("theta_source='control' 需要传入 control 统计量")
        source = control
    else:
        source = pooled

    var_x = source.var_x
    if not np.isfinite(var_x) or var_x <= 0:
        raise ValueError(
            "前置协变量没有方差，CUPED 无法估计 theta"
            "（检查前置指标是否全为常数，或窗口是否取在了曝光之后）"
        )

    return CupedFit(
        theta=float(source.cov_xy / var_x),
        theta_source=theta_source,
        n_used=source.n,
        covariate_mean=float(pooled.mean_x),
        covariate_var=float(var_x),
        covariance=float(source.cov_xy),
        correlation=float(pooled.corr_xy),
    )


def _balance_diagnostic(treatment: AggregateStats, control: AggregateStats) -> Diagnostic:
    """前置协变量的组间平衡性检验。

    这是实验可信度的体检项：前置指标在**分流前**测量，两臂不应有系统性差异。
    （单次实现的随机不平衡是正常的，频繁出现才说明分流或埋点有问题。）
    """
    var_t = treatment.var_x / treatment.n
    var_c = control.var_x / control.n
    se = math.sqrt(var_t + var_c)
    gap = treatment.mean_x - control.mean_x

    if se == 0 or not np.isfinite(se):
        return Diagnostic(
            name="协变量平衡",
            status="info",
            message="前置协变量无波动，平衡性检验不可用",
        )

    z = gap / se
    p = float(2 * stats.norm.sf(abs(z)))
    status: Status = "warn" if p < _IMBALANCE_ALPHA else "pass"
    msg = (
        f"前置指标组间差 {gap:+.4f}（{z:+.2f} 个标准误），p={p:.4g}"
        + ("；**失衡**——CUPED 收益最大的情形，但也提示分流链路值得复查" if status == "warn" else "")
    )
    return Diagnostic(
        name="协变量平衡",
        status=status,
        message=msg,
        statistic=float(z),
        p_value=p,
    )


def cuped_estimate(
    treatment: AggregateStats,
    control: AggregateStats,
    *,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
    theta_source: ThetaSource = "pooled",
    expected_weights: dict[str, float] | None = None,
    srm_alpha: float = 1e-3,
) -> tuple[Estimate, CupedFit]:
    """从汇总统计量做 CUPED 校正 + Welch 检验，返回 (估计, 拟合)。

    ``treatment``/``control`` 里的 ``x`` 是**实验前**指标，``y`` 是实验后指标。
    """
    pooled = treatment.merge(control)
    fit = fit_cuped(pooled, control, theta_source=theta_source)
    theta = fit.theta

    # 中心化常数对组间差没有影响（两臂各减同一个 pooled.mean_x），
    # 只影响"校正后均值"的绝对水平，因此这里统一用合并均值。
    center = pooled.mean_x
    adj_mean_t = treatment.mean_y - theta * (treatment.mean_x - center)
    adj_mean_c = control.mean_y - theta * (control.mean_x - center)
    effect = adj_mean_t - adj_mean_c

    # 校正后每一臂内的方差：Var(y - theta*x) = Var(y) - 2*theta*Cov + theta^2*Var(x)
    var_adj_t = treatment.var_y - 2 * theta * treatment.cov_xy + theta**2 * treatment.var_x
    var_adj_c = control.var_y - 2 * theta * control.cov_xy + theta**2 * control.var_x
    # 数值保护：理论上非负，浮点误差可能给出极小负数
    var_adj_t = max(float(var_adj_t), 0.0)
    var_adj_c = max(float(var_adj_c), 0.0)

    inference = welch_inference(
        effect,
        n_treatment=treatment.n,
        var_treatment=var_adj_t,
        n_control=control.n,
        var_control=var_adj_c,
        alpha=alpha,
    )
    ci_low, ci_high = inference.interval(effect)

    diagnostics: list[Diagnostic] = [_balance_diagnostic(treatment, control)]

    if abs(fit.correlation) < _WEAK_CORRELATION:
        diagnostics.append(
            Diagnostic(
                name="CUPED 收益",
                status="warn",
                message=(
                    f"|rho|={abs(fit.correlation):.4f} 太低，方差缩减仅 "
                    f"{fit.variance_reduction:.2%}；换一个与结果更相关的前置指标，"
                    "或考虑 CUPAC（用模型预测值当协变量）"
                ),
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                name="CUPED 收益",
                status="pass",
                message=(
                    f"方差缩减 {fit.variance_reduction:.2%}（残余 {fit.remaining_variance:.2%}，"
                    f"标准误降 {fit.se_shrinkage:.2%}），等效样本量 x"
                    f"{fit.effective_sample_multiplier:.2f}"
                ),
                statistic=float(fit.correlation),
            )
        )

    if expected_weights is not None:
        diagnostics.append(
            srm_check(
                {control_name: control.n, variant: treatment.n},
                expected_weights,
                alpha=srm_alpha,
            )
        )

    relative = effect / adj_mean_c if adj_mean_c != 0 else float("nan")

    return (
        Estimate(
            metric=metric,
            variant=variant,
            control=control_name,
            method="CUPED + Welch t-test",
            absolute_effect=float(effect),
            relative_effect=float(relative),
            std_error=inference.se,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            p_value=inference.p_value,
            n_treatment=int(treatment.n),
            n_control=int(control.n),
            mean_treatment=float(adj_mean_t),
            mean_control=float(adj_mean_c),
            alpha=alpha,
            diagnostics=tuple(diagnostics),
        ),
        fit,
    )


def cuped_ttest(
    treatment_pre,
    treatment_post,
    control_pre,
    control_post,
    **kwargs,
) -> tuple[Estimate, CupedFit]:
    """逐单元明细入口：``(前, 后)`` 两臂各一对数组。"""
    return cuped_estimate(
        AggregateStats.from_arrays(treatment_post, treatment_pre),
        AggregateStats.from_arrays(control_post, control_pre),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 多协变量 CUPED：theta = Sigma_X^{-1} Cov(X, Y)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MultivariateCupedFit:
    """多协变量 CUPED 的拟合，以及两个**必须一起看**的风险指标。

    单协变量时"方差缩减 = rho^2"是个可以用眼睛验证的等式；
    协变量一多，有两件事会悄悄发生：

    1. **样本内 theta 会过拟合**。用同一批数据估 theta 再算缩减，
       噪声协变量也会"贡献"一点缩减 —— 于是它**看起来有效**。
       诚实的做法是交叉拟合（``n_folds > 1``）：theta 在别的折上估、
       在留出的那一折上调整。两个数都给，差值就是过拟合的幅度；
    2. **共线性会让 theta 爆炸**。``Sigma_X`` 接近奇异时，
       ``Sigma_X^{-1}`` 的元素会到 1e6 量级，theta 不可信。
       所以这里报**条件数**，并支持 ``ridge`` 正则化。
    """

    theta: np.ndarray
    n_used: int
    n_covariates: int
    #: **实测**去掉的方差比例 = 1 − Var(Y_adj)/Var(Y)。
    #:
    #: 踩过一次：第一版把它写成 ``rho(y, y_adj)^2`` —— 那是**残余**比例，
    #: 因为 ``Corr(y, y−x'θ)^2 = Var(y−x'θ)/Var(y)``。于是"两协变量的缩减
    #: 0.109"看起来比"单协变量 0.85"还差，方向正好反了。
    #: 现在直接量：两个口径（样本内 / 交叉拟合）的差别就是过拟合的幅度。
    variance_reduction_measured: float
    #: Corr(Y, Y_adj)：与上面那个互补（``rho^2 = 残余比例``）
    rho: float
    #: 逐个协变量单独做 CUPED 的 rho^2（对照：多协变量到底多拿了多少）
    univariate_reductions: np.ndarray
    #: X 的协方差矩阵条件数 —— 多协变量最实际的风险
    condition_number: float
    #: 实际用到的 ridge（0 表示没加）
    ridge: float
    #: 交叉拟合的折数（1 = 样本内，会有过拟合）
    n_folds: int

    @property
    def variance_reduction(self) -> float:
        """去掉的方差比例。``n_folds=1`` 时它是**样本内**的，会偏乐观。"""
        return self.variance_reduction_measured

    @property
    def remaining_variance(self) -> float:
        return 1.0 - self.variance_reduction_measured

    @property
    def se_shrinkage(self) -> float:
        return 1.0 - math.sqrt(self.remaining_variance)

    @property
    def ill_conditioned(self) -> bool:
        """``Σ_X`` 的条件数是否大到"解不可信"。

        **必须有这道闸门**：``np.linalg.solve`` 对数值奇异的矩阵**不会报错**，
        它照常返回一个巨大的解。实测 p=120/n=80 时条件数 2.3e18、样本内 R²=1.0000，
        而留出集上的缩减是 −1.1e4 —— 如果只看样本内，会得出"完美"的结论。
        阈值取 1e10（float64 的相对精度是 1e-16，条件数到 1e10 时解已丢 6 位）。
        """
        return bool(np.isfinite(self.condition_number) and self.condition_number > 1e10)

    @property
    def best_univariate_reduction(self) -> float:
        """单独用最好的那一个协变量能拿到的缩减。"""
        if self.univariate_reductions.size == 0:
            return float("nan")
        return float(np.max(self.univariate_reductions))

    @property
    def extra_from_multivariate(self) -> float:
        """多协变量相对"最好的单协变量"多拿了多少（可能为负）。"""
        return self.variance_reduction - self.best_univariate_reduction

    def summary(self) -> str:
        mode = "样本内（偏乐观）" if self.n_folds == 1 else f"交叉拟合 {self.n_folds} 折"
        warn = (
            "\n  **警告：Σ_X 条件数 "
            f"{self.condition_number:.1e} > 1e10 —— θ̂ 不可信，请减协变量或加 ridge**"
            if self.ill_conditioned
            else ""
        )
        return (
            f"多协变量 CUPED（p={self.n_covariates}, n={self.n_used:,}, {mode}）\n"
            f"  rho(X'θ, Y) = {self.rho:.4f} → 方差缩减 {self.variance_reduction:.4f}"
            f"（标准误降 {self.se_shrinkage:.4f}）\n"
            f"  最好的单协变量缩减 = {self.best_univariate_reduction:.4f}"
            f"，多协变量多拿 {self.extra_from_multivariate:+.4f}\n"
            f"  Sigma_X 条件数 = {self.condition_number:.3e}"
            f"（ridge={self.ridge:g}）" + warn
        )


def _covariance_matrix(X: np.ndarray) -> np.ndarray:
    """中心化的协方差矩阵 ``Sigma_X``（除以 n，与 AggregateStats 的口径一致）。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / X.shape[0]


def multivariate_theta(
    X: np.ndarray, y: np.ndarray, *, ridge: float = 0.0
) -> np.ndarray:
    """``θ = Σ_X⁻¹ Cov(X, Y)``（可加 ridge 正则）。

    单协变量时它退化成 ``Cov(x,y)/Var(x)`` —— 与 ``fit_cuped`` 完全一致，
    有测试钉着这个等价。``ridge`` 加在对角线上（``Σ_X + ridge·I``）：
    共线性严重时它把解拉回来，代价是引入一点偏差 —— 这一点必须显式选择，
    不能藏在实现里。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.size:
        raise ValueError("X 与 y 的样本量必须一致")
    if X.shape[1] == 0:
        raise ValueError("至少要一个协变量")
    if ridge < 0:
        raise ValueError("ridge 不能为负")

    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    sigma = (Xc.T @ Xc) / n
    cov = (Xc.T @ yc) / n
    if ridge > 0:
        sigma = sigma + ridge * np.eye(sigma.shape[0])
    try:
        return np.linalg.solve(sigma, cov)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - 需要极端共线性
        raise ValueError(
            "Sigma_X 奇异，无法求 θ —— 协变量之间完全共线；"
            "请去掉冗余协变量或显式给 ridge"
        ) from exc


def fit_multivariate_cuped(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 0.0,
    n_folds: int = 1,
    seed: int = 0,
) -> MultivariateCupedFit:
    """拟合多协变量 CUPED。``n_folds > 1`` 时用**交叉拟合**的 θ̂。

    交叉拟合的含义：留出那一折的调整量用的是**别的折**估出来的 θ̂，
    所以 ``variance_reduction`` 不再包含"用同一批数据挑 θ 造成的乐观偏差"。
    这是判断"多协变量到底有没有用"的唯一诚实口径。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.size:
        raise ValueError("X 与 y 的样本量必须一致")
    if n_folds < 1:
        raise ValueError("n_folds 必须 >= 1")

    n, p = X.shape
    sigma = _covariance_matrix(X)
    # **条件数要算在实际用于求解的那个矩阵上**：加了 ridge 之后
    # Σ_X + ridge·I 的条件数才是"解有多可信"的依据。
    # 第一版算的是未正则化的 Σ_X，于是"ridge 有没有改善条件数"这条断言
    # 得到两个一模一样的数（实测 9.9e17 vs 9.9e17）—— 报告里那个指标是假的。
    sigma_used = sigma + ridge * np.eye(p) if ridge > 0 else sigma
    cond = float(np.linalg.cond(sigma_used)) if p else float("nan")

    if n_folds == 1:
        theta = multivariate_theta(X, y, ridge=ridge)
        y_adj = y - (X - X.mean(axis=0, keepdims=True)) @ theta
    else:
        from sklearn.model_selection import KFold

        y_adj = np.empty(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train, test in kf.split(X):
            theta_fold = multivariate_theta(X[train], y[train], ridge=ridge)
            y_adj[test] = y[test] - (
                X[test] - X[train].mean(axis=0, keepdims=True)
            ) @ theta_fold
        theta = multivariate_theta(X, y, ridge=ridge)

    # 实测的方差缩减 = 1 − Var(Y_adj)/Var(Y)（**直接量**，不从 rho 反推）
    if np.var(y) > 0:
        reduction = 1.0 - float(np.var(y_adj) / np.var(y))
    else:  # pragma: no cover - 常数结果
        reduction = 0.0
    if np.std(y_adj) > 0 and np.std(y) > 0:
        rho = float(np.corrcoef(y, y_adj)[0, 1])
    else:  # pragma: no cover - 常数结果
        rho = 0.0

    univariate = np.array(
        [
            float(np.corrcoef(X[:, j], y)[0, 1]) ** 2
            if np.std(X[:, j]) > 0 and np.std(y) > 0
            else 0.0
            for j in range(p)
        ]
    )

    return MultivariateCupedFit(
        theta=np.asarray(theta, dtype=float),
        n_used=n,
        n_covariates=p,
        variance_reduction_measured=reduction,
        rho=rho,
        univariate_reductions=univariate,
        condition_number=cond,
        ridge=float(ridge),
        n_folds=int(n_folds),
    )
