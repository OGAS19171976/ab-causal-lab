"""异质处理效应：DGP、DML、元学习器。

M4 的验证台难点
---------------
前三个阶段可以只用一个 DGP，因为估计量不挑数据的**函数形式**。
但 CATE 估计量会：树模型天生擅长阶梯函数，线性模型天生擅长线性函数，
核方法擅长平滑函数。**只用一个 DGP 去评 CATE，等于在挑赢家。**

所以这里的 DGP 提供一族 ``cate_form``，并且刻意让
``m(X)``（倾向得分）、``g(X)``（结果函数）、``tau(X)``（异质效应）
用**不同的特征子集 + 不同的函数形式**，谁也不占便宜。
验证台在四种形式上都跑一遍，报告完整的对照表 ——
"没有单一赢家"本身就是结论。

DML 为什么值得单独做
--------------------
把 ``Y`` 对 ``[D, X]`` 直接跑一个弹性的 ML 模型，它的 D 系数是有偏的：
D 与 ``g(X)`` 的非线性部分相关，正则化会把 D 的系数也一起压下去。
Chernozhukov et al. (2018) 的 DML 用**正交化 + 交叉拟合**解决：

    Y = theta*D + g(X) + e,    D = m(X) + v
    残差对残差回归：  theta_hat = <Y - g_hat(X), D - m_hat(X)> / ||D - m_hat(X)||^2

关键在于 **Neyman 正交**：只要 ``g_hat`` 和 ``m_hat`` 的收敛速度好于 ``n^{-1/4}``，
``theta_hat`` 就是 ``sqrt(n)`` 一致且渐近正态的 —— 即使 nuisance 模型本身很差。
交叉拟合（在 K-1 折上拟合、在第 K 折上预测）是为了消掉过拟合带来的自身偏置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

__all__ = [
    "HTEConfig",
    "HTEData",
    "generate_hte_data",
    "DMLResult",
    "dml_partial_linear",
    "naive_plugin",
    "s_learner",
    "t_learner",
    "x_learner",
    "CATE_FORMS",
]

CATE_FORMS = ("constant", "linear", "threshold", "nonlinear")


# --------------------------------------------------------------------------- #
# DGP
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HTEConfig:
    """异质效应数据的超参数。

    ``cate_form`` 决定 ``tau(X)`` 的函数形式，也就决定了"哪个模型会赢"：

    * ``constant``  —— 没有异质性，最好的策略是报一个常数；弹性模型只会过拟合
    * ``linear``    —— 线性模型占优
    * ``threshold`` —— 树模型占优（阶梯函数正是它的归纳偏置）
    * ``nonlinear`` —— 平滑非线性，需要足够弹性的模型，但树要靠很多次分裂逼近

    报告必须同时给出这四种，否则"我的模型更好"只是一句关于 DGP 的话。
    """

    n: int = 4000
    n_features: int = 20
    n_informative: int = 6
    cate_form: str = "nonlinear"
    cate_scale: float = 1.0
    outcome_curvature: float = 1.0
    propensity_strength: float = 1.0
    noise_sd: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cate_form not in CATE_FORMS:
            raise ValueError(f"cate_form 必须是 {CATE_FORMS} 之一，收到 {self.cate_form!r}")
        if self.n_informative > self.n_features:
            raise ValueError("n_informative 不能超过 n_features")
        if self.n < 50:
            raise ValueError("样本量太小")


@dataclass(frozen=True)
class HTEData:
    """带真值的异质效应数据。"""

    X: np.ndarray  # (n, p)
    D: np.ndarray  # (n,) 0/1
    Y: np.ndarray  # (n,)
    tau: np.ndarray  # (n,) 真实个体效应
    propensity: np.ndarray  # (n,) 真实倾向得分
    outcome_function: np.ndarray  # (n,) g(X)
    config: HTEConfig

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def ate(self) -> float:
        return float(self.tau.mean())

    @property
    def treated_share(self) -> float:
        return float(self.D.mean())

    def summary(self) -> str:
        return (
            f"HTE 数据：n={self.n}, p={self.X.shape[1]}, "
            f"cate_form={self.config.cate_form}\n"
            f"  真实 ATE = {self.ate:+.4f}   "
            f"真实 CATE 范围 [{self.tau.min():+.3f}, {self.tau.max():+.3f}]   "
            f"SD = {self.tau.std():.4f}\n"
            f"  处置比例 = {self.treated_share:.3f}   "
            f"倾向得分范围 [{self.propensity.min():.3f}, {self.propensity.max():.3f}]"
        )


def _tau_function(X: np.ndarray, form: str, scale: float) -> np.ndarray:
    """真实 CATE。四种形式各自偏袒不同的模型族。"""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    if form == "constant":
        base = np.ones(X.shape[0])
    elif form == "linear":
        base = 1.0 + 0.8 * x1 + 0.5 * x2
    elif form == "threshold":
        base = np.where(x1 > 0.0, 1.0, -1.0) + 0.5 * np.where(x2 > 0.5, 1.0, 0.0)
    elif form == "nonlinear":
        # 平滑但带交互：树要靠很多次分裂逼近，线性模型完全抓不到
        base = np.sin(np.pi * x1 / 2.0) * np.exp(0.4 * x2) + 0.3 * x3**2
    else:  # pragma: no cover - 已在 config 里校验
        raise ValueError(form)
    return scale * base


def _outcome_function(X: np.ndarray, curvature: float) -> np.ndarray:
    """g(X)：与 tau 用**不同**的特征子集和函数形式，避免互相"送分"。"""
    x = X
    return curvature * (
        1.5 * x[:, 2]
        + 0.8 * x[:, 3] ** 2
        + 0.6 * np.sin(x[:, 4])
        + 0.5 * x[:, 5] * x[:, 2]
    )


def _propensity_function(X: np.ndarray, strength: float) -> np.ndarray:
    """倾向得分用**另外一组**特征，且非线性 —— 制造真实的混淆。"""
    linear = 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 2]
    nonlinear = 0.6 * X[:, 1] ** 2 - 0.4 * np.cos(X[:, 3])
    logit = strength * (linear + nonlinear)
    return 1.0 / (1.0 + np.exp(-logit))


def generate_hte_data(
    config: HTEConfig | None = None,
    **overrides,
) -> HTEData:
    """生成带已知 CATE 的观测数据（处理是**非随机**的）。"""
    cfg = config or HTEConfig(**overrides)
    rng = np.random.default_rng(cfg.seed)

    X = rng.normal(0.0, 1.0, (cfg.n, cfg.n_features))
    propensity = _propensity_function(X, cfg.propensity_strength)
    D = (rng.random(cfg.n) < propensity).astype(float)

    g = _outcome_function(X, cfg.outcome_curvature)
    tau = _tau_function(X, cfg.cate_form, cfg.cate_scale)
    Y = g + tau * D + rng.normal(0.0, cfg.noise_sd, cfg.n)

    return HTEData(
        X=X,
        D=D,
        Y=Y,
        tau=tau,
        propensity=propensity,
        outcome_function=g,
        config=cfg,
    )


# --------------------------------------------------------------------------- #
# DML
# --------------------------------------------------------------------------- #
@dataclass
class DMLResult:
    """DML 的部分线性模型估计。"""

    theta: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_folds: int
    true_theta: float | None = None

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def bias(self) -> float:
        if self.true_theta is None:
            return float("nan")
        return self.theta - self.true_theta

    @property
    def covers_truth(self) -> bool:
        if self.true_theta is None:
            return False
        return self.ci_low <= self.true_theta <= self.ci_high

    def summary(self) -> str:
        lines = [
            f"DML 部分线性估计（{self.n_folds} 折交叉拟合）",
            f"  theta = {self.theta:+.4f}  SE {self.se:.4f}",
            f"  95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  p={self.p_value:.4g}",
        ]
        if self.true_theta is not None:
            lines.append(
                f"  真值 {self.true_theta:+.4f}  偏置 {self.bias:+.4f}  "
                f"覆盖真值 {self.covers_truth}"
            )
        return "\n".join(lines)


def _default_nuisance():
    """默认 nuisance 学习器：小随机森林。

    刻意**不是**线性模型 —— 因为 DML 的价值恰恰在于 nuisance 可以任意复杂。

    **必须是单进程。** ``n_jobs=-1`` 会走 joblib 的多进程后端，
    而受限（沙箱）环境禁止命名管道，直接抛 ``WinError 5``。
    规模因此取"够复杂但不拖垮仿真循环"：交叉拟合一次要在此基础上
    再乘 ``2 * n_folds`` 次拟合，所以单次拟合的开销会被放大十倍。
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=60,
        min_samples_leaf=10,
        random_state=0,
        n_jobs=1,
    )


def dml_partial_linear(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    model_y=None,
    model_d=None,
    n_folds: int = 5,
    true_theta: float | None = None,
    seed: int = 0,
) -> DMLResult:
    """DML 部分线性模型：``Y = theta*D + g(X) + e``。

    ``n_folds=1`` 表示**不做交叉拟合**（在同一批数据上拟合和预测），
    会因过拟合而偏离 —— 保留它是为了让这个偏置可以被量出来。
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = Y.size
    if not (D.size == n and X.shape[0] == n):
        raise ValueError("Y / D / X 的样本量必须一致")
    if n_folds < 1:
        raise ValueError("n_folds 必须 >= 1")

    model_y = model_y or _default_nuisance()
    model_d = model_d or _default_nuisance()

    if n_folds == 1:
        # 不交叉拟合：拟合与预测用同一批数据
        g_hat = model_y.fit(X, Y).predict(X)
        m_hat = model_d.fit(X, D).predict(X)
    else:
        from sklearn.model_selection import KFold

        g_hat = np.empty(n)
        m_hat = np.empty(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train, test in kf.split(X):
            g_hat[test] = _clone_fit_predict(model_y, X[train], Y[train], X[test])
            m_hat[test] = _clone_fit_predict(model_d, X[train], D[train], X[test])

    y_res = Y - g_hat
    d_res = D - m_hat

    denom = float(d_res @ d_res)
    if denom <= 0:
        raise ValueError("正交化后的 D 没有变异，无法识别 theta")

    theta = float(d_res @ y_res / denom)

    resid = y_res - theta * d_res
    sigma2 = float(resid @ resid / n)
    se = float(np.sqrt(sigma2 / denom))

    df = max(n - 1, 1)
    crit = float(stats.t.ppf(0.975, df))
    p = float(2 * stats.t.sf(abs(theta / se), df)) if se > 0 else 1.0

    return DMLResult(
        theta=theta,
        se=se,
        ci_low=theta - crit * se,
        ci_high=theta + crit * se,
        p_value=p,
        n_folds=n_folds,
        true_theta=true_theta,
    )


def _clone_fit_predict(model, X_train, y_train, X_test):
    from sklearn.base import clone

    return clone(model).fit(X_train, y_train).predict(X_test)


def naive_plugin(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    true_theta: float | None = None,
) -> DMLResult:
    """**反例**：把 D 直接塞进一个线性回归，读它的系数。

    当 ``g(X)`` 含非线性项时，这些项进了误差项，而 D 与 X 相关，
    于是 D 的系数被遗漏变量偏置污染。这就是 DML 要解决的问题。
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = Y.size

    design = np.column_stack([D, np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    resid = Y - design @ beta
    sigma2 = float(resid @ resid / (n - design.shape[1]))

    xtx_inv = np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    theta = float(beta[0])

    df = max(n - design.shape[1], 1)
    crit = float(stats.t.ppf(0.975, df))
    p = float(2 * stats.t.sf(abs(theta / se), df)) if se > 0 else 1.0

    return DMLResult(
        theta=theta,
        se=se,
        ci_low=theta - crit * se,
        ci_high=theta + crit * se,
        p_value=p,
        n_folds=0,
        true_theta=true_theta,
    )


# --------------------------------------------------------------------------- #
# 元学习器
# --------------------------------------------------------------------------- #
def s_learner(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    *,
    learner=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """S-learner：把 D 当成一个普通特征训练 ``mu(X, D)``，CATE = mu(x,1) - mu(x,0)。

    最简单，但当 D 的信号很弱时，模型会**忽略** D 这个特征，
    于是 CATE 被压向 0（正则化偏置）。
    """
    learner = learner or _default_nuisance()
    XD = np.column_stack([X, D])
    model = learner.fit(XD, Y)

    def cate(X_new: np.ndarray) -> np.ndarray:
        X_new = np.atleast_2d(X_new)
        ones = np.column_stack([X_new, np.ones(X_new.shape[0])])
        zeros = np.column_stack([X_new, np.zeros(X_new.shape[0])])
        return model.predict(ones) - model.predict(zeros)

    return cate


def t_learner(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    *,
    learner=None,
) -> Callable[[np.ndarray], np.ndarray]:
    """T-learner：分别在处理组和对照组各训一个模型，作差。

    在倾向得分极端（某组样本很少）时方差很大 —— 这是它的软肋。
    """
    learner = learner or _default_nuisance()
    from sklearn.base import clone

    t_mask = D > 0.5
    if t_mask.sum() < 5 or (~t_mask).sum() < 5:
        raise ValueError("两组的样本量都至少要有 5 个")

    m1 = clone(learner).fit(X[t_mask], Y[t_mask])
    m0 = clone(learner).fit(X[~t_mask], Y[~t_mask])

    def cate(X_new: np.ndarray) -> np.ndarray:
        X_new = np.atleast_2d(X_new)
        return m1.predict(X_new) - m0.predict(X_new)

    return cate


def x_learner(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    *,
    learner=None,
    propensity: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """X-learner（Künzel et al. 2019）：适合两组样本量悬殊的情况。

    1. 先训 T-learner 的 ``mu1, mu0``
    2. 用**对侧模型**插补个体效应：处理组 ``D1 = Y - mu0(X)``，对照组 ``D0 = mu1(X) - Y``
    3. 分别在两组上训 ``tau1, tau0``
    4. 用倾向得分加权：``tau(x) = g(x)*tau0(x) + (1-g(x))*tau1(x)``
    """
    learner = learner or _default_nuisance()
    from sklearn.base import clone

    t_mask = D > 0.5
    n1, n0 = int(t_mask.sum()), int((~t_mask).sum())
    if n1 < 5 or n0 < 5:
        raise ValueError("两组的样本量都至少要有 5 个")

    m1 = clone(learner).fit(X[t_mask], Y[t_mask])
    m0 = clone(learner).fit(X[~t_mask], Y[~t_mask])

    d1 = Y[t_mask] - m0.predict(X[t_mask])
    d0 = m1.predict(X[~t_mask]) - Y[~t_mask]

    tau1 = clone(learner).fit(X[t_mask], d1)
    tau0 = clone(learner).fit(X[~t_mask], d0)

    if propensity is None:
        g = float(n1) / (n1 + n0)
    else:
        g = propensity

    def cate(X_new: np.ndarray) -> np.ndarray:
        X_new = np.atleast_2d(X_new)
        if np.isscalar(g) or np.ndim(g) == 0:
            return float(g) * tau0.predict(X_new) + (1.0 - float(g)) * tau1.predict(X_new)
        return g * tau0.predict(X_new) + (1.0 - g) * tau1.predict(X_new)

    return cate
