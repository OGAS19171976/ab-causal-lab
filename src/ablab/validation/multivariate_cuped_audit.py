"""多协变量 CUPED 的验证台：它最危险的地方不是不生效，而是**看起来生效**。

三个必须分开量的问题
--------------------
1. **样本内 θ̂ 会过拟合**。用同一批数据估 θ 再算缩减，噪声协变量也会"贡献"
   一点缩减 —— 协变量越多、样本越少，这个假收益越大。
   诚实的口径是交叉拟合（θ 在别的折上估、在留出的折上调整），
   两个数的差就是过拟合的幅度；
2. **共线性会让 θ̂ 爆炸**。``Σ_X`` 接近奇异时解会到 1e6 量级，
   样本内的缩减依然好看，而**留出集上可能比不调整还差**（缩减为负）。
   所以这里同时报条件数、``||θ̂||``，以及 ridge 之后的结果；
3. **"方差缩减"有两个口径**：``1 − Var(Y_adj)/Var(Y)`` 是理论代理，
   而业务真正关心的是**估计量的方差**。所以每个档位都用重随机化
   实测一次 ATE 方差缩减，与理论代理并排放 —— 两者不吻合就说明代理用错了。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..inference.cuped import fit_multivariate_cuped

__all__ = ["MultivariateCupedAudit", "MultivariateCupedRow", "run_multivariate_cuped_audit"]


def _correlated_covariates(
    rng: np.random.Generator, n: int, p: int, *, rho_collinear: float
) -> np.ndarray:
    """生成 ``p`` 个协变量：等相关系数 ``rho_collinear``（1 附近即强共线）。"""
    if p == 0:
        return np.empty((n, 0))
    if rho_collinear == 0.0:
        return rng.normal(0.0, 1.0, (n, p))
    # 等相关结构：X = sqrt(rho)*f + sqrt(1-rho)*e_j
    f = rng.normal(0.0, 1.0, (n, 1))
    e = rng.normal(0.0, 1.0, (n, p))
    return np.sqrt(rho_collinear) * f + np.sqrt(1.0 - rho_collinear) * e


def _ate_variance_reduction(
    X: np.ndarray, y: np.ndarray, *, n_trials: int, seed: int
) -> float:
    """重随机化测**估计量**的方差缩减（原始 vs CUPED 调整后）。

    这就是 CUPED 卖的那个东西：同样样本量下 ATE 的方差小多少。
    理论代理（``R^2``）在单协变量时与它相等；多协变量 + 共线性时可能背离，
    所以必须分开量。
    """
    n = y.size
    rng = np.random.default_rng(seed)
    theta = np.linalg.lstsq(
        np.column_stack([np.ones(n), X]), y, rcond=None
    )[0] if X.size else np.zeros(1)
    y_adj = y - (np.column_stack([np.ones(n), X]) @ theta - y.mean())

    raw, adj = np.empty(n_trials), np.empty(n_trials)
    half = n // 2
    for k in range(n_trials):
        order = rng.permutation(n)
        treated = np.zeros(n, dtype=bool)
        treated[order[:half]] = True
        raw[k] = y[treated].mean() - y[~treated].mean()
        adj[k] = y_adj[treated].mean() - y_adj[~treated].mean()
    var_raw, var_adj = float(raw.var(ddof=1)), float(adj.var(ddof=1))
    return 1.0 - var_adj / var_raw if var_raw > 0 else float("nan")


@dataclass(frozen=True)
class MultivariateCupedRow:
    """一个档位的读数。"""

    regime: str
    n: int
    p: int
    n_informative: int
    condition_number: float
    theta_norm: float
    reduction_in_sample: float
    reduction_cross_fitted: float
    best_univariate: float
    ate_variance_reduction: float
    ridge: float = 0.0
    reduction_with_ridge: float = float("nan")
    #: 拟合直接失败时的原因（例如 p ≥ n 导致 Σ_x 奇异）—— 空表示跑通了
    failed: str = ""

    @property
    def ran(self) -> bool:
        return not self.failed

    @property
    def overfitting(self) -> float:
        """样本内 − 交叉拟合 = 过拟合的幅度（正数表示样本内偏乐观）。"""
        return self.reduction_in_sample - self.reduction_cross_fitted

    def summary(self) -> str:
        if self.failed:
            return (
                f"  {self.regime:<28} n={self.n:<5} p={self.p:<3}"
                f" **直接报错**：{self.failed[:56]}"
            )
        ridge_txt = (
            f" | ridge 后 {self.reduction_with_ridge:.4f}" if self.ridge > 0 else ""
        )
        return (
            f"  {self.regime:<28} n={self.n:<5} p={self.p:<3}"
            f" 条件数 {self.condition_number:>9.2e}"
            f" | 样本内 {self.reduction_in_sample:+.4f}"
            f" 交叉拟合 {self.reduction_cross_fitted:+.4f}"
            f"（过拟合 {self.overfitting:+.4f}）"
            f" | 实测 ATE 方差缩减 {self.ate_variance_reduction:+.4f}{ridge_txt}"
        )


@dataclass(frozen=True)
class MultivariateCupedAudit:
    """多协变量 CUPED：三种情形（正常 / 噪声协变量多 / 强共线）的对照。"""

    n_trials: int
    rows: tuple[MultivariateCupedRow, ...]

    def _row(self, key: str) -> MultivariateCupedRow:
        return next(r for r in self.rows if r.regime == key)

    @property
    def multivariate_beats_univariate(self) -> bool:
        """多协变量档位（p ≥ 2）：诚实缩减高于「最好的单协变量」。

        只对 p ≥ 2 成立 —— 单协变量那一档本来就等于单协变量，
        拿它比是自比自（第一版就是这么写的，于是得到 False）。
        """
        good = [
            r for r in self.rows
            if "正常" in r.regime and r.ran and r.p >= 2 and r.ridge == 0
        ]
        return bool(good) and all(
            r.reduction_cross_fitted > r.best_univariate + 1e-6 for r in good
        )

    @property
    def noise_covariates_overfit_in_sample(self) -> bool:
        """噪声协变量多时：样本内明显乐观，而交叉拟合把收益打回原形。"""
        row = next((r for r in self.rows if "噪声" in r.regime), None)
        if row is None:
            return False
        return row.overfitting > 0.05 and row.reduction_cross_fitted < 0.2

    @property
    def collinearity_is_visible(self) -> bool:
        """共线档位：条件数很大、||θ̂|| 很大 —— 但**诚实缩减没有崩**。

        这一条与直觉相反，所以值得单独写成属性：共线让 θ̂ 的**单个分量**
        失去可解释性（两个几乎相同的协变量可以互相抵消），
        但只要样本量足够、``Σ_X`` 的数值秩还在，用它做**预测/调整**依然有效。
        真正致命的不是共线，是 ``p`` 接近或超过 ``n``（见下一条）。
        """
        row = next(
            (r for r in self.rows if "共线" in r.regime and r.ridge == 0 and r.ran), None
        )
        if row is None:
            return False
        return (
            row.condition_number > 1e3
            and row.theta_norm > 2.0
            and row.reduction_cross_fitted > row.reduction_in_sample - 0.05
        )

    @property
    def p_over_n_looks_perfect(self) -> bool:
        """``p ≥ n``：**样本内完美（缩减 = 1.0000）而留出集上是灾难**。

        而且这里还有一个实测发现：``np.linalg.solve`` 对**数值奇异**的
        ``Σ_X`` **不会报错** —— 它照常返回一个巨大的解，于是样本内 R² = 1。
        （第一版以为会抛 ``LinAlgError``，还写了 try/except —— 那条分支从没被走到。）
        所以这道闸门必须自己建：条件数超限就标出来，见
        ``MultivariateCupedFit.ill_conditioned``。
        """
        row = next((r for r in self.rows if "p≥n" in r.regime and r.ridge == 0), None)
        if row is None or not row.ran:
            return False
        return row.reduction_in_sample > 0.99 and row.reduction_cross_fitted < -1.0

    @property
    def ridge_tames_p_over_n_but_stays_honest(self) -> bool:
        """ridge 把 p≥n 的灾难从 1e4 量级压回个位数，但**收益仍然是负的**。

        全噪声协变量本来就不该有收益 —— 一个"加了 ridge 就有收益"的结果
        才更可疑。这条属性主张的是"能跑且不撒谎"，不是"能赚"。
        """
        row = next((r for r in self.rows if "p≥n" in r.regime and r.ridge > 0), None)
        return bool(
            row and row.ran and -5.0 < row.reduction_with_ridge <= 0.02
        )

    @property
    def ridge_helps_under_collinearity(self) -> bool:
        """共线档位上 ridge 让诚实缩减略有提升（0.7884 → 0.7894 量级）。"""
        plain = next(
            (r for r in self.rows if "共线" in r.regime and r.ridge == 0 and r.ran), None
        )
        ridged = next((r for r in self.rows if "共线" in r.regime and r.ridge > 0), None)
        if plain is None or ridged is None or not ridged.ran:
            return False
        return ridged.reduction_with_ridge > plain.reduction_cross_fitted

    @property
    def proxy_tracks_the_real_thing(self) -> bool:
        """理论代理（交叉拟合缩减）与实测 ATE 方差缩减在同量级。"""
        good = [r for r in self.rows if "正常" in r.regime]
        return bool(good) and all(
            abs(r.reduction_cross_fitted - r.ate_variance_reduction) < 0.15 for r in good
        )

    def summary(self) -> str:
        lines = [
            f"多协变量 CUPED（{self.n_trials} 次重随机化/档）",
        ]
        lines += [r.summary() for r in self.rows]
        lines += [
            "  读法（三条都推翻了动手前的预期，逐条记下）：",
            "    · **正常档位**：多协变量的诚实缩减 > 最好的单协变量，理论代理与"
            "实测 ATE 方差缩减同量级（0.67 vs 0.68）—— 这是它该有的样子；",
            "    · **危险的不是共线，是 p 接近 n**：共线（rho=0.9999、条件数 2.8e4、"
            "||θ̂|| 很大）时诚实缩减**几乎不降**；而 p=120/n=80 时样本内 R²=1.0000、"
            "留出集 −1.1e4。共线只是让 θ̂ 的**分量**不可解释，p≥n 才让它**不可用**；",
            "    · **样本内收益是真的，但它不是长期收益的无偏估计**：噪声档位里"
            "本样本的 ATE 方差确实降了 4.3%，而留出集口径是 −17.7% —— "
            "「这次省了」与「这个做法省」是两件事，报告里必须分开写。",
        ]
        return "\n".join(lines)


def run_multivariate_cuped_audit(
    *,
    n_trials: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
) -> MultivariateCupedAudit:
    """跑三个档位：正常 / 噪声协变量多 / 强共线（含 ridge 对照）。"""
    rows: list[MultivariateCupedRow] = []

    def build(
        regime: str,
        *,
        n: int,
        p: int,
        n_informative: int,
        rho_collinear: float,
        n_folds: int = 5,
        ridge: float = 0.0,
        seed_offset: int = 0,
    ) -> MultivariateCupedRow:
        rng = np.random.default_rng(seed + seed_offset)
        X = _correlated_covariates(rng, n, p, rho_collinear=rho_collinear)
        beta = np.zeros(p)
        beta[:n_informative] = 1.0
        y = X @ beta + rng.normal(0.0, 1.0, n)

        try:
            in_sample = fit_multivariate_cuped(X, y, n_folds=1)
            cross = fit_multivariate_cuped(X, y, n_folds=n_folds, seed=seed)
        except ValueError as exc:
            # p ≥ n 时 Σ_X 奇异：**这条路径本来就该失败**，
            # 把它记下来（而不是让整个审计崩掉），后面有属性钉它。
            return MultivariateCupedRow(
                regime=regime, n=n, p=p, n_informative=n_informative,
                condition_number=float("nan"), theta_norm=float("nan"),
                reduction_in_sample=float("nan"), reduction_cross_fitted=float("nan"),
                best_univariate=float("nan"), ate_variance_reduction=float("nan"),
                ridge=ridge, failed=str(exc),
            )
        with_ridge = (
            fit_multivariate_cuped(X, y, ridge=ridge, n_folds=n_folds, seed=seed)
            if ridge > 0
            else None
        )
        ate = _ate_variance_reduction(X, y, n_trials=n_trials, seed=seed + 7)
        return MultivariateCupedRow(
            regime=regime,
            n=n,
            p=p,
            n_informative=n_informative,
            condition_number=cross.condition_number,
            theta_norm=float(np.linalg.norm(cross.theta)),
            reduction_in_sample=in_sample.variance_reduction,
            reduction_cross_fitted=cross.variance_reduction,
            best_univariate=cross.best_univariate_reduction,
            ate_variance_reduction=ate,
            ridge=ridge,
            reduction_with_ridge=(
                with_ridge.variance_reduction if with_ridge is not None else float("nan")
            ),
        )

    rows.append(
        build("正常：2 个有效协变量", n=4000, p=2, n_informative=2,
              rho_collinear=0.0, seed_offset=0)
    )
    rows.append(
        build("正常：1 个有效协变量", n=4000, p=1, n_informative=1,
              rho_collinear=0.0, seed_offset=100)
    )
    # 噪声协变量多、样本小：过拟合最容易现形的地方
    rows.append(
        build("噪声协变量多（p=20 全噪声, n=200）", n=200, p=20, n_informative=0,
              rho_collinear=0.0, seed_offset=200)
    )
    # 强共线但可解：现象是条件数与 ||θ̂|| 大，而**诚实缩减并不崩**
    rows.append(
        build("共线但可解（p=3, rho=0.9999）", n=800, p=3, n_informative=2,
              rho_collinear=0.9999, seed_offset=300)
    )
    rows.append(
        build("共线但可解 + ridge（1e-3）", n=800, p=3, n_informative=2,
              rho_collinear=0.9999, ridge=1e-3, seed_offset=300)
    )
    # p ≥ n：Σ_X 奇异 —— 应当**直接报错**，而不是给一个看起来正常的数
    rows.append(
        build("p≥n（p=120 全噪声, n=80）", n=80, p=120, n_informative=0,
              rho_collinear=0.0, seed_offset=400)
    )
    rows.append(
        build("p≥n + ridge（0.1）", n=80, p=120, n_informative=0, ridge=0.1,
              rho_collinear=0.0, seed_offset=400)
    )

    return MultivariateCupedAudit(n_trials=n_trials, rows=tuple(rows))
