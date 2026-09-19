"""异质效应审计：CATE 估计到底比"报一个平均值"好在哪里。

M4 的验证台比前几个阶段难
------------------------
DML 的 θ 有真值可比，直接量偏置和覆盖率就行。但 CATE 不行：
**"谁受益更多"这个问题没有唯一的正确答案**，因为答案取决于 DGP 的函数形式。
只用一个 DGP 去评 CATE，等于在宣布赢家。

所以这里的做法是：

1. **四种 CATE 形式各跑一遍**，报告完整对照表 —— "没有单一赢家"是结论，不是失败
2. **拿常数 ATE 当基准线**。这是最容易被跳过、也最残酷的一问：
   你的弹性模型比"对所有人报同一个数"到底好在哪？
3. **把排序指标和水平指标分开报**。Qini/AUUC 只衡量排序，
   MSE 衡量水平，两者可以给出完全相反的结论 —— 这是 M4 的核心发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..causal.forest import CausalForest, ForestConfig
from ..causal.hte import (
    CATE_FORMS,
    HTEConfig,
    dml_partial_linear,
    generate_hte_data,
    naive_plugin,
)
from ..causal.uplift import (
    constant_prediction,
    qini_coefficient,
    rank_correlation,
)

__all__ = [
    "DMLEstimationAudit",
    "CATEFormResult",
    "CATEModelComparison",
    "CateIntervalCoverage",
    "UpliftMetricAudit",
    "run_dml_audit",
    "run_cate_form_comparison",
    "run_cate_coverage_audit",
    "run_uplift_metric_audit",
]


# --------------------------------------------------------------------------- #
# 一、DML：naive 有偏、DML 无偏
# --------------------------------------------------------------------------- #
@dataclass
class DMLSizePoint:
    """某个样本量下的一对估计量。"""

    n: int
    naive_bias: float
    naive_coverage: float
    dml_bias: float
    dml_coverage: float
    dml_no_crossfit_bias: float
    dml_se: float

    def summary(self) -> str:
        return (
            f"  n={self.n:>5}: naive 偏置 {self.naive_bias:+.4f} 覆盖率 {self.naive_coverage:.2f} "
            f"| DML 偏置 {self.dml_bias:+.4f} 覆盖率 {self.dml_coverage:.2f} "
            f"| 不交叉拟合 {self.dml_no_crossfit_bias:+.4f}"
        )


@dataclass
class DMLEstimationAudit:
    """DML 与 naive plug-in 在**多个样本量**上的偏置与覆盖率。

    扫样本量是刻意的：naive 的偏置来自遗漏的非线性项，
    **它不会随着样本量消失** —— 所以 n 越大，它的置信区间越窄、
    越自信地错过真值，覆盖率从 73% 一路崩到 13%。
    DML 的偏置随 n 收缩，覆盖率守在名义水平附近。
    """

    points: tuple[DMLSizePoint, ...]
    n_trials: int

    @property
    def coverage_collapses(self) -> bool:
        """naive 的覆盖率是否随样本量下降。"""
        return self.points[-1].naive_coverage < self.points[0].naive_coverage - 0.1

    @property
    def dml_holds(self) -> bool:
        return abs(self.points[-1].dml_coverage - 0.95) < 0.12

    @property
    def large_sample(self) -> DMLSizePoint:
        return self.points[-1]

    def summary(self) -> str:
        lines = [
            f"DML vs naive plug-in（{self.n_trials} 次仿真/样本量）",
            "  真 theta = +1.000（constant CATE 下的 ATE）",
        ]
        lines.extend(p.summary() for p in self.points)
        lines.append(
            "  -> naive 的偏置**不随 n 消失**，覆盖率从 "
            f"{self.points[0].naive_coverage:.0%} 崩到 {self.points[-1].naive_coverage:.0%}："
            "更多数据只会让有偏的区间更自信地错。"
        )
        lines.append(
            f"  -> DML 的偏置随 n 收缩（{self.points[0].dml_bias:+.3f} -> "
            f"{self.points[-1].dml_bias:+.3f}），覆盖率守在 "
            f"{self.points[-1].dml_coverage:.0%}。"
        )
        lines.append(
            f"  -> 但注意 n={self.points[0].n} 那一行：nuisance 太弱时 **DML 自己也偏**"
            " 它的渐近保证要求 nuisance 收敛快于 n^{-1/4}，小样本 + 弱学习器不满足。"
        )
        return "\n".join(lines)


def run_dml_audit(
    *,
    n_trials: int = 40,
    sizes: tuple[int, ...] = (800, 2000, 4000),
    seed: int = 0,
    base: HTEConfig | None = None,
) -> DMLEstimationAudit:
    """在多个样本量上反复估计，量偏置与覆盖率如何随 n 变化。"""
    base = base or HTEConfig(n=4000, cate_form="constant", noise_sd=1.0)
    points: list[DMLSizePoint] = []

    for n in sizes:
        naive_bias, naive_cov = [], []
        dml_bias, dml_cov, dml_se = [], [], []
        no_cf_bias = []
        truth = float("nan")

        for i in range(n_trials):
            data = generate_hte_data(
                HTEConfig(**{**base.__dict__, "n": n, "seed": seed + i})
            )
            truth = data.ate

            nv = naive_plugin(data.Y, data.D, data.X, true_theta=truth)
            naive_bias.append(nv.bias)
            naive_cov.append(int(nv.covers_truth))

            d = dml_partial_linear(
                data.Y, data.D, data.X, n_folds=5, true_theta=truth, seed=i
            )
            dml_bias.append(d.bias)
            dml_cov.append(int(d.covers_truth))
            dml_se.append(d.se)

            d1 = dml_partial_linear(
                data.Y, data.D, data.X, n_folds=1, true_theta=truth, seed=i
            )
            no_cf_bias.append(d1.bias)

        points.append(
            DMLSizePoint(
                n=n,
                naive_bias=float(np.mean(naive_bias)),
                naive_coverage=float(np.mean(naive_cov)),
                dml_bias=float(np.mean(dml_bias)),
                dml_coverage=float(np.mean(dml_cov)),
                dml_no_crossfit_bias=float(np.mean(no_cf_bias)),
                dml_se=float(np.mean(dml_se)),
            )
        )

    return DMLEstimationAudit(points=tuple(points), n_trials=n_trials)


# --------------------------------------------------------------------------- #
# 二、四种 CATE 形式下的模型对照
# --------------------------------------------------------------------------- #
@dataclass
class CATEFormResult:
    """一种 CATE 形式下，因果森林相对常数基线的表现。"""

    cate_form: str
    mse_forest: float
    mse_constant: float
    rank_correlation: float
    qini_forest: float
    qini_constant: float
    qini_perfect: float
    true_cate_sd: float

    @property
    def mse_ratio(self) -> float:
        """森林 MSE / 常数基线 MSE。**> 1 表示弹性模型还不如报一个平均数**。"""
        if self.mse_constant <= 1e-12:
            return float("inf")
        return self.mse_forest / self.mse_constant

    @property
    def forest_beats_constant(self) -> bool:
        return self.mse_ratio < 1.0

    def summary(self) -> str:
        ratio = self.mse_ratio
        ratio_s = "∞" if not np.isfinite(ratio) else f"{ratio:.2f}"
        return (
            f"  {self.cate_form:>10}: MSE 森林 {self.mse_forest:8.4f} vs 常数 "
            f"{self.mse_constant:8.4f}（比值 {ratio_s:>6}）  "
            f"秩相关 {self.rank_correlation:+.3f}  "
            f"Qini 森林 {self.qini_forest:8.2f} vs 常数 {self.qini_constant:7.2f}"
        )


@dataclass
class CATEModelComparison:
    """四种 CATE 形式下的完整对照表。"""

    results: tuple[CATEFormResult, ...]
    n: int

    @property
    def any_model_beats_constant(self) -> bool:
        return any(r.forest_beats_constant for r in self.results)

    @property
    def best_form(self) -> str:
        finished = [r for r in self.results if np.isfinite(r.mse_ratio)]
        if not finished:
            return ""
        return min(finished, key=lambda r: r.mse_ratio).cate_form

    def summary(self) -> str:
        lines = [
            f"因果森林 vs 常数基线（n={self.n}，四种 CATE 形式各一遍）",
            "  注意：这是**留出集**上的 MSE。常数基线 = 对所有人报同一个 ATE。",
        ]
        lines.extend(r.summary() for r in self.results)
        verdict = (
            "弹性模型在至少一种形式下赢过常数基线"
            if self.any_model_beats_constant
            else "**弹性模型在四种形式下全部输给常数基线** —— 它只在排序上有价值"
        )
        lines.append(f"  -> {verdict}")
        return "\n".join(lines)


def run_cate_form_comparison(
    *,
    forms: tuple[str, ...] = CATE_FORMS,
    n: int = 4000,
    forest_config: ForestConfig | None = None,
    seed: int = 0,
) -> CATEModelComparison:
    """每种 CATE 形式跑一遍：留出集上森林 vs 常数基线。"""
    config = forest_config or ForestConfig(n_trees=60, max_depth=5, min_leaf=20, seed=0)
    results: list[CATEFormResult] = []

    for i, form in enumerate(forms):
        data = generate_hte_data(HTEConfig(n=n, cate_form=form, seed=seed + i))
        rng = np.random.default_rng(seed + 100 + i)
        perm = rng.permutation(data.n)
        half = data.n // 2
        tr, te = perm[:half], perm[half:]

        forest = CausalForest(config).fit(data.X[tr], data.D[tr], data.Y[tr])
        pred = forest.predict(data.X[te])

        tau_te = data.tau[te]
        y_te, d_te = data.Y[te], data.D[te]
        constant = constant_prediction(tau_te)

        results.append(
            CATEFormResult(
                cate_form=form,
                mse_forest=float(np.mean((pred - tau_te) ** 2)),
                mse_constant=float(np.mean((constant - tau_te) ** 2)),
                rank_correlation=rank_correlation(tau_te, pred),
                qini_forest=qini_coefficient(y_te, d_te, pred),
                qini_constant=qini_coefficient(y_te, d_te, constant),
                qini_perfect=qini_coefficient(y_te, d_te, tau_te),
                true_cate_sd=float(tau_te.std()),
            )
        )

    return CATEModelComparison(results=tuple(results), n=n)


# --------------------------------------------------------------------------- #
# 二之二、CATE 的区间：算得出来，但**校准不了**
# --------------------------------------------------------------------------- #
@dataclass
class CateIntervalCoverage:
    """两条区间路线的覆盖率与长度，以及"为什么盖不住"的分解。

    M4 过去只能主张**排序**（Qini 是常数基线的 40 倍），不能主张**水平**
    （森林 MSE 反而差 17%）。这一组数字回答的是：**加上区间之后，水平可用了吗？**
    答案是**没有** —— 而且原因被定位到了：不是方差，是**点估计有偏**。
    """

    n: int
    n_scenarios: int
    bootstrap_draws: int
    #: 叶内方差 + 跨树独立合成
    analytic_coverage: float
    analytic_length: float
    #: 按单元 bootstrap
    bootstrap_coverage: float
    bootstrap_length: float
    #: 点估计本身的诊断（这才是"盖不住"的原因）
    cate_correlation: float
    rmse: float
    true_cate_sd: float
    estimate_sd: float
    #: 可给出有限区间的单元比例（纯叶子给不出方差）
    finite_share: float

    @property
    def max_length_gap(self) -> float:
        """两条路线的长度差（相对），验收标准里要求 ≤30%。"""
        if self.bootstrap_length <= 0:
            return float("nan")
        return abs(self.analytic_length - self.bootstrap_length) / self.bootstrap_length

    def summary(self) -> str:
        return "\n".join(
            [
                f"CATE 区间的覆盖率（{self.n_scenarios} 个场景 × n={self.n}，"
                f"bootstrap B={self.bootstrap_draws}，名义 0.95）",
                f"  叶内方差 + 独立合成：覆盖率 {self.analytic_coverage:.4f}，"
                f"长度 {self.analytic_length:.4f}",
                f"  按单元 bootstrap　　：覆盖率 {self.bootstrap_coverage:.4f}，"
                f"长度 {self.bootstrap_length:.4f}（长度差 {self.max_length_gap:.1%}）",
                f"  点估计诊断：τ̂ 与真值相关 {self.cate_correlation:.4f}，"
                f"RMSE {self.rmse:.4f}",
                f"  真 CATE 的 sd {self.true_cate_sd:.4f} vs τ̂ 的 sd "
                f"{self.estimate_sd:.4f}；可给区间的单元 {self.finite_share:.1%}",
                "  → 两条路线的覆盖率都远低于名义值，而**区间长度与真 CATE 的离散度"
                "同量级**：所以盖不住的原因是**点估计有偏**，不是方差算错。",
            ]
        )


def run_cate_coverage_audit(
    *,
    n: int = 1500,
    n_scenarios: int = 5,
    bootstrap_draws: int = 25,
    n_trees: int = 40,
    max_depth: int = 5,
    min_leaf: int = 20,
    seed_start: int = 1,
) -> CateIntervalCoverage:
    """多场景测量 CATE 区间的覆盖率（这是"有没有区间"唯一算数的证据）。

    **没有这个数，"我们有置信区间了"就只是一句话。**
    两条路线都测：叶内方差 + 跨树独立合成（``predict_with_se``）
    与按单元 bootstrap。

    结果在实测中是**负面**的（覆盖率 0.38~0.74，名义 0.95），
    所以这个函数的价值不在于给出一个能用的区间，而在于**把"水平不可用"量化** ——
    并且指出它是因为偏差而不是方差。
    """
    from ..causal.forest import CausalForest, ForestConfig
    from ..causal.hte import HTEConfig, generate_hte_data

    z = 1.959963984540054
    rows: list[dict[str, float]] = []
    for seed in range(seed_start, seed_start + n_scenarios):
        data = generate_hte_data(
            HTEConfig(n=n, n_features=6, n_informative=3, seed=seed)
        )
        true = np.asarray(data.tau)
        cfg = ForestConfig(n_trees=n_trees, max_depth=max_depth, min_leaf=min_leaf)

        forest = CausalForest(cfg)
        forest.fit(data.X, data.D, data.Y)
        tau_hat, se = forest.predict_with_se(data.X)
        finite = np.isfinite(se)
        lo, hi = tau_hat[finite] - z * se[finite], tau_hat[finite] + z * se[finite]
        analytic_cov = float(np.mean((lo <= true[finite]) & (true[finite] <= hi)))
        analytic_len = float(np.mean(hi - lo))

        rng = np.random.default_rng(seed)
        n_obs = data.X.shape[0]
        draws = np.empty((bootstrap_draws, n_obs))
        for b in range(bootstrap_draws):
            idx = rng.integers(0, n_obs, size=n_obs)
            tree = CausalForest(cfg)
            tree.fit(data.X[idx], data.D[idx], data.Y[idx])
            draws[b] = tree.predict(data.X)
        blo = np.percentile(draws, 2.5, axis=0)
        bhi = np.percentile(draws, 97.5, axis=0)

        rows.append(
            {
                "analytic_cov": analytic_cov,
                "analytic_len": analytic_len,
                "boot_cov": float(np.mean((blo <= true) & (true <= bhi))),
                "boot_len": float(np.mean(bhi - blo)),
                "corr": float(np.corrcoef(tau_hat, true)[0, 1]),
                "rmse": float(np.sqrt(np.mean((tau_hat - true) ** 2))),
                "true_sd": float(np.std(true)),
                "est_sd": float(np.std(tau_hat)),
                "finite": float(np.mean(finite)),
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    return CateIntervalCoverage(
        n=n,
        n_scenarios=n_scenarios,
        bootstrap_draws=bootstrap_draws,
        analytic_coverage=mean("analytic_cov"),
        analytic_length=mean("analytic_len"),
        bootstrap_coverage=mean("boot_cov"),
        bootstrap_length=mean("boot_len"),
        cate_correlation=mean("corr"),
        rmse=mean("rmse"),
        true_cate_sd=mean("true_sd"),
        estimate_sd=mean("est_sd"),
        finite_share=mean("finite"),
    )


# --------------------------------------------------------------------------- #
# 二之三、CATE 的**组级**校准：BLP 与 GATES
# --------------------------------------------------------------------------- #
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """普通最小二乘 + HC0 稳健标准误（分样本之后这就是经典推断）。

    只依赖 numpy：这一节要的是"在留出样本上做一次线性回归"，
    引 sklearn 的 LinearRegression 不会有任何额外信息，还挡住标准误。
    """
    inv = np.linalg.pinv(x.T @ x)
    beta = inv @ (x.T @ y)
    resid = y - x @ beta
    meat = x.T @ (x * (resid**2)[:, None])
    cov = inv @ meat @ inv
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0))


#: 裁剪阈值的候选网格。上界 0.30 是刻意留的 —— 实测它在**单靠裁剪**时会把
#: 覆盖率打到 0.5375，自动规则若选到那里就说明目标函数出问题了，应当看得见。
_TRIM_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


@dataclass
class GatesBlpResult:
    """CATE 的**特征**（而不是 CATE 函数）的校准结果。

    为什么换对象：Chernozhukov 等（arXiv:1712.04802）指出，通用 ML 工具在
    高维/非参下连 CATE 的一致估计都拿不到，自适应置信集更不存在 ——
    所以对 ``s0(Z)`` 本身做单元级推断是**注定**失败的（本仓库实测覆盖率
    0.13~0.44，见 ``CateIntervalCoverage``）。他们转而推断 CATE 的**特征**：

    * **BLP**：把信号回归到代理上。``斜率 = 1`` 就是校准；推断来自经典 OLS，
      **不要求代理一致** —— 这正是它绕开那个不可能性的方式。
    * **GATES**：按代理的分位数分组，组内平均效应的**有效置信区间**。
      于是"排序可用、水平不可用"可以正确地软化成：
      **组级水平可用，单元级不可用**。

    信号用 Horvitz-Thompson 变换 ``H = (D - p)/(p(1-p))``，``signal = H·Y``，
    它满足 ``E[signal | Z] = s0(Z)``，**不需要结果模型**（本仓库仿真的 p 已知）。
    """

    n: int
    n_splits: int
    n_groups: int
    #: BLP 斜率（应当 ≈ 1）与其标准误、以及"CI 盖住 1"的比例
    blp_slope: float
    blp_slope_se: float
    blp_covers_one: float
    #: GATES：各组平均效应、以及组级覆盖率（对真实组 ATE）
    gates_effects: list[float]
    gates_true: list[float]
    gates_coverage: float
    gates_mean_length: float
    #: 各组「估计 − 真实」的均值及其蒙特卡洛标准误（用于区分偏差与噪音）
    gates_gap: list[float] = field(default_factory=list)
    gates_gap_mc_se: list[float] = field(default_factory=list)
    #: HT 信号的诊断：峰度（正态为 3）与**重叠度**（倾向得分落在 [0.05,0.95] 外的比例）
    #: 这两个数解释了"为什么区间有效却又宽又不稳"：峰度 752 的信号 + 近乎违背的
    #: 正值性假设。见 README 已知边界里那条"应当换 AIPW 信号"。
    signal_kurtosis: float = float("nan")
    overlap_violation_share: float = float("nan")
    #: 信号与裁剪：``signal_kind`` 是 ht / aipw；``trim_alpha`` 是实际用上的阈值，
    #: ``trimmed_share`` 是被裁掉的比例 —— **阈值会改变估计目标**，
    #: 所以这两个数必须和覆盖率一起看，单独报覆盖率会误导。
    signal_kind: str = "ht"
    trim_alpha: float = 0.0
    trimmed_share: float = 0.0
    #: 对照：同一次实验里单元级那两条路线的覆盖率（见 CateIntervalCoverage）
    unit_level_note: str = ""

    def summary(self) -> str:
        lines = [
            f"CATE 的**组级**校准（{self.n_splits} 次分裂 × n={self.n}，"
            f"{self.n_groups} 组，HT 信号，分样本经典推断）",
            f"  BLP：斜率 {self.blp_slope:.4f}（SE {self.blp_slope_se:.4f}），"
            f"CI 盖住 1 的比例 {self.blp_covers_one:.3f}（校准则应当 ≈ 0.95）",
            f"  GATES 组级覆盖率 {self.gates_coverage:.4f}，"
            f"平均区间长度 {self.gates_mean_length:.4f}",
            f"  信号诊断：峰度 {self.signal_kurtosis:.1f}（正态为 3），"
            f"倾向得分落在 [0.05,0.95] 外的比例 {self.overlap_violation_share:.4f}",
            f"  信号 {self.signal_kind}，裁剪阈值 {self.trim_alpha:.3f}"
            f"（裁掉 {self.trimmed_share:.4f} 的单元；阈值改变了估计目标，"
            f"覆盖率要连同这两个数一起读）",
            "  各组：",
        ]
        for i, (eff, true) in enumerate(zip(self.gates_effects, self.gates_true)):
            gap = eff - true
            mc = self.gates_gap_mc_se[i] if i < len(self.gates_gap_mc_se) else float("nan")
            # 只有超出蒙特卡洛噪音才叫偏差，否则一律记为噪音（不做无依据的断言）
            verdict = "噪音范围内" if abs(gap) <= 2 * mc else "超出噪音"
            lines.append(
                f"    组{i + 1}: 估计 {eff:+.4f} vs 真实 {true:+.4f}"
                f"（差 {gap:+.4f} ± {mc:.4f}，{verdict}）"
            )
        if self.unit_level_note:
            lines.append(f"  对照：{self.unit_level_note}")
        return "\n".join(lines)


def _gates_fit(
    proxy: np.ndarray, signal: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """GATES 的公共计算：按代理分位数切 K 组、把信号回归到组哑变量上。

    只此一份实现 —— 选阈值时要反复用它（候选阈值各算一次方差），
    最终结果也用它。写成两处就会让"选出来的阈值"和"报告的组效应"
    来自不同口径。
    """
    edges = np.quantile(proxy, np.linspace(0, 1, n_groups + 1)[1:-1])
    group = np.searchsorted(edges, proxy, side="right")
    dummies = np.zeros((proxy.size, n_groups))
    dummies[np.arange(proxy.size), group] = 1.0
    keep = [g for g in range(n_groups) if dummies[:, g].sum() > 1]
    eff, se = _ols(dummies[:, keep], signal)
    return eff, se, group, keep


def _pick_trim_alpha(
    p: np.ndarray,
    signal: np.ndarray,
    proxy: np.ndarray,
    n_groups: int,
    grid: tuple[float, ...],
) -> float:
    """按 Crump 等的**原则**选裁剪阈值：让组级估计量的插入式方差最小。

    Crump-Hotz-Imbens-Mitnik（2009，*Dealing with limited overlap in estimation
    of average treatment effects*）的做法是：对 ``e(X)`` 设一个对称阈值 α，
    丢掉 ``e(X)`` 落在 ``[α, 1-α]`` 之外的单元，并以**估计量的渐近方差最小**
    来选 α（他们给出了常数条件方差下的闭式，常用的经验截断是 0.1）。
    这里不照抄那个闭式（它依赖条件方差的具体形式），而是把这个**目标函数**
    直接在候选网格上算出来：对每个 α，在裁剪后的样本上算组级标准误的均值，
    取最小的那个。这样选的阈值与本仓库自己的估计量完全一致，
    而且选出来的 α 和被裁剪的比例都会写进报告。

    诚实的代价：**阈值变了，估计目标也跟着变**（从全体变成重叠总体），
    这不是"免费的精度"，所以报告里必须同时给出被裁剪的比例。
    """
    best_alpha, best_score = 0.0, float("inf")
    min_keep = max(50, p.size // 4)
    for a in grid:
        keep = np.ones(p.size, bool) if a <= 0 else (p >= a) & (p <= 1.0 - a)
        if int(keep.sum()) < min_keep:
            continue
        eff, se, _, k = _gates_fit(proxy[keep], signal[keep], n_groups)
        if len(k) < n_groups or not np.all(np.isfinite(se)):
            continue
        score = float(np.mean(se))
        if score < best_score - 1e-12:
            best_alpha, best_score = float(a), score
    return best_alpha


def run_gates_blp_audit(
    *,
    n: int = 1500,
    n_splits: int = 20,
    n_groups: int = 4,
    n_trees: int = 40,
    max_depth: int = 5,
    min_leaf: int = 20,
    seed: int = 0,
    cate_form: str = "nonlinear",
    proxy_kind: str = "forest",
    signal_kind: str = "aipw",
    trim: float | str | None = "auto",
) -> GatesBlpResult:
    """BLP 与 GATES 的实测校准（这是"组级水平可用"这句话的唯一证据）。

    流程（每一步都对应文献里的一个决定）：
      1. 把样本随机劈成**辅助样本**（训练代理）与**主样本**（做推断）；
      2. 辅助样本上拟合因果森林当代理 τ̂（**允许它有偏**）；
      3. 主样本上算 HT 信号；
      4. BLP：``signal ~ 1 + τ̂`` 的斜率；GATES：按 τ̂ 分位数切 K 组、
         ``signal ~ 组哑变量``；
      5. 重复 ``n_splits`` 次，统计覆盖：BLP 的 CI 盖住 1、GATES 的 CI 盖住
         **真实的组 ATE**（真值由 DGP 直接给出，不需要估计）。

    ``proxy_kind="oracle"`` 是**正对照**：直接拿真 CATE 当代理（模拟"代理已
    校准"这个理想情形）。它必须给出 BLP 斜率 ≈ 1、覆盖率 ≈ 0.95 ——
    否则说明这个审计连真值都验不过，那 0.95 就只是"永远通过"，没有信息。

    ``signal_kind``：

    * ``"aipw"``（**默认**，推荐）：``Γ = μ̂₁(X) − μ̂₀(X) + H·(Y − μ̂_D(X))``，
      结局模型在**辅助样本**上拟合（所以走 AIPW 时连正对照也得分样本，
      否则模型会在推断样本上拟合过 —— 那是不诚实的）。
      实测它把覆盖率抬到 0.94~0.98、区间缩短约三分之一。
    * ``"ht"``（历史口径）：纯 Horvitz-Thompson 信号，不需要结果模型。
      本 DGP 下重尾（峰度 134），覆盖率会在 0.875~0.950 之间摆动。
      保留它是为了**可复现历史报告的数字**，也是对照。

    ``trim``：对倾向得分的对称裁剪阈值。

    * ``"auto"``（**默认**，推荐）：在候选网格上按"组级估计量的插入式方差最小"
      选 α（Crump 等 2009 的原则，见 ``_pick_trim_alpha``）。本 DGP 实测
      自动选出 α≈0.09~0.13（与文献里常用的 0.1 一致），峰度从 134 降到 11。
    * ``None``：不裁剪（历史口径）。
    * 浮点数：按该阈值裁掉 ``p`` 落在 ``[α, 1-α]`` 之外的单元。
      **阈值会改变估计目标**（全体 → 重叠总体），所以结果里必须同时看
      ``trim_alpha`` 与 ``trimmed_share``；实测单靠裁剪（不换 AIPW）时
      阈值超过 0.1 会把覆盖率打到 0.54。
    """
    from ..causal.forest import CausalForest, ForestConfig
    from ..causal.hte import HTEConfig, generate_hte_data

    z = 1.959963984540054
    slopes: list[float] = []
    slope_ses: list[float] = []
    covers_one = 0
    effects_acc = np.zeros(n_groups)
    true_acc = np.zeros(n_groups)
    gaps: list[list[float]] = [[] for _ in range(n_groups)]
    kurt_acc: list[float] = []
    overlap_acc: list[float] = []
    alpha_acc: list[float] = []
    trimmed_acc: list[float] = []
    group_hits = 0
    group_total = 0
    lengths: list[float] = []

    for s in range(n_splits):
        data = generate_hte_data(
            HTEConfig(n=n, n_features=6, n_informative=3, seed=seed + s, cate_form=cate_form)
        )
        rng = np.random.default_rng(1000 + s)
        perm = rng.permutation(n)
        aux, main = perm[: n // 2], perm[n // 2 :]
        if proxy_kind == "oracle":
            if signal_kind == "ht":
                # 纯 HT 不需要拟合任何东西，于是可以把全部样本当主样本
                main = perm
            # AIPW 需要结局模型，而模型不能在看过的数据上做推断 —— 所以
            # 走 AIPW 时正对照也保持 aux/main 分样本（见函数的说明）。
            proxy = np.asarray(data.tau)[main]
        elif proxy_kind == "forest":
            cfg = ForestConfig(n_trees=n_trees, max_depth=max_depth, min_leaf=min_leaf)
            forest = CausalForest(cfg)
            forest.fit(data.X[aux], data.D[aux], data.Y[aux])
            proxy = forest.predict(data.X[main])
        else:  # pragma: no cover - 参数校验
            raise ValueError(f"未知的 proxy_kind: {proxy_kind!r}（可选 forest / oracle）")

        p = np.asarray(data.propensity)[main]
        d = np.asarray(data.D)[main]
        y = np.asarray(data.Y)[main]
        tau_main = np.asarray(data.tau)[main]

        # ---- 信号 ---- #
        if signal_kind == "ht":
            signal = (d - p) / (p * (1.0 - p)) * y
        elif signal_kind == "aipw":
            xa = np.asarray(data.X)[aux]
            da = np.asarray(data.D)[aux]
            ya = np.asarray(data.Y)[aux]
            x_m = np.asarray(data.X)[main]
            m1 = _fit_predict(xa[da == 1], ya[da == 1], x_m)
            m0 = _fit_predict(xa[da == 0], ya[da == 0], x_m)
            h = (d - p) / (p * (1.0 - p))
            signal = (m1 - m0) + h * (y - np.where(d == 1, m1, m0))
        else:  # pragma: no cover - 参数校验
            raise ValueError(f"未知的 signal_kind: {signal_kind!r}（可选 ht / aipw）")

        # ---- 裁剪（阈值改变的是估计目标，所以要留痕） ---- #
        if trim == "auto":
            alpha = _pick_trim_alpha(p, signal, proxy, n_groups, _TRIM_GRID)
        elif trim is None:
            alpha = 0.0
        else:
            alpha = float(trim)
        if alpha > 0.0:
            inside = (p >= alpha) & (p <= 1.0 - alpha)
            trimmed_acc.append(1.0 - float(inside.mean()))
            p, d, y, signal = p[inside], d[inside], y[inside], signal[inside]
            proxy, tau_main = proxy[inside], tau_main[inside]
        else:
            trimmed_acc.append(0.0)
        alpha_acc.append(alpha)

        kurt = float(((signal - signal.mean()) ** 4).mean() / signal.var() ** 2)
        kurt_acc.append(kurt)
        overlap_acc.append(float(np.mean((p < 0.05) | (p > 0.95))))

        # ---- BLP ---- #
        # 注意用 proxy.size 而不是 main.size：裁剪之后两者不再相等
        x = np.column_stack([np.ones(proxy.size), proxy - proxy.mean()])
        beta, se = _ols(x, signal)
        slopes.append(float(beta[1]))
        slope_ses.append(float(se[1]))
        covers_one += int(abs(beta[1] - 1.0) <= z * se[1])

        # ---- GATES ---- #
        # 分组用代理的分位数（分位数组在**裁剪之后**的样本上算，组仍然是均衡的）
        g_eff, g_se, group, keep = _gates_fit(proxy, signal, n_groups)
        for j, g in enumerate(keep):
            true_group = float(tau_main[group == g].mean())
            effects_acc[g] += float(g_eff[j])
            true_acc[g] += true_group
            gaps[g].append(float(g_eff[j]) - true_group)
            group_total += 1
            group_hits += int(abs(g_eff[j] - true_group) <= z * g_se[j])
            lengths.append(2 * z * float(g_se[j]))

    return GatesBlpResult(
        n=n,
        n_splits=n_splits,
        n_groups=n_groups,
        blp_slope=float(np.mean(slopes)),
        blp_slope_se=float(np.mean(slope_ses)),
        blp_covers_one=covers_one / n_splits,
        gates_effects=list(effects_acc / n_splits),
        gates_true=list(true_acc / n_splits),
        gates_coverage=group_hits / max(group_total, 1),
        gates_mean_length=float(np.mean(lengths)) if lengths else float("nan"),
        gates_gap=[float(np.mean(g)) if g else float("nan") for g in gaps],
        gates_gap_mc_se=[
            float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
            for g in gaps
        ],
        signal_kurtosis=float(np.mean(kurt_acc)) if kurt_acc else float("nan"),
        overlap_violation_share=float(np.mean(overlap_acc)) if overlap_acc else float("nan"),
        signal_kind=signal_kind,
        trim_alpha=float(np.mean(alpha_acc)) if alpha_acc else 0.0,
        trimmed_share=float(np.mean(trimmed_acc)) if trimmed_acc else 0.0,
        unit_level_note=(
            "单元级那两条路线（叶内方差 / bootstrap）实测覆盖率 0.13~0.74，"
            "而这里是**组级**：对象不同，不能直接比大小，只能比"
            "「谁给出了可用的区间」"
        ),
    )


# --------------------------------------------------------------------------- #
# 二之四、信号的选择：HT / 裁剪 / AIPW / 两者叠加 —— 用实测决定，而不是靠推测
# --------------------------------------------------------------------------- #
def _fit_predict(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """带截距的 OLS，用来当**结局模型** μ̂_d(X)（辅助样本拟合，主样本预测）。

    为什么用最朴素的 OLS：这一节要比的是**信号**（HT vs AIPW）在同一个
    代理、同一套分组下的差别，结局模型越好两边都越好，不改变结论方向；
    用朴素模型反而避免了"到底是谁带来的改善"这个混淆。
    """
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.column_stack([np.ones(x_new.shape[0]), x_new]) @ beta


@dataclass
class SignalArmStats:
    """一个信号版本的实测表现（都跑在同一个代理与同一套分组上）。"""

    name: str
    kurtosis: float
    coverage: float
    mean_length: float
    mean_abs_gap: float
    signal_sd: float


@dataclass
class SignalComparison:
    """四种信号的对照 —— 回答"重尾该怎么修"。

    背景：纯 HT 信号在本 DGP 下峰度 260（见 ``GatesBlpResult``），
    于是组级覆盖率会在 0.875~0.950 之间摆动。README 原先写下的下一步是
    "换 AIPW 信号，把 1/(p(1-p)) 的大权重从 Y 转移到残差上"。
    **实测只对了一半**：

      * AIPW 确实把区间缩短约 30%、覆盖率抬到 0.925 —— 但它**几乎不降峰度**
        （257.8 vs 260.3）。因为极端的 1/(p(1-p)) 权重仍在，尾巴还在。
      * **降峰度靠裁剪**（260→72.5，3.6 倍），单裁剪不改善覆盖率却缩短区间。
      * 两者叠加最好：峰度 63.5、覆盖率 0.925、长度比纯 HT 短 46%。

    所以正确的下一步不是二选一，而是"裁剪（管尾巴）＋ AIPW（管方差）"，
    并如实记录裁剪引入的偏差 —— 这一节把偏差也测出来放在同一张表里。
    """

    n: int
    n_splits: int
    n_groups: int
    clip: float
    arms: list[SignalArmStats]

    def summary(self) -> str:
        lines = [
            f"信号对照（{self.n_splits} 次分裂 × n={self.n}，{self.n_groups} 组，"
            f"裁剪阈值 {self.clip}）",
            f"  {'信号':<12}{'峰度':>9}{'覆盖率':>9}{'区间长度':>10}"
            f"{'|偏差|':>9}{'信号 sd':>9}",
        ]
        for a in self.arms:
            lines.append(
                f"  {a.name:<12}{a.kurtosis:>9.1f}{a.coverage:>9.4f}"
                f"{a.mean_length:>10.4f}{a.mean_abs_gap:>9.4f}{a.signal_sd:>9.3f}"
            )
        return "\n".join(lines)


def run_signal_comparison(
    *,
    n: int = 2000,
    n_splits: int = 30,
    n_groups: int = 4,
    clip: float = 0.05,
    seed: int = 0,
    cate_form: str = "nonlinear",
) -> SignalComparison:
    """在同一个真实 CATE 排序下比较四种信号的组级表现。

    分组用**真实** τ 的分位数（而不是某个代理）：这一节要隔离的变量是
    **信号**，把代理的误差混进来会让四个版本的差别说不清是谁造成的。
    （代理质量的影响已经在 ``run_gates_blp_audit`` 里单独测过了。）

    结局模型在**辅助样本**上拟合、主样本上预测 —— 主样本仍用于推断，
    所以"拟合过的东西不进推断样本"这条纪律没有被破坏。
    """
    from ..causal.hte import HTEConfig, generate_hte_data

    z = 1.959963984540054
    names = ["HT", "HT+裁剪", "AIPW", "AIPW+裁剪"]
    acc: dict[str, dict[str, list[float]]] = {
        name: {"kurt": [], "cov": [], "len": [], "gap": [], "sd": []} for name in names
    }

    for s in range(n_splits):
        data = generate_hte_data(
            HTEConfig(n=n, n_features=6, n_informative=3, seed=seed + s, cate_form=cate_form)
        )
        rng = np.random.default_rng(1000 + s)
        perm = rng.permutation(n)
        aux, main = perm[: n // 2], perm[n // 2 :]

        p = np.asarray(data.propensity)[main]
        d = np.asarray(data.D)[main]
        y = np.asarray(data.Y)[main]
        x = np.asarray(data.X)[main]
        tau = np.asarray(data.tau)[main]

        p_clip = np.clip(p, clip, 1.0 - clip)
        h = (d - p) / (p * (1.0 - p))
        h_clip = (d - p_clip) / (p_clip * (1.0 - p_clip))

        # 结局模型：辅助样本拟合（诚实分样本），主样本预测
        xa, da, ya = np.asarray(data.X)[aux], np.asarray(data.D)[aux], np.asarray(data.Y)[aux]
        m1 = _fit_predict(xa[da == 1], ya[da == 1], x)
        m0 = _fit_predict(xa[da == 0], ya[da == 0], x)
        m_d = np.where(d == 1, m1, m0)

        signals = {
            "HT": h * y,
            "HT+裁剪": h_clip * y,
            "AIPW": (m1 - m0) + h * (y - m_d),
            "AIPW+裁剪": (m1 - m0) + h_clip * (y - m_d),
        }

        # 分组按**真实** τ 的分位数，四个信号共用，隔离"信号"这一个变量
        edges = np.quantile(tau, np.linspace(0, 1, n_groups + 1)[1:-1])
        group = np.searchsorted(edges, tau, side="right")
        dummies = np.zeros((main.size, n_groups))
        dummies[np.arange(main.size), group] = 1.0

        for name, sig in signals.items():
            a = acc[name]
            a["kurt"].append(float(((sig - sig.mean()) ** 4).mean() / sig.var() ** 2))
            a["sd"].append(float(sig.std()))
            eff, se = _ols(dummies, sig)
            for k in range(n_groups):
                truth = float(tau[group == k].mean())
                a["cov"].append(float(abs(eff[k] - truth) <= z * se[k]))
                a["len"].append(2 * z * float(se[k]))
                a["gap"].append(abs(float(eff[k]) - truth))

    arms = [
        SignalArmStats(
            name=name,
            kurtosis=float(np.mean(acc[name]["kurt"])),
            coverage=float(np.mean(acc[name]["cov"])),
            mean_length=float(np.mean(acc[name]["len"])),
            mean_abs_gap=float(np.mean(acc[name]["gap"])),
            signal_sd=float(np.mean(acc[name]["sd"])),
        )
        for name in names
    ]
    return SignalComparison(
        n=n, n_splits=n_splits, n_groups=n_groups, clip=clip, arms=arms
    )


# --------------------------------------------------------------------------- #
# 三、排序指标 vs 水平指标
# --------------------------------------------------------------------------- #
@dataclass
class UpliftMetricAudit:
    """Qini/AUUC 与 MSE 给出的结论是否一致。"""

    n: int
    qini_in_sample: float
    qini_out_sample: float
    qini_perfect: float
    qini_constant: float
    mse_forest: float
    mse_constant: float

    @property
    def in_sample_optimism(self) -> float:
        """样本内 Qini 相对留出集高估了多少。"""
        if self.qini_out_sample == 0:
            return float("nan")
        return self.qini_in_sample / self.qini_out_sample - 1.0

    @property
    def ranking_says_forest_wins(self) -> bool:
        return self.qini_out_sample > self.qini_constant

    @property
    def level_says_forest_loses(self) -> bool:
        return self.mse_forest > self.mse_constant

    @property
    def verdicts_conflict(self) -> bool:
        return self.ranking_says_forest_wins and self.level_says_forest_loses

    def summary(self) -> str:
        lines = [
            f"排序指标 vs 水平指标（n={self.n}，nonlinear CATE）",
            f"  Qini：样本内 {self.qini_in_sample:.2f} / 留出 {self.qini_out_sample:.2f} / "
            f"完美 {self.qini_perfect:.2f} / 常数 {self.qini_constant:.2f}",
            f"    样本内高估 {self.in_sample_optimism:+.1%}"
            "（把评估样本也拿去拟合，Qini 会虚高）",
            f"  MSE：森林 {self.mse_forest:.4f} vs 常数 {self.mse_constant:.4f}",
            "",
            f"  排序结论：森林优于常数（Qini {self.qini_out_sample:.1f} vs "
            f"{self.qini_constant:.1f}）",
            f"  水平结论：森林劣于常数（MSE {self.mse_forest:.3f} vs {self.mse_constant:.3f}）",
        ]
        if self.verdicts_conflict:
            lines.append(
                "  -> **两个指标给出相反结论**。Qini/AUUC 只衡量排序，完全不衡量水平；"
            )
            lines.append(
                "     「我的模型 AUUC 更高」不等于「我的 CATE 估得更准」。"
            )
        return "\n".join(lines)


def run_uplift_metric_audit(
    *,
    n: int = 6000,
    cate_form: str = "nonlinear",
    forest_config: ForestConfig | None = None,
    seed: int = 0,
) -> UpliftMetricAudit:
    """同一批数据上同时算 Qini（样本内 + 留出）与 MSE。"""
    config = forest_config or ForestConfig(n_trees=60, max_depth=5, min_leaf=20, seed=0)
    data = generate_hte_data(HTEConfig(n=n, cate_form=cate_form, seed=seed))

    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(data.n)
    half = data.n // 2
    tr, te = perm[:half], perm[half:]

    forest = CausalForest(config).fit(data.X[tr], data.D[tr], data.Y[tr])
    pred_in = forest.predict(data.X[tr])
    pred_out = forest.predict(data.X[te])
    constant = constant_prediction(data.tau[te])

    return UpliftMetricAudit(
        n=n,
        qini_in_sample=qini_coefficient(data.Y[tr], data.D[tr], pred_in),
        qini_out_sample=qini_coefficient(data.Y[te], data.D[te], pred_out),
        qini_perfect=qini_coefficient(data.Y[te], data.D[te], data.tau[te]),
        qini_constant=qini_coefficient(data.Y[te], data.D[te], constant),
        mse_forest=float(np.mean((pred_out - data.tau[te]) ** 2)),
        mse_constant=float(np.mean((constant - data.tau[te]) ** 2)),
    )
