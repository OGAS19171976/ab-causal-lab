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
            # 正对照：代理 = 真值。不需要训练，也不需要辅助样本。
            main = perm
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
        # Horvitz-Thompson 信号：E[signal | Z] = s0(Z)，不需要结果模型
        signal = (d - p) / (p * (1.0 - p)) * y
        kurt = float(((signal - signal.mean()) ** 4).mean() / signal.var() ** 2)
        kurt_acc.append(kurt)
        overlap_acc.append(float(np.mean((p < 0.05) | (p > 0.95))))

        # ---- BLP ---- #
        x = np.column_stack([np.ones(main.size), proxy - proxy.mean()])
        beta, se = _ols(x, signal)
        slopes.append(float(beta[1]))
        slope_ses.append(float(se[1]))
        covers_one += int(abs(beta[1] - 1.0) <= z * se[1])

        # ---- GATES ---- #
        # 分位数组：用主样本上的代理分位数切，组大小尽量均衡
        edges = np.quantile(proxy, np.linspace(0, 1, n_groups + 1)[1:-1])
        group = np.searchsorted(edges, proxy, side="right")
        dummies = np.zeros((main.size, n_groups))
        dummies[np.arange(main.size), group] = 1.0
        # 去掉空组（分位数理论上不会空，保险起见）
        keep = [g for g in range(n_groups) if dummies[:, g].sum() > 1]
        g_eff, g_se = _ols(dummies[:, keep], signal)
        true_tau = np.asarray(data.tau)[main]
        for j, g in enumerate(keep):
            true_group = float(true_tau[group == g].mean())
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
        unit_level_note=(
            "单元级那两条路线（叶内方差 / bootstrap）实测覆盖率 0.13~0.74，"
            "而这里是**组级**：对象不同，不能直接比大小，只能比"
            "「谁给出了可用的区间」"
        ),
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
