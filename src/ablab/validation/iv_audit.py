"""IV 的验证台：OLS 的偏、2SLS 的偏、以及**弱工具到底伤在哪里**。

先说一件被实测推翻的预期
------------------------
动手前我写下的判据是"弱工具会让 Wald 区间覆盖率崩到 0.6 上下"（教科书结论）。
**实测没有崩**：F ≈ 1.0 时 Wald 覆盖率 0.96~0.97，同方差口径与稳健口径
一模一样（见 ``reports/m3_validation.md`` 的 2.8 节）。原因是 SE 与点估计的
尾部**一起**变大 —— 区间宽到中位数 7.9，所以"盖得住"。

弱工具真正伤的是另外三件事，这一节就量它们：

* **点估计朝 OLS 靠**：中位偏差 ÷ OLS 偏差，F≈1 时是 **62%**。
  也就是说"我用了工具变量"并不能救回点估计，它只是把 OLS 的偏拉回来一部分；
* **均值与 RMSE 爆炸**：同一档位上 2SLS 的均值偏差 +87.8、RMSE 1281 ——
  报告里那个"平均效应"在这种数据上没有任何意义；
* **AR 区间宽到无用**：它覆盖率守住了（0.98），代价是 **90~94% 的区间无界** ——
  "这批数据排除不掉任何 β"。这才是诚实结论该有的样子：
  不是给一个窄而错的区间，而是说"给不出区间"。

所以这一节报的量是：``ols_bias``（为什么需要工具）、
``tsls_median_bias`` 与 ``median_bias_ratio``（朝 OLS 靠了多少）、
``tsls_rmse``（散到什么程度）、``wald_coverage`` 与 ``ar_coverage``（区间守不守）、
``ar_unbounded_share`` / ``ar_empty_share``（守住的代价）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..causal.iv import two_sls
from ..sim import IVScenarioConfig, generate_iv_scenario

__all__ = ["IVAudit", "IVStrengthRow", "run_iv_audit"]


def _ols_beta(y: np.ndarray, d: np.ndarray, x: np.ndarray) -> float:
    """把 D 当外生直接回归 —— 这就是"不做工具变量"的做法。"""
    design = np.column_stack([d, x, np.ones(d.size)])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][0])


@dataclass(frozen=True)
class IVStrengthRow:
    """一个工具强度档位下的全部读数。"""

    pi: float
    first_stage_f: float
    ols_bias: float
    ols_rmse: float
    tsls_bias: float
    tsls_median_bias: float
    tsls_rmse: float
    wald_coverage: float
    ar_coverage: float
    wald_width_median: float
    ar_width_median: float
    ar_unbounded_share: float
    ar_empty_share: float
    weak_share: float

    @property
    def median_bias_ratio(self) -> float:
        """2SLS 的中位偏差 ÷ OLS 的偏差 —— "朝 OLS 靠了多少"。

        这是弱工具最经典的读数：``0`` 表示完全没被带偏，``1`` 表示 2SLS
        给出的就是 OLS 的答案（工具白用了）。用中位数而不是均值：
        弱工具下 2SLS 的均值不存在（实测 RMSE 上千），比值会被一两个样本决定。
        """
        if abs(self.ols_bias) < 1e-12:
            return float("nan")
        return self.tsls_median_bias / self.ols_bias

    def summary(self) -> str:
        return (
            f"  pi={self.pi:<5g} F={self.first_stage_f:>7.2f} | "
            f"OLS 偏差 {self.ols_bias:+.4f} | 2SLS 中位偏差 {self.tsls_median_bias:+.4f}"
            f"（占 OLS 的 {self.median_bias_ratio:+.1%}）均值偏差 {self.tsls_bias:+.2f} "
            f"RMSE {self.tsls_rmse:.2f} | "
            f"覆盖 Wald {self.wald_coverage:.4f} / AR {self.ar_coverage:.4f} | "
            f"AR 中位宽 {self.ar_width_median:.2f} 无界 {self.ar_unbounded_share:.2f} "
            f"空集 {self.ar_empty_share:.2f}"
        )


@dataclass(frozen=True)
class IVAudit:
    """工具强度的谱系：从"几乎没用"到"很强"，看每个诊断怎么变。"""

    n: int
    n_replications: int
    tau: float
    rho: float
    rows: tuple[IVStrengthRow, ...]
    #: 正对照：rho = 0（没有内生性）时 OLS 应当无偏
    ols_bias_no_endogeneity: float
    tsls_bias_no_endogeneity: float

    @property
    def ols_is_biased(self) -> bool:
        """内生性确实把 OLS 推偏了（第一档的偏差应当明显大于 0）。"""
        return bool(self.rows and abs(self.rows[0].ols_bias) > 0.3)

    @property
    def weak_pulls_to_ols(self) -> bool:
        """最弱那一档上，2SLS 的中位偏差被拉到了 OLS 偏差的 30% 以上。"""
        return bool(self.rows and abs(self.rows[0].median_bias_ratio) > 0.3)

    @property
    def wald_does_not_break(self) -> bool:
        """**与教科书预期相反的那一条**：最弱档位的 Wald 覆盖率仍 ≥ 0.90。

        这不是"我们做对了什么"，而是"这个设计下的实际行为"：
        稳健三明治的 SE 与点估计的尾部一起变大，于是区间宽到盖得住。
        把它写成一个断言，是为了让后来人**不会**在没量过的前提下
        在 README 里写下"覆盖率会崩"。
        """
        return bool(self.rows and self.rows[0].wald_coverage >= 0.90)

    @property
    def ar_holds_throughout(self) -> bool:
        """所有档位上 AR 覆盖率都不低于 0.90 —— 这是它存在的理由。"""
        return all(r.ar_coverage >= 0.90 for r in self.rows)

    @property
    def ar_cost_is_visible(self) -> bool:
        """最弱档位上过半的 AR 区间是无界的 —— 稳健的代价必须看得见。"""
        return bool(self.rows and self.rows[0].ar_unbounded_share > 0.5)

    def summary(self) -> str:
        lines = [
            f"工具变量审计：n={self.n}，{self.n_replications} 次重抽，"
            f"真实效应 {self.tau:g}，内生性 rho={self.rho:g}",
        ]
        lines += [r.summary() for r in self.rows]
        lines += [
            f"  正对照（rho=0，没有内生性）：OLS 偏差 {self.ols_bias_no_endogeneity:+.4f}，"
            f"2SLS 偏差 {self.tsls_bias_no_endogeneity:+.4f}",
            "  读法（都是量出来的，不是照抄教科书）：",
            "    · OLS 的偏差来自**混淆**，与工具强度无关（每一档都差不多）；",
            "    · 工具越弱，2SLS 的**中位偏差**越朝 OLS 靠，而均值与 RMSE 会爆炸 ——",
            "      「我用了工具变量」救不回点估计，它只是把 OLS 的偏拉回来一部分；",
            "    · **Wald 覆盖率没有崩**（最弱档位仍 ≈0.96）：SE 与尾部一起变大，",
            "      区间宽到盖得住。教科书那句「弱工具让区间过窄」在这个设计下不成立；",
            "    · AR 区间守住了名义覆盖，代价是大部分区间**无界** ——",
            "      正确的话不是「β 在某个窄区间里」，而是「这批数据排除不掉任何 β」。",
        ]
        return "\n".join(lines)


def run_iv_audit(
    *,
    n: int = 2000,
    n_replications: int = 200,
    pis: tuple[float, ...] = (0.03, 0.05, 0.10, 0.20, 0.50),
    rho: float = 0.8,
    tau: float = 2.0,
    alpha: float = 0.05,
    seed_start: int = 1,
) -> IVAudit:
    """扫工具强度，逐档量 OLS/2SLS 的偏差与两条区间的覆盖率。"""
    rows: list[IVStrengthRow] = []

    for pi in pis:
        f_stats: list[float] = []
        ols_betas: list[float] = []
        tsls_betas: list[float] = []
        wald_cover = ar_cover = 0
        wald_widths: list[float] = []
        ar_widths: list[float] = []
        n_unbounded = n_empty = n_weak = 0

        for r in range(n_replications):
            sample = generate_iv_scenario(
                IVScenarioConfig(
                    n=n, pi=pi, rho=rho, tau=tau, seed=seed_start + r
                )
            )
            ols_betas.append(_ols_beta(sample.Y, sample.D, sample.X))
            res = two_sls(sample.Y, sample.D, sample.Z, sample.X, alpha=alpha)
            tsls_betas.append(res.beta)
            f_stats.append(res.first_stage_f)
            wald_widths.append(res.ci[1] - res.ci[0])
            wald_cover += int(res.ci[0] <= tau <= res.ci[1])
            n_weak += int(res.weak)

            if res.ar_ci is None:
                n_empty += 1
            else:
                n_unbounded += int(res.ar_unbounded)
                ar_cover += int(res.ar_ci[0] <= tau <= res.ar_ci[1])
                if np.isfinite(res.ar_width):
                    ar_widths.append(res.ar_width)

        ols_arr = np.asarray(ols_betas)
        tsls_arr = np.asarray(tsls_betas)
        rows.append(
            IVStrengthRow(
                pi=pi,
                first_stage_f=float(np.mean(f_stats)),
                ols_bias=float(ols_arr.mean() - tau),
                ols_rmse=float(np.sqrt(np.mean((ols_arr - tau) ** 2))),
                tsls_bias=float(tsls_arr.mean() - tau),
                tsls_median_bias=float(np.median(tsls_arr) - tau),
                tsls_rmse=float(np.sqrt(np.mean((tsls_arr - tau) ** 2))),
                wald_coverage=wald_cover / n_replications,
                ar_coverage=ar_cover / n_replications,
                wald_width_median=float(np.median(wald_widths)),
                ar_width_median=float(np.median(ar_widths)) if ar_widths else float("inf"),
                ar_unbounded_share=n_unbounded / n_replications,
                ar_empty_share=n_empty / n_replications,
                weak_share=n_weak / n_replications,
            )
        )

    # 正对照：把内生性关掉（rho=0），OLS 应当无偏
    ols0: list[float] = []
    tsls0: list[float] = []
    for r in range(max(20, n_replications // 4)):
        sample = generate_iv_scenario(
            IVScenarioConfig(n=n, pi=pis[-1], rho=0.0, tau=tau, seed=seed_start + r)
        )
        ols0.append(_ols_beta(sample.Y, sample.D, sample.X))
        tsls0.append(two_sls(sample.Y, sample.D, sample.Z, sample.X, alpha=alpha).beta)

    return IVAudit(
        n=n,
        n_replications=n_replications,
        tau=tau,
        rho=rho,
        rows=tuple(rows),
        ols_bias_no_endogeneity=float(np.mean(ols0) - tau),
        tsls_bias_no_endogeneity=float(np.mean(tsls0) - tau),
    )
