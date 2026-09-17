"""频率派检验（M0 最小集）。

只实现到"把 A/B 对比这件事做对"为止：
  * Welch t 检验（不假设方差齐性，线上指标方差几乎从不齐）
  * 差值的置信区间
  * 自动挂上 SRM 诊断

两个入口，共用**同一份**统计实现（有单元测试守着二者数值一致）：

``welch_ttest(treatment, control)``
    拿到逐单元明细时用。
``welch_ttest_from_stats(n, mean, var, ...)``
    只拿到汇总统计量时用。数仓链路天然是这个形态：
    ADS 层输出 (n, mean, var)，不该也不需要把明细拉进 Python。

M1 会在这里加 CUPED 方差缩减与 ratio metric（delta method），
M2 加序贯检验与贝叶斯决策规则。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from .result import Diagnostic, Estimate
from .srm import srm_check
from .welch import welch_inference

__all__ = ["welch_ttest", "welch_ttest_from_stats", "two_proportion_ztest"]


def welch_ttest_from_stats(
    *,
    n_treatment: int,
    mean_treatment: float,
    var_treatment: float,
    n_control: int,
    mean_control: float,
    var_control: float,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
    expected_weights: dict[str, float] | None = None,
    srm_alpha: float = 1e-3,
) -> Estimate:
    """从汇总统计量直接做 Welch t 检验。

    Parameters
    ----------
    var_treatment, var_control:
        **样本方差**（ddof=1），不是标准误。
    expected_weights:
        设计权重 ``{分支名: 权重}``，传入则自动附加 SRM 体检。
    """
    effect = mean_treatment - mean_control
    inference = welch_inference(
        effect,
        n_treatment=n_treatment,
        var_treatment=var_treatment,
        n_control=n_control,
        var_control=var_control,
        alpha=alpha,
    )
    ci_low, ci_high = inference.interval(effect)

    diagnostics: list[Diagnostic] = []
    if inference.degenerate:
        diagnostics.append(
            Diagnostic(
                name="方差退化",
                status="warn",
                message="两组样本方差均为 0，标准误无法估计；置信区间已退化为点。",
            )
        )

    relative = effect / mean_control if mean_control != 0 else float("nan")

    if expected_weights is not None:
        counts = {control_name: n_control, variant: n_treatment}
        weights = {k: expected_weights[k] for k in (control_name, variant)}
        diagnostics.append(srm_check(counts, weights, alpha=srm_alpha))

    return Estimate(
        metric=metric,
        variant=variant,
        control=control_name,
        method="Welch t-test",
        absolute_effect=effect,
        relative_effect=float(relative),
        std_error=inference.se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=inference.p_value,
        n_treatment=int(n_treatment),
        n_control=int(n_control),
        mean_treatment=float(mean_treatment),
        mean_control=float(mean_control),
        alpha=alpha,
        diagnostics=tuple(diagnostics),
    )


def welch_ttest(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    metric: str = "metric",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
    expected_weights: dict[str, float] | None = None,
    srm_alpha: float = 1e-3,
) -> Estimate:
    """Welch t 检验 + 置信区间 + 自动 SRM 体检（逐单元明细入口）。

    用 Welch 而不是 Student 的合并方差版本：线上指标在实验组与对照组
    的方差经常不同（尤其当处理会改变用户行为的波动性时），
    合并方差会低估标准误、抬高假阳性。
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    t = t[np.isfinite(t)]
    c = c[np.isfinite(c)]

    if t.size < 2 or c.size < 2:
        raise ValueError(f"每组至少需要 2 个有效样本，收到 {t.size} / {c.size}")

    return welch_ttest_from_stats(
        n_treatment=int(t.size),
        mean_treatment=float(t.mean()),
        var_treatment=float(t.var(ddof=1)),
        n_control=int(c.size),
        mean_control=float(c.mean()),
        var_control=float(c.var(ddof=1)),
        metric=metric,
        variant=variant,
        control_name=control_name,
        alpha=alpha,
        expected_weights=expected_weights,
        srm_alpha=srm_alpha,
    )


def two_proportion_ztest(
    successes_t: int,
    n_t: int,
    successes_c: int,
    n_c: int,
    *,
    metric: str = "conversion",
    variant: str = "treatment",
    control_name: str = "control",
    alpha: float = 0.05,
    expected_weights: dict[str, float] | None = None,
) -> Estimate:
    """两比例 z 检验。用于转化率这类 0/1 指标，比 t 检验更贴合其分布假设。

    注意：这只适用于**用户级**的 0/1 指标。如果是"点击/曝光"这种
    以请求为分母的比值指标，方差不能这样算，M1 会用 delta method 处理。
    """
    if n_t <= 0 or n_c <= 0:
        raise ValueError("每组样本量必须为正")

    p_t = successes_t / n_t
    p_c = successes_c / n_c
    effect = p_t - p_c
    p_pool = (successes_t + successes_c) / (n_t + n_c)
    se_pooled = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c)))

    diagnostics: list[Diagnostic] = []

    if se_pooled == 0.0:
        z_stat, p_value = 0.0, 1.0
        ci_low = ci_high = effect
    else:
        z_stat = effect / se_pooled
        p_value = float(2 * stats.norm.sf(abs(z_stat)))
        # 区间用非合并标准误，与点估计的对偶性更一致
        se_unpooled = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
        crit = float(stats.norm.ppf(1 - alpha / 2))
        ci_low, ci_high = effect - crit * se_unpooled, effect + crit * se_unpooled

    if expected_weights is not None:
        diagnostics.append(
            srm_check({control_name: n_c, variant: n_t}, expected_weights)
        )

    return Estimate(
        metric=metric,
        variant=variant,
        control=control_name,
        method="two-proportion z-test",
        absolute_effect=effect,
        relative_effect=float(effect / p_c) if p_c else float("nan"),
        std_error=se_pooled,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        n_treatment=int(n_t),
        n_control=int(n_c),
        mean_treatment=p_t,
        mean_control=p_c,
        alpha=alpha,
        diagnostics=tuple(diagnostics),
    )
