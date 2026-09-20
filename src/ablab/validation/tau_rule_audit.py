"""``tau`` 怎么选：一条能算出来的规则，以及一个**必须避开的陷阱**。

为什么单独一节
--------------
mSPRT 的**有效性不依赖 tau** —— 先验给多宽，I 类错误都 ≤ alpha。但**功效极度依赖**它。
线上原来取的是 ``tau = 2 * 末次标准误``：那是个经验值，量过才知道它离最优有多远。
这一节把三件事分开量：

1. **固定 tau 的扫描**：每个 tau 的 FWER 与功效。FWER 应当处处 ≤ alpha（这就是
   "有效性不依赖 tau"），而功效有一个内部极大点 —— 实测在 ``2.87 * SE``（alpha=0.05）；
2. **规则点**：``optimal_tau``（阈值最小化，解析可算）、线上的 ``2*SE``、
   以及"与目标效应匹配"三种取法各自的实测功效，并给出它们离网格最优有多远；
3. **陷阱：让数据选先验**。always-valid 的保证要求 tau 是**可预测量**
   （事前固定，或只依赖过去的查看）。若每次查看都用**当前**观测到的效应当先验尺度
   （``tau_k = |theta_hat_k|``，或更"聪明"的 ``sqrt(theta_hat_k^2 - V_k)``
   —— 后者恰好是**当次似然比最大的那个 tau**），那个承诺就作废了。
   实测：它并不会让 I 类错误"小一点"或"大一点"，而是**系统性变大**
   （每个监测密度下都更大，密集监测时翻倍以上）。

   注意这里量的是**相对**变化，不是"越过 alpha"：5 次查看时连固定 tau 都只有
   0.0066（远低于名义值，mSPRT 在小查看次数下很保守），
   所以"越过 alpha"要等监测足够密 —— 实测 400 次查看时才到 0.0437。
   **结论是"保证没了"，而不是"一定超 alpha"** —— 这两句话的强度不一样。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sequential import choose_tau, msprt_p_value
from ..sim.sequential import default_information_fractions, simulate_canonical_sequences

__all__ = ["TauRuleAudit", "TauRuleRow", "run_tau_rule_audit"]


@dataclass(frozen=True)
class TauRuleRow:
    """一个固定 ``tau`` 的读数。"""

    label: str
    tau: float
    tau_over_se: float
    fwer: float
    power: float

    def summary(self) -> str:
        return (
            f"  {self.label:<24} tau = {self.tau:>7.4f}（{self.tau_over_se:>5.2f} SE）"
            f"  FWER {self.fwer:.4f}  功效 {self.power:.4f}"
        )


@dataclass(frozen=True)
class TauRuleAudit:
    """固定 tau 的扫描 + 三个规则点 + 数据依赖的陷阱。"""

    se_final: float
    alpha: float
    effect_scale: float
    n_trials: int
    rows: tuple[TauRuleRow, ...]
    #: 网格上的经验最优（功效最高那一档）
    empirical_best_tau_over_se: float
    empirical_best_power: float
    rule_tau_over_se: float
    rule_power: float
    online_tau_over_se: float
    online_power: float
    matched_tau_over_se: float
    matched_power: float
    #: 每次查看重算 tau（让数据选先验）时的 FWER —— 与 fixed_fwer 同密度
    datadependent_fwer: float
    #: 同一密度下固定 tau 的 FWER
    fixed_fwer: float
    #: 固定 tau 时最差的 FWER（应当 ≤ alpha）
    worst_fixed_fwer: float
    #: 监测密度扫描：``(n_looks, 固定 tau 的 FWER, 数据依赖的 FWER)``
    density_rows: tuple[tuple[int, float, float], ...]

    @property
    def validity_holds_for_any_fixed_tau(self) -> bool:
        """固定 tau 时 FWER 处处不超过 alpha（含蒙特卡洛误差的余量）。"""
        return self.worst_fixed_fwer <= self.alpha + 0.01

    @property
    def rule_is_near_the_empirical_optimum(self) -> bool:
        """规则点（阈值最小化）的功效在经验最优的 1 个百分点之内。"""
        return self.empirical_best_power - self.rule_power <= 0.01

    @property
    def online_heuristic_is_not_far_off(self) -> bool:
        """线上那个 ``2*SE`` 也不算错 —— 但它确实比规则低一截（实测差距）。"""
        return self.online_power >= self.empirical_best_power - 0.05

    @property
    def data_dependent_tau_voids_the_guarantee(self) -> bool:
        """让数据选先验：**每个监测密度下 FWER 都严格更大**，密集时至少翻倍。

        注意主张的强度：不是"一定超过 alpha"（小查看次数下 mSPRT 本身很保守，
        连固定 tau 都远低于 alpha），而是"那个 always-valid 的承诺不再成立"。
        """
        if not self.density_rows:
            return False
        always_worse = all(fixed < dd for _, fixed, dd in self.density_rows)
        dense_doubles = all(
            dd >= 2.0 * fixed for looks, fixed, dd in self.density_rows if looks >= 100
        )
        return always_worse and dense_doubles

    def summary(self) -> str:
        lines = [
            f"mSPRT 的 tau 选择（se_final={self.se_final:g}，alpha={self.alpha}，"
            f"备择效应 = {self.effect_scale:g}×SE，{self.n_trials:,} 次）",
        ]
        lines += [r.summary() for r in self.rows]
        lines += [
            f"  规则（阈值最小化）tau = {self.rule_tau_over_se:.2f} SE"
            f"（{self.rule_power:.4f}） vs 经验最优 {self.empirical_best_tau_over_se:.2f} SE"
            f"（{self.empirical_best_power:.4f}）",
            f"  线上旧口径 2*SE（{self.online_power:.4f}）差 "
            f"{self.empirical_best_power - self.online_power:+.4f}；"
            f"匹配目标效应（{self.matched_power:.4f}）差 "
            f"{self.empirical_best_power - self.matched_power:+.4f}",
            f"  固定 tau 时的最差 FWER {self.worst_fixed_fwer:.4f}"
            f"（名义 {self.alpha}）—— **有效性不依赖 tau**；",
            f"  陷阱：让数据选先验 → 同一密度下 FWER 从 {self.fixed_fwer:.4f} 抬到 "
            f"{self.datadependent_fwer:.4f}（x{self.datadependent_fwer / max(self.fixed_fwer, 1e-12):.2f}）",
            "  监测密度扫描（固定 tau vs 让数据选先验）：",
        ]
        lines += [
            f"    {looks:>4} 次查看：固定 {fixed:.4f} → 数据依赖 {dd:.4f}"
            f"（x{dd / max(fixed, 1e-12):.2f}）"
            for looks, fixed, dd in self.density_rows
        ]
        lines += [
            "  读法：tau 是**设计期的常数**（或只依赖过去的查看）；",
            "        从**当前**数据里挑它，等于事后挑备择假设 —— 保证没了，",
            "        而且实测方向是**变大**，不是「小一点」。",
        ]
        return "\n".join(lines)


def run_tau_rule_audit(
    *,
    n_trials: int = 20_000,
    n_looks: int = 5,
    alpha: float = 0.05,
    se_final: float = 0.42,
    effect_scale: float = 2.0,
    grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.87, 4.0, 6.0, 10.0, 20.0),
    seed: int = 0,
) -> TauRuleAudit:
    """跑固定 tau 的网格、三个规则点，以及数据依赖的陷阱。"""
    t = default_information_fractions(n_looks)
    null = simulate_canonical_sequences(
        n_trials=n_trials, information_fractions=t, se_final=se_final,
        effect=0.0, seed=seed,
    )
    alt = simulate_canonical_sequences(
        n_trials=n_trials, information_fractions=t, se_final=se_final,
        effect=effect_scale * se_final, seed=seed + 1,
    )

    rows: list[TauRuleRow] = []
    for mult in grid:
        tau = float(mult) * se_final
        fwer = float((msprt_p_value(null.estimates, null.standard_errors, tau) <= alpha).any(axis=1).mean())
        power = float((msprt_p_value(alt.estimates, alt.standard_errors, tau) <= alpha).any(axis=1).mean())
        rows.append(
            TauRuleRow(
                label="网格", tau=tau, tau_over_se=float(mult), fwer=fwer, power=power
            )
        )

    best = max(rows, key=lambda r: r.power)
    rule_tau = choose_tau(std_error=se_final, alpha=alpha)
    online_tau = 2.0 * se_final
    matched_tau = effect_scale * se_final

    def _fixed(tau: float) -> tuple[float, float]:
        fwer = float(
            (msprt_p_value(null.estimates, null.standard_errors, tau) <= alpha).any(axis=1).mean()
        )
        power = float(
            (msprt_p_value(alt.estimates, alt.standard_errors, tau) <= alpha).any(axis=1).mean()
        )
        return fwer, power

    rule_fwer, rule_power = _fixed(rule_tau)
    online_fwer, online_power = _fixed(online_tau)
    matched_fwer, matched_power = _fixed(matched_tau)

    # 陷阱：每次查看都用**当前**观测到的效应当先验尺度
    # （sqrt(θ̂² − V) 恰好是当次似然比最大的 tau —— "最聪明"的那种选法）
    tau_path = np.sqrt(
        np.maximum(null.estimates**2 - null.standard_errors**2, 1e-8)
    )
    dd_fwer = float(
        (msprt_p_value(null.estimates, null.standard_errors, tau_path) <= alpha).any(axis=1).mean()
    )
    rule_fwer_fixed = float(
        (msprt_p_value(null.estimates, null.standard_errors, rule_tau) <= alpha).any(axis=1).mean()
    )

    # 监测密度扫描：mSPRT 在小查看次数下很保守，"越过 alpha"要等监测足够密
    density_rows: list[tuple[int, float, float]] = []
    for looks in (5, 25, 100):
        fractions = np.arange(1, looks + 1) / looks
        path = simulate_canonical_sequences(
            n_trials=n_trials, information_fractions=fractions, se_final=se_final,
            effect=0.0, seed=seed,
        )
        f_fixed = float(
            (msprt_p_value(path.estimates, path.standard_errors, rule_tau) <= alpha)
            .any(axis=1).mean()
        )
        f_dd = float(
            (
                msprt_p_value(
                    path.estimates,
                    path.standard_errors,
                    np.sqrt(np.maximum(path.estimates**2 - path.standard_errors**2, 1e-8)),
                )
                <= alpha
            )
            .any(axis=1)
            .mean()
        )
        density_rows.append((looks, f_fixed, f_dd))

    rows = [
        *rows,
        TauRuleRow("规则：阈值最小化", rule_tau, rule_tau / se_final, rule_fwer, rule_power),
        TauRuleRow("线上旧口径 2×SE", online_tau, 2.0, online_fwer, online_power),
        TauRuleRow("匹配目标效应", matched_tau, effect_scale, matched_fwer, matched_power),
    ]

    return TauRuleAudit(
        se_final=se_final,
        alpha=alpha,
        effect_scale=effect_scale,
        n_trials=n_trials,
        rows=tuple(rows),
        empirical_best_tau_over_se=best.tau_over_se,
        empirical_best_power=best.power,
        rule_tau_over_se=rule_tau / se_final,
        rule_power=rule_power,
        online_tau_over_se=2.0,
        online_power=online_power,
        matched_tau_over_se=effect_scale,
        matched_power=matched_power,
        datadependent_fwer=dd_fwer,
        fixed_fwer=rule_fwer_fixed,
        worst_fixed_fwer=max(r.fwer for r in rows if r.label == "网格"),
        density_rows=tuple(density_rows),
    )
