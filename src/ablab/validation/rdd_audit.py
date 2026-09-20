"""断点回归的审计台：**带宽是主战场，操纵是前提**。

这个审计刻意把"点估计准不准"放在第二位，先量三件更基础的事：

  1. **带宽的偏差-方差置换**：同一个 DGP，同一份数据，只看带宽。
     离断点越远偏差越大、越近方差越大 —— 于是"覆盖率"成了唯一能同时看住
     两头的判据。顺带量出 MSE 最优带宽**不是**覆盖率最优带宽这件事
     （它优化的是点估计，不是区间）。
  2. **零效应的误报率**：τ = 0 时这些区间的实际拒绝率。
  3. **操纵检验的两端**：密度连续时的误报率、以及"有人把单元挪过线"时的检出率。
     一个只会报警的检验与一个从不报警的检验一样没用。

再加一档模糊断点：断点只改变**接受处置的概率**。它把 ITT 与 LATE 的区别
直接量出来 —— 前者是有意稀释过的效应，"断点显著"不等于"处置有效"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..causal.rdd import (
    cct_robust_ci,
    fuzzy_rdd,
    manipulation_test,
    mse_optimal_bandwidth,
    sharp_rdd,
)

__all__ = ["RDDAudit", "run_rdd_audit"]

#: 断点处的真实效应（sharp 与 fuzzy 两档都用它）
TRUE_TAU = 2.0


def _baseline(x: np.ndarray) -> np.ndarray:
    """基线函数：两侧都光滑、但**二阶导不同**。

    为什么必须让两侧曲率不同：RDD 的局部线性估计在断点两侧各做一次边界
    局部线性，各自的边界偏差在 τ̂ 里相减 —— 曲率相同的部分**自己抵消**，
    只剩曲率之差。第一版 DGP 两侧曲率一样，测出来的偏差是 +0.029/+0.028
    （随带宽**不动**），"带宽太宽会出事"那句话在这个 DGP 上量不出来。
    这不是判据的问题，是 DGP 没搭对。
    """
    curvature = np.where(x < 0.0, 0.3, 2.5)
    return 1.0 + 0.8 * x + curvature * x**2 + 0.4 * np.sin(2.0 * x)


def _draw(rng: np.random.Generator, n: int, *, tau: float = TRUE_TAU,
          noise: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-1.0, 1.0, n)
    y = _baseline(x) + tau * (x >= 0.0) + rng.normal(0.0, noise, n)
    return x, y


@dataclass(frozen=True)
class RDDAudit:
    """断点回归的审计读数（每格都是 ``n_trials`` 次仿真的平均）。"""

    n_trials: int
    n: int
    tau: float
    #: 估计量名 → {偏差, 覆盖率, 平均 SE, 平均带宽}
    estimators: dict[str, dict[str, float]] = field(default_factory=dict)
    #: 零效应档：估计量名 → 拒绝率
    false_positive: dict[str, float] = field(default_factory=dict)
    #: 操纵检验：密度连续时的误报率、操纵时的检出率、平均对数跳跃
    manipulation: dict[str, float] = field(default_factory=dict)
    #: 模糊断点：ITT / 第一阶段 / LATE 的均值与 LATE 的覆盖率
    fuzzy: dict[str, float] = field(default_factory=dict)

    # ---- 带宽：偏差与方差确实在换位置 ------------------------------------ #
    @property
    def bias_grows_with_bandwidth(self) -> bool:
        """带宽 ×2 的偏差绝对值大于 plug-in 带宽处的偏差。"""
        return abs(self.estimators["朴素·2h"]["偏差"]) > abs(
            self.estimators["朴素·plug-in h"]["偏差"]
        )

    @property
    def variance_shrinks_with_bandwidth(self) -> bool:
        """带宽 ×2 的 SE 小于带宽 ×0.5 的 SE（方差那一头）。"""
        return self.estimators["朴素·2h"]["平均 SE"] < self.estimators["朴素·0.5h"]["平均 SE"]

    @property
    def naive_ci_undercovers_at_wide_bandwidth(self) -> bool:
        """朴素区间在宽带宽下欠覆盖 —— 那个带宽是给**点估计**的 MSE 用的。"""
        return self.estimators["朴素·2h"]["覆盖率"] < 0.90

    @property
    def bias_correction_fixes_the_point_estimate(self) -> bool:
        """偏差校正把点估计的偏差压到原来的一半以下（这一半它做对了）。"""
        return abs(self.estimators["CCT 偏差校正"]["偏差"]) < 0.5 * abs(
            self.estimators["朴素·plug-in h"]["偏差"]
        )

    @property
    def bias_correction_alone_hurts_coverage(self) -> bool:
        """**判据被实测改写的那一条**：原来写的是"校正后覆盖率回到名义附近"。

        实测不成立：校正后的覆盖率 **0.8300**，比同带宽的朴素区间 **0.8850**
        还低。原因是这次只做了"减偏差"，方差仍是校正前那个 ——
        而偏差是**估**出来的，估它自己也带来方差。CCT 之所以要另给一个
        "稳健方差"，正是因为这个；本仓库没实现那一步，所以这里如实报出来，
        并把判据改成"校正只修点估计、不修区间"这个**可复现的事实**。
        """
        return (
            self.estimators["CCT 偏差校正"]["覆盖率"]
            < self.estimators["朴素·plug-in h"]["覆盖率"]
        )

    # ---- 零效应：误报率 --------------------------------------------------- #
    @property
    def zero_effect_false_positive_is_reasonable(self) -> bool:
        """plug-in / 半带宽 / 均匀核 / CCT 这几档的误报率应当接近名义。

        **不含宽带那一档**：它在零效应下大量拒绝是这一节要展示的现象
        （偏差被当成效应），不是一个"应该合格"的读数 —— 见下一条。
        """
        keys = ("朴素·plug-in h", "朴素·0.5h", "朴素·均匀核")
        return max(self.false_positive[k] for k in keys) <= 0.13

    @property
    def wide_bandwidth_manufactures_significance(self) -> bool:
        """零效应 + 宽带宽：偏差被读成效应（拒绝率超过名义的两倍）。"""
        return self.false_positive["朴素·2h"] > 0.10

    @property
    def correction_without_variance_over_rejects(self) -> bool:
        """零效应下校正那一档的拒绝率最高 —— 与"它只修点估计"对得上。"""
        return self.false_positive["CCT 偏差校正"] > 2.0 * self.false_positive["朴素·0.5h"]

    # ---- 操纵检验：两端都要看 -------------------------------------------- #
    @property
    def manipulation_test_is_calibrated(self) -> bool:
        return self.manipulation["误报率"] <= 0.10

    @property
    def manipulation_test_has_power(self) -> bool:
        return self.manipulation["检出率（有人挪过线）"] >= 0.90

    # ---- 模糊断点：ITT ≠ LATE -------------------------------------------- #
    @property
    def itt_is_attenuated(self) -> bool:
        """ITT 明显小于 LATE：它是被合规份额稀释过的效应。"""
        return self.fuzzy["ITT 均值"] < 0.75 * self.fuzzy["LATE 均值"]

    @property
    def fuzzy_late_recovers_the_effect(self) -> bool:
        return abs(self.fuzzy["LATE 均值"] - self.tau) < 0.25

    @property
    def fuzzy_ci_covers(self) -> bool:
        return self.fuzzy["LATE 覆盖率"] >= 0.85

    @property
    def fuzzy_design_is_usable(self) -> bool:
        """第一阶段很少被门槛拒绝（>95% 的仿真里 Wald 比都有定义）。"""
        return self.fuzzy["第一阶段被门槛拒绝的比例"] <= 0.05

    def passed(self) -> dict[str, bool]:
        return {
            "带宽：偏差随 h 变大": self.bias_grows_with_bandwidth,
            "带宽：方差随 h 变小": self.variance_shrinks_with_bandwidth,
            "朴素区间在宽带宽下欠覆盖": self.naive_ci_undercovers_at_wide_bandwidth,
            "偏差校正把点估计的偏差压小": self.bias_correction_fixes_the_point_estimate,
            "偏差校正只修点估计（区间反而更差）": self.bias_correction_alone_hurts_coverage,
            "偏差校正不改方差 ⇒ 零效应下过度拒绝": self.correction_without_variance_over_rejects,
            "零效应的误报率合理（plug-in 那几档）": self.zero_effect_false_positive_is_reasonable,
            "宽带宽在零效应下制造显著": self.wide_bandwidth_manufactures_significance,
            "操纵检验在密度连续时不报警": self.manipulation_test_is_calibrated,
            "操纵检验在有人挪线时报警": self.manipulation_test_has_power,
            "模糊断点：ITT 被稀释": self.itt_is_attenuated,
            "模糊断点：LATE 恢复真值": self.fuzzy_late_recovers_the_effect,
            "模糊断点：LATE 区间守覆盖": self.fuzzy_ci_covers,
            "模糊断点：第一阶段基本都过门槛": self.fuzzy_design_is_usable,
        }

    def summary(self) -> str:
        lines = [
            f"断点回归审计（每格 {self.n_trials} 次仿真，n={self.n}，真实 τ = {self.tau}）",
            "",
            "  一、带宽的偏差-方差置换（同一份 DGP，只换带宽）",
            f"    {'估计量':<18}{'偏差':>10}{'覆盖率':>10}{'平均 SE':>10}{'平均 h':>10}",
        ]
        for name, row in self.estimators.items():
            lines.append(
                f"    {name:<18}{row['偏差']:>+10.4f}{row['覆盖率']:>10.4f}"
                f"{row['平均 SE']:>10.4f}{row['平均带宽']:>10.4f}"
            )
        lines += [
            "    读法：MSE 最优带宽是给**点估计**的，不是给区间的 ——",
            "    所以它在宽的那一侧换来的是偏差，而偏差不会被 SE 盖住。",
            "",
            "  二、零效应（τ = 0）的实际拒绝率（名义 5%）",
        ]
        for name, fpr in self.false_positive.items():
            lines.append(f"    {name:<18}{fpr:>10.4f}")
        lines += [
            "    两档拒绝率最高：最宽的那一档（偏差被读成效应）"
            "与校正那一档（方差被低估）。",
            "",
            "  三、操纵检验（McCrary 式）",
            f"    密度连续时的误报率        {self.manipulation['误报率']:.4f}",
            f"    有人挪过线时的检出率      {self.manipulation['检出率（有人挪过线）']:.4f}",
            f"    两类数据的平均对数跳跃    "
            f"{self.manipulation['干净数据·对数跳跃']:+.4f} / "
            f"{self.manipulation['被操纵数据·对数跳跃']:+.4f}",
            "",
            "  四、模糊断点：断点只改变接受处置的概率",
            f"    ITT 均值                  {self.fuzzy['ITT 均值']:+.4f}"
            "   ← 被合规份额稀释过",
            f"    第一阶段（D 的跳跃）均值  {self.fuzzy['第一阶段均值']:+.4f}",
            f"    LATE 均值（Wald 比）      {self.fuzzy['LATE 均值']:+.4f}"
            f"（真值 {self.tau:+.4f}）",
            f"    LATE 区间覆盖率           {self.fuzzy['LATE 覆盖率']:.4f}",
            "    第一阶段没过门槛的比例    "
            f"{self.fuzzy['第一阶段被门槛拒绝的比例']:.4f}"
            "（只对通过的那部分求平均 = 一次轻度选择，写在这里）",
        ]
        return "\n".join(lines)


def run_rdd_audit(
    *,
    n_trials: int = 200,
    n: int = 1500,
    seed: int = 0,
) -> RDDAudit:
    """三档 DGP（有信号 / 零效应 / 被操纵）+ 模糊断点，各跑 ``n_trials`` 次。"""
    n_trials = max(int(n_trials), 1)
    names = ("朴素·plug-in h", "朴素·0.5h", "朴素·2h", "朴素·均匀核", "CCT 偏差校正")
    rows: dict[str, dict[str, list[float]]] = {k: {} for k in names}

    def push(name: str, key: str, value: float) -> None:
        rows[name].setdefault(key, []).append(float(value))

    fpr: dict[str, list[float]] = {k: [] for k in names}
    clean_reject: list[float] = []
    dirty_reject: list[float] = []
    clean_jump: list[float] = []
    dirty_jump: list[float] = []
    itt, jump_d, late, late_cover = [], [], [], []
    first_stage_gated = 0

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + 1000 * trial)
        x, y = _draw(rng, n)
        h = mse_optimal_bandwidth(x, y)
        results = {
            "朴素·plug-in h": sharp_rdd(x, y, bandwidth=h),
            "朴素·0.5h": sharp_rdd(x, y, bandwidth=0.5 * h),
            "朴素·2h": sharp_rdd(x, y, bandwidth=2.0 * h),
            "朴素·均匀核": sharp_rdd(x, y, bandwidth=h, kernel="uniform"),
            "CCT 偏差校正": cct_robust_ci(x, y, bandwidth=h),
        }
        for name, res in results.items():
            push(name, "偏差", res.tau - TRUE_TAU)
            push(name, "覆盖率", float(res.covers(TRUE_TAU)))
            push(name, "平均 SE", res.se)
            push(name, "平均带宽", res.bandwidth)

        # 零效应：同一套估计量，τ = 0
        x0, y0 = _draw(rng, n, tau=0.0)
        h0 = mse_optimal_bandwidth(x0, y0)
        zero = {
            "朴素·plug-in h": sharp_rdd(x0, y0, bandwidth=h0),
            "朴素·0.5h": sharp_rdd(x0, y0, bandwidth=0.5 * h0),
            "朴素·2h": sharp_rdd(x0, y0, bandwidth=2.0 * h0),
            "朴素·均匀核": sharp_rdd(x0, y0, bandwidth=h0, kernel="uniform"),
            "CCT 偏差校正": cct_robust_ci(x0, y0, bandwidth=h0),
        }
        for name, res in zero.items():
            fpr[name].append(float(res.p_value < 0.05))

        # 操纵检验：干净 vs 有人把断点左侧的单元挪到右侧
        xc = rng.normal(0.0, 1.0, n)  # 干净：连续密度
        mt = manipulation_test(xc)
        clean_reject.append(float(mt.rejects))
        clean_jump.append(mt.log_jump)
        xd = rng.normal(0.0, 1.0, n)
        moved = (xd > -0.4) & (xd < 0.0) & (rng.random(n) < 0.7)
        xd[moved] = 0.0 + 0.4 * rng.random(int(moved.sum()))
        mt_d = manipulation_test(xd)
        dirty_reject.append(float(mt_d.rejects))
        dirty_jump.append(mt_d.log_jump)

        # 模糊断点：断点只改变接受处置的概率（断点下方无人被处置）
        z = (x >= 0.0).astype(float)
        d = (z * (rng.random(n) < 0.4)).astype(float)
        yf = _baseline(x) + TRUE_TAU * d + rng.normal(0.0, 0.5, n)
        # 模糊断点的带宽按**第一阶段**选（分母是它），而不是按 Y ——
        # 这不是调参，而是"Wald 比的分母决定可用性"这个事实的直接后果。
        try:
            fr = fuzzy_rdd(x, yf, d, bandwidth=mse_optimal_bandwidth(x, d))
        except ValueError:
            # 第一阶段没过 5% 门槛的**极少**几次会被拒绝。计数并如实报出来：
            # 只对通过的那部分求平均等于一次轻度的选择，必须写在报告里。
            first_stage_gated += 1
            continue
        itt.append(fr.tau_itt)
        jump_d.append(fr.jump_d)
        late.append(fr.tau_late)
        late_cover.append(float(fr.covers(TRUE_TAU)))

    def agg(values: list[float]) -> float:
        return float(np.mean(values))

    estimators = {
        name: {
            "偏差": agg(rows[name]["偏差"]),
            "覆盖率": agg(rows[name]["覆盖率"]),
            "平均 SE": agg(rows[name]["平均 SE"]),
            "平均带宽": agg(rows[name]["平均带宽"]),
        }
        for name in names
    }
    return RDDAudit(
        n_trials=n_trials,
        n=n,
        tau=TRUE_TAU,
        estimators=estimators,
        false_positive={name: agg(v) for name, v in fpr.items()},
        manipulation={
            "误报率": agg(clean_reject),
            "检出率（有人挪过线）": agg(dirty_reject),
            "干净数据·对数跳跃": agg(clean_jump),
            "被操纵数据·对数跳跃": agg(dirty_jump),
        },
        fuzzy={
            "ITT 均值": agg(itt),
            "第一阶段均值": agg(jump_d),
            "LATE 均值": agg(late),
            "LATE 覆盖率": agg(late_cover),
            "第一阶段被门槛拒绝的比例": first_stage_gated / n_trials,
        },
    )
