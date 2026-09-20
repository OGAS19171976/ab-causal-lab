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
from scipy import stats

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
    # 两侧曲率差要够大，偏差才不会被抽样噪声盖住：
    # (0.3, 2.5) 时 plug-in 带宽处的偏差只有 -0.006 ~ -0.013（SE 0.11），
    # 那时"带宽太宽会出事""偏差校正有没有用"都量不出来 —— DGP 没搭对，
    # 不是判据太严。(0.3, 8.0) 下偏差 -0.0836，才是这一节要讨论的量级。
    curvature = np.where(x < 0.0, 0.3, 8.0)
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
    #: 偏差带宽 b 的敏感性表：键是 "b=h" / "b=1.5h" / "b=2h"
    bias_bandwidth_rows: dict[str, dict[str, float]] = field(default_factory=dict)

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
        return abs(self.estimators["CCT 校正·稳健方差"]["偏差"]) < 0.5 * abs(
            self.estimators["朴素·plug-in h"]["偏差"]
        )

    @property
    def conventional_variance_after_correction_undercovers(self) -> bool:
        """**上一轮记下的欠账**：同一个校正后的点估计，只换方差就看得见差别。

        上一轮那句判据是"校正后的覆盖率低于朴素区间"（0.8300 < 0.8850）。
        换成两侧曲率更大的 DGP 之后，那句判据**不再成立**（校正把偏差压掉七成，
        常规方差的覆盖率 0.8667 反而高于朴素 0.7333）—— 这不矛盾：
        前一个 DGP 的朴素区间本来就准，校正只带来了方差；
        后一个 DGP 里朴素区间被偏差毁掉，减偏差立刻回本。
        所以判据改成**与同一估计量的稳健方差比**，这条在任何 DGP 上都成立：
        少算的那一块方差就是覆盖率少掉的那一块。
        """
        return (
            self.estimators["CCT 校正·常规方差"]["覆盖率"]
            < self.estimators["CCT 校正·稳健方差"]["覆盖率"] - 0.03
        )

    @property
    def robust_variance_holds_coverage(self) -> bool:
        """稳健方差（把"估偏差"带来的方差也算进去）应当把覆盖率拉回名义附近。"""
        return self.estimators["CCT 校正·稳健方差"]["覆盖率"] >= 0.90

    @property
    def bias_bandwidth_tradeoff_is_measured(self) -> bool:
        """b 的敏感性表要真的呈现出置换：b 越大区间越短、覆盖率越低。"""
        rows = self.bias_bandwidth_rows
        if len(rows) < 3:
            return False
        ses = [rows[k]["平均 SE"] for k in rows]
        covs = [rows[k]["覆盖率"] for k in rows]
        # SE 那一头是结构性的（b 越大 → 偏差估得越稳 → 区间越短），必须严格；
        # 覆盖率那一头在几十次仿真里会打平，只要求不反着走。
        return ses[0] > ses[-1] and covs[0] >= covs[-1]

    # ---- 零效应：误报率 --------------------------------------------------- #
    @property
    def naive_ci_over_rejects_because_of_bias(self) -> bool:
        """**判据被实测改写的那一条**：原来写的是"plug-in 那几档误报率 ≤0.13"。

        换成两侧曲率差更大的 DGP（偏差 -0.0784，SE 0.1136）之后不成立了：
        朴素·plug-in 的误报率 **0.1650**，均匀核 **0.2250** —— 它们的区间
        没有把偏差算进去，于是偏差被读成了效应。这不是实现坏了，
        而是"带宽选得对"与"区间算得对"是两件事：MSE 最优带宽修的是前者。
        所以判据改成"朴素区间在这个 DGP 上会过度拒绝"这个**可复现的事实**，
        守名义的那两条挪到稳健方差上（见下）。
        """
        return (
            self.false_positive["朴素·plug-in h"] > 0.10
            and self.false_positive["朴素·均匀核"] > 0.10
        )

    @property
    def wide_bandwidth_manufactures_significance(self) -> bool:
        """零效应 + 宽带宽：偏差被读成效应（拒绝率超过名义的两倍）。"""
        return self.false_positive["朴素·2h"] > 0.10

    @property
    def correction_without_variance_over_rejects(self) -> bool:
        """零效应下"只减偏差"那一档的拒绝率远高于稳健方差那一档。"""
        return (
            self.false_positive["CCT 校正·常规方差"]
            > self.false_positive["CCT 校正·稳健方差"] + 0.15
        )

    @property
    def robust_variance_controls_false_positives(self) -> bool:
        """补上方差之后，零效应下的拒绝率回到名义附近。"""
        return self.false_positive["CCT 校正·稳健方差"] <= 0.10

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
            "对照行：只减偏差时覆盖率反而更差": (
                self.conventional_variance_after_correction_undercovers
            ),
            "稳健方差把覆盖率拉回名义附近": self.robust_variance_holds_coverage,
            "对照行：只减偏差时零效应下过度拒绝": (
                self.correction_without_variance_over_rejects
            ),
            "稳健方差把零效应误报率压回名义": self.robust_variance_controls_false_positives,
            "偏差带宽 b 的置换被量出来（b↑ ⇒ 区间短、覆盖低）": (
                self.bias_bandwidth_tradeoff_is_measured
            ),
            "对照行：朴素区间在零效应下过度拒绝（偏差没进区间）": (
                self.naive_ci_over_rejects_because_of_bias
            ),
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
            "    偏差带宽 b 的置换（同一个点估计与稳健方差公式，只换 b）：",
            f"      {'b':<8}{'偏差':>10}{'覆盖率':>10}{'平均 SE':>10}",
        ]
        for tag, row in self.bias_bandwidth_rows.items():
            lines.append(
                f"      {tag:<8}{row['偏差']:>+10.4f}{row['覆盖率']:>10.4f}"
                f"{row['平均 SE']:>10.4f}"
            )
        lines += [
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
    names = (
        "朴素·plug-in h",
        "朴素·0.5h",
        "朴素·2h",
        "朴素·均匀核",
        # 偏差校正那一支拆成两行：**同一个点估计**，只换方差。
        # 这样"校正只修了一半"与"稳健方差把它补完"才是可比的。
        "CCT 校正·常规方差",
        "CCT 校正·稳健方差",
    )
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
    b_rows: dict[str, dict[str, list[float]]] = {
        "b=h": {}, "b=1.5h": {}, "b=2h": {}
    }

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + 1000 * trial)
        x, y = _draw(rng, n)
        h = mse_optimal_bandwidth(x, y)
        results = {
            "朴素·plug-in h": sharp_rdd(x, y, bandwidth=h),
            "朴素·0.5h": sharp_rdd(x, y, bandwidth=0.5 * h),
            "朴素·2h": sharp_rdd(x, y, bandwidth=2.0 * h),
            "朴素·均匀核": sharp_rdd(x, y, bandwidth=h, kernel="uniform"),
            "CCT 校正·常规方差": cct_robust_ci(x, y, bandwidth=h),
            "CCT 校正·稳健方差": cct_robust_ci(x, y, bandwidth=h),
        }
        for name, res in results.items():
            # 校正那一支：点估计是同一个，喂给两行的 SE 不同
            se = res.se_conventional if name.endswith("常规方差") else res.se
            covered = abs(res.tau - TRUE_TAU) <= 1.959964 * se
            push(name, "偏差", res.tau - TRUE_TAU)
            push(name, "覆盖率", float(covered))
            push(name, "平均 SE", se)
            push(name, "平均带宽", res.bandwidth)

        # 偏差带宽 b 的敏感性：同一个点估计、同一个稳健方差公式，只换 b
        for tag, factor in (("b=h", 1.0), ("b=1.5h", 1.5), ("b=2h", 2.0)):
            rb = cct_robust_ci(x, y, bandwidth=h, bias_bandwidth=factor * h)
            store_b = b_rows[tag]
            store_b.setdefault("偏差", []).append(rb.tau - TRUE_TAU)
            store_b.setdefault("覆盖率", []).append(float(rb.covers(TRUE_TAU)))
            store_b.setdefault("平均 SE", []).append(rb.se)

        # 零效应：同一套估计量，τ = 0
        x0, y0 = _draw(rng, n, tau=0.0)
        h0 = mse_optimal_bandwidth(x0, y0)
        zero = {
            "朴素·plug-in h": sharp_rdd(x0, y0, bandwidth=h0),
            "朴素·0.5h": sharp_rdd(x0, y0, bandwidth=0.5 * h0),
            "朴素·2h": sharp_rdd(x0, y0, bandwidth=2.0 * h0),
            "朴素·均匀核": sharp_rdd(x0, y0, bandwidth=h0, kernel="uniform"),
            "CCT 校正·常规方差": cct_robust_ci(x0, y0, bandwidth=h0),
            "CCT 校正·稳健方差": cct_robust_ci(x0, y0, bandwidth=h0),
        }
        for name, res in zero.items():
            se = res.se_conventional if name.endswith("常规方差") else res.se
            z = res.tau / se if se > 0 else float("nan")
            p_value = float(2.0 * stats.norm.sf(abs(z)))
            fpr[name].append(float(p_value < 0.05))

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
        above = (x >= 0.0).astype(float)
        d = (above * (rng.random(n) < 0.4)).astype(float)
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
        bias_bandwidth_rows={
            tag: {k: agg(v) for k, v in vals.items()} for tag, vals in b_rows.items()
        },
        fuzzy={
            "ITT 均值": agg(itt),
            "第一阶段均值": agg(jump_d),
            "LATE 均值": agg(late),
            "LATE 覆盖率": agg(late_cover),
            "第一阶段被门槛拒绝的比例": first_stage_gated / n_trials,
        },
    )
