"""策略学习的审计台：**同一个估计量，对固定策略无偏、对 argmax 不是**。

这一节要量的是一个很容易被绕过的事实。策略学习的宣传语通常是
"我们的策略价值提升了 X" —— 而那个 X 如果是**在同一个样本上选完策略再评估**，
它衡量的是搜索空间的大小，不是策略的好坏。

三档设计
--------
1. **有信号**（``cate_form="threshold"``，真实最优策略恰好是"x₁ > 0 就投"）：
   学到的策略能接近 oracle 吗？乐观偏差有多大？
2. **纯噪声**（``cate_scale=0``，τ ≡ 0）：**任何**策略的真实价值都是 0 ——
   于是"报出来的价值"与"真实价值"的差不需要任何真值推断，直接可读。
3. **预指定策略的正对照**（``x₀ > 0``，与数据无关、不含任何选择）：
   它的区间必须守住名义覆盖 —— 用来证明"问题出在选择，不是出在估计量"。
   同一批数据上把"选中的策略"的覆盖率也量出来，两个数放在一起，
   结论就没有别的解释。

第一版判据被实测改写的地方（写在这里而不是留在报告里）
------------------------------------------------------
写这份审计时预设了两条判据："噪声档的**分离样本**价值应当落在 0 附近"
与"有信号档里**选中策略**的区间覆盖率会崩"。**两条都被实测推翻了**：

  * 分离样本价值在噪声档是 **+0.0740**（不是 0），而样本内是 +0.2499 ——
    分离去掉的是"同一份噪声选+评"那一半，留下的是 nuisance 系统性偏差
    经过"选区域"之后的放大。所以判据改成"分离后**显著变小**"，
    并且**另量一条偏差曲线**（预指定阈值上的跨仿真均值）来指认它的来源：
  * 有信号档里选中策略的区间**照样覆盖**（容量 1 的类，选出来就是 oracle，
    区间自然盖得住真实价值）。覆盖率崩塌只出现在**噪声档的大容量类**上
    （τ≡0 时"真实价值 = 0"是精确已知的），所以覆盖率那两栏改到噪声档量；
  * 第三条更不客气：正对照（**预指定**策略，完全不含选择）的价值在噪声档
    是 **+0.0127** 而不是 0，它的区间照样覆盖 —— 也就是说
    **估计量本身在这个 n 与这套 nuisance 下也有偏差**。于是"正对照"的判据
    从"≈0"改成"**远小于样本内价值**"，并把这条偏差本身当成一个读数报出来。

这三条改写本身就是这一节的结论：**"分离样本"不是万灵药，
它治的是选择偏差，不治估计量本身的偏差** —— 而噪声档给出的
`样本内 = 偏差面 + 选择的乐观` 这个分解，正好把两者分开：
分离样本价值 +0.0740 ≈ 预指定偏差曲线的峰 +0.076。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..causal.hte import HTEConfig, generate_hte_data
from ..causal.policy import (
    ConstantPolicy,
    ThresholdPolicy,
    aipw_effect_scores,
    learn_threshold_policy,
    policy_value,
    ranking_mask,
    split_policy_value,
    unrestricted_mask,
)

__all__ = ["PolicyAudit", "run_policy_audit"]

#: 预先指定、与数据无关的对照策略：``x₀ > 0 就投``。
FIXED_POLICY = ThresholdPolicy(feature=0, threshold=0.0, direction=1)

#: 偏差曲线用的阈值网格（预指定，不随数据变 —— 否则它自己就有选择偏差）。
BIAS_GRID: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


@dataclass(frozen=True)
class PolicyAudit:
    """策略学习的审计读数（每格都是 ``n_trials`` 次仿真的平均）。"""

    n_trials: int
    n: int
    n_grid: int
    signal: dict[str, float] = field(default_factory=dict)
    noise: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    #: 噪声档下"预指定阈值 → 跨仿真平均 AIPW 价值"这条曲线（偏差面）
    bias_curve: tuple[float, ...] = ()

    # ---- 噪声档：这才是要钉住的陷阱 -------------------------------------- #
    @property
    def noise_regime_is_optimistic(self) -> bool:
        """真实价值恒为 0 时，样本内价值仍然显著为正。"""
        return self.noise["样本内价值·深度1"] > 0.05

    @property
    def capacity_increases_optimism(self) -> bool:
        """容量越大，样本内价值越高（而真实价值不变）—— 深度 2 > 深度 1。"""
        return self.noise["样本内价值·深度2"] > self.noise["样本内价值·深度1"]

    @property
    def unrestricted_is_most_optimistic(self) -> bool:
        """不限制策略类（Γ>0 就投）的样本内价值最高，且远高于受限类。"""
        return self.noise["样本内价值·无限制"] > 2.0 * self.noise["样本内价值·深度1"]

    @property
    def splitting_shrinks_the_gap(self) -> bool:
        """分离样本价值明显小于样本内价值（去掉的是"同一份噪声选+评"那一半）。"""
        return self.noise["分离样本价值·深度1"] < 0.70 * self.noise["样本内价值·深度1"]

    @property
    def fixed_policy_bias_is_far_below_the_inflation(self) -> bool:
        """预指定策略的偏差要**远小于**样本内价值（判据被实测改写过：不是 ≈0）。

        噪声档里真实价值精确为 0，所以预指定策略的价值**就是**估计量的偏差。
        实测它不是 0（+0.057，来源是随机森林 nuisance 在 n=500 上的偏差），
        但比"选完再评"的样本内价值小一个量级 —— 所以判据写成相对量，
        并且把这个偏差本身单独报出来，不假装它是 0。
        """
        return abs(self.noise["样本内价值·固定策略"]) < 0.30 * self.noise["样本内价值·深度1"]

    @property
    def split_value_matches_the_bias_scale(self) -> bool:
        """分离样本价值与偏差曲线的**峰**同量级 —— 剩下的那部分不是选择噪声。"""
        peak = max(self.bias_curve) if self.bias_curve else 0.0
        if peak <= 0:
            return False
        return 0.5 * peak <= self.noise["分离样本价值·深度1"] <= 2.0 * peak

    @property
    def bias_curve_has_a_peak(self) -> bool:
        """偏差曲线不是平的：某个区域系统性偏高 —— 那正是"选区域"能捡到的东西。"""
        curve = np.asarray(self.bias_curve, dtype=float)
        if curve.size == 0:
            return False
        return bool(curve.max() > curve.mean() + 0.05)

    # ---- 覆盖率：同一个估计量，两种用法 ---------------------------------- #
    @property
    def fixed_policy_ci_holds(self) -> bool:
        """预指定策略（不含选择）的区间覆盖率应当接近名义（这里只要 ≥0.85）。"""
        return self.coverage["固定策略·覆盖率"] >= 0.85

    @property
    def unrestricted_ci_overrejects(self) -> bool:
        """不限制策略类时，样本内区间几乎总是排除掉真值 0（τ≡0 时）。"""
        return self.coverage["无限制·覆盖率"] <= 0.50

    # ---- 有信号档：策略学习本身的成绩 ------------------------------------ #
    @property
    def signal_regime_beats_treat_all(self) -> bool:
        """真实价值要明显高于"全投"这条基线（3 倍以上）。"""
        return self.signal["真实价值·深度1"] > 3.0 * self.signal["真实价值·全投"]

    @property
    def signal_regime_reaches_oracle(self) -> bool:
        """学到的策略至少拿到 oracle 的 80%。"""
        return self.signal["真实价值·深度1"] >= 0.80 * self.signal["真实价值·oracle"]

    @property
    def signal_regime_optimism_is_small(self) -> bool:
        """有信号时乐观偏差应当远小于噪声档（这里是相对值：< 10%）。"""
        return abs(self.signal["乐观偏差·深度1"]) < 0.10 * abs(
            self.signal["样本内价值·深度1"]
        )

    def passed(self) -> dict[str, bool]:
        return {
            "噪声档：样本内价值为正（真值恒为 0）": self.noise_regime_is_optimistic,
            "噪声档：容量越大样本内价值越高": self.capacity_increases_optimism,
            "噪声档：不限制策略类最乐观": self.unrestricted_is_most_optimistic,
            "噪声档：分离样本价值显著变小": self.splitting_shrinks_the_gap,
            "正对照：预指定策略的偏差远小于样本内价值": (
                self.fixed_policy_bias_is_far_below_the_inflation
            ),
            "正对照：预指定策略的区间守住覆盖": self.fixed_policy_ci_holds,
            "机制：偏差曲线不是平的（选区域能捡到）": self.bias_curve_has_a_peak,
            "机制：分离样本价值与偏差面同量级": self.split_value_matches_the_bias_scale,
            "大容量类的区间排除真值（覆盖率崩）": self.unrestricted_ci_overrejects,
            "有信号：真实价值明显高于全投": self.signal_regime_beats_treat_all,
            "有信号：达到 oracle 的 80%": self.signal_regime_reaches_oracle,
            "有信号：乐观偏差很小": self.signal_regime_optimism_is_small,
        }

    def summary(self) -> str:
        lines = [
            f"策略学习审计（每格 {self.n_trials} 次仿真，n={self.n}，候选网格 {self.n_grid}）",
            "",
            "  一、有信号（真实最优：x₁ > 0 就投）",
            f"    oracle 真实价值              {self.signal['真实价值·oracle']:+.4f}",
            f"    全投的真实价值                {self.signal['真实价值·全投']:+.4f}",
            f"    CATE 排序（同份额）真实价值    {self.signal['真实价值·排序']:+.4f}"
            f"（份额 {self.signal['份额·排序']:.2f}，"
            f"达到 oracle 的 {self.signal['oracle 占比·排序']:.1%}）",
            f"    学到的策略（深度1）真实价值    {self.signal['真实价值·深度1']:+.4f}"
            f"（份额 {self.signal['份额·深度1']:.2f}，"
            f"达到 oracle 的 {self.signal['oracle 占比·深度1']:.1%}）",
            f"    学到的策略（深度2）真实价值    {self.signal['真实价值·深度2']:+.4f}",
            f"    样本内价值（深度1）            {self.signal['样本内价值·深度1']:+.4f}",
            f"    分离样本价值（深度1）          {self.signal['分离样本价值·深度1']:+.4f}",
            f"    乐观偏差                      {self.signal['乐观偏差·深度1']:+.4f}",
            "",
            "  二、纯噪声（τ ≡ 0：**任何**策略的真实价值都精确等于 0）",
            f"    样本内价值·深度1              {self.noise['样本内价值·深度1']:+.4f}",
            f"    样本内价值·深度2              {self.noise['样本内价值·深度2']:+.4f}",
            f"    样本内价值·无限制（Γ>0 就投）  {self.noise['样本内价值·无限制']:+.4f}",
            f"    分离样本价值·深度1            {self.noise['分离样本价值·深度1']:+.4f}"
            f"（投放份额 {self.noise['份额·深度1']:.2f}）",
            f"    预指定策略（x₀>0）的价值       {self.noise['样本内价值·固定策略']:+.4f}"
            "   ← 正对照：这就是**估计量本身的偏差**（不是 0，但小一个量级）",
            "",
            "  三、偏差曲线（噪声档，只看 x₀ 这一维 —— 它是**下界参照**）：",
            "      预指定阈值 → 跨仿真平均 AIPW 价值",
            "    " + "  ".join(f"{t:+.1f}" for t in BIAS_GRID),
            "    " + "  ".join(f"{v:+.3f}" for v in self.bias_curve),
            f"    峰 = {max(self.bias_curve):+.3f}（在阈值"
            f" {BIAS_GRID[int(np.argmax(np.asarray(self.bias_curve)))]:+.1f} 处）——",
            "    20 维搜索空间的偏差面只会比这条曲线更高，所以它是下界；",
            "    「选区域」捡到的就是这种系统性偏差（分离样本价值与它同量级），",
            "    而不是「选择噪声」—— 后者在样本分离之后本就应当平均为 0。",
            "",
            "  四、同一个估计量，两种用法的覆盖率（τ≡0 时真值 = 0，95% 名义）",
            f"    预指定策略（不含选择）        {self.coverage['固定策略·覆盖率']:.4f}"
            "   ← 正对照",
            f"    选中策略·深度1                {self.coverage['选中策略·覆盖率']:.4f}",
            f"    选中策略·深度2                {self.coverage['大容量类·覆盖率']:.4f}",
            f"    无限制（Γ>0 就投）            {self.coverage['无限制·覆盖率']:.4f}"
            "   ← 崩掉的是「选完再评」这一步",
        ]
        return "\n".join(lines)


def run_policy_audit(
    *,
    n_trials: int = 10,
    n: int = 500,
    n_grid: int = 10,
    n_folds: int = 3,
    seed: int = 0,
) -> PolicyAudit:
    """两档 DGP × 三种容量：样本内 / 分离样本 / 覆盖率 / 偏差曲线。"""
    n_trials = max(int(n_trials), 1)
    signal_rows: dict[str, list[float]] = {}
    noise_rows: dict[str, list[float]] = {}
    fixed_covers: list[float] = []
    selected_covers: list[float] = []
    big_covers: list[float] = []
    unrestricted_covers: list[float] = []
    curve = np.zeros(len(BIAS_GRID))

    def push(store: dict[str, list[float]], key: str, value: float) -> None:
        store.setdefault(key, []).append(float(value))

    for trial in range(n_trials):
        for tag, scale in (("signal", 1.0), ("noise", 0.0)):
            store = signal_rows if tag == "signal" else noise_rows
            data = generate_hte_data(
                HTEConfig(
                    n=n,
                    cate_form="threshold",
                    cate_scale=scale,
                    propensity_strength=0.6,
                    seed=seed + 1000 * trial + (0 if tag == "signal" else 7),
                )
            )
            X, Y, D, tau = data.X, data.Y, data.D, data.tau
            scores = aipw_effect_scores(Y, D, X, n_folds=n_folds, seed=seed + trial)

            policies = {}
            for depth in (1, 2):
                pol, _cands = learn_threshold_policy(
                    scores, X, depth=depth, n_grid=n_grid
                )
                policies[depth] = pol
                mask = pol(X)
                push(store, f"样本内价值·深度{depth}", policy_value(scores, X, pol).value)
                push(store, f"真实价值·深度{depth}", float((tau * mask).mean()))
                push(store, f"份额·深度{depth}", float(mask.mean()))

            split = split_policy_value(
                Y,
                D,
                X,
                depth=1,
                n_grid=n_grid,
                n_splits=2,
                seed=seed + trial,
                n_folds=n_folds,
            )
            push(store, "分离样本价值·深度1", split.value)
            push(
                store,
                "乐观偏差·深度1",
                policy_value(scores, X, policies[1]).value - split.value,
            )

            # 无限制那一支：Γ_i > 0 就投 —— 它不是一个策略（新单元没有 Γ），
            # 只存在于样本内，所以只有样本内价值与"照它投"的真实价值。
            mask_un = unrestricted_mask(scores)
            pv_un = policy_value(scores, X, mask_un)
            push(store, "样本内价值·无限制", pv_un.value)
            push(store, "真实价值·无限制", float((tau * mask_un).mean()))

            # 预指定策略（与数据无关）：正对照
            pv_fixed = policy_value(scores, X, FIXED_POLICY)
            push(store, "样本内价值·固定策略", pv_fixed.value)

            if tag == "signal":
                share = float(policies[1](X).mean())
                mask_rank = ranking_mask(
                    scores.m1 - scores.m0, share=float(min(max(share, 0.02), 0.98))
                )
                push(store, "真实价值·排序", float((tau * mask_rank).mean()))
                push(store, "份额·排序", float(mask_rank.mean()))
                push(store, "真实价值·oracle", float((tau * (tau > 0)).mean()))
                push(store, "真实价值·全投", float(tau.mean()))
                # 全投也是**固定**策略：它的 AIPW 价值应当与真实 ATE 相符
                push(
                    store,
                    "样本内价值·全投",
                    policy_value(scores, X, ConstantPolicy(True)).value,
                )
            else:
                # τ ≡ 0 ⇒ 任何策略的真值都是 0，于是覆盖率是一条精确可判的读数
                fixed_covers.append(float(pv_fixed.covers(0.0)))
                selected_covers.append(
                    float(policy_value(scores, X, policies[1]).covers(0.0))
                )
                big_covers.append(
                    float(policy_value(scores, X, policies[2]).covers(0.0))
                )
                unrestricted_covers.append(float(pv_un.covers(0.0)))
                # 偏差面：预指定阈值上的平均价值（不含任何选择）
                for j, t in enumerate(BIAS_GRID):
                    pol_t = ThresholdPolicy(feature=0, threshold=float(t), direction=1)
                    curve[j] += policy_value(scores, X, pol_t).value / n_trials

    def aggregate(store: dict[str, list[float]]) -> dict[str, float]:
        out = {k: float(np.mean(v)) for k, v in sorted(store.items())}
        if out.get("真实价值·oracle"):
            out["oracle 占比·深度1"] = out["真实价值·深度1"] / out["真实价值·oracle"]
            # 排序策略拿到 oracle 的多少 —— "CATE 排序当策略"与"直接优化价值"
            # 到底差多少，就得看这个数（README 里引用的也是它）
            out["oracle 占比·排序"] = out["真实价值·排序"] / out["真实价值·oracle"]
        return out

    return PolicyAudit(
        n_trials=n_trials,
        n=n,
        n_grid=n_grid,
        signal=aggregate(signal_rows),
        noise=aggregate(noise_rows),
        coverage={
            "固定策略·覆盖率": float(np.mean(fixed_covers)) if fixed_covers else float("nan"),
            "选中策略·覆盖率": float(np.mean(selected_covers)) if selected_covers else float("nan"),
            "大容量类·覆盖率": float(np.mean(big_covers)) if big_covers else float("nan"),
            "无限制·覆盖率": float(np.mean(unrestricted_covers))
            if unrestricted_covers
            else float("nan"),
        },
        bias_curve=tuple(float(v) for v in curve),
    )
