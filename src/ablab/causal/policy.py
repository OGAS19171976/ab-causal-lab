"""策略学习：直接优化**策略价值**，而不是"先估 CATE 再排序"。

为什么这是另一件事
------------------
"按估计的 CATE 排序、投前 q 份额"是这个仓库一直在用的启发式（``causal/uplift.py``
量过它的排序能力）。它没有目标函数：不知道离最优有多远、不知道怎么定 q、
也无法回答"我的策略比现状好多少"。

策略学习把目标写出来。以"相对谁都不投的增量"为口径：

    Δ(π) = E[ τ(X) · π(X) ]                       （π(X) ∈ {0,1}，1 = 投放）

用交叉拟合的 nuisance 造**单元级效应的 AIPW 伪结果**：

    Γ_i = m̂₁(X_i) − m̂₀(X_i)
          + D_i (Y_i − m̂₁(X_i)) / ê_i
          − (1 − D_i) (Y_i − m̂₀(X_i)) / (1 − ê_i)

则 Δ̂(π) = (1/n) Σ π(X_i) Γ_i。这个估计量对**固定**策略 π 是无偏的，
而且它的影响函数就是 ``π(X)(Γ − Δ)`` —— 所以 SE 直接可得，不用 bootstrap。

这一节真正要量的陷阱
--------------------
Δ̂(π) 对**固定** π 无偏，**对 argmax 不是**。不限制策略类时最优解是
"Γ_i > 0 的全投"：一个把噪声也当信号的规则，样本内价值恒为正，
而它的真实价值可能恰好是 0。策略类越小偏差越小；把"选策略"和"评策略"
放到两份数据上（``split_policy_value``）能把剩下的那部分也去掉。

所以模块给出三件东西，缺一件这个对照就不成立：
  1. ``aipw_effect_scores``：伪结果 + 交叉拟合的诊断（裁剪比例）；
  2. ``learn_threshold_policy``：在**受限策略类**（阈值规则的并集，深度 1/2）上
     最大化 Δ̂ —— 类的大小是显式参数，便于量"容量 ↑ ⇒ 乐观偏差 ↑"；
  3. ``split_policy_value``：样本内价值 vs 分离样本价值，两者之差就是乐观偏差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .hte import _clip_propensity, _clone_fit_predict, _default_nuisance

__all__ = [
    "AipwScores",
    "ConstantPolicy",
    "GreedyThresholdPolicy",
    "LearnedPolicy",
    "PolicyValue",
    "SplitValue",
    "ThresholdPolicy",
    "aipw_effect_scores",
    "aipw_policy_learner",
    "learn_threshold_policy",
    "policy_value",
    "ranking_mask",
    "split_policy_value",
    "unrestricted_mask",
]

#: 任何"策略"都可以是：``X -> 0/1 掩码`` 的可调用对象，或直接给一个掩码。
Policy = Callable[[np.ndarray], np.ndarray] | np.ndarray


# --------------------------------------------------------------------------- #
# 策略类
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstantPolicy:
    """投全部 / 谁都不投 —— 策略学习的两条平凡基线。"""

    treat: bool

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], 1.0 if self.treat else 0.0)

    def describe(self) -> str:
        return "全投" if self.treat else "全不投"


@dataclass(frozen=True)
class ThresholdPolicy:
    """单条阈值规则：``x_j > t`` 就投。

    ``direction = -1`` 表示"小于 t 才投"。只用一维特征的规则族刻意**很笨** ——
    容量小是它的优点，不是缺点：这一节量的正是"容量换偏差"。
    """

    feature: int
    threshold: float
    direction: int = 1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        col = X[:, self.feature]
        if self.direction >= 0:
            return (col > self.threshold).astype(float)
        return (col < self.threshold).astype(float)

    def describe(self) -> str:
        sign = ">" if self.direction >= 0 else "<"
        return f"x{self.feature} {sign} {self.threshold:+.3f}"


@dataclass(frozen=True)
class GreedyThresholdPolicy:
    """若干阈值规则的**并集**（深度 = 规则条数）。

    取并集而不是交集，是因为"满足任一条件就投"在这个应用里更自然，
    而且贪心加规则的搜索空间小、可复现。深度 1 是 ``ThresholdPolicy``，
    深度 2 起容量明显变大 —— 审计里就是靠这个对比量乐观偏差。
    """

    rules: tuple[ThresholdPolicy, ...]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mask = np.zeros(X.shape[0])
        for rule in self.rules:
            mask = np.maximum(mask, rule(X))
        return mask

    @property
    def depth(self) -> int:
        return len(self.rules)

    def describe(self) -> str:
        return " 或 ".join(r.describe() for r in self.rules)


def ranking_mask(score: np.ndarray, *, share: float) -> np.ndarray:
    """按 ``score`` 从大到小投前 ``share`` 份额 —— "CATE 排序当策略"那一支。"""
    score = np.asarray(score, dtype=float).ravel()
    n = score.size
    if not 0.0 < share < 1.0:
        raise ValueError("share 必须在 (0, 1) 内")
    k = int(round(share * n))
    k = min(max(k, 0), n)
    mask = np.zeros(n)
    if k:
        top = np.argsort(-score, kind="stable")[:k]
        mask[top] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# AIPW 伪结果与策略价值
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AipwScores:
    """一份数据上的单元级 AIPW 伪结果 + 它用到的 nuisance 诊断。"""

    gamma: np.ndarray  # (n,) AIPW 伪结果
    propensity: np.ndarray  # (n,) 交叉拟合的 ê
    m1: np.ndarray  # (n,) 交叉拟合的 m̂₁
    m0: np.ndarray  # (n,) 交叉拟合的 m̂₀
    n_folds: int
    clipped_share: float  # 被裁剪的 ê 比例（重叠差时这个数会变大）

    @property
    def n(self) -> int:
        return int(self.gamma.size)

    @property
    def ate(self) -> float:
        """伪结果的均值 = ATE 的 AIPW 估计（= 全投策略的价值）。"""
        return float(self.gamma.mean())

    def summary(self) -> str:
        return (
            f"AIPW 伪结果：n={self.n}，交叉拟合 {self.n_folds} 折\n"
            f"  Γ 的均值（= 全投的价值）= {self.ate:+.4f}，"
            f"SD = {self.gamma.std(ddof=1):.4f}\n"
            f"  ê 范围 [{self.propensity.min():.4f}, {self.propensity.max():.4f}]，"
            f"被裁剪比例 {self.clipped_share:.4f}\n"
            f"  Γ > 0 的单元占 {float((self.gamma > 0).mean()):.4f}"
            f"（这是「不限制策略类」时最优策略投的份额）"
        )


def aipw_effect_scores(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    n_folds: int = 5,
    clip: float = 0.02,
    seed: int = 0,
) -> AipwScores:
    """交叉拟合的 AIPW 单元级伪结果（DR-learner 的 ψ，但不往下回归）。"""
    X = np.asarray(X, dtype=float)
    D = np.asarray(D, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = Y.size
    if not (D.size == n and X.shape[0] == n):
        raise ValueError("X / D / Y 的样本量必须一致")
    treated = D > 0.5
    if treated.sum() < 5 or (~treated).sum() < 5:
        raise ValueError("两组的样本量都至少要有 5 个")

    from sklearn.model_selection import KFold

    if n_folds <= 1:
        folds = [(np.arange(n), np.arange(n))]
    else:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))

    m1 = np.empty(n)
    m0 = np.empty(n)
    e_hat = np.empty(n)
    for train, test in folds:
        tr_t, tr_c = train[treated[train]], train[~treated[train]]
        if tr_t.size < 5 or tr_c.size < 5:
            raise ValueError("某个训练折里某一臂的样本不足 5 个")
        m1[test] = _clone_fit_predict(_default_nuisance(), X[tr_t], Y[tr_t], X[test])
        m0[test] = _clone_fit_predict(_default_nuisance(), X[tr_c], Y[tr_c], X[test])
        e_hat[test] = _clone_fit_predict(_default_nuisance(), X[train], D[train], X[test])

    e_clip = _clip_propensity(e_hat, clip)
    weight = (D - e_clip) / (e_clip * (1.0 - e_clip))
    gamma = m1 - m0 + weight * (Y - np.where(treated, m1, m0))

    return AipwScores(
        gamma=gamma,
        propensity=e_clip,
        m1=m1,
        m0=m0,
        n_folds=max(n_folds, 1),
        clipped_share=float(np.mean(np.abs(e_clip - e_hat) > 1e-12)),
    )


@dataclass(frozen=True)
class PolicyValue:
    """一个**固定**策略的 AIPW 价值 + 影响函数 SE。"""

    value: float
    se: float
    n: int
    treated_share: float

    @property
    def ci(self) -> tuple[float, float]:
        return (self.value - 1.96 * self.se, self.value + 1.96 * self.se)

    def covers(self, truth: float) -> bool:
        lo, hi = self.ci
        return bool(lo <= truth <= hi)

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"策略价值 Δ̂ = {self.value:+.4f}（SE {self.se:.4f}，"
            f"95% 区间 [{lo:+.4f}, {hi:+.4f}]），投放份额 {self.treated_share:.4f}"
        )


def _as_mask(policy: Policy, X: np.ndarray) -> np.ndarray:
    """策略 → 0/1 掩码：可调用对象或直接给的数组都接受（但形状要查）。"""
    if callable(policy):
        mask = np.asarray(policy(X), dtype=float)
    else:
        mask = np.asarray(policy, dtype=float)
    mask = mask.ravel()
    if mask.size != X.shape[0]:
        raise ValueError(f"策略给出的掩码长度 {mask.size} 与样本量 {X.shape[0]} 不一致")
    if not np.all(np.isin(mask, (0.0, 1.0))):
        raise ValueError("策略掩码只能是 0/1（这里不做软策略/随机策略）")
    return mask


def policy_value(scores: AipwScores, X: np.ndarray, policy: Policy) -> PolicyValue:
    """固定策略的 AIPW 价值与 SE（影响函数：``π(X)(Γ − Δ)``）。"""
    X = np.asarray(X, dtype=float)
    mask = _as_mask(policy, X)
    n = mask.size
    contribution = mask * scores.gamma
    value = float(contribution.mean())
    influence = contribution - value
    se = float(influence.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return PolicyValue(
        value=value, se=se, n=n, treated_share=float(mask.mean())
    )


def unrestricted_mask(scores: AipwScores) -> np.ndarray:
    """**不限制策略类**时的"最优"策略：Γ_i > 0 就投。

    这看起来像"每个单元都按自己的估计效应决定"，实际是把噪声当成信号 ——
    它的样本内价值恒为正（正部均值），真实价值却未必。审计里它是乐观偏差的上界。
    """
    return (scores.gamma > 0).astype(float)


# --------------------------------------------------------------------------- #
# 在受限策略类上最大化价值
# --------------------------------------------------------------------------- #
def learn_threshold_policy(
    scores: AipwScores,
    X: np.ndarray,
    *,
    depth: int = 1,
    n_grid: int = 25,
    min_share: float = 0.02,
    max_share: float = 0.98,
) -> tuple[GreedyThresholdPolicy, int]:
    """贪心前向搜索：每次加一条让 Δ̂ 提升最多的阈值规则。

    返回 ``(策略, 评估过的候选数)`` —— 候选数是"容量"的另一个刻度，
    审计里"容量 ↑ ⇒ 乐观偏差 ↑"就是靠它和深度一起量的。
    """
    X = np.asarray(X, dtype=float)
    if depth < 1:
        raise ValueError("depth 至少为 1")
    n, p = X.shape
    quantiles = np.linspace(min_share, max_share, n_grid)
    grid = [np.quantile(X[:, j], quantiles) for j in range(p)]

    rules: list[ThresholdPolicy] = []
    n_candidates = 0
    current = np.zeros(n)
    current_value = 0.0
    for _ in range(depth):
        best_gain = 0.0
        best_rule: ThresholdPolicy | None = None
        best_mask: np.ndarray | None = None
        for j in range(p):
            for t in grid[j]:
                for direction in (1, -1):
                    n_candidates += 1
                    rule = ThresholdPolicy(feature=j, threshold=float(t), direction=direction)
                    mask = np.maximum(current, rule(X))
                    value = float((mask * scores.gamma).mean())
                    if value - current_value > best_gain:
                        best_gain, best_rule, best_mask = value - current_value, rule, mask
        if best_rule is None or best_mask is None:
            break
        rules.append(best_rule)
        current, current_value = best_mask, current_value + best_gain
    if not rules:  # 一条规则都没能让价值变好 —— 返回"全不投"
        return GreedyThresholdPolicy(rules=()), n_candidates
    return GreedyThresholdPolicy(rules=tuple(rules)), n_candidates


@dataclass(frozen=True)
class SplitValue:
    """把"选策略"和"评策略"分到两份数据上得到的价值（每折一个读数）。"""

    value: float
    fold_values: tuple[float, ...]
    fold_shares: tuple[float, ...]

    def summary(self) -> str:
        folds = ", ".join(f"{v:+.4f}" for v in self.fold_values)
        return f"分离样本价值 = {self.value:+.4f}（各折：{folds}）"


def split_policy_value(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    depth: int = 1,
    n_grid: int = 25,
    n_splits: int = 2,
    clip: float = 0.02,
    n_folds: int = 5,
    seed: int = 0,
) -> SplitValue:
    """样本分离的策略价值：在第 k 份上**选**策略，在第 k+1 份上**评**它。

    两折时正反各做一次（A 选 B 评、B 选 A 评），所以每个单元既当过选择集、
    也当过评价集 —— 但**从不同时**。这是策略学习里最便宜的一种诚实：
    代价是每折只能用一半数据学策略，容量越大的类损失越多。
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = Y.size
    if n_splits < 2:
        raise ValueError("n_splits 至少为 2（否则没有分离）")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    chunks = np.array_split(order, n_splits)

    values: list[float] = []
    shares: list[float] = []
    for k in range(n_splits):
        test = chunks[k]
        train = np.concatenate([chunks[j] for j in range(n_splits) if j != k])
        train_scores = aipw_effect_scores(
            Y[train], D[train], X[train], clip=clip, n_folds=n_folds, seed=seed + k
        )
        policy, _ = learn_threshold_policy(
            train_scores, X[train], depth=depth, n_grid=n_grid
        )
        # 评价用**这一折自己的** nuisance —— 策略是固定的，这里没有二次选择。
        test_scores = aipw_effect_scores(
            Y[test], D[test], X[test], clip=clip, n_folds=n_folds, seed=seed + 100 + k
        )
        pv = policy_value(test_scores, X[test], policy)
        values.append(pv.value)
        shares.append(pv.treated_share)
    return SplitValue(
        value=float(np.mean(values)),
        fold_values=tuple(values),
        fold_shares=tuple(shares),
    )


@dataclass(frozen=True)
class LearnedPolicy:
    """一次策略学习的全部读数 —— 关键是**两个**价值，不是一个。"""

    policy: GreedyThresholdPolicy
    in_sample: PolicyValue
    split: SplitValue
    n_candidates: int
    depth: int

    @property
    def optimism(self) -> float:
        """样本内价值 − 分离样本价值 = 选择带来的乐观偏差。"""
        return self.in_sample.value - self.split.value

    @property
    def optimism_share(self) -> float:
        """乐观偏差占样本内价值的比例（样本内价值接近 0 时无意义，返回 nan）。"""
        if abs(self.in_sample.value) < 1e-12:
            return float("nan")
        return self.optimism / self.in_sample.value

    def summary(self) -> str:
        return (
            f"学到的策略（深度 {self.depth}，评估 {self.n_candidates} 个候选）："
            f"{self.policy.describe()}\n"
            f"  样本内：{self.in_sample.summary()}\n"
            f"  分离样本：{self.split.summary()}\n"
            f"  乐观偏差 = {self.optimism:+.4f}"
            f"（占样本内价值 {self.optimism_share:.1%}）"
        )


def aipw_policy_learner(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    depth: int = 1,
    n_grid: int = 25,
    n_splits: int = 2,
    clip: float = 0.02,
    seed: int = 0,
) -> LearnedPolicy:
    """AIPW 策略学习：在受限策略类上最大化 Δ̂，并同时报出分离样本价值。

    这个函数**刻意同时返回两个数**。只报样本内价值的策略学习，在任何数据上
    都能给出一个正数 —— 那个数衡量的是搜索空间有多大，不是策略有多好。
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    scores = aipw_effect_scores(Y, D, X, clip=clip, seed=seed)
    policy, n_candidates = learn_threshold_policy(
        scores, X, depth=depth, n_grid=n_grid
    )
    in_sample = policy_value(scores, X, policy)
    split = split_policy_value(
        Y, D, X, depth=depth, n_grid=n_grid, n_splits=n_splits, clip=clip, seed=seed
    )
    return LearnedPolicy(
        policy=policy,
        in_sample=in_sample,
        split=split,
        n_candidates=n_candidates,
        depth=depth,
    )


def _describe_rules(policies: Sequence[Policy]) -> str:
    """给审计用：把一组策略压成一行（只给能自描述的那些）。"""
    parts = []
    for pol in policies:
        desc = getattr(pol, "describe", None)
        parts.append(desc() if callable(desc) else type(pol).__name__)
    return "; ".join(parts)
