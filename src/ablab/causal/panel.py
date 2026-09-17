"""面板数据与交错处置的合成 DGP。

M3 第一次离开"有随机化撑腰"的安全区。这里的数据生成器必须**同时**给出：

1. 真实的处理效应路径（才知道估计量偏了多少）
2. **可控的平行趋势违背**（才知道假设破了会怎样）
3. 两种违背：事先能看出来的（队列专属趋势）和**看不出来的**（处置后才分岔）

第 3 条是 M3 最重要的一课：事前的平行趋势检验只能验证"处置前看起来平行"，
**它无法排除"处置后才分岔"**。这个 DGP 就是为了把那件事量出来。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StaggeredPanelConfig",
    "Panel",
    "GroundTruth",
    "generate_staggered_panel",
    "never_treated_code",
]

#: 未处置队列的内部编码（比任何真实期数都大）
never_treated_code = 10**6


@dataclass(frozen=True)
class StaggeredPanelConfig:
    """交错处置面板的超参数。

    Parameters
    ----------
    cohorts:
        各队列的**首次处置期**（1-based）。未处置单元由 ``never_treated_share`` 控制。
    effects:
        动态效应 ``effects[k]`` = 处置后第 ``k`` 期的效应。超出长度后沿用最后一个值。
    trend_violation:
        队列专属线性趋势的强度。**这会同时污染处置前的期数**，
        所以事前的平行趋势检验能发现它。
    post_divergence:
        处置后才出现的趋势分岔强度。**它不影响处置前**，
        所以事前检验对它**完全没有功效** —— 这是平行趋势假设的真正盲区。
    """

    n_units: int = 400
    n_periods: int = 10
    cohorts: tuple[int, ...] = (3, 5, 7)
    cohort_weights: tuple[float, ...] = (0.25, 0.25, 0.25)
    never_treated_share: float = 0.25
    unit_fe_sd: float = 1.0
    time_trend: float = 0.4
    noise_sd: float = 0.5
    effects: tuple[float, ...] = (1.0, 2.0, 3.0, 3.0)
    #: 队列间的效应倍数（按队列先后排序）。**这是 TWFE 出现负权重的燃料**：
    #: 后处置的队列效应更大时，"早组 vs 晚组"这个比较会把晚组自己的效应当成
    #: 对照组的变化扣掉，从而给出偏小甚至负的估计。留空则各队列相同。
    cohort_effect_multiplier: tuple[float, ...] = ()
    trend_violation: float = 0.0
    post_divergence: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.cohorts) != len(self.cohort_weights):
            raise ValueError("cohorts 与 cohort_weights 长度必须一致")
        if self.cohort_effect_multiplier and len(self.cohort_effect_multiplier) != len(
            self.cohorts
        ):
            raise ValueError("cohort_effect_multiplier 长度必须与 cohorts 一致")
        if not 0.0 <= self.never_treated_share < 1.0:
            raise ValueError("never_treated_share 必须落在 [0, 1)")
        if self.never_treated_share == 0.0 and len(self.cohorts) < 2:
            raise ValueError(
                "没有未处置组时至少要两个处置队列，否则不存在干净的两两比较"
            )
        if not self.effects:
            raise ValueError("effects 不能为空")
        for c in self.cohorts:
            if not 1 <= c <= self.n_periods:
                raise ValueError(f"处置期 {c} 必须落在 [1, {self.n_periods}]")


@dataclass(frozen=True)
class Panel:
    """平衡面板。内部用 ``(N, T)`` 矩阵，便于双向固定效应。"""

    outcome: np.ndarray  # (N, T)
    treated: np.ndarray  # (N, T)，吸收型：一旦处置就不再回退
    cohort: np.ndarray  # (N,)，首次处置期；未处置为 never_treated_code
    periods: np.ndarray  # (T,) 1..T

    @property
    def n_units(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.outcome.shape[1])

    @property
    def treated_units(self) -> np.ndarray:
        return self.cohort != never_treated_code

    @property
    def never_treated_units(self) -> np.ndarray:
        return self.cohort == never_treated_code

    def cohorts(self) -> np.ndarray:
        """所有出现过的处置期（升序）。"""
        return np.unique(self.cohort[self.treated_units])

    def group_size(self, cohort: int) -> int:
        return int(np.sum(self.cohort == cohort))

    def long(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """摊平成 ``(unit, period, outcome, treated)`` 四条长向量。"""
        n, t = self.outcome.shape
        unit = np.repeat(np.arange(n), t)
        period = np.tile(self.periods, n)
        return unit, period, self.outcome.ravel(), self.treated.ravel()

    def summary(self) -> str:
        parts = [f"面板：{self.n_units} 个单元 × {self.n_periods} 期"]
        for g in self.cohorts():
            parts.append(f"  队列 {g}: {self.group_size(int(g))} 个单元")
        n_never = int(self.never_treated_units.sum())
        if n_never:
            parts.append(f"  未处置: {n_never} 个单元")
        return "\n".join(parts)


@dataclass(frozen=True)
class GroundTruth:
    """仿真才知道的真值。"""

    #: ``att[g][k]`` = 队列 g 在相对期数 k 的真实效应（不是所有队列都有全部 k）
    event_study: dict[int, float]
    overall_att: float
    #: 每个 (队列, 绝对期数) 的真实效应，便于逐格对照
    group_time: dict[tuple[int, int], float]
    #: 各队列单元上的平均处理效应（用来算加权的 truth）
    cohort_att: dict[int, float]

    def summary(self) -> str:
        lines = [f"真值：整体 ATT = {self.overall_att:+.4f}"]
        for g, v in sorted(self.cohort_att.items()):
            lines.append(f"  队列 {g} 的平均效应 = {v:+.4f}")
        lines.append("  事件研究（相对期数 k -> 真实效应）：")
        for k, v in sorted(self.event_study.items()):
            lines.append(f"    k={k:>2}: {v:+.4f}")
        return "\n".join(lines)


def generate_staggered_panel(
    config: StaggeredPanelConfig | None = None,
    **overrides,
) -> tuple[Panel, GroundTruth]:
    """生成交错处置面板并返回真值。"""
    cfg = config or StaggeredPanelConfig(**overrides)
    rng = np.random.default_rng(cfg.seed)

    n, T = cfg.n_units, cfg.n_periods

    # ---- 分配队列 -------------------------------------------------------- #
    n_never = int(round(n * cfg.never_treated_share))
    n_treated = n - n_never
    weights = np.asarray(cfg.cohort_weights, dtype=float)
    weights = weights / weights.sum()
    counts = np.floor(weights * n_treated).astype(int)
    counts[-1] += n_treated - counts.sum()  # 余数给最后一个队列

    # 一次洗牌决定谁属于哪个队列；排序靠后的 n_never 个保持未处置
    order = rng.permutation(n)
    cohort = np.full(n, never_treated_code, dtype=int)
    cursor = 0
    for g, c in zip(cfg.cohorts, counts):
        cohort[order[cursor : cursor + c]] = g
        cursor += c

    periods = np.arange(1, T + 1)
    treated = (periods[None, :] >= cohort[:, None]) & (cohort[:, None] != never_treated_code)

    # ---- 结构项 ---------------------------------------------------------- #
    unit_fe = rng.normal(0.0, cfg.unit_fe_sd, n)

    # 队列专属趋势：让平行趋势不成立，且**处置前就能看出来**
    cohort_rank = np.zeros(n)
    for rank, g in enumerate(sorted(cfg.cohorts)):
        cohort_rank[cohort == g] = rank + 1
    unit_trend = cfg.time_trend + cfg.trend_violation * cohort_rank

    outcome = unit_fe[:, None] + unit_trend[:, None] * periods[None, :]

    # ---- 动态处理效应 ---------------------------------------------------- #
    ordered_cohorts = sorted(cfg.cohorts)
    multiplier = (
        cfg.cohort_effect_multiplier
        if cfg.cohort_effect_multiplier
        else (1.0,) * len(ordered_cohorts)
    )
    scale_of = {g: multiplier[i] for i, g in enumerate(ordered_cohorts)}

    group_time: dict[tuple[int, int], float] = {}
    for g in cfg.cohorts:
        mask = cohort == g
        if not mask.any():
            continue
        scale = scale_of[g]
        for t in periods:
            k = int(t - g)
            if k < 0:
                continue
            eff = cfg.effects[min(k, len(cfg.effects) - 1)] * scale
            outcome[mask, int(t) - 1] += eff
            group_time[(int(g), int(t))] = float(eff)

    # ---- 处置后才分岔的平行趋势违背（事前检验的盲区） -------------------- #
    if cfg.post_divergence != 0.0:
        for g in cfg.cohorts:
            mask = cohort == g
            if not mask.any():
                continue
            for t in periods:
                k = int(t - g)
                if k >= 0:
                    outcome[mask, int(t) - 1] += cfg.post_divergence * (k + 1)

    outcome += rng.normal(0.0, cfg.noise_sd, (n, T))

    panel = Panel(
        outcome=outcome,
        treated=treated,
        cohort=cohort,
        periods=periods,
    )

    # ---- 真值 ------------------------------------------------------------ #
    event_study: dict[int, float] = {}
    for g in cfg.cohorts:
        for t in periods:
            k = int(t - g)
            if k >= 0:
                event_study[k] = float(cfg.effects[min(k, len(cfg.effects) - 1)])

    # 整体 ATT：按 (单元, 处置后期数) 等权，这是 CS 的 "simple" 聚合口径
    per_obs = []
    for i in range(n):
        if cohort[i] == never_treated_code:
            continue
        scale = scale_of.get(int(cohort[i]), 1.0)
        for t in periods:
            k = int(t - cohort[i])
            if k >= 0:
                per_obs.append(cfg.effects[min(k, len(cfg.effects) - 1)] * scale)
    overall_att = float(np.mean(per_obs)) if per_obs else float("nan")

    cohort_att = {}
    for g in cfg.cohorts:
        mask = cohort == g
        if not mask.any():
            continue
        scale = scale_of[g]
        vals = [
            cfg.effects[min(int(t - g), len(cfg.effects) - 1)] * scale
            for t in periods
            if t >= g
        ]
        cohort_att[int(g)] = float(np.mean(vals)) if vals else float("nan")

    return panel, GroundTruth(
        event_study=event_study,
        overall_att=overall_att,
        group_time=group_time,
        cohort_att=cohort_att,
    )
