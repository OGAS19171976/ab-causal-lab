"""合成控制（Synthetic Control）与安慰剂推断。

核心想法
--------
一个处置单元（一座城市、一个国家）没有好的单一对照。那就用**捐赠池的加权组合**
造一个"合成对照"，让它在**处置前**尽可能复刻处置单元的轨迹；
处置后的差距就是效应估计。

    w* = argmin ||X1 - X0 w||²   s.t.  w >= 0,  Σw = 1

权重非负且和为 1，是为了让合成对照仍是"一个单元的平均"，可以解释、不会外推。

推断：没有标准误可用
--------------------
只有一个处置单元，渐近理论无从谈起。标准做法是**空间安慰剂检验**：
把每个捐赠单元轮流当成"处置单元"跑一遍同样的流程，看真实处置单元的
处置后/处置前 RMSE 比值在安慰剂分布里排第几。那个排名比例就是"p 值"。

这不是常规意义上的 p 值 —— 它衡量的是"这个效应相对于捐赠池里随机一个单元
自己造出来的差距，算不算异常大"。本模块把它明确命名并在报告里标注。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

__all__ = [
    "SCMConfig",
    "SCMData",
    "SCMResult",
    "PlaceboResult",
    "generate_scm_scenario",
    "synthetic_control",
    "placebo_inference",
    "TimePlaceboResult",
    "LeaveOneOutResult",
    "time_placebo",
    "leave_one_out",
]


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SCMConfig:
    """合成控制场景的超参数（因子结构让捐赠池有机会复刻处置单元）。"""

    n_units: int = 40
    n_pre: int = 20
    n_post: int = 10
    n_factors: int = 2
    factor_sd: float = 1.0
    loading_sd: float = 0.6
    noise_sd: float = 0.4
    #: 处置后的真实效应（逐期）
    effect: float = 3.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_units < 5:
            raise ValueError("捐赠池太小，至少 5 个单元")
        if self.n_pre < 5 or self.n_post < 1:
            raise ValueError("处置前至少 5 期、处置后至少 1 期")


@dataclass(frozen=True)
class SCMData:
    """一次合成控制实验的观测。"""

    outcome: np.ndarray  # (n_units, n_pre + n_post)
    n_pre: int
    true_effect: float
    treated_index: int = 0

    @property
    def n_units(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_post(self) -> int:
        return int(self.outcome.shape[1] - self.n_pre)

    @property
    def pre(self) -> np.ndarray:
        return self.outcome[:, : self.n_pre]

    @property
    def post(self) -> np.ndarray:
        return self.outcome[:, self.n_pre :]


def generate_scm_scenario(
    config: SCMConfig | None = None,
    *,
    effect: float | None = None,
    seed: int | None = None,
    **overrides,
) -> SCMData:
    """生成带因子结构的面板：处置单元是捐赠单元的（近似）凸组合。"""
    cfg = config or SCMConfig(**overrides)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    effect = cfg.effect if effect is None else effect

    T = cfg.n_pre + cfg.n_post
    factors = rng.normal(0.0, cfg.factor_sd, (cfg.n_factors, T))
    loadings = rng.normal(0.0, cfg.loading_sd, (cfg.n_units, cfg.n_factors))
    level = rng.normal(0.0, 1.0, cfg.n_units)

    outcome = level[:, None] + loadings @ factors + rng.normal(0.0, cfg.noise_sd, (cfg.n_units, T))
    outcome[0, cfg.n_pre :] += effect

    return SCMData(outcome=outcome, n_pre=cfg.n_pre, true_effect=float(effect), treated_index=0)


# --------------------------------------------------------------------------- #
# 估计
# --------------------------------------------------------------------------- #
def _fit_weights(target_pre: np.ndarray, donors_pre: np.ndarray) -> np.ndarray:
    """求解非负、和为 1 的权重（最小化处置前的平方误差）。"""
    n_donors = donors_pre.shape[0]
    if n_donors == 0:
        raise ValueError("捐赠池为空")

    def loss(w: np.ndarray) -> float:
        resid = target_pre - w @ donors_pre
        return float(resid @ resid)

    def grad(w: np.ndarray) -> np.ndarray:
        resid = target_pre - w @ donors_pre
        return -2.0 * donors_pre @ resid

    w0 = np.full(n_donors, 1.0 / n_donors)
    res = optimize.minimize(
        loss,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


@dataclass
class SCMResult:
    """一次合成控制拟合。"""

    weights: np.ndarray
    synthetic_pre: np.ndarray
    synthetic_post: np.ndarray
    treated_pre: np.ndarray
    treated_post: np.ndarray

    @property
    def pre_rmse(self) -> float:
        return float(np.sqrt(np.mean((self.treated_pre - self.synthetic_pre) ** 2)))

    @property
    def post_rmse(self) -> float:
        return float(np.sqrt(np.mean((self.treated_post - self.synthetic_post) ** 2)))

    @property
    def gap(self) -> np.ndarray:
        """处置后的逐期差距 —— 效应估计。"""
        return self.treated_post - self.synthetic_post

    @property
    def att(self) -> float:
        return float(self.gap.mean())

    @property
    def rmse_ratio(self) -> float:
        """处置后 RMSE / 处置前 RMSE —— 安慰剂检验用的统计量。"""
        return self.post_rmse / self.pre_rmse if self.pre_rmse > 0 else float("inf")

    @property
    def effective_donors(self) -> int:
        return int(np.sum(self.weights > 1e-4))

    def summary(self) -> str:
        return (
            f"合成控制：ATT = {self.att:+.4f}\n"
            f"  处置前 RMSE = {self.pre_rmse:.4f}   处置后 RMSE = {self.post_rmse:.4f}   "
            f"比值 = {self.rmse_ratio:.3f}\n"
            f"  有效捐赠单元数 = {self.effective_donors} / {self.weights.size}"
        )


def synthetic_control(data: SCMData, *, treated_index: int | None = None) -> SCMResult:
    """对指定单元拟合合成对照，返回权重与差距路径。"""
    t = data.treated_index if treated_index is None else treated_index
    mask = np.ones(data.n_units, dtype=bool)
    mask[t] = False

    w = _fit_weights(data.pre[t], data.pre[mask])
    syn = w @ data.outcome[mask]

    return SCMResult(
        weights=w,
        synthetic_pre=syn[: data.n_pre],
        synthetic_post=syn[data.n_pre :],
        treated_pre=data.pre[t],
        treated_post=data.post[t],
    )


# --------------------------------------------------------------------------- #
# 安慰剂推断
# --------------------------------------------------------------------------- #
@dataclass
class PlaceboResult:
    """空间安慰剂检验的结果。

    ``p_value`` 的含义是"捐赠池里随机挑一个单元，它自己造出来的差距
    比真实处置单元还大（或一样大）的比例"。**它不是常规 p 值** ——
    没有渐近分布做后盾，只是排名的比例。
    """

    treated_ratio: float
    placebo_ratios: np.ndarray
    p_value: float
    treated_att: float
    placebo_atts: np.ndarray

    @property
    def n_placebos(self) -> int:
        return int(self.placebo_ratios.size)

    @property
    def rank(self) -> int:
        """真实单元在比值排序里的名次（1 = 最大）。"""
        return int(1 + np.sum(self.placebo_ratios > self.treated_ratio))

    def summary(self) -> str:
        q = np.percentile(self.placebo_ratios, [50, 90, 95])
        return (
            f"空间安慰剂检验（{self.n_placebos} 个安慰剂）\n"
            f"  真实单元的处置后/处置前 RMSE 比值 = {self.treated_ratio:.3f}"
            f"（排名 {self.rank}/{self.n_placebos + 1}）\n"
            f"  安慰剂比值分位数 50%/90%/95% = {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}\n"
            f"  排名 p 值 = {self.p_value:.4f}"
            f"   注意：这不是常规 p 值，只是排名的比例，没有渐近分布做后盾"
        )


def placebo_inference(data: SCMData, *, treated_index: int | None = None) -> PlaceboResult:
    """空间安慰剂检验：把每个捐赠单元轮流当作处置单元。"""
    t = data.treated_index if treated_index is None else treated_index
    main = synthetic_control(data, treated_index=t)

    ratios, atts = [], []
    for j in range(data.n_units):
        if j == t:
            continue
        try:
            r = synthetic_control(data, treated_index=j)
        except Exception:  # pragma: no cover - 拟合失败时跳过
            continue
        if not np.isfinite(r.rmse_ratio):
            continue
        ratios.append(r.rmse_ratio)
        atts.append(r.att)

    # 名字要分开：``ratios`` 是构造用的 list，``ratios_arr`` 才是要进结果的数组。
    # 第一版写成 ``ratios = np.asarray(ratios)`` —— 变量先被推断成 ``list[float]``，
    # 后面那行就变成"给 list 赋一个 ndarray"，类型检查器说的没错，是代码在骗人。
    ratios_arr = np.asarray(ratios)
    # 排名 p 值：真实单元的比值在（真实 + 安慰剂）里的排名比例
    p = float((1 + np.sum(ratios_arr >= main.rmse_ratio)) / (1 + ratios_arr.size))

    return PlaceboResult(
        treated_ratio=main.rmse_ratio,
        placebo_ratios=ratios_arr,
        p_value=p,
        treated_att=main.att,
        placebo_atts=np.asarray(atts),
    )


# --------------------------------------------------------------------------- #
# 时间安慰剂与留一法：空间安慰剂之外的两种"自打脸"检查
# --------------------------------------------------------------------------- #
@dataclass
class TimePlaceboResult:
    """时间安慰剂：把"处置"往前挪到处置前窗口里，看它会不会也造出一个效应。

    做法（Abadie 的 in-time placebo）：只用 ``[0, k)`` 这段拟合权重，
    然后在 ``[k, n_pre)`` 这段**什么都没发生**的窗口上算差距。
    如果这里也能"检出"一个可观的效果，说明这条差距路径本来就不可信 ——
    它可能是过拟合（捐赠池被拟合得太紧）或者结构性漂移，而不是处置造成的。

    与空间安慰剂的分工：空间安慰剂问"**别的单元**会不会也这样"，
    时间安慰剂问"**别的时段**会不会也这样"。两种都过，结论才站得住。
    """

    n_pre_used: int
    placebo_att: float
    placebo_rmse_ratio: float
    #: 同一个"处置前窗口"里，真实拟合的 RMSE（用来给 placebo_att 定标尺）
    pre_rmse_scale: float

    @property
    def scaled_placebo(self) -> float:
        """把安慰剂效应按处置前拟合误差标准化 —— 它才是"大不大"的读数。"""
        return abs(self.placebo_att) / self.pre_rmse_scale if self.pre_rmse_scale > 0 else float("nan")

    @property
    def clean(self) -> bool:
        """安慰剂效应不超过处置前拟合误差的 2 倍。"""
        return bool(self.scaled_placebo <= 2.0)

    def summary(self) -> str:
        return (
            f"时间安慰剂（前 {self.n_pre_used} 期拟合，后 {self.n_pre_used} 期当安慰剂窗口）\n"
            f"  安慰剂窗口的 ATT = {self.placebo_att:+.4f}\n"
            f"  处置前拟合 RMSE = {self.pre_rmse_scale:.4f} → 标准化后"
            f" {self.scaled_placebo:.3f}（≤2 视为干净：{self.clean}）\n"
            f"  安慰剂窗口内的处置后/处置前 RMSE 比值 = {self.placebo_rmse_ratio:.3f}"
        )


def time_placebo(
    data: SCMData, *, treated_index: int | None = None, split: float = 0.5
) -> TimePlaceboResult:
    """见 ``TimePlaceboResult``。``split`` 是"假想处置时点"在处置前窗口里的位置。"""
    t_index = data.treated_index if treated_index is None else treated_index
    if not 0.2 <= split <= 0.8:
        raise ValueError("split 必须落在 [0.2, 0.8]，否则两侧样本太少")
    k = int(round(data.n_pre * split))
    if k < 3 or data.n_pre - k < 2:
        raise ValueError(
            f"处置前只有 {data.n_pre} 期，拆成 {k}/{data.n_pre - k} 之后不够拟合与检验"
        )

    mask = np.ones(data.n_units, dtype=bool)
    mask[t_index] = False
    donors = data.outcome[mask]

    # 只用前半段拟合权重
    w = _fit_weights(data.outcome[t_index, :k], donors[:, :k])
    synthetic = w @ donors

    # 安慰剂窗口 = [k, n_pre)：这一段真实效应为 0
    placebo_gap = data.outcome[t_index, k : data.n_pre] - synthetic[k : data.n_pre]
    placebo_att = float(np.mean(placebo_gap))
    fit_resid = data.outcome[t_index, :k] - synthetic[:k]
    pre_rmse = float(np.sqrt(np.mean(fit_resid**2)))
    placebo_rmse = (
        float(np.sqrt(np.mean(placebo_gap**2))) if placebo_gap.size else float("nan")
    )
    ratio = placebo_rmse / pre_rmse if pre_rmse > 0 else float("nan")

    return TimePlaceboResult(
        n_pre_used=k,
        placebo_att=placebo_att,
        placebo_rmse_ratio=float(ratio),
        pre_rmse_scale=pre_rmse,
    )


@dataclass
class LeaveOneOutResult:
    """留一法：逐个丢掉一个捐赠单元，看估计的处置效应稳不稳。

    这条检查针对的是"整个结论靠某一个捐赠单元撑着"的情形 ——
    在合成控制里很常见（权重集中在一两个单元上时尤其如此）。
    """

    att_full: float
    att_dropped: np.ndarray
    weights_full: np.ndarray

    @property
    def att_min(self) -> float:
        return float(np.min(self.att_dropped))

    @property
    def att_max(self) -> float:
        return float(np.max(self.att_dropped))

    @property
    def att_sd(self) -> float:
        return float(np.std(self.att_dropped, ddof=1)) if self.att_dropped.size > 1 else 0.0

    @property
    def most_influential(self) -> int:
        """丢掉之后 ATT 变得最多的那个捐赠单元的下标（捐赠池内）。"""
        return int(np.argmax(np.abs(self.att_dropped - self.att_full)))

    @property
    def sign_is_stable(self) -> bool:
        """所有留一估计的符号都与全样本一致。"""
        return bool(
            np.all(self.att_dropped > 0) if self.att_full > 0 else np.all(self.att_dropped < 0)
        )

    @property
    def max_shift_share(self) -> float:
        """最坏的那次删掉，ATT 相对全样本移动了多少比例。"""
        if self.att_full == 0:
            return float("nan")
        return float(
            np.max(np.abs(self.att_dropped - self.att_full)) / abs(self.att_full)
        )

    def summary(self) -> str:
        return (
            f"留一法（逐个丢掉 {self.att_dropped.size} 个捐赠单元）\n"
            f"  全样本 ATT = {self.att_full:+.4f}\n"
            f"  留一 ATT 范围 [{self.att_min:+.4f}, {self.att_max:+.4f}]，"
            f"sd {self.att_sd:.4f}\n"
            f"  最坏一次移动 {self.max_shift_share:.1%}；符号是否稳定：{self.sign_is_stable}"
        )


def leave_one_out(data: SCMData, *, treated_index: int | None = None) -> LeaveOneOutResult:
    """见 ``LeaveOneOutResult``：逐个丢掉捐赠单元重估。"""
    t_index = data.treated_index if treated_index is None else treated_index
    main = synthetic_control(data, treated_index=t_index)
    donor_indices = [j for j in range(data.n_units) if j != t_index]

    atts: list[float] = []
    for drop in donor_indices:
        keep = [j for j in donor_indices if j != drop]
        sub = SCMData(
            outcome=data.outcome[[t_index, *keep]],
            n_pre=data.n_pre,
            true_effect=data.true_effect,
            treated_index=0,
        )
        atts.append(float(synthetic_control(sub, treated_index=0).att))

    return LeaveOneOutResult(
        att_full=float(main.att),
        att_dropped=np.asarray(atts),
        weights_full=main.weights,
    )
