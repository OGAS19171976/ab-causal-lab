"""M1 的两种合成实验场景：比值指标、聚类随机化。

与 ``generator.py`` 同一个思路 —— **必须知道 ground truth 才能验证方法**，
但这两个场景的 DGP 和连续指标实验不同，所以单独放一个模块。

两个场景都复用项目的确定性哈希分流（比值场景按用户、聚类场景按簇），
于是"重新随机化"就是换一个 salt 再哈希一遍，和 M0 的仿真台保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assignment import ExperimentSpec, Randomizer, Variant

__all__ = [
    "RatioScenarioConfig",
    "RatioSample",
    "ClusterScenarioConfig",
    "ClusterSample",
    "generate_ratio_scenario",
    "generate_cluster_scenario",
    "two_arm",
]


def two_arm(name: str, salt: str) -> ExperimentSpec:
    """本模块内部用的标准两臂 1:1 定义。"""
    return ExperimentSpec(
        name=name,
        variants=(Variant("control", 0.5), Variant("treatment", 0.5)),
        salt=salt,
    )


# --------------------------------------------------------------------------- #
# 场景一：比值指标（CTR = 点击 / 曝光）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RatioScenarioConfig:
    """比值指标实验的超参数。"""

    n_users: int = 20_000
    seed: int = 20260101
    #: 曝光量用负二项分布生成 —— 重尾是比值指标的本质特征：
    #: 大量用户只曝光一两次（其 r_i ∈ {0,1}），少数用户曝光成百上千次
    views_n: float = 1.0
    views_p: float = 0.08
    #: 基础点击率
    click_prob: float = 0.12
    #: 用户级点击率异质性（logit 尺度上的标准差）
    click_logit_sd: float = 0.6
    #: 潜在活跃度与曝光量的相关强度。
    #: **它才是两种口径分道扬镳的驱动因素**：高点击率的用户同时更高频，
    #: 合并比值就被高频用户"加权"到更高的水平，与人均比值拉开系统性差距。
    #: 实测 engagement_corr=0 时差距约 0，=0.3 时约 13%，=0.6 时约 25%。
    engagement_corr: float = 0.3

    def __post_init__(self) -> None:
        if self.n_users < 100:
            raise ValueError("n_users 太小")
        if not 0 < self.click_prob < 1:
            raise ValueError("click_prob 必须在 (0,1)")
        if not 0 < self.views_p <= 1:
            raise ValueError("views_p 必须在 (0,1]")
        if self.click_logit_sd < 0:
            raise ValueError("click_logit_sd 不能为负")

    @property
    def base_views_mean(self) -> float:
        return self.views_n * (1.0 - self.views_p) / self.views_p


@dataclass(frozen=True)
class RatioSample:
    """一次比值指标实验的观测。"""

    user_id: np.ndarray
    treated: np.ndarray  # (n,) bool
    views: np.ndarray  # (n,) 分母
    clicks: np.ndarray  # (n,) 分子
    base_click_prob: float
    relative_lift: float

    def __len__(self) -> int:
        return int(self.views.size)

    @property
    def true_effect(self) -> float:
        """真实效应。

        在 ``relative_lift=0`` 的 A/A 场景下两个口径的真效应都是 0，
        所以校准检验用 0 作参照是准确的。
        """
        return self.base_click_prob * self.relative_lift

    def pooled_ratio(self, treated: bool) -> float:
        """业务口径的合并比值 Σy/Σx。"""
        m = self.treated if treated else ~self.treated
        return float(self.clicks[m].sum() / self.views[m].sum())

    def mean_unit_ratio(self, treated: bool) -> float:
        """naive 做法实际估计的人均比值 mean(y_i/x_i)（忽略分母为 0 的单元）。"""
        m = (self.treated if treated else ~self.treated) & (self.views != 0)
        return float((self.clicks[m] / self.views[m]).mean())

    def estimand_gap(self, treated: bool = False) -> float:
        """人均比值 − 合并比值。非零就说明两种做法在回答不同的问题。"""
        return self.mean_unit_ratio(treated) - self.pooled_ratio(treated)


def generate_ratio_scenario(
    config: RatioScenarioConfig | None = None,
    *,
    relative_lift: float = 0.0,
    salt: str | None = None,
    seed: int | None = None,
) -> RatioSample:
    """生成一次比值指标实验。

    ``relative_lift`` 是**相对**提升（例如 0.02 表示点击率 +2%），
    因为这才是比值指标上业务真正关心的量。
    """
    cfg = config or RatioScenarioConfig()
    # 用户级 1:1 分流（用项目自己的哈希，可离线复算）
    spec = two_arm("ratio_exp", salt or "ratio_exp_v1")

    # ``seed`` 决定**整批样本**（用户异质性、曝光量）与点击噪声。
    # 传 ``seed`` 就是重新抽一批用户 —— 这是 delta method 所声称的
    # "超总体重复抽样"框架；不传则复用 ``config.seed`` 那一批固定用户。
    if seed is None:
        pop_rng = np.random.default_rng(cfg.seed)
        noise_rng = np.random.default_rng(cfg.seed)
    else:
        pop_rng = np.random.default_rng(seed)
        noise_rng = np.random.default_rng(seed + 1)

    user_id = np.array([f"u{i:07d}" for i in range(cfg.n_users)], dtype=object)

    # z 是用户的潜在活跃度：既影响曝光量，也影响点击率（两者正相关）
    z = pop_rng.normal(0.0, 1.0, cfg.n_users)
    lam = cfg.base_views_mean * np.exp(cfg.engagement_corr * z)
    views = pop_rng.poisson(lam) + 1

    logit_base = np.log(cfg.click_prob / (1.0 - cfg.click_prob))
    logit_p = logit_base + cfg.click_logit_sd * z
    p_user = 1.0 / (1.0 + np.exp(-logit_p))

    rz = Randomizer()
    codes = rz.assign_codes(list(user_id), spec)
    treated = codes == 1

    prob = np.where(treated, p_user * (1.0 + relative_lift), p_user)
    # 概率要夹到 [0,1]，relative_lift 较大时 p_user*(1+lift) 可能越界
    prob = np.clip(prob, 0.0, 1.0)
    clicks = noise_rng.binomial(views, prob)

    return RatioSample(
        user_id=user_id,
        treated=treated,
        views=views.astype(float),
        clicks=clicks.astype(float),
        base_click_prob=cfg.click_prob,
        relative_lift=relative_lift,
    )


# --------------------------------------------------------------------------- #
# 场景二：聚类随机化
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterScenarioConfig:
    """聚类随机化实验的超参数（城市/门店级分流）。"""

    n_clusters: int = 200
    users_per_cluster: int = 100
    #: 簇级随机效应的标准差 —— 它决定 ICC 有多大
    cluster_sd: float = 8.0
    user_sd: float = 10.0
    baseline: float = 50.0
    seed: int = 20260101
    #: 簇大小是否不等（1.0 表示完全相等）
    size_cv: float = 0.0

    def __post_init__(self) -> None:
        if self.n_clusters < 8:
            raise ValueError("簇数太少，至少 8 个")
        if self.users_per_cluster < 2:
            raise ValueError("每簇至少 2 个用户")
        if self.cluster_sd < 0 or self.user_sd <= 0:
            raise ValueError("标准差设置非法")
        if self.size_cv < 0:
            raise ValueError("size_cv 不能为负")


@dataclass(frozen=True)
class ClusterSample:
    """一次聚类随机化实验的观测。"""

    cluster_id: np.ndarray
    user_id: np.ndarray
    cluster_treated: np.ndarray  # (G,) bool
    treated: np.ndarray  # (n,) bool，由簇决定
    outcome: np.ndarray  # (n,)
    true_lift: float

    def __len__(self) -> int:
        return int(self.outcome.size)

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_treated.size)


def _assign_clusters(cluster_ids: list[str], salt: str) -> np.ndarray:
    """按**簇**做确定性 1:1 分流。

    直接复用项目的 Randomizer —— 只是把 unit_id 换成 cluster_id，
    于是"重新随机化"仍然是换一个 salt 再哈希一遍，与 M0 完全一致。
    """
    codes = Randomizer().assign_codes(cluster_ids, two_arm("cluster_exp", salt))
    if (codes < 0).any():
        raise RuntimeError("簇分流出现了未命中，检查 salt 与权重设置")
    return codes == 1


def generate_cluster_scenario(
    config: ClusterScenarioConfig | None = None,
    *,
    true_lift: float = 0.0,
    salt: str = "cluster_exp_v1",
    seed: int | None = None,
) -> ClusterSample:
    """生成一次聚类随机化实验（处理在簇级别分配，用户嵌在簇内）。"""
    cfg = config or ClusterScenarioConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    cluster_ids = [f"c{i:05d}" for i in range(cfg.n_clusters)]

    if cfg.size_cv > 0:
        sizes = np.maximum(
            (rng.lognormal(0.0, cfg.size_cv, cfg.n_clusters) * cfg.users_per_cluster).astype(int), 1
        )
    else:
        sizes = np.full(cfg.n_clusters, cfg.users_per_cluster)

    cluster_treated = _assign_clusters(cluster_ids, salt)

    # 簇级随机效应：同一座城市的用户共享它 —— 这正是组内相关的来源
    cluster_effect = rng.normal(0.0, cfg.cluster_sd, cfg.n_clusters)
    user_effect = rng.normal(0.0, cfg.user_sd, int(sizes.sum()))

    cid = np.repeat(np.arange(cfg.n_clusters), sizes)
    outcome = (
        cfg.baseline
        + cluster_effect[cid]
        + user_effect
        + true_lift * cluster_treated[cid]
    )

    return ClusterSample(
        cluster_id=np.repeat(np.array(cluster_ids, dtype=object), sizes),
        user_id=np.array([f"u{i:07d}" for i in range(int(sizes.sum()))], dtype=object),
        cluster_treated=cluster_treated,
        treated=cluster_treated[cid],
        outcome=outcome.astype(float),
        true_lift=true_lift,
    )
