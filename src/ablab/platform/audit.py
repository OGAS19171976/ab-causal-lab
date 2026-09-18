"""平台层审计：这条"注册表 → 分析 → 接口"的链路本身可信吗？

为什么平台层还要再验一遍
------------------------
M0–M4 验的是**方法**（分流、CUPED、序贯、因果、HTE），平台层不引入新方法。
但它引入了三种新的出错方式：

  1. 编排顺序错了（先看指标再看 SRM、把 naive 口径丢掉）
  2. 随机流分离错了（"同一个实验重启后数字全变"、协变量流与结果流串了）
  3. 校验逻辑长出第二份实现（注册表和 ``ExperimentSpec`` 各校一遍，迟早不一致）

所以这里要验的主要是**工程不变量**。但有一个例外必须在平台层测：
**整条管道的 A/A 校准**。因为平台把"一次实验"变成了"点一下按钮"，
而按钮背后比 M0 的验证台多用了一路随机数（合成协变量、结果噪声、查看顺序各一路）。
流一旦串了，零效应实验就会以远超 5% 的概率报显著 —— 这正是我们真的怀疑过的事。

一条纪律：审计必须调用**真实的 ``analyse_experiment``**。
重新实现一个"简化版分析"再证明它校准，等于什么都没证明。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..hashing import murmur3_32
from .analysis import PLATFORM_POPULATION, analyse_experiment, analyse_experiment_from_warehouse
from .registry import ExperimentRecord

__all__ = [
    "PlatformAAResult",
    "StreamIndependenceResult",
    "NoiseComponentResult",
    "DemoDecomposition",
    "SourceEquivalence",
    "MonitoringAuditResult",
    "run_platform_aa_audit",
    "run_stream_independence_audit",
    "run_noise_component_audit",
    "run_demo_decomposition",
    "run_source_equivalence_audit",
    "run_monitoring_fwer_audit",
    "UnitAwarenessResult",
    "run_unit_awareness_audit",
]


def _record(
    name: str,
    salt: str,
    *,
    weights: tuple[float, ...] = (0.5, 0.5),
    traffic_ratio: float = 1.0,
    true_lift: float = 0.0,
    metric: str = "metric",
) -> ExperimentRecord:
    """直接构造一条记录（不经过注册表）——审计要的是分析管道，不是存储层。"""
    names = ["control", "treatment"] if len(weights) == 2 else [f"arm{i}" for i in range(len(weights))]
    return ExperimentRecord(
        name=name,
        salt=salt,
        variants=[{"name": n, "weight": w} for n, w in zip(names, weights)],
        traffic_ratio=traffic_ratio,
        true_lift=true_lift,
        primary_metric=metric,
    )


def _wilson(hits: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """比例的 Wilson 区间（比正态近似稳，尤其是比例接近 0 时）。"""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# 1. 整条管道的 A/A 校准
# --------------------------------------------------------------------------- #
@dataclass
class PlatformAAResult:
    """换 salt 重复做零效应实验，看整条管道会不会超发。"""

    n_salts: int
    n_units: int
    alpha: float
    traffic_ratio: float
    naive_fpr: float
    cuped_fpr: float
    naive_fpr_interval: tuple[float, float]
    cuped_fpr_interval: tuple[float, float]
    naive_z_mean: float
    naive_z_sd: float
    cuped_z_mean: float
    cuped_z_sd: float
    coverage: float
    coverage_interval: tuple[float, float]
    #: 逐次运行的 z，供画图/复核
    naive_z: list[float] = field(default_factory=list)
    cuped_z: list[float] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        """naive 与 CUPED 的 FPR 区间都要盖住 alpha，且 z 的均值≈0、sd≈1。"""
        return (
            self.naive_fpr_interval[0] <= self.alpha <= self.naive_fpr_interval[1]
            and self.cuped_fpr_interval[0] <= self.alpha <= self.cuped_fpr_interval[1]
            and abs(self.naive_z_sd - 1.0) < 0.15
            and abs(self.cuped_z_sd - 1.0) < 0.15
        )


def run_platform_aa_audit(
    *,
    n_salts: int = 400,
    n_units: int = 20_000,
    alpha: float = 0.05,
    traffic_ratio: float = 1.0,
    metric: str = "interaction_per_user_14d",
    progress: Any = None,
) -> PlatformAAResult:
    """跑 ``n_salts`` 次零效应实验，每次是一个**新 salt**（= 真正意义上的重复实验）。

    每一次都完整走 ``analyse_experiment``：合成人群 → 哈希分流 → 生成结果 →
    SRM → CUPED + naive → 序贯监控。所以任何一环的流分离错了都会在这里露出来。
    """
    naive_hits = cuped_hits = coverage = 0
    nz: list[float] = []
    cz: list[float] = []

    for i in range(n_salts):
        rec = _record(
            f"aa_{i}", f"aa_audit_{i}", traffic_ratio=traffic_ratio, metric=metric
        )
        rep = analyse_experiment(rec, n_users=n_units, alpha=alpha)
        nz.append(rep.naive.absolute_effect / rep.naive.std_error)
        cz.append(rep.cuped.absolute_effect / rep.cuped.std_error)
        naive_hits += int(rep.naive.significant)
        cuped_hits += int(rep.cuped.significant)
        # 真效应为 0，所以"覆盖"就是区间盖住 0
        coverage += int(rep.cuped.ci_low <= 0.0 <= rep.cuped.ci_high)
        if progress is not None and (i + 1) % max(1, n_salts // 10) == 0:
            progress(i + 1, n_salts)

    naive_arr = np.array(nz)
    cuped_arr = np.array(cz)
    return PlatformAAResult(
        n_salts=n_salts,
        n_units=n_units,
        alpha=alpha,
        traffic_ratio=traffic_ratio,
        naive_fpr=naive_hits / n_salts,
        cuped_fpr=cuped_hits / n_salts,
        naive_fpr_interval=_wilson(naive_hits, n_salts),
        cuped_fpr_interval=_wilson(cuped_hits, n_salts),
        naive_z_mean=float(naive_arr.mean()),
        naive_z_sd=float(naive_arr.std(ddof=1)),
        cuped_z_mean=float(cuped_arr.mean()),
        cuped_z_sd=float(cuped_arr.std(ddof=1)),
        coverage=coverage / n_salts,
        coverage_interval=_wilson(coverage, n_salts),
        naive_z=nz,
        cuped_z=cz,
    )


# --------------------------------------------------------------------------- #
# 2. 随机流分离：相邻整数种子到底独立不独立
# --------------------------------------------------------------------------- #
@dataclass
class StreamIndependenceResult:
    """两条随机流（协变量、结果噪声）在同一分组掩码下的相关系数。"""

    n_reps: int
    adjacent_corr: float
    far_corr: float
    spawn_corr: float
    adjacent_sd_pre: float
    adjacent_sd_eps: float
    tolerance: float

    @property
    def independent(self) -> bool:
        """三种种子构造方式的相关都应在抽样误差内为 0。"""
        return all(
            abs(c) <= self.tolerance
            for c in (self.adjacent_corr, self.far_corr, self.spawn_corr)
        )


def run_stream_independence_audit(
    *,
    n_reps: int = 2_000,
    n_units: int = 20_000,
    salt: str = "exp_rec_emb_v1",
    traffic_ratio: float = 0.6,
) -> StreamIndependenceResult:
    """``analyse_experiment`` 用 ``default_rng(S)`` 抽协变量、``default_rng(S+1)`` 抽结果噪声。

    两个流必须独立。若相邻整数种子在 SeedSequence 下混合不充分，
    协变量噪声与结果噪声就会相关 —— 零效应实验会超发。
    这里把"相邻"单独拎出来测：只改相邻性，其它全一样。
    """
    from ..assignment import ExperimentSpec, Randomizer, Variant
    from ..hashing import KeyBatcher

    spec = ExperimentSpec(
        name="stream_audit",
        variants=(Variant("control", 0.5), Variant("treatment", 0.5)),
        salt=salt,
        traffic_ratio=traffic_ratio,
    )
    ids = [f"u{i:07d}" for i in range(n_units)]
    codes = Randomizer().assign_codes(ids, spec, KeyBatcher(ids))
    treated, control = codes == 1, codes == 0
    w = np.where(treated, 1.0 / treated.sum(), np.where(control, -1.0 / control.sum(), 0.0))
    norm = float(np.linalg.norm(w))

    cfg = PLATFORM_POPULATION
    eps_sd = cfg.post_sd * float(np.sqrt(1.0 - cfg.corr_pre_post**2))
    base = murmur3_32(salt.encode("utf-8"))

    def _pair_z(s_pre, s_eps) -> tuple[float, float]:
        pre = np.random.default_rng(s_pre).normal(cfg.pre_mean, cfg.pre_sd, n_units)
        eps = np.random.default_rng(s_eps).normal(0.0, eps_sd, n_units)
        return (
            float(w @ pre) / float(pre.std(ddof=1) * norm),
            float(w @ eps) / float(eps.std(ddof=1) * norm),
        )

    def _run(pairs) -> tuple[float, np.ndarray, np.ndarray]:
        dx, de = [], []
        for s_pre, s_eps in pairs:
            a, b = _pair_z(s_pre, s_eps)
            dx.append(a)
            de.append(b)
        dx, de = np.array(dx), np.array(de)
        return float(np.corrcoef(dx, de)[0, 1]), dx, de

    adj_corr, adj_dx, adj_de = _run(
        [(base + 7 * k, base + 7 * k + 1) for k in range(n_reps)]
    )
    far_corr, _, _ = _run(
        [(base + 7 * k, base + 7 * k + 10**6) for k in range(n_reps)]
    )
    # 理想情形：SeedSequence.spawn 出的独立子流
    root = np.random.SeedSequence(20260101)
    spawn_pairs = []
    for child in root.spawn(n_reps):
        a, b = child.spawn(2)
        spawn_pairs.append((a, b))
    spawn_corr, _, _ = _run(spawn_pairs)

    return StreamIndependenceResult(
        n_reps=n_reps,
        adjacent_corr=adj_corr,
        far_corr=far_corr,
        spawn_corr=spawn_corr,
        adjacent_sd_pre=float(adj_dx.std(ddof=1)),
        adjacent_sd_eps=float(adj_de.std(ddof=1)),
        tolerance=3.0 / float(np.sqrt(n_reps)),
    )


# --------------------------------------------------------------------------- #
# 3. 两个成分在**不同 salt 之间**的分布
# --------------------------------------------------------------------------- #
@dataclass
class NoiseComponentResult:
    """跨 salt 的 ΔX̄ 与 Δeps 的标准化分布。

    这是解释"演示实验看起来离谱"的唯一硬证据：两个成分各自是不是标准正态。

    注意 ``run_stream_independence_audit`` 回答的是另一个问题 ——
    它固定一个 salt，看两条流**在同一个掩码下**相不相关。
    这里换的问题是：跨 salt（掩码和噪声同时变）时，这两个成分的分布对不对。
    两个问题都问完了，才谈得上"演示那条负对照到底是不是巧合"。
    """

    n_salts: int
    n_units: int
    d_pre_z_mean: float
    d_pre_z_sd: float
    d_noise_z_mean: float
    d_noise_z_sd: float
    d_pre_max_abs: float
    d_noise_max_abs: float
    d_pre_over3: int
    d_noise_over3: int
    d_pre_z: list[float] = field(default_factory=list)
    d_noise_z: list[float] = field(default_factory=list)

    def percentile_of(self, z: float) -> float:
        """某个观测 z 在跨 salt 分布里的单侧经验分位（用噪声成分）。"""
        arr = np.asarray(self.d_noise_z)
        return float((np.abs(arr) <= abs(z)).mean())

    @property
    def looks_standard_normal(self) -> bool:
        return (
            abs(self.d_pre_z_mean) < 4 / np.sqrt(self.n_salts)
            and abs(self.d_noise_z_mean) < 4 / np.sqrt(self.n_salts)
            and abs(self.d_pre_z_sd - 1.0) < 0.15
            and abs(self.d_noise_z_sd - 1.0) < 0.15
        )


def run_noise_component_audit(
    *,
    n_salts: int = 2_000,
    n_units: int = 20_000,
    traffic_ratio: float = 0.6,
    weights: tuple[float, float] = (0.5, 0.5),
    salt_prefix: str = "component_audit",
) -> NoiseComponentResult:
    """对每个 salt 分别算 ΔX̄ 与 Δeps 的标准化值，看分布。"""
    from ..assignment import ExperimentSpec, Randomizer, Variant
    from ..hashing import KeyBatcher

    cfg = PLATFORM_POPULATION
    eps_sd = cfg.post_sd * float(np.sqrt(1.0 - cfg.corr_pre_post**2))
    ids = [f"u{i:07d}" for i in range(n_units)]

    d_pre_z: list[float] = []
    d_noise_z: list[float] = []
    for i in range(n_salts):
        salt = f"{salt_prefix}_{i}"
        spec = ExperimentSpec(
            name=salt,
            variants=(Variant("control", weights[0]), Variant("treatment", weights[1])),
            salt=salt,
            traffic_ratio=traffic_ratio,
        )
        codes = Randomizer().assign_codes(ids, spec, KeyBatcher(ids))
        treated, control = codes == 1, codes == 0
        base_seed = murmur3_32(salt.encode("utf-8"))
        pre = np.random.default_rng(base_seed).normal(cfg.pre_mean, cfg.pre_sd, n_units)
        eps = np.random.default_rng(base_seed + 1).normal(0.0, eps_sd, n_units)
        scale = float(np.sqrt(1 / treated.sum() + 1 / control.sum()))
        d_pre_z.append(
            float(pre[treated].mean() - pre[control].mean()) / float(pre.std(ddof=1) * scale)
        )
        d_noise_z.append(
            float(eps[treated].mean() - eps[control].mean()) / float(eps.std(ddof=1) * scale)
        )

    a, b = np.array(d_pre_z), np.array(d_noise_z)
    return NoiseComponentResult(
        n_salts=n_salts,
        n_units=n_units,
        d_pre_z_mean=float(a.mean()),
        d_pre_z_sd=float(a.std(ddof=1)),
        d_noise_z_mean=float(b.mean()),
        d_noise_z_sd=float(b.std(ddof=1)),
        d_pre_max_abs=float(np.abs(a).max()),
        d_noise_max_abs=float(np.abs(b).max()),
        d_pre_over3=int((np.abs(a) > 3).sum()),
        d_noise_over3=int((np.abs(b) > 3).sum()),
        d_pre_z=d_pre_z,
        d_noise_z=d_noise_z,
    )


# --------------------------------------------------------------------------- #
# 4. 演示实验的效应分解
# --------------------------------------------------------------------------- #
@dataclass
class DemoDecomposition:
    """一条演示实验：观测到的效应到底由什么组成。"""

    name: str
    true_lift: float
    n_users: int
    naive_effect: float
    cuped_effect: float
    naive_z: float
    cuped_z: float
    naive_significant: bool
    cuped_significant: bool
    imbalance_component: float
    residual_component: float
    balance_statistic: float
    balance_p_value: float
    pre_difference: float
    pre_difference_se: float
    noise_difference: float
    noise_difference_se: float
    split_z: float  # ΔX 的 z
    noise_z: float  # Δeps 的 z
    health: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "true_lift": self.true_lift,
            "naive_effect": self.naive_effect,
            "cuped_effect": self.cuped_effect,
            "naive_z": self.naive_z,
            "cuped_z": self.cuped_z,
            "imbalance_component": self.imbalance_component,
            "residual_component": self.residual_component,
            "split_z": self.split_z,
            "noise_z": self.noise_z,
        }


def run_demo_decomposition(
    *,
    n_users: int = 20_000,
    demos: Any = None,
) -> list[DemoDecomposition]:
    """把每条演示实验的观测效应拆成"前置协变量失衡"+"结果噪声"。

    为什么要拆：负对照（``true_lift=0``）在默认种子下**确实显著**。
    说"框架错了"很容易，说清"它由什么组成"才有用。
    拆的依据是生成模型：

        Y = post_mean + beta*(X - pre_mean) + eps + tau*T
        naive  = beta*ΔX + Δeps + tau       <- 含协变量失衡
        CUPED ~= Δeps + tau                 <- θ̂ 收敛到 beta，把 beta*ΔX 扣掉

    所以 ``naive - CUPED`` 恰好就是失衡那一项，可以拿真实数据直接验证。
    """
    from ..assignment import ExperimentSpec, Randomizer, Variant
    from ..hashing import KeyBatcher
    from ..sim.generator import generate_population

    if demos is None:
        from .demo import DEMO_EXPERIMENTS

        demos = DEMO_EXPERIMENTS

    out: list[DemoDecomposition] = []
    for demo in demos:
        salt = demo["salt"]
        weights = tuple(v["weight"] for v in demo["variants"])
        traffic = demo.get("traffic_ratio", 1.0)
        true_lift = demo.get("true_lift", 0.0)

        rec = _record(
            demo["name"], salt, weights=weights, traffic_ratio=traffic,
            true_lift=true_lift, metric=demo.get("primary_metric", "metric"),
        )
        rep = analyse_experiment(rec, n_users=n_users)

        # 用同一条记录重新生成一次数据，把 β·ΔX 与 Δeps 分开量出来
        base_seed = murmur3_32(salt.encode("utf-8"))
        cfg = replace(PLATFORM_POPULATION, n_units=n_users, seed=base_seed)
        pop = generate_population(cfg)
        spec = ExperimentSpec(
            name=demo["name"],
            variants=tuple(Variant(v["name"], v["weight"]) for v in demo["variants"]),
            salt=salt,
            traffic_ratio=traffic,
        )
        ids = pop.ids()
        codes = Randomizer().assign_codes(ids, spec, KeyBatcher(ids))
        treated, control = codes == len(spec.variants) - 1, codes == 0

        eps_sd = cfg.post_sd * float(np.sqrt(1.0 - cfg.corr_pre_post**2))
        eps = np.random.default_rng(base_seed + 1).normal(0.0, eps_sd, pop.pre_metric.size)
        pre = pop.pre_metric

        scale = float(np.sqrt(1 / treated.sum() + 1 / control.sum()))
        d_pre = float(pre[treated].mean() - pre[control].mean())
        se_pre = float(pre.std(ddof=1) * scale)
        d_noise = float(eps[treated].mean() - eps[control].mean())
        se_noise = float(eps.std(ddof=1) * scale)

        balance = next(c for c in rep.checks if c.name == "协变量平衡")
        assert rep.naive is not None and rep.cuped is not None
        out.append(
            DemoDecomposition(
                name=demo["name"],
                true_lift=true_lift,
                n_users=n_users,
                naive_effect=rep.naive.absolute_effect,
                cuped_effect=rep.cuped.absolute_effect,
                naive_z=rep.naive.absolute_effect / rep.naive.std_error,
                cuped_z=rep.cuped.absolute_effect / rep.cuped.std_error,
                naive_significant=rep.naive.significant,
                cuped_significant=rep.cuped.significant,
                imbalance_component=float(rep.imbalance_component),
                residual_component=float(rep.residual_component),
                balance_statistic=float(balance.statistic),
                balance_p_value=float(balance.p_value),
                pre_difference=d_pre,
                pre_difference_se=se_pre,
                noise_difference=d_noise,
                noise_difference_se=se_noise,
                split_z=d_pre / se_pre,
                noise_z=d_noise / se_noise,
                health=rep.health,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 5. 数据源等价性：同一批数据，三条读取路径必须给同一个答案
# --------------------------------------------------------------------------- #
@dataclass
class SourceEquivalence:
    """一个实验在"三条路径"下的推断结果对比。

    M0 的交叉验证只有两条路径（DWS 汇总 vs DWD 明细）。
    平台接入数仓之后多出第三条：**平台自己的编排** ``analyse_data``。
    三条都要对得上 —— 任何一条不同，就说明有一层在用自己的口径重新实现了一遍。
    """

    experiment: str
    n_users: int
    n_looks: int
    #: (naive_effect, naive_se, cuped_effect, cuped_se) 三条路径各一组
    from_detail: tuple[float, float, float, float]
    from_ads: tuple[float, float, float, float]
    from_platform: tuple[float, float, float, float]
    #: 监控的最后一次查看是否等于主结论（跨数据源的同一条不变量）
    last_look_matches: bool
    #: 监控的信息比例（数仓路径下是实际累计，不一定是等距）
    information_fractions: list[float]

    @property
    def max_deviation(self) -> float:
        """三条路径之间的最大差异。"""
        groups = (self.from_detail, self.from_ads, self.from_platform)
        worst = 0.0
        for i in range(4):
            values = [g[i] for g in groups]
            worst = max(worst, max(values) - min(values))
        return float(worst)

    @property
    def agree(self) -> bool:
        return self.max_deviation < 1e-9 and self.last_look_matches


def run_source_equivalence_audit(con, experiment: str, *, n_looks: int = 5) -> SourceEquivalence:
    """把 ``experiment`` 用三条路径各算一遍，比较四个核心数。

    三条路径：
      1. **DWD 明细**：``welch_ttest`` / ``cuped_ttest`` 直接吃逐用户数组
      2. **ADS 汇总**：``ablab.warehouse.analyse_ads``（M0 写的独立实现）
      3. **平台编排**：``analyse_experiment_from_warehouse``（M5 写的，共用监控层）

    第 3 条与本审计同属 M5，所以它单独通过说明不了什么 —— 关键是
    1 和 2 是**在平台存在之前就写好的**，拿它们当参照才有意义。
    """
    from ..inference import cuped_ttest, welch_ttest
    from ..warehouse import analyse_ads
    from .datasource import build_warehouse_data

    detail = con.execute(
        """
        SELECT variant, pre_metric, post_metric
        FROM dwd_experiment_user WHERE experiment = ?
        """,
        [experiment],
    ).df()
    if detail.empty:
        raise ValueError(f"DWD 里找不到实验 {experiment!r}")
    t = detail[detail["variant"] == "treatment"]
    c = detail[detail["variant"] == "control"]
    if t.empty or c.empty:
        raise ValueError(f"实验 {experiment!r} 在 DWD 里缺少某一臂")

    naive_detail = welch_ttest(
        t["post_metric"].to_numpy(), c["post_metric"].to_numpy(), metric="post_metric_14d"
    )
    cuped_detail, _ = cuped_ttest(
        t["pre_metric"].to_numpy(),
        t["post_metric"].to_numpy(),
        c["pre_metric"].to_numpy(),
        c["post_metric"].to_numpy(),
        metric="post_metric_14d",
    )
    from_detail = (
        naive_detail.absolute_effect,
        naive_detail.std_error,
        cuped_detail.absolute_effect,
        cuped_detail.std_error,
    )

    reference = next(a for a in analyse_ads(con) if a.experiment == experiment)
    from_ads = (
        reference.naive.absolute_effect,
        reference.naive.std_error,
        reference.cuped.absolute_effect,
        reference.cuped.std_error,
    )

    record = _record(experiment, f"{experiment}_v1")
    record.warehouse_experiment = experiment
    record.primary_metric = "post_metric_14d"
    rep = analyse_experiment_from_warehouse(record, con, n_looks=n_looks)
    # 平台把"头条口径"和"对照口径"分开报，而参照实现是按 (post-only, CUPED) 排的。
    # 这里**按名字取**，不依赖声明顺序 —— 否则把 estimator 改成 post_only 就会静默错位。
    by_name = {rep.primary_estimator_name: rep.primary, rep.alt_estimator_name: rep.alt}
    post_est, cuped_est = by_name["post_only"], by_name["cuped"]
    from_platform = (
        post_est.absolute_effect,
        post_est.std_error,
        cuped_est.absolute_effect,
        cuped_est.std_error,
    )

    data = build_warehouse_data(con, experiment, n_looks=n_looks)
    last = rep.monitoring[-1]
    return SourceEquivalence(
        experiment=experiment,
        n_users=rep.n_users,
        n_looks=len(rep.monitoring),
        from_detail=from_detail,
        from_ads=from_ads,
        from_platform=from_platform,
        last_look_matches=(
            abs(last["effect"] - rep.primary.absolute_effect) < 1e-12
            and abs(last["std_error"] - rep.primary.std_error) < 1e-12
        ),
        information_fractions=[lk.information_fraction for lk in data.looks],
    )


# --------------------------------------------------------------------------- #
# 6. 监控口径与判定口径是否一致（M6.1）
# --------------------------------------------------------------------------- #
@dataclass
class MonitoringAuditResult:
    """序贯监控用哪个估计量，FWER 都还稳得住吗。

    M6.1 之前，平台的头条结论是 CUPED，而**监控曲线用的是 post-only 的 z**。
    那是两个不同的估计量，于是会出现"曲线越界了、结论卡说不显著"这类互相打架的报告。

    对齐之后的新问题必须靠仿真实证，不能靠推演：
    CUPED 的 z 配同一组 OBF 边界，FWER 到底还是不是 α？

    * 若 ≈ α：CUPED 就是纯粹的收益（同样边界下功效更高），可以直接用；
    * 若明显 > α：说明 z 的联合分布偏离了边界递归所依赖的典型结构，
      必须按 CUPED 的信息量重新解边界 —— 那就不能"白拿"这笔收益。
    """

    n_salts: int
    n_units: int
    n_looks: int
    alpha: float
    true_lift: float
    #: "至少越界一次"的比例 = 序贯 FWER（零效应时）
    post_only_fwer: float
    cuped_fwer: float
    #: 只看最后一次的显著率（固定样本口径），用来对照"序贯付出的代价"
    post_only_final_rate: float
    cuped_final_rate: float
    #: 两个口径的越界结论不一致的比例 —— 口径不一致的可见后果
    disagreement: float
    #: 末次查看 z 的标准差（应为 1）
    post_only_z_sd: float
    cuped_z_sd: float
    #: 末次查看 z 的均值（零效应下应为 0，有真实效应时应为正）
    post_only_z_mean: float
    cuped_z_mean: float
    #: 两个口径末次 z 的相关系数
    z_correlation: float
    fwer_interval: tuple[float, float]
    cuped_fwer_interval: tuple[float, float]
    #: 逐次运行的末次 z，供画图/复核
    post_only_z: list[float] = field(default_factory=list)
    cuped_z: list[float] = field(default_factory=list)

    @property
    def cuped_calibrated(self) -> bool:
        """CUPED 监控的 FWER 区间要盖住 alpha。"""
        return self.cuped_fwer_interval[0] <= self.alpha <= self.cuped_fwer_interval[1]

    @property
    def cuped_advantage(self) -> float:
        """CUPED 相对 post-only 的功效/检出率提升。"""
        if self.post_only_final_rate <= 0:
            return float("nan")
        return self.cuped_final_rate / self.post_only_final_rate - 1.0


def run_monitoring_fwer_audit(
    *,
    n_salts: int = 400,
    n_units: int = 20_000,
    n_looks: int = 5,
    alpha: float = 0.05,
    true_lift: float = 0.0,
) -> MonitoringAuditResult:
    """走**真实管道**（``analyse_experiment``）跑 n_salts 个新 salt，统计各口径的越界率。

    ``true_lift=0`` 时得到的是 FWER；给一个真实效应就得到功效。
    """
    post_fwer = cuped_fwer = 0
    post_final = cuped_final = 0
    disagree = 0
    post_z: list[float] = []
    cuped_z: list[float] = []

    for i in range(n_salts):
        rec = _record(f"mon_{i}", f"mon_audit_{i}", true_lift=true_lift)
        rep = analyse_experiment(rec, n_users=n_units, n_looks=n_looks, alpha=alpha)
        rows = rep.monitoring

        # 主口径 = 记录上声明的那个（这里是 cuped）；对照口径 = alt_*
        p_cross = any(m["crossed"] for m in rows)
        c_cross = any(
            m["alt_z"] is not None and abs(m["alt_z"]) >= m["boundary"] for m in rows
        )
        post_fwer += int(c_cross)
        cuped_fwer += int(p_cross)
        disagree += int(p_cross != c_cross)
        # 末次查看单独看 = 固定样本口径
        post_final += int(
            rows[-1]["alt_z"] is not None and abs(rows[-1]["alt_z"]) >= rows[-1]["boundary"]
        )
        cuped_final += int(rep.primary.significant)
        post_z.append(float(rows[-1]["alt_z"]))
        cuped_z.append(float(rows[-1]["z"]))

    p_arr, c_arr = np.array(post_z), np.array(cuped_z)
    return MonitoringAuditResult(
        n_salts=n_salts,
        n_units=n_units,
        n_looks=n_looks,
        alpha=alpha,
        true_lift=true_lift,
        post_only_fwer=post_fwer / n_salts,
        cuped_fwer=cuped_fwer / n_salts,
        post_only_final_rate=post_final / n_salts,
        cuped_final_rate=cuped_final / n_salts,
        disagreement=disagree / n_salts,
        post_only_z_sd=float(p_arr.std(ddof=1)),
        cuped_z_sd=float(c_arr.std(ddof=1)),
        post_only_z_mean=float(p_arr.mean()),
        cuped_z_mean=float(c_arr.mean()),
        z_correlation=float(np.corrcoef(p_arr, c_arr)[0, 1]),
        fwer_interval=_wilson(post_fwer, n_salts),
        cuped_fwer_interval=_wilson(cuped_fwer, n_salts),
        post_only_z=post_z,
        cuped_z=cuped_z,
    )


# --------------------------------------------------------------------------- #
# 7. 分析单元：随机化单元 ≠ 分析单元时会发生什么（M6.2）
# --------------------------------------------------------------------------- #
@dataclass
class UnitAwarenessResult:
    """整簇随机化下，"用错分析单元"的代价。

    M1 已经用验证台量过这件事（用户级 t 检验 FPR 64.5%，CR1 5.25%）。
    这里量的是**平台**：走真实管道 ``analyse_experiment``，
    看它声明的 ``analysis_unit`` 到底有没有把这件事接住。

    零效应下的"越界率"就是 I 类错误率；两个口径给出的是同一份数据的两个答案，
    差别只在"把谁当观测单位"。
    """

    n_salts: int
    n_users: int
    n_clusters: int
    n_looks: int
    alpha: float
    #: 单元级（错误做法）与簇级（正确做法）的固定样本显著率
    unit_level_fpr: float
    cluster_level_fpr: float
    #: 序贯口径下的越界率
    unit_level_sequential: float
    cluster_level_sequential: float
    #: 两者的 SE 之比 = 簇级 SE / 单元级 SE（> 1 表示单元级低估了）
    se_ratio: float
    unit_level_interval: tuple[float, float]
    cluster_level_interval: tuple[float, float]

    @property
    def cluster_calibrated(self) -> bool:
        return self.cluster_level_interval[0] <= self.alpha <= self.cluster_level_interval[1]

    @property
    def se_understatement(self) -> float:
        """单元级标准误被**低估**的比例。

        注意方向：``se_ratio`` 是"簇级 SE ÷ 单元级 SE"（> 1）。
        单元级 SE 相对正确值小了多少，是 ``1 - 1/se_ratio``，不是 ``1 - se_ratio`` ——
        后者会给出 -530% 这种把比例算反的数（第一版就是这么错的）。
        """
        if self.se_ratio <= 0:
            return float("nan")
        return 1.0 - 1.0 / self.se_ratio


def run_unit_awareness_audit(
    *,
    n_salts: int = 300,
    n_users: int = 10_000,
    n_looks: int = 5,
    alpha: float = 0.05,
) -> UnitAwarenessResult:
    """整簇随机化 + 零效应，走真实管道，比较单元级与簇级两个口径。"""
    unit_hits = cluster_hits = 0
    unit_seq = cluster_seq = 0
    se_ratios: list[float] = []
    n_clusters_seen = 0

    for i in range(n_salts):
        rec = _record(f"unit_{i}", f"unit_audit_{i}")
        rec.analysis_unit = "cluster"
        rec.estimator = "post_only"
        rep = analyse_experiment(rec, n_users=n_users, n_looks=n_looks, alpha=alpha)

        unit_hits += int(rep.alt.significant)
        cluster_hits += int(rep.primary.significant)
        unit_seq += int(
            any(m["alt_z"] is not None and abs(m["alt_z"]) >= m["boundary"] for m in rep.monitoring)
        )
        cluster_seq += int(any(m["crossed"] for m in rep.monitoring))
        se_ratios.append(rep.primary.std_error / rep.alt.std_error)
        n_clusters_seen = rep.primary.n_treatment + rep.primary.n_control

    ratios = np.array(se_ratios)
    return UnitAwarenessResult(
        n_salts=n_salts,
        n_users=n_users,
        n_clusters=n_clusters_seen,
        n_looks=n_looks,
        alpha=alpha,
        unit_level_fpr=unit_hits / n_salts,
        cluster_level_fpr=cluster_hits / n_salts,
        unit_level_sequential=unit_seq / n_salts,
        cluster_level_sequential=cluster_seq / n_salts,
        se_ratio=float(ratios.mean()),
        unit_level_interval=_wilson(unit_hits, n_salts),
        cluster_level_interval=_wilson(cluster_hits, n_salts),
    )
