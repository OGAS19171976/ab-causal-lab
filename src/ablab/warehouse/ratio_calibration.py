"""数仓比值链路的**序贯校准**：100 个 A/A 复制实验。

为什么必须"重复实现"
--------------------
比值链路（06 / 07 两路 SQL）到这一轮之前只被证明过**一致性**：
平台编排与 M1 的独立实现在同一条链路上算出同一个数（实测偏差 0.00e+00 ~ 7.1e-15）。
一致性回答的是"两份实现有没有写岔"，它**回答不了校准**：

* 序贯监控声称"随时可以看，FWER 仍然是 α"—— 这句话是关于**一个分布**的，
  一个 salt 只能给一个数，给不出分布；
* 区间声称 95% 覆盖 —— 同样需要一个分布来数"盖住了多少次"；
* z 的方差是不是 1 —— 这是"SE 诚不诚实"的直接证据，也只能靠分布看出来。

所以这一轮造了 ``n`` 个 **A/A 复制实验**（``generate.ratio_replicate_experiments``），
每个自带一层、自带 salt、真实效应为 0，然后**走完整条真实链路**
（ODS → DWD → DWS → ADS → 平台编排 → 序贯判定）跑一遍。

这套校准为什么是**精确**的（随机化视角）
----------------------------------------
设每个用户的结果 ``Y_i`` 是固定的（这就是数仓里那一份事件序列），复制实验
``r`` 只是对同一批用户再抽一次签 ``A_r``。于是对每个 ``r``，尖锐零假设
``Y_i(1) = Y_i(0)`` **逐字成立** —— 因为这份 ``Y`` 根本不随 ``r`` 的处置而变化
（别的实验的效应跟着用户走，与本实验的分到哪一臂无关）。

* 给定固定的 ``Y``，``A_1, ..., A_n`` 由不同 salt 独立生成 →
  ``T(Y, A_r)`` 是**独立同分布**的随机化分布抽样。所以 FWER 的估计量就是
  一个二项比例，蒙特卡洛误差可以直接算；
* 条件在 ``Y`` 上的拒绝概率若 ≤ α，则无条件概率也 ≤ α（全概率公式）——
  所以这套校准不依赖任何渐近论，也不依赖 DGP 猜得对不对。

**代价也要说清**：正因为共享同一份 ``Y``，这个设计只能校准**零效应**。
要校准真实效应下的功效/覆盖，必须让复制实验走**自己的事件名和自己的 DWD 链路**
（否则它的效应会加进共享序列、把别的实验的数字改掉）。这一条记在报告的"仍未做"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..platform.analysis import analyse_experiment_from_warehouse
from ..platform.registry import ExperimentRecord
from ..validation.aa import wilson_interval
from .generate import LIFT_REPLICATE_PREFIX, RATIO_REPLICATE_PREFIX

__all__ = [
    "RatioLinkCalibration",
    "RatioLinkPowerCalibration",
    "RerandomizationReference",
    "lift_replicate_name",
    "lift_replicate_record",
    "replicate_name",
    "replicate_record",
    "rerandomization_reference",
    "run_ratio_link_calibration",
    "run_ratio_link_power_calibration",
]


def replicate_name(i: int, prefix: str = RATIO_REPLICATE_PREFIX) -> str:
    """第 ``i`` 个复制实验的名字。三个数字位，与生成器保持一致。"""
    return f"{prefix}{i:03d}"


def replicate_record(i: int, prefix: str = RATIO_REPLICATE_PREFIX) -> ExperimentRecord:
    """把复制实验包成一条平台记录（数仓路径 + 比值口径）。"""
    name = replicate_name(i, prefix)
    return ExperimentRecord(
        id=f"rep_{i:03d}",
        name=name,
        salt=f"{name}_v1",
        variants=[
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ],
        primary_metric="post_metric_14d",
        warehouse_experiment=name,
        estimator="post_only",
        metric_type="ratio",
    )


@dataclass
class RatioLinkCalibration:
    """比值链路在真实数仓上的序贯校准结果。

    三个数字回答三件事，缺一个都不算校准过：

    * ``fwer``：零效应下"至少越界一次"的比例（名义 α）—— 序贯承诺的核心；
    * ``coverage``：末次重复区间盖住 0 的比例（名义 1-α）—— 水平承诺；
    * ``z_sd_final`` / ``z_mean_final``：末次 z 的标准差与均值（应为 1 与 0）——
      SE 是否诚实的直接证据，也是前两个数字**为什么**成立的解释。
    """

    n_replicates: int
    n_looks: int
    alpha: float
    #: 每次查看的信息分数（累计分母质量之比），所有复制实验共用一套
    information_fractions: tuple[float, ...]
    final_boundary: float
    #: 序贯 FWER 与它的 Wilson 区间
    fwer: float
    fwer_interval: tuple[float, float]
    #: 逐次查看的越界率（第 k 个数 = 第 k 次查看就越界的比例）
    per_look_rate: tuple[float, ...]
    #: 只看末次查看的显著率（固定样本口径，用来对照"序贯付出的代价"）
    final_rate: float
    coverage: float
    coverage_interval: tuple[float, float]
    #: z 的分布诊断（末次查看）
    z_mean_final: float
    z_sd_final: float
    z_skew_final: float
    z_kurtosis_final: float
    #: 全部查看的 z 合在一起（只看均值与标准差；跨查看高度相关，别当独立样本读）
    z_mean_all: float
    z_sd_all: float
    #: 点估计的分布诊断：均值应 ≈ 0，sd 应 ≈ 平均 SE
    effect_mean: float
    effect_sd: float
    se_mean: float
    #: SRM：p < α 的比例（应为 α）与 Wilson 区间
    srm_rate: float
    srm_interval: tuple[float, float]
    #: salt 独立性：两两之间"同一个用户落在同一臂"的一致率（独立时应 ≈ 0.5）
    salt_agreement: float
    salt_pairs: int
    salt_shared_users: float
    #: 各复制实验的处置臂占比：均值 / 最小 / 最大（0.5 附近的抖动就是分流质量）
    arm_share_mean: float
    arm_share_min: float
    arm_share_max: float
    #: 其中失衡最明显的那个的 SRM p 值（用来解释"SRM 触发率"里的那几个）
    srm_min_p: float
    #: 过度离散检验：把 100 个 salt 的臂占比标准化成 z 再平方求和（df = 复制实验数）。
    #: p 值小说明**分流本身**在 salt 之间有系统失衡（那会是整个项目的根问题）；
    #: p 值正常而某个 salt 很极端，那只是尾部 —— 两者必须分开读。
    arm_share_chi2: float
    arm_share_chi2_p: float
    arm_share_sd_ratio: float
    #: 不变量：末次查看的 z 与主结论的 z 逐位一致的比例（M6.1 那条）
    last_look_matches_primary: float
    #: 每个复制实验的末次 z（供作图与复核）
    final_z: list[float] = field(default_factory=list)
    srm_p_values: list[float] = field(default_factory=list)

    @property
    def fwer_covers_alpha(self) -> bool:
        """FWER 的 Wilson 区间是否盖住名义 α。"""
        return self.fwer_interval[0] <= self.alpha <= self.fwer_interval[1]

    @property
    def coverage_covers_nominal(self) -> bool:
        return self.coverage_interval[0] <= 1.0 - self.alpha <= self.coverage_interval[1]

    @property
    def z_is_honest(self) -> bool:
        """z 的标准差落在 1 附近（±15%），均值落在 ±0.15 内。"""
        return abs(self.z_sd_final - 1.0) <= 0.15 and abs(self.z_mean_final) <= 0.15

    def summary(self) -> str:
        lines = [
            f"比值链路的序贯校准（{self.n_replicates} 个 A/A 复制实验 × {self.n_looks} 次查看，"
            f"名义 α={self.alpha}）",
            f"  信息分数 {[round(t, 3) for t in self.information_fractions]}，"
            f"末次边界 {self.final_boundary:.4f}",
            f"  序贯 FWER = {self.fwer:.4f}（Wilson [{self.fwer_interval[0]:.4f}, "
            f"{self.fwer_interval[1]:.4f}]）—— 名义 {self.alpha}",
            f"  逐次越界率 {[round(r, 4) for r in self.per_look_rate]}",
            f"  只看末次（固定样本口径）显著率 {self.final_rate:.4f}；"
            f"末次区间覆盖 0 的比例 {self.coverage:.4f}"
            f"（Wilson [{self.coverage_interval[0]:.4f}, {self.coverage_interval[1]:.4f}]）",
            f"  末次 z：均值 {self.z_mean_final:+.4f}，sd {self.z_sd_final:.4f}，"
            f"偏度 {self.z_skew_final:+.3f}，峰度 {self.z_kurtosis_final:+.3f}",
            f"  点估计：均值 {self.effect_mean:+.6f}，sd {self.effect_sd:.6f}，"
            f"平均 SE {self.se_mean:.6f}（sd/SE = {self.effect_sd / self.se_mean:.4f}）",
            f"  SRM 触发率 {self.srm_rate:.4f}（应为 {self.alpha}，Wilson "
            f"[{self.srm_interval[0]:.4f}, {self.srm_interval[1]:.4f}]）",
            f"  salt 独立性：{self.salt_pairs} 对复制实验、平均共享 "
            f"{self.salt_shared_users:.0f} 个用户，同臂一致率 {self.salt_agreement:.6f}"
            "（独立时应 ≈ 0.5）",
            f"  处置臂占比：均值 {self.arm_share_mean:.4f}，"
            f"范围 [{self.arm_share_min:.4f}, {self.arm_share_max:.4f}]"
            f"；最小的 SRM p 值 {self.srm_min_p:.3g}",
            f"  过度离散检验：chi2 = {self.arm_share_chi2:.2f}"
            f"（df = {self.n_replicates}），p = {self.arm_share_chi2_p:.4f}；"
            f"臂占比 sd / 二项理论 = {self.arm_share_sd_ratio:.3f}",
            f"  不变量的复核：末次查看 z 与主结论 z 逐位一致的比例 "
            f"{self.last_look_matches_primary:.4f}（应为 1.0）",
        ]
        verdict = [
            "  → 三个承诺各自的表现：",
            f"    · 序贯 FWER {'盖住' if self.fwer_covers_alpha else '**没盖住**'}名义值；",
            f"    · 末次区间覆盖 {'盖住' if self.coverage_covers_nominal else '**没盖住**'}名义值；",
            f"    · z 的 sd {'在 1 附近' if self.z_is_honest else '**偏离 1**'}"
            f"（{self.z_sd_final:.4f}）—— SE 这条链路是"
            f"{'诚实' if self.z_is_honest else '有问题'}的。",
            "  仍未做的是**真实效应**下的功效/覆盖：复制实验共享同一份结果序列、"
            "真实效应为 0，所以它只能校准零效应。",
            "  要给真实效应，必须让复制实验走自己的事件名与自己的 DWD 链路"
            "（否则它的效应会加进共享序列、改掉别的实验的数字）。",
        ]
        return "\n".join(lines + verdict)


def _srm_p_values(con, prefix: str) -> pd.Series:
    """从 ADS 的 SRM 表取出每个复制实验的卡方 p 值。

    ADS 只落了 ``chi2_statistic`` 与自由度（可加量），p 值在这里按
    ``chi2.sf`` 还原 —— 与 ``inference.srm`` 同一套做法。
    """
    frame = con.execute(
        "SELECT experiment, chi2_statistic, degrees_of_freedom FROM ads_experiment_srm"
        " WHERE experiment LIKE ? ORDER BY experiment",
        [f"{prefix}%"],
    ).df()
    if frame.empty:
        return pd.Series(dtype=float)
    p = stats.chi2.sf(
        frame["chi2_statistic"].to_numpy(dtype=float),
        frame["degrees_of_freedom"].to_numpy(dtype=float),
    )
    return pd.Series(np.asarray(p, dtype=float), index=frame["experiment"].to_numpy())


def _salt_independence(con, prefix: str) -> tuple[float, int, float]:
    """两两核对"不同 salt 的分流是否独立"。

    做法：把复制实验的曝光表自连接（同一个用户出现在两个复制实验里），
    数"两个实验把它分到同一臂"的比例。独立时应 ≈ 0.5。

    这一条不能省：整套校准假设"不同 salt 给出独立的随机化"。如果哈希在
    不同 salt 之间有结构（比如某些用户总是落在同一臂），复制实验就不是
    独立重复，FWER 的蒙特卡洛误差会被低估 —— 而那正是本仓库反复在防的
    "看起来没问题的假设"。
    """
    row = con.execute(
        """
        SELECT COUNT(*) AS shared,
               SUM(CASE WHEN a.variant = b.variant THEN 1 ELSE 0 END) AS agree
        FROM ods_exposure_log a
        JOIN ods_exposure_log b ON a.user_id = b.user_id
        WHERE a.experiment LIKE ? AND b.experiment LIKE ? AND a.experiment < b.experiment
        """,
        [f"{prefix}%", f"{prefix}%"],
    ).fetchone()
    shared = int(row[0] or 0)
    agree = int(row[1] or 0)
    if shared == 0:  # pragma: no cover - 复制实验必然有重叠用户
        return float("nan"), 0, float("nan")
    # 对数（实验对）与每对平均重叠用户数：前者是"独立重复"的检验个数
    n_exp = con.execute(
        "SELECT COUNT(DISTINCT experiment) FROM ods_exposure_log WHERE experiment LIKE ?",
        [f"{prefix}%"],
    ).fetchone()[0]
    pairs = int(n_exp) * (int(n_exp) - 1) // 2
    return agree / shared, pairs, shared / max(pairs, 1)


def _arm_shares(con, prefix: str) -> tuple[float, float, float, float, float, float]:
    """各复制实验的处置臂占比：均值 / 最小 / 最大 + 过度离散检验。

    50/50 分流下的抖动是**分流质量**的直接读数。这里做两件事，必须分开读：

    * 过度离散检验：把每个 salt 的臂占比标准化 ``z = (share - 0.5)/sqrt(0.25/n)``，
      ``Σz²`` 在"分流真的在随机"下服从 ``χ²(n)``。它回答的是**分流器整体**
      有没有系统性失衡 —— 这是整个项目的根假设，值得用 100 个 salt 验一次；
    * 极值：某一个 salt 特别偏，在 100 个里是正常的尾部现象，
      **不能**因为最小值很小就宣布分流坏了（那正是"从 100 次里挑最差的一次"）。
    """
    frame = con.execute(
        """
        SELECT SUM(CASE WHEN variant = 'treatment' THEN 1 ELSE 0 END) AS t, COUNT(*) AS n
        FROM ods_exposure_log WHERE experiment LIKE ?
        GROUP BY experiment
        """,
        [f"{prefix}%"],
    ).df()
    if frame.empty:  # pragma: no cover - 复制实验必然有曝光行
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    share = frame["t"].to_numpy(dtype=float) / frame["n"].to_numpy(dtype=float)
    n = frame["n"].to_numpy(dtype=float)
    se = np.sqrt(0.25 / n)
    z = (share - 0.5) / se
    chi2 = float(np.sum(z**2))
    theo = float(se.mean())
    return (
        float(share.mean()),
        float(share.min()),
        float(share.max()),
        chi2,
        float(stats.chi2.sf(chi2, share.size)),
        float(share.std(ddof=1) / theo),
    )


def run_ratio_link_calibration(
    con,
    *,
    n_replicates: int = 100,
    n_looks: int = 5,
    alpha: float = 0.05,
    prefix: str = RATIO_REPLICATE_PREFIX,
) -> RatioLinkCalibration:
    """在真实数仓上跑 ``n_replicates`` 个 A/A 复制实验，量序贯校准。

    ``con`` 必须是**含复制实验**的那条数仓连接（见 ``scripts/run_warehouse.py``
    的 6b 节：它单独建一条库，不污染默认演示库的数字）。
    """
    if n_replicates < 2:
        raise ValueError("复制实验至少要 2 个，否则量不出分布")

    n_cross = 0
    n_final = 0
    per_look_cross = np.zeros(n_looks, dtype=int)
    z_final: list[float] = []
    z_all: list[float] = []
    effects: list[float] = []
    ses: list[float] = []
    matched = 0
    info: tuple[float, ...] = ()
    final_boundary = float("nan")

    for i in range(n_replicates):
        rep = analyse_experiment_from_warehouse(
            replicate_record(i, prefix), con, n_looks=n_looks, alpha=alpha
        )
        rows = rep.monitoring
        if not rows:  # pragma: no cover - 数仓路径必然给出查看序列
            raise RuntimeError(f"{replicate_name(i, prefix)} 没有监控序列")
        if i == 0:
            info = tuple(float(m["information_fraction"]) for m in rows)
            final_boundary = float(rows[-1]["boundary"])

        crossed = [bool(m["crossed"]) for m in rows]
        n_cross += int(any(crossed))
        n_final += int(crossed[-1])
        per_look_cross += np.asarray(crossed, dtype=int)
        z_final.append(float(rows[-1]["z"]))
        z_all.extend(float(m["z"]) for m in rows)
        effects.append(float(rows[-1]["effect"]))
        ses.append(float(rows[-1]["std_error"]))

        # M6.1 的不变量：末次查看必须与主结论同口径、同一个数
        primary = rep.primary
        if primary is not None and primary.std_error > 0:
            z_primary = primary.absolute_effect / primary.std_error
            matched += int(abs(z_primary - float(rows[-1]["z"])) < 1e-12)

    z_arr = np.asarray(z_final, dtype=float)
    z_all_arr = np.asarray(z_all, dtype=float)
    eff = np.asarray(effects, dtype=float)
    se = np.asarray(ses, dtype=float)

    srm = _srm_p_values(con, prefix)
    srm_hits = int((srm < alpha).sum())
    srm_rate = srm_hits / max(len(srm), 1)
    salt_agree, salt_pairs, salt_shared = _salt_independence(con, prefix)
    arm_mean, arm_min, arm_max, arm_chi2, arm_p, arm_sd_ratio = _arm_shares(con, prefix)
    srm_min_p = float(srm.min()) if len(srm) else float("nan")

    return RatioLinkCalibration(
        n_replicates=n_replicates,
        n_looks=n_looks,
        alpha=alpha,
        information_fractions=info,
        final_boundary=final_boundary,
        fwer=n_cross / n_replicates,
        fwer_interval=wilson_interval(n_cross, n_replicates),
        per_look_rate=tuple((per_look_cross / n_replicates).tolist()),
        final_rate=n_final / n_replicates,
        coverage=1.0 - n_final / n_replicates,
        coverage_interval=(
            1.0 - wilson_interval(n_final, n_replicates)[1],
            1.0 - wilson_interval(n_final, n_replicates)[0],
        ),
        z_mean_final=float(z_arr.mean()),
        z_sd_final=float(z_arr.std(ddof=1)),
        z_skew_final=float(stats.skew(z_arr)),
        z_kurtosis_final=float(stats.kurtosis(z_arr)),
        z_mean_all=float(z_all_arr.mean()),
        z_sd_all=float(z_all_arr.std(ddof=1)),
        effect_mean=float(eff.mean()),
        effect_sd=float(eff.std(ddof=1)),
        se_mean=float(se.mean()),
        srm_rate=srm_rate,
        srm_interval=wilson_interval(srm_hits, max(len(srm), 1)),
        salt_agreement=salt_agree,
        salt_pairs=salt_pairs,
        salt_shared_users=salt_shared,
        arm_share_mean=arm_mean,
        arm_share_min=arm_min,
        arm_share_max=arm_max,
        srm_min_p=srm_min_p,
        arm_share_chi2=arm_chi2,
        arm_share_chi2_p=arm_p,
        arm_share_sd_ratio=arm_sd_ratio,
        last_look_matches_primary=matched / n_replicates,
        final_z=z_final,
        srm_p_values=[float(v) for v in srm.to_numpy()],
    )


def lift_replicate_name(i: int, prefix: str = LIFT_REPLICATE_PREFIX) -> str:
    """第 ``i`` 个**带真实效应**的复制实验的名字。"""
    return f"{prefix}{i:03d}"


def lift_replicate_record(i: int, prefix: str = LIFT_REPLICATE_PREFIX) -> ExperimentRecord:
    """把带真实效应的复制实验包成平台记录（数仓路径 + 比值口径 + post-only）。"""
    name = lift_replicate_name(i, prefix)
    return ExperimentRecord(
        id=f"pow_{i:03d}",
        name=name,
        salt=f"{name}_v1",
        variants=[
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ],
        primary_metric="post_metric_14d",
        warehouse_experiment=name,
        estimator="post_only",
        metric_type="ratio",
    )


@dataclass
class RatioLinkPowerCalibration:
    """比值链路在**真实效应**下的校准：覆盖、功效、以及 SE 是否诚实。

    为什么真值可以取 ``true_lift`` 本身：生成器把效应用
    ``post_effect * is_post`` 加进**每一条**后置互动记录的取值里，而分母
    ``post_cnt`` 数的就是后置互动**条数** —— 于是
    ``Y_i(1) = Y_i(0) + lift * X_i``，两边同除 ``ΣX`` 得 ``R_t = R_t(0) + lift``。
    所以真实效应 ≡ ``lift``，**逐字相等**，不需要估一个"大概的真值"。

    三个数字回答三件事：

    * ``coverage``：末次 95% 区间盖住真值的比例 —— 真实效应下的水平承诺；
    * ``power`` / ``sequential_power``：末次与序贯口径的检出率 —— 功效；
    * ``se_over_sd`` 与 ``z_sd``：平均 SE ÷ 估计的跨复制 sd（=1 才叫诚实）。
      小样本下 sd 本身有噪声，所以**必须**同时给它的卡方区间。
    """

    n_replicates: int
    n_looks: int
    alpha: float
    true_lift: float
    #: 末次固定样本区间（z=1.96）覆盖真值的比例与 Wilson 区间
    coverage: float
    coverage_interval: tuple[float, float]
    #: 末次 p < alpha 的比例（功效）与 Wilson 区间
    power: float
    power_interval: tuple[float, float]
    #: 序贯口径：至少越界一次的比例（= 序贯功效）、末次重复区间覆盖真值的比例
    sequential_power: float
    sequential_coverage: float
    #: 真值 ÷ 平均 SE：这批设置的**非中心度**（解释功效的量）
    noncentrality: float
    #: z = (τ̂ − 真值)/SE 的分布
    z_mean: float
    z_sd: float
    #: sd 的卡方区间（n−1 自由度）—— 小样本下不给它就没法判断"sd 是不是 1"
    z_sd_interval: tuple[float, float]
    #: 点估计的分布
    effect_mean: float
    effect_sd: float
    se_mean: float
    #: 平均 SE ÷ 跨复制 sd：1 才是诚实（<1 反保守，>1 保守）
    se_over_sd: float
    se_over_sd_interval: tuple[float, float]
    n_per_arm_mean: float

    @property
    def coverage_covers_nominal(self) -> bool:
        return self.coverage_interval[0] <= 1.0 - self.alpha <= self.coverage_interval[1]

    @property
    def se_is_honest(self) -> bool:
        """``se_over_sd`` 的区间是否盖住 1。"""
        return self.se_over_sd_interval[0] <= 1.0 <= self.se_over_sd_interval[1]

    def summary(self) -> str:
        return "\n".join(
            [
                f"比值链路在**真实效应**下的校准（{self.n_replicates} 个复制实验 × "
                f"{self.n_looks} 次查看，真值 = 注入的每条互动 +{self.true_lift:g}）",
                f"  末次 95% 区间覆盖真值的比例 {self.coverage:.4f}"
                f"（Wilson [{self.coverage_interval[0]:.4f}, {self.coverage_interval[1]:.4f}]，"
                f"名义 {1 - self.alpha:.2f}）",
                f"  功效：末次显著率 {self.power:.4f}"
                f"（Wilson [{self.power_interval[0]:.4f}, {self.power_interval[1]:.4f}]）；"
                f"序贯至少越界一次 {self.sequential_power:.4f}",
                f"  序贯末次重复区间覆盖真值 {self.sequential_coverage:.4f}",
                f"  非中心度（真值 ÷ 平均 SE）= {self.noncentrality:.3f}，"
                f"每臂平均 {self.n_per_arm_mean:.0f} 个单元",
                f"  z =（τ̂ − 真值）/SE：均值 {self.z_mean:+.4f}，sd {self.z_sd:.4f}"
                f"（sd 的卡方区间 [{self.z_sd_interval[0]:.4f}, {self.z_sd_interval[1]:.4f}]）",
                f"  点估计：均值 {self.effect_mean:+.6f}（真值 {self.true_lift:g}，"
                f"差 {self.effect_mean - self.true_lift:+.6f}），sd {self.effect_sd:.6f}，"
                f"平均 SE {self.se_mean:.6f}",
                f"  **平均 SE ÷ 跨复制 sd = {self.se_over_sd:.4f}**"
                f"（区间 [{self.se_over_sd_interval[0]:.4f}, {self.se_over_sd_interval[1]:.4f}]）"
                " —— 1 才是诚实",
                "  → 读法：覆盖率对着**真值本身**数（真值是逐字相等的，不是估出来的）；",
                "    非中心度决定功效，所以功效低不一定是链路的问题，可能只是这批设置太小；",
                "    SE 诚实与否看最后一行：区间盖住 1 才叫校准，否则要么反保守要么保守。",
            ]
        )


def _sd_interval(sd: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """标准差估计的卡方区间 —— 小样本下**必须**给，否则"sd 偏离 1"这句话站不住。

    12 个样本时 sd 的 95% 区间宽到 [0.59, 1.41]×，也就是"0.65 的 sd"与 1
    在这点样本量下**根本区分不开**。不报区间的 sd 对比就是过度解读。
    """
    if n < 2 or not np.isfinite(sd):
        return float("nan"), float("nan")
    lo = np.sqrt((n - 1) * sd**2 / stats.chi2.ppf(1 - alpha / 2, n - 1))
    hi = np.sqrt((n - 1) * sd**2 / stats.chi2.ppf(alpha / 2, n - 1))
    return float(lo), float(hi)


def run_ratio_link_power_calibration(
    con,
    *,
    n_replicates: int = 40,
    true_lift: float = 2.0,
    n_looks: int = 5,
    alpha: float = 0.05,
    prefix: str = LIFT_REPLICATE_PREFIX,
) -> RatioLinkPowerCalibration:
    """在真实数仓上跑 ``n_replicates`` 个**带真实效应**的复制实验。

    ``con`` 必须是只含这批实验的那条数仓连接（见 ``scripts/run_warehouse.py``
    的 6c 节）—— 它们会真的往结果里加效应，混进默认演示库会把已有数字改掉。
    """
    if n_replicates < 2:
        raise ValueError("复制实验至少要 2 个，否则量不出分布")
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    n_cover = 0
    n_power = 0
    n_seq_power = 0
    n_seq_cover = 0
    z_values: list[float] = []
    effects: list[float] = []
    ses: list[float] = []
    n_arms: list[int] = []

    for i in range(n_replicates):
        rep = analyse_experiment_from_warehouse(
            lift_replicate_record(i, prefix), con, n_looks=n_looks, alpha=alpha
        )
        primary = rep.primary
        rows = rep.monitoring
        if primary is None or not rows:  # pragma: no cover - 数仓路径必然给出
            raise RuntimeError(f"{lift_replicate_name(i, prefix)} 没有主结论或监控序列")
        est = float(primary.absolute_effect)
        se_i = float(primary.std_error)
        effects.append(est)
        ses.append(se_i)
        n_arms.append(int(rows[-1]["n_per_arm"]))
        z_values.append((est - true_lift) / se_i)
        n_cover += int(abs(est - true_lift) <= z_crit * se_i)
        n_power += int(float(primary.p_value) < alpha)
        crossed = [bool(m["crossed"]) for m in rows]
        n_seq_power += int(any(crossed))
        # 序贯末次重复区间：用设计给的末次边界而不是 1.96
        bound = float(rows[-1]["boundary"])
        n_seq_cover += int(abs(est - true_lift) <= bound * se_i)

    z_arr = np.asarray(z_values, dtype=float)
    eff = np.asarray(effects, dtype=float)
    se_arr = np.asarray(ses, dtype=float)
    sd_eff = float(eff.std(ddof=1))
    sd_z = float(z_arr.std(ddof=1))
    se_mean = float(se_arr.mean())
    ratio = se_mean / sd_eff if sd_eff > 0 else float("nan")
    sd_lo, sd_hi = _sd_interval(sd_eff, n_replicates)

    return RatioLinkPowerCalibration(
        n_replicates=n_replicates,
        n_looks=n_looks,
        alpha=alpha,
        true_lift=true_lift,
        coverage=n_cover / n_replicates,
        coverage_interval=wilson_interval(n_cover, n_replicates),
        power=n_power / n_replicates,
        power_interval=wilson_interval(n_power, n_replicates),
        sequential_power=n_seq_power / n_replicates,
        sequential_coverage=n_seq_cover / n_replicates,
        noncentrality=true_lift / se_mean,
        z_mean=float(z_arr.mean()),
        z_sd=sd_z,
        z_sd_interval=_sd_interval(sd_z, n_replicates),
        effect_mean=float(eff.mean()),
        effect_sd=sd_eff,
        se_mean=se_mean,
        se_over_sd=ratio,
        # se/sd 的区间就是把 sd 的区间倒过来（分子的噪声随 n 增长而消失）
        se_over_sd_interval=(
            se_mean / sd_hi if sd_hi > 0 else float("nan"),
            se_mean / sd_lo if sd_lo > 0 else float("nan"),
        ),
        n_per_arm_mean=float(np.mean(n_arms)),
    )


@dataclass
class RerandomizationReference:
    """对**固定结果**做重随机化，直接量估计量与 SE 的分布。

    为什么需要它：40 个复制实验给出的跨复制 sd 只有 40 个点，它的卡方区间宽到
    ±30% —— **量不出** SE 到底准不准。而 DGP 是已知的
    （``Y_i(1) = Y_i(0) + lift·X_i``），所以潜在结果可以**逐字重建**：

        处置用户  ``Y_i(0) = y_i − lift·x_i``，``Y_i(1) = y_i``
        对照用户  ``Y_i(0) = y_i``，            ``Y_i(1) = y_i + lift·x_i``

    然后在同一批用户上重抽 50/50 分流，直接得到估计量的设计分布。
    这是"SE 诚不诚实"最锋利的证据：``SE/sd`` 的蒙特卡洛误差只由重抽次数决定
    （400 次 ≈ ±3.5%），与复制实验个数无关。

    **注意它检验的是同一件事的另一面**：跨复制校准把"数据生成"的随机性也算进去，
    重随机化只留"分流"的随机性。两个都报，因为它们回答的问题不同。
    """

    experiment: str
    n_units: int
    n_splits: int
    true_lift: float
    #: 全处置 vs 全对照的真实效应（由重建的潜在结果算出，应逐字等于 ``true_lift``）
    truth: float
    estimate_mean: float
    bias: float
    sd: float
    se_mean: float
    se_over_sd: float
    #: 观测到的这一次分流的估计与 SE（对照用）
    observed_estimate: float
    observed_se: float

    def summary(self) -> str:
        return "\n".join(
            [
                f"  重随机化对照（{self.experiment}，n={self.n_units}，"
                f"{self.n_splits} 次重抽分流）",
                f"    真值 {self.truth:.6f}（DGP 注入的每条互动 +{self.true_lift:g}）",
                f"    估计均值 {self.estimate_mean:+.6f}（偏差 {self.bias:+.6f}），"
                f"sd {self.sd:.6f}",
                f"    delta method 的 SE 均值 {self.se_mean:.6f} → "
                f"**SE/sd = {self.se_over_sd:.4f}**",
                f"    观测到的那一次分流：估计 {self.observed_estimate:+.6f}，"
                f"SE {self.observed_se:.6f}",
            ]
        )


def rerandomization_reference(
    con,
    *,
    experiment: str,
    true_lift: float,
    n_splits: int = 400,
    seed: int = 0,
) -> RerandomizationReference:
    """见 ``RerandomizationReference``：重建潜在结果，重抽分流。"""
    from ..inference import ratio_delta_method
    from ..inference.aggregates import AggregateStats

    # **必须 ORDER BY**：DuckDB 并行扫描的行序不保证稳定，而重抽是对**行位置**
    # 做置换的 —— 行序一变，同一个 seed 抽到的用户组合就变了（实测同一份库、
    # 同一次重抽，SE/sd 会在 1.01 ~ 1.06 之间跳，而汇总量一位不差）。
    # 这是"报告必须可复算"的直接要求，不是洁癖。
    frame = con.execute(
        "SELECT variant, post_metric, post_cnt FROM dwd_experiment_user"
        " WHERE experiment = ? ORDER BY user_id",
        [experiment],
    ).df()
    if frame.empty:
        raise ValueError(f"DWD 里找不到实验 {experiment!r}")
    y = frame["post_metric"].to_numpy(dtype=float)
    x = frame["post_cnt"].to_numpy(dtype=float)
    treated = (frame["variant"] == "treatment").to_numpy()
    y0 = np.where(treated, y - true_lift * x, y)
    y1 = np.where(treated, y, y + true_lift * x)
    truth = float(y1.sum() / x.sum() - y0.sum() / x.sum())

    def _stats(sel: np.ndarray, values: np.ndarray) -> AggregateStats:
        xs, ys = x[sel], values[sel]
        return AggregateStats.from_sums(
            n=int(sel.sum()),
            sum_x=float(xs.sum()),
            sum_y=float(ys.sum()),
            sum_xx=float((xs * xs).sum()),
            sum_yy=float((ys * ys).sum()),
            sum_xy=float((xs * ys).sum()),
        )

    rng = np.random.default_rng(seed)
    n = x.size
    half = n // 2
    estimates = np.empty(n_splits)
    ses = np.empty(n_splits)
    for k in range(n_splits):
        order = rng.permutation(n)
        sel = np.zeros(n, dtype=bool)
        sel[order[:half]] = True
        res = ratio_delta_method(_stats(sel, y1), _stats(~sel, y0))
        estimates[k] = res.absolute_effect
        ses[k] = res.std_error

    observed = ratio_delta_method(_stats(treated, y), _stats(~treated, y))
    sd = float(estimates.std(ddof=1))
    se_mean = float(ses.mean())
    return RerandomizationReference(
        experiment=experiment,
        n_units=n,
        n_splits=n_splits,
        true_lift=true_lift,
        truth=truth,
        estimate_mean=float(estimates.mean()),
        bias=float(estimates.mean() - truth),
        sd=sd,
        se_mean=se_mean,
        se_over_sd=se_mean / sd if sd > 0 else float("nan"),
        observed_estimate=float(observed.absolute_effect),
        observed_se=float(observed.std_error),
    )
