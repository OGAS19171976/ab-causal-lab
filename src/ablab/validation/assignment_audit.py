"""分流层正确性审计。

仿真台校准的是**推断层**（t 检验算得对不对）；
这个模块校准的是**分流层**（哈希分得均不均、SRM 检验灵不灵、分层正不正交）。

四组审计，每组都给出"应该等于多少"和"实测多少"：

``hash_uniformity``
    哈希值在 [0, 2^32) 上应服从均匀分布。用 KS 检验直接测**未分桶的哈希值**
    （比分桶卡方更灵敏，且不需要桶数足够大），再辅以分桶卡方。
    跑多个 salt：好的哈希应该在 5% 的 salt 上被拒绝（而不是 0% 或 50%）。

``srm_calibration``
    SRM 检验本身是不是校准的？在**确实均匀**的分流上反复跑 SRM，
    若 SRM 的 p 值服从均匀分布，则触发率应等于设定阈值。
    这是"体检工具的体检"。

``layer_orthogonality``
    跨层实验应相互独立。对多层 pair 做列联表卡方独立性检验，
    拒绝率应约等于 5%。

``ramp_stability``
    放量时必须保证"已进组用户不换组、命中集合单调扩张"。
    这是可以确定性验证的工程性质，不靠统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ..assignment import (
    N_BUCKETS,
    ExperimentSpec,
    Layer,
    LayerSlot,
    Randomizer,
)
from ..hashing import encode_keys, murmur3_32_matrix
from ..inference.srm import srm_check
from ..sim.generator import Population, two_arm_spec

__all__ = [
    "HashUniformityAudit",
    "SRMCalibrationAudit",
    "OrthogonalityAudit",
    "RampStabilityAudit",
    "AssignmentAudit",
    "audit_hash_uniformity",
    "audit_srm_calibration",
    "audit_layer_orthogonality",
    "audit_ramp_stability",
    "run_assignment_audit",
    "make_orthogonal_layers",
]

_UINT32 = 1 << 32


def _hashes(unit_ids: list[str], salt: str) -> np.ndarray:
    """直接对 ``salt:unit_id`` 哈希，返回 uint64 数组。"""
    return murmur3_32_matrix(encode_keys(f"{salt}:", unit_ids))


# --------------------------------------------------------------------------- #
# 1. 哈希均匀性
# --------------------------------------------------------------------------- #
@dataclass
class HashUniformityAudit:
    n_salts: int
    alpha: float
    n_buckets: int
    ks_reject_rate: float
    chi2_reject_rate: float
    ks_pvalue_uniformity_p: float
    mean_ks_D: float
    worst_ks_D: float

    def summary(self) -> str:
        return (
            f"[分流层] 哈希均匀性审计（{self.n_salts} 个 salt, n 用户逐 salt 重哈希）\n"
            f"  未分桶 KS 检验拒绝率 = {self.ks_reject_rate:.4f} "
            f"(应为 {self.alpha:.2f})，均值 D={self.mean_ks_D:.4f}，最大 D={self.worst_ks_D:.4f}\n"
            f"  {self.n_buckets} 桶卡方拒绝率 = {self.chi2_reject_rate:.4f} (应为 {self.alpha:.2f})\n"
            f"  KS p 值自身的均匀性 = p={self.ks_pvalue_uniformity_p:.4g}\n"
        )


def audit_hash_uniformity(
    population: Population,
    *,
    n_salts: int = 200,
    alpha: float = 0.05,
    n_buckets: int = 100,
    salt_prefix: str = "uniformity_audit",
) -> HashUniformityAudit:
    """对多个 salt 检验哈希值的均匀性。"""
    ids = population.ids()
    ks_stats = np.empty(n_salts)
    ks_pvals = np.empty(n_salts)
    chi2_rejects = 0

    for i in range(n_salts):
        h = _hashes(ids, f"{salt_prefix}_{i}")
        u = h.astype(np.float64) / _UINT32
        res = stats.kstest(u, "uniform")
        ks_stats[i] = res.statistic
        ks_pvals[i] = res.pvalue

        counts = np.bincount(
            (h % np.uint64(n_buckets)).astype(np.int64), minlength=n_buckets
        )
        expected = np.full(n_buckets, counts.sum() / n_buckets)
        chi2_rejects += int(stats.chisquare(counts, expected).pvalue < alpha)

    return HashUniformityAudit(
        n_salts=n_salts,
        alpha=alpha,
        n_buckets=n_buckets,
        ks_reject_rate=float(np.mean(ks_pvals < alpha)),
        chi2_reject_rate=chi2_rejects / n_salts,
        ks_pvalue_uniformity_p=float(stats.kstest(ks_pvals, "uniform").pvalue),
        mean_ks_D=float(ks_stats.mean()),
        worst_ks_D=float(ks_stats.max()),
    )


# --------------------------------------------------------------------------- #
# 2. SRM 检验的校准
# --------------------------------------------------------------------------- #
@dataclass
class SRMCalibrationAudit:
    n_salts: int
    alpha: float
    trigger_rate: float
    pvalue_uniformity_p: float
    min_pvalue: float

    def summary(self) -> str:
        return (
            f"[分流层] SRM 检验校准审计（{self.n_salts} 次真实分流）\n"
            f"  触发率 = {self.trigger_rate:.4f} (应为 {self.alpha:.2f})，"
            f"最小 p={self.min_pvalue:.4g}\n"
            f"  SRM p 值均匀性 = p={self.pvalue_uniformity_p:.4g}\n"
        )


def audit_srm_calibration(
    population: Population,
    spec: ExperimentSpec,
    *,
    n_salts: int = 500,
    alpha: float = 0.05,
) -> SRMCalibrationAudit:
    """在确实均匀的分流上反复跑 SRM，检验 SRM 自身是否校准。"""
    rz = Randomizer()
    ids = population.ids()
    p_values = np.empty(n_salts)

    for i in range(n_salts):
        fresh = ExperimentSpec(
            name=spec.name,
            variants=spec.variants,
            salt=f"{spec.salt_}#srm{i}",
            traffic_ratio=spec.traffic_ratio,
            layer=spec.layer,
        )
        dist = rz.distribution(ids, fresh)
        counts = {v.name: dist[v.name] for v in spec.variants}
        diag = srm_check(counts, spec.weights, alpha=alpha)
        p_values[i] = diag.p_value if diag.p_value is not None else 1.0

    return SRMCalibrationAudit(
        n_salts=n_salts,
        alpha=alpha,
        trigger_rate=float(np.mean(p_values < alpha)),
        pvalue_uniformity_p=float(stats.kstest(p_values, "uniform").pvalue),
        min_pvalue=float(p_values.min()),
    )


# --------------------------------------------------------------------------- #
# 3. 分层正交性
# --------------------------------------------------------------------------- #
@dataclass
class OrthogonalityAudit:
    n_pairs: int
    alpha: float
    reject_rate: float
    pvalue_uniformity_p: float
    max_conditional_deviation: float

    def summary(self) -> str:
        return (
            f"[分流层] 分层正交性审计（{self.n_pairs} 组 layer pair）\n"
            f"  独立性检验拒绝率 = {self.reject_rate:.4f} (应为 {self.alpha:.2f})\n"
            f"  p 值均匀性 = p={self.pvalue_uniformity_p:.4g}，"
            f"条件分布最大偏差 = {self.max_conditional_deviation:.4f}\n"
        )


def make_orthogonal_layers(
    suffix: str = "audit", coverage: float = 0.6
) -> tuple[Layer, Layer]:
    """构造两个互不相干的层，各自装两个实验。"""
    half = int(N_BUCKETS * coverage / 2)
    layer_a = Layer(
        name="layer_a",
        salt=f"layer_a_{suffix}",
        slots=(
            LayerSlot("exp_a1", 0, half),
            LayerSlot("exp_a2", half, 2 * half),
        ),
    )
    layer_b = Layer(
        name="layer_b",
        salt=f"layer_b_{suffix}",
        slots=(
            LayerSlot("exp_b1", 0, half),
            LayerSlot("exp_b2", half, 2 * half),
        ),
    )
    return layer_a, layer_b


def audit_layer_orthogonality(
    population: Population,
    *,
    n_pairs: int = 200,
    alpha: float = 0.05,
) -> OrthogonalityAudit:
    """反复构造跨层 pair，做列联表独立性检验，看拒绝率是否等于 alpha。"""
    ids = population.ids()
    p_values = np.empty(n_pairs)
    worst_dev = 0.0

    for i in range(n_pairs):
        la, lb = make_orthogonal_layers(suffix=f"o{i}")
        ra = np.array([r if r else "__none__" for r in la.route_many(ids)])
        rb = np.array([r if r else "__none__" for r in lb.route_many(ids)])

        table = np.zeros((len(la.slots) + 1, len(lb.slots) + 1), dtype=np.int64)
        cats_a = ["__none__"] + [s.experiment for s in la.slots]
        cats_b = ["__none__"] + [s.experiment for s in lb.slots]
        idx_a = {c: j for j, c in enumerate(cats_a)}
        idx_b = {c: j for j, c in enumerate(cats_b)}
        for a, b in zip(ra, rb):
            table[idx_a[a], idx_b[b]] += 1

        p_values[i] = stats.chi2_contingency(table).pvalue

        # 条件分布 vs 边际分布的最大偏差：直观衡量"正交程度"
        row_sums = table.sum(axis=1, keepdims=True)
        cond = np.divide(table, row_sums, out=np.zeros_like(table, dtype=float), where=row_sums > 0)
        marginal = table.sum(axis=0) / table.sum()
        worst_dev = max(worst_dev, float(np.abs(cond - marginal).max()))

    return OrthogonalityAudit(
        n_pairs=n_pairs,
        alpha=alpha,
        reject_rate=float(np.mean(p_values < alpha)),
        pvalue_uniformity_p=float(stats.kstest(p_values, "uniform").pvalue),
        max_conditional_deviation=worst_dev,
    )


# --------------------------------------------------------------------------- #
# 4. 放量稳定性（确定性性质，不靠统计）
# --------------------------------------------------------------------------- #
@dataclass
class RampStabilityAudit:
    ratios: tuple[float, ...]
    nested_ok: bool
    variant_stable_ok: bool
    enrolled_counts: tuple[int, ...]
    total: int

    @property
    def passed(self) -> bool:
        return self.nested_ok and self.variant_stable_ok

    def summary(self) -> str:
        steps = ", ".join(
            f"{r:.0%}->{c:,}" for r, c in zip(self.ratios, self.enrolled_counts)
        )
        return (
            f"[分流层] 放量稳定性审计（命中集合单调扩张 且 老用户不换组）\n"
            f"  命中人数: {steps}  (总 {self.total:,})\n"
            f"  单调扩张 = {self.nested_ok}，分组不变 = {self.variant_stable_ok}  "
            f"-> {'PASS' if self.passed else 'FAIL'}\n"
        )


def audit_ramp_stability(
    population: Population,
    spec: ExperimentSpec,
    *,
    ratios: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
) -> RampStabilityAudit:
    """验证灰度放量时"只增不改"：命中集合嵌套、已命中用户分组不变。"""
    rz = Randomizer()
    ids = population.ids()
    total = len(ids)

    enrolled_sets: list[np.ndarray] = []
    variant_arrays: list[np.ndarray] = []
    counts: list[int] = []

    for ratio in ratios:
        ramped = ExperimentSpec(
            name=spec.name,
            variants=spec.variants,
            salt=spec.salt_,
            traffic_ratio=ratio,
            layer=spec.layer,
        )
        assigned = np.array(
            [a if a is not None else "" for a in rz.assign_many(ids, ramped)], dtype=object
        )
        enrolled = assigned != ""
        enrolled_sets.append(enrolled)
        variant_arrays.append(assigned)
        counts.append(int(enrolled.sum()))

    nested = all(
        np.all(enrolled_sets[i - 1] <= enrolled_sets[i]) for i in range(1, len(ratios))
    )
    stable = all(
        np.array_equal(
            variant_arrays[-1][enrolled_sets[i]], variant_arrays[i][enrolled_sets[i]]
        )
        for i in range(len(ratios) - 1)
    )

    return RampStabilityAudit(
        ratios=tuple(ratios),
        nested_ok=bool(nested),
        variant_stable_ok=bool(stable),
        enrolled_counts=tuple(counts),
        total=total,
    )


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
@dataclass
class AssignmentAudit:
    uniformity: HashUniformityAudit
    srm: SRMCalibrationAudit
    orthogonality: OrthogonalityAudit
    ramp: RampStabilityAudit
    details: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """四组审计的硬性通过条件。"""
        return (
            self.ramp.passed
            and abs(self.uniformity.ks_reject_rate - self.uniformity.alpha) < 0.05
            and abs(self.srm.trigger_rate - self.srm.alpha) < 0.05
            and abs(self.orthogonality.reject_rate - self.orthogonality.alpha) < 0.10
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "=" * 72,
                "分流层正确性审计",
                "=" * 72,
                self.uniformity.summary().rstrip(),
                self.srm.summary().rstrip(),
                self.orthogonality.summary().rstrip(),
                self.ramp.summary().rstrip(),
                f"结论: {'PASS' if self.passed else 'FAIL'}",
            ]
        )


def run_assignment_audit(
    population: Population,
    spec: ExperimentSpec | None = None,
    *,
    n_salts_uniformity: int = 200,
    n_salts_srm: int = 500,
    n_layer_pairs: int = 200,
    alpha: float = 0.05,
) -> AssignmentAudit:
    """跑完整的四组分流层审计。"""
    spec = spec or two_arm_spec("audit_spec", salt="audit_spec_v1")

    return AssignmentAudit(
        uniformity=audit_hash_uniformity(
            population, n_salts=n_salts_uniformity, alpha=alpha
        ),
        srm=audit_srm_calibration(population, spec, n_salts=n_salts_srm, alpha=alpha),
        orthogonality=audit_layer_orthogonality(
            population, n_pairs=n_layer_pairs, alpha=alpha
        ),
        ramp=audit_ramp_stability(population, spec),
    )
