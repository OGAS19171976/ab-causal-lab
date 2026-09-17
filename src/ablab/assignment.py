"""确定性分流引擎。

三个设计决策，每一个都是面试考点：

1. **哈希分流，而不是自增取模**
   ``bucket = hash(salt + unit_id) % N``。分流结果只取决于用户身份，
   与"谁先到"无关 —— 因此可离线复算、可跨服务对齐、可灰度放量。

2. **"是否命中"与"分到哪组"用两把独立的哈希**
   放量（``traffic_ratio`` 从 10% 提到 50%）时，已经进组的用户
   **分组保持不变**，新增的只是原本未命中的用户。这是灰度放量的正确实现；
   如果两级共用一个哈希，"半路换组"会让实验数据彻底报废。

3. **分层正交（Layer / Domain）**
   不同层用不同 ``salt``，得到互相独立的哈希空间；同层内的实验瓜分桶区间。
   于是：**同层互斥、跨层正交**。同一批流量可以同时跑几十个实验而不相互污染。
   这是 Google 重叠实验框架（Tang et al., 2010）的核心思想。

桶空间固定为 10000 个桶：既保证分组精度（最小可开 0.01% 流量），
又能用查表法做到 O(1) 分组。

性能：``assign_many`` 在 unit_id 等宽时自动走 numpy 向量化哈希
（见 ``hashing.murmur3_32_matrix``），20k 用户从 79ms 降到 6.5ms（约 12 倍）；
不等宽时自动回落到逐条标量实现，结果逐位一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hashing import encode_keys, murmur3_32, murmur3_32_matrix

__all__ = [
    "N_BUCKETS",
    "Variant",
    "ExperimentSpec",
    "LayerSlot",
    "Layer",
    "Randomizer",
]

N_BUCKETS = 10_000
_UINT32 = 1 << 32
_WEIGHT_TOL = 1e-6


# --------------------------------------------------------------------------- #
# 实验定义
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """实验的一个分支（对照组也算一个分支）。"""

    name: str
    weight: float

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"variant {self.name!r} 的权重必须为正，收到 {self.weight}")
        if self.weight > 1:
            raise ValueError(f"variant {self.name!r} 的权重不能超过 1，收到 {self.weight}")


@dataclass(frozen=True)
class ExperimentSpec:
    """一个实验的分流定义。

    Parameters
    ----------
    name:
        实验唯一名。会写进曝光日志，是后续所有分析的 join key。
    variants:
        分支列表，权重之和必须为 1。
    salt:
        分流盐。**默认等于 name，但强烈建议显式指定一个不随改名的短串** ——
        实验改名在真实业务里很常见，用 name 当 salt 会因为改名把全部用户重新分组。
    traffic_ratio:
        实验覆盖的流量比例，用于灰度放量。放量时老用户分组不变。
    layer:
        所属层。同层实验互斥；不同层正交。
    """

    name: str
    variants: Sequence[Variant]
    salt: str | None = None
    unit: str = "user_id"
    traffic_ratio: float = 1.0
    layer: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("实验名不能为空")
        if not self.variants:
            raise ValueError(f"实验 {self.name!r} 至少需要一个分支")

        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"实验 {self.name!r} 存在重复分支名: {names}")

        total = sum(v.weight for v in self.variants)
        if abs(total - 1.0) > _WEIGHT_TOL:
            raise ValueError(
                f"实验 {self.name!r} 的分支权重之和必须为 1，当前为 {total:.6f}"
            )
        if not 0.0 < self.traffic_ratio <= 1.0:
            raise ValueError(
                f"实验 {self.name!r} 的 traffic_ratio 必须在 (0, 1]，收到 {self.traffic_ratio}"
            )

    @property
    def salt_(self) -> str:
        """实际生效的分流盐。"""
        return self.salt or self.name

    @property
    def control(self) -> str:
        return self.variants[0].name

    @property
    def treatment(self) -> str:
        return self.variants[-1].name

    @property
    def weights(self) -> dict[str, float]:
        return {v.name: v.weight for v in self.variants}


@dataclass(frozen=True)
class LayerSlot:
    """层内一片连续的桶区间，分配给某个实验。"""

    experiment: str
    start: int
    end: int  # 左闭右开

    @property
    def size(self) -> int:
        return self.end - self.start


# --------------------------------------------------------------------------- #
# 分层
# --------------------------------------------------------------------------- #
class Layer:
    """一层流量域：层内实验互斥，层间靠 salt 正交。"""

    def __init__(
        self,
        name: str,
        salt: str,
        slots: Sequence[LayerSlot],
        n_buckets: int = N_BUCKETS,
    ) -> None:
        self.name = name
        self.salt = salt
        self.n_buckets = n_buckets
        self.slots = tuple(slots)
        self._validate()
        self._lookup = self._build_lookup()

    def _validate(self) -> None:
        ordered = sorted(self.slots, key=lambda s: s.start)
        prev_end = 0
        for slot in ordered:
            if slot.start < 0 or slot.end > self.n_buckets:
                raise ValueError(
                    f"层 {self.name!r} 的 {slot.experiment!r} 桶区间 "
                    f"[{slot.start}, {slot.end}) 越界 (0~{self.n_buckets})"
                )
            if slot.start >= slot.end:
                raise ValueError(f"层 {self.name!r} 的 {slot.experiment!r} 桶区间为空")
            if slot.start < prev_end:
                raise ValueError(
                    f"层 {self.name!r} 内桶区间重叠：{slot.experiment!r} "
                    f"起点 {slot.start} 落在前一个实验区间内（层内必须互斥）"
                )
            prev_end = slot.end

        names = [s.experiment for s in ordered]
        if len(set(names)) != len(names):
            raise ValueError(f"层 {self.name!r} 内存在重名实验: {names}")

    def _build_lookup(self) -> list[str | None]:
        table: list[str | None] = [None] * self.n_buckets
        for slot in self.slots:
            for b in range(slot.start, slot.end):
                table[b] = slot.experiment
        return table

    @property
    def coverage(self) -> float:
        """层内已被实验占用的流量比例。"""
        return sum(s.size for s in self.slots) / self.n_buckets

    def position(self, unit_id: str) -> int:
        """该用户在本层的落桶位置。"""
        return murmur3_32(f"{self.salt}:{unit_id}".encode("utf-8")) % self.n_buckets

    def route(self, unit_id: str) -> str | None:
        """返回该用户命中的实验名；未命中任何实验则返回 ``None``。"""
        return self._lookup[self.position(unit_id)]

    def route_many(self, unit_ids: Sequence[str], batcher=None) -> list[str | None]:
        """批量路由；等宽 unit_id 时走向量化哈希。"""
        positions = _vector_positions(unit_ids, self.salt, self.n_buckets, batcher)
        if positions is not None:
            return [self._lookup[p] for p in positions.tolist()]
        return [self.route(u) for u in unit_ids]

    def slot_of(self, experiment: str) -> LayerSlot:
        for slot in self.slots:
            if slot.experiment == experiment:
                return slot
        raise KeyError(f"层 {self.name!r} 中不存在实验 {experiment!r}")

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        body = ", ".join(f"{s.experiment}[{s.start},{s.end})" for s in self.slots)
        return f"Layer({self.name!r}, salt={self.salt!r}, coverage={self.coverage:.1%}, {body})"


def _vector_positions(
    unit_ids: Sequence[str], salt: str, n_buckets: int, batcher=None
) -> np.ndarray | None:
    """层内落桶位置；不等宽返回 ``None`` 让调用方回落标量。"""
    hashes = _hash_prefix(unit_ids, f"{salt}:", batcher)
    if hashes is None:
        return None
    return (hashes % np.uint64(n_buckets)).astype(np.int64)


def _hash_prefix(
    unit_ids: Sequence[str], prefix: str, batcher=None
) -> np.ndarray | None:
    """``hash(prefix + unit_id)`` 的向量化版本。

    传入 ``KeyBatcher`` 可复用已编码的 unit_id 字节，换 salt 时只做一次内存拷贝。
    不等宽时返回 ``None``，调用方回落逐条标量实现。
    """
    try:
        matrix = batcher.build(prefix) if batcher is not None else encode_keys(prefix, unit_ids)
    except ValueError:
        return None
    return murmur3_32_matrix(matrix)


# --------------------------------------------------------------------------- #
# 分流器
# --------------------------------------------------------------------------- #
class Randomizer:
    """无状态分流器：给定 (spec, unit_id)，永远返回同一个分支。"""

    def __init__(self, n_buckets: int = N_BUCKETS) -> None:
        self.n_buckets = n_buckets
        self._lookup_cache: dict[tuple, tuple[list[str | None], np.ndarray, np.ndarray]] = {}

    # -- 内部：把权重表编译成"桶 -> 分支"的查表结构 ------------------------ #
    def _compile(
        self, spec: ExperimentSpec
    ) -> tuple[list[str | None], np.ndarray, np.ndarray]:
        """返回 (桶->分支名 list, 桶->分支下标 ndarray, 分支名 ndarray)。"""
        key = (
            spec.salt_,
            self.n_buckets,
            tuple((v.name, v.weight) for v in spec.variants),
        )
        cached = self._lookup_cache.get(key)
        if cached is not None:
            return cached

        names = np.array([v.name for v in spec.variants], dtype=object)
        table: list[str | None] = [None] * self.n_buckets
        idx = np.empty(self.n_buckets, dtype=np.int16)

        cursor = 0
        last = len(spec.variants) - 1
        for i, variant in enumerate(spec.variants):
            # 末位分支吸收四舍五入误差，保证桶空间被完整覆盖
            end = self.n_buckets if i == last else cursor + int(round(variant.weight * self.n_buckets))
            for b in range(cursor, end):
                table[b] = variant.name
            idx[cursor:end] = i
            cursor = end

        self._lookup_cache[key] = (table, idx, names)
        return table, idx, names

    # -- 公开 API ---------------------------------------------------------- #
    def is_enrolled(self, unit_id: str, spec: ExperimentSpec) -> bool:
        """是否被本次实验的流量覆盖（灰度放量闸门）。

        用**独立的哈希**判断命中，与分组哈希解耦，
        因此放量只增加新用户，不会让老用户换组。
        """
        return bool(self.enrolled_mask([unit_id], spec)[0])

    def enrolled_mask(
        self, unit_ids: Sequence[str], spec: ExperimentSpec, batcher=None
    ) -> np.ndarray:
        """批量判断是否进入实验流量。"""
        n = len(unit_ids)
        if spec.traffic_ratio >= 1.0:
            return np.ones(n, dtype=bool)
        h = _hash_prefix(unit_ids, f"{spec.salt_}:traffic:", batcher)
        if h is None:
            return np.array([self._scalar_enrolled(u, spec) for u in unit_ids], dtype=bool)
        return (h.astype(np.float64) / _UINT32) < spec.traffic_ratio

    def _scalar_enrolled(self, unit_id: str, spec: ExperimentSpec) -> bool:
        if spec.traffic_ratio >= 1.0:
            return True
        h = murmur3_32(f"{spec.salt_}:traffic:{unit_id}".encode("utf-8"))
        return (h / _UINT32) < spec.traffic_ratio

    def bucket(self, unit_id: str, spec: ExperimentSpec) -> int:
        """用户在本实验分组空间中的桶号（0 ~ n_buckets-1）。"""
        return (
            murmur3_32(f"{spec.salt_}:group:{unit_id}".encode("utf-8")) % self.n_buckets
        )

    def assign(self, unit_id: str, spec: ExperimentSpec) -> str | None:
        """返回分支名；未进入实验流量则返回 ``None``。"""
        if not self._scalar_enrolled(unit_id, spec):
            return None
        table, _, _ = self._compile(spec)
        return table[self.bucket(unit_id, spec)]

    def assign_many(
        self, unit_ids: Sequence[str], spec: ExperimentSpec, batcher=None
    ) -> list[str | None]:
        """批量分流。等宽 unit_id 时走向量化路径，否则逐条标量。"""
        ids = list(unit_ids)
        if not ids:
            return []

        table, idx, names = self._compile(spec)
        buckets = _hash_prefix(ids, f"{spec.salt_}:group:", batcher)

        if buckets is None:
            return [self.assign(u, spec) for u in ids]

        bucket_idx = (buckets % np.uint64(self.n_buckets)).astype(np.int64)
        out = names[idx[bucket_idx]]
        enrolled = self.enrolled_mask(ids, spec, batcher)
        if not enrolled.all():
            out = out.copy()
            out[~enrolled] = None
        return out.tolist()

    def assign_codes(
        self, unit_ids: Sequence[str], spec: ExperimentSpec, batcher=None
    ) -> np.ndarray:
        """批量分流，返回**整数分支下标**；未进入实验流量为 ``-1``。

        这是仿真循环的推荐入口：相比 ``assign_many`` 返回 Python 对象列表，
        整数编码省掉了每轮两万次对象分配、字符串比较和 ``tolist()``，
        单次分流从 7.4ms 降到 3ms 左右。
        """
        ids = list(unit_ids)
        if not ids:
            return np.empty(0, dtype=np.int16)

        _, idx, names = self._compile(spec)
        buckets = _hash_prefix(ids, f"{spec.salt_}:group:", batcher)

        if buckets is None:
            lookup = {str(n): i for i, n in enumerate(names)}
            codes = np.empty(len(ids), dtype=np.int16)
            for i, u in enumerate(ids):
                a = self.assign(u, spec)
                codes[i] = -1 if a is None else lookup[a]
            return codes

        codes = idx[(buckets % np.uint64(self.n_buckets)).astype(np.int64)]
        enrolled = self.enrolled_mask(ids, spec, batcher)
        if not enrolled.all():
            codes = codes.copy()
            codes[~enrolled] = -1
        return codes

    def variant_probabilities(self, spec: ExperimentSpec) -> dict[str, float]:
        """理论分流比例（用于 SRM 的期望值）。"""
        _, idx, names = self._compile(spec)
        counts = np.bincount(idx, minlength=len(names))
        return {str(names[i]): int(counts[i]) / self.n_buckets for i in range(len(names))}

    def distribution(
        self, unit_ids: Sequence[str], spec: ExperimentSpec, batcher=None
    ) -> dict[str, int]:
        """实际分流结果计数，含未命中（键为 ``"__not_enrolled__"``）。"""
        ids = list(unit_ids)
        table, _, _ = self._compile(spec)
        counts: dict[str, int] = {v.name: 0 for v in spec.variants}
        counts["__not_enrolled__"] = 0

        buckets = _hash_prefix(ids, f"{spec.salt_}:group:", batcher)
        if buckets is None:
            for unit_id in ids:
                if not self._scalar_enrolled(unit_id, spec):
                    counts["__not_enrolled__"] += 1
                else:
                    counts[table[self.bucket(unit_id, spec)]] += 1
            return counts

        bucket_idx = (buckets % np.uint64(self.n_buckets)).astype(np.int64)
        enrolled = self.enrolled_mask(ids, spec, batcher)

        chosen = np.empty(len(ids), dtype=object)
        chosen[enrolled] = [table[b] for b in bucket_idx[enrolled].tolist()]
        for name, cnt in zip(*np.unique(chosen[enrolled], return_counts=True)):
            counts[str(name)] = int(cnt)
        counts["__not_enrolled__"] = int((~enrolled).sum())
        return counts


def _vector_hashes(unit_ids: Sequence[str], salt: str, kind: str) -> np.ndarray | None:
    """``hash(f"{salt}:{kind}:{unit_id}")`` 的向量化版本；不等宽返回 ``None``。"""
    try:
        matrix = encode_keys(f"{salt}:{kind}:", unit_ids)
    except ValueError:
        return None
    return murmur3_32_matrix(matrix)
