"""MurmurHash3 (x86, 32-bit)：标量实现 + numpy 向量化实现。

分流引擎为什么必须自带哈希，而不是用内置 ``hash()``：

1. CPython 对 ``str`` 的 ``hash()`` 受 ``PYTHONHASHSEED`` 随机盐影响，
   同一份数据两次进程启动后分组结果**不同**，实验不可复现。
2. 线上分流服务通常由 Java / Go / C++ 提供，离线复算必须与线上**逐位一致**。
   MurmurHash3 是业界分流的事实标准（Google 重叠实验框架、各家 AB 平台）。
3. 自带实现让"分流结果可复现"不依赖任何第三方包。

两个入口，结果完全一致（有单元测试守着）：

``murmur3_32``
    纯 Python、零依赖。可以直接拷到任何地方做对照实现。
``murmur3_32_matrix``
    一次对整批等长 key 做哈希的 numpy 实现。仿真台要跑上千次
    "重新分流"，逐条调用标量版本每次 20k 用户要 79ms；
    向量化后降到 6.5ms（约 12 倍）—— 这是"能跑几千次真实哈希 A/A"的前提。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # 仅供类型检查；运行时 numpy 是惰性的（见 _require_numpy）
    import numpy as np

__all__ = [
    "murmur3_32",
    "murmur3_32_str",
    "murmur3_32_matrix",
    "encode_keys",
    "KeyBatcher",
]

_C1 = 0xCC9E2D51
_C2 = 0x1B873593
_MASK = 0xFFFFFFFF

_NUMPY = None


def _require_numpy():
    """惰性加载 numpy。

    标量 ``murmur3_32`` 刻意保持零依赖（可以直接拷到生成脚本或对照实现里），
    所以 numpy 只在真正走向量化路径时才导入，并且只解析一次。
    """
    global _NUMPY
    if _NUMPY is None:
        import numpy

        _NUMPY = numpy
    return _NUMPY


def _rotl32(x: int, r: int) -> int:
    """32 位循环左移。"""
    return ((x << r) | (x >> (32 - r))) & _MASK


# --------------------------------------------------------------------------- #
# 标量实现（零依赖）
# --------------------------------------------------------------------------- #
def murmur3_32(key: bytes, seed: int = 0) -> int:
    """返回 ``key`` 在给定 ``seed`` 下的 32 位无符号哈希值 (0 ~ 2**32-1)。

    已知向量：``murmur3_32(b"") == 0``，``murmur3_32(b"hello") == 613153351``。
    """
    length = len(key)
    h1 = seed & _MASK
    n_blocks = length // 4

    # ---- 主体：按小端序每次吞 4 字节 ----
    for i in range(n_blocks):
        off = i * 4
        k1 = (
            key[off]
            | (key[off + 1] << 8)
            | (key[off + 2] << 16)
            | (key[off + 3] << 24)
        )
        k1 = (k1 * _C1) & _MASK
        k1 = _rotl32(k1, 15)
        k1 = (k1 * _C2) & _MASK

        h1 ^= k1
        h1 = _rotl32(h1, 13)
        h1 = (h1 * 5 + 0xE6546B64) & _MASK

    # ---- 尾巴：剩余 1~3 字节 ----
    tail = length & 3
    if tail:
        off = n_blocks * 4
        k1 = 0
        if tail == 3:
            k1 ^= key[off + 2] << 16
        if tail >= 2:
            k1 ^= key[off + 1] << 8
        k1 ^= key[off]

        k1 = (k1 * _C1) & _MASK
        k1 = _rotl32(k1, 15)
        k1 = (k1 * _C2) & _MASK
        h1 ^= k1

    # ---- 终结：混入长度后做 fmix32 雪崩 ----
    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & _MASK
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & _MASK
    h1 ^= h1 >> 16
    return h1


def murmur3_32_str(key: str, seed: int = 0) -> int:
    """``murmur3_32`` 的字符串便捷入口，统一按 UTF-8 编码。"""
    return murmur3_32(key.encode("utf-8"), seed)


# --------------------------------------------------------------------------- #
# 批量编码 + 向量化实现
# --------------------------------------------------------------------------- #
def encode_keys(prefix: str, keys: Sequence[str]) -> "object":
    """把 ``prefix + key`` 拼成 (n, L) 的 uint8 矩阵。

    只有当所有 key 的 UTF-8 字节长度**完全一致**时才能向量化（矩阵必须等宽）；
    否则抛 ``ValueError``，调用方应退回标量实现。
    """
    np = _require_numpy()

    prefix_bytes = prefix.encode("utf-8")
    encoded = [k.encode("utf-8") for k in keys]
    if not encoded:
        raise ValueError("keys 不能为空")

    widths = {len(e) for e in encoded}
    if len(widths) != 1:
        raise ValueError(
            f"key 的 UTF-8 字节长度不一致（{sorted(widths)[:5]}...），无法向量化；"
            "请改用标量 murmur3_32，或把 unit_id 补齐为定宽"
        )

    buf = b"".join(prefix_bytes + e for e in encoded)
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(encoded), len(prefix_bytes) + widths.pop())


class KeyBatcher:
    """把固定的 unit_id 字节只编码一次，之后每次换前缀只做一次内存拷贝。

    仿真台要在同一个用户集合上反复换 salt 重新分流。``encode_keys`` 每次都要
    对两万个字符串做 ``encode`` + ``join``，占掉单次分流 1/3 的时间；
    而这些字节其实是**不变的**。预分配缓冲区后，换前缀的成本从 2.5ms 降到微秒级。

    要求所有 unit_id 等宽（UTF-8 字节数一致），否则 ``build`` 抛 ``ValueError``。
    """

    def __init__(self, unit_ids: Sequence[str]) -> None:
        np = _require_numpy()

        encoded = [u.encode("utf-8") for u in unit_ids]
        if not encoded:
            raise ValueError("unit_ids 不能为空")
        widths = {len(e) for e in encoded}
        if len(widths) != 1:
            raise ValueError(
                f"unit_id 的 UTF-8 字节长度不一致（{sorted(widths)[:5]}...），"
                "KeyBatcher 需要等宽 id；请改用逐条标量分流"
            )
        self._width = widths.pop()
        self._n = len(encoded)
        self._id_bytes = encoded
        self._id_matrix = np.frombuffer(
            b"".join(encoded), dtype=np.uint8
        ).reshape(self._n, self._width)
        self._prefix_cache: dict[str, np.ndarray] = {}

    @property
    def n_units(self) -> int:
        return self._n

    @property
    def key_width(self) -> int:
        return self._width

    def _prefix_bytes(self, prefix: str) -> np.ndarray:
        cached = self._prefix_cache.get(prefix)
        if cached is None:
            np = _require_numpy()
            cached = np.frombuffer(prefix.encode("utf-8"), dtype=np.uint8)
            if len(self._prefix_cache) < 4096:  # 简单上限，避免无限增长
                self._prefix_cache[prefix] = cached
        return cached

    def build(self, prefix: str) -> "object":
        """返回 (n, len(prefix)+width) 的 uint8 矩阵，可直接喂给 ``murmur3_32_matrix``。"""
        np = _require_numpy()

        pb = self._prefix_bytes(prefix)
        buf = np.empty((self._n, len(pb) + self._width), dtype=np.uint8)
        buf[:, : len(pb)] = pb
        buf[:, len(pb):] = self._id_matrix
        return buf


def murmur3_32_matrix(block: "object", seed: int = 0) -> "object":
    """对 (n, L) 的 uint8 矩阵按行做 MurmurHash3，返回 (n,) 的 uint64 结果。

    与 ``murmur3_32(bytes(row))`` 逐位一致。主体循环按 4 字节分组，
    每组一次向量运算，所以 Python 层的循环次数只跟 key 长度有关，
    与样本量无关。
    """
    np = _require_numpy()

    block = np.ascontiguousarray(block, dtype=np.uint8)
    if block.ndim != 2:
        raise ValueError(f"需要二维 (n, L) 矩阵，收到 shape={block.shape}")

    n, length = block.shape
    u64 = np.uint64
    c1, c2 = u64(_C1), u64(_C2)
    mask = u64(_MASK)

    def rotl(a, r: int):
        r = u64(r)
        return ((a << r) | (a >> (u64(32) - r))) & mask

    h1 = np.full(n, u64(seed & _MASK), dtype=np.uint64)
    n_blocks = length // 4

    if n_blocks:
        body = block[:, : n_blocks * 4].reshape(n, n_blocks, 4).astype(np.uint64)
        k1 = (
            body[:, :, 0]
            | (body[:, :, 1] << u64(8))
            | (body[:, :, 2] << u64(16))
            | (body[:, :, 3] << u64(24))
        )
        k1 = (k1 * c1) & mask
        k1 = rotl(k1, 15)
        k1 = (k1 * c2) & mask

        h1 = h1.astype(np.uint64)
        for j in range(n_blocks):
            hj = h1 ^ k1[:, j]
            hj = rotl(hj, 13)
            h1 = (hj * u64(5) + u64(0xE6546B64)) & mask

    tail = length & 3
    if tail:
        off = n_blocks * 4
        k1 = np.zeros(n, dtype=np.uint64)
        if tail == 3:
            k1 ^= block[:, off + 2].astype(np.uint64) << u64(16)
        if tail >= 2:
            k1 ^= block[:, off + 1].astype(np.uint64) << u64(8)
        k1 ^= block[:, off].astype(np.uint64)

        k1 = (k1 * c1) & mask
        k1 = rotl(k1, 15)
        k1 = (k1 * c2) & mask
        h1 = h1 ^ k1

    h1 = h1 ^ u64(length)
    h1 = h1 ^ (h1 >> u64(16))
    h1 = (h1 * u64(0x85EBCA6B)) & mask
    h1 = h1 ^ (h1 >> u64(13))
    h1 = (h1 * u64(0xC2B2AE35)) & mask
    h1 = h1 ^ (h1 >> u64(16))
    return h1
