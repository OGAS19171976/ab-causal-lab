"""MurmurHash3 与批量编码的测试。"""

import numpy as np
import pytest

from ablab.hashing import (
    KeyBatcher,
    encode_keys,
    murmur3_32,
    murmur3_32_matrix,
    murmur3_32_str,
)


class TestKnownVectors:
    """对照 Appleby 参考实现的公开测试向量。"""

    def test_empty(self):
        assert murmur3_32(b"") == 0

    def test_hello(self):
        assert murmur3_32(b"hello") == 613153351

    def test_str_wrapper_matches_bytes(self):
        assert murmur3_32_str("hello") == murmur3_32(b"hello")

    def test_utf8_encoding(self):
        # 中文必须按 UTF-8 编码，与线上服务一致
        assert murmur3_32_str("用户") == murmur3_32("用户".encode("utf-8"))

    def test_seed_changes_result(self):
        assert murmur3_32(b"hello", seed=1) != murmur3_32(b"hello", seed=0)


class TestVectorizedMatchesScalar:
    """向量化实现必须与标量实现逐位一致，否则分流结果会随调用方式变化。"""

    @pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 5, 7, 8, 16])
    def test_various_widths(self, width):
        keys = [f"k{i:04d}".ljust(width, "x")[:width] if width else "" for i in range(50)]
        if width == 0:
            keys = [""] * 50
        matrix = encode_keys("", keys)
        assert matrix.shape == (50, width)

        expected = np.array([murmur3_32(k.encode()) for k in keys], dtype=np.uint64)
        assert np.array_equal(murmur3_32_matrix(matrix), expected)

    def test_random_keys(self):
        rng = np.random.default_rng(0)
        keys = [f"salt_{i:04d}:group:u{rng.integers(0, 10**9):09d}" for i in range(500)]
        matrix = encode_keys("", keys)
        expected = np.array([murmur3_32(k.encode()) for k in keys], dtype=np.uint64)
        assert np.array_equal(murmur3_32_matrix(matrix), expected)

    def test_values_fit_uint32(self):
        matrix = encode_keys("p:", [f"u{i:05d}" for i in range(100)])
        values = murmur3_32_matrix(matrix)
        assert values.max() < (1 << 32)


class TestEncodeKeysValidation:
    def test_ragged_width_raises(self):
        with pytest.raises(ValueError, match="字节长度不一致"):
            encode_keys("p:", ["short", "much-longer-id"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            encode_keys("p:", [])

    def test_multibyte_width_counted_in_bytes(self):
        # "用户" 是 3+3 字节，与 6 个 ASCII 字符等宽
        matrix = encode_keys("", ["用户", "abcdef"])
        assert matrix.shape == (2, 6)


class TestKeyBatcher:
    def test_matches_encode_keys(self):
        ids = [f"u{i:07d}" for i in range(200)]
        batcher = KeyBatcher(ids)
        for prefix in ("a:", "much_longer_prefix:", ""):
            assert np.array_equal(batcher.build(prefix), encode_keys(prefix, ids))

    def test_rejects_ragged_ids(self):
        with pytest.raises(ValueError, match="字节长度不一致"):
            KeyBatcher(["a", "bb"])

    def test_reuse_is_consistent(self):
        ids = [f"u{i:07d}" for i in range(50)]
        batcher = KeyBatcher(ids)
        first = murmur3_32_matrix(batcher.build("x:"))
        second = murmur3_32_matrix(batcher.build("x:"))
        assert np.array_equal(first, second)


class TestUniformity:
    """哈希的核心性质：输出在 [0, 2^32) 上均匀。"""

    def test_ks_uniform(self):
        from scipy import stats

        ids = [f"u{i:07d}" for i in range(20_000)]
        h = murmur3_32_matrix(encode_keys("salt:", ids)).astype(np.float64) / (1 << 32)
        assert stats.kstest(h, "uniform").pvalue > 0.01

    def test_avalanche_one_bit_flip(self):
        """翻转一个 bit，输出应有约一半 bit 改变（雪崩效应）。"""
        base = murmur3_32(b"u0000001")
        flipped = murmur3_32(b"u0000002")
        differing = bin(base ^ flipped).count("1")
        # 32 bit 里改变 8~24 bit 都属正常范围
        assert 8 <= differing <= 24

    def test_sequential_ids_look_random(self):
        """相邻 id 的哈希必须无结构 —— 这是"用 id 自增取模"会踩的坑。"""
        ids = [f"u{i:07d}" for i in range(2000)]
        h = murmur3_32_matrix(encode_keys("s:", ids))
        # 相邻差值不应该单调
        diffs = np.diff(h.astype(np.int64))
        assert np.unique(np.sign(diffs)).size > 1
