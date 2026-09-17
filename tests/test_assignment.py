"""分流引擎测试：确定性、权重、放量稳定性、分层互斥与正交。"""

import numpy as np
import pytest

from ablab.assignment import (
    N_BUCKETS,
    ExperimentSpec,
    Layer,
    LayerSlot,
    Randomizer,
    Variant,
)

IDS = [f"u{i:07d}" for i in range(20_000)]


def spec_1to1(name="exp", salt=None, traffic_ratio=1.0, layer=None):
    return ExperimentSpec(
        name=name,
        variants=(Variant("control", 0.5), Variant("treatment", 0.5)),
        salt=salt,
        traffic_ratio=traffic_ratio,
        layer=layer,
    )


class TestSpecValidation:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="权重之和"):
            ExperimentSpec("e", (Variant("a", 0.5), Variant("b", 0.4)))

    def test_duplicate_variant_names_rejected(self):
        with pytest.raises(ValueError, match="重复分支名"):
            ExperimentSpec("e", (Variant("a", 0.5), Variant("a", 0.5)))

    def test_traffic_ratio_bounds(self):
        with pytest.raises(ValueError, match="traffic_ratio"):
            spec_1to1(traffic_ratio=0.0)
        with pytest.raises(ValueError, match="traffic_ratio"):
            spec_1to1(traffic_ratio=1.5)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="必须为正"):
            Variant("a", -0.1)

    def test_salt_defaults_to_name(self):
        assert spec_1to1(name="abc").salt_ == "abc"
        assert spec_1to1(name="abc", salt="xyz").salt_ == "xyz"


class TestDeterminism:
    """分流的头号要求：同样的输入永远给同样的输出。"""

    def test_repeated_calls_agree(self):
        rz = Randomizer()
        spec = spec_1to1()
        assert rz.assign_many(IDS, spec) == rz.assign_many(IDS, spec)

    def test_order_independent(self):
        """分流结果不能依赖调用顺序 —— 否则并发环境下会不一致。"""
        rz = Randomizer()
        spec = spec_1to1()
        forward = rz.assign_many(IDS[:500], spec)
        backward = rz.assign_many(list(reversed(IDS[:500])), spec)
        assert forward == list(reversed(backward))

    def test_new_randomizer_same_result(self):
        """换一个进程/实例，结果必须一样（这就是不能用内置 hash() 的原因）。"""
        spec = spec_1to1()
        assert Randomizer().assign_many(IDS[:500], spec) == Randomizer().assign_many(
            IDS[:500], spec
        )

    def test_scalar_and_vector_agree(self):
        rz = Randomizer()
        spec = spec_1to1()
        codes = rz.assign_codes(IDS[:1000], spec)
        names = np.array([v.name for v in spec.variants], dtype=object)
        vector = names[codes].tolist()
        scalar = [rz.assign(u, spec) for u in IDS[:1000]]
        assert vector == scalar

    def test_ragged_ids_fall_back_and_still_agree(self):
        """不等宽 id 应自动回落标量实现，结果与向量化路径语义一致。"""
        rz = Randomizer()
        spec = spec_1to1()
        ragged = ["u1", "u22", "u333", "u4444", "u55555"]
        codes = rz.assign_codes(ragged, spec)
        names = np.array([v.name for v in spec.variants], dtype=object)
        assert names[codes].tolist() == [rz.assign(u, spec) for u in ragged]


class TestWeightAllocation:
    @pytest.mark.parametrize("weights", [(0.5, 0.5), (0.9, 0.1), (0.01, 0.99)])
    def test_split_matches_weights(self, weights):
        rz = Randomizer()
        spec = ExperimentSpec(
            "e", (Variant("a", weights[0]), Variant("b", weights[1]))
        )
        dist = rz.distribution(IDS, spec)
        total = len(IDS)
        for name, w in (("a", weights[0]), ("b", weights[1])):
            observed = dist[name] / total
            assert abs(observed - w) < 0.01, f"{name}: {observed} vs {w}"

    def test_three_way_split(self):
        rz = Randomizer()
        spec = ExperimentSpec(
            "e",
            (Variant("a", 0.2), Variant("b", 0.3), Variant("c", 0.5)),
        )
        dist = rz.distribution(IDS, spec)
        total = len(IDS)
        assert abs(dist["a"] / total - 0.2) < 0.01
        assert abs(dist["b"] / total - 0.3) < 0.01
        assert abs(dist["c"] / total - 0.5) < 0.01
        assert dist["__not_enrolled__"] == 0

    def test_odd_weights_still_cover_all_buckets(self):
        """权重除不尽时，末位分支必须吸收舍入误差，不能留下未分配桶。"""
        rz = Randomizer()
        spec = ExperimentSpec(
            "e", (Variant("a", 0.333), Variant("b", 0.333), Variant("c", 0.334))
        )
        dist = rz.distribution(IDS, spec)
        assert dist["__not_enrolled__"] == 0
        assert sum(dist[k] for k in ("a", "b", "c")) == len(IDS)


class TestTrafficRamp:
    """灰度放量最关键的工程性质：只增不改。"""

    def test_traffic_ratio_controls_enrollment(self):
        rz = Randomizer()
        for ratio in (0.1, 0.3, 0.5):
            dist = rz.distribution(IDS, spec_1to1(traffic_ratio=ratio))
            enrolled = len(IDS) - dist["__not_enrolled__"]
            assert abs(enrolled / len(IDS) - ratio) < 0.01

    def test_enrolled_set_is_nested(self):
        """放量后必须包含放量前的全部用户。"""
        rz = Randomizer()
        prev = None
        for ratio in (0.05, 0.1, 0.25, 0.5, 1.0):
            spec = spec_1to1(traffic_ratio=ratio)
            assigned = rz.assign_codes(IDS, spec)
            mask = assigned != -1
            if prev is not None:
                assert np.all(prev <= mask), f"放量到 {ratio} 时丢掉了老用户"
            prev = mask

    def test_variant_never_changes_during_ramp(self):
        """放量的核心承诺：已进组用户的分组保持不变。"""
        rz = Randomizer()
        full = rz.assign_codes(IDS, spec_1to1(traffic_ratio=1.0))
        for ratio in (0.05, 0.2, 0.6):
            ramped = rz.assign_codes(IDS, spec_1to1(traffic_ratio=ratio))
            mask = ramped != -1
            assert np.array_equal(ramped[mask], full[mask]), f"{ratio} 时发生了换组"


class TestLayers:
    def make_layer(self, name, salt):
        return Layer(
            name=name,
            salt=salt,
            slots=(LayerSlot("exp1", 0, 4000), LayerSlot("exp2", 4000, 7000)),
        )

    def test_slot_coverage_and_mutual_exclusion(self):
        """同层实验互斥：各实验人数 + 未命中人数 == 总人数，没有重复计数。"""
        layer = self.make_layer("L", "salt_L")
        routed = layer.route_many(IDS)
        counts = {"exp1": routed.count("exp1"), "exp2": routed.count("exp2")}
        none = routed.count(None)

        assert counts["exp1"] / len(IDS) == pytest.approx(0.4, abs=0.01)
        assert counts["exp2"] / len(IDS) == pytest.approx(0.3, abs=0.01)
        assert none / len(IDS) == pytest.approx(0.3, abs=0.01)
        # 这一条就是互斥性的可验证形式：谁也没被算两次
        assert counts["exp1"] + counts["exp2"] + none == len(IDS)

    def test_overlapping_slots_rejected(self):
        with pytest.raises(ValueError, match="重叠"):
            Layer("L", "s", (LayerSlot("a", 0, 5000), LayerSlot("b", 4000, 6000)))

    def test_out_of_range_slot_rejected(self):
        with pytest.raises(ValueError, match="越界"):
            Layer("L", "s", (LayerSlot("a", 0, N_BUCKETS + 1),))

    def test_duplicate_experiment_rejected(self):
        with pytest.raises(ValueError, match="重名"):
            Layer("L", "s", (LayerSlot("a", 0, 100), LayerSlot("a", 100, 200)))

    def test_different_layers_are_orthogonal(self):
        """跨层正交：层 A 的路由不应携带层 B 的信息。"""
        from scipy import stats as st

        la = self.make_layer("A", "salt_A")
        lb = self.make_layer("B", "salt_B")
        ra = [r or "__none__" for r in la.route_many(IDS)]
        rb = [r or "__none__" for r in lb.route_many(IDS)]

        cats_a = sorted(set(ra))
        cats_b = sorted(set(rb))
        table = np.zeros((len(cats_a), len(cats_b)), dtype=np.int64)
        ia = {c: i for i, c in enumerate(cats_a)}
        ib = {c: i for i, c in enumerate(cats_b)}
        for a, b in zip(ra, rb):
            table[ia[a], ib[b]] += 1

        assert st.chi2_contingency(table).pvalue > 0.01

    def test_same_position_different_layers_can_both_hit(self):
        """两层用相同桶区间时，同一用户应能同时命中两层的实验 —— 这正是正交的意义。"""
        la = self.make_layer("A", "salt_A")
        lb = self.make_layer("B", "salt_B")
        both = sum(
            1
            for u in IDS
            if la.route(u) is not None and lb.route(u) is not None
        )
        # 两层各覆盖 70%，独立时交集应约 49%
        assert both / len(IDS) == pytest.approx(0.49, abs=0.02)

    def test_slot_lookup(self):
        layer = self.make_layer("L", "s")
        assert layer.slot_of("exp1").size == 4000
        with pytest.raises(KeyError):
            layer.slot_of("nope")
