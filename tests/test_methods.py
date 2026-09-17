"""M1 方法校准测试：每个新方法都必须在正确的重复抽样框架下校准。

这些断言就是"这三个方法是可信的"的可执行证据。
种子固定，所以结果是确定性的、可复现的。
"""

import numpy as np
import pytest

from ablab.sim import PopulationConfig, generate_population, two_arm_spec
from ablab.sim.scenarios import ClusterScenarioConfig, RatioScenarioConfig
from ablab.validation import (
    CLUSTER_LEVEL,
    CLUSTER_NAIVE,
    CLUSTER_ROBUST,
    CUPED_LABEL,
    CUPED_NAIVE,
    RATIO_DELTA,
    RATIO_NAIVE,
    run_cluster_comparison,
    run_cuped_bias_decomposition,
    run_cuped_comparison,
    run_ratio_comparison,
)

POP_CFG = PopulationConfig(n_units=6_000, seed=20260101)


@pytest.fixture(scope="module")
def pop():
    return generate_population(POP_CFG)


@pytest.fixture(scope="module")
def spec():
    return two_arm_spec("cuped_test", salt="cuped_test_v1")


# --------------------------------------------------------------------------- #
# CUPED
# --------------------------------------------------------------------------- #
class TestCupedCalibration:
    @pytest.fixture(scope="class")
    def randomized(self, pop, spec):
        return run_cuped_comparison(pop, spec, n_trials=600, mode="randomized", seed=7)

    def test_cuped_is_calibrated(self, randomized):
        """CUPED 的 I 类错误必须落在名义 5% 附近。"""
        cuped = randomized.get(CUPED_LABEL)
        assert 0.025 <= cuped.fpr() <= 0.085, f"CUPED FPR = {cuped.fpr()}"

    def test_cuped_coverage(self, randomized):
        cuped = randomized.get(CUPED_LABEL)
        assert 0.92 <= cuped.coverage() <= 0.98

    def test_cuped_pvalues_uniform(self, randomized):
        _d, p = randomized.get(CUPED_LABEL).uniformity_test()
        assert p > 0.01, f"CUPED 的 p 值拒绝均匀分布 (p={p})"

    def test_cuped_unbiased(self, randomized):
        assert abs(randomized.get(CUPED_LABEL).bias) < 0.06

    def test_variance_reduction_matches_rho_squared(self, randomized):
        """实测方差缩减应等于 rho^2 —— 这是 CUPED 的核心恒等式。"""
        naive = randomized.get(CUPED_NAIVE)
        cuped = randomized.get(CUPED_LABEL)
        measured = 1 - (cuped.sd_effect / naive.sd_effect) ** 2
        theory = POP_CFG.corr_pre_post**2
        assert measured == pytest.approx(theory, abs=0.10), (
            f"实测方差缩减 {measured}，理论 rho^2 {theory}"
        )

    def test_cuped_has_smaller_se_than_naive(self, randomized):
        naive = randomized.get(CUPED_NAIVE)
        cuped = randomized.get(CUPED_LABEL)
        expected_ratio = np.sqrt(1 - POP_CFG.corr_pre_post**2)
        assert cuped.mean_se / naive.mean_se == pytest.approx(expected_ratio, abs=0.06)

    def test_se_estimate_matches_actual_sd(self, randomized):
        """平均标准误估计必须与效应估计的实际标准差吻合。"""
        for method in randomized.methods:
            assert method.mean_se == pytest.approx(method.sd_effect, rel=0.08)


class TestCupedRemovesConditionalBias:
    """固定分流下，CUPED 必须消掉协变量失衡造成的偏置。"""

    @pytest.fixture(scope="class")
    def conditional(self, pop, spec):
        return run_cuped_comparison(pop, spec, n_trials=600, mode="conditional", seed=7)

    def test_naive_is_biased(self, conditional):
        naive = conditional.get(CUPED_NAIVE)
        assert abs(naive.bias) > 0.05, "固定分流下 post-only 应当有可见偏置"

    def test_cuped_removes_most_of_the_bias(self, conditional):
        naive = conditional.get(CUPED_NAIVE)
        cuped = conditional.get(CUPED_LABEL)
        removed = 1 - abs(cuped.bias) / abs(naive.bias)
        assert removed > 0.7, f"只消掉 {removed:.1%} 的偏置"

    def test_cuped_is_calibrated_under_conditional_mode(self, conditional):
        """CUPED 在固定分流下也应当校准 —— 这正是不用重抽总体时它的价值。"""
        cuped = conditional.get(CUPED_LABEL)
        assert 0.025 <= cuped.fpr() <= 0.085
        _d, p = cuped.uniformity_test()
        assert p > 0.01

    def test_naive_is_not_calibrated(self, conditional):
        naive = conditional.get(CUPED_NAIVE)
        assert abs(naive.fpr() - 0.05) > 0.02

    def test_cuped_se_matches_noise_only_theory(self, conditional):
        """固定分流下真实波动只来自噪声，CUPED 的标准误恰好估到这个量。"""
        cuped = conditional.get(CUPED_LABEL)
        assert cuped.mean_se == pytest.approx(cuped.sd_effect, rel=0.08)


class TestBiasDecomposition:
    """跨随机化的偏置分解：比单次实现稳健得多的判据。

    单次实现的失衡大小是随机的，可能恰好接近 0，那一次就演示不出"消偏置"。
    看斜率则不受影响。
    """

    @pytest.fixture(scope="class")
    def decomposition(self, pop, spec):
        return run_cuped_bias_decomposition(
            pop, spec, n_assignments=60, n_noise=5, seed=91
        )

    def test_naive_slope_equals_beta(self, decomposition):
        """post-only 的效应对前置失衡的斜率必须等于 beta。"""
        assert decomposition.naive_slope == pytest.approx(
            decomposition.theoretical_slope, abs=4 * decomposition.naive_slope_se
        )
        assert decomposition.naive_slope_p < 1e-6

    def test_cuped_slope_is_zero(self, decomposition):
        """CUPED 的斜率必须统计上不显著于 0 —— 偏置被整项扣掉。"""
        assert decomposition.cuped_slope_p > 0.01

    def test_decomposition_passes(self, decomposition):
        assert decomposition.naive_bias_is_driven_by_imbalance
        assert decomposition.cuped_is_free_of_imbalance_bias
        assert decomposition.passed

    def test_summary_renders(self, decomposition):
        text = decomposition.summary()
        assert "斜率" in text
        assert "理论 beta" in text


class TestCupedPower:
    def test_cuped_has_more_power(self, pop, spec):
        """方差更小 → 同等样本量下功效更高。"""
        comparison = run_cuped_comparison(
            pop, spec, n_trials=400, mode="randomized", true_lift=0.8, seed=21
        )
        naive_power = comparison.get(CUPED_NAIVE).fpr()
        cuped_power = comparison.get(CUPED_LABEL).fpr()
        assert cuped_power > naive_power


# --------------------------------------------------------------------------- #
# 比值指标
# --------------------------------------------------------------------------- #
class TestRatioCalibration:
    @pytest.fixture(scope="class")
    def comparison(self):
        return run_ratio_comparison(
            RatioScenarioConfig(n_users=6_000), n_trials=500, seed=21
        )

    def test_delta_method_is_calibrated(self, comparison):
        delta = comparison.get(RATIO_DELTA)
        assert 0.025 <= delta.fpr() <= 0.085
        assert 0.92 <= delta.coverage() <= 0.98

    def test_delta_method_pvalues_uniform(self, comparison):
        _d, p = comparison.get(RATIO_DELTA).uniformity_test()
        assert p > 0.01

    def test_naive_is_also_calibrated(self, comparison):
        """关键的一课：naive 做法的 I 类错误也是校准的。

        **校准不等于正确** —— 一个检验可以既不偏高也不偏低，
        却系统性地报出错误的量。这正是比值指标这一节要讲的事。
        """
        naive = comparison.get(RATIO_NAIVE)
        assert 0.02 <= naive.fpr() <= 0.09

    def test_naive_estimates_a_different_estimand(self, comparison):
        """两者估计的口径有系统性差距 —— 这才是 naive 真正的错误。

        **差距体现在水平值上，不体现在效应上**：真效应为 0 时两者的效应都该是 0，
        但它们各自对着不同的"0"（人均比值 vs 合并比值）。
        """
        delta = comparison.get(RATIO_DELTA)
        naive = comparison.get(RATIO_NAIVE)
        relative_gap = abs(naive.control_level - delta.control_level) / delta.control_level
        assert relative_gap > 0.05, f"口径差距只有 {relative_gap:.2%}"

    def test_estimand_gap_is_reported(self, comparison):
        note = " ".join(comparison.notes)
        assert "口径差距" in note
        assert "校准不等于正确" in note

    def test_zero_relative_lift_gives_zero_effect(self, comparison):
        for method in comparison.methods:
            assert abs(method.bias) < 3 * method.mean_se


# --------------------------------------------------------------------------- #
# 聚类随机化
# --------------------------------------------------------------------------- #
class TestClusterCalibration:
    @pytest.fixture(scope="class")
    def comparison(self):
        return run_cluster_comparison(
            ClusterScenarioConfig(n_clusters=60, users_per_cluster=40),
            n_trials=400,
            seed=31,
        )

    def test_naive_is_catastrophically_wrong(self, comparison):
        """聚类随机化 + 用户级 t 检验 → I 类错误飙升到 30% 以上。"""
        naive = comparison.get(CLUSTER_NAIVE)
        assert naive.fpr() > 0.25, f"naive FPR = {naive.fpr()}，没有体现出问题"

    def test_naive_underestimates_se(self, comparison):
        naive = comparison.get(CLUSTER_NAIVE)
        robust = comparison.get(CLUSTER_ROBUST)
        assert naive.mean_se < robust.mean_se / 2

    def test_cr1_is_calibrated(self, comparison):
        cr = comparison.get(CLUSTER_ROBUST)
        assert 0.02 <= cr.fpr() <= 0.09, f"CR1 FPR = {cr.fpr()}"
        assert 0.91 <= cr.coverage() <= 0.99

    def test_cr1_pvalues_uniform(self, comparison):
        _d, p = comparison.get(CLUSTER_ROBUST).uniformity_test()
        assert p > 0.01

    def test_cluster_level_is_calibrated(self, comparison):
        level = comparison.get(CLUSTER_LEVEL)
        assert 0.02 <= level.fpr() <= 0.09
        assert 0.91 <= level.coverage() <= 0.99

    def test_all_methods_agree_on_point_estimate(self, comparison):
        """三种方法估计的是同一个效应，点估计必须一致（差别只在标准误）。"""
        effects = [m.mean_effect for m in comparison.methods]
        assert max(effects) - min(effects) < 1e-9

    def test_design_effect_explains_the_failure(self, comparison):
        note = " ".join(comparison.notes)
        assert "设计效应" in note
        naive = comparison.get(CLUSTER_NAIVE)
        cr = comparison.get(CLUSTER_ROBUST)
        # 标准误被低估的倍数应当与 sqrt(deff) 同量级
        assert cr.mean_se / naive.mean_se > 2.0
