"""仿真台测试：这些断言就是"框架是对的"的可执行证据。

断言都是**统计性**的，但种子固定，所以结果是确定性的、可复现的。
容差按名义水平的 2~3 个标准误设置，既能抓住真实错误，又不会随机翻车。
"""

import numpy as np
import pytest
from scipy import stats

from ablab.sim import PopulationConfig, generate_population, two_arm_spec
from ablab.sim.generator import make_experiment_sample, simulate_outcomes
from ablab.validation import (
    audit_hash_uniformity,
    audit_layer_orthogonality,
    audit_ramp_stability,
    audit_srm_calibration,
    run_aa_trials,
    run_peeking_simulation,
    run_power_trials,
    wilson_interval,
)

SMALL = PopulationConfig(n_units=6_000, seed=20260101)


@pytest.fixture(scope="module")
def pop():
    return generate_population(SMALL)


@pytest.fixture(scope="module")
def spec():
    return two_arm_spec("test_exp", salt="test_exp_v1")


class TestPopulation:
    def test_reproducible(self):
        a = generate_population(PopulationConfig(n_units=100, seed=5))
        b = generate_population(PopulationConfig(n_units=100, seed=5))
        assert np.array_equal(a.pre_metric, b.pre_metric)
        assert list(a.user_id) == list(b.user_id)

    def test_different_seed_differs(self):
        a = generate_population(PopulationConfig(n_units=100, seed=5))
        b = generate_population(PopulationConfig(n_units=100, seed=6))
        assert not np.array_equal(a.pre_metric, b.pre_metric)

    def test_ids_are_fixed_width(self):
        """定宽 id 是向量化哈希能生效的前提。"""
        p = generate_population(PopulationConfig(n_units=1000, seed=1))
        assert len({len(u) for u in p.ids()}) == 1

    def test_realized_correlation_matches_config(self):
        """生成的 post 与 pre 的相关必须等于配置的 rho —— CUPED 的理论收益靠它。"""
        cfg = PopulationConfig(n_units=50_000, seed=3, corr_pre_post=0.7)
        p = generate_population(cfg)
        variant = np.array(["control"] * len(p), dtype=object)
        post = simulate_outcomes(p, variant, true_lift=0.0, seed=1)
        rho = float(np.corrcoef(p.pre_metric, post)[0, 1])
        assert rho == pytest.approx(0.7, abs=0.02)

    def test_post_sd_matches_config(self):
        cfg = PopulationConfig(n_units=50_000, seed=4, post_sd=30.0, corr_pre_post=0.7)
        p = generate_population(cfg)
        variant = np.array(["control"] * len(p), dtype=object)
        post = simulate_outcomes(p, variant, true_lift=0.0, seed=2)
        assert float(post.std(ddof=1)) == pytest.approx(30.0, rel=0.03)

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError, match="corr_pre_post"):
            PopulationConfig(corr_pre_post=1.5)
        with pytest.raises(ValueError, match="标准差"):
            PopulationConfig(post_sd=0.0)

    def test_cuped_variance_reduction_is_rho_squared(self):
        """方差缩减 = rho^2，残余方差 = 1-rho^2 —— 这两个量最容易写反。"""
        cfg = PopulationConfig(corr_pre_post=0.7)
        assert cfg.cuped_variance_reduction == pytest.approx(0.49)
        assert cfg.cuped_remaining_variance == pytest.approx(0.51)
        assert cfg.cuped_variance_reduction + cfg.cuped_remaining_variance == pytest.approx(1.0)

    def test_cuped_se_shrinkage_is_not_variance_reduction(self):
        """标准误降幅是 1-sqrt(1-rho^2)，不等于方差缩减。"""
        cfg = PopulationConfig(corr_pre_post=0.7)
        se_shrinkage = 1 - np.sqrt(cfg.cuped_remaining_variance)
        assert se_shrinkage == pytest.approx(1 - np.sqrt(0.51), abs=1e-12)
        assert se_shrinkage != pytest.approx(cfg.cuped_variance_reduction, abs=0.01)

    def test_true_lift_shifts_only_treatment(self):
        p = generate_population(PopulationConfig(n_units=20_000, seed=9))
        variant = np.array(["control"] * len(p), dtype=object)
        variant[: len(p) // 2] = "treatment"
        base = simulate_outcomes(p, variant, true_lift=0.0, seed=1)
        lifted = simulate_outcomes(p, variant, true_lift=5.0, seed=1)
        delta = lifted - base
        assert delta[variant == "treatment"].mean() == pytest.approx(5.0, abs=1e-9)
        assert delta[variant == "control"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_sample_has_both_arms(self, pop, spec):
        s = make_experiment_sample(pop, spec, seed=1)
        assert set(np.unique(s.variant)) == {"control", "treatment"}


class TestWilsonInterval:
    def test_contains_point_estimate(self):
        lo, hi = wilson_interval(50, 1000)
        assert lo < 0.05 < hi

    def test_narrows_with_n(self):
        w_small = np.subtract(*reversed(wilson_interval(50, 1000)))
        w_big = np.subtract(*reversed(wilson_interval(500, 10000)))
        assert w_big < w_small

    def test_empty(self):
        lo, hi = wilson_interval(0, 0)
        assert np.isnan(lo) and np.isnan(hi)


class TestAACalibration:
    """核心断言：随机化模式下 t 检验必须是校准的。"""

    @pytest.fixture(scope="class")
    def randomized(self, pop, spec):
        return run_aa_trials(pop, spec, n_trials=1200, mode="randomized", seed=7)

    def test_fpr_near_nominal(self, randomized):
        fpr = randomized.empirical_fpr()
        assert 0.025 <= fpr <= 0.085, f"经验 I 类错误 {fpr} 偏离名义 0.05 过多"

    def test_coverage_near_nominal(self, randomized):
        cov = randomized.coverage()
        assert 0.92 <= cov <= 0.98, f"覆盖率 {cov} 偏离名义 0.95 过多"

    def test_pvalues_are_uniform(self, randomized):
        """比"FPR≈5%"更强的证据：整个 p 值分布必须是均匀的。"""
        _d, p = randomized.uniformity_test()
        assert p > 0.01, f"p 值拒绝均匀分布 (p={p})，检验未校准"

    def test_effect_estimate_unbiased(self, randomized):
        assert abs(randomized.mean_effect) < 0.05

    def test_effect_sd_matches_theory(self, randomized):
        """效应估计的 sd 应等于 sqrt(2*post_sd^2/n)。"""
        theory = np.sqrt(2 * SMALL.post_sd**2 / randomized.mean_n_per_arm)
        assert randomized.sd_effect == pytest.approx(theory, rel=0.06)

    def test_summary_renders(self, randomized):
        assert "经验 I 类错误" in randomized.summary()


class TestConditionalModeIsNotCalibrated:
    """条件模式（固定分流）必须复现出"边际保证 ≠ 条件保证"这一现象。

    注意**不能**断言"固定分流一定偏保守"：偏离方向由这一次实现的
    协变量失衡决定，可以偏保守也可以偏激进。可断言的是三条不变的定律：

    1. 效应估计有固定偏置，约等于 beta × (前置协变量组间差)
    2. 估计量的真实波动只来自噪声，小于 t 检验使用的标准误
    3. p 值分布不再是均匀的
    """

    @pytest.fixture(scope="class")
    def conditional(self, pop, spec):
        return run_aa_trials(pop, spec, n_trials=1200, mode="conditional", seed=7)

    def test_pvalues_reject_uniformity(self, conditional):
        _d, p = conditional.uniformity_test()
        assert p < 0.01, "固定分流下 p 值不应服从均匀分布"

    def test_error_rate_away_from_nominal(self, conditional):
        """实际错误率明显偏离 5% —— 方向不定，但一定不是校准的。"""
        assert abs(conditional.empirical_fpr() - 0.05) > 0.02

    def test_effect_estimate_is_biased(self, pop, spec):
        """偏置必须等于 beta × 前置组间差，而不是 0。"""
        from ablab.sim import assign_variants

        variant = assign_variants(pop, spec)
        imbalance = (
            pop.pre_metric[variant == "treatment"].mean()
            - pop.pre_metric[variant == "control"].mean()
        )
        beta = SMALL.corr_pre_post * SMALL.post_sd / SMALL.pre_sd
        expected_bias = beta * imbalance
        assert abs(expected_bias) > 0.02, "本次实现的失衡太小，测试数据需要调整"

        result = run_aa_trials(pop, spec, n_trials=800, mode="conditional", seed=7)
        # 噪声平均掉之后，剩下的就是固定偏置
        assert result.mean_effect == pytest.approx(expected_bias, abs=0.02)

    def test_effect_sd_matches_noise_only_theory(self, conditional):
        """条件模式下估计量的波动只来自噪声，小于 t 检验用的标准误。"""
        eps_sd = SMALL.post_sd * np.sqrt(1 - SMALL.corr_pre_post**2)
        theory_noise = np.sqrt(2 * eps_sd**2 / conditional.mean_n_per_arm)
        theory_full = np.sqrt(2 * SMALL.post_sd**2 / conditional.mean_n_per_arm)

        assert conditional.sd_effect == pytest.approx(theory_noise, rel=0.08)
        assert conditional.sd_effect < theory_full * 0.85

    def test_randomized_mode_is_the_calibrated_one(self, pop, spec):
        """对照：只有重新随机化才能得到校准的误差率。"""
        cond = run_aa_trials(pop, spec, n_trials=600, mode="conditional", seed=21)
        rand = run_aa_trials(pop, spec, n_trials=600, mode="randomized", seed=21)
        assert abs(rand.empirical_fpr() - 0.05) < abs(cond.empirical_fpr() - 0.05) + 0.05


class TestPower:
    def test_large_effect_is_always_detected(self, pop, spec):
        r = run_power_trials(pop, spec, true_lift=5.0, n_trials=200, seed=11)
        assert r.empirical_power > 0.99

    def test_zero_effect_gives_alpha_level_power(self, pop, spec):
        r = run_power_trials(pop, spec, true_lift=0.0, n_trials=600, seed=12)
        assert r.empirical_power == pytest.approx(0.05, abs=0.035)

    def test_empirical_matches_analytic(self, pop, spec):
        """经验功效必须落在解析功效附近 —— 说明功效公式没写错。"""
        r = run_power_trials(pop, spec, true_lift=1.5, n_trials=600, seed=13)
        assert r.empirical_power == pytest.approx(r.analytic_power, abs=0.08)

    def test_power_increases_with_effect(self, pop, spec):
        weak = run_power_trials(pop, spec, true_lift=0.4, n_trials=400, seed=14)
        strong = run_power_trials(pop, spec, true_lift=1.6, n_trials=400, seed=15)
        assert strong.empirical_power > weak.empirical_power


class TestPeeking:
    def test_single_look_equals_fixed_horizon(self, pop):
        """只看一次时，窥视就是普通的固定时点检验，假阳性率必须是 5%。

        这条断言专门守着 ``_look_sizes`` 不会被写错成"第一次只看 2 个样本"。
        """
        r = run_peeking_simulation(pop, n_looks=1, n_trials=1500, seed=101)
        assert r.naive_fpr == pytest.approx(0.05, abs=0.03)
        assert r.fixed_horizon_fpr == pytest.approx(0.05, abs=0.03)
        assert r.naive_fpr == r.fixed_horizon_fpr

    def test_naive_peeking_inflates_type_one_error(self, pop):
        r = run_peeking_simulation(pop, n_looks=10, n_trials=1500, seed=101)
        assert r.naive_fpr > 0.15, "朴素窥视应当显著抬高假阳性率"

    def test_rejects_too_many_looks(self, pop):
        """观察次数多到每次只剩几个样本时必须报错，而不是悄悄给出错数字。"""
        with pytest.raises(ValueError, match="样本量"):
            run_peeking_simulation(pop, n_looks=500, n_trials=10, seed=1)

    def test_fixed_horizon_stays_nominal(self, pop):
        r = run_peeking_simulation(pop, n_looks=10, n_trials=1500, seed=101)
        assert r.fixed_horizon_fpr == pytest.approx(0.05, abs=0.03)

    def test_calibrated_boundary_controls_fwer(self, pop):
        r = run_peeking_simulation(pop, n_looks=10, n_trials=1500, seed=101)
        assert r.calibrated_fpr < 0.09
        assert r.boundary > stats.norm.ppf(0.975), "标定边界必须比 1.96 更严"

    def test_more_looks_more_inflation(self, pop):
        few = run_peeking_simulation(pop, n_looks=2, n_trials=900, seed=55)
        many = run_peeking_simulation(pop, n_looks=25, n_trials=900, seed=55)
        assert many.naive_fpr > few.naive_fpr


class TestAssignmentAudit:
    def test_hash_uniformity(self, pop):
        a = audit_hash_uniformity(pop, n_salts=60)
        assert abs(a.ks_reject_rate - 0.05) < 0.10
        assert a.ks_pvalue_uniformity_p > 0.01

    def test_srm_calibration(self, pop, spec):
        a = audit_srm_calibration(pop, spec, n_salts=200)
        assert abs(a.trigger_rate - 0.05) < 0.08

    def test_layer_orthogonality(self, pop):
        a = audit_layer_orthogonality(pop, n_pairs=80)
        assert abs(a.reject_rate - 0.05) < 0.12
        assert a.max_conditional_deviation < 0.05

    def test_ramp_is_stable(self, pop, spec):
        a = audit_ramp_stability(pop, spec)
        assert a.passed
        assert list(a.enrolled_counts) == sorted(a.enrolled_counts)
