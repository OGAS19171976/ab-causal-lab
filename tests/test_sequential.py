"""M2 单元测试：alpha 消耗、群序贯边界、mSPRT、贝叶斯决策。"""

import numpy as np
import pytest
from scipy import stats

from ablab.sequential import (
    BoundarySolver,
    NormalPrior,
    adjusted_p_value,
    always_valid_path,
    build_design,
    decide,
    expected_loss,
    get_spending,
    kim_demets,
    msprt_p_value,
    msprt_statistic,
    posterior,
    posterior_mean_sd,
    probability_better,
    repeated_ci,
)
from ablab.sequential.always_valid import rejection_threshold
from ablab.sim import PopulationConfig, generate_population
from ablab.sim.sequential import (
    default_information_fractions,
    simulate_experiment_sequence,
)
from ablab.validation import (
    run_monitoring_intensity,
    run_stopping_rule_comparison,
    verify_adjusted_p_value,
    verify_boundary_accuracy,
)


# --------------------------------------------------------------------------- #
# 消耗函数
# --------------------------------------------------------------------------- #
class TestSpendingFunctions:
    @pytest.mark.parametrize("name", ["obf", "pocock", "linear", "kim-demets-2", "kim-demets-3"])
    def test_spends_exactly_alpha_at_the_end(self, name):
        fn = get_spending(name)
        assert float(fn(np.array([1.0]), 0.05)[0]) == pytest.approx(0.05, rel=1e-9)

    @pytest.mark.parametrize("name", ["obf", "pocock", "linear"])
    def test_spends_nothing_at_the_start(self, name):
        fn = get_spending(name)
        assert float(fn(np.array([0.0]), 0.05)[0]) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ["obf", "pocock", "linear"])
    def test_monotone(self, name):
        fn = get_spending(name)
        t = np.linspace(0, 1, 50)
        spend = fn(t, 0.05)
        assert np.all(np.diff(spend) >= -1e-15)

    def test_obf_is_front_loaded_conservative(self):
        """OBF 前期几乎不消耗 alpha（浮点下溢级别）；线性最激进。"""
        t = np.array([0.2, 0.6, 1.0])
        obf = get_spending("obf")(t, 0.05)
        lin = get_spending("linear")(t, 0.05)
        assert obf[0] < 1e-4
        assert obf[0] < lin[0] * 1e-2
        assert obf[1] < lin[1]

    def test_pocock_formula(self):
        t = np.array([0.5])
        expected = 0.05 * np.log(1 + (np.e - 1) * 0.5)
        assert float(get_spending("pocock")(t, 0.05)[0]) == pytest.approx(expected)

    def test_kim_demets_rho(self):
        fn = kim_demets(2.0)
        assert float(fn(np.array([0.5]), 0.05)[0]) == pytest.approx(0.05 * 0.25)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="未知的消耗函数"):
            get_spending("nope")

    def test_kim_demets_bad_rho(self):
        with pytest.raises(ValueError, match="rho 必须为正"):
            kim_demets(0.0)


# --------------------------------------------------------------------------- #
# 边界
# --------------------------------------------------------------------------- #
class TestBoundaries:
    def test_pocock_boundary_is_roughly_constant(self):
        """Pocock 的特征就是各次边界近似相等（文献 K=5 约 2.41）。"""
        d = build_design(alpha=0.05, n_looks=5, spending="pocock")
        assert d.boundaries.std() < 0.03
        assert d.boundaries.mean() == pytest.approx(2.41, abs=0.06)

    def test_obf_tracks_one_over_sqrt_t(self):
        """Lan-DeMets OBF 的边界应贴近 c/sqrt(t) 的形状。"""
        d = build_design(alpha=0.05, n_looks=5, spending="obf")
        scaled = d.boundaries * np.sqrt(d.information_fractions)
        assert scaled.std() < 0.06
        assert scaled.mean() == pytest.approx(1.99, abs=0.06)

    def test_more_looks_means_stricter_final_boundary(self):
        few = build_design(alpha=0.05, n_looks=2, spending="obf")
        many = build_design(alpha=0.05, n_looks=10, spending="obf")
        assert many.final_boundary > few.final_boundary

    def test_final_nominal_p_below_alpha(self):
        """末次边界对应的名义 p 必须小于 alpha —— 这就是窥视的代价。"""
        for name in ("obf", "pocock", "linear"):
            d = build_design(alpha=0.05, n_looks=5, spending=name)
            assert d.final_nominal_p < 0.05

    def test_single_look_matches_fixed_horizon(self):
        """只看一次时，边界必须退化成 1.96。"""
        d = build_design(alpha=0.05, n_looks=1, spending="obf")
        assert d.boundaries[0] == pytest.approx(1.959964, abs=1e-4)

    def test_crossed_1d_and_2d(self):
        d = build_design(alpha=0.05, n_looks=3, spending="obf")
        z1 = d.boundaries * 0.5
        assert not d.crossed(z1).any()
        z2 = d.boundaries * 1.5
        assert d.crossed(z2).all()
        batch = np.vstack([z1, z2])
        assert d.crossed(batch).shape == (2, 3)
        assert not d.crossed(batch)[0].any()

    def test_reject_and_first_crossing(self):
        d = build_design(alpha=0.05, n_looks=3, spending="obf")
        assert not d.reject(np.zeros(3))
        assert d.first_crossing_look(np.zeros(3)) is None
        z = np.zeros(3)
        z[1] = d.boundaries[1] + 0.01
        assert d.reject(z)
        assert d.first_crossing_look(z) == 2

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="alpha"):
            build_design(alpha=1.5, n_looks=3)
        with pytest.raises(ValueError, match="递增"):
            BoundarySolver([0.5, 0.4, 1.0])


class TestBoundarySolverConsistency:
    def test_exit_probabilities_sum_to_spend(self):
        """递归的增量越界概率之和必须等于消耗函数（自洽性）。"""
        for name in ("obf", "pocock", "linear"):
            d = build_design(alpha=0.05, n_looks=5, spending=name)
            solver = BoundarySolver(d.information_fractions)
            increments = solver.exit_probabilities(d.boundaries)
            assert np.all(increments >= -1e-12)
            assert np.cumsum(increments)[-1] == pytest.approx(0.05, rel=0.02)

    def test_grid_size_does_not_change_the_answer(self):
        """网格精度只影响数值误差量级，不该改变边界本身。

        ``exit_probabilities`` 的误差主要由插值往返造成（约 1e-4 量级），
        它**不随网格加密单调下降** —— 所以这里断言的是"两种网格给出一致结论"，
        而不是"越细越准"。真正的精度验证在 ``verify_boundary_accuracy`` 里用蒙特卡洛做。
        """
        for n_grid in (201, 801, 1601):
            solver = BoundarySolver(default_information_fractions(5), n_grid=n_grid)
            d = build_design(alpha=0.05, n_looks=5, spending="obf", solver=solver)
            total = float(np.sum(solver.exit_probabilities(d.boundaries)))
            assert total == pytest.approx(0.05, rel=0.05), f"n_grid={n_grid}: {total}"
            assert d.boundaries[-1] == pytest.approx(2.066, abs=0.02)

    def test_reliable_flag(self):
        assert build_design(alpha=0.05, n_looks=5, spending="obf").reliable
        assert build_design(alpha=0.05, n_looks=50, spending="obf").reliable
        assert build_design(alpha=0.05, n_looks=50, spending="pocock").reliable

    def test_infinite_early_boundary_is_not_a_failure(self):
        """OBF 观察次数多时首次消耗下溢，边界为 inf —— 这是对的，不是错误。"""
        d = build_design(alpha=0.05, n_looks=500, spending="obf")
        assert d.boundaries[0] > 8.0 or not np.isfinite(d.boundaries[0])
        assert np.isfinite(d.boundaries[-1])
        # |z| >= 一个巨大边界永远为假：这次查看不可能拒绝
        assert not bool(0.0 >= d.boundaries[0])

    def test_grossly_too_many_looks_is_flagged(self):
        """网格太粗时数值会崩，必须自报不可靠而不是给出错数字。

        注意这种失效表现为边界**偏小**（不是撞到网格边缘），
        所以只查边缘是不够的 —— 得靠网格细化自检。
        """
        good = build_design(alpha=0.05, n_looks=500, spending="pocock")
        assert good.reliable
        assert good.boundaries[-1] == pytest.approx(2.64, abs=0.05)

        coarse = build_design(
            alpha=0.05,
            n_looks=500,
            spending="pocock",
            solver=BoundarySolver(default_information_fractions(500), n_grid=201),
        )
        # 粗网格给出 1.24，明显偏离 —— 必须被标记
        assert coarse.boundaries[-1] < 2.0
        assert not coarse.reliable
        assert coarse.refinement_error > 0.01

    def test_refinement_error_is_small_when_accurate(self):
        d = build_design(alpha=0.05, n_looks=5, spending="obf")
        assert d.refinement_error < 0.01
        assert d.reliable

    def test_unequal_spacing(self):
        d = build_design(alpha=0.05, information_fractions=[0.25, 0.5, 1.0], spending="obf")
        assert d.n_looks == 3
        assert np.all(np.diff(d.boundaries) < 0)


class TestBoundaryAccuracy:
    def test_monte_carlo_matches_spending(self):
        acc = verify_boundary_accuracy(n_trials=60_000)
        assert acc.passed, acc.summary()
        assert acc.max_sigma < 5.0


class TestAdjustedPValue:
    def test_on_boundary_equals_alpha(self):
        """把观测值放在边界上，调整 p 值必须正好等于 alpha。"""
        ok, rows = verify_adjusted_p_value(n_looks=5)
        assert ok, rows
        for _k, _b, p in rows:
            assert p == pytest.approx(0.05, abs=1e-3)

    def test_inside_boundary_gives_larger_p(self):
        d = build_design(alpha=0.05, n_looks=5, spending="obf")
        t = d.information_fractions
        z = np.zeros(5)
        z[-1] = d.boundaries[-1] * 0.9
        assert adjusted_p_value(z, information_fractions=t) > 0.05

    def test_larger_z_gives_smaller_p(self):
        d = build_design(alpha=0.05, n_looks=5, spending="obf")
        t = d.information_fractions
        p_small = adjusted_p_value(np.r_[np.zeros(4), 3.0], information_fractions=t)
        p_big = adjusted_p_value(np.r_[np.zeros(4), 2.5], information_fractions=t)
        assert p_small < p_big


class TestRepeatedCI:
    def test_symmetric(self):
        lo, hi = repeated_ci(1.0, 0.5, 2.0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(2.0)

    def test_wider_than_fixed_horizon_early(self):
        d = build_design(alpha=0.05, n_looks=5, spending="obf")
        early = repeated_ci(0.0, 1.0, d.boundary_at(1))
        late = repeated_ci(0.0, 1.0, d.boundary_at(5))
        assert (early[1] - early[0]) > (late[1] - late[0])

    def test_negative_se_raises(self):
        with pytest.raises(ValueError, match="标准误不能为负"):
            repeated_ci(0.0, -1.0, 2.0)


# --------------------------------------------------------------------------- #
# mSPRT
# --------------------------------------------------------------------------- #
class TestMsprt:
    def test_statistic_at_zero_estimate(self):
        """估计为 0 时混合似然比 = sqrt(V/(V+tau^2))。"""
        se, tau = 0.4, 0.5
        V, tau2 = se**2, tau**2
        assert float(msprt_statistic(0.0, se, tau)) == pytest.approx(
            np.sqrt(V / (V + tau2))
        )

    def test_statistic_increases_with_effect(self):
        vals = [float(msprt_statistic(d, 0.4, 0.5)) for d in (0.0, 0.5, 1.0, 2.0)]
        assert vals == sorted(vals)

    def test_pvalue_is_capped_at_one(self):
        assert float(msprt_p_value(0.0, 0.4, 0.5)) == pytest.approx(1.0)

    def test_pvalue_decreases_with_effect(self):
        big = float(msprt_p_value(1.0, 0.4, 0.5))
        small = float(msprt_p_value(0.1, 0.4, 0.5))
        assert big < small

    def test_tau_must_be_positive(self):
        with pytest.raises(ValueError, match="tau 必须为正"):
            msprt_statistic(1.0, 0.4, 0.0)

    def test_rejection_threshold_inverts_statistic(self):
        """阈值必须真的把似然比推到 1/alpha。"""
        se, tau, alpha = 0.4, 0.5, 0.05
        thr = rejection_threshold(se, tau, alpha)
        assert float(msprt_statistic(thr, se, tau)) == pytest.approx(1 / alpha, rel=1e-9)
        assert float(msprt_p_value(thr, se, tau)) == pytest.approx(alpha, rel=1e-9)
        assert float(msprt_p_value(thr * 0.99, se, tau)) > alpha

    def test_path_helper(self):
        est = np.array([0.1, 0.5, 1.5, 2.0])
        se = np.full(4, 0.4)
        r = always_valid_path(est, se, tau=0.5, alpha=0.05)
        assert r.n_looks == 4
        assert r.min_p == pytest.approx(float(np.min(r.p_values)))
        assert r.thresholds.shape == (4,)

    def test_path_length_mismatch(self):
        with pytest.raises(ValueError, match="长度必须一致"):
            always_valid_path([0.1, 0.2], [0.4], tau=0.5)


# --------------------------------------------------------------------------- #
# 贝叶斯
# --------------------------------------------------------------------------- #
class TestTauChoice:
    """``tau`` 怎么选：一条能算出来的规则 + 一个必须避开的陷阱。

    与 ``TestMsprt`` 的分工：那一组测 mSPRT 本身（统计量、p 值、阈值反解），
    这一组测"先验尺度该取多少"这条**设计决策**。
    """

    def test_optimal_tau_matches_a_brute_force_grid(self):
        """黄金分割解出来的 tau 必须与暴力网格的最小点一致。"""
        from ablab.sequential import optimal_tau, rejection_threshold

        for alpha in (0.05, 0.01):
            for se in (0.1, 0.42):
                tau_star = optimal_tau(se, alpha=alpha)
                grid = np.linspace(0.2 * se, 12.0 * se, 4001)
                best = min(grid, key=lambda x: rejection_threshold(se, float(x), alpha))
                assert tau_star == pytest.approx(best, rel=0.01)
                # 阈值在 tau* 处确实是最小的
                assert rejection_threshold(se, tau_star, alpha) <= rejection_threshold(
                    se, best, alpha
                ) + 1e-9

    def test_optimal_tau_scales_with_alpha_and_se(self):
        from ablab.sequential import optimal_tau

        # 与 SE 成正比（尺度不变）
        a = optimal_tau(0.2, alpha=0.05) / 0.2
        b = optimal_tau(1.0, alpha=0.05) / 1.0
        assert a == pytest.approx(b, rel=1e-6)
        assert 2.5 < a < 3.2  # alpha=0.05 时约 2.87
        # alpha 越小（越严）需要越宽的先验
        assert optimal_tau(1.0, alpha=0.01) > optimal_tau(1.0, alpha=0.05)

    def test_choose_tau_rules(self):
        from ablab.sequential import choose_tau, optimal_tau

        assert choose_tau(std_error=0.42) == pytest.approx(optimal_tau(0.42))
        assert choose_tau(std_error=0.42, target_effect=1.5, rule="match") == 1.5
        with pytest.raises(ValueError, match="target_effect"):
            choose_tau(std_error=0.42, rule="match")
        with pytest.raises(ValueError, match="rule"):
            choose_tau(std_error=0.42, rule="magic")
        with pytest.raises(ValueError, match="std_error"):
            choose_tau(std_error=0.0)

    def test_msprt_statistic_accepts_an_array_tau(self):
        """数组 tau 只为"让数据选先验"那个反例服务 —— 但必须能算。"""
        from ablab.sequential import msprt_p_value

        est = np.array([0.1, 0.5, 1.0])
        se = np.array([0.2, 0.2, 0.2])
        p_arr = msprt_p_value(est, se, np.array([0.1, 0.2, 0.3]))
        assert p_arr.shape == (3,)
        assert np.all((p_arr > 0) & (p_arr <= 1))
        with pytest.raises(ValueError, match="tau"):
            msprt_p_value(est, se, np.array([0.1, -0.2, 0.3]))

    def test_data_dependent_tau_voids_the_guarantee(self):
        """审计的核心结论：让数据选先验，FWER 在每个监测密度下都变大。"""
        from ablab.validation import run_tau_rule_audit

        audit = run_tau_rule_audit(n_trials=4000)
        assert audit.validity_holds_for_any_fixed_tau
        assert audit.rule_is_near_the_empirical_optimum
        assert audit.data_dependent_tau_voids_the_guarantee
        # 规则点与经验最优在同一个量级（不是碰巧）
        assert abs(audit.rule_tau_over_se - audit.empirical_best_tau_over_se) < 0.6


class TestBayesian:
    def test_posterior_is_between_prior_and_data(self):
        prior = NormalPrior(sd=1.0)
        post = posterior(2.0, 0.5, prior)
        assert 0.0 < post.mean < 2.0
        assert post.sd < 0.5

    def test_posterior_precision_adds(self):
        prior = NormalPrior(sd=0.5)
        post = posterior(1.0, 0.25, prior)
        assert post.precision == pytest.approx(1 / 0.5**2 + 1 / 0.25**2)

    def test_diffuse_prior_recovers_data(self):
        post = posterior(1.0, 0.3, NormalPrior(sd=1e6))
        assert post.mean == pytest.approx(1.0, rel=1e-4)
        assert post.sd == pytest.approx(0.3, rel=1e-4)

    def test_probability_better_at_zero_is_half(self):
        prior = NormalPrior(sd=1.0)
        assert posterior(0.0, 0.5, prior).probability_better() == pytest.approx(0.5)

    def test_probability_better_increases_with_effect(self):
        prior = NormalPrior(sd=1.0)
        vals = [posterior(d, 0.5, prior).probability_better() for d in (0.0, 0.5, 1.0)]
        assert vals == sorted(vals)

    def test_vectorized_matches_scalar(self):
        prior = NormalPrior(sd=0.7)
        est = np.array([0.1, 0.5, -0.3])
        se = np.array([0.4, 0.5, 0.6])
        vec = probability_better(est, se, prior)
        for i in range(est.size):
            assert float(vec[i]) == pytest.approx(
                posterior(float(est[i]), float(se[i]), prior).probability_better()
            )

    def test_posterior_mean_sd_shape(self):
        mean, sd = posterior_mean_sd(np.zeros((3, 2)), np.full(2, 0.5), NormalPrior(sd=1.0))
        assert mean.shape == (3, 2) and sd.shape == (2,)

    def test_expected_loss_symmetry_at_zero(self):
        """mu=0 时两边期望损失相等，且等于 tau*sqrt(2/pi)/2 * 2 的一半。"""
        loss_t = expected_loss(0.0, 1.0, side="treatment")
        loss_c = expected_loss(0.0, 1.0, side="control")
        assert loss_t == pytest.approx(loss_c)
        assert loss_t == pytest.approx(stats.norm.pdf(0.0))

    def test_expected_loss_sums_to_absolute_first_moment(self):
        for mu in (-1.0, 0.0, 0.5, 2.0):
            total = expected_loss(mu, 1.0, side="treatment") + expected_loss(
                mu, 1.0, side="control"
            )
            analytic = (
                2 * stats.norm.pdf(mu) + mu * (2 * stats.norm.cdf(mu) - 1)
            )
            assert total == pytest.approx(analytic, abs=1e-12)

    def test_expected_loss_shrinks_with_large_positive_effect(self):
        assert expected_loss(5.0, 1.0, side="treatment") < 1e-5

    def test_credible_interval(self):
        post = posterior(1.0, 0.4, NormalPrior(sd=1e6))
        lo, hi = post.credible_interval(0.95)
        assert lo == pytest.approx(1.0 - 1.96 * 0.4, abs=0.01)
        assert hi == pytest.approx(1.0 + 1.96 * 0.4, abs=0.01)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="先验标准差必须为正"):
            NormalPrior(sd=0.0)
        with pytest.raises(ValueError, match="标准误必须为正"):
            posterior(1.0, 0.0, NormalPrior(sd=1.0))
        with pytest.raises(ValueError, match="side"):
            expected_loss(0.0, 1.0, side="nope")

    def test_decide_requires_a_criterion(self):
        post = posterior(1.0, 0.3, NormalPrior(sd=0.5))
        with pytest.raises(ValueError, match="至少要给一个判据"):
            decide(post)

    def test_decide_actions(self):
        prior = NormalPrior(sd=0.5)
        strong = decide(posterior(2.0, 0.2, prior), probability_threshold=0.95)
        assert strong.action == "ship_treatment"
        weak = decide(posterior(0.01, 0.5, prior), probability_threshold=0.95)
        assert weak.action == "continue"
        reverse = decide(posterior(-2.0, 0.2, prior), probability_threshold=0.95)
        assert reverse.action == "keep_control"

    def test_loss_threshold_rule(self):
        prior = NormalPrior(sd=0.5)
        post = posterior(1.0, 0.3, prior)
        triggered = decide(post, loss_threshold=post.expected_loss("treatment") + 1e-9)
        assert triggered.should_stop


# --------------------------------------------------------------------------- #
# 校准（仿真）
# --------------------------------------------------------------------------- #
class TestSequentialCalibration:
    def test_naive_peeking_inflates(self):
        rules, _ = run_stopping_rule_comparison(
            n_trials=4_000, n_looks=5, effect=0.0, seed=0
        )
        naive = next(r for r in rules if r.label.startswith("naive"))
        assert naive.rate > 0.10

    def test_group_sequential_is_calibrated(self):
        rules, design = run_stopping_rule_comparison(
            n_trials=6_000, n_looks=5, effect=0.0, seed=0
        )
        seq = next(r for r in rules if r.label.startswith("sequential"))
        assert seq.rate == pytest.approx(0.05, abs=0.015)
        assert design.reliable

    def test_fixed_horizon_is_calibrated(self):
        rules, _ = run_stopping_rule_comparison(
            n_trials=6_000, n_looks=5, effect=0.0, seed=0
        )
        fixed = next(r for r in rules if r.label.startswith("fixed"))
        assert fixed.rate == pytest.approx(0.05, abs=0.015)
        assert fixed.mean_looks == 5.0

    def test_msprt_never_exceeds_alpha(self):
        """always-valid 的全部意义就在这里：任何 tau 下都不超 alpha。"""
        from ablab.validation import run_tau_sensitivity

        points = run_tau_sensitivity(
            n_trials=6_000, taus=np.array([0.2, 0.5, 1.0, 2.0, 5.0]), seed=0
        )
        assert all(p.msprt_fwer <= 0.05 for p in points)

    def test_bayesian_threshold_is_not_self_calibrating(self):
        """0.95 阈值随先验从极保守变到偏激进 —— 这不是校准规则。"""
        from ablab.validation import run_tau_sensitivity

        points = run_tau_sensitivity(
            n_trials=6_000, taus=np.array([0.2, 1.0, 5.0]), seed=0
        )
        assert points[0].bayes_fwer < 0.01
        assert points[-1].bayes_fwer > 0.05

    def test_monitoring_intensity_trends(self):
        points = run_monitoring_intensity(
            look_counts=(2, 20, 100), n_trials=4_000, seed=0
        )
        naive = [p.naive_fwer for p in points]
        seq = [p.sequential_fwer for p in points]
        assert naive == sorted(naive)
        assert all(abs(s - 0.05) < 0.015 for s in seq)


class TestUserLevelSequence:
    def test_sequence_shapes_and_monotone_information(self):
        pop = generate_population(PopulationConfig(n_units=12_000, seed=1))
        seq = simulate_experiment_sequence(pop, n_looks=5, true_lift=0.0, seed=7)
        assert seq.n_looks == 5
        assert np.all(np.diff(seq.per_arm) > 0)
        assert seq.information_fractions[-1] == pytest.approx(1.0)

    def test_sequence_se_is_decreasing(self):
        pop = generate_population(PopulationConfig(n_units=12_000, seed=2))
        seq = simulate_experiment_sequence(pop, n_looks=5, true_lift=0.0, seed=8)
        assert np.all(np.diff(seq.standard_errors) < 0)

    def test_sequence_detects_large_effect(self):
        pop = generate_population(PopulationConfig(n_units=20_000, seed=3))
        seq = simulate_experiment_sequence(pop, n_looks=5, true_lift=8.0, seed=9)
        assert seq.z_statistics[-1] > 2.0
