"""M3 单元测试：面板 DGP、DiD 估计量、合成控制、敏感性分析。

M3 与前三个阶段不同：这里验证的不再是"数值算得对不对"，
而是"**假设被打破时估计量会怎样**"。所以测试里既有一致性检查，
也有"注入违背后偏置应当出现"这类断言。
"""

import numpy as np
import pytest

from ablab.causal import (
    StaggeredPanelConfig,
    callaway_santanna,
    event_study_leads,
    generate_scm_scenario,
    generate_staggered_panel,
    placebo_inference,
    pretrend_test,
    synthetic_control,
    trend_sensitivity,
    twfe,
    twfe_decomposition,
    two_by_two_did,
)
from ablab.validation import (
    run_pretrend_audit,
    run_scm_audit,
    run_staggered_estimator_comparison,
)

HEADLINE = StaggeredPanelConfig(
    n_units=500,
    n_periods=7,
    cohorts=(2, 4),
    cohort_weights=(0.5, 0.5),
    never_treated_share=0.05,
    effects=(1.0, 2.0, 3.0, 4.0),
    cohort_effect_multiplier=(1.0, 0.25),
    noise_sd=0.5,
)


# --------------------------------------------------------------------------- #
# 面板与 DGP
# --------------------------------------------------------------------------- #
class TestPanel:
    def test_shapes_and_cohorts(self):
        panel, truth = generate_staggered_panel(HEADLINE)
        assert panel.n_units == 500
        assert panel.n_periods == 7
        assert set(panel.cohorts()) == {2, 4}
        assert panel.never_treated_units.sum() == 25
        assert not truth.overall_att != truth.overall_att  # 非 NaN

    def test_treatment_is_absorbing(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        for g in panel.cohorts():
            rows = panel.treated[panel.cohort == g]
            # 每行必须是 0...0 1...1
            assert np.all(np.diff(rows.astype(int), axis=1) >= 0)

    def test_never_treated_never_treated(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        assert not panel.treated[panel.never_treated_units].any()

    def test_reproducible(self):
        a, ta = generate_staggered_panel(HEADLINE)
        b, tb = generate_staggered_panel(HEADLINE)
        assert np.array_equal(a.outcome, b.outcome)
        assert ta.overall_att == tb.overall_att

    def test_truth_matches_injected_effects(self):
        """真值必须等于注入的效应，不能自己算错。"""
        cfg = StaggeredPanelConfig(
            n_units=400, n_periods=6, cohorts=(3,), cohort_weights=(1.0,),
            never_treated_share=0.5, effects=(2.0,), noise_sd=0.0,
            unit_fe_sd=0.0, time_trend=0.0,
        )
        panel, truth = generate_staggered_panel(cfg)
        assert truth.overall_att == pytest.approx(2.0)
        treated = panel.treated_units
        assert panel.outcome[treated][:, 2:].mean() == pytest.approx(2.0, abs=1e-9)
        assert panel.outcome[treated][:, :2].mean() == pytest.approx(0.0, abs=1e-9)

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="长度必须一致"):
            StaggeredPanelConfig(cohorts=(2, 4), cohort_weights=(1.0,))
        with pytest.raises(ValueError, match="never_treated_share"):
            StaggeredPanelConfig(never_treated_share=1.0)
        with pytest.raises(ValueError, match="至少要两个处置队列"):
            StaggeredPanelConfig(cohorts=(2,), cohort_weights=(1.0,), never_treated_share=0.0)
        with pytest.raises(ValueError, match="必须落在"):
            StaggeredPanelConfig(n_periods=5, cohorts=(9,), cohort_weights=(1.0,))
        with pytest.raises(ValueError, match="cohort_effect_multiplier"):
            StaggeredPanelConfig(
                cohorts=(2, 4), cohort_weights=(0.5, 0.5), cohort_effect_multiplier=(1.0,)
            )


# --------------------------------------------------------------------------- #
# 2x2 与 TWFE
# --------------------------------------------------------------------------- #
class TestTwoByTwo:
    def test_recovers_known_effect(self):
        cfg = StaggeredPanelConfig(
            n_units=1000, n_periods=6, cohorts=(4,), cohort_weights=(1.0,),
            never_treated_share=0.5, effects=(3.0,), noise_sd=0.5, seed=3,
        )
        panel, _ = generate_staggered_panel(cfg)
        est = two_by_two_did(
            panel,
            treated_units=panel.treated_units,
            control_units=panel.never_treated_units,
            pre_periods=[1, 2, 3],
            post_periods=[4, 5, 6],
        )
        assert est.absolute_effect == pytest.approx(3.0, abs=0.15)
        assert est.significant

    def test_no_effect_gives_near_zero(self):
        cfg = StaggeredPanelConfig(
            n_units=1000, n_periods=6, cohorts=(4,), cohort_weights=(1.0,),
            never_treated_share=0.5, effects=(0.0,), noise_sd=0.5, seed=4,
        )
        panel, _ = generate_staggered_panel(cfg)
        est = two_by_two_did(
            panel,
            treated_units=panel.treated_units,
            control_units=panel.never_treated_units,
            pre_periods=[1, 2, 3],
            post_periods=[4, 5, 6],
        )
        assert abs(est.absolute_effect) < 4 * est.std_error

    def test_empty_periods_raise(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        with pytest.raises(ValueError, match="不能为空"):
            two_by_two_did(
                panel,
                treated_units=panel.treated_units,
                control_units=panel.never_treated_units,
                pre_periods=[],
                post_periods=[3],
            )


class TestTWFE:
    def test_single_cohort_matches_two_by_two(self):
        """只有一个处置时点时，TWFE 应当近似等于 2×2（识别上等价）。"""
        cfg = StaggeredPanelConfig(
            n_units=1500, n_periods=8, cohorts=(4,), cohort_weights=(1.0,),
            never_treated_share=0.4, effects=(2.0,), noise_sd=0.5, seed=5,
            time_trend=0.3,
        )
        panel, _ = generate_staggered_panel(cfg)
        t = twfe(panel).absolute_effect
        b = two_by_two_did(
            panel,
            treated_units=panel.treated_units,
            control_units=panel.never_treated_units,
            pre_periods=[1, 2, 3],
            post_periods=[4, 5, 6, 7, 8],
        ).absolute_effect
        assert t == pytest.approx(b, abs=0.15)

    def test_warns_about_staggered_adoption(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        est = twfe(panel)
        diag = est.diagnostics_of("交错处置")
        assert diag is not None and diag.status == "warn"

    def test_no_staggered_warning_for_single_cohort(self):
        cfg = StaggeredPanelConfig(
            n_units=400, n_periods=6, cohorts=(4,), cohort_weights=(1.0,),
            never_treated_share=0.5, effects=(1.0,), seed=6,
        )
        panel, _ = generate_staggered_panel(cfg)
        assert twfe(panel).diagnostics_of("交错处置") is None


class TestTWFEDecomposition:
    def test_weights_reconstruct_twfe(self):
        """恒等式：Σ weight × y 必须精确等于 TWFE 估计。"""
        panel, truth = generate_staggered_panel(HEADLINE)
        dec = twfe_decomposition(panel, truth)
        assert float((dec.weight * panel.outcome).sum()) == pytest.approx(dec.tau, rel=1e-9)
        assert dec.tau == pytest.approx(twfe(panel).absolute_effect, rel=1e-9)

    def test_weights_sum_is_zero(self):
        """双向去均值后 D̃ 的均值是 0，所以权重和也是 0。"""
        panel, truth = generate_staggered_panel(HEADLINE)
        dec = twfe_decomposition(panel, truth)
        assert dec.weight.sum() == pytest.approx(0.0, abs=1e-12)

    def test_negative_weights_exist(self):
        panel, truth = generate_staggered_panel(HEADLINE)
        dec = twfe_decomposition(panel, truth)
        assert dec.negative_post_weight_share > 0.1

    def test_weights_are_negatively_correlated_with_effect(self):
        """核心机制：权重方向和效应方向相反。"""
        panel, truth = generate_staggered_panel(HEADLINE)
        dec = twfe_decomposition(panel, truth)
        assert dec.weight_effect_correlation() < -0.3

    def test_negative_weights_carry_most_of_the_effect(self):
        panel, truth = generate_staggered_panel(HEADLINE)
        dec = twfe_decomposition(panel, truth)
        assert dec.negative_weight_effect_share > 0.3


# --------------------------------------------------------------------------- #
# Callaway-Sant'Anna
# --------------------------------------------------------------------------- #
class TestCallawaySantanna:
    def test_recovers_truth_under_staggered_adoption(self):
        panel, truth = generate_staggered_panel(HEADLINE)
        cs = callaway_santanna(panel)
        assert cs.overall.absolute_effect == pytest.approx(truth.overall_att, abs=0.25)

    def test_beats_twfe_when_twfe_flips_sign(self):
        panel, truth = generate_staggered_panel(HEADLINE)
        assert twfe(panel).absolute_effect < 0
        assert callaway_santanna(panel).overall.absolute_effect > 0

    def test_weights_are_non_negative(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        diag = callaway_santanna(panel).overall.diagnostics_of("权重")
        assert diag is not None and diag.status == "pass"

    def test_leads_are_near_zero_under_parallel_trends(self):
        """平行趋势成立时，处置前的 placebo 系数应当接近 0。"""
        cfg = StaggeredPanelConfig(
            n_units=2000, n_periods=9, cohorts=(4, 7), cohort_weights=(0.5, 0.5),
            never_treated_share=0.4, effects=(2.0, 2.0), noise_sd=0.5, seed=7,
        )
        panel, _ = generate_staggered_panel(cfg)
        leads = event_study_leads(panel)
        assert leads
        for est in leads.values():
            assert abs(est.absolute_effect) < 5 * est.std_error

    def test_never_treated_control_group(self):
        panel, truth = generate_staggered_panel(HEADLINE)
        cs = callaway_santanna(panel, control_group="never_treated")
        assert np.isfinite(cs.overall.absolute_effect)

    def test_event_study_has_post_and_pre(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        cs = callaway_santanna(panel)
        assert any(k >= 0 for k in cs.event_study)
        assert any(k < 0 for k in cs.event_study)


class TestPretrendTest:
    def test_passes_under_parallel_trends(self):
        cfg = StaggeredPanelConfig(
            n_units=1500, n_periods=9, cohorts=(4, 7), cohort_weights=(0.5, 0.5),
            never_treated_share=0.4, effects=(2.0, 2.0), noise_sd=0.5, seed=8,
        )
        panel, _ = generate_staggered_panel(cfg)
        assert pretrend_test(panel).status == "pass"

    def test_detects_cohort_specific_trends(self):
        cfg = StaggeredPanelConfig(
            n_units=1500, n_periods=9, cohorts=(4, 7), cohort_weights=(0.5, 0.5),
            never_treated_share=0.4, effects=(2.0, 2.0), noise_sd=0.5, seed=9,
            trend_violation=0.4,
        )
        panel, _ = generate_staggered_panel(cfg)
        assert pretrend_test(panel).status == "warn"

    def test_blind_to_post_treatment_divergence(self):
        """**核心断言**：处置后才分岔时，检验照样通过。"""
        cfg = StaggeredPanelConfig(
            n_units=1500, n_periods=9, cohorts=(4, 7), cohort_weights=(0.5, 0.5),
            never_treated_share=0.4, effects=(2.0, 2.0), noise_sd=0.5, seed=10,
            post_divergence=0.8,
        )
        panel, _ = generate_staggered_panel(cfg)
        assert pretrend_test(panel).status == "pass"

    def test_reports_lead_count(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        diag = pretrend_test(panel)
        assert diag.detail["n_leads"] > 0


# --------------------------------------------------------------------------- #
# 合成控制
# --------------------------------------------------------------------------- #
class TestSyntheticControl:
    def test_weights_are_a_simplex(self):
        data = generate_scm_scenario(seed=1)
        res = synthetic_control(data)
        assert res.weights.min() >= -1e-9
        assert res.weights.sum() == pytest.approx(1.0)

    def test_pre_fit_beats_naive_donor_average(self):
        data = generate_scm_scenario(seed=2)
        res = synthetic_control(data)
        naive = np.sqrt(np.mean((data.pre[0] - data.pre[1:].mean(axis=0)) ** 2))
        assert res.pre_rmse < naive

    def test_recovers_known_effect(self):
        data = generate_scm_scenario(effect=4.0, seed=3)
        res = synthetic_control(data)
        assert res.att == pytest.approx(4.0, abs=0.6)

    def test_zero_effect_gives_near_zero_gap(self):
        data = generate_scm_scenario(effect=0.0, seed=4)
        res = synthetic_control(data)
        assert abs(res.att) < 1.0

    def test_placebo_p_value_in_range(self):
        data = generate_scm_scenario(effect=3.0, seed=5)
        p = placebo_inference(data)
        assert 0 < p.p_value <= 1
        assert p.n_placebos == data.n_units - 1
        assert p.rank >= 1

    def test_placebo_ranks_true_effect_high(self):
        data = generate_scm_scenario(effect=5.0, seed=6)
        p = placebo_inference(data)
        assert p.p_value < 0.15

    def test_no_effect_gives_unremarkable_rank(self):
        data = generate_scm_scenario(effect=0.0, seed=7)
        p = placebo_inference(data)
        assert p.p_value > 0.10


# --------------------------------------------------------------------------- #
# 敏感性分析
# --------------------------------------------------------------------------- #
class TestSensitivity:
    def test_breakdown_is_finite_and_positive(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        sens = trend_sensitivity(callaway_santanna(panel), panel)
        assert np.isfinite(sens.breakdown_delta)
        assert sens.breakdown_delta > 0

    def test_larger_effect_needs_larger_violation(self):
        """效应越大，需要越大的违背才能翻转 —— 这是翻转点的基本性质。"""
        breaks = []
        for scale in (0.5, 3.0):
            cfg = StaggeredPanelConfig(
                **{**HEADLINE.__dict__, "cohort_effect_multiplier": (scale, scale),
                   "effects": (2.0, 2.0, 2.0), "seed": 11}
            )
            panel, _ = generate_staggered_panel(cfg)
            breaks.append(trend_sensitivity(callaway_santanna(panel), panel).breakdown_delta)
        assert breaks[1] > breaks[0]

    def test_scale_free_measure(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        sens = trend_sensitivity(callaway_santanna(panel), panel)
        assert np.isfinite(sens.breakdown_in_scale)
        assert sens.scale > 0

    def test_summary_warns_about_exploding_ratio(self):
        """处置前平坦时比值会爆炸 —— 摘要必须把这件事说清楚。"""
        panel, _ = generate_staggered_panel(HEADLINE)
        text = trend_sensitivity(callaway_santanna(panel), panel).summary()
        assert "翻转点" in text
        assert "不可用" in text or "比值" in text


# --------------------------------------------------------------------------- #
# 审计（仿真）
# --------------------------------------------------------------------------- #
class TestCausalAudit:
    def test_twfe_bias_and_sign_flip(self):
        c = run_staggered_estimator_comparison(n_trials=25, seed=0)
        assert c.twfe_bias < -1.0
        assert c.twfe_sign_flip_rate > 0.8
        assert abs(c.cs_bias) < 0.1
        assert c.cs_sign_flip_rate < 0.1

    def test_pretrend_test_has_a_blind_spot(self):
        a = run_pretrend_audit(n_trials=60, seed=0)
        assert a.size < 0.15
        assert a.power_trend_violation > 0.8
        assert a.blind_spot, a.summary()

    def test_scm_placebo_is_roughly_calibrated(self):
        s = run_scm_audit(n_trials=30, seed=0)
        assert abs(s.false_positive_rate - 0.05) < 0.15
        assert s.power > 0.4
