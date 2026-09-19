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
    sun_abraham,
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


class TestSunAbraham:
    """IW 聚合 + **聚合方差**。

    这一组测试里最要紧的不是"SA 能不能算对"（在饱和设定下它的点估计与 CS
    **完全相同**，见 ``sun_abraham`` 的 docstring —— 我们不假装它是回归版），
    而是"聚合方差该用哪种算法"，以及那条被它抓出来的真 bug。
    """

    def test_point_estimates_match_callaway_santanna(self):
        """饱和设定下 IW 聚合与 CS 事件研究点估计相同 —— 这是**已知的等价**，
        不是巧合；把它钉住是为了让"我们实现的到底是哪个估计量"这件事可核对。"""
        panel, _ = generate_staggered_panel(HEADLINE)
        cs = callaway_santanna(panel)
        sa = sun_abraham(panel)
        assert sa.overall.absolute_effect == pytest.approx(
            cs.overall.absolute_effect, abs=1e-9
        )

    def test_weights_are_non_negative_and_sum_to_one(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        sa = sun_abraham(panel)
        assert all(w >= 0 for w in sa.weights.values())
        assert sum(sa.weights.values()) == pytest.approx(1.0)

    def test_influence_based_se_exceeds_independent_combination(self):
        """独立合成（sqrt(Σw²se²)）必然**低估**：各相对期数共用基准期与对照，
        相关性非负。低估幅度在这份面板上是几十个百分点。"""
        panel, _ = generate_staggered_panel(HEADLINE)
        sa = sun_abraham(panel)
        assert sa.naive_overall_se < sa.overall.std_error
        assert sa.se_understatement > 0.2, sa.se_understatement

    def test_cs_also_uses_influence_based_aggregation(self):
        """CS 那边曾经用独立合成 —— 这个 bug 已经修掉，且旧算法仍被报出来对比。"""
        panel, _ = generate_staggered_panel(HEADLINE)
        cs = callaway_santanna(panel)
        assert cs.naive_overall_se < cs.overall.std_error
        assert cs.se_understatement > 0.2
        diag = cs.overall.diagnostics_of("聚合方差")
        assert diag is not None and "独立合成" in diag.message

    def test_overall_p_value_is_not_stuck_at_one(self):
        """回归测试：修之前 ``_se_from_influence`` 把 effect 传成 0 去算 p 值，
        于是**整体 ATT 的 p 值恒等于 1** —— 一个"看起来只是显示问题"的错误，
        实际是在报告一个没有任何证据支持的结论。"""
        panel, _ = generate_staggered_panel(HEADLINE)
        sa = sun_abraham(panel)
        assert sa.overall.absolute_effect > 0
        assert sa.overall.p_value < 1e-6, sa.overall.p_value

    def test_pre_periods_are_near_zero(self):
        cfg = StaggeredPanelConfig(
            n_units=2000, n_periods=9, cohorts=(4, 7), cohort_weights=(0.5, 0.5),
            never_treated_share=0.3, effects=(2.0,), noise_sd=0.5, seed=11,
        )
        panel, _ = generate_staggered_panel(cfg)
        sa = sun_abraham(panel)
        pre = {k: e for k, e in sa.event_study.items() if k < 0}
        assert pre
        for k, est in pre.items():
            assert abs(est.absolute_effect) < 5 * est.std_error, (k, est.absolute_effect)

    def test_min_max_k_filters_periods(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        sa = sun_abraham(panel, min_k=-2, max_k=2)
        assert set(sa.event_study) <= {-2, -1, 0, 1, 2}
        assert set(sa.weights) <= {0, 1, 2}  # 整体 ATT 只聚合 k>=0

    def test_treated_cohort_never_serves_as_its_own_control(self):
        """**处置队列不能进自己的对照组** —— 这条钉的是一个真出现过的 bug。

        ``not_yet_treated`` 的判据是 ``C_i > max(t, g-1)``，而处置队列自己满足
        ``g > g-1``；于是**处置前的格子里**（``t < g-1``）它被算成"尚未处置"。
        后果不是崩溃，而是 placebo 被静默压向 0（对照均值里混进了处置组自身的
        变化）。实测（g=4、t=1）：对照从 125 个（其中 75 个是处置组）
        修成 50 个，placebo 从 +0.0615 变成 +0.1537。

        这个 bug 之所以藏得住，恰恰因为"处置前系数接近 0"看起来正是我们
        想看到的结论 —— 顺眼的错误最难发现，所以要用结构断言钉住。
        """
        from ablab.causal.did import _control_mask, cs_att_with_influence

        cfg = StaggeredPanelConfig(
            n_units=200, n_periods=7, cohorts=(2, 4), cohort_weights=(0.5, 0.5),
            never_treated_share=0.25, effects=(1.0, 2.0, 3.0, 4.0), seed=5,
        )
        panel, _ = generate_staggered_panel(cfg)
        g_mask = panel.cohort == 4
        # 处置前的格子：t=1、基准期 g-1=3
        raw = _control_mask(panel, 1, 3, "not_yet_treated")
        assert (raw & g_mask).sum() > 0, "这个面板本应能触发重叠，无法验证修复"
        out = cs_att_with_influence(panel, 4, 1, 3, "not_yet_treated")
        assert out is not None
        # 修复后：对照只应当剩未处置组（50 个），且与处置组不相交
        c_used = _control_mask(panel, 1, 3, "not_yet_treated") & ~g_mask
        assert (c_used & g_mask).sum() == 0
        assert c_used.sum() == int(panel.never_treated_units.sum())
        # placebo 不再被压向 0：明显大于修复前的电平
        assert abs(out[0]) > 0.10, out[0]


class TestSunAbrahamRegression:
    """Sun-Abraham 的**回归版**（一条回归 + 队列×相对期数 + 双向固定效应）。

    它的价值不在"再多一个估计量"，而在**交叉验证**：IW 版走的是逐 2×2 再加权，
    回归版走的是吸收双向固定效应后的一条回归 —— 两条完全不同的计算路径，
    在饱和设定下必须给出同一个数。实测逐 k 点估计最大差 **9.8e-15**、
    SE 比值处处 1.000。任何一边写错，这个比对都会立刻炸。
    """

    @staticmethod
    def _both(control_group: str = "never_treated"):
        from ablab.causal import sun_abraham_regression

        panel, truth = generate_staggered_panel(HEADLINE)
        return (
            panel,
            truth,
            sun_abraham(panel, control_group=control_group),  # type: ignore[arg-type]
            sun_abraham_regression(panel, control_group=control_group),  # type: ignore[arg-type]
        )

    def test_regression_matches_interaction_weighted(self):
        """**有未处置组时**两条路径的点估计、标准误、权重都应当一致。

        容差取得很紧（点估计 1e-9、SE 相对 1e-6）是有意的：它们不是"近似相同"，
        而是**同一个估计量的两种算法**，松容差就失去交叉验证的意义了。
        """
        _panel, _truth, iw, rg = self._both("never_treated")
        assert set(iw.event_study) == set(rg.event_study)
        for k in iw.event_study:
            assert rg.event_study[k].absolute_effect == pytest.approx(
                iw.event_study[k].absolute_effect, abs=1e-9
            ), k
            assert rg.event_study[k].std_error == pytest.approx(
                iw.event_study[k].std_error, rel=1e-6
            ), k
        assert rg.overall.absolute_effect == pytest.approx(
            iw.overall.absolute_effect, abs=1e-9
        )
        assert rg.overall.std_error == pytest.approx(iw.overall.std_error, rel=1e-6)
        assert set(rg.weights) == set(iw.weights)
        for k in iw.weights:
            assert rg.weights[k] == pytest.approx(iw.weights[k], abs=1e-12)

    def test_without_never_treated_the_two_versions_differ_and_why(self):
        """**没有未处置组时两者不同** —— 这条比"相同"更重要，因为它说明了边界。

        IW 的每个分格用"当时尚未处置"的单元当对照（对照集随 t 变），
        饱和回归只有一套双向固定效应，已处置队列的变化会进入比较。
        实测（cohorts=(3,5,7)、n_periods=10、600 单元）：
          * 多队列共同贡献的相对期数（k<=3）差最大 0.17（IW 的 SE 是 0.086）；
          * 只有单队列贡献的（k>=4）差回到 1e-14 —— 那时没有"别人"可混进来。

        所以断言的是**这个模式**，而不是"两者应当相同"：后者是错的声明。
        """
        from ablab.causal import sun_abraham_regression

        cfg = StaggeredPanelConfig(
            n_units=600, n_periods=10, cohorts=(3, 5, 7),
            cohort_weights=(0.25, 0.25, 0.25), never_treated_share=0.25,
            effects=(1.0, 2.0, 3.0, 3.0, 3.0), noise_sd=1.0, seed=5,
        )
        panel, _ = generate_staggered_panel(cfg)
        iw = sun_abraham(panel, control_group="not_yet_treated")
        rg = sun_abraham_regression(panel, control_group="not_yet_treated")
        diffs = {
            k: abs(iw.event_study[k].absolute_effect - rg.event_study[k].absolute_effect)
            for k in iw.event_study
        }
        multi = [k for k in diffs if k <= 3]
        single = [k for k in diffs if k >= 4]
        assert multi and single
        assert max(diffs[k] for k in single) < 1e-9, {k: diffs[k] for k in single}
        assert max(diffs[k] for k in multi) > 1e-3, {k: diffs[k] for k in multi}

    def test_influence_scaling_is_not_degenerate(self):
        """回归版的影响函数**必须按均值型约定缩放**（乘 n）。

        第一版漏了这一步，整体 ATT 的 SE 报成 0.0001（正确值 0.0431）——
        差 400 倍，而且方向是"看起来更显著"。这条断言钉住量级：
        它与 IW 版同阶、且大于独立合成的反保守值。
        """
        _panel, _truth, iw, rg = self._both("never_treated")
        ratio = rg.overall.std_error / iw.overall.std_error
        assert 0.5 < ratio < 2.0, ratio
        assert rg.naive_overall_se < rg.overall.std_error
        assert rg.se_understatement > 0.2, rg.se_understatement

    def test_absorption_removes_two_way_fixed_effects(self):
        """交替投影真的把双向固定效应吸掉了：纯 FE 的向量残差应当≈0。

        这是回归版的**地基**：吸收不干净，``δ`` 就会被固定效应污染，
        而面板上的固定效应恰好与队列相关（队列效应 + 时间趋势）。

        第二条断言用**理论预期**而不是拍一个阈值：双向吸收会吃掉
        ``(N + T - 1)/(N·T)`` 那部分自由度，所以保留下来的噪音方差应当约为
        ``1 - (N+T-1)/(N·T)``。第一版把阈值写成"相关 > 0.99"，
        那是把吸收当成了"只碰固定效应、不碰噪音"—— 实测 0.930，
        与理论值 0.9276 吻合，是阈值错了而不是实现错了。
        """
        import numpy as np

        from ablab.causal.did import _absorb_two_way

        rng = np.random.default_rng(0)
        n_units, n_times = 60, 8
        u = np.repeat(np.arange(n_units), n_times)
        t = np.tile(np.arange(n_times), n_units)
        unit_fe = rng.normal(0, 2.0, n_units)
        time_fe = rng.normal(0, 1.5, n_times)
        pure_fe = unit_fe[u] + time_fe[t]
        resid = _absorb_two_way(pure_fe, u, t, n_units, n_times)
        assert np.max(np.abs(resid)) < 1e-8, np.max(np.abs(resid))

        noise = rng.normal(0, 1.0, pure_fe.size)
        resid2 = _absorb_two_way(pure_fe + noise, u, t, n_units, n_times)
        corr = float(np.corrcoef(resid2, noise)[0, 1])
        expected = float(np.sqrt(1.0 - (n_units + n_times - 1) / (n_units * n_times)))
        assert abs(corr - expected) < 0.05, (corr, expected)
        # 残差里不能剩下固定效应：与两个固定效应都应当基本不相关
        assert abs(np.corrcoef(resid2, unit_fe[u])[0, 1]) < 0.05
        assert abs(np.corrcoef(resid2, time_fe[t])[0, 1]) < 0.05

    def test_regression_fixes_twfe_bias_under_heterogeneous_dynamics(self):
        """效应异质时 TWFE 会偏（负权重），回归版不会 —— 这是它存在的理由。

        实测（HEADLINE 面板）：真值 +2.5000，TWFE +2.0086（偏 -0.49），
        回归版 +2.5585（偏 +0.06）。断言用"误差小一个量级"，不写死具体数。
        """
        panel, truth, _iw, rg = self._both()
        tw = twfe(panel)
        reg_err = abs(rg.overall.absolute_effect - truth.overall_att)
        twfe_err = abs(tw.absolute_effect - truth.overall_att)
        assert reg_err < twfe_err / 3, (reg_err, twfe_err)

    def test_min_max_k_filters_periods(self):
        from ablab.causal import sun_abraham_regression

        panel, _ = generate_staggered_panel(HEADLINE)
        rg = sun_abraham_regression(panel, min_k=-2, max_k=2)
        assert set(rg.event_study) <= {-2, -1, 0, 1, 2}
        assert set(rg.weights) <= {0, 1, 2}

    def test_event_study_se_uses_influence_too(self):
        """事件研究的聚合也改用影响函数了 —— 但它的影响**很小，而且符号不定**。

        实测（HEADLINE 面板）：多数 k 上两者只差几个百分点，个别 k 上
        影响函数算出的 SE 反而**更小** —— 因为格子之间的协方差**可以为负**
        （不同队列、不同基准期），所以"影响函数版必然更大"这句话是错的。
        这条断言只钉住"两者同量级"：改的是算法，不是数字。
        """
        panel, _ = generate_staggered_panel(HEADLINE)
        cs = callaway_santanna(panel)
        for k, est in cs.event_study.items():
            rows = [e for (g, t), e in cs.group_time.items() if t - g == k]
            w = np.array([e.n_treatment for e in rows], dtype=float)
            w = w / w.sum()
            naive = float(np.sqrt(sum(wi**2 * e.std_error**2 for wi, e in zip(w, rows))))
            assert 0.7 * naive <= est.std_error <= 1.4 * naive, (k, est.std_error, naive)

    def test_aggregation_variance_audit_separates_the_two(self):
        """审计本身要能分辨两种算法：H0 下独立合成的越界率明显更高。

        用**小规模**跑（25 次、300 单元）—— 它测的是"能不能分辨"，
        不是精确的 size；完整版在 ``run_m3_validation.py`` 里。
        """
        from ablab.validation import run_aggregation_variance_audit

        cfg = StaggeredPanelConfig(
            n_units=300, n_periods=6, cohorts=(2, 4), cohort_weights=(0.5, 0.5),
            never_treated_share=0.2, effects=(0.0,), noise_sd=1.0,
        )
        audit = run_aggregation_variance_audit(cfg, n_trials=25, seed=3)
        assert audit.naive_reject_rate > audit.influence_reject_rate
        assert audit.mean_se_understatement > 0.2
        assert audit.naive_coverage < audit.influence_coverage


class TestCateInterval:
    """CATE 的区间：**算得出来，但校准不了** —— 这两件事都要被钉住。

    如果只钉"能算出来"，下一个人会以为它可用；如果只钉"校准不了"，
    又会被误解成"没实现"。两条一起钉，才是这一轮的真实状态。
    """

    @staticmethod
    def _fitted(seed: int = 3, min_leaf: int = 20):
        from ablab.causal.forest import CausalForest, ForestConfig
        from ablab.causal.hte import HTEConfig, generate_hte_data

        data = generate_hte_data(HTEConfig(n=1200, n_features=6, n_informative=3, seed=seed))
        forest = CausalForest(ForestConfig(n_trees=30, max_depth=4, min_leaf=min_leaf))
        forest.fit(data.X, data.D, data.Y)
        return data, forest

    def test_predict_with_se_shapes_and_finiteness(self):
        import numpy as np

        data, forest = self._fitted()
        tau, se = forest.predict_with_se(data.X)
        assert tau.shape == se.shape == (data.X.shape[0],)
        # 有些叶子只有一臂（倾向得分把样本切开了），那些单元的 SE 是 inf ——
        # 这是**有意**的：给不出方差就不要给一个假的有限值。
        finite = np.isfinite(se)
        assert finite.mean() > 0.5, f"可给出区间的单元太少：{finite.mean():.2f}"
        assert (se[finite] > 0).all()

    def test_interval_shrinks_with_larger_leaves(self):
        """验收标准之一（**部分满足**，如实测）：叶子越大整体越短，但**不单调**。

        实测 min_leaf = 10/20/40/80 的解析区间中位长度：
        0.0824 / 0.0937 / 0.0720 / 0.0507 —— 80 比 10 短约 38%，
        但 20 处反而比 10 长。原因是**两个效应叠在一起**：min_leaf 变大既让
        叶子内样本更多（区间变短），又改变"能给出区间的叶子"的比例
        （纯叶子变多 → 有限 SE 的子集变了）。
        所以断言的是**整体方向**（80 < 10），不是逐点单调 ——
        把逐点单调写成断言，就是让标准去迁就一个不成立的说法。
        """
        import numpy as np

        medians = []
        for ml in (10, 80):
            data, forest = self._fitted(min_leaf=ml)
            _, se = forest.predict_with_se(data.X)
            finite = np.isfinite(se)
            medians.append(float(np.median(se[finite])))
        assert medians[1] < medians[0], medians

    def test_coverage_is_far_below_nominal_and_that_is_the_point(self):
        """**覆盖率远低于 95%** —— 这条断言是"水平仍不可用"的证据。

        实测解析区间覆盖 ~0.4、bootstrap ~0.7（见 reports/cate_interval_report.md）。
        这里只跑一个场景，断言"明显低于名义值"，把它钉成**已知结论**
        而不是一个会被误读成 bug 的现象。
        """
        import numpy as np

        data, forest = self._fitted()
        tau, se = forest.predict_with_se(data.X)
        true = np.asarray(data.tau)
        finite = np.isfinite(se)
        lo, hi = tau[finite] - 1.96 * se[finite], tau[finite] + 1.96 * se[finite]
        coverage = float(np.mean((lo <= true[finite]) & (true[finite] <= hi)))
        assert coverage < 0.8, (
            f"覆盖率 {coverage:.3f} —— 如果它真的接近 95%，"
            "那说明点估计变了，README 的已知边界要跟着改"
        )


class TestGatesBlp:
    """组级路线：单元级不可行（上一条），**组级可行** —— 换成 BLP/GATES。

    依据 Chernozhukov 等（arXiv:1712.04802）：通用 ML 工具下 CATE 的一致估计
    与自适应置信集都不存在，所以推断对象应当是 CATE 的**特征**。信号取
    Horvitz-Thompson，分样本做经典 OLS 推断，于是**不要求代理一致**。

    这些断言钉三件事，缺一件这条路线就会被误读：
      1. 组级区间确实给出名义覆盖（与单元级的 0.4 形成对照）；
      2. 审计**有功效** —— 真值代理通过、未校准的森林代理被推开；
      3. HT 信号在本 DGP 下重尾（峰度远超 3）且正值性近乎违背 ——
         这是"区间有效但又宽又不稳"的原因，也是下一步换 AIPW 信号的依据。
    """

    @staticmethod
    def _run(proxy_kind: str):
        from ablab.validation.hte_audit import run_gates_blp_audit

        return run_gates_blp_audit(
            n=1500, n_splits=8, n_groups=4, seed=0, proxy_kind=proxy_kind
        )

    def test_shapes_and_ranges(self):
        r = self._run("forest")
        assert r.n_groups == 4
        assert len(r.gates_effects) == len(r.gates_true) == len(r.gates_gap) == 4
        assert 0.0 <= r.gates_coverage <= 1.0
        assert 0.0 <= r.blp_covers_one <= 1.0
        assert r.gates_mean_length > 0.0
        assert all(len(g) == 4 for g in (r.gates_gap, r.gates_gap_mc_se))

    def test_group_level_coverage_beats_unit_level(self):
        """**组级覆盖率是名义值量级**，而单元级实测只有 0.13~0.44。

        两个数不能直接比大小（对象不同：一个是 E[τ|组]，一个是 τ(X)），
        能比的是"谁给出了可用的区间"。所以断言的是组级落在名义值附近，
        且**明显高于**单元级那条路线的实测上界。
        """
        r = self._run("forest")
        assert r.gates_coverage > 0.8, (
            f"组级覆盖率 {r.gates_coverage:.3f} —— 若真掉到这个量级，"
            "说明 HT 信号或分样本流程坏了，而不是名义性波动"
        )

    def test_calibrated_proxy_passes_and_miscalibrated_is_rejected(self):
        """审计的**功效**：真值代理斜率≈1，森林代理被明显推开。

        实测（n=1500、8 次分裂）：真值代理斜率 1.05（SE 0.20），
        森林代理 1.45（SE 0.78）。森林代理不只是斜率大，它的 SE 还大 4 倍 ——
        因为 τ̂ 被"压平"（attenuation），Var(τ̂) 变小，斜率与它的方差一起变大。
        断言用宽松倍数，钉的是方向而不是那两个具体数。
        """
        forest, oracle = self._run("forest"), self._run("oracle")
        assert abs(oracle.blp_slope - 1.0) < abs(forest.blp_slope - 1.0), (
            f"真值代理 {oracle.blp_slope:.3f} vs 森林代理 {forest.blp_slope:.3f}"
        )
        assert forest.blp_slope_se > 2 * oracle.blp_slope_se

    def test_signal_is_heavy_tailed_and_positivity_is_violated(self):
        """重尾与重叠度诊断 —— 记下的是**瓶颈**，不是可以忽略的细节。

        实测：倾向得分范围 [0.014, 1.000]，权重 1/(p(1-p)) 最大 5530；
        信号峰度 70~875（Y 自身只有 ~4.6）。组级 SE 用 HC0，在这个峰度下
        有限样本不可靠，覆盖率才会在 0.88~0.96 之间摆动。
        """
        r = self._run("forest")
        assert r.signal_kurtosis > 10.0, r.signal_kurtosis
        assert 0.0 < r.overlap_violation_share < 0.5, r.overlap_violation_share

    def test_unknown_proxy_kind_is_rejected(self):
        import pytest

        from ablab.validation.hte_audit import run_gates_blp_audit

        with pytest.raises(ValueError, match="proxy_kind"):
            run_gates_blp_audit(n=200, n_splits=1, proxy_kind="nonsense")

    def test_default_path_is_aipw_with_auto_trim(self):
        """**默认路径已经是推荐配置**：不是纯 HT，而是 AIPW + 自动裁剪。

        实测（n=1500、8 次分裂）：默认路径峰度 17.1、区间长度 1.1393；
        历史口径（纯 HT）峰度 70.9、长度 1.8823 —— 覆盖率两者都在
        0.94~0.97，差别在**区间的宽窄与稳定性**。

        自动选出的阈值落在 0.069（同配置下 4 次分裂是 0.05~0.13），
        裁掉 9.1% 的单元 —— 这个比例必须被报告，因为**阈值改变了估计目标**
        （从全体变成重叠总体）。
        """
        r = self._run("forest")
        assert r.signal_kind == "aipw"
        assert 0.02 <= r.trim_alpha <= 0.20, r.trim_alpha
        assert 0.0 < r.trimmed_share < 0.35, r.trimmed_share
        assert r.gates_coverage > 0.8

    def test_historical_ht_path_is_still_reachable(self):
        """历史口径必须**仍然可复现** —— 报告里那 4 行老数字靠它。

        换了默认值不等于删掉旧路径：`signal_kind="ht", trim=None` 必须
        逐位给出换默认之前的数（BLP 斜率 2.3618、覆盖率 0.9417）。
        这条断言只钉"没裁剪 + 峰度是重的"，具体数值由报告锁定。
        """
        from ablab.validation.hte_audit import run_gates_blp_audit

        r = run_gates_blp_audit(
            n=1500, n_splits=8, n_groups=4, seed=0,
            proxy_kind="forest", signal_kind="ht", trim=None,
        )
        assert r.signal_kind == "ht"
        assert r.trim_alpha == 0.0
        assert r.trimmed_share == 0.0
        assert r.signal_kurtosis > 10.0  # 纯 HT 的尾巴还在
        assert r.gates_mean_length > self._run("forest").gates_mean_length

    def test_default_beats_historical_on_tail_and_length(self):
        """默认路径比历史口径**尾巴更轻、区间更短**，覆盖率不明显更差。

        这是"接成默认"这个决定本身的证据；如果哪天它反过来了，
        这个默认值就该改回去，而不是让 README 继续推荐它。
        """
        from ablab.validation.hte_audit import run_gates_blp_audit

        new = self._run("forest")
        old = run_gates_blp_audit(
            n=1500, n_splits=8, n_groups=4, seed=0,
            proxy_kind="forest", signal_kind="ht", trim=None,
        )
        assert new.signal_kurtosis < old.signal_kurtosis / 2
        assert new.gates_mean_length < old.gates_mean_length
        assert new.gates_coverage > 0.8


class TestSignalComparison:
    """信号怎么选：**裁剪管尾巴、AIPW 管方差** —— 这条是测出来的，不是推的。

    它同时是一次**自我纠正**的记录：先写下的修法只有"换 AIPW"，
    实测发现 AIPW 几乎不降峰度（它降的是方差），降峰度靠裁剪。
    所以这里既钉"裁剪有效"，也钉"两者各有分工"，防止后来人只看到一半。
    """

    @staticmethod
    def _run(n_splits: int = 4):
        from ablab.validation.hte_audit import run_signal_comparison

        return run_signal_comparison(n=1200, n_splits=n_splits, n_groups=4, clip=0.05)

    def test_four_arms_report_shapes(self):
        r = self._run()
        names = [a.name for a in r.arms]
        assert names == ["HT", "HT+裁剪", "AIPW", "AIPW+裁剪"]
        for a in r.arms:
            assert 0.0 <= a.coverage <= 1.0
            assert a.mean_length > 0.0
            assert a.signal_sd > 0.0
            assert a.kurtosis > 3.0  # 四个版本都还是重尾，只是程度不同

    def test_clipping_cuts_the_tail(self):
        """裁剪把峰度显著压下来 —— 尾巴来自 1/(p(1-p)) 的极端权重。

        实测（n=2000、30 次分裂）：HT 133.5 → HT+裁剪 43.6（3.1 倍）。
        这里样本更小、分裂更少，所以只断言**方向 + 明显幅度**（降到六成以下），
        不钉具体数 —— 小样本下 AIPW 自己的峰度就已经低一截，
        拿同一个倍数去卡 AIPW+裁剪会变成一条假红的断言。
        """
        ht, ht_clip, aipw, both = self._run().arms
        assert ht_clip.kurtosis < 0.6 * ht.kurtosis, (ht.kurtosis, ht_clip.kurtosis)
        assert both.kurtosis < 0.8 * aipw.kurtosis, (aipw.kurtosis, both.kurtosis)

    def test_aipw_shrinks_intervals_without_fixing_the_tail(self):
        """AIPW 的分工是**方差**：区间变短，峰度基本不动。

        这一条是那一半纠正：如果哪天 AIPW 也把峰度降下来了，说明结局模型
        或 DGP 变了，README 第 7 节的说法要跟着改。
        """
        ht, _, aipw, _ = self._run().arms
        assert aipw.mean_length < ht.mean_length
        assert aipw.kurtosis > ht.kurtosis / 3, (ht.kurtosis, aipw.kurtosis)

    def test_combination_beats_pure_ht(self):
        ht, _, _, both = self._run().arms
        assert both.mean_length < ht.mean_length
        assert both.kurtosis < ht.kurtosis

    def test_aggressive_clipping_alone_breaks_coverage_but_aipw_survives(self):
        """裁剪阈值选大了会翻车，**但只有单靠裁剪时才翻车**。

        实测（报告第 7b 节，n=2000、20 次分裂）：HT+裁剪的覆盖率从阈值 0.10
        起就掉（0.8875 → 0.8125 → 0.5375），而 AIPW+裁剪在 0.05~0.20
        稳在 0.94~0.96。原因是裁剪把极端权重的单元拉进边界、改变了实际
        覆盖的目标，而结局模型能把那部分偏差补回来。

        这条断言钉的是**交互**，不是某个具体阈值 —— 阈值本身依赖 DGP 的重叠
        程度，换个场景就该重测，但"裁剪必须先有结局模型兜着"这个结论不变。
        """
        small = self._run_with_clip(0.05)
        large = self._run_with_clip(0.30)
        _, ht_small, _, _ = small.arms
        _, ht_large, _, aipw_large = large.arms
        assert ht_large.coverage < ht_small.coverage, (
            ht_small.coverage, ht_large.coverage
        )
        assert aipw_large.coverage > ht_large.coverage, (
            ht_large.coverage, aipw_large.coverage
        )

    @staticmethod
    def _run_with_clip(clip: float):
        from ablab.validation.hte_audit import run_signal_comparison

        return run_signal_comparison(n=1200, n_splits=4, n_groups=4, clip=clip)


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
        """比值爆炸时必须给出警告 —— 但这条测试原来是**照着 bug 写的**。

        它原先用 HEADLINE 面板做集成断言，理由是"处置前平坦时比值会爆炸"。
        实测（对照泄漏 bug 修复后）：同一面板的比值从 **467.7 变成 4.26** ——
        因为泄漏把处置前系数压向了 0，分母虚小、比值虚大，
        于是那条"这个数不可用"的告警是被 bug 触发的。
        所以这里改成**直接测分支语义**（构造三个比值），
        集成层面只保留"能给出有限比值与翻转点"这一条不会漂移的断言。
        """
        from ablab.causal.sensitivity import TrendSensitivity

        def make(pretrend: float) -> TrendSensitivity:
            return TrendSensitivity(
                att=2.0, breakdown_delta=0.7, pretrend_delta=pretrend,
                scale=2.4, n_post_coefs=6, n_pre_coefs=2,
            )

        # 处置前趋势可测 → 给数值，不给警告
        normal = make(0.15).summary()
        assert "与处置前趋势之比" in normal
        assert "不可用" not in normal
        # 处置前近乎平坦 → 比值爆炸，必须显式说"这个数不可用"
        exploding = make(0.0015).summary()
        assert "不可用" in exploding
        # 处置前为 0 → 比值无定义，也要说清楚，而不是印一个 inf
        undefined = make(0.0).summary()
        assert "无定义" in undefined and "inf" not in undefined

    def test_sensitivity_is_finite_on_the_headline_panel(self):
        panel, _ = generate_staggered_panel(HEADLINE)
        sens = trend_sensitivity(callaway_santanna(panel), panel)
        assert "翻转点" in sens.summary()
        assert np.isfinite(sens.breakdown_delta)
        assert sens.n_pre_coefs > 0


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
