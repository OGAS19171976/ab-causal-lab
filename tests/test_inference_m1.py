"""M1 三个新方法的单元测试：可加统计量、CUPED、比值指标、聚类稳健。"""

import numpy as np
import pytest
from scipy import stats

from ablab.inference import (
    AggregateStats,
    cluster_level_ttest,
    cluster_robust_ttest,
    cuped_estimate,
    cuped_ttest,
    estimate_icc,
    fit_cuped,
    naive_unit_ratio_ttest,
    ratio_delta_method,
    t_inference,
    welch_inference,
    welch_inference_from_components,
    welch_ttest,
    welch_ttest_from_stats,
    wild_cluster_bootstrap,
)


# --------------------------------------------------------------------------- #
# AggregateStats
# --------------------------------------------------------------------------- #
class TestAggregateStats:
    def test_from_arrays_matches_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 2, 500)
        y = 0.6 * x + rng.normal(0, 3, 500)
        s = AggregateStats.from_arrays(y, x)

        assert s.mean_x == pytest.approx(x.mean())
        assert s.mean_y == pytest.approx(y.mean())
        assert s.var_x == pytest.approx(x.var(ddof=1))
        assert s.var_y == pytest.approx(y.var(ddof=1))
        assert s.cov_xy == pytest.approx(np.cov(x, y, ddof=1)[0, 1])
        assert s.corr_xy == pytest.approx(np.corrcoef(x, y)[0, 1])

    def test_merge_is_additive(self):
        """merge 的可加性是数仓分层成立的前提 —— 必须与拼接后一次算完全一致。"""
        rng = np.random.default_rng(1)
        x1, y1 = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
        x2, y2 = rng.normal(0, 1, 700), rng.normal(0, 1, 700)

        merged = AggregateStats.from_arrays(y1, x1).merge(AggregateStats.from_arrays(y2, x2))
        whole = AggregateStats.from_arrays(np.r_[y1, y2], np.r_[x1, x2])

        assert merged.n == whole.n
        assert merged.mean_x == pytest.approx(whole.mean_x, rel=1e-12)
        assert merged.var_x == pytest.approx(whole.var_x, rel=1e-12)
        assert merged.cov_xy == pytest.approx(whole.cov_xy, rel=1e-12)

    def test_merge_three_way(self):
        rng = np.random.default_rng(2)
        parts = [
            AggregateStats.from_arrays(rng.normal(0, 1, 100), rng.normal(0, 1, 100))
            for _ in range(3)
        ]
        combined = parts[0].merge(*parts[1:])
        assert combined.n == 300

    def test_from_sums_roundtrip(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
        s = AggregateStats.from_arrays(y, x)
        rebuilt = AggregateStats.from_sums(
            s.n, s.sum_x, s.sum_y, s.sum_xx, s.sum_yy, s.sum_xy
        )
        assert rebuilt.var_y == pytest.approx(s.var_y, rel=1e-12)

    def test_nan_dropped_pairwise(self):
        """任一列为 NaN 的单元整条丢弃，避免两列样本量不一致。"""
        y = np.array([1.0, 2.0, np.nan, 4.0])
        x = np.array([1.0, np.nan, 3.0, 4.0])
        s = AggregateStats.from_arrays(y, x)
        assert s.n == 2  # 只剩第 1、4 个

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            AggregateStats.from_arrays([1.0, 2.0], [1.0])

    def test_outcomes_only(self):
        s = AggregateStats.from_arrays([1.0, 2.0, 3.0])
        assert s.mean_x == 0.0
        assert s.var_y == pytest.approx(1.0)

    def test_ratio(self):
        s = AggregateStats.from_arrays([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
        assert s.ratio == pytest.approx(2.0)


class TestRatioVariance:
    def test_matches_manual_delta_method(self):
        rng = np.random.default_rng(4)
        x = rng.poisson(10, 5000).astype(float) + 1
        y = rng.binomial(x.astype(int), 0.2).astype(float)
        s = AggregateStats.from_arrays(y, x)

        r = s.ratio
        residual_var = s.var_y - 2 * r * s.cov_xy + r * r * s.var_x
        expected = residual_var / (s.n * s.mean_x**2)
        assert s.ratio_variance() == pytest.approx(expected, rel=1e-12)

    def test_non_negative(self):
        rng = np.random.default_rng(5)
        x = rng.poisson(5, 2000).astype(float) + 1
        y = rng.binomial(x.astype(int), 0.3).astype(float)
        assert AggregateStats.from_arrays(y, x).ratio_variance() >= 0


# --------------------------------------------------------------------------- #
# welch_inference 三层结构
# --------------------------------------------------------------------------- #
class TestWelchInference:
    def test_t_inference_matches_scipy(self):
        res = t_inference(1.0, se=0.5, degrees_of_freedom=100)
        assert res.p_value == pytest.approx(2 * stats.t.sf(2.0, 100), rel=1e-12)
        lo, hi = res.interval(1.0)
        assert lo < 1.0 < hi

    def test_welch_matches_detail_entry(self):
        rng = np.random.default_rng(6)
        t = rng.normal(1, 2, 400)
        c = rng.normal(0, 3, 500)
        detail = welch_ttest(t, c)
        summary = welch_ttest_from_stats(
            n_treatment=400, mean_treatment=float(t.mean()), var_treatment=float(t.var(ddof=1)),
            n_control=500, mean_control=float(c.mean()), var_control=float(c.var(ddof=1)),
        )
        assert summary.p_value == pytest.approx(detail.p_value, rel=1e-12)
        assert summary.std_error == pytest.approx(detail.std_error, rel=1e-12)

    def test_from_components_matches_welch(self):
        rng = np.random.default_rng(7)
        t = rng.normal(0, 2, 300)
        c = rng.normal(0, 3, 400)
        a = welch_inference(
            0.5, n_treatment=300, var_treatment=float(t.var(ddof=1)),
            n_control=400, var_control=float(c.var(ddof=1)),
        )
        b = welch_inference_from_components(
            0.5,
            var_treatment=float(t.var(ddof=1)) / 300,
            df_treatment=299,
            var_control=float(c.var(ddof=1)) / 400,
            df_control=399,
        )
        assert a.se == pytest.approx(b.se, rel=1e-12)
        assert a.degrees_of_freedom == pytest.approx(b.degrees_of_freedom, rel=1e-10)

    def test_zero_se_degenerates(self):
        res = t_inference(0.0, se=0.0, degrees_of_freedom=10)
        assert res.degenerate and res.p_value == 1.0

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="方差不能为负"):
            welch_inference(0.0, n_treatment=10, var_treatment=-1.0, n_control=10, var_control=1.0)


# --------------------------------------------------------------------------- #
# CUPED
# --------------------------------------------------------------------------- #
class TestCuped:
    def make_data(self, *, n=20_000, rho=0.7, effect=0.0, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(100, 30, n)
        y = 50 + rho * (x - 100) + rng.normal(0, 30 * np.sqrt(1 - rho**2), n)
        treated = rng.random(n) < 0.5
        y = y + effect * treated
        return x, y, treated

    def test_theta_recovers_regression_coefficient(self):
        x, y, _ = self.make_data(rho=0.7)
        fit = fit_cuped(AggregateStats.from_arrays(y, x))
        # DGP 里 y 对 x 的斜率就是 rho
        assert fit.theta == pytest.approx(0.7, abs=0.03)

    def test_variance_reduction_equals_rho_squared(self):
        """方差缩减必须精确等于 rho^2 —— 这是 CUPED 的核心恒等式。"""
        x, y, treated = self.make_data(rho=0.7, seed=1)
        t = AggregateStats.from_arrays(y[treated], x[treated])
        c = AggregateStats.from_arrays(y[~treated], x[~treated])

        naive = welch_ttest_from_stats(
            n_treatment=t.n, mean_treatment=t.mean_y, var_treatment=t.var_y,
            n_control=c.n, mean_control=c.mean_y, var_control=c.var_y,
        )
        cuped, fit = cuped_estimate(t, c)

        ratio = (cuped.std_error / naive.std_error) ** 2
        assert ratio == pytest.approx(1 - fit.correlation**2, abs=0.02)
        assert fit.variance_reduction == pytest.approx(fit.correlation**2, rel=1e-12)

    def test_three_quantities_are_distinct(self):
        """方差缩减 / 残余方差 / 标准误降幅是三个不同的量，不能互相混用。"""
        x, y, _ = self.make_data(rho=0.7, seed=2)
        fit = fit_cuped(AggregateStats.from_arrays(y, x))

        assert fit.variance_reduction + fit.remaining_variance == pytest.approx(1.0)
        assert fit.se_shrinkage == pytest.approx(1 - np.sqrt(fit.remaining_variance))
        # rho=0.7 时：方差缩减≈0.49，残余≈0.51，SE 降幅≈0.29
        assert fit.variance_reduction == pytest.approx(0.49, abs=0.03)
        assert fit.se_shrinkage == pytest.approx(0.286, abs=0.02)
        assert not np.isclose(fit.se_shrinkage, fit.variance_reduction, atol=0.1)

    def test_effective_sample_multiplier(self):
        x, y, _ = self.make_data(rho=0.7, seed=3)
        fit = fit_cuped(AggregateStats.from_arrays(y, x))
        assert fit.effective_sample_multiplier == pytest.approx(1 / fit.remaining_variance)

    def test_removes_covariate_imbalance_bias(self):
        """CUPED 的核心承诺：扣掉前置协变量失衡带来的固定偏置。"""
        n = 40_000
        rng = np.random.default_rng(4)
        x = rng.normal(100, 30, n)
        treated = rng.random(n) < 0.5
        # 造一个明显的失衡：处理组的前置指标整体偏高
        x = x + 3.0 * treated

        beta = 0.7
        y = 50 + beta * (x - 100) + rng.normal(0, 30 * np.sqrt(1 - beta**2), n)

        t = AggregateStats.from_arrays(y[treated], x[treated])
        c = AggregateStats.from_arrays(y[~treated], x[~treated])

        naive = welch_ttest_from_stats(
            n_treatment=t.n, mean_treatment=t.mean_y, var_treatment=t.var_y,
            n_control=c.n, mean_control=c.mean_y, var_control=c.var_y,
        )
        cuped, _ = cuped_estimate(t, c)

        # 真效应为 0，naive 被失衡推到约 beta*3 = 2.1
        assert abs(naive.absolute_effect) > 1.5
        assert abs(cuped.absolute_effect) < 0.3

    def test_control_only_theta(self):
        x, y, treated = self.make_data(seed=5)
        t = AggregateStats.from_arrays(y[treated], x[treated])
        c = AggregateStats.from_arrays(y[~treated], x[~treated])
        pooled_fit = fit_cuped(t.merge(c))
        control_fit = fit_cuped(t.merge(c), c, theta_source="control")
        assert control_fit.theta == pytest.approx(c.cov_xy / c.var_x, rel=1e-12)
        assert abs(control_fit.theta - pooled_fit.theta) < 0.05

    def test_control_only_requires_control(self):
        x, y, _ = self.make_data()
        with pytest.raises(ValueError, match="需要传入 control"):
            fit_cuped(AggregateStats.from_arrays(y, x), theta_source="control")

    def test_constant_covariate_raises(self):
        with pytest.raises(ValueError, match="没有方差"):
            fit_cuped(AggregateStats.from_arrays([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))

    def test_weak_correlation_warns(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, 5000)
        y = rng.normal(0, 1, 5000)  # 与 x 无关
        treated = rng.random(5000) < 0.5
        est, _ = cuped_estimate(
            AggregateStats.from_arrays(y[treated], x[treated]),
            AggregateStats.from_arrays(y[~treated], x[~treated]),
        )
        diag = est.diagnostics_of("CUPED 收益")
        assert diag is not None and diag.status == "warn"

    def test_balance_diagnostic_present(self):
        x, y, treated = self.make_data(seed=7)
        est, _ = cuped_estimate(
            AggregateStats.from_arrays(y[treated], x[treated]),
            AggregateStats.from_arrays(y[~treated], x[~treated]),
        )
        assert est.diagnostics_of("协变量平衡") is not None

    def test_balance_diagnostic_flags_imbalance(self):
        n = 20_000
        x = np.r_[np.zeros(n // 2), np.ones(n // 2) * 10.0] + np.random.default_rng(8).normal(0, 1, n)
        treated = np.r_[np.ones(n // 2, dtype=bool), np.zeros(n // 2, dtype=bool)]
        y = x + np.random.default_rng(9).normal(0, 1, n)
        est, _ = cuped_estimate(
            AggregateStats.from_arrays(y[treated], x[treated]),
            AggregateStats.from_arrays(y[~treated], x[~treated]),
        )
        assert est.diagnostics_of("协变量平衡").status == "warn"

    def test_detail_entry_matches_stats_entry(self):
        x, y, treated = self.make_data(seed=10)
        a, fit_a = cuped_ttest(x[treated], y[treated], x[~treated], y[~treated])
        b, fit_b = cuped_estimate(
            AggregateStats.from_arrays(y[treated], x[treated]),
            AggregateStats.from_arrays(y[~treated], x[~treated]),
        )
        assert a.absolute_effect == pytest.approx(b.absolute_effect, abs=1e-12)
        assert a.std_error == pytest.approx(b.std_error, abs=1e-12)
        assert fit_a.theta == pytest.approx(fit_b.theta, rel=1e-12)

    def test_cuped_is_unbiased_with_real_effect(self):
        """CUPED 的置信区间应覆盖真实效应（用自身标准误作为尺度，避免写死容差）。"""
        x, y, treated = self.make_data(rho=0.7, effect=2.0, seed=11, n=50_000)
        est, _ = cuped_estimate(
            AggregateStats.from_arrays(y[treated], x[treated]),
            AggregateStats.from_arrays(y[~treated], x[~treated]),
        )
        assert est.ci_low <= 2.0 <= est.ci_high
        assert abs(est.absolute_effect - 2.0) < 3 * est.std_error


# --------------------------------------------------------------------------- #
# 比值指标
# --------------------------------------------------------------------------- #
class TestRatioMetric:
    def make_ratio_data(self, *, n=20_000, seed=0):
        rng = np.random.default_rng(seed)
        views = rng.poisson(12, n) + 1
        treated = rng.random(n) < 0.5
        clicks = rng.binomial(views, 0.15).astype(float)
        return views.astype(float), clicks, treated

    def test_delta_method_estimates_pooled_ratio(self):
        views, clicks, treated = self.make_ratio_data()
        est = ratio_delta_method(
            AggregateStats.from_arrays(clicks[treated], views[treated]),
            AggregateStats.from_arrays(clicks[~treated], views[~treated]),
        )
        expected = clicks[treated].sum() / views[treated].sum() - (
            clicks[~treated].sum() / views[~treated].sum()
        )
        assert est.absolute_effect == pytest.approx(expected, rel=1e-12)

    def test_naive_estimates_a_different_quantity(self):
        """两种口径在均值上就不同 —— 这是 naive 做法最根本的问题。"""
        views, clicks, treated = self.make_ratio_data()
        delta = ratio_delta_method(
            AggregateStats.from_arrays(clicks[treated], views[treated]),
            AggregateStats.from_arrays(clicks[~treated], views[~treated]),
        )
        naive = naive_unit_ratio_ttest(
            clicks[treated], views[treated], clicks[~treated], views[~treated]
        )
        # 即使分子分母同分布，人均比值与合并比值也不相等
        assert naive.control_mean_unit_ratio != pytest.approx(delta.mean_control, rel=1e-6)
        assert naive.estimate.mean_control == pytest.approx(naive.control_mean_unit_ratio)

    def test_naive_gap_vs_is_not_identically_zero(self):
        """守住一个真实踩过的坑：gap_vs 不能拿同口径的两个量相减。"""
        views, clicks, treated = self.make_ratio_data()
        delta = ratio_delta_method(
            AggregateStats.from_arrays(clicks[treated], views[treated]),
            AggregateStats.from_arrays(clicks[~treated], views[~treated]),
        )
        naive = naive_unit_ratio_ttest(
            clicks[treated], views[treated], clicks[~treated], views[~treated]
        )
        assert naive.gap_vs(delta) == pytest.approx(
            naive.control_mean_unit_ratio - delta.mean_control
        )

    def test_zero_denominator_reported(self):
        views = np.array([0.0, 2.0, 3.0, 4.0])
        clicks = np.array([0.0, 1.0, 1.0, 2.0])
        naive = naive_unit_ratio_ttest(clicks, views, clicks, views)
        # 两臂各有 1 个分母为 0 的单元被丢弃，计数是跨两臂累加的
        assert naive.n_units_dropped == 2

    def test_reversed_arguments_raise(self):
        """把分子分母传反时必须报错，而不是悄悄给出一个数。"""
        with pytest.raises(ValueError, match="分母均值必须为正"):
            ratio_delta_method(
                AggregateStats.from_arrays([1.0, 2.0], [-1.0, -1.0]),
                AggregateStats.from_arrays([1.0, 2.0], [-1.0, -1.0]),
            )

    def test_diagnostic_present(self):
        views, clicks, treated = self.make_ratio_data()
        est = ratio_delta_method(
            AggregateStats.from_arrays(clicks[treated], views[treated]),
            AggregateStats.from_arrays(clicks[~treated], views[~treated]),
        )
        assert est.diagnostics_of("比值指标") is not None


# --------------------------------------------------------------------------- #
# 聚类稳健
# --------------------------------------------------------------------------- #
class TestClustered:
    def make_cluster_data(self, *, G=60, m=40, cluster_sd=8.0, user_sd=10.0, seed=0):
        rng = np.random.default_rng(seed)
        cid = np.repeat(np.arange(G), m)
        cluster_treated = np.repeat(rng.random(G) < 0.5, m)
        effect = rng.normal(0, cluster_sd, G)
        y = 50 + effect[cid] + rng.normal(0, user_sd, G * m)
        return cid, cluster_treated, y

    def test_icc_near_zero_without_clustering(self):
        rng = np.random.default_rng(0)
        cid = np.repeat(np.arange(100), 50)
        y = rng.normal(0, 1, 5000)  # 无簇效应
        icc, _ = estimate_icc(cid, y)
        assert icc < 0.03

    def test_icc_recovers_theory(self):
        """ICC 应接近 sigma_c^2 / (sigma_c^2 + sigma_u^2)。"""
        cid, _, y = self.make_cluster_data(G=200, m=50, cluster_sd=8.0, user_sd=10.0, seed=1)
        icc, _ = estimate_icc(cid, y)
        theory = 8.0**2 / (8.0**2 + 10.0**2)  # = 0.390
        assert icc == pytest.approx(theory, abs=0.06)

    def test_cluster_robust_widens_se(self):
        """有组内相关时，聚类稳健标准误必须远大于朴素标准误。"""
        cid, treated, y = self.make_cluster_data(seed=2)
        naive = welch_ttest(y[treated], y[~treated])
        robust = cluster_robust_ttest(cid, treated, y)
        assert robust.absolute_effect == pytest.approx(naive.absolute_effect, rel=1e-12)
        assert robust.std_error > naive.std_error * 2

    def test_singleton_clusters_match_welch(self):
        """每簇只有一个单元时，CR1 应退化成异方差稳健版本，接近 Welch 标准误。"""
        rng = np.random.default_rng(3)
        n = 4000
        cid = np.arange(n)  # 每簇一个
        treated = rng.random(n) < 0.5
        y = rng.normal(0, 1, n) + 2.0 * treated
        robust = cluster_robust_ttest(cid, treated, y)
        naive = welch_ttest(y[treated], y[~treated])
        assert robust.std_error == pytest.approx(naive.std_error, rel=0.15)

    def test_cluster_level_uses_cluster_counts(self):
        cid, treated, y = self.make_cluster_data(seed=4)
        est = cluster_level_ttest(cid, treated, y)
        assert est.n_treatment + est.n_control == len(np.unique(cid))

    def test_cluster_level_and_robust_agree_on_balanced_design(self):
        """簇大小相等时，两种口径的估计量相同，标准误也应接近。"""
        cid, treated, y = self.make_cluster_data(G=80, m=50, seed=5)
        robust = cluster_robust_ttest(cid, treated, y)
        level = cluster_level_ttest(cid, treated, y)
        assert robust.absolute_effect == pytest.approx(level.absolute_effect, rel=1e-9)
        assert robust.std_error == pytest.approx(level.std_error, rel=0.15)

    def test_mixed_cluster_assignment_raises(self):
        """同一簇里既有处理又有对照 → 不是聚类随机化，必须报错而不是硬算。"""
        cid = np.repeat(np.arange(20), 10)
        treated = np.zeros(200, dtype=bool)
        treated[::2] = True  # 每个簇内部都有处理与对照
        y = np.random.default_rng(0).normal(0, 1, 200)
        with pytest.raises(ValueError, match="不是聚类随机化"):
            cluster_robust_ttest(cid, treated, y)

    def test_diagnostic_reports_design_effect(self):
        cid, treated, y = self.make_cluster_data(seed=6)
        est = cluster_robust_ttest(cid, treated, y)
        detail = est.diagnostics[0].detail
        assert detail["design_effect"] > 1.0
        assert detail["se_inflation"] == pytest.approx(np.sqrt(detail["design_effect"]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度必须一致"):
            cluster_robust_ttest([1, 2, 3], [True, False], [1.0, 2.0])

    # ---- wild cluster bootstrap（簇数很少时的推断） ------------------------ #

    @staticmethod
    def _wild_data(*, G=12, m=60, seed=0, size_cv=0.0):
        from ablab.validation.wild_bootstrap_audit import _draw_cluster_experiment

        return _draw_cluster_experiment(
            n_clusters=G, users_per_cluster=m, cluster_sd=8.0, user_sd=10.0,
            lift=0.0, seed=seed, size_cv=size_cv,
        )

    def test_returns_a_valid_p_value_and_records_the_design(self):
        cid, treated, y = self._wild_data()
        res = wild_cluster_bootstrap(cid, treated, y, n_bootstrap=299, seed=1)
        assert 0.0 < res.p_value <= 1.0
        assert res.p_value_cr1 >= 0.0
        assert res.n_clusters == 12
        assert res.n_clusters_treated + res.n_clusters_control == 12
        assert res.ci_low < res.effect < res.ci_high
        assert res.weights == "webb" and res.null == "imposed"

    def test_is_deterministic_for_a_fixed_seed(self):
        """同一个 seed 必须给同一个 p 值 —— 报告要可复算。"""
        cid, treated, y = self._wild_data()
        a = wild_cluster_bootstrap(cid, treated, y, n_bootstrap=299, seed=3)
        b = wild_cluster_bootstrap(cid, treated, y, n_bootstrap=299, seed=3)
        assert (a.p_value, a.ci_low, a.ci_high) == (b.p_value, b.ci_low, b.ci_high)

    def test_rademacher_resolution_is_two_to_the_minus_g(self):
        """Rademacher 的分辨率是 2^-G —— 这条就是 G<12 推荐 Webb 的理由。"""
        cid, treated, y = self._wild_data(G=6)
        rad = wild_cluster_bootstrap(
            cid, treated, y, n_bootstrap=999, weights="rademacher", seed=0
        )
        webb = wild_cluster_bootstrap(cid, treated, y, n_bootstrap=999, seed=0)
        assert rad.p_resolution == pytest.approx(2.0 ** -6)
        assert webb.p_resolution == pytest.approx(1.0 / 1000)
        assert rad.p_resolution > webb.p_resolution

    def test_agrees_with_cr1_when_clusters_are_many_and_balanced(self):
        """簇多且均衡时，两条路应当给出接近的 p 值（否则说明实现有问题）。"""
        cid, treated, y = self._wild_data(G=40, m=50, seed=5)
        cr1 = cluster_robust_ttest(cid, treated, y)
        wild = wild_cluster_bootstrap(cid, treated, y, n_bootstrap=999, seed=5)
        assert abs(wild.p_value - cr1.p_value) < 0.2, (wild.p_value, cr1.p_value)

    def test_unrestricted_null_also_runs(self):
        cid, treated, y = self._wild_data()
        res = wild_cluster_bootstrap(
            cid, treated, y, n_bootstrap=299, null="unrestricted", seed=2
        )
        assert 0.0 < res.p_value <= 1.0
        assert res.null == "unrestricted"

    def test_input_validation(self):
        cid, treated, y = self._wild_data(G=3)  # 每臂不足 2 个簇
        with pytest.raises(ValueError, match="识别问题"):
            wild_cluster_bootstrap(cid, treated, y, n_bootstrap=199)
        cid, treated, y = self._wild_data()
        with pytest.raises(ValueError, match="n_bootstrap"):
            wild_cluster_bootstrap(cid, treated, y, n_bootstrap=10)
        with pytest.raises(ValueError, match="weights"):
            wild_cluster_bootstrap(cid, treated, y, n_bootstrap=199, weights="normal")
        with pytest.raises(ValueError, match="null"):
            wild_cluster_bootstrap(cid, treated, y, n_bootstrap=199, null="both")

    def test_audit_is_fast_and_structurally_sound(self):
        """审计本体：小规模跑一遍，钉结构与"实测属性"的一致性。"""
        from ablab.validation.wild_bootstrap_audit import run_wild_bootstrap_audit

        audit = run_wild_bootstrap_audit(
            n_trials=20, n_clusters_grid=(6,), size_cv_grid=(0.0, 1.0)
        )
        assert len(audit.rows) == 2
        assert {r.size_cv for r in audit.rows} == {0.0, 1.0}
        for row in audit.rows:
            for rate in (row.size_cr1, row.size_wild_webb, row.size_wild_rademacher):
                assert 0.0 <= rate <= 1.0
            assert row.rademacher_min_p == pytest.approx(2.0 ** -6)
        assert np.isfinite(audit.worst_power_cost)
