"""推断层测试：两条统计路径必须一致，SRM 该报警时报警、不该报警时不报警。"""

import numpy as np
import pytest
from scipy import stats

from ablab.inference import (
    srm_check,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_stats,
)


class TestWelchConsistency:
    """汇总统计量路径与明细路径必须给出完全相同的结果。

    这是数仓链路可信的前提：ADS 层只输出 (n, mean, var)，
    如果两条路径有偏差，说明有一侧算错了，而线上只会跑其中一条。
    """

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_summary_matches_detail(self, seed):
        rng = np.random.default_rng(seed)
        t = rng.normal(10.0, 3.0, 500)
        c = rng.normal(9.5, 4.0, 700)

        detail = welch_ttest(t, c)
        summary = welch_ttest_from_stats(
            n_treatment=t.size,
            mean_treatment=float(t.mean()),
            var_treatment=float(t.var(ddof=1)),
            n_control=c.size,
            mean_control=float(c.mean()),
            var_control=float(c.var(ddof=1)),
        )

        assert summary.absolute_effect == pytest.approx(detail.absolute_effect, rel=0, abs=1e-12)
        assert summary.std_error == pytest.approx(detail.std_error, rel=0, abs=1e-12)
        assert summary.p_value == pytest.approx(detail.p_value, rel=0, abs=1e-12)
        assert summary.ci_low == pytest.approx(detail.ci_low, rel=0, abs=1e-12)
        assert summary.ci_high == pytest.approx(detail.ci_high, rel=0, abs=1e-12)

    def test_matches_scipy(self):
        """对照 scipy 的独立实现，确认自由度与 p 值没写错。"""
        rng = np.random.default_rng(7)
        t = rng.normal(0, 1, 100)
        c = rng.normal(0.5, 2, 150)

        est = welch_ttest(t, c)
        ref = stats.ttest_ind(t, c, equal_var=False)
        assert est.p_value == pytest.approx(float(ref.pvalue), rel=1e-12)

    def test_ci_covers_truth(self):
        rng = np.random.default_rng(3)
        covered = 0
        for _ in range(300):
            t = rng.normal(1.0, 1.0, 200)
            c = rng.normal(1.0, 1.0, 200)
            est = welch_ttest(t, c)
            covered += int(est.ci_low <= 0.0 <= est.ci_high)
        assert 0.90 <= covered / 300 <= 0.99

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="至少需要 2 个"):
            welch_ttest([1.0], [2.0, 3.0])

    def test_nan_is_dropped(self):
        est = welch_ttest([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0, 4.0])
        assert est.n_treatment == 3

    def test_zero_variance_warns_not_crashes(self):
        est = welch_ttest([5.0] * 10, [5.0] * 10)
        assert est.p_value == 1.0
        assert any(d.name == "方差退化" for d in est.diagnostics)


class TestSRM:
    def test_balanced_passes(self):
        diag = srm_check({"control": 5000, "treatment": 5000}, {"control": 0.5, "treatment": 0.5})
        assert diag.status == "pass"
        assert diag.ok

    def test_imbalance_detected(self):
        """40/60 的分组，在 10000 样本下必然被判定为失衡。"""
        diag = srm_check({"control": 4000, "treatment": 6000}, {"control": 0.5, "treatment": 0.5})
        assert diag.status == "fail"
        assert not diag.ok
        assert diag.statistic == pytest.approx(400.0, rel=1e-6)

    def test_three_way(self):
        diag = srm_check(
            {"a": 2000, "b": 3000, "c": 5000},
            {"a": 0.2, "b": 0.3, "c": 0.5},
        )
        assert diag.status == "pass"

    def test_weight_can_be_9_to_1(self):
        diag = srm_check({"a": 9000, "b": 1000}, {"a": 0.9, "b": 0.1})
        assert diag.status == "pass"

    def test_false_positive_rate_is_low(self):
        """SRM 是每天每实验都跑的常规检验，误报率必须低。"""
        rng = np.random.default_rng(0)
        alarms = 0
        for _ in range(500):
            n = 20000
            treated = rng.binomial(n, 0.5)
            diag = srm_check(
                {"control": n - treated, "treatment": treated},
                {"control": 0.5, "treatment": 0.5},
            )
            alarms += int(diag.status == "fail")
        assert alarms / 500 < 0.02

    def test_missing_variant_raises(self):
        with pytest.raises(ValueError, match="缺失"):
            srm_check({"control": 100}, {"control": 0.5, "treatment": 0.5})

    def test_empty_counts_fails(self):
        diag = srm_check({"control": 0, "treatment": 0}, {"control": 0.5, "treatment": 0.5})
        assert diag.status == "fail"

    def test_small_expected_warns(self):
        diag = srm_check({"a": 3, "b": 1}, {"a": 0.5, "b": 0.5})
        assert diag.status == "warn"


class TestEstimateObject:
    def test_health_reflects_diagnostics(self):
        rng = np.random.default_rng(0)
        t = rng.normal(0, 1, 1000)
        c = rng.normal(0, 1, 1000)
        healthy = welch_ttest(
            t, c, expected_weights={"control": 0.5, "treatment": 0.5}
        )
        assert healthy.is_healthy

        unhealthy = welch_ttest(
            t, np.r_[c, np.zeros(3000)], expected_weights={"control": 0.5, "treatment": 0.5}
        )
        assert not unhealthy.is_healthy

    def test_report_contains_key_fields(self):
        rng = np.random.default_rng(1)
        est = welch_ttest(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        text = est.report()
        assert "Welch" in text
        assert "diagnostics" in text


class TestProportionZTest:
    def test_matches_manual_calculation(self):
        est = two_proportion_ztest(120, 1000, 100, 1000)
        p_pool = 220 / 2000
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / 1000 + 1 / 1000))
        z = (0.12 - 0.10) / se
        assert est.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-12)

    def test_no_difference_gives_p_one(self):
        est = two_proportion_ztest(100, 1000, 100, 1000)
        assert est.p_value == pytest.approx(1.0)
