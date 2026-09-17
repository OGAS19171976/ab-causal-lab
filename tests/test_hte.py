"""M4 单元测试：HTE 数据、DML、元学习器、因果森林、Uplift 指标。"""

import numpy as np
import pytest

from ablab.causal import (
    CATE_FORMS,
    CausalForest,
    ForestConfig,
    HTEConfig,
    constant_prediction,
    dml_partial_linear,
    generate_hte_data,
    naive_plugin,
    qini_coefficient,
    rank_correlation,
    s_learner,
    scaled_perfect,
    t_learner,
    uplift_curve,
    x_learner,
)
from ablab.validation import (
    run_cate_form_comparison,
    run_dml_audit,
    run_uplift_metric_audit,
)

SMALL = HTEConfig(n=1200, cate_form="nonlinear", seed=0)


@pytest.fixture(scope="module")
def data():
    return generate_hte_data(SMALL)


@pytest.fixture(scope="module")
def forest(data):
    cfg = ForestConfig(n_trees=25, max_depth=4, min_leaf=20, seed=0)
    return CausalForest(cfg).fit(data.X, data.D, data.Y)


# --------------------------------------------------------------------------- #
# DGP
# --------------------------------------------------------------------------- #
class TestHTEData:
    def test_shapes(self, data):
        assert data.X.shape == (SMALL.n, SMALL.n_features)
        assert data.D.shape == (SMALL.n,)
        assert data.Y.shape == (SMALL.n,)
        assert data.tau.shape == (SMALL.n,)

    @pytest.mark.parametrize("form", CATE_FORMS)
    def test_each_form_has_expected_heterogeneity(self, form):
        d = generate_hte_data(HTEConfig(n=1000, cate_form=form, seed=1))
        if form == "constant":
            assert d.tau.std() == pytest.approx(0.0, abs=1e-12)
        else:
            assert d.tau.std() > 0.3

    def test_true_ate_matches_arm_difference(self):
        """无混淆时，两组均值差应接近 ATE。"""
        d = generate_hte_data(
            HTEConfig(n=20000, cate_form="linear", propensity_strength=0.0, seed=2)
        )
        raw = d.Y[d.D > 0.5].mean() - d.Y[d.D < 0.5].mean()
        assert raw == pytest.approx(d.ate, abs=0.15)

    def test_confounding_exists(self):
        """有混淆时，原始两组差应当**偏离** ATE。"""
        d = generate_hte_data(
            HTEConfig(n=20000, cate_form="constant", propensity_strength=1.5, seed=3)
        )
        raw = d.Y[d.D > 0.5].mean() - d.Y[d.D < 0.5].mean()
        assert abs(raw - d.ate) > 0.1

    def test_reproducible(self):
        a = generate_hte_data(HTEConfig(n=500, seed=4))
        b = generate_hte_data(HTEConfig(n=500, seed=4))
        assert np.array_equal(a.Y, b.Y)

    def test_propensity_in_unit_interval(self):
        d = generate_hte_data(HTEConfig(n=2000, propensity_strength=3.0, seed=5))
        # 极端 logit 下 sigmoid 会浮点下溢到恰好 0 或 1，这是正常的
        assert d.propensity.min() >= 0.0 and d.propensity.max() <= 1.0
        assert d.propensity.std() > 0.05, "倾向得分应当有实质变异"

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="cate_form"):
            HTEConfig(cate_form="nope")
        with pytest.raises(ValueError, match="n_informative"):
            HTEConfig(n_features=3, n_informative=5)


# --------------------------------------------------------------------------- #
# DML
# --------------------------------------------------------------------------- #
class TestDML:
    def test_dml_beats_naive_at_large_n(self):
        """多次仿真取平均 —— 单次试验的 DML 偏置波动太大，不足以做判据。"""
        naive_biases, dml_biases = [], []
        for s in range(5):
            d = generate_hte_data(HTEConfig(n=3000, cate_form="constant", seed=100 + s))
            naive_biases.append(naive_plugin(d.Y, d.D, d.X, true_theta=d.ate).bias)
            dml_biases.append(
                dml_partial_linear(
                    d.Y, d.D, d.X, n_folds=5, true_theta=d.ate, seed=s
                ).bias
            )
        avg_naive = float(np.mean(np.abs(naive_biases)))
        avg_dml = float(np.mean(np.abs(dml_biases)))
        assert avg_dml < avg_naive / 2, f"naive {avg_naive:.4f} vs DML {avg_dml:.4f}"

    def test_naive_ci_misses_truth(self):
        d = generate_hte_data(HTEConfig(n=4000, cate_form="constant", seed=7))
        nv = naive_plugin(d.Y, d.D, d.X, true_theta=d.ate)
        assert not nv.covers_truth

    def test_dml_ci_covers_truth(self):
        d = generate_hte_data(HTEConfig(n=4000, cate_form="constant", seed=8))
        dm = dml_partial_linear(d.Y, d.D, d.X, n_folds=5, true_theta=d.ate, seed=0)
        assert dm.covers_truth

    def test_naive_bias_does_not_shrink_with_n(self):
        """核心性质：遗漏变量偏置不随样本量消失。"""
        biases = []
        for n in (1000, 4000):
            d = generate_hte_data(HTEConfig(n=n, cate_form="constant", seed=9))
            biases.append(abs(naive_plugin(d.Y, d.D, d.X, true_theta=d.ate).bias))
        assert biases[1] > biases[0] * 0.5

    def test_dml_bias_shrinks_with_n(self):
        biases = []
        for n in (1000, 4000):
            d = generate_hte_data(HTEConfig(n=n, cate_form="constant", seed=10))
            biases.append(
                abs(
                    dml_partial_linear(
                        d.Y, d.D, d.X, n_folds=5, true_theta=d.ate, seed=0
                    ).bias
                )
            )
        assert biases[1] < biases[0]

    def test_naive_se_shrinks_with_n(self):
        ses = []
        for n in (1000, 4000):
            d = generate_hte_data(HTEConfig(n=n, cate_form="constant", seed=11))
            ses.append(naive_plugin(d.Y, d.D, d.X).se)
        assert ses[1] < ses[0] * 0.6

    def test_invalid_input(self):
        d = generate_hte_data(HTEConfig(n=200, seed=12))
        with pytest.raises(ValueError, match="样本量必须一致"):
            dml_partial_linear(d.Y, d.D[:-1], d.X)
        with pytest.raises(ValueError, match="n_folds"):
            dml_partial_linear(d.Y, d.D, d.X, n_folds=0)

    def test_no_crossfit_runs(self):
        d = generate_hte_data(HTEConfig(n=1000, cate_form="constant", seed=13))
        res = dml_partial_linear(d.Y, d.D, d.X, n_folds=1)
        assert np.isfinite(res.theta) and res.n_folds == 1


# --------------------------------------------------------------------------- #
# 元学习器
# --------------------------------------------------------------------------- #
class TestMetaLearners:
    @pytest.mark.parametrize("name", ["s", "t", "x"])
    def test_produces_finite_cate(self, data, name):
        fn = {"s": s_learner, "t": t_learner, "x": x_learner}[name]
        cate = fn(data.X, data.D, data.Y)
        pred = cate(data.X[:200])
        assert pred.shape == (200,)
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize("name", ["t", "x"])
    def test_captures_ordering(self, data, name):
        """T / X-learner 应当抓到非线性 CATE 的排序（哪怕水平有偏）。"""
        fn = {"t": t_learner, "x": x_learner}[name]
        pred = fn(data.X, data.D, data.Y)(data.X)
        rho = rank_correlation(data.tau, pred)
        assert rho > 0.25, f"{name}-learner 秩相关仅 {rho}"

    def test_x_learner_accepts_propensity(self, data):
        cate = x_learner(data.X, data.D, data.Y, propensity=0.5)
        assert np.all(np.isfinite(cate(data.X[:50])))

    def test_too_few_in_one_arm_raises(self):
        d = generate_hte_data(HTEConfig(n=300, cate_form="linear", seed=14))
        D = np.ones(300)
        with pytest.raises(ValueError, match="至少要有 5 个"):
            t_learner(d.X, D, d.Y)


# --------------------------------------------------------------------------- #
# 因果森林
# --------------------------------------------------------------------------- #
class TestCausalForest:
    def test_prediction_shape(self, forest, data):
        assert forest.predict(data.X).shape == (data.n,)

    def test_captures_ordering(self, forest, data):
        rho = rank_correlation(data.tau, forest.predict(data.X))
        assert rho > 0.4, f"森林秩相关仅 {rho}"

    def test_honest_beats_dishonest_on_noise(self):
        """不诚实分裂会对噪声做最大化，把效应估计放大。

        用一个**没有异质性**的 DGP：真实 CATE 处处为 1.05 上下，
        不诚实的森林应当给出方差更大的预测。
        """
        d = generate_hte_data(HTEConfig(n=3000, cate_form="constant", seed=15))
        honest = CausalForest(ForestConfig(n_trees=25, honest=True, seed=0)).fit(
            d.X, d.D, d.Y
        )
        dishonest = CausalForest(ForestConfig(n_trees=25, honest=False, seed=0)).fit(
            d.X, d.D, d.Y
        )
        v_h = float(honest.predict(d.X).var())
        v_d = float(dishonest.predict(d.X).var())
        assert v_d > v_h, f"不诚实 {v_d:.4f} 应大于诚实 {v_h:.4f}"

    def test_shrinkage_pulls_toward_constant(self, data):
        base = CausalForest(ForestConfig(n_trees=25, shrinkage=0.0, seed=0)).fit(
            data.X, data.D, data.Y
        )
        shrunk = CausalForest(ForestConfig(n_trees=25, shrinkage=500.0, seed=0)).fit(
            data.X, data.D, data.Y
        )
        assert shrunk.predict(data.X).std() < base.predict(data.X).std()

    def test_feature_importance_normalized(self, forest):
        imp = forest.feature_importance
        assert imp.sum() == pytest.approx(1.0)
        assert (imp >= 0).all()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="先调用 fit"):
            CausalForest(ForestConfig(n_trees=2)).predict(np.zeros((3, 2)))

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="n_trees"):
            ForestConfig(n_trees=0)
        with pytest.raises(ValueError, match="subsample"):
            ForestConfig(subsample=1.5)


# --------------------------------------------------------------------------- #
# Uplift 指标
# --------------------------------------------------------------------------- #
class TestUplift:
    def test_qini_is_scale_invariant(self, data):
        """Qini 只看排序，正缩放不改变它 —— 这正是它的边界所在。"""
        a = qini_coefficient(data.Y, data.D, data.tau)
        b = qini_coefficient(data.Y, data.D, scaled_perfect(data.tau, factor=1000.0))
        assert a == pytest.approx(b, rel=1e-9)

    def test_constant_has_near_zero_qini(self, data):
        q = qini_coefficient(data.Y, data.D, constant_prediction(data.tau))
        assert abs(q) < abs(qini_coefficient(data.Y, data.D, data.tau)) * 0.2

    def test_oracle_beats_scrambled(self, data):
        rng = np.random.default_rng(0)
        scrambled = rng.permutation(data.tau)
        assert qini_coefficient(data.Y, data.D, data.tau) > qini_coefficient(
            data.Y, data.D, scrambled
        )

    def test_reversed_ranking_is_negative(self, data):
        assert qini_coefficient(data.Y, data.D, -data.tau) < 0

    def test_curve_shape(self, data):
        curve = uplift_curve(data.Y, data.D, data.tau, n_points=50)
        assert curve.fractions.shape == curve.qini.shape
        assert curve.fractions[0] > 0 and curve.fractions[-1] == pytest.approx(1.0)
        assert np.all(np.diff(curve.fractions) > 0)

    def test_rank_correlation_handles_constant(self, data):
        assert np.isnan(rank_correlation(data.tau, constant_prediction(data.tau)))
        assert np.isnan(rank_correlation(np.ones(10), np.arange(10.0)))

    def test_rank_correlation_perfect(self, data):
        assert rank_correlation(data.tau, 3.0 * data.tau) == pytest.approx(1.0)

    def test_scaled_perfect_has_bad_mse(self, data):
        p = scaled_perfect(data.tau, factor=1000.0)
        assert np.mean((p - data.tau) ** 2) > 1e5

    def test_length_mismatch_raises(self, data):
        with pytest.raises(ValueError, match="样本量必须一致"):
            uplift_curve(data.Y, data.D[:-1], data.tau)


# --------------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------------- #
class TestHTEAudit:
    def test_dml_coverage_collapses_for_naive(self):
        a = run_dml_audit(n_trials=8, sizes=(600, 2500), seed=0)
        assert a.coverage_collapses
        assert a.dml_holds

    def test_dml_audit_summary(self):
        a = run_dml_audit(n_trials=6, sizes=(800, 2000), seed=0)
        text = a.summary()
        assert "不随 n 消失" in text

    def test_cate_form_comparison_covers_all_forms(self):
        c = run_cate_form_comparison(n=1000, seed=0)
        assert len(c.results) == 4
        assert {r.cate_form for r in c.results} == set(CATE_FORMS)

    def test_constant_form_has_infinite_ratio(self):
        c = run_cate_form_comparison(n=1000, seed=0)
        const = next(r for r in c.results if r.cate_form == "constant")
        assert not np.isfinite(const.mse_ratio)
        assert const.mse_constant == pytest.approx(0.0, abs=1e-9)

    def test_uplift_metric_audit_detects_conflict(self):
        a = run_uplift_metric_audit(n=3000, seed=0)
        assert a.ranking_says_forest_wins
        assert a.level_says_forest_loses
        assert a.verdicts_conflict
        assert a.in_sample_optimism > 0

    def test_uplift_audit_summary_mentions_the_conflict(self):
        a = run_uplift_metric_audit(n=2500, seed=0)
        assert "相反结论" in a.summary()
