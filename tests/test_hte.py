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
    dr_learner,
    generate_hte_data,
    naive_plugin,
    qini_coefficient,
    r_learner,
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


class TestRLearnerAndDRLearner:
    """R-learner 与 DR-learner：正交化 + 交叉拟合的元学习器。

    重点钉三件事：
    (1) 线性 τ 的 DGP 上它们能真的把 CATE 估准（而 T-learner 不能）；
    (2) **闭式解与"伪结果 + 加权拟合"两条路径是同一个损失** ——
        这是文档里那句"两者恰好相等"的可执行版本；
    (3) 对照实验只许改一个变量：``n_folds=1`` 那一支的 ``ê`` 也必须照样估，
        否则量出来的差不属于"交叉拟合"。
    """

    @staticmethod
    def _linear_data(n: int = 1500, seed: int = 3):
        return generate_hte_data(
            HTEConfig(n=n, n_features=6, n_informative=3, cate_form="linear", seed=seed)
        )

    def test_r_learner_beats_t_learner_on_a_linear_cate(self):
        d = self._linear_data()
        r = r_learner(d.X, d.D, d.Y, seed=0)(d.X)
        t_pred = t_learner(d.X, d.D, d.Y)(d.X)
        mse_r = float(np.mean((r - d.tau) ** 2))
        mse_t = float(np.mean((t_pred - d.tau) ** 2))
        assert mse_r < mse_t, (mse_r, mse_t)
        assert rank_correlation(d.tau, r) > 0.8

    def test_dr_learner_produces_finite_cate(self):
        d = generate_hte_data(
            HTEConfig(n=800, n_features=6, n_informative=3, cate_form="nonlinear", seed=5)
        )
        pred = dr_learner(d.X, d.D, d.Y, seed=0)(d.X[:200])
        assert pred.shape == (200,)
        assert np.all(np.isfinite(pred))
        assert float(np.std(pred)) > 0

    def test_closed_form_matches_the_weighted_path(self):
        """闭式解 ≡ 线性学习器 + 伪结果加权拟合（同一个 R-loss）。"""
        from sklearn.linear_model import LinearRegression

        d = self._linear_data(n=1200, seed=7)
        closed = r_learner(d.X, d.D, d.Y, seed=0)(d.X)
        weighted = r_learner(
            d.X, d.D, d.Y, learner=LinearRegression(), seed=0
        )(d.X)
        # 两条路径的差别只来自线性代数实现，量级应当是 1e-8 而不是"差不多"
        assert np.max(np.abs(closed - weighted)) < 1e-6, float(
            np.max(np.abs(closed - weighted))
        )

    def test_cross_fitting_flag_actually_changes_something(self):
        d = self._linear_data(n=1000, seed=11)
        cf = r_learner(d.X, d.D, d.Y, n_folds=5, seed=0)(d.X)
        no_cf = r_learner(d.X, d.D, d.Y, n_folds=1, seed=0)(d.X)
        assert not np.allclose(cf, no_cf)

    def test_clip_is_validated_and_reported(self):
        d = self._linear_data(n=600, seed=13)
        with pytest.raises(ValueError, match="clip"):
            r_learner(d.X, d.D, d.Y, clip=0.6)
        with pytest.raises(ValueError, match="clip"):
            dr_learner(d.X, d.D, d.Y, clip=0.0)

    def test_sample_size_mismatch_raises(self):
        d = self._linear_data(n=400, seed=17)
        with pytest.raises(ValueError, match="样本量"):
            r_learner(d.X, d.D[:100], d.Y)


# --------------------------------------------------------------------------- #
# 策略学习（AIPW）
# --------------------------------------------------------------------------- #
class TestAipwPolicyLearning:
    """钉住三件事：固定策略的估计是无偏的、选择会带来乐观偏差、
    以及"Γ>0 就投"根本不是一个策略。"""

    @staticmethod
    def _data(n: int, scale: float, seed: int):
        from ablab.causal import HTEConfig, generate_hte_data

        return generate_hte_data(
            HTEConfig(
                n=n,
                cate_form="threshold",
                cate_scale=scale,
                propensity_strength=0.6,
                seed=seed,
            )
        )

    def test_fixed_policy_value_is_close_and_covers(self):
        """预指定策略（不含选择）：AIPW 价值应当贴近真值，区间盖住真值。"""
        from ablab.causal import ThresholdPolicy
        from ablab.causal.policy import aipw_effect_scores, policy_value

        d = self._data(n=8000, scale=1.0, seed=3)
        scores = aipw_effect_scores(d.Y, d.D, d.X, seed=0)
        pol = ThresholdPolicy(feature=0, threshold=0.0)
        pv = policy_value(scores, d.X, pol)
        truth = float((d.tau * pol(d.X)).mean())
        assert abs(pv.value - truth) < 3.0 * pv.se
        assert pv.covers(truth)
        assert pv.se < 0.25  # 这个 n 下 SE 应当很小

    def test_se_shrinks_with_sample_size(self):
        """影响函数 SE 的 1/√n 行为（否则它不是 SE，只是个数字）。"""
        from ablab.causal import ThresholdPolicy
        from ablab.causal.policy import aipw_effect_scores, policy_value

        pol = ThresholdPolicy(feature=0, threshold=0.0)
        ses = []
        for n in (1000, 4000):
            d = self._data(n=n, scale=1.0, seed=5)
            scores = aipw_effect_scores(d.Y, d.D, d.X, seed=0)
            ses.append(policy_value(scores, d.X, pol).se)
        ratio = ses[0] / ses[1]
        assert 1.5 < ratio < 2.6, ratio

    def test_mask_must_be_binary_and_aligned(self):
        from ablab.causal.policy import aipw_effect_scores, policy_value

        d = self._data(n=400, scale=1.0, seed=7)
        scores = aipw_effect_scores(d.Y, d.D, d.X, seed=0)
        with pytest.raises(ValueError, match="长度"):
            policy_value(scores, d.X, np.ones(10))
        with pytest.raises(ValueError, match="0/1"):
            policy_value(scores, d.X, np.full(400, 0.5))

    def test_ranking_mask_takes_the_top_share(self):
        from ablab.causal.policy import ranking_mask

        score = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2])
        mask = ranking_mask(score, share=0.5)
        assert mask.sum() == 3
        assert set(np.flatnonzero(mask)) == {1, 2, 4}
        with pytest.raises(ValueError, match="share"):
            ranking_mask(score, share=1.2)

    def test_learned_policy_reaches_oracle_on_signal(self):
        """有信号时：学到的策略要真的拿到 oracle 的大部分（不是样本内数字）。"""
        from ablab.causal.policy import aipw_policy_learner

        d = self._data(n=2000, scale=1.0, seed=11)
        learned = aipw_policy_learner(d.Y, d.D, d.X, depth=1, n_grid=12, seed=0)
        mask = learned.policy(d.X)
        true_value = float((d.tau * mask).mean())
        oracle = float((d.tau * (d.tau > 0)).mean())
        treat_all = float(d.tau.mean())
        assert true_value >= 0.80 * oracle, (true_value, oracle)
        assert true_value > 3.0 * treat_all, (true_value, treat_all)
        assert 0.05 < mask.mean() < 0.95  # 不是"全投"或"全不投"这种平凡解

    def test_noise_regime_shows_optimism_and_splitting_shrinks_it(self):
        """τ ≡ 0：样本内价值为正（真值为 0），分离样本价值明显更小。"""
        from ablab.causal.policy import aipw_policy_learner

        d = self._data(n=1500, scale=0.0, seed=13)
        learned = aipw_policy_learner(d.Y, d.D, d.X, depth=1, n_grid=12, seed=0)
        assert learned.in_sample.value > 0.05
        assert learned.split.value < learned.in_sample.value
        assert learned.optimism > 0.0
        # 真值精确为 0（τ ≡ 0 是构造出来的，不是估出来的）
        assert float((d.tau * learned.policy(d.X)).mean()) == pytest.approx(0.0)

    def test_capacity_increases_the_optimism(self):
        """同一份噪声数据：深度 2 的样本内价值高于深度 1（容量换偏差）。"""
        from ablab.causal.policy import (
            aipw_effect_scores,
            learn_threshold_policy,
            policy_value,
        )

        d = self._data(n=1200, scale=0.0, seed=17)
        scores = aipw_effect_scores(d.Y, d.D, d.X, seed=0)
        values = {}
        for depth in (1, 2):
            pol, cands = learn_threshold_policy(scores, d.X, depth=depth, n_grid=10)
            values[depth] = (policy_value(scores, d.X, pol).value, cands)
        assert values[2][0] > values[1][0]
        assert values[2][1] > values[1][1]  # 候选数也更多（容量确实更大）

    def test_unrestricted_rule_is_not_a_policy(self):
        """「Γ>0 就投」要求知道单元自己的 Γ —— 样本内价值很高，真实价值是 0。"""
        from ablab.causal.policy import (
            aipw_effect_scores,
            policy_value,
            unrestricted_mask,
        )

        d = self._data(n=1200, scale=0.0, seed=19)
        scores = aipw_effect_scores(d.Y, d.D, d.X, seed=0)
        mask = unrestricted_mask(scores)
        pv = policy_value(scores, d.X, mask)
        assert pv.value > 0.3
        assert float((d.tau * mask).mean()) == pytest.approx(0.0)

    def test_audit_reports_the_decomposition(self):
        """审计本体（小规模）：偏差面 + 选择乐观这个分解要真的看得见。"""
        from ablab.validation.policy_audit import run_policy_audit

        audit = run_policy_audit(n_trials=2, n=300, n_grid=6, n_folds=2)
        passed = audit.passed()
        assert passed["噪声档：样本内价值为正（真值恒为 0）"]
        assert passed["机制：偏差曲线不是平的（选区域能捡到）"]
        assert passed["大容量类的区间排除真值（覆盖率崩）"]
        # 覆盖率的两端：预指定策略守住，无限制那一支崩掉
        assert audit.coverage["固定策略·覆盖率"] >= audit.coverage["无限制·覆盖率"]
        assert audit.noise["样本内价值·无限制"] > audit.noise["样本内价值·深度1"]


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
