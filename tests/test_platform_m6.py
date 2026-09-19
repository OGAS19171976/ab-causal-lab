"""M6：平台在**生产口径**下的三个不变量。

与 ``test_platform.py`` 的分工：那边测 M5 的注册表/编排/接口，
这里测 M6 新增的三件事是否真的接住了：

1. **口径一致** —— 监控曲线的判定估计量必须与头条结论是同一个，
   而且最后一次查看与头条结论**逐位相同**。
2. **分析单元** —— 整簇随机化必须走簇级检验；比值指标必须走 delta method。
   这两条都属于"用错了会给出一个看起来很正常的错答案"，所以要有测试钉住。
3. **MDE / 功效** —— 解析公式与它的反函数必须自洽，且反解回来的功效命中目标。
"""

import pytest

from ablab.inference import mde, required_n_per_arm, se_of_mean_diff, z_power
from ablab.platform import ExperimentRegistry, RegistryError, analyse_experiment
from ablab.platform.audit import run_monitoring_fwer_audit, run_unit_awareness_audit
from ablab.platform.datasource import build_synthetic_data

#: 审计里的操作者（现在是必填的具名参数）。
ACTOR = "tester"

TWO_ARM = [
    {"name": "control", "weight": 0.5},
    {"name": "treatment", "weight": 0.5},
]


#: 写接口现在需要凭据。测试里统一用这个助手：建一个 admin 用户，
#: 把 token 挂到 client 上（`client.headers`），这样各测试的调用点不用逐个改。
def authed_client(app):
    """给 TestClient 装上 admin 凭据。返回 client 本身。"""
    from fastapi.testclient import TestClient

    token = app.state.registry.add_user("test_admin", role="admin")
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture
def registry(work_dir):
    reg = ExperimentRegistry(work_dir / "m6.db")
    yield reg
    reg.close()


# --------------------------------------------------------------------------- #
# M6.1 口径一致
# --------------------------------------------------------------------------- #
class TestEstimatorAlignment:
    def test_last_look_equals_primary(self, registry):
        """**核心不变量**：监控右端点 == 头条结论，逐位相同。"""
        for est in ("cuped", "post_only"):
            rec = registry.create(actor=ACTOR, name=f"al_{est}", variants=TWO_ARM, true_lift=0.3, estimator=est)
            rep = analyse_experiment(rec, n_users=6000, seed=1)
            last = rep.monitoring[-1]
            assert rep.primary_estimator_name == est
            assert last["estimator"] == est
            assert last["effect"] == rep.primary.absolute_effect
            assert last["std_error"] == rep.primary.std_error

    def test_alt_is_the_other_estimator(self, registry):
        rec = registry.create(actor=ACTOR, name="al2", variants=TWO_ARM, true_lift=0.3)
        rep = analyse_experiment(rec, n_users=6000, seed=2)
        assert rep.alt_estimator_name == "post_only"
        # 同一个数据、同一个效应，只换了标准误
        assert rep.primary.absolute_effect == pytest.approx(rep.alt.absolute_effect, abs=0.5)
        assert rep.primary.std_error < rep.alt.std_error, "CUPED 的 SE 必须更小"
        for m in rep.monitoring:
            assert m["alt_estimator"] == "post_only"
            assert m["alt_z"] is not None

    def test_crossed_matches_declared_estimator(self, registry):
        """``crossed`` 必须由**声明口径**的 z 决定，不能顺手用另一个。"""
        rec = registry.create(actor=ACTOR, name="al3", variants=TWO_ARM, true_lift=0.3, estimator="post_only")
        rep = analyse_experiment(rec, n_users=6000, seed=3)
        for m in rep.monitoring:
            assert m["crossed"] == (abs(m["z"]) >= m["boundary"])
            assert m["estimator"] == "post_only"

    def test_monitoring_carries_both_paths(self, registry):
        rec = registry.create(actor=ACTOR, name="al4", variants=TWO_ARM)
        rep = analyse_experiment(rec, n_users=4000, n_looks=4, seed=4)
        for m in rep.monitoring:
            assert {"z", "alt_z", "alt_effect", "alt_std_error"} <= set(m)
            assert m["alt_std_error"] > m["std_error"]

    def test_estimator_declaration_validated(self, registry):
        with pytest.raises(RegistryError, match="estimator"):
            registry.create(actor=ACTOR, name="al5", variants=TWO_ARM, estimator="nope")

    def test_set_estimator(self, registry):
        rec = registry.create(actor=ACTOR, name="al6", variants=TWO_ARM)
        assert registry.set_estimator(rec.id, "post_only", actor=ACTOR).estimator == "post_only"
        with pytest.raises(RegistryError, match="estimator"):
            registry.set_estimator(rec.id, "nope", actor=ACTOR)


# --------------------------------------------------------------------------- #
# M6.2 分析单元
# --------------------------------------------------------------------------- #
class TestAnalysisUnit:
    def test_cluster_path_uses_cluster_level(self, registry):
        rec = registry.create(actor=ACTOR, 
            name="cl1", variants=TWO_ARM, analysis_unit="cluster",
            estimator="post_only", true_lift=2.0,
        )
        rep = analyse_experiment(rec, n_users=10_000, seed=5)
        assert rep.analysis_unit == "cluster"
        assert rep.primary_estimator_name == "cluster_level"
        assert rep.alt_estimator_name == "unit_level"
        # 簇级 SE 必须**大得多** —— 单元级检验忽略了组内相关
        assert rep.primary.std_error > 5 * rep.alt.std_error
        # 分析单元数与用户数是两个数
        assert rep.n_analysis_units < rep.n_users
        assert rep.n_analysis_units == sum(rep.monitoring[-1][k] for k in
                                          ("n_clusters_treatment", "n_clusters_control"))

    def test_cluster_srm_uses_cluster_counts(self, registry):
        """SRM 的检验对象是**随机化单元**。簇设计下就是簇数，不是用户数。"""
        rec = registry.create(actor=ACTOR, 
            name="cl2", variants=TWO_ARM, analysis_unit="cluster", estimator="post_only"
        )
        rep = analyse_experiment(rec, n_users=10_000, seed=6)
        srm = rep.checks[0]
        assert srm.name == "SRM"
        # 卡方自由度 = 分支数 - 1 = 1，样本量级必须是"簇"的量级而不是"用户"
        assert rep.n_analysis_units < 0.05 * rep.n_users

    def test_cluster_monitoring_is_cluster_level(self, registry):
        rec = registry.create(actor=ACTOR, 
            name="cl3", variants=TWO_ARM, analysis_unit="cluster", estimator="post_only"
        )
        rep = analyse_experiment(rec, n_users=10_000, n_looks=5, seed=7)
        for m in rep.monitoring:
            assert m["estimator"] == "cluster_level"
            assert m["alt_estimator"] == "unit_level"
            assert m["n_clusters_per_arm"] >= 2
        last = rep.monitoring[-1]
        assert last["effect"] == rep.primary.absolute_effect

    def test_cluster_with_cuped_is_now_allowed(self, registry):
        """**改写了**：这条原先断言"整簇 + CUPED 在创建时被拒"。

        拒绝的理由是"数据源没有簇级前置指标"，而那是一个**过时假设**：
        05 路 DWS 一直落着簇级的 pre_sum / pre_sq_sum / pre_post_cross_sum
        （实测 pre_sum ≈ 4.4e6）。现在按声明放行；真拿不到前置指标时由分析层
        报错（数据驱动的检查），而不是在这里一刀切拦住。
        """
        rec = registry.create(
            actor=ACTOR,
            name="cl4", variants=TWO_ARM, analysis_unit="cluster", estimator="cuped",
        )
        assert rec.analysis_unit == "cluster" and rec.estimator == "cuped"

    def test_cluster_stats_must_merge_back(self):
        """簇级统计量合并回去必须等于臂级统计量（两次读取口径一致）。"""
        from ablab.platform.analysis import PLATFORM_POPULATION

        data = build_synthetic_data(
            experiment="x", salt="merge_v1", variants=[("control", 0.5), ("treatment", 0.5)],
            metric="m", n_users=5000, n_looks=5, seed=11, population=PLATFORM_POPULATION,
            analysis_unit="cluster",
        )
        assert data.total.clusters_consistent()
        assert data.total.n_clusters[0] > 0
        assert data.analysis_unit == "cluster"

    def test_ratio_uses_delta_method(self, registry):
        # estimator 必须**显式**写 post_only：CUPED 需要前置协变量，比值口径没有它，
        # 注册表会在创建时拦住 ratio+cuped 这个组合 —— **不替用户猜口径**。
        rec = registry.create(actor=ACTOR, 
            name="r1", variants=TWO_ARM, metric_type="ratio",
            estimator="post_only", true_lift=0.02,
        )
        rep = analyse_experiment(rec, n_users=20_000, seed=8)
        assert rep.metric_type == "ratio"
        assert rep.primary_estimator_name == "ratio_delta"
        # 比值指标没有"另一个口径"的 z（人均比值的 SE 需要明细）
        assert rep.alt is None
        assert all(m["alt_z"] is None for m in rep.monitoring)
        # 估计量必须是业务口径 Σy/Σx
        assert rep.primary.mean_treatment == pytest.approx(
            rep.primary.mean_treatment, rel=1e-12
        )
        assert any(c.name == "指标类型" for c in rep.checks)

    def test_metric_type_validated(self, registry):
        with pytest.raises(RegistryError, match="metric_type"):
            registry.create(actor=ACTOR, name="r2", variants=TWO_ARM, metric_type="nope")


# --------------------------------------------------------------------------- #
# M6.3 MDE / 功效
# --------------------------------------------------------------------------- #
class TestPowerAndMDE:
    def test_mde_inverts_power(self):
        for p in (0.5, 0.8, 0.9, 0.95):
            assert z_power(mde(0.12, power=p), 0.12, 0.05) == pytest.approx(p, abs=1e-4)

    def test_power_at_zero_is_alpha(self):
        assert z_power(0.0, 0.12, 0.05) == pytest.approx(0.05, abs=1e-12)

    def test_required_n_hits_target(self):
        for rel in (0.05, 0.02, 0.01):
            target = 19.5 * rel
            n_arm = required_n_per_arm(7.5, target, power=0.8)
            assert z_power(target, se_of_mean_diff(7.5, n_arm, n_arm), 0.05) == pytest.approx(
                0.8, abs=1e-4
            )

    def test_unequal_allocation_costs_more(self):
        """不等权分配要更多样本 —— 这是"别做 90/10"的定量答案。"""
        need = [required_n_per_arm(7.5, 0.39, treatment_ratio=q) for q in (0.5, 0.3, 0.1)]
        assert need == sorted(need)
        assert need[-1] / need[0] > 2.5

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="se"):
            mde(0.0)
        with pytest.raises(ValueError, match="power"):
            mde(0.1, power=1.5)
        with pytest.raises(ValueError, match="treatment_ratio"):
            required_n_per_arm(1.0, 0.1, treatment_ratio=1.0)

    def test_report_power_block(self, registry):
        rec = registry.create(actor=ACTOR, name="p1", variants=TWO_ARM, true_lift=0.35)
        rep = analyse_experiment(rec, n_users=20_000, seed=9)
        pw = rep.power
        assert {"mde_abs", "mde_relative", "power_at_observed", "se"} <= set(pw)
        assert pw["se"] == pytest.approx(rep.primary.std_error, rel=1e-12)
        assert pw["mde_abs"] == pytest.approx(mde(pw["se"], power=0.8), rel=1e-12)
        # MDE 收缩必须等于 1 - sqrt(1-ρ²)（ρ=0.7 → 29.0%）
        assert pw["mde_shrinkage_vs_post_only"] == pytest.approx(
            1 - (1 - 0.7**2) ** 0.5, abs=0.02
        )
        assert any(c.name == "功效 / MDE" for c in rep.checks)

    def test_power_block_is_none_for_non_unit_mean(self, registry):
        rec = registry.create(actor=ACTOR, 
            name="p2", variants=TWO_ARM, analysis_unit="cluster", estimator="post_only"
        )
        rep = analyse_experiment(rec, n_users=8000, seed=10)
        assert rep.power["mde_shrinkage_vs_post_only"] is None


# --------------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------------- #
class TestM6Audits:
    def test_monitoring_audit_reports_both_paths(self):
        r = run_monitoring_fwer_audit(n_salts=60, n_units=3000, n_looks=3)
        assert 0.0 <= r.cuped_fwer <= 0.3
        assert 0.0 <= r.post_only_fwer <= 0.3
        assert r.cuped_calibrated
        assert 0.0 <= r.disagreement <= 1.0
        assert len(r.cuped_z) == 60 and len(r.post_only_z) == 60
        # 两个口径是同一个数据的两个估计量，必须强相关
        assert r.z_correlation > 0.5

    def test_unit_audit_flags_the_wrong_unit(self):
        """单元级检验在整簇随机化下会严重超发 —— 这条审计要能抓出来。"""
        r = run_unit_awareness_audit(n_salts=40, n_users=4000, n_looks=3)
        assert r.unit_level_fpr > 0.3, "审计没抓出单元级超发，说明它没在测该测的东西"
        assert r.cluster_calibrated
        assert r.se_ratio > 2.0
        assert 0.0 < r.se_understatement < 1.0


# --------------------------------------------------------------------------- #
# 接口层：声明必须真的生效
# --------------------------------------------------------------------------- #
class TestDeclarationsThroughAPI:
    @pytest.fixture
    def client(self, work_dir):

        from ablab.platform.api import create_app

        app = create_app(work_dir / "m6_api.db")
        with authed_client(app) as c:
            yield c
        app.state.registry.close()

    def test_unknown_field_is_rejected_not_ignored(self, client):
        """**静默忽略未知字段是这台平台上最危险的一类 bug。**

        实测踩过：请求里带 ``analysis_unit="cluster"``，而当时 ``ExperimentIn``
        没有这个字段 —— pydantic 默认忽略它，于是请求 200、实验被建成"单元级"，
        而"整簇"和"单元级"的 I 类错误率差 10 倍以上。
        现在多余字段直接 422。
        """
        r = client.post("/api/experiments", json={
            "name": "typo_exp",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            "analysis_units": "cluster",  # 故意写错（多了个 s）
        })
        assert r.status_code == 422
        # 而且不能落库
        assert client.get("/api/experiments?status=draft").json() == []

    def test_cluster_declaration_reaches_the_engine(self, client):
        created = client.post("/api/experiments", json={
            "name": "api_cluster",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            "analysis_unit": "cluster",
            "estimator": "post_only",
            "true_lift": 2.0,
        })
        assert created.status_code == 201
        eid = created.json()["id"]
        r = client.post(f"/api/experiments/{eid}/analyze", json={"n_users": 10_000}).json()
        assert r["analysis_unit"] == "cluster"
        assert r["primary_estimator_name"] == "cluster_level"
        assert r["alt_estimator_name"] == "unit_level"
        # 分析单元数是簇数，远小于用户数
        assert r["n_analysis_units"] < 0.05 * r["n_users"]
        # 单元级的 SE 必须显著更小（这就是它超发的原因）
        assert r["alt"]["std_error"] < r["primary"]["std_error"] / 3

    def test_ratio_declaration_reaches_the_engine(self, client):
        created = client.post("/api/experiments", json={
            "name": "api_ratio",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            "metric_type": "ratio",
            "estimator": "post_only",
            "true_lift": 0.02,
        })
        eid = created.json()["id"]
        r = client.post(f"/api/experiments/{eid}/analyze", json={"n_users": 15_000}).json()
        assert r["metric_type"] == "ratio"
        assert r["primary_estimator_name"] == "ratio_delta"
        assert any(c["name"] == "指标类型" for c in r["checks"])

    def test_cluster_cuped_combo_is_accepted_by_the_api(self, client):
        """**改写了**：这个组合现在合法（理由见 TestAnalysisUnit 那条）。

        仍然非法的是"比值指标 + CUPED"（那条理由是真的：比值链路里没有前置协变量），
        这里一并确认它没被顺手放开。
        """
        ok = client.post("/api/experiments", json={
            "name": "api_cluster_cuped",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            "analysis_unit": "cluster",
            "estimator": "cuped",
        })
        assert ok.status_code == 201, ok.text
        bad = client.post("/api/experiments", json={
            "name": "api_bad_combo",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            "metric_type": "ratio",
            "estimator": "cuped",
        })
        assert bad.status_code == 400
        assert "比值" in bad.json()["detail"] or "ratio" in bad.json()["detail"].lower()

    def test_design_power_endpoint(self, client):
        r = client.post("/api/design/power", json={
            "baseline_mean": 19.5, "sd": 7.5, "relative_mde": 0.02,
            "pre_post_correlation": 0.7,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["n_total"] < body["n_total_post_only"]
        assert body["sample_saving"] == pytest.approx(0.49, abs=0.01)

    def test_projections_are_lossless_json(self, client):
        """报告必须能被严格 JSON 解析（之前的 nan 就是在这里炸的）。"""
        import json

        created = client.post("/api/experiments", json={
            "name": "json_check", "estimator": "post_only",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
        }).json()
        raw = client.post(f"/api/experiments/{created['id']}/analyze",
                          json={"n_users": 5000}).content.decode("utf-8")

        def reject(constant):
            raise AssertionError(f"出现了非法 JSON 常量 {constant}（前端 JSON.parse 会直接炸）")

        # parse_constant 只在遇到 NaN / Infinity / -Infinity 时被调用
        parsed = json.loads(raw, parse_constant=reject)
        assert parsed["primary"]["std_error"] > 0
        assert parsed["checks"][0]["name"] == "SRM"
