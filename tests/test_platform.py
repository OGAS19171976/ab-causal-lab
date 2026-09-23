"""平台层测试：注册表校验、分析编排、HTTP 接口。

这一层的测试重点不是统计（那些在 M0–M4 已经测透了），
而是**三件工程上的事**：

1. 校验逻辑是否真的只有一份（注册表复用 ``ExperimentSpec``）
2. salt 是否真的不可变（改了等于把所有用户重新分组）
3. 分析接口是否返回**完整体检报告**而不是一个数
"""

import pytest
from fastapi.testclient import TestClient

from ablab.hashing import murmur3_32
from ablab.platform import (
    ExperimentRegistry,
    RegistryError,
    analyse_experiment,
    run_aa_validation,
)
from ablab.platform.api import create_app
from ablab.platform.demo import DEMO_EXPERIMENTS, seed_demo

#: 审计里的操作者（现在是必填的具名参数）。
ACTOR = "tester"

TWO_ARM = [
    {"name": "control", "weight": 0.5},
    {"name": "treatment", "weight": 0.5},
]


#: 读写接口都要凭据。测试里统一用这个助手：建一个 admin 用户，
#: 把 token 挂到 client 上（`client.headers`），这样各测试的调用点不用逐个改。
def authed_client(app):
    """给 TestClient 装上 admin 凭据。返回 client 本身。"""
    token = app.state.registry.add_user("test_admin", role="admin")
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture
def registry(work_dir):
    reg = ExperimentRegistry(work_dir / "reg.db")
    yield reg
    reg.close()


@pytest.fixture
def client(work_dir):
    app = create_app(work_dir / "api.db")
    seed_demo(app.state.registry)
    with authed_client(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_create_and_get(self, registry):
        rec = registry.create(actor=ACTOR, name="exp_a", variants=TWO_ARM, true_lift=0.3)
        assert rec.id and rec.salt == "exp_a_v1"
        assert registry.get(rec.id).name == "exp_a"
        assert registry.count() == 1

    def test_weights_must_sum_to_one(self, registry):
        """校验复用 M0 的 ExperimentSpec —— 不是在这层另写一遍。"""
        with pytest.raises(RegistryError, match="权重之和"):
            registry.create(actor=ACTOR, 
                name="bad", variants=[{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.4}]
            )
        assert registry.count() == 0, "校验失败时不应写入任何记录"

    def test_duplicate_name_rejected(self, registry):
        registry.create(actor=ACTOR, name="dup", variants=TWO_ARM)
        with pytest.raises(RegistryError, match="已存在"):
            registry.create(actor=ACTOR, name="dup", variants=TWO_ARM)

    def test_invalid_traffic_ratio(self, registry):
        with pytest.raises(RegistryError, match="traffic_ratio"):
            registry.create(actor=ACTOR, name="bad", variants=TWO_ARM, traffic_ratio=1.5)

    def test_default_salt_is_stable(self, registry):
        """salt 由名字派生，必须是确定的（否则每次重启都会重新分组）。"""
        a = registry.create(actor=ACTOR, name="s1", variants=TWO_ARM)
        assert a.salt == "s1_v1"

    def test_explicit_salt_used(self, registry):
        rec = registry.create(actor=ACTOR, name="s2", variants=TWO_ARM, salt="my_salt_v9")
        assert rec.salt == "my_salt_v9"

    def test_empty_salt_rejected(self, registry):
        with pytest.raises(RegistryError, match="salt 不能为空"):
            registry.create(actor=ACTOR, name="s3", variants=TWO_ARM, salt="   ")

    def test_status_transitions(self, registry):
        rec = registry.create(actor=ACTOR, name="s4", variants=TWO_ARM)
        assert rec.status == "draft"
        assert registry.set_status(rec.id, "running", actor=ACTOR).status == "running"
        with pytest.raises(RegistryError, match="status"):
            registry.set_status(rec.id, "nope", actor=ACTOR)

    def test_no_update_method_for_spec(self, registry):
        """**刻意不提供**修改分流定义的方法 —— 那会让已有数据报废。"""
        assert not hasattr(registry, "update")
        assert not hasattr(registry, "update_spec")

    def test_delete(self, registry):
        rec = registry.create(actor=ACTOR, name="s5", variants=TWO_ARM)
        registry.delete(rec.id, actor=ACTOR)
        with pytest.raises(RegistryError, match="找不到"):
            registry.get(rec.id)

    def test_get_missing_raises(self, registry):
        with pytest.raises(RegistryError, match="找不到"):
            registry.get("nope")

    def test_list_filter_by_status(self, registry):
        registry.create(actor=ACTOR, name="a", variants=TWO_ARM, status="draft")
        registry.create(actor=ACTOR, name="b", variants=TWO_ARM, status="running")
        assert len(registry.list()) == 2
        assert [r.name for r in registry.list(status="running")] == ["b"]

    def test_to_spec_roundtrip(self, registry):
        rec = registry.create(actor=ACTOR, name="s6", variants=TWO_ARM, traffic_ratio=0.4, layer="L")
        spec = rec.to_spec()
        assert spec.traffic_ratio == 0.4
        assert spec.layer == "L"
        assert spec.salt_ == "s6_v1"

    def test_persistence_across_connections(self, work_dir):
        p = work_dir / "persist.db"
        r1 = ExperimentRegistry(p)
        r1.create(actor=ACTOR, name="keepme", variants=TWO_ARM)
        r1.close()
        r2 = ExperimentRegistry(p)
        assert r2.count() == 1
        r2.close()


class TestDemoSeed:
    def test_seed_is_idempotent(self, registry):
        first = seed_demo(registry)
        assert first == len(DEMO_EXPERIMENTS)
        assert seed_demo(registry) == 0, "重复调用不应重复写入"

    def test_seed_includes_a_negative_control(self):
        """演示数据必须包含真实效应为零的负对照。"""
        assert any(e["true_lift"] == 0.0 for e in DEMO_EXPERIMENTS)

    def test_seed_force_overwrites(self, registry):
        seed_demo(registry)
        assert seed_demo(registry, force=True) == len(DEMO_EXPERIMENTS)


# --------------------------------------------------------------------------- #
# 分析编排
# --------------------------------------------------------------------------- #
class TestAnalysis:
    def test_report_shape(self, registry):
        rec = registry.create(actor=ACTOR, name="an1", variants=TWO_ARM, true_lift=0.4)
        rep = analyse_experiment(rec, n_users=3000, seed=1)
        assert rep.cuped is not None and rep.naive is not None
        assert rep.sequential is not None and len(rep.monitoring) == 5
        assert rep.health in ("pass", "warn", "fail")

    def test_srm_appears_exactly_once(self, registry):
        """回归测试：曾经显式加一次、CUPED 自己又加一次，导致重复。"""
        rec = registry.create(actor=ACTOR, name="an2", variants=TWO_ARM, true_lift=0.3)
        rep = analyse_experiment(rec, n_users=3000, seed=2)
        names = [c.name for c in rep.checks]
        assert names.count("SRM") == 1
        assert names[0] == "SRM", "SRM 必须排第一位"

    def test_cuped_beats_naive_se(self, registry):
        rec = registry.create(actor=ACTOR, name="an3", variants=TWO_ARM, true_lift=0.4)
        rep = analyse_experiment(rec, n_users=6000, seed=3)
        assert rep.cuped.std_error < rep.naive.std_error
        assert rep.cuped_fit.variance_reduction > 0.3

    def test_true_lift_shows_up(self, registry):
        """注入 0.4 的真实效应，CUPED 应当检出。

        前提是平台用的是**业务量纲**而不是验证台的 100/30 抽象量纲：
        若 post_sd=30，0.4 只有 0.013σ，再多样本也检不出来。
        """
        rec = registry.create(actor=ACTOR, name="an4", variants=TWO_ARM, true_lift=0.4)
        rep = analyse_experiment(rec, n_users=12_000, seed=4)
        assert rep.cuped.significant
        assert rep.cuped.absolute_effect > 0

    def test_default_seed_is_deterministic(self, registry):
        """回归测试：默认种子曾经用内置 ``hash()``，它每个进程都加盐，
        于是"重启一次演示的数字就全变了"。必须由 murmur3 确定性派生，
        而且要挂在 **salt**（不可变）上而不是 uuid 形式的 id 上。"""
        rec = registry.create(actor=ACTOR, name="an5b", variants=TWO_ARM, true_lift=0.2)
        expected = murmur3_32(rec.salt.encode("utf-8"))
        a = analyse_experiment(rec, n_users=1500, seed=None)
        b = analyse_experiment(rec, n_users=1500, seed=expected)
        assert a.cuped.absolute_effect == b.cuped.absolute_effect
        assert a.naive.absolute_effect == b.naive.absolute_effect

    def test_same_salt_survives_registry_rebuild(self, work_dir):
        """删库重建后（新 uuid、同 salt）演示数字必须一模一样。"""
        p = work_dir / "rebuild.db"
        r1 = ExperimentRegistry(p)
        first = analyse_experiment(
            r1.create(actor=ACTOR, name="rb", variants=TWO_ARM, true_lift=0.3), n_users=1500
        )
        r1.delete(r1.get_by_name("rb").id, actor=ACTOR)
        second = analyse_experiment(
            r1.create(actor=ACTOR, name="rb", variants=TWO_ARM, true_lift=0.3), n_users=1500
        )
        r1.close()
        assert first.cuped.absolute_effect == second.cuped.absolute_effect

    def test_zero_lift_false_positive_rate(self, registry):
        """真效应为零时的体检 —— 不追求"一定不显著"，而是看校准。

        单次运行当然可能显著（那就是 5% 的 I 类错误），
        所以这里跑 24 个种子看整体比例，而不是断言某一次的结果。
        """
        rec = registry.create(actor=ACTOR, name="an5", variants=TWO_ARM, true_lift=0.0)
        hits = 0
        ratios = []
        trials = 24
        for s in range(trials):
            rep = analyse_experiment(rec, n_users=2500, seed=100 + s)
            hits += int(rep.cuped.significant)
            ratios.append(rep.cuped.std_error / rep.naive.std_error)
        assert hits / trials <= 0.35, f"零效应下显著比例 {hits / trials:.3f} 明显超标"
        # 干净随机化下 CUPED 与 naive 点估计同源，唯一区别是 SE ——
        # 理论收缩 sqrt(1-0.7^2)=0.714，实测应当贴着它。
        mean_ratio = sum(ratios) / trials
        assert 0.60 < mean_ratio < 0.82, f"SE 收缩比 {mean_ratio:.3f} 偏离理论值 0.714"

    def test_sequential_path_is_consistent(self, registry):
        rec = registry.create(actor=ACTOR, name="an6", variants=TWO_ARM, true_lift=0.3)
        rep = analyse_experiment(rec, n_users=4000, seed=6, n_looks=5)
        fracs = [m["information_fraction"] for m in rep.monitoring]
        assert fracs == sorted(fracs)
        assert fracs[-1] == pytest.approx(1.0)
        bounds = [m["boundary"] for m in rep.monitoring]
        assert bounds == sorted(bounds, reverse=True), "OBF 边界应递减"
        # "任何时候看都有效"对应的量是运行最小值，它必须单调不增，
        # 且等于全部逐次 p 的最小值 —— 否则报出来的就不是 anytime-valid 的那个数。
        mins = [m["always_valid_p_running_min"] for m in rep.monitoring]
        assert mins == sorted(mins, reverse=True)
        assert mins[-1] == min(m["always_valid_p"] for m in rep.monitoring)

    def test_to_dict_is_json_safe(self, registry):
        """用 ``allow_nan=False``：nan/inf 会被序列化成非法 JSON 的 NaN/Infinity，
        前端的 ``JSON.parse`` 遇到就直接炸 —— 这里必须严格。"""
        import json

        rec = registry.create(actor=ACTOR, name="an7", variants=TWO_ARM, true_lift=0.2)
        rep = analyse_experiment(rec, n_users=2000, seed=7)
        json.dumps(rep.to_dict(), allow_nan=False)

    def test_small_sample_raises_clear_error(self, registry):
        """回归测试：每组样本太少时 (n-1)=0 会算出 nan 而不是报错。"""
        rec = registry.create(actor=ACTOR, 
            name="an9",
            variants=[{"name": "control", "weight": 0.9}, {"name": "treatment", "weight": 0.1}],
            traffic_ratio=0.3,
        )
        with pytest.raises(ValueError, match="不足以构造"):
            analyse_experiment(rec, n_users=200, n_looks=5, seed=9)

    def test_multi_arm_monitoring_uses_only_the_two_compared_arms(self, registry):
        """回归测试：序贯监控曾经用 ``~treated`` 当对照组，
        于是三臂实验会把**中间臂和未进组的人**也算进对照组，
        而主口径是"最后一臂 vs 第一臂" —— 两条路径看的不是同一件事。
        """
        rec = registry.create(actor=ACTOR, 
            name="an13",
            variants=[
                {"name": "control", "weight": 0.4},
                {"name": "mid", "weight": 0.3},
                {"name": "treatment", "weight": 0.3},
            ],
        )
        rep = analyse_experiment(rec, n_users=6000, seed=13)
        # 主口径只用了第 0 臂与第 2 臂
        assert rep.cuped.n_treatment + rep.cuped.n_control < 6000
        # 监控每次查看的每组样本量 = min(第 2 臂, 第 0 臂)，而不是 min(第 2 臂, 其余全部)
        assert rep.monitoring[-1]["n_per_arm"] == min(
            rep.cuped.n_treatment, rep.cuped.n_control
        )
        sizes = [m["n_per_arm"] for m in rep.monitoring]
        assert sizes == sorted(sizes), "每次查看的样本量只增不减"

    def test_monitoring_is_finite_or_raises(self, registry):
        """绝不允许返回 nan/inf：要么给出有限值，要么明确报错。

        这条比"某个具体样本量报 400"更本质 —— 它守的是边界条件本身，
        而边界条件正是当初算出 nan 的地方（(n-1)=0）。
        """
        import numpy as np

        rec = registry.create(actor=ACTOR, 
            name="an12",
            variants=[{"name": "control", "weight": 0.9}, {"name": "treatment", "weight": 0.1}],
            traffic_ratio=0.3,
        )
        keys = (
            "n_per_arm", "z", "boundary", "effect", "std_error",
            "always_valid_p", "always_valid_p_running_min",
        )
        checked = 0
        for n in (200, 260, 340, 500, 800):
            for k in (2, 5, 10, 20):
                try:
                    rep = analyse_experiment(rec, n_users=n, n_looks=k, seed=3)
                except ValueError:
                    continue  # 明确报错是允许的
                for m in rep.monitoring:
                    for key in keys:
                        assert np.isfinite(m[key]), f"n={n} k={k} 的 {key} 不是有限值"
                checked += 1
        assert checked >= 5, "样本量组合几乎全部被拒，覆盖不到边界"

    def test_effect_decomposition_is_exact(self, registry):
        """naive − CUPED 必须**恰好**等于 θ̂·ΔX̄，也就是协变量失衡的贡献。

        这个恒等式是平台上"显著效应里有多少只是处置前就不平衡"的唯一依据，
        算错了整个分解就没有意义。
        """
        rec = registry.create(actor=ACTOR, name="an10", variants=TWO_ARM, true_lift=0.3)
        rep = analyse_experiment(rec, n_users=5000, seed=11)
        assert rep.residual_component == rep.cuped.absolute_effect
        assert rep.imbalance_component + rep.residual_component == pytest.approx(
            rep.naive.absolute_effect, abs=1e-12
        )
        # θ̂·ΔX̄ = (naive − CUPED) 本身
        assert rep.imbalance_component == pytest.approx(
            rep.naive.absolute_effect - rep.cuped.absolute_effect, abs=1e-12
        )
        assert any(c.name == "效应分解" for c in rep.checks)

    def test_decomposition_sign_matches_balance_check(self, registry):
        """失衡贡献的方向必须和协变量平衡诊断给出的方向一致 ——
        两者同号才说明分解与体检说的是同一件事，而不是各算各的。"""
        rec = registry.create(actor=ACTOR, name="an11", variants=TWO_ARM, true_lift=0.0)
        for s in range(6):
            rep = analyse_experiment(rec, n_users=4000, seed=500 + s)
            balance = next(c for c in rep.checks if c.name == "协变量平衡")
            if abs(rep.imbalance_component) < 1e-9:
                continue
            assert (rep.imbalance_component > 0) == (balance.statistic > 0)

    def test_traffic_ratio_respected(self, registry):
        rec = registry.create(actor=ACTOR, name="an8", variants=TWO_ARM, traffic_ratio=0.5)
        rep = analyse_experiment(rec, n_users=4000, seed=8)
        total = rep.cuped.n_treatment + rep.cuped.n_control
        assert total < 4000 * 0.6, "半流量实验不应把所有人都收进来"


class TestAAValidation:
    def test_returns_calibration_evidence(self):
        r = run_aa_validation(n_trials=200, n_units=3000, seed=0)
        assert 0.0 <= r["empirical_fpr"] <= 0.2
        assert r["fpr_interval"][0] < r["fpr_interval"][1]
        assert "现场快速检查" in r["note"]

    def test_result_is_reproducible(self):
        a = run_aa_validation(n_trials=100, n_units=2000, seed=1)
        b = run_aa_validation(n_trials=100, n_units=2000, seed=1)
        assert a["empirical_fpr"] == b["empirical_fpr"]


# --------------------------------------------------------------------------- #
# HTTP 接口
# --------------------------------------------------------------------------- #
class TestAPI:
    def test_healthz(self, client):
        r = client.get("/healthz").json()
        assert r["status"] == "ok" and r["experiments"] == len(DEMO_EXPERIMENTS)

    def test_index_is_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "ab-causal-lab" in r.text

    def test_openapi_available(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/experiments" in spec["paths"]

    def test_list(self, client):
        items = client.get("/api/experiments").json()
        assert len(items) == len(DEMO_EXPERIMENTS)
        assert {i["name"] for i in items} == {e["name"] for e in DEMO_EXPERIMENTS}

    def test_list_bad_status(self, client):
        assert client.get("/api/experiments?status=nope").status_code == 400

    def test_create(self, client):
        r = client.post(
            "/api/experiments",
            json={"name": "api_new", "variants": TWO_ARM, "true_lift": 0.25},
        )
        assert r.status_code == 201
        assert r.json()["salt"] == "api_new_v1"

    def test_create_validation_error(self, client):
        r = client.post(
            "/api/experiments",
            json={"name": "bad", "variants": [{"name": "a", "weight": 0.3}]},
        )
        assert r.status_code == 400
        assert "权重之和" in r.json()["detail"]

    def test_create_duplicate(self, client):
        r = client.post(
            "/api/experiments", json={"name": DEMO_EXPERIMENTS[0]["name"], "variants": TWO_ARM}
        )
        assert r.status_code == 400

    def test_get_missing_is_404(self, client):
        assert client.get("/api/experiments/nope").status_code == 404

    def test_status_patch(self, client):
        eid = client.get("/api/experiments").json()[0]["id"]
        r = client.patch(f"/api/experiments/{eid}/status", json={"status": "stopped"})
        assert r.status_code == 200 and r.json()["status"] == "stopped"

    def test_delete(self, client):
        eid = client.get("/api/experiments").json()[0]["id"]
        assert client.delete(f"/api/experiments/{eid}").status_code == 204
        assert client.get(f"/api/experiments/{eid}").status_code == 404

    def test_analyze_returns_full_report(self, client):
        eid = client.get("/api/experiments").json()[0]["id"]
        r = client.post(f"/api/experiments/{eid}/analyze", json={"n_users": 3000})
        assert r.status_code == 200
        body = r.json()
        # 核心断言：返回的是**体检报告**，不是一个效应值
        assert {"health", "checks", "cuped", "naive", "sequential", "monitoring"} <= set(body)
        assert body["checks"][0]["name"] == "SRM"
        assert len(body["monitoring"]) == 5

    def test_analyze_missing_is_404(self, client):
        assert client.post("/api/experiments/nope/analyze", json={}).status_code == 404

    def test_analyze_rejects_bad_params(self, client):
        eid = client.get("/api/experiments").json()[0]["id"]
        assert client.post(
            f"/api/experiments/{eid}/analyze", json={"alpha": 0.9}
        ).status_code == 422

    def test_analyze_n_looks_bounds(self, client):
        eid = client.get("/api/experiments").json()[0]["id"]
        assert client.post(
            f"/api/experiments/{eid}/analyze", json={"n_looks": 50}
        ).status_code == 422

    def test_analyze_explicit_seed_is_reproducible(self, client):
        eid = client.get("/api/experiments").json()[0]["id"]
        body = {"n_users": 3000, "seed": 42}
        a = client.post(f"/api/experiments/{eid}/analyze", json=body).json()
        b = client.post(f"/api/experiments/{eid}/analyze", json=body).json()
        assert a["cuped"]["absolute_effect"] == b["cuped"]["absolute_effect"]
        assert a["monitoring"] == b["monitoring"]

    def test_analyze_small_sample_returns_400(self, client):
        """守卫要求每组至少 2·n_looks 个样本。

        这个配置在 n_users=200 时每组只有 10 个，所以把 n_looks 提到 20
        （需要 40 个）才明确越界 —— 用 5 次会**正好压在边界上**而不报错，
        那样测的就不是守卫，而是运气。
        """
        items = {i["name"]: i["id"] for i in client.get("/api/experiments").json()}
        r = client.post(
            f"/api/experiments/{items['exp_ui_density']}/analyze",
            json={"n_users": 200, "n_looks": 20},
        )
        assert r.status_code == 400
        assert "不足以构造" in r.json()["detail"]

    def test_analyze_negative_control_shows_both_methods(self, client):
        """负对照的要点不是"naive 中招而 CUPED 不中招"（干净随机化下那是反的），
        而是两种口径同时给出区间、且 CUPED 的区间更窄。"""
        items = {i["name"]: i["id"] for i in client.get("/api/experiments").json()}
        body = client.post(
            f"/api/experiments/{items['exp_rec_emb']}/analyze", json={"n_users": 20000}
        ).json()
        assert body["naive"] is not None and body["cuped"] is not None
        assert body["cuped"]["std_error"] < body["naive"]["std_error"]
        assert body["cuped_fit"]["variance_reduction"] > 0.3

    def test_validate_aa_endpoint(self, client):
        r = client.post("/api/validate/aa", json={"n_trials": 120, "n_units": 2000})
        assert r.status_code == 200
        assert "empirical_fpr" in r.json()
