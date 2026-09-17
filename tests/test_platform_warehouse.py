"""平台 × 数仓的集成测试。

与 ``test_platform.py`` 的分工：那边测的是**平台自己**（注册表、编排、接口），
这边测的是**接上真实数仓之后**才出现的东西 —— 数据源等价性、绑定语义、结构迁移。

一个会话级的小数仓 fixture
-------------------------
``WarehouseConfig(n_users=3000)`` 走完整的 ODS→DWD→DWS→ADS 链路，
几秒钟就能建好。**刻意不用生产那份 20000 人的数仓** ——
测试要的是链路口径，不是样本量。
"""

import sqlite3

import duckdb
import pytest
from fastapi.testclient import TestClient

from ablab.platform import (
    ExperimentRegistry,
    RegistryError,
    analyse_experiment,
    run_source_equivalence_audit,
)
from ablab.platform.api import create_app
from ablab.platform.datasource import build_warehouse_data, list_warehouse_experiments
from ablab.platform.demo import WAREHOUSE_DEMO, seed_demo

TWO_ARM = [
    {"name": "control", "weight": 0.5},
    {"name": "treatment", "weight": 0.5},
]
NINE_ONE = [
    {"name": "control", "weight": 0.9},
    {"name": "treatment", "weight": 0.1},
]


@pytest.fixture(scope="session")
def warehouse_path(project_root):
    """建一个小数仓，返回**文件路径**（不在 fixture 里持有连接）。

    为什么不顺手把连接 yield 出去：DuckDB 在同一进程内**不允许对同一个文件
    混用不同配置的连接** —— 建仓的连接是读写，平台按请求开的是只读，
    两者共存会直接抛 ``Can't open a connection to same database file with
    a different configuration``。所以建完就关，把文件交出去，
    后续谁要读谁自己开只读连接。
    """
    from ablab.warehouse import WarehouseConfig, build_warehouse

    base = project_root / "build" / "_test_tmp" / "warehouse_fixture"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "wh.duckdb"
    con = build_warehouse(
        path,
        base / "source",
        project_root / "sql",
        config=WarehouseConfig(n_users=3000),
        force_data=True,
        verbose=False,
    )
    con.close()
    return path


@pytest.fixture
def warehouse_con(warehouse_path):
    """只读连接 —— 与平台按请求开的连接配置一致。"""
    con = duckdb.connect(str(warehouse_path), read_only=True)
    yield con
    con.close()


@pytest.fixture
def registry(work_dir):
    reg = ExperimentRegistry(work_dir / "reg.db")
    yield reg
    reg.close()


# --------------------------------------------------------------------------- #
# 数据源层：合成路径
# --------------------------------------------------------------------------- #
class TestSyntheticSource:
    def test_last_look_equals_the_main_analysis(self, registry):
        """**最重要的一条不变量**：监控曲线的右端点必须与主结论逐位相同。

        这是 M6.1 的核心：判定的估计量必须与头条结论是同一个。
        默认口径是 CUPED，所以最后一次查看要等于 ``primary``（CUPED），
        而 ``alt``（post-only）是另一个口径、不该被拿来比。
        """
        rec = registry.create(name="ds1", variants=TWO_ARM, true_lift=0.3)
        rep = analyse_experiment(rec, n_users=6000, seed=1)
        last = rep.monitoring[-1]
        assert rep.primary_estimator_name == "cuped"
        assert last["estimator"] == "cuped"
        assert last["effect"] == rep.primary.absolute_effect
        assert last["std_error"] == rep.primary.std_error
        assert last["information_fraction"] == 1.0
        # 对照口径是另一回事，绝不能顺手相等
        assert last["alt_estimator"] == "post_only"
        assert last["alt_effect"] == rep.alt.absolute_effect

    def test_last_look_is_full_even_for_unequal_arms(self, registry):
        """90/10 分支下最容易写错的地方。

        如果按"两臂都取 min(n_t, n_c) 的前缀"来构造查看，小臂永远取不到全量，
        最后一次查看就不等于全量分析了。正确做法是**每臂内部**各自按比例取前缀。
        """
        rec = registry.create(name="ds2", variants=NINE_ONE)
        rep = analyse_experiment(rec, n_users=8000, seed=2)
        last = rep.monitoring[-1]
        assert last["n_treatment"] == rep.primary.n_treatment
        assert last["n_control"] == rep.primary.n_control
        assert last["std_error"] == rep.primary.std_error

    def test_information_fractions_are_uniform(self, registry):
        rec = registry.create(name="ds3", variants=TWO_ARM)
        rep = analyse_experiment(rec, n_users=4000, n_looks=8, seed=3)
        fracs = [m["information_fraction"] for m in rep.monitoring]
        assert fracs == [round(i / 8, 10) for i in range(1, 9)]

    def test_n_per_arm_is_monotone(self, registry):
        rec = registry.create(name="ds4", variants=NINE_ONE)
        rep = analyse_experiment(rec, n_users=6000, n_looks=5, seed=4)
        sizes = [m["n_per_arm"] for m in rep.monitoring]
        assert sizes == sorted(sizes)
        assert all(s >= 2 for s in sizes)

    def test_source_is_marked(self, registry):
        rec = registry.create(name="ds5", variants=TWO_ARM)
        rep = analyse_experiment(rec, n_users=2000, seed=5)
        assert rep.source == "synthetic"
        assert rep.population_size == 2000


# --------------------------------------------------------------------------- #
# 注册表：绑定语义与结构迁移
# --------------------------------------------------------------------------- #
class TestBinding:
    def test_create_with_binding(self, registry):
        rec = registry.create(name="b1", variants=TWO_ARM, warehouse_experiment="exp_rank_v2")
        assert rec.warehouse_experiment == "exp_rank_v2"
        assert registry.get(rec.id).warehouse_experiment == "exp_rank_v2"

    def test_bind_and_unbind(self, registry):
        rec = registry.create(name="b2", variants=TWO_ARM)
        assert rec.warehouse_experiment is None
        assert registry.bind_warehouse(rec.id, "  exp_rec_emb  ").warehouse_experiment == "exp_rec_emb"
        assert registry.bind_warehouse(rec.id, None).warehouse_experiment is None

    def test_blank_binding_rejected(self, registry):
        rec = registry.create(name="b3", variants=TWO_ARM)
        with pytest.raises(RegistryError, match="空白"):
            registry.bind_warehouse(rec.id, "   ")
        with pytest.raises(RegistryError, match="空白"):
            registry.create(name="b3b", variants=TWO_ARM, warehouse_experiment="  ")

    def test_binding_is_mutable_but_salt_is_not(self, registry):
        """绑定可变、salt 不可变 —— 两者的区别必须清晰。

        salt 决定了每个用户的分组，改它等于让已有数据报废；
        ``warehouse_experiment`` 只决定"从哪里读数"，不影响任何分组。
        """
        rec = registry.create(name="b4", variants=TWO_ARM)
        salt_before = rec.salt
        registry.bind_warehouse(rec.id, "exp_rank_v2")
        registry.bind_warehouse(rec.id, "exp_rec_emb")
        after = registry.get(rec.id)
        assert after.salt == salt_before, "绑定不该碰到 salt"
        assert after.warehouse_experiment == "exp_rec_emb"

    def test_migration_adds_column_to_old_db(self, work_dir):
        """老库（没有 warehouse_experiment 列）必须能被自动补齐。

        ``CREATE TABLE IF NOT EXISTS`` 对已存在的表什么都不做，
        缺了迁移就会在第一次写入时报 no such column。
        """
        legacy = work_dir / "legacy.db"
        con = sqlite3.connect(legacy)
        con.execute(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                hypothesis TEXT NOT NULL DEFAULT '', owner TEXT NOT NULL DEFAULT '',
                layer TEXT, unit TEXT NOT NULL DEFAULT 'user_id', salt TEXT NOT NULL,
                traffic_ratio REAL NOT NULL DEFAULT 1.0, variants TEXT NOT NULL,
                primary_metric TEXT NOT NULL DEFAULT 'metric',
                guardrails TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'draft', start_ds TEXT, end_ds TEXT,
                true_lift REAL NOT NULL DEFAULT 0.0, created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO experiments (id,name,salt,variants,created_at) "
            "VALUES ('old1','legacy','legacy_v1','[]','2026-01-01T00:00:00+00:00')"
        )
        con.commit()
        con.close()

        reg = ExperimentRegistry(legacy)
        try:
            # 老行读得出来，新列取默认值
            assert reg.get("old1").warehouse_experiment is None
            # 新行写得进去
            rec = reg.create(name="new_after_migration", variants=TWO_ARM)
            assert rec.warehouse_experiment is None
            assert reg.bind_warehouse(rec.id, "exp_rank_v2").warehouse_experiment == "exp_rank_v2"
        finally:
            reg.close()


# --------------------------------------------------------------------------- #
# 数仓路径
# --------------------------------------------------------------------------- #
class TestWarehouseSource:
    def test_lists_bindable_experiments(self, warehouse_con):
        items = list_warehouse_experiments(warehouse_con)
        names = {i["experiment"] for i in items}
        assert {"exp_rank_v2", "exp_rec_emb"} <= names
        for i in items:
            assert i["n_variants"] == 2 and i["n_users"] > 0

    def test_looks_are_daily_cumulative(self, warehouse_con):
        data = build_warehouse_data(warehouse_con, "exp_rank_v2", n_looks=5)
        assert data.source == "warehouse"
        fracs = [lk.information_fraction for lk in data.looks]
        assert fracs == sorted(fracs) and fracs[-1] == 1.0
        # 数仓路径的查看标签是**日期**，不是 "look k"
        assert all("（累计）" in lk.label for lk in data.looks)
        # 累计样本量单调增
        sizes = [lk.n_per_arm for lk in data.looks]
        assert sizes == sorted(sizes)

    def test_last_look_equals_total(self, warehouse_con):
        data = build_warehouse_data(warehouse_con, "exp_rec_emb", n_looks=5)
        assert data.total.treatment.n == data.counts["treatment"]
        assert data.total.control.n == data.counts["control"]

    def test_information_fractions_are_actual_not_calendar(self, warehouse_con):
        """信息比例必须取自实际累计样本量，不是日历天数。

        每天进入实验的人数并不相等，用天数比会高估早期信息量、让早期边界偏松。
        """
        data = build_warehouse_data(warehouse_con, "exp_rank_v2", n_looks=5)
        fracs = [lk.information_fraction for lk in data.looks]
        # 与等距目标有偏差（这就是"实际"的含义），但偏差不会大到离谱
        assert any(abs(f - i / 5) > 1e-6 for i, f in enumerate(fracs, start=1))
        assert all(abs(f - i / 5) < 0.12 for i, f in enumerate(fracs, start=1))

    def test_unknown_experiment_raises(self, warehouse_con):
        with pytest.raises(ValueError, match="找不到"):
            build_warehouse_data(warehouse_con, "no_such_experiment")

    def test_three_paths_agree(self, warehouse_con):
        """**头条证据**：三条读取路径给同一个答案。

        1. DWD 明细（``welch_ttest`` / ``cuped_ttest``）
        2. ADS 汇总（``ablab.warehouse.analyse_ads`` —— 平台存在之前就写好的独立实现）
        3. 平台编排（``analyse_experiment_from_warehouse``）

        1 和 2 是 M0 写的，拿它们当参照才有意义。
        """
        for experiment in ("exp_rank_v2", "exp_rec_emb"):
            r = run_source_equivalence_audit(warehouse_con, experiment, n_looks=5)
            assert r.agree, f"{experiment} 三条路径不一致，最大偏差 {r.max_deviation:.3e}"
            assert r.max_deviation < 1e-9
            assert r.last_look_matches
            # 换算成"相对偏差"看更直观：效应量级远大于 1e-9
            assert abs(r.from_platform[0]) > 1.0


# --------------------------------------------------------------------------- #
# 数仓路径：HTTP 接口
# --------------------------------------------------------------------------- #
@pytest.fixture
def wh_client(work_dir, warehouse_path):
    app = create_app(work_dir / "wh_api.db", warehouse_path=warehouse_path)
    seed_demo(app.state.registry, warehouse_available=True)
    with TestClient(app) as c:
        yield c
    app.state.registry.close()


@pytest.fixture
def plain_client(work_dir):
    """没配数仓的平台 —— 必须优雅退化，不是崩掉。"""
    app = create_app(work_dir / "plain_api.db")
    seed_demo(app.state.registry)
    with TestClient(app) as c:
        yield c
    app.state.registry.close()


class TestWarehouseAPI:
    def test_lists_warehouse_experiments(self, wh_client):
        body = wh_client.get("/api/warehouse/experiments").json()
        assert body["available"] is True
        names = {e["experiment"] for e in body["experiments"]}
        assert "exp_rank_v2" in names

    def test_without_warehouse_degrades_gracefully(self, plain_client):
        body = plain_client.get("/api/warehouse/experiments").json()
        assert body["available"] is False and body["experiments"] == []
        # 合成路径照常工作
        eid = plain_client.get("/api/experiments").json()[0]["id"]
        r = plain_client.post(f"/api/experiments/{eid}/analyze", json={"n_users": 3000})
        assert r.status_code == 200 and r.json()["source"] == "synthetic"

    def test_seed_adds_a_warehouse_bound_demo(self, wh_client):
        items = {i["name"]: i for i in wh_client.get("/api/experiments").json()}
        assert WAREHOUSE_DEMO["name"] in items
        assert items[WAREHOUSE_DEMO["name"]]["warehouse_experiment"] == "exp_rank_v2"

    def test_analyze_bound_record_reads_warehouse(self, wh_client):
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        r = wh_client.post(
            f"/api/experiments/{items[WAREHOUSE_DEMO['name']]}/analyze",
            json={"alpha": 0.05, "n_looks": 5},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "warehouse"
        assert body["population_size"] is None
        # 与 M0 数仓报告里的数字同源（真值 +2.0/天 × 14 天）
        assert body["cuped"]["absolute_effect"] > 0
        assert body["checks"][0]["name"] == "SRM"
        # 监控的查看标签是日期
        assert "（累计）" in body["monitoring"][0]["label"]

    def test_analyze_bound_record_rejects_seed(self, wh_client):
        """数仓数据是既成的，seed 没有意义 —— 明确报错，不静默忽略。"""
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        r = wh_client.post(
            f"/api/experiments/{items[WAREHOUSE_DEMO['name']]}/analyze",
            json={"seed": 7},
        )
        assert r.status_code == 400
        assert "seed" in r.json()["detail"]

    def test_bind_valid(self, wh_client):
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        r = wh_client.post(
            f"/api/experiments/{items['exp_rec_emb']}/bind",
            json={"warehouse_experiment": "exp_rec_emb"},
        )
        assert r.status_code == 200
        assert r.json()["warehouse_experiment"] == "exp_rec_emb"

    def test_bind_unknown_rolls_back(self, wh_client):
        """绑一个不存在的数仓实验必须当场失败并回滚。

        否则用户要到点"分析"时才发现 —— 那是更晚、更贵的反馈。
        """
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        eid = items["exp_rec_emb"]
        r = wh_client.post(
            f"/api/experiments/{eid}/bind", json={"warehouse_experiment": "nope"}
        )
        assert r.status_code == 400
        assert "没有实验" in r.json()["detail"]
        # 回滚检查：仍然是未绑定状态
        assert wh_client.get(f"/api/experiments/{eid}").json()["warehouse_experiment"] is None

    def test_unbind(self, wh_client):
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        eid = items[WAREHOUSE_DEMO["name"]]
        r = wh_client.post(f"/api/experiments/{eid}/bind", json={"warehouse_experiment": None})
        assert r.status_code == 200 and r.json()["warehouse_experiment"] is None
        # 解绑后回到合成路径，此时 seed 又生效了
        body = wh_client.post(
            f"/api/experiments/{eid}/analyze", json={"n_users": 3000, "seed": 1}
        ).json()
        assert body["source"] == "synthetic"

    def test_bind_without_warehouse_still_records(self, plain_client):
        """没配数仓时允许先记下绑定 —— 但分析会明确报错，而不是悄悄走合成数据。

        悄悄降级是最坏的选择：报告里写着"数仓链路"，数字其实是平台编的。
        """
        items = {i["name"]: i["id"] for i in plain_client.get("/api/experiments").json()}
        eid = items["exp_rec_emb"]
        assert plain_client.post(
            f"/api/experiments/{eid}/bind", json={"warehouse_experiment": "exp_rec_emb"}
        ).status_code == 200
        r = plain_client.post(f"/api/experiments/{eid}/analyze", json={})
        assert r.status_code == 400
        assert "数仓" in r.json()["detail"]
