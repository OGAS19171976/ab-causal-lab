"""治理功能的测试：操作审计（append-only）+ 护栏指标显式化。

为什么单独一个文件：这两件事都不是统计方法，而是**平台治理**——
"谁改了什么有没有留痕"、"声明了但没人看的东西有没有被说出来"。
它们验证的是"这个平台会不会静默地骗人"，值得有自己的名字。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ablab.platform.analysis import analyse_experiment
from ablab.platform.api import create_app
from ablab.platform.registry import ExperimentRecord, ExperimentRegistry, RegistryError

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}]


@pytest.fixture()
def registry():
    reg = ExperimentRegistry(":memory:")
    yield reg
    reg.close()


def make(reg: ExperimentRegistry, name: str = "audit_demo", **kwargs) -> ExperimentRecord:
    return reg.create(name=name, variants=VARIANTS, salt=f"{name}_v1", **kwargs)


class TestAuditTrail:
    """"谁改了什么"必须留痕，而且**改不掉**。"""

    def test_create_is_recorded(self, registry):
        rec = make(registry)
        events = registry.events(rec.id)
        assert [e.action for e in events] == ["create"]
        assert events[0].after == rec.name

    def test_every_mutation_is_recorded_with_before_and_after(self, registry):
        rec = make(registry)
        registry.set_status(rec.id, "running")
        registry.set_estimator(rec.id, "post_only")
        registry.bind_warehouse(rec.id, "exp_rank_v2")

        events = registry.events(rec.id)
        assert [e.action for e in events] == [
            "create", "set_status", "set_estimator", "bind_warehouse",
        ]
        by_action = {e.action: e for e in events}
        assert (by_action["set_status"].before, by_action["set_status"].after) == (
            "draft", "running",
        )
        assert (by_action["set_estimator"].before, by_action["set_estimator"].after) == (
            "cuped", "post_only",
        )
        assert by_action["bind_warehouse"].after == "exp_rank_v2"

    def test_estimator_change_is_marked_as_a_judgement_rule_change(self, registry):
        """改判定口径是最需要留痕的一类改动 —— 备注里必须说清后果。"""
        rec = make(registry)
        registry.set_estimator(rec.id, "post_only")
        event = next(e for e in registry.events(rec.id) if e.action == "set_estimator")
        assert "判定规则" in event.note, event.note

    def test_audit_survives_deletion(self, registry):
        """删掉实验之后审计必须还在 —— 那正是最需要回答"谁删的"的时候。"""
        rec = make(registry)
        registry.set_status(rec.id, "stopped")
        registry.delete(rec.id)

        with pytest.raises(RegistryError):
            registry.get(rec.id)  # 实验确实没了
        events = registry.events(rec.id)
        assert [e.action for e in events] == ["create", "set_status", "delete"]
        assert events[-1].note, "删除事件要留下说明"

    def test_recent_events_are_newest_first(self, registry):
        a = make(registry, "exp_a")
        b = make(registry, "exp_b")
        registry.set_status(b.id, "running")
        recent = registry.recent_events(limit=3)
        assert recent[0].experiment_id == b.id
        assert recent[0].action == "set_status"
        assert {e.experiment_id for e in recent} == {a.id, b.id}

    def test_events_are_append_only_enforced_by_the_database(self, registry):
        """**不是"我们不写 UPDATE"，是写不动。** 用触发器挡，并有测试证明它挡得住。

        靠约定的不可变性迟早会被某次维护脚本破坏；而审计被改过之后
        没有任何办法看出来 —— 它自己就是"有没有被动过"的依据。
        """
        rec = make(registry)
        for sql in (
            "UPDATE experiment_events SET after = 'x' WHERE seq = 1",
            "DELETE FROM experiment_events WHERE seq = 1",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                registry._conn.execute(sql)
        assert len(registry.events(rec.id)) == 1

    def test_failed_write_leaves_no_event(self, registry):
        """校验失败的写入不能留下审计 —— 否则审计会记下"没发生的事"。

        （``set_estimator`` 先校验再写，所以非法值在碰到事务之前就被拒了。）
        """
        rec = make(registry)
        with pytest.raises(RegistryError):
            registry.set_estimator(rec.id, "not_an_estimator")
        assert [e.action for e in registry.events(rec.id)] == ["create"]

    def test_events_survive_reopening_the_database(self, tmp_path):
        """落盘之后再打开，审计还在（真实用法是文件库，不是内存库）。"""
        path = tmp_path / "registry.db"
        reg = ExperimentRegistry(path)
        rec = make(reg)
        reg.set_status(rec.id, "running")
        reg.close()

        reopened = ExperimentRegistry(path)
        try:
            assert [e.action for e in reopened.events(rec.id)] == ["create", "set_status"]
        finally:
            reopened.close()


class TestAuditAPI:
    @staticmethod
    def _client(tmp_path: Path):
        """走真实的建应用路径（``create_app(库路径)``），而不是自己拼一个 registry。

        这样接口测的是**产品里那套装配**，不是测试自己接的一根线。
        """
        from fastapi.testclient import TestClient

        return TestClient(create_app(tmp_path / "audit_api.db"))

    def test_events_endpoint_lists_history(self, tmp_path):
        client = self._client(tmp_path)
        rec = client.post(
            "/api/experiments",
            json={"name": "api_audit", "variants": VARIANTS, "salt": "api_audit_v1"},
        ).json()
        client.patch(f"/api/experiments/{rec['id']}/status", json={"status": "running"})

        payload = client.get(f"/api/experiments/{rec['id']}/events").json()
        assert payload["count"] == 2
        assert [e["action"] for e in payload["events"]] == ["create", "set_status"]

        recent = client.get("/api/events?limit=1").json()
        assert recent["count"] == 1
        assert recent["events"][0]["action"] == "set_status"

    def test_events_endpoint_still_works_after_delete(self, tmp_path):
        """删掉之后审计接口要还能用 —— 不能因为"实验不存在"就 404。"""
        client = self._client(tmp_path)
        rec = client.post(
            "/api/experiments",
            json={"name": "api_audit_del", "variants": VARIANTS, "salt": "api_audit_del_v1"},
        ).json()
        assert client.delete(f"/api/experiments/{rec['id']}").status_code == 204

        resp = client.get(f"/api/experiments/{rec['id']}/events")
        assert resp.status_code == 200
        assert [e["action"] for e in resp.json()["events"]] == ["create", "delete"]


class TestGuardrailVisibility:
    """声明了护栏却没人分析 —— 这件事必须在报告里**明说**。"""

    @staticmethod
    def record(name: str, guardrails: list[str]) -> ExperimentRecord:
        return ExperimentRecord(
            name=name,
            variants=list(VARIANTS),
            salt=f"{name}_v1",
            primary_metric="post_metric_14d",
            guardrails=guardrails,
        )

    def test_declared_guardrails_are_reported_as_not_analysed(self):
        rep = analyse_experiment(
            self.record("g_declared", ["latency_p99", "crash_rate"]), n_users=4_000
        )
        item = next((c for c in rep.checks if c.name == "护栏指标"), None)
        assert item is not None, [c.name for c in rep.checks]
        assert item.statistic == 2.0
        assert "latency_p99" in item.message and "crash_rate" in item.message
        assert "尚不分析" in item.message, "必须明说没有分析，而不是含糊其辞"

    def test_guardrail_notice_does_not_raise_health(self):
        """它不该把 health 拉成 warn。

        理由：护栏未接入是**平台级**缺口，声明了护栏的每个实验都会一直 warn，
        而"一条永远亮的告警等于没有告警" —— health 会因此失去意义。
        信息要显式（永远在报告里），但不占用"这次运行有问题"这个信号。
        """
        rep = analyse_experiment(self.record("g_health", ["latency_p99"]), n_users=4_000)
        item = next(c for c in rep.checks if c.name == "护栏指标")
        assert item.status == "info"
        assert rep.health == "pass", rep.health

    def test_no_guardrails_no_notice(self):
        rep = analyse_experiment(self.record("g_none", []), n_users=4_000)
        assert not [c for c in rep.checks if c.name == "护栏指标"]

    def test_notice_reaches_the_api_payload(self):
        """报告字典是 API 返回的东西 —— 那条声明要能在它里面看到。"""
        rep = analyse_experiment(self.record("g_api", ["scroll_depth"]), n_users=4_000)
        names = [c["name"] for c in rep.to_dict()["checks"]]
        assert "护栏指标" in names
