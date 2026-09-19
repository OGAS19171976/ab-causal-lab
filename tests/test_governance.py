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
from ablab.platform.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    RegistryConflict,
    RegistryError,
)

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}]
#: 审计里的操作者现在是**必填**的具名参数（忘了记是谁会在调用点报错）。
ACTOR = "tester"


@pytest.fixture()
def registry():
    reg = ExperimentRegistry(":memory:")
    yield reg
    reg.close()


def make(reg: ExperimentRegistry, name: str = "audit_demo", **kwargs) -> ExperimentRecord:
    return reg.create(
        actor=ACTOR, name=name, variants=VARIANTS, salt=f"{name}_v1", **kwargs
    )


class TestAuditTrail:
    """"谁改了什么"必须留痕，而且**改不掉**。"""

    def test_create_is_recorded(self, registry):
        rec = make(registry)
        events = registry.events(rec.id)
        assert [e.action for e in events] == ["create"]
        assert events[0].after == rec.name

    def test_every_mutation_is_recorded_with_before_and_after(self, registry):
        rec = make(registry)
        registry.set_status(rec.id, "running", actor=ACTOR)
        registry.set_estimator(rec.id, "post_only", actor=ACTOR)
        registry.bind_warehouse(rec.id, "exp_rank_v2", actor=ACTOR)

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
        registry.set_estimator(rec.id, "post_only", actor=ACTOR)
        event = next(e for e in registry.events(rec.id) if e.action == "set_estimator")
        assert "判定规则" in event.note, event.note

    def test_audit_survives_deletion(self, registry):
        """删掉实验之后审计必须还在 —— 那正是最需要回答"谁删的"的时候。"""
        rec = make(registry)
        registry.set_status(rec.id, "stopped", actor=ACTOR)
        registry.delete(rec.id, actor=ACTOR)

        with pytest.raises(RegistryError):
            registry.get(rec.id)  # 实验确实没了
        events = registry.events(rec.id)
        assert [e.action for e in events] == ["create", "set_status", "delete"]
        assert events[-1].note, "删除事件要留下说明"

    def test_recent_events_are_newest_first(self, registry):
        a = make(registry, "exp_a")
        b = make(registry, "exp_b")
        registry.set_status(b.id, "running", actor=ACTOR)
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
            registry.set_estimator(rec.id, "not_an_estimator", actor=ACTOR)
        assert [e.action for e in registry.events(rec.id)] == ["create"]

    def test_events_survive_reopening_the_database(self, tmp_path):
        """落盘之后再打开，审计还在（真实用法是文件库，不是内存库）。"""
        path = tmp_path / "registry.db"
        reg = ExperimentRegistry(path)
        rec = make(reg)
        reg.set_status(rec.id, "running", actor=ACTOR)
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
        返回 ``(client, auth)``：写接口现在需要凭据，``auth`` 是一个 admin 的请求头。
        """
        from fastapi.testclient import TestClient

        client = TestClient(create_app(tmp_path / "audit_api.db"))
        token = client.app.state.registry.add_user("api_admin", role="admin")
        return client, {"Authorization": f"Bearer {token}"}

    def test_events_endpoint_lists_history(self, tmp_path):
        client, auth = self._client(tmp_path)
        rec = client.post(
            "/api/experiments",
            json={"name": "api_audit", "variants": VARIANTS, "salt": "api_audit_v1"},
            headers=auth,
        ).json()
        client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "running"},
            headers=auth,
        )

        payload = client.get(f"/api/experiments/{rec['id']}/events").json()
        assert payload["count"] == 2
        assert [e["action"] for e in payload["events"]] == ["create", "set_status"]
        # 审计里现在有操作者，而且来自凭据（api_admin 这个用户），不是请求里写的
        assert {e["actor"] for e in payload["events"]} == {"api_admin"}

        recent = client.get("/api/events?limit=1").json()
        assert recent["count"] == 1
        assert recent["events"][0]["action"] == "set_status"

    def test_events_endpoint_still_works_after_delete(self, tmp_path):
        """删掉之后审计接口要还能用 —— 不能因为"实验不存在"就 404。"""
        client, auth = self._client(tmp_path)
        rec = client.post(
            "/api/experiments",
            json={"name": "api_audit_del", "variants": VARIANTS, "salt": "api_audit_del_v1"},
            headers=auth,
        ).json()
        assert (
            client.delete(f"/api/experiments/{rec['id']}", headers=auth).status_code == 204
        )

        resp = client.get(f"/api/experiments/{rec['id']}/events")
        assert resp.status_code == 200
        assert [e["action"] for e in resp.json()["events"]] == ["create", "delete"]


class TestAuthAndActor:
    """身份：**服务端从凭据推导**，请求里说的名字一律不算数。

    这一组测试要钉的不是"能不能登录"，而是三件更要紧的事：

    1. 审计里的操作者**不可能被请求方指定**（伪造尝试必须无效）；
    2. 被拒绝的操作**不写审计**（否则审计会被失败的尝试淹没）；
    3. **每一个写路由都要凭据** —— 而这件事由"自动枚举路由"来保证，
       而不是靠人记得在新增端点时加一行（那种保证迟早会漏）。
    """

    @staticmethod
    def _app(tmp_path: Path):
        from fastapi.testclient import TestClient

        client = TestClient(create_app(tmp_path / "auth.db"))
        reg = client.app.state.registry
        tokens = {
            "admin": reg.add_user("boss", role="admin"),
            "editor": reg.add_user("alice", role="editor"),
            "viewer": reg.add_user("bob", role="viewer"),
        }
        return client, tokens

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_missing_or_wrong_credentials_get_401(self, tmp_path):
        client, tokens = self._app(tmp_path)
        body = {"name": "e_auth", "variants": VARIANTS, "salt": "e_auth_v1"}
        assert client.post("/api/experiments", json=body).status_code == 401
        for header in (
            "Bearer nope",
            tokens["editor"],  # 少了 Bearer 前缀
            "Basic YWxpY2U6cHc=",  # 换了认证方案
        ):
            resp = client.post(
                "/api/experiments", json=body, headers={"Authorization": header}
            )
            assert resp.status_code == 401, header

    def test_viewer_cannot_write_and_editor_cannot_delete(self, tmp_path):
        client, tokens = self._app(tmp_path)
        rec = client.post(
            "/api/experiments",
            json={"name": "e_roles", "variants": VARIANTS, "salt": "e_roles_v1"},
            headers=self._auth(tokens["editor"]),
        ).json()
        # viewer：连建实验都不行
        assert (
            client.post(
                "/api/experiments",
                json={"name": "e_viewer", "variants": VARIANTS, "salt": "e_viewer_v1"},
                headers=self._auth(tokens["viewer"]),
            ).status_code
            == 403
        )
        # editor：能改状态，但不能删（删是 admin 的事）
        assert (
            client.patch(
                f"/api/experiments/{rec['id']}/status",
                json={"status": "running"},
                headers=self._auth(tokens["editor"]),
            ).status_code
            == 200
        )
        assert (
            client.delete(
                f"/api/experiments/{rec['id']}", headers=self._auth(tokens["editor"])
            ).status_code
            == 403
        )
        assert (
            client.delete(
                f"/api/experiments/{rec['id']}", headers=self._auth(tokens["admin"])
            ).status_code
            == 204
        )

    def test_disabled_user_is_locked_out_immediately(self, tmp_path):
        client, tokens = self._app(tmp_path)
        client.app.state.registry.disable_user("alice")
        resp = client.post(
            "/api/experiments",
            json={"name": "e_dis", "variants": VARIANTS, "salt": "e_dis_v1"},
            headers=self._auth(tokens["editor"]),
        )
        assert resp.status_code == 401

    def test_actor_cannot_be_spoofed_by_the_request(self, tmp_path):
        """**伪造尝试必须无效**：请求里写谁都不算数，只认凭据。

        这一条是这一整轮存在的理由。如果 actor 能从请求体里读，
        审计表就会变成"看起来权威的假证据"—— 比没有更糟。
        所以这里同时试三种伪造：请求体里的字段、附加 header、query 参数。
        """
        client, tokens = self._app(tmp_path)
        # 1) 请求体里塞一个 actor 字段：_Strict 直接 422，根本进不了审计
        assert (
            client.post(
                "/api/experiments",
                json={
                    "name": "e_spoof",
                    "variants": VARIANTS,
                    "salt": "e_spoof_v1",
                    "owner": "boss",
                    "actor": "boss",
                },
                headers={**self._auth(tokens["editor"]), "X-Actor": "boss"},
            ).status_code
            == 422
        )
        # 2) query 参数 + 附加 header 都写 boss：审计里仍然必须是 alice
        rec = client.post(
            "/api/experiments?actor=boss",
            json={
                "name": "e_spoof2",
                "variants": VARIANTS,
                "salt": "e_spoof2_v1",
                "owner": "boss",
            },
            headers={**self._auth(tokens["editor"]), "X-Actor": "boss"},
        ).json()
        events = client.get(f"/api/experiments/{rec['id']}/events").json()["events"]
        assert [e["actor"] for e in events] == ["alice"], events

    def test_rejected_writes_leave_no_audit_row(self, tmp_path):
        """被拒的操作不写审计 —— 否则"谁改了什么"要翻十页失败的尝试才看得见。"""
        client, tokens = self._app(tmp_path)
        body = {"name": "e_rej", "variants": VARIANTS, "salt": "e_rej_v1"}
        assert client.post("/api/experiments", json=body).status_code == 401
        assert (
            client.post(
                "/api/experiments",
                json={**body, "name": "e_rej2"},
                headers=self._auth(tokens["viewer"]),
            ).status_code
            == 403
        )
        assert client.get("/api/events").json()["count"] == 0

    def test_every_mutating_route_needs_a_token(self, tmp_path):
        """**自动枚举所有写路由**，逐个断言无凭据时是 401。

        这是防"新加了一个端点但忘了鉴权"的唯一可靠办法：
        人会在新增端点时忘记加校验，但这条测试会在 CI 上直接红。
        免检名单为空 —— 本仓库没有"本来就该公开"的写接口。
        """
        client, _tokens = self._app(tmp_path)
        exempt: set[str] = set()
        checked = 0
        for route in client.app.routes:
            methods = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            mutating = methods & {"POST", "PATCH", "PUT", "DELETE"}
            if not mutating or path in exempt or not path.startswith("/api"):
                continue
            for method in sorted(mutating):
                resp = client.request(
                    method, path.replace("{experiment_id}", "x"), json={}
                )
                assert resp.status_code == 401, (method, path, resp.status_code)
                checked += 1
        assert checked >= 5, f"只检查到 {checked} 个写路由，枚举逻辑可能失效了"

    def test_stored_credentials_are_hashes_not_tokens(self, tmp_path):
        """库里存的是 sha256，不是 token —— 库文件泄露不等于凭据泄露。"""
        client, tokens = self._app(tmp_path)
        reg = client.app.state.registry
        rows = reg._conn.execute("SELECT id, token_hash FROM users").fetchall()
        assert rows
        for row in rows:
            assert len(row["token_hash"]) == 64
        for token in tokens.values():
            hits = reg._conn.execute(
                "SELECT COUNT(*) AS c FROM users WHERE token_hash = ?", (token,)
            ).fetchone()["c"]
            assert hits == 0
        # list_users 也不能把哈希漏出去
        assert all("token" not in u for u in reg.list_users())

    def test_pre_migration_events_show_unknown_actor(self, tmp_path):
        """老库（事件表没有 actor 列）迁移之后：**旧行写"未知"，绝不猜名字**。

        `experiment_events` 是 append-only，触发器禁止 UPDATE ——
        所以迁移只能是 `ADD COLUMN` + 默认值。想"回填"就得先关掉触发器，
        那等于毁掉审计的卖点。这里手工造一个老结构的库来验证。
        """
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE experiment_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL, at TEXT NOT NULL, action TEXT NOT NULL,
                field TEXT, before TEXT, after TEXT, note TEXT NOT NULL DEFAULT ''
            );
            CREATE TRIGGER experiment_events_no_update BEFORE UPDATE ON experiment_events
            BEGIN SELECT RAISE(ABORT, 'append-only'); END;
            """
        )
        conn.execute(
            "INSERT INTO experiment_events (experiment_id, at, action, note) "
            "VALUES ('old_exp', '2026-01-01T00:00:00+00:00', 'create', '迁移前的老记录')"
        )
        conn.commit()
        conn.close()

        reg = ExperimentRegistry(path)
        try:
            events = reg.events("old_exp")
            assert len(events) == 1
            assert events[0].actor == "（迁移前未知）"
            # 新写入的行则必须带真实操作者
            rec = make(reg, "after_migration")
            assert reg.events(rec.id)[0].actor == ACTOR
        finally:
            reg.close()


class TestOptimisticLocking:
    """并发：**两个人同时改，不能有一方的改动被静默吃掉**。

    这一组测试要钉的不是"版本号加一"，而是"丢失更新"这个具体故障：
    两个客户端都读到 v1，A 先写、B 后写 —— 没有检查时 B 的写会成功，
    而 A 的改动**从世界上消失**（谁都没报错，审计里也只有 B 那条）。
    这就是"后写覆盖"，也是为什么它比崩溃更难查。
    """

    @staticmethod
    def _app(tmp_path: Path):
        from fastapi.testclient import TestClient

        client = TestClient(create_app(tmp_path / "optlock.db"))
        token = client.app.state.registry.add_user("lock_admin", role="admin")
        return client, {"Authorization": f"Bearer {token}"}

    def _new(self, client, auth, name="lock_demo"):
        return client.post(
            "/api/experiments",
            json={"name": name, "variants": VARIANTS, "salt": f"{name}_v1"},
            headers=auth,
        ).json()

    def test_lost_update_is_prevented_by_if_match(self, tmp_path):
        """核心断言：A 写成功之后，B 拿着**读过的旧版本**再写必须被拒。"""
        client, auth = self._app(tmp_path)
        rec = self._new(client, auth)
        stale = rec["version"]  # 两个客户端都读到 v1

        first = client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "running"},
            headers={**auth, "If-Match": str(stale)},
        )
        assert first.status_code == 200
        assert first.json()["version"] == stale + 1

        second = client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "stopped"},
            headers={**auth, "If-Match": str(stale)},
        )
        assert second.status_code == 412
        assert "版本冲突" in second.json()["detail"]
        # A 的改动还在（没有被 B 覆盖），而且 B 的写**没有**进审计
        current = client.get(f"/api/experiments/{rec['id']}").json()
        assert current["status"] == "running"
        assert current["version"] == stale + 1
        actions = [
            e["action"]
            for e in client.get(f"/api/experiments/{rec['id']}/events").json()["events"]
        ]
        assert actions == ["create", "set_status"]

    def test_without_if_match_it_is_last_write_wins(self, tmp_path):
        """不带 `If-Match` 时按后写覆盖放行 —— 这是**默认行为**，不是漏洞。

        把它钉住，是为了让"乐观锁是可选的"这件事有据可查：
        要防覆盖就带版本号；不带就表示调用方接受覆盖。
        """
        client, auth = self._app(tmp_path)
        rec = self._new(client, auth, "lock_lww")
        stale = rec["version"]
        assert (
            client.patch(
                f"/api/experiments/{rec['id']}/status",
                json={"status": "running"},
                headers={**auth, "If-Match": str(stale)},
            ).status_code
            == 200
        )
        overwrite = client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "stopped"},
            headers=auth,  # 故意不带
        )
        assert overwrite.status_code == 200
        assert overwrite.json()["status"] == "stopped"

    def test_stale_delete_does_not_remove_the_experiment(self, tmp_path):
        """删除也要检查版本：否则"我正打算改，别人把它删了"会变成静默失败。"""
        client, auth = self._app(tmp_path)
        rec = self._new(client, auth, "lock_del")
        client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "running"},
            headers={**auth, "If-Match": str(rec["version"])},
        )
        stale = client.delete(
            f"/api/experiments/{rec['id']}", headers={**auth, "If-Match": str(rec["version"])}
        )
        assert stale.status_code == 412
        assert client.get(f"/api/experiments/{rec['id']}").status_code == 200

    def test_malformed_if_match_is_400_and_quoted_form_is_accepted(self, tmp_path):
        client, auth = self._app(tmp_path)
        rec = self._new(client, auth, "lock_fmt")
        bad = client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "running"},
            headers={**auth, "If-Match": "abc"},
        )
        assert bad.status_code == 400
        # HTTP 规范里 ETag 是带引号的，所以引号写法必须能用
        quoted = client.patch(
            f"/api/experiments/{rec['id']}/status",
            json={"status": "running"},
            headers={**auth, "If-Match": f'"{rec["version"]}"'},
        )
        assert quoted.status_code == 200
        assert quoted.json()["version"] == rec["version"] + 1

    def test_every_write_bumps_the_version_and_the_audit_says_so(self, tmp_path):
        """每次成功的写 +1，且审计里留下版本变迁 —— 复现时能对齐到具体某一版。"""
        client, auth = self._app(tmp_path)
        rec = self._new(client, auth, "lock_bump")
        eid = rec["id"]
        client.patch(
            f"/api/experiments/{eid}/status", json={"status": "running"}, headers=auth
        )
        client.post(
            f"/api/experiments/{eid}/estimator",
            json={"estimator": "post_only"},
            headers=auth,
        )
        client.post(
            f"/api/experiments/{eid}/bind",
            json={"warehouse_experiment": None},
            headers=auth,
        )
        assert client.get(f"/api/experiments/{eid}").json()["version"] == 4
        notes = [
            e["note"]
            for e in client.get(f"/api/experiments/{eid}/events").json()["events"]
        ]
        assert any("v1 -> v2" in n for n in notes), notes
        assert sum("-> v" in n for n in notes) >= 3

    def test_conflict_type_is_distinct_from_bad_input(self, tmp_path):
        """冲突与"参数写错"必须是两种类型：前者重读后重试通常会成功。"""
        reg = ExperimentRegistry(tmp_path / "types.db")
        try:
            rec = make(reg, "lock_types")
            reg.set_status(rec.id, "running", actor=ACTOR, expected_version=rec.version)
            with pytest.raises(RegistryConflict):
                reg.set_status(
                    rec.id, "stopped", actor=ACTOR, expected_version=rec.version
                )
            # 冲突是 RegistryError 的子类（调用方可以只捕获父类），但类型可区分
            with pytest.raises(RegistryError):
                reg.set_status(
                    rec.id, "stopped", actor=ACTOR, expected_version=rec.version
                )
            with pytest.raises(RegistryError):
                reg.set_status(rec.id, "not_a_status", actor=ACTOR)
        finally:
            reg.close()

    def test_old_database_gets_version_column_at_one(self, tmp_path):
        """老库迁移：既有行的版本从 **1** 开始 —— 不编造历史。

        我们并不知道那些行被改过几次；写 1 的意思是"从这里开始计数"。
        编一个更大的数会假装我们知道历史，那正是审计要避免的事。
        """
        path = tmp_path / "old_version.db"
        reg = ExperimentRegistry(path)
        rec = make(reg, "old_version_row")
        reg._conn.execute("UPDATE experiments SET version = 99 WHERE id = ?", (rec.id,))
        reg._conn.commit()
        reg.close()

        # 模拟"加列之前"的库：删掉 version 列在 sqlite 里做不到，
        # 所以直接建一个不含该列的表结构，再走一次迁移。
        raw = sqlite3.connect(tmp_path / "old_shape.db")
        raw.executescript(
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
            );
            INSERT INTO experiments (id, name, salt, variants, created_at)
            VALUES ('old1', 'legacy', 'legacy_v1', '[]', '2026-01-01T00:00:00+00:00');
            """
        )
        raw.commit()
        raw.close()

        migrated = ExperimentRegistry(tmp_path / "old_shape.db")
        try:
            row = migrated.get("old1")
            assert row.version == 1
            migrated.set_status("old1", "running", actor=ACTOR)
            assert migrated.get("old1").version == 2
        finally:
            migrated.close()


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
