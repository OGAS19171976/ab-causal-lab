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


class TestGuardrailCalibration:
    """护栏判定的**运行特征**：误停率与功效。

    规则问的是"伤害是否超过容忍度"，所以它不是 5% 水平的检验 ——
    H0（真实伤害 0）下几乎不会喊停，而这是**有意**的取舍。
    这一组把"保守到什么程度、真有害时能不能喊出来"钉住：
    只写规则合理、不给运行特征，等于让别人替我们相信。
    """

    def test_h0_almost_never_stops_and_h1_almost_always_stops(self):
        """H0 误停 0 / 边界点开始出现停止 / H1 全部停止。

        实测（80 次/场景、n=3000、容忍度 5%）：H0 停 0 次；
        边界点（伤害正好 5%）停 4 次、观察 36 次；注入 12% 时停 80 次。
        这里用小样本跑（6 次）以控制测试时长，断言方向而不是具体比例。
        """
        from ablab.validation.guardrail_audit import run_guardrail_audit

        cal = run_guardrail_audit(n_trials=6, n_users=1500, max_harm=0.05, harm=0.30)
        assert cal.false_stop_rate == 0.0, cal.statuses_h0
        assert cal.power >= 0.5, cal.statuses_h1
        assert cal.statuses_h0.get("unknown", 0) == 0
        # 判定分布必须覆盖到"通过"这一档：否则是规则没跑起来
        assert cal.statuses_h0.get("ok", 0) == 6

    def test_boundary_point_is_the_real_operating_point(self):
        """边界点（伤害正好等于容忍度）落在"观察/通过"之间，而不是全停。

        这一条是这条规则的性格：**点估计要越界、置信下界也要越界**才停，
        所以压线时大量进入"观察"带。把它钉住，免得有人把阈值改到 0
        还以为只是"更灵敏"——那会让 H0 误停率跳到 alpha 附近。
        """
        from ablab.validation.guardrail_audit import run_guardrail_audit

        cal = run_guardrail_audit(n_trials=6, n_users=1500, max_harm=0.05, harm=0.05)
        assert cal.boundary_statuses.get("unknown", 0) == 0
        assert cal.boundary_statuses.get("ok", 0) + cal.boundary_statuses.get("watch", 0) > 0

    def test_declared_tolerance_drives_the_verdict(self):
        """同一条数据、同一个伤害：容忍度收紧到 1% 就该停，放宽到 50% 就通过。

        这是"阈值必须事先声明"的可测含义：判定随**声明**变，
        而不是随数据变。
        """
        from ablab.validation.guardrail_audit import run_guardrail_audit

        tight = run_guardrail_audit(n_trials=3, n_users=1500, max_harm=0.01, harm=0.12)
        loose = run_guardrail_audit(n_trials=3, n_users=1500, max_harm=0.50, harm=0.12)
        assert tight.power == 1.0, tight.statuses_h1
        assert loose.power == 0.0, loose.statuses_h1


class TestGuardrailAnalysis:
    """护栏：**真的判定**，并且把"无法判断"与"通过"严格分开。

    这一块原来的毛病不是缺功能，而是"字段存了、界面显示了、引擎没读过"——
    用户合理地以为护栏被看着。所以测试的重点不在"能不能算出一个数"，
    而在三件事：**越界要能触发停实验**、**缺声明/缺数据不能被当成通过**、
    **判定用的是置信下界而不是点估计**。
    """

    @staticmethod
    def _series(name: str, values_t, values_c):
        """按**规格里的名字**建键 —— 名字对不上会被判成"没有数据"（这很常见）。"""
        import numpy as np

        from ablab.inference.aggregates import AggregateStats

        return {
            name: {
                "control": AggregateStats.from_arrays(np.asarray(values_c, dtype=float)),
                "treatment": AggregateStats.from_arrays(np.asarray(values_t, dtype=float)),
            }
        }

    def _run(self, spec, treated_values, control_values):
        from ablab.platform.guardrails import analyse_guardrails

        return analyse_guardrails(
            [spec],
            self._series(spec.name, treated_values, control_values),
            treated="treatment",
            control="control",
        )

    def test_harm_beyond_tolerance_triggers_stop(self):
        """伤害的**置信下界**越过容忍度 -> fail + 建议停实验 + check 状态 fail。"""
        import numpy as np

        from ablab.platform.guardrails import GuardrailSpec

        rng = np.random.default_rng(0)
        spec = GuardrailSpec("latency_p99", "lower_is_better", 0.05)
        report = self._run(
            spec,
            rng.normal(1.20, 0.05, 4000),  # +20% 延迟
            rng.normal(1.00, 0.05, 4000),
        )
        assert report.verdict == "stop"
        assert report.check_status == "fail"
        assert "停止实验" in report.recommendation
        outcome = report.outcomes[0]
        assert outcome.status == "fail"
        assert outcome.harm > 0.15
        assert outcome.harm_ci_low > spec.max_harm
        assert outcome.p_value < 0.01

    def test_no_harm_passes_and_direction_matters(self):
        """方向决定"伤害"的正负号 —— 越高越好的指标变低也是伤害。"""
        import numpy as np

        from ablab.platform.guardrails import GuardrailSpec

        rng = np.random.default_rng(1)
        # 收入类指标（越高越好）：处置组变低 20% 就是伤害
        report = self._run(
            GuardrailSpec("revenue_per_user", "higher_is_better", 0.05),
            rng.normal(0.80, 0.02, 4000),
            rng.normal(1.00, 0.02, 4000),
        )
        assert report.verdict == "stop"
        assert report.outcomes[0].harm > 0.15  # 伤害是正的（变低被折算成正伤害）
        # 同一个数据，方向写反 -> 伤害变成负数 -> 通过（所以方向绝不能猜）
        reversed_report = self._run(
            GuardrailSpec("revenue_per_user", "lower_is_better", 0.05),
            rng.normal(0.80, 0.02, 4000),
            rng.normal(1.00, 0.02, 4000),
        )
        assert reversed_report.verdict == "ok"

    def test_point_estimate_over_limit_is_only_a_warning(self):
        """点估计超了、但置信下界没超 -> ``warn``（继续观察），不停实验。

        这条是把"停机"这个动作做保守的关键：否则样本少的时候会频繁误停。
        """
        import numpy as np

        from ablab.platform.guardrails import GuardrailSpec

        rng = np.random.default_rng(2)
        report = self._run(
            GuardrailSpec("latency_p99", "lower_is_better", 0.05),
            rng.normal(1.06, 0.20, 60),  # 点估计 +6%，但噪声大、样本小
            rng.normal(1.00, 0.20, 60),
        )
        assert report.outcomes[0].status == "warn", report.summary()
        assert report.verdict == "watch"
        assert report.check_status == "warn"

    def test_missing_spec_is_warn_missing_data_is_info(self):
        """两种"无法判断"要分开，因为补救办法不同 —— 而且都**不是通过**。"""
        from ablab.platform.guardrails import GuardrailSpec, analyse_guardrails

        bare = analyse_guardrails(
            [GuardrailSpec("latency_p99")], {}, treated="treatment", control="control"
        )
        assert bare.verdict == "unknown"
        assert bare.missing_specs and not bare.missing_data
        assert bare.check_status == "warn"  # 这一次就能补：direction + max_harm
        assert "补上声明" in bare.recommendation

        no_data = analyse_guardrails(
            [GuardrailSpec("latency_p99", "lower_is_better", 0.05)],
            {},
            treated="treatment",
            control="control",
        )
        assert no_data.verdict == "unknown"
        assert no_data.missing_data and not no_data.missing_specs
        assert no_data.check_status == "info"  # 平台级缺口，不该永远 warn
        assert "不等于通过" in no_data.recommendation

    def test_bonferroni_adjustment_is_applied_across_guardrails(self):
        """K 条护栏就是 K 次检验：校正后的 alpha 随 K 变小。"""
        import numpy as np

        from ablab.platform.guardrails import GuardrailSpec, analyse_guardrails

        rng = np.random.default_rng(3)
        specs = [
            GuardrailSpec(f"g{i}", "lower_is_better", 0.05) for i in range(4)
        ]
        series = {
            s.name: {
                "control": __import__(
                    "ablab.inference.aggregates", fromlist=["AggregateStats"]
                ).AggregateStats.from_arrays(rng.normal(1.0, 0.05, 2000)),
                "treatment": __import__(
                    "ablab.inference.aggregates", fromlist=["AggregateStats"]
                ).AggregateStats.from_arrays(rng.normal(1.0, 0.05, 2000)),
            }
            for s in specs
        }
        report = analyse_guardrails(
            specs, series, treated="treatment", control="control", alpha=0.05
        )
        assert all(abs(o.alpha_adjusted - 0.05 / 4) < 1e-12 for o in report.outcomes)

    def test_analysis_marks_fail_and_stop_in_the_report(self):
        """走完真实入口：带规格的实验 -> 护栏 fail -> health=fail。"""
        from ablab.platform.analysis import analyse_experiment
        from ablab.platform.guardrails import GuardrailSpec

        rec = ExperimentRecord(
            name="guardrail_e2e",
            variants=list(VARIANTS),
            salt="guardrail_e2e_v1",
            primary_metric="post_metric_14d",
            guardrails=["latency_p99"],
            guardrail_specs=[
                GuardrailSpec("latency_p99", "lower_is_better", 0.05, demo_harm=0.12)
            ],
        )
        report = analyse_experiment(rec, n_users=4000)
        item = next(c for c in report.checks if c.name == "护栏指标")
        assert item.status == "fail"
        assert report.health == "fail"
        assert "停止实验" in item.message

    def test_warehouse_path_says_unknown_not_pass(self):
        """数仓路径还没有护栏表 -> 判 unknown 并说清原因，**不是通过**。

        这条钉的是这一整块最容易犯的错：缺数据时给人一个"检查通过"。
        """
        from ablab.platform.analysis import _with_record_metadata
        from ablab.platform.datasource import build_synthetic_data
        from ablab.platform.guardrails import GuardrailSpec

        rec = ExperimentRecord(
            name="guardrail_wh",
            variants=list(VARIANTS),
            salt="guardrail_wh_v1",
            primary_metric="post_metric_14d",
            guardrails=["latency_p99"],
            guardrail_specs=[GuardrailSpec("latency_p99", "lower_is_better", 0.05)],
            warehouse_experiment="exp_rank_v2",
        )
        data = build_synthetic_data(
            experiment=rec.name, salt=rec.salt, variants=[("control", 0.5), ("treatment", 0.5)],
            metric=rec.primary_metric, n_users=2000, n_looks=3, seed=7,
            population=__import__("ablab.platform.analysis", fromlist=["PLATFORM_POPULATION"]).PLATFORM_POPULATION,
        )
        # 数仓路径的护栏数据是空的（还没有那张表）
        wh_data = _with_record_metadata(data, rec)
        wh_data = __import__("dataclasses").replace(wh_data, guardrail_series={})
        from ablab.platform.analysis import analyse_data

        report = analyse_data(wh_data)
        item = next(c for c in report.checks if c.name == "护栏指标")
        assert item.status == "info"
        assert "不等于通过" in item.message
        assert "停止实验" not in item.message


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
    """护栏要么被**判定**，要么**说清为什么判不了** —— 不能含糊其辞。

    这一组测试的名字与断言在本轮被改写过：它们原来钉的是"声明了护栏却没人分析，
    报告里必须明说'尚不分析'"。现在引擎真的会判定了，于是同等重要的关切变成：
    **没有被判定时，报告必须写清原因，而且绝不能写成通过**。
    换句话说，钉的不是"某个字符串还在"，而是"这句话还成立"。
    """

    @staticmethod
    def record(name: str, guardrails: list[str]) -> ExperimentRecord:
        return ExperimentRecord(
            name=name,
            variants=list(VARIANTS),
            salt=f"{name}_v1",
            primary_metric="post_metric_14d",
            guardrails=guardrails,
        )

    def test_declared_names_without_specs_say_why_they_cannot_be_judged(self):
        """只声明名字、没给方向与容忍度 -> warn，并说清"补上声明才能判定"。

        它**不是通过**：缺声明时判通过会让人以为护栏被看着 ——
        那正是这一块原来的毛病。
        """
        rep = analyse_experiment(
            self.record("g_declared", ["latency_p99", "crash_rate"]), n_users=4_000
        )
        item = next((c for c in rep.checks if c.name == "护栏指标"), None)
        assert item is not None, [c.name for c in rep.checks]
        assert item.statistic == 2.0
        assert "latency_p99" in item.message and "crash_rate" in item.message
        assert item.status == "warn", item.status
        assert "不等于通过" in item.message
        assert "direction + max_harm" in item.message

    def test_platform_level_gap_does_not_raise_health(self):
        """**有规格但没数据**（数仓还没有护栏表）-> info，不把 health 拉成 warn。

        这是平台级缺口：它对每个实验都一样，永远 warn 就等于没有告警
        （第 31 条那条教训），health 会因此失去意义。
        注意与上一条的区别：**用户要补的声明 -> warn**（这一次就能补），
        **平台要补的数据 -> info**（不该由用户承担）。
        """
        import dataclasses

        from ablab.platform.analysis import PLATFORM_POPULATION, analyse_data
        from ablab.platform.datasource import build_synthetic_data
        from ablab.platform.guardrails import GuardrailSpec

        rec = self.record("g_health", ["latency_p99"])
        rec.guardrail_specs = [GuardrailSpec("latency_p99", "lower_is_better", 0.05)]
        # 合成路径**会**生成护栏数据；这里显式清空，模拟"数仓路径还没有护栏表"
        data = build_synthetic_data(
            experiment=rec.name, salt=rec.salt,
            variants=[("control", 0.5), ("treatment", 0.5)],
            metric=rec.primary_metric, n_users=2000, n_looks=3, seed=7,
            population=PLATFORM_POPULATION, guardrail_specs=tuple(rec.guardrail_specs),
        )
        assert data.guardrail_series, "合成路径本应生成护栏数据，否则这条测试没测到东西"
        empty = dataclasses.replace(data, guardrail_series={})
        item = next(c for c in analyse_data(empty).checks if c.name == "护栏指标")
        assert item.status == "info", item.status
        assert "不等于通过" in item.message
        assert "数仓" in item.message

    def test_no_guardrails_no_notice(self):
        rep = analyse_experiment(self.record("g_none", []), n_users=4_000)
        assert not [c for c in rep.checks if c.name == "护栏指标"]

    def test_notice_reaches_the_api_payload(self):
        """报告字典是 API 返回的东西 —— 那条声明要能在它里面看到。"""
        rep = analyse_experiment(self.record("g_api", ["scroll_depth"]), n_users=4_000)
        names = [c["name"] for c in rep.to_dict()["checks"]]
        assert "护栏指标" in names
