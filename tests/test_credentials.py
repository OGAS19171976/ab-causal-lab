"""凭据的生命周期与限速：过期、带宽限期的轮换、读接口要凭据、429。

为什么值得单独一个文件
----------------------
这一轮之前，README 把三件事写在"边界"里：静态 token **没有过期、没有轮换、
没有限速**，而且**读接口匿名**。这四条当时都是真的 —— 但它们是**文字**，
而文字会被时间漂移：功能补上了、话还在，读的人就得到相反的结论
（这个仓库已经栽过三次，见设计决策 57）。

所以这里每条边界都配一个能红的断言，外加一句"为什么这样选"。
时间**不靠 sleep**：注册表那三个凭据方法都收一个可选的 ``at=``，
于是"过期"是摆出来的，不是等出来的。
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ablab.platform.api import create_app
from ablab.platform.registry import (
    DEFAULT_TOKEN_TTL_DAYS,
    ExperimentRegistry,
    RegistryError,
)

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def reg(work_dir: Path):
    r = ExperimentRegistry(work_dir / "cred.db")
    yield r
    r.close()


class TestTokenLifecycle:
    """注册表层：过期与轮换。全是确定性的 —— 时间用 ``at=`` 给。"""

    def test_fresh_token_gets_the_default_ttl(self, reg):
        raw = reg.add_user("a", role="viewer", at=NOW)
        status = reg.auth_status(raw, at=NOW)
        assert status.ok and status.reason == "ok"
        assert status.expires_at == (NOW + timedelta(days=DEFAULT_TOKEN_TTL_DAYS)).isoformat(
            timespec="seconds"
        )
        # 到期的**那一刻**就算过期（<=，不是 <）：边界不留含糊
        assert not reg.auth_status(raw, at=NOW + timedelta(days=DEFAULT_TOKEN_TTL_DAYS)).ok

    def test_zero_ttl_never_expires_and_says_so(self, reg):
        """``ttl_days=0`` = 永不过期。它是**显式选择**，不是默认值。"""
        raw = reg.add_user("a", role="viewer", ttl_days=0, at=NOW)
        assert reg.auth_status(raw, at=NOW).expires_at is None
        assert reg.auth_status(raw, at=NOW + timedelta(days=3650)).ok
        assert reg.list_users()[0]["expires_at"] is None

    def test_expired_token_is_reported_as_expired_not_unknown(self, reg):
        """过期说"过期"。

        说成"不认识"会让运维去查一个不存在的问题（token 拼错了？），
        而正确的动作是"换发" —— 401 的文案要能指导动作。
        """
        raw = reg.add_user("a", role="viewer", ttl_days=1, at=NOW - timedelta(days=2))
        status = reg.auth_status(raw, at=NOW)
        assert not status.ok
        assert status.reason == "expired"
        assert status.expires_at is not None
        assert reg.authenticate(raw, at=NOW) is None

    def test_rotation_with_grace_keeps_the_old_token_for_the_window(self, reg):
        """宽限期：换发是**滚动动作**，不是停机动作。代价是两个 token 同时有效。"""
        old = reg.add_user("a", role="editor", at=NOW)
        new = reg.rotate_token("a", grace_minutes=30, at=NOW)
        assert new != old
        assert reg.auth_status(old, at=NOW).ok, "换发后旧 token 立刻失效了？"
        # 窗口内有效
        assert reg.auth_status(old, at=NOW + timedelta(minutes=29)).ok
        # 窗口外立刻失效（边界同样取 <=）
        assert reg.auth_status(old, at=NOW + timedelta(minutes=30)).reason == "expired"
        assert reg.auth_status(old, at=NOW + timedelta(hours=1)).reason == "expired"
        # 新 token 不受旧窗口影响
        assert reg.auth_status(new, at=NOW + timedelta(hours=1)).ok

    def test_rotation_without_grace_kills_the_old_token_at_once(self, reg):
        old = reg.add_user("a", role="editor", at=NOW)
        reg.rotate_token("a", grace_minutes=0, at=NOW)
        assert reg.auth_status(old, at=NOW).reason == "expired"

    def test_rotation_renews_the_expiry(self, reg):
        """换发同时续期 —— 否则第 89 天轮换完，第 90 天又过期。"""
        reg.add_user("a", role="editor", ttl_days=90, at=NOW)
        reg.rotate_token("a", ttl_days=90, at=NOW + timedelta(days=89))
        # 旧 token 在这个时刻已经过期，新 token 还有 90 天
        assert reg.list_users()[0]["expires_at"] == (
            NOW + timedelta(days=179)
        ).isoformat(timespec="seconds")

    def test_negative_grace_is_refused(self, reg):
        reg.add_user("a", role="editor", at=NOW)
        with pytest.raises(RegistryError):
            reg.rotate_token("a", grace_minutes=-1, at=NOW)

    def test_rotation_of_unknown_user_raises(self, reg):
        with pytest.raises(RegistryError):
            reg.rotate_token("nobody", at=NOW)

    def test_disabled_beats_everything(self, reg):
        """停用是**立即**止损手段：它不看时钟，也不受宽限期影响。"""
        raw = reg.add_user("a", role="editor", at=NOW)
        reg.rotate_token("a", grace_minutes=30, at=NOW)  # 旧 token 落在宽限期里
        reg.disable_user("a")
        assert reg.auth_status(raw, at=NOW).reason == "disabled"

    def test_list_users_never_leaks_hashes_but_shows_expiry(self, reg):
        reg.add_user("a", role="viewer", ttl_days=7, at=NOW)
        reg.add_user("b", role="viewer", ttl_days=0, at=NOW)
        users = {u["id"]: u for u in reg.list_users()}
        assert users["a"]["expires_at"] is not None
        assert users["b"]["expires_at"] is None
        for u in users.values():
            assert "token" not in u and "token_hash" not in u

    def test_old_database_keeps_its_tokens_working(self, work_dir: Path):
        """**老库迁移后旧的 token 必须还能用**，而且它们是"永不过期"。

        加一列就把所有人踢下线，等于用一次迁移换一次停机；而给老 token 编一个
        到达日期又是**编造**（我们并不知道它们是什么时候签发的）。
        所以迁移的结果只能是 NULL = 永不过期 —— 让管理员显式轮换一次来收紧。
        """
        path = work_dir / "legacy_users.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL CHECK (role IN ('viewer','editor','admin')),
                created_at TEXT NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0,
                note TEXT NOT NULL DEFAULT ''
            );
            """
        )
        legacy = "legacy-token-in-the-clear"
        conn.execute(
            "INSERT INTO users (id, token_hash, role, created_at, disabled, note) "
            "VALUES ('old', ?, 'editor', '2026-01-01T00:00:00+00:00', 0, '迁移前')",
            (ExperimentRegistry.hash_token(legacy),),
        )
        conn.commit()
        conn.close()

        migrated = ExperimentRegistry(path)
        try:
            assert migrated.auth_status(legacy, at=NOW).ok
            assert migrated.list_users()[0]["expires_at"] is None
            # 迁移之后新签发的仍然有有效期（默认值没有因为兼容而被关掉）
            fresh = migrated.add_user("new", role="editor", at=NOW)
            assert migrated.auth_status(fresh, at=NOW).expires_at is not None
        finally:
            migrated.close()


class TestReadAuthAPI:
    """接口层：读也要凭据，401 的文案按原因分。"""

    @staticmethod
    def _fill(path: str) -> str:
        return re.sub(r"\{[^}]+\}", "x", path)

    def test_every_api_route_needs_a_token(self, work_dir: Path):
        """**自动枚举所有 /api 路由**（读和写都算），逐个断言无凭据是 401。

        人会在新增端点时忘记"这个也要鉴权"，枚举测试不会 —— 它与
        ``TestAuthAndActor::test_every_mutating_route_needs_a_token`` 是一对：
        那条管写，这条管读。两条都靠枚举，而不是靠一份手抄的名单。
        """
        client = TestClient(create_app(work_dir / "readauth.db"))
        checked = 0
        for route in client.app.routes:
            methods = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            if not path.startswith("/api"):
                continue
            for method in sorted(methods - {"HEAD", "OPTIONS"}):
                resp = client.request(method, self._fill(path), json={})
                assert resp.status_code == 401, (method, path, resp.status_code)
                checked += 1
        assert checked >= 10, f"只枚举到 {checked} 个 /api 路由，枚举逻辑可能失效了"

    def test_healthz_stays_public(self, work_dir: Path):
        """健康检查必须公开：否则监控先于业务死掉（它拿不到 token）。"""
        client = TestClient(create_app(work_dir / "health.db"))
        assert client.get("/healthz").status_code == 200

    def test_401_detail_depends_on_the_reason(self, work_dir: Path):
        client = TestClient(create_app(work_dir / "reasons.db"))
        reg = client.app.state.registry
        expired = reg.add_user("expired", role="viewer", ttl_days=1, at=NOW - timedelta(days=3))
        fresh = reg.add_user("fresh", role="viewer")

        missing = client.get("/api/experiments")
        unknown = client.get("/api/experiments", headers=bearer("nope"))
        gone = client.get("/api/experiments", headers=bearer(expired))
        ok = client.get("/api/experiments", headers=bearer(fresh))
        assert [missing.status_code, unknown.status_code, gone.status_code] == [401] * 3
        assert "需要凭据" in missing.json()["detail"]
        assert "不在库里" in unknown.json()["detail"]
        # 过期的那条要指向**换发**这个动作，而不是让人去查"token 是不是拼错了"
        assert "过期" in gone.json()["detail"] and "rotate" in gone.json()["detail"]
        assert ok.status_code == 200


class TestRateLimitAPI:
    """接口层：限速。额度在测试里调到个位数，于是几条请求就能证明 429。"""

    def test_429_carries_retry_after_and_the_limit(self, work_dir: Path):
        app = create_app(work_dir / "rl.db", rate_limits={"read": 2})
        token = app.state.registry.add_user("a", role="viewer")
        client = TestClient(app)
        client.headers.update(bearer(token))

        assert [client.get("/api/experiments").status_code for _ in range(2)] == [200, 200]
        third = client.get("/api/experiments")
        assert third.status_code == 429
        # 没有 Retry-After，客户端只能靠猜什么时候再来
        assert int(third.headers["Retry-After"]) >= 1
        assert third.headers["X-RateLimit-Limit"] == "2"

    def test_the_budget_is_per_identity(self, work_dir: Path):
        """限速按**身份**分桶：一个人跑飞了不该把同事一起挡住。"""
        app = create_app(work_dir / "rl2.db", rate_limits={"read": 1})
        a = app.state.registry.add_user("a", role="viewer")
        b = app.state.registry.add_user("b", role="viewer")
        client = TestClient(app)
        assert client.get("/api/experiments", headers=bearer(a)).status_code == 200
        assert client.get("/api/experiments", headers=bearer(a)).status_code == 429
        assert client.get("/api/experiments", headers=bearer(b)).status_code == 200

    def test_writes_have_their_own_smaller_budget(self, work_dir: Path):
        app = create_app(work_dir / "rl3.db", rate_limits={"read": 5, "write": 1})
        token = app.state.registry.add_user("a", role="editor")
        client = TestClient(app)
        client.headers.update(bearer(token))
        body = {
            "name": "x",
            "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            "salt": "x_v1",
        }
        assert client.post("/api/experiments", json=body).status_code == 201
        assert client.post("/api/experiments", json={**body, "name": "y"}).status_code == 429
        # 写额度用完不该影响读（两个窗口）
        assert client.get("/api/experiments").status_code == 200

    def test_guessing_tokens_is_limited_too(self, work_dir: Path):
        """**被拒的请求也要计入限速**（按来源 IP）。

        否则"猜 token"这条路径一次都不花成本 —— 而认证发生在限速之前的话，
        正好就是这个结果：攻击者可以无限试。
        """
        app = create_app(work_dir / "rl4.db", rate_limits={"anonymous": 2})
        client = TestClient(app)
        codes = [
            client.get("/api/experiments", headers=bearer(f"guess-{i}")).status_code
            for i in range(3)
        ]
        assert codes[:2] == [401, 401]
        assert codes[2] == 429, "连续猜 token 没有被限速"
