"""实验注册表：把"一个实验"从散落的配置变成有状态、可校验、可追溯的记录。

三个设计决定
------------
**1. 用 sqlite3 而不是 DuckDB。**
数仓那层分析明细数据，用 DuckDB；这一层是低频的**配置类**数据（几十到几千条），
sqlite3 是标准库、零依赖、单文件、天然适合。

**2. 创建时必须过 ``ExperimentSpec`` 的校验。**
权重之和、流量比例、分支重名、处置期…… 这些规则在 M0 就写好了。
注册表不重复实现一遍 —— 它直接构造一个 ``ExperimentSpec``，
非法配置在写入之前就被拒掉。**校验逻辑只有一份。**

**3. 记录里保留 ``salt`` 且不可变。**
salt 决定了每一个用户的分组。改 salt 等于把所有用户重新分组，
实验数据直接报废 —— 所以它在 API 层是只读的（``update`` 不接受它）。

**4. ``warehouse_experiment`` 是可变的，这跟第 3 条不矛盾。**
它只决定"从哪里读数"，不改变任何用户的分组，所以允许后置绑定 ——
而且必须允许：数仓表要等实验跑完才有，创建实验时它还不存在。
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from ..assignment import ExperimentSpec, Variant

__all__ = [
    "DEFAULT_TOKEN_TTL_DAYS",
    "AuthStatus",
    "ExperimentEvent",
    "ExperimentRecord",
    "ExperimentRegistry",
    "RegistryError",
    "STATUSES",
]

STATUSES = ("draft", "running", "stopped")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    hypothesis    TEXT NOT NULL DEFAULT '',
    owner         TEXT NOT NULL DEFAULT '',
    layer         TEXT,
    unit          TEXT NOT NULL DEFAULT 'user_id',
    salt          TEXT NOT NULL,
    traffic_ratio REAL NOT NULL DEFAULT 1.0,
    variants      TEXT NOT NULL,
    primary_metric TEXT NOT NULL DEFAULT 'metric',
    guardrails    TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'draft',
    start_ds      TEXT,
    end_ds        TEXT,
    true_lift     REAL NOT NULL DEFAULT 0.0,
    estimator     TEXT NOT NULL DEFAULT 'cuped',
    analysis_unit TEXT NOT NULL DEFAULT 'unit',
    metric_type   TEXT NOT NULL DEFAULT 'mean',
    created_at    TEXT NOT NULL
);

-- 操作审计：**append-only**。
--
-- 为什么需要它：M6 特意允许改 ``estimator``（口径是策略不是数据），
-- 也允许改状态、绑数仓、删实验 —— 于是"谁在什么时候把判定口径从 CUPED
-- 改成 post-only"这件事必须留痕。否则改口径就是一个**事后挑口径的通道**：
-- 报告里始终写着"用的是哪个口径"，但没人知道它是不是在看到结果之后才改的。
--
-- 三条不可动摇的设计：
--   1. **写在业务变更的同一个事务里**。分开写就会出现"改了但没记"，
--      而那种缺失是静默的 —— 审计表看起来"没有这条记录"，与"没发生过"无法区分。
--   2. **触发器禁止 UPDATE / DELETE**。不是"我们不写 UPDATE"，是**写不动**。
--      靠约定的不可变性，迟早被某次维护脚本破坏。
--   3. **没有外键级联**。删掉实验之后审计必须还在 —— 那恰恰是最需要它的时候。
CREATE TABLE IF NOT EXISTS experiment_events (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    at            TEXT NOT NULL,
    action        TEXT NOT NULL,
    field         TEXT,
    before        TEXT,
    after         TEXT,
    note          TEXT NOT NULL DEFAULT '',
    -- 操作者。**由服务端从凭据推导，永远不从请求体里读** ——
    -- 客户端能填的名字不是身份，只是声明（见 README 设计决策第 45 条）。
    -- 老库迁移时这一列取默认值：宁可写"未知"，也不拿一个猜出来的名字
    -- 冒充历史记录（`experiment_events` 是 append-only，回填就是改写历史）。
    actor         TEXT NOT NULL DEFAULT '（迁移前未知）'
);

CREATE INDEX IF NOT EXISTS idx_events_experiment
    ON experiment_events(experiment_id, seq);

-- 用户与凭据。**存 token 的 sha256，不存 token 本身**：
-- 库文件泄露不应该等于凭据泄露。token 是高熵随机串（`secrets.token_urlsafe`），
-- 所以直接哈希就够，不需要抗暴力破解的口令哈希 —— 没有"猜口令"这条捷径。
--
-- 凭据的**生命周期**（这一轮补上，之前三件都没有）：
--   * expires_at：到期即失效。NULL = 永不过期 —— 老库迁移后就是 NULL，
--     这是刻意的：加一列就把所有人踢下线，等于用一次迁移换一次停机；
--   * prev_token_hash / prev_valid_until：轮换时旧 token 的**宽限期**。
--     换发是滚动动作而不是停机动作，代价是那一小段时间里两个 token 同时有效。
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    token_hash    TEXT NOT NULL UNIQUE,
    role          TEXT NOT NULL CHECK (role IN ('viewer', 'editor', 'admin')),
    created_at    TEXT NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    note          TEXT NOT NULL DEFAULT '',
    expires_at    TEXT,
    prev_token_hash   TEXT,
    prev_valid_until  TEXT
);

CREATE TRIGGER IF NOT EXISTS experiment_events_no_update
BEFORE UPDATE ON experiment_events
BEGIN
    SELECT RAISE(ABORT, 'experiment_events 是 append-only：不允许 UPDATE');
END;

CREATE TRIGGER IF NOT EXISTS experiment_events_no_delete
BEFORE DELETE ON experiment_events
BEGIN
    SELECT RAISE(ABORT, 'experiment_events 是 append-only：不允许 DELETE');
END;
"""

#: 建表之后新增的列（列名 -> 列定义）。
#:
#: 为什么需要这个：``CREATE TABLE IF NOT EXISTS`` 对**已存在**的表什么都不做，
#: 于是"给老库加一列"这件事它管不了。上线过一版之后再改结构，
#: 不写迁移就会在 ``INSERT`` 时报 no such column。
_MIGRATIONS: dict[str, str] = {
    "warehouse_experiment": "TEXT",
    "estimator": "TEXT NOT NULL DEFAULT 'cuped'",
    "analysis_unit": "TEXT NOT NULL DEFAULT 'unit'",
    "metric_type": "TEXT NOT NULL DEFAULT 'mean'",
    # 乐观锁版本号：老库的既有行都从 1 开始（"我们不知道它被改过几次"，
    # 但 1 是唯一诚实的选择 —— 编一个更大的数会假装我们知道历史）
    "version": "INTEGER NOT NULL DEFAULT 1",
    # 护栏的**规格**（方向 + 容忍度），JSON。名字仍在 guardrails 列里：
    # 只有名字没有规格时，护栏分析会判 unknown（"没声明"不等于"通过"）。
    "guardrail_specs": "TEXT NOT NULL DEFAULT '[]'",
}

#: 审计表新增的列。单独一张表，因为它的迁移规则不一样：
#: ``experiment_events`` 上的触发器禁止 UPDATE，所以**只能用
#: ``ALTER TABLE ADD COLUMN`` + 默认值**，绝不能用"先加列再 UPDATE 回填"——
#: 那样会被自己的触发器拒绝，或者（更糟）逼着人去关掉触发器，等于毁掉审计。
_EVENT_MIGRATIONS: dict[str, str] = {
    "actor": "TEXT NOT NULL DEFAULT '（迁移前未知）'",
}

#: 用户表新增的列。三条都允许 NULL，理由见上面 ``users`` 表的注释：
#: 老的 token 迁移后**不设过期**，否则一次迁移就把所有在用的凭据踢下线。
_USER_MIGRATIONS: dict[str, str] = {
    "expires_at": "TEXT",
    "prev_token_hash": "TEXT",
    "prev_valid_until": "TEXT",
}

#: 新签发的 token 默认有效期（天）。
#:
#: 90 天不是行业标准数字，是这台机器上的权衡：比它短，长跑的自动化
#: （CI、看板、别人终端里的 curl）会频繁断开；比它长，"泄露窗口"就接近
#: 永不过期 —— 而这一列存在的全部意义就是让泄露有个头。
#: 要别的值就在 `add_user` / `rotate_token` 时显式给（0 或 None = 永不过期）。
DEFAULT_TOKEN_TTL_DAYS = 90


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at(ttl_days: int | None, at: datetime) -> str | None:
    """TTL -> ``expires_at``。``None`` 与 ``0`` 都表示**永不过期**（要显式选）。"""
    if not ttl_days:
        return None
    return (at + timedelta(days=ttl_days)).isoformat(timespec="seconds")


def _is_past(stamp: str | None, at: datetime) -> bool:
    """``stamp`` 是否已经过期。空字符串与 NULL 都算"没有这条限制"。"""
    if not stamp:
        return False
    return datetime.fromisoformat(stamp) <= at


@dataclass(frozen=True)
class AuthStatus:
    """凭据解析的结果。

    ``reason`` 存在的唯一目的是**说清为什么被拒**（401 的文案要能指导动作：
    是没给、是过期了、还是被停用了），它**绝不参与授权判断** ——
    判断只看 ``user_id`` 是否为空（见 ``ok``）。
    """

    user_id: str | None
    reason: str
    expires_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.user_id is not None

def _coerce_spec(raw: Any) -> Any:
    """把 dict / GuardrailSpec 统一成 ``GuardrailSpec``。

    放在注册表层而不是 API 层：直接调注册表的调用方（脚本、测试、demo 播种）
    也要走同一条规整化路径，否则同一份声明在不同入口会有不同的解释。
    """
    from .guardrails import GuardrailSpec

    if isinstance(raw, GuardrailSpec):
        return raw
    if isinstance(raw, dict):
        return GuardrailSpec.from_dict(raw)
    raise RegistryError(f"护栏规格必须是 dict 或 GuardrailSpec，收到 {type(raw).__name__}")


def _specs_from_json(raw: str | None) -> list[Any]:
    if not raw:
        return []
    from .guardrails import GuardrailSpec

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # 库里存着坏 JSON 时不要把整个读路径打挂：护栏降级成"没有规格"，
        # 于是分析会判 unknown（"没声明"）—— 而不是假装通过。
        return []
    return [GuardrailSpec.from_dict(d) for d in items if isinstance(d, dict)]


#: 角色。最小三分法：读 / 写 / 删。
ROLES = ("viewer", "editor", "admin")

#: 判定口径。**必须与头条结论同一个估计量**，否则监控曲线与结论卡会互相打架。
ESTIMATORS = ("cuped", "post_only")
#: 分析单元。``cluster`` 表示整簇随机化 —— 此时必须用簇级检验，
#: 否则那个看起来完全正常的 p 值背后是 64.5% 的 I 类错误率。
ANALYSIS_UNITS = ("unit", "cluster")
#: 指标类型。``ratio`` 表示比值指标（Σy/Σx），必须走 delta method。
METRIC_TYPES = ("mean", "ratio")


class RegistryError(ValueError):
    """注册表层的输入错误（参数非法、重名、找不到等）。"""


class RegistryConflict(RegistryError):
    """**并发冲突**：调用方拿的版本已经不是当前版本。

    单独一个类型，因为它与"参数非法"是两件事：
      * 参数非法是调用方**写错了**（重试也没用）；
      * 冲突是"你读到的世界已经变了"（重新读、再决定，通常会成功）。
    API 层据此映射成 412 Precondition Failed 而不是 400。
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExperimentEvent:
    """审计表里的一条记录。**只读**（表本身也禁止 UPDATE / DELETE）。"""

    seq: int
    experiment_id: str
    at: str
    action: str
    field: str | None = None
    before: str | None = None
    after: str | None = None
    note: str = ""
    #: 操作者（由服务端从凭据推导）。老库里迁移过来的行是"（迁移前未知）"。
    actor: str = "（迁移前未知）"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        """给人看的一行。"""
        if self.field:
            return (
                f"[{self.at}] {self.action} {self.field}: "
                f"{self.before or '（空）'} -> {self.after or '（空）'}"
            )
        return f"[{self.at}] {self.action} {self.after or ''}".rstrip()


@dataclass
class ExperimentRecord:
    """注册表里的一条实验记录。"""

    name: str
    variants: list[dict[str, Any]]
    salt: str
    hypothesis: str = ""
    owner: str = ""
    layer: str | None = None
    unit: str = "user_id"
    traffic_ratio: float = 1.0
    primary_metric: str = "metric"
    guardrails: list[str] = field(default_factory=list)
    status: str = "draft"
    start_ds: str | None = None
    end_ds: str | None = None
    #: 仅演示用：分析时注入的真实效应。真实平台不会有这一列。
    true_lift: float = 0.0
    #: 绑定的数仓实验名（``ads_experiment_result.experiment``）。
    #: 留空则分析走**合成数据**路径；填了就读数仓。
    #: 与 ``salt`` 不同，这一列**允许后置修改** —— 数仓表要等实验跑完才有，
    #: 而 binding 不改变任何用户的分组，改它不会让已有数据报废。
    warehouse_experiment: str | None = None
    #: 判定口径：``cuped`` 或 ``post_only``。序贯边界、显著性判定、
    #: always-valid p 都按它算 —— **必须与头条结论同一个估计量**。
    #: 和 ``salt`` 一样属于"现在定了就别改"的那类：改了等于换了个判定规则，
    #: 但历史上已经据此做过决定。所以它有 setter（口径是策略不是数据），
    #: 但报告里会把用到的口径写出来。
    estimator: str = "cuped"
    #: 分析单元：``unit``（随机化单元 = 分析单元）或 ``cluster``（整簇随机化）。
    analysis_unit: str = "unit"
    #: 指标类型：``mean``（人均指标）或 ``ratio``（比值指标 Σy/Σx）。
    metric_type: str = "mean"
    id: str = ""
    created_at: str = ""
    #: 乐观锁版本号。每次成功的写 +1；调用方带 ``expected_version`` 且对不上时，
    #: 这次写会被拒（``RegistryConflict``）—— 这就是"防止把别人的改动覆盖掉"。
    version: int = 1
    #: 护栏的**规格**（``GuardrailSpec``）：方向 + 容忍度 + 演示用的真实伤害。
    #: ``guardrails`` 只有名字 —— 名字决定"界面上显示什么"，
    #: 规格决定"能不能判定"（只有名字时判 ``unknown``）。
    guardrail_specs: list[Any] = field(default_factory=list)

    def to_spec(self) -> ExperimentSpec:
        """把记录还原成分流定义 —— 构造即校验。"""
        return ExperimentSpec(
            name=self.name,
            variants=tuple(
                Variant(str(v["name"]), float(v["weight"])) for v in self.variants
            ),
            salt=self.salt,
            unit=self.unit,
            traffic_ratio=float(self.traffic_ratio),
            layer=self.layer,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """实验注册表（sqlite3 后端）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """把老库补齐到当前结构（幂等）。

        ``CREATE TABLE IF NOT EXISTS`` 只管"表不存在"，管不了"表少一列"。
        没有这一步，给老库加列之后第一次写入就会 no such column。

        审计表单独处理：它只能用 ``ADD COLUMN`` + 默认值（触发器禁止 UPDATE，
        回填历史等于改写历史）。见 ``_EVENT_MIGRATIONS``。
        """
        existing = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(experiments)")
        }
        for column, decl in _MIGRATIONS.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE experiments ADD COLUMN {column} {decl}")

        event_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(experiment_events)")
        }
        for column, decl in _EVENT_MIGRATIONS.items():
            if column not in event_cols:
                self._conn.execute(
                    f"ALTER TABLE experiment_events ADD COLUMN {column} {decl}"
                )

        user_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(users)")
        }
        for column, decl in _USER_MIGRATIONS.items():
            if column not in user_cols:
                self._conn.execute(f"ALTER TABLE users ADD COLUMN {column} {decl}")

    def close(self) -> None:
        self._conn.close()

    # -- 用户与凭据 ---------------------------------------------------------- #
    #
    # 这一节的边界（写在代码里，不是只写在 README 里）：
    #   * 这是**静态 token** 鉴权：这一轮补上了过期、带宽限期的轮换与限速
    #     （在此之前三件都没有 —— README 设计决策 58）。
    #     但它仍然**不是** OAuth：没有刷新令牌、没有撤销列表下发、没有设备绑定，
    #     所以 token 在有效期内泄露仍然等于该用户被冒充，`disable_user`
    #     才是立即止损的手段（它不依赖任何时钟）。
    #   * 时间只在凭据这一个面上被用到，所以这里不注入全局时钟，
    #     而是给三个方法留一个**可选的 ``at``**：生产路径用系统时间，
    #     测试可以显式传"现在"。这样"过期"是能测的，而不是要靠 sleep 等出来。
    #   * 它只解决"写操作记谁"，不解决"两个人同时改会互相覆盖"（那是并发控制）。
    #   * 读接口从这一轮起**不再匿名**：鉴权中间件覆盖所有 ``/api``。
    #     ``/healthz``、``/docs`` 与静态页面仍然公开 —— 它们不是 ``/api``。
    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def add_user(
        self,
        user_id: str,
        *,
        role: str = "editor",
        note: str = "",
        token: str | None = None,
        ttl_days: int | None = DEFAULT_TOKEN_TTL_DAYS,
        at: datetime | None = None,
    ) -> str:
        """建用户并返回**明文 token（只此一次）**。

        库里只留哈希，所以这个返回值是拿到凭据的唯一机会 ——
        丢了只能重新发一个（`rotate_token`），不能"再查一次"。

        ``ttl_days`` 默认 90 天；传 ``0`` 或 ``None`` 表示**永不过期**（显式选）。
        """
        if role not in ROLES:
            raise RegistryError(f"角色必须是 {ROLES} 之一，收到 {role!r}")
        if not user_id.strip():
            raise RegistryError("user_id 不能为空")
        now = at or _utcnow()
        raw = token if token is not None else secrets.token_urlsafe(32)
        with self._conn:
            self._conn.execute(
                "INSERT INTO users "
                "(id, token_hash, role, created_at, disabled, note, expires_at) "
                "VALUES (?,?,?,?,0,?,?)",
                (
                    user_id,
                    self.hash_token(raw),
                    role,
                    _now(),
                    note,
                    _expires_at(ttl_days, now),
                ),
            )
        return raw

    def rotate_token(
        self,
        user_id: str,
        *,
        grace_minutes: int = 0,
        ttl_days: int | None = DEFAULT_TOKEN_TTL_DAYS,
        at: datetime | None = None,
    ) -> str:
        """换发 token：新的立即生效，旧的按 ``grace_minutes`` 决定还能用多久。

        默认 ``grace_minutes=0``（旧 token 立刻失效，与这一轮之前的行为一致）。
        宽限期解决的是一个很具体的运维问题：换发之后**正在跑的客户端**
        （CI、看板、别人终端里的 curl）手里只有旧 token，会同时 401 ——
        给一段窗口，换发就从"停机动作"变成"滚动动作"。
        代价必须说清楚：**那段时间里两个 token 同时有效**（不是"更安全"，是"更平滑"）。

        换发同时**重置有效期**（``ttl_days``）：这是"续期"的标准做法，
        否则一个 90 天的 token 在第 89 天轮换后立刻又要过期。
        """
        if grace_minutes < 0:
            raise RegistryError("grace_minutes 不能是负数")
        now = at or _utcnow()
        raw = secrets.token_urlsafe(32)
        grace_until = (
            (now + timedelta(minutes=grace_minutes)).isoformat(timespec="seconds")
            if grace_minutes > 0
            else None
        )
        with self._conn:
            # SQL 的赋值右侧读的都是**旧行**，所以 prev_token_hash 拿到的是换发前的哈希
            cur = self._conn.execute(
                "UPDATE users SET prev_token_hash = token_hash, prev_valid_until = ?, "
                "token_hash = ?, expires_at = ? WHERE id = ?",
                (grace_until, self.hash_token(raw), _expires_at(ttl_days, now), user_id),
            )
            if cur.rowcount == 0:
                raise RegistryError(f"用户不存在：{user_id}")
        return raw

    def disable_user(self, user_id: str, *, disabled: bool = True) -> None:
        with self._conn:
            cur = self._conn.execute(
                "UPDATE users SET disabled = ? WHERE id = ?",
                (1 if disabled else 0, user_id),
            )
            if cur.rowcount == 0:
                raise RegistryError(f"用户不存在：{user_id}")

    def list_users(self) -> list[dict[str, Any]]:
        """用户清单。**永不返回 token 或哈希** —— 连哈希也没必要给人看。"""
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "created_at": row["created_at"],
                "disabled": bool(row["disabled"]),
                "note": row["note"],
                "expires_at": row["expires_at"],
                "grace_until": row["prev_valid_until"],
            }
            for row in self._conn.execute(
                "SELECT id, role, created_at, disabled, note, expires_at, "
                "prev_valid_until FROM users ORDER BY id"
            )
        ]

    def auth_status(self, token: str | None, *, at: datetime | None = None) -> AuthStatus:
        """凭据 -> 用户 id **与拒绝原因**。

        ``reason`` 取值：``ok`` / ``missing`` / ``unknown`` / ``disabled`` / ``expired``。
        宽限期内的旧 token 认（``ok``），过期的旧 token 报 ``expired`` ——
        它的确"存在过"，说成 ``unknown`` 会让运维去查一个不存在的问题。
        """
        if not token:
            return AuthStatus(None, "missing")
        now = at or _utcnow()
        digest = self.hash_token(token)
        row = self._conn.execute(
            "SELECT id, disabled, expires_at, token_hash, prev_valid_until "
            "FROM users WHERE token_hash = ? OR prev_token_hash = ?",
            (digest, digest),
        ).fetchone()
        if row is None:
            return AuthStatus(None, "unknown")
        if row["disabled"]:
            return AuthStatus(None, "disabled", row["expires_at"])
        if row["token_hash"] != digest:
            # 轮换前的旧 token：只在宽限期内认
            if _is_past(row["prev_valid_until"], now) or not row["prev_valid_until"]:
                return AuthStatus(None, "expired", row["expires_at"])
        if _is_past(row["expires_at"], now):
            return AuthStatus(None, "expired", row["expires_at"])
        return AuthStatus(str(row["id"]), "ok", row["expires_at"])

    def authenticate(self, token: str | None, *, at: datetime | None = None) -> str | None:
        """凭据 -> 用户 id。认不出来、被停用、已过期、或没给，都返回 ``None``。

        **这是身份的唯一起点**：调用方（API 层）只能从这里拿 actor，
        绝不允许把请求体里的名字当身份传进 ``_record_event``。

        想看"为什么被拒"就用 ``auth_status`` —— 401 的文案需要它。
        """
        return self.auth_status(token, at=at).user_id

    def role_of(self, user_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT role, disabled FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None or row["disabled"]:
            return None
        return str(row["role"])

    # -- 乐观锁 ------------------------------------------------------------- #
    @staticmethod
    def _check_version(
        record: ExperimentRecord, expected_version: int | None
    ) -> None:
        """版本对不上就拒绝这次写。

        ``expected_version=None`` 表示"调用方没有声明它读的是哪一版" ——
        此时按**后写覆盖**放行（这是默认行为，不是漏洞：单机实验平台上
        大部分调用就是这么用的）。要防止覆盖，就带上版本号。
        """
        if expected_version is None:
            return
        if int(expected_version) != record.version:
            raise RegistryConflict(
                f"版本冲突：你读到的是 v{int(expected_version)}，"
                f"当前已经是 v{record.version} —— "
                "说明这中间有人改过。请重新读取后再提交（这次写没有生效）。"
            )

    def _bump_version(self, experiment_id: str, note_prefix: str) -> int:
        """把版本 +1，返回新版本号。**与业务更新在同一事务里**。"""
        self._conn.execute(
            "UPDATE experiments SET version = version + 1 WHERE id = ?",
            (experiment_id,),
        )
        row = self._conn.execute(
            "SELECT version FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return int(row["version"]) if row else 1

    # -- 审计（append-only） ------------------------------------------------ #
    def _record_event(
        self,
        experiment_id: str,
        action: str,
        *,
        actor: str,
        field: str | None = None,
        before: Any = None,
        after: Any = None,
        note: str = "",
    ) -> None:
        """把一条审计写进**当前事务**。

        **故意不 commit**：调用方把业务变更和这一条放在同一个 ``with self._conn``
        里，要么都落盘、要么都不落。分开写就会出现"改了但没记"——
        而审计表里"没有这条"与"这件事没发生"长得一模一样，是最难发现的那种缺失。

        ``actor`` 是**必填的具名参数**：这样"忘了记是谁"在调用点就报错，
        而不是静默写进一个空字符串（那正是这一列以前一直是空的原因）。
        """
        self._conn.execute(
            """
            INSERT INTO experiment_events
            (experiment_id, at, action, field, before, after, note, actor)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                experiment_id,
                _now(),
                action,
                field,
                None if before is None else str(before),
                None if after is None else str(after),
                note,
                actor,
            ),
        )

    def events(self, experiment_id: str, *, limit: int | None = None) -> list[ExperimentEvent]:
        """某个实验的全部审计，按发生顺序。

        **实验被删掉之后这里仍然查得到** —— 审计表没有外键级联，这是有意的：
        "谁删了它"恰恰是删掉之后最需要回答的问题。
        """
        sql = "SELECT * FROM experiment_events WHERE experiment_id = ? ORDER BY seq"
        params: tuple[Any, ...] = (experiment_id,)
        if limit is not None:
            sql = (
                "SELECT * FROM (SELECT * FROM experiment_events WHERE experiment_id = ? "
                "ORDER BY seq DESC LIMIT ?) ORDER BY seq"
            )
            params = (experiment_id, int(limit))
        return [
            ExperimentEvent(
                seq=row["seq"],
                experiment_id=row["experiment_id"],
                at=row["at"],
                action=row["action"],
                field=row["field"],
                before=row["before"],
                after=row["after"],
                note=row["note"],
                actor=row["actor"],
            )
            for row in self._conn.execute(sql, params)
        ]

    def recent_events(self, *, limit: int = 50) -> list[ExperimentEvent]:
        """全局最近的操作（跨实验，倒序）——用于"最近发生了什么"。"""
        rows = self._conn.execute(
            "SELECT * FROM experiment_events ORDER BY seq DESC LIMIT ?", (int(limit),)
        )
        return [
            ExperimentEvent(
                seq=row["seq"],
                experiment_id=row["experiment_id"],
                at=row["at"],
                action=row["action"],
                field=row["field"],
                before=row["before"],
                after=row["after"],
                note=row["note"],
                actor=row["actor"],
            )
            for row in rows
        ]

    # -- 校验 -------------------------------------------------------------- #
    @staticmethod
    def _validate(
        name: str,
        variants: Sequence[dict[str, Any]],
        salt: str,
        unit: str,
        traffic_ratio: float,
        layer: str | None,
        status: str,
    ) -> None:
        if not name or not name.strip():
            raise RegistryError("实验名不能为空")
        if status not in STATUSES:
            raise RegistryError(f"status 必须是 {STATUSES} 之一，收到 {status!r}")
        if not salt or not salt.strip():
            raise RegistryError("salt 不能为空（它决定每个用户的分组）")
        # 直接复用 M0 的分流定义校验：权重之和、重名分支、流量比例……
        try:
            ExperimentSpec(
                name=name,
                variants=tuple(
                    Variant(str(v.get("name", "")), float(v.get("weight", 0.0)))
                    for v in variants
                ),
                salt=salt,
                unit=unit,
                traffic_ratio=float(traffic_ratio),
                layer=layer,
            )
        except ValueError as exc:  # 转成注册表层的错误类型，便于 API 统一处理
            raise RegistryError(str(exc)) from exc

    # -- 写 ---------------------------------------------------------------- #
    def create(
        self,
        *,
        actor: str,
        name: str,
        variants: Sequence[dict[str, Any]],
        salt: str | None = None,
        hypothesis: str = "",
        owner: str = "",
        layer: str | None = None,
        unit: str = "user_id",
        traffic_ratio: float = 1.0,
        primary_metric: str = "metric",
        guardrails: Sequence[str] = (),
        guardrail_specs: Sequence[Any] = (),
        status: str = "draft",
        start_ds: str | None = None,
        end_ds: str | None = None,
        true_lift: float = 0.0,
        warehouse_experiment: str | None = None,
        estimator: str = "cuped",
        analysis_unit: str = "unit",
        metric_type: str = "mean",
    ) -> ExperimentRecord:
        """新建实验。非法配置在写入前就被拒掉。"""
        if not variants:
            raise RegistryError("至少需要一个分支")
        if estimator not in ESTIMATORS:
            raise RegistryError(f"estimator 必须是 {ESTIMATORS} 之一，收到 {estimator!r}")
        if analysis_unit not in ANALYSIS_UNITS:
            raise RegistryError(
                f"analysis_unit 必须是 {ANALYSIS_UNITS} 之一，收到 {analysis_unit!r}"
            )
        if metric_type not in METRIC_TYPES:
            raise RegistryError(f"metric_type 必须是 {METRIC_TYPES} 之一，收到 {metric_type!r}")
        # 簇级 CUPED 现在**放行**了：05 路 DWS 一直落着簇级的
        # pre_sum / pre_sq_sum / pre_post_cross_sum（当时的注释就写着
        # "将来做簇级 CUPED 时不必改这一层"），只是平台侧硬编码成了 post_only。
        # 那个硬编码与这条闸门都基于一个**过时的假设**（"簇 DGP 没有前置期"），
        # 实测数仓里 pre_sum ≈ 4.4e6，前置期是有的。
        # 现在按声明走：estimator 是什么就用什么；真拿不到簇级前置指标时，
        # 由分析层报错（而不是在这里提前拦住一条其实走得通的路）。

        if metric_type == "ratio" and estimator == "cuped":
            # 同一个理由、同一个位置：CUPED 需要**前置协变量**，
            # 而比值链路（DWS/ADS 06/07）里落下的是分子与分母，没有前置指标。
            # 比值指标只能用 delta method。
            raise RegistryError(
                "metric_type='ratio' 与 estimator='cuped' 不能同时选："
                "CUPED 需要前置协变量，而比值 ADS 里只有分子/分母两列；"
                "请把 estimator 设为 'post_only'（比值指标用 delta method）"
            )

        # 未指定 salt 时派生一个**不随改名变化**的稳定 salt
        resolved_salt = salt or f"{name.strip()}_v1"
        self._validate(
            name, variants, resolved_salt, unit, traffic_ratio, layer, status
        )
        binding = warehouse_experiment.strip() if warehouse_experiment else None
        if warehouse_experiment is not None and not binding:
            raise RegistryError("warehouse_experiment 不能是空白字符串（留空请传 None）")

        if self.get_by_name(name) is not None:
            raise RegistryError(f"实验名 {name!r} 已存在（名字是唯一键）")

        record = ExperimentRecord(
            id=uuid.uuid4().hex[:12],
            name=name.strip(),
            variants=[dict(v) for v in variants],
            salt=resolved_salt,
            hypothesis=hypothesis,
            owner=owner,
            layer=layer,
            unit=unit,
            traffic_ratio=float(traffic_ratio),
            primary_metric=primary_metric,
            guardrails=list(guardrails),
            guardrail_specs=[_coerce_spec(s) for s in guardrail_specs],
            status=status,
            start_ds=start_ds,
            end_ds=end_ds,
            true_lift=float(true_lift),
            warehouse_experiment=binding,
            estimator=estimator,
            analysis_unit=analysis_unit,
            metric_type=metric_type,
            created_at=_now(),
        )

        self._conn.execute(
            """
            INSERT INTO experiments
            (id, name, hypothesis, owner, layer, unit, salt, traffic_ratio,
             variants, primary_metric, guardrails, status, start_ds, end_ds,
             true_lift, warehouse_experiment, estimator, analysis_unit, metric_type,
             created_at, version, guardrail_specs)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.id, record.name, record.hypothesis, record.owner, record.layer,
                record.unit, record.salt, record.traffic_ratio,
                json.dumps(record.variants, ensure_ascii=False),
                record.primary_metric,
                json.dumps(record.guardrails, ensure_ascii=False),
                record.status, record.start_ds, record.end_ds,
                record.true_lift, record.warehouse_experiment, record.estimator,
                record.analysis_unit, record.metric_type,
                record.created_at, record.version,
                json.dumps([s.to_dict() for s in record.guardrail_specs], ensure_ascii=False),
            ),
        )
        self._record_event(
            record.id,
            "create",
            actor=actor,
            after=record.name,
            note=f"salt={record.salt}；口径={record.estimator}；单元={record.analysis_unit}",
        )
        self._conn.commit()
        return record

    def set_status(
        self,
        experiment_id: str,
        status: str,
        *,
        actor: str,
        expected_version: int | None = None,
    ) -> ExperimentRecord:
        """只允许改状态 —— 分流定义一旦上线就不能动。

        带 ``expected_version`` 时是**乐观锁**语义：版本对不上直接拒，
        不会被"后写覆盖"悄悄吃掉别人的改动。
        """
        if status not in STATUSES:
            raise RegistryError(f"status 必须是 {STATUSES} 之一，收到 {status!r}")
        record = self.get(experiment_id)
        self._check_version(record, expected_version)
        with self._conn:  # 变更 + 版本 + 审计同一事务
            self._conn.execute(
                "UPDATE experiments SET status = ? WHERE id = ?", (status, experiment_id)
            )
            record.version = self._bump_version(experiment_id, "set_status")
            self._record_event(
                experiment_id, "set_status", actor=actor, field="status",
                before=record.status, after=status,
                note=f"v{record.version - 1} -> v{record.version}",
            )
        record.status = status
        return record

    def stop_with_reason(
        self,
        experiment_id: str,
        *,
        actor: str,
        reason: str,
        expected_version: int | None = None,
    ) -> ExperimentRecord:
        """**带理由地停止实验**（护栏触发、人工叫停都走这里）。

        为什么不用 ``set_status(id, "stopped")`` 了事：审计要能回答"**为什么**停的"，
        而 ``set_status`` 只记 ``draft -> stopped``。停实验是这一整套平台里
        **最不可逆**的动作（分流随时可以重开，但已经造成的伤害收不回来），
        它的审计必须比"改了个状态"更厚：谁停的、依据是什么、当时的值是多少。
        所以理由进 ``note``，动作名单独记为 ``stop``。
        """
        if not reason.strip():
            raise RegistryError("停止实验必须给出理由（审计要能回答为什么）")
        record = self.get(experiment_id)
        if record.status == "stopped":
            raise RegistryError(f"实验 {record.name!r} 已经是 stopped")
        self._check_version(record, expected_version)
        with self._conn:
            self._conn.execute(
                "UPDATE experiments SET status = 'stopped' WHERE id = ?",
                (experiment_id,),
            )
            record.version = self._bump_version(experiment_id, "stop")
            self._record_event(
                experiment_id, "stop", actor=actor, field="status",
                before=record.status, after="stopped",
                note=f"{reason}；v{record.version - 1} -> v{record.version}",
            )
        record.status = "stopped"
        return record

    def set_estimator(
        self,
        experiment_id: str,
        estimator: str,
        *,
        actor: str,
        expected_version: int | None = None,
    ) -> ExperimentRecord:
        """切换判定口径。

        允许改 —— 口径是**策略**不是数据，改了不会让任何人的分组失效。
        但要注意它改变的是判定规则本身：切完之后，"历史上有没有越界"这个问题
        的答案会跟着变。所以报告里始终写着用的是哪个口径，
        而**这次改动会进审计表** —— 否则"改口径"就是一条事后挑口径的通道。
        """
        if estimator not in ESTIMATORS:
            raise RegistryError(f"estimator 必须是 {ESTIMATORS} 之一，收到 {estimator!r}")
        record = self.get(experiment_id)
        self._check_version(record, expected_version)
        with self._conn:
            self._conn.execute(
                "UPDATE experiments SET estimator = ? WHERE id = ?", (estimator, experiment_id)
            )
            record.version = self._bump_version(experiment_id, "set_estimator")
            self._record_event(
                experiment_id, "set_estimator", actor=actor, field="estimator",
                before=record.estimator, after=estimator,
                note="判定口径变更：历史结论的判定规则会随之改变；"
                     f"v{record.version - 1} -> v{record.version}",
            )
        record.estimator = estimator
        return record

    def bind_warehouse(
        self,
        experiment_id: str,
        warehouse_experiment: str | None,
        *,
        actor: str,
        expected_version: int | None = None,
    ) -> ExperimentRecord:
        """绑定/解绑数仓实验。

        **允许后置修改，而且不影响任何已有结论的正确性** —— 因为这个字段
        只决定"从哪里读数"，不改变任何用户的分组。真正不可变的是 ``salt``。
        """
        record = self.get(experiment_id)
        binding = warehouse_experiment.strip() if warehouse_experiment else None
        if warehouse_experiment is not None and not binding:
            raise RegistryError("warehouse_experiment 不能是空白字符串（解绑请传 None）")
        self._check_version(record, expected_version)
        with self._conn:
            self._conn.execute(
                "UPDATE experiments SET warehouse_experiment = ? WHERE id = ?",
                (binding, experiment_id),
            )
            record.version = self._bump_version(experiment_id, "bind_warehouse")
            self._record_event(
                experiment_id, "bind_warehouse", actor=actor, field="warehouse_experiment",
                before=record.warehouse_experiment, after=binding,
                note="数据源绑定变更：改变读哪份数据，不改变任何用户的分组；"
                     f"v{record.version - 1} -> v{record.version}",
            )
        record.warehouse_experiment = binding
        return record

    def delete(self, experiment_id: str, *, actor: str, expected_version: int | None = None) -> None:
        record = self.get(experiment_id)
        if record is None:
            raise RegistryError(f"找不到实验 {experiment_id!r}")
        # 删除也要检查版本：一个人正打算改状态，另一个人把实验删了 ——
        # 前者应当收到"你读到的世界已经变了"，而不是写进一个不存在的行（静默无效果）。
        self._check_version(record, expected_version)
        with self._conn:
            self._conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            # 审计**不删**：删掉实验之后，"谁删的、删之前是什么状态"正是要回答的问题。
            self._record_event(
                experiment_id, "delete", actor=actor,
                before=experiment_id, after=None,
                note="实验已删除；这条审计保留（append-only，无级联）",
            )

    # -- 读 ---------------------------------------------------------------- #
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            id=row["id"],
            name=row["name"],
            hypothesis=row["hypothesis"],
            owner=row["owner"],
            layer=row["layer"],
            unit=row["unit"],
            salt=row["salt"],
            traffic_ratio=row["traffic_ratio"],
            variants=json.loads(row["variants"]),
            primary_metric=row["primary_metric"],
            guardrails=json.loads(row["guardrails"]),
            status=row["status"],
            start_ds=row["start_ds"],
            end_ds=row["end_ds"],
            true_lift=row["true_lift"],
            warehouse_experiment=row["warehouse_experiment"],
            estimator=row["estimator"],
            analysis_unit=row["analysis_unit"],
            metric_type=row["metric_type"],
            created_at=row["created_at"],
            version=int(row["version"]),
            guardrail_specs=_specs_from_json(row["guardrail_specs"]),
        )

    def get(self, experiment_id: str) -> ExperimentRecord:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise RegistryError(f"找不到实验 {experiment_id!r}")
        return self._row_to_record(row)

    def get_by_name(self, name: str) -> ExperimentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self, *, status: str | None = None) -> list[ExperimentRecord]:
        sql = "SELECT * FROM experiments"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY created_at DESC, name"
        return [self._row_to_record(r) for r in self._conn.execute(sql, params)]

    def count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()["n"]
        )
