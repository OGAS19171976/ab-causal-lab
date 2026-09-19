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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..assignment import ExperimentSpec, Variant

__all__ = [
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
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    token_hash    TEXT NOT NULL UNIQUE,
    role          TEXT NOT NULL CHECK (role IN ('viewer', 'editor', 'admin')),
    created_at    TEXT NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    note          TEXT NOT NULL DEFAULT ''
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
}

#: 审计表新增的列。单独一张表，因为它的迁移规则不一样：
#: ``experiment_events`` 上的触发器禁止 UPDATE，所以**只能用
#: ``ALTER TABLE ADD COLUMN`` + 默认值**，绝不能用"先加列再 UPDATE 回填"——
#: 那样会被自己的触发器拒绝，或者（更糟）逼着人去关掉触发器，等于毁掉审计。
_EVENT_MIGRATIONS: dict[str, str] = {
    "actor": "TEXT NOT NULL DEFAULT '（迁移前未知）'",
}

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

    def close(self) -> None:
        self._conn.close()

    # -- 用户与凭据 ---------------------------------------------------------- #
    #
    # 这一节的边界（写在代码里，不是只写在 README 里）：
    #   * 这是**静态 token** 鉴权：没有过期、没有轮换、没有限速。
    #     token 泄露 = 该用户被冒充，且只有 `disable_user` 能止损。
    #   * 它只解决"写操作记谁"，不解决"两个人同时改会互相覆盖"（那是并发控制）。
    #   * 读接口仍然匿名（单机实验平台，读不是威胁面）—— 这一点必须说出来，
    #     否则"平台做了鉴权"会被理解成"什么都挡住了"。
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
    ) -> str:
        """建用户并返回**明文 token（只此一次）**。

        库里只留哈希，所以这个返回值是拿到凭据的唯一机会 ——
        丢了只能重新发一个（`rotate_token`），不能"再查一次"。
        """
        if role not in ROLES:
            raise RegistryError(f"角色必须是 {ROLES} 之一，收到 {role!r}")
        if not user_id.strip():
            raise RegistryError("user_id 不能为空")
        raw = token if token is not None else secrets.token_urlsafe(32)
        with self._conn:
            self._conn.execute(
                "INSERT INTO users (id, token_hash, role, created_at, disabled, note) "
                "VALUES (?,?,?,?,0,?)",
                (user_id, self.hash_token(raw), role, _now(), note),
            )
        return raw

    def rotate_token(self, user_id: str) -> str:
        """换发 token（旧 token 立即失效）。用户不存在时抛错。"""
        raw = secrets.token_urlsafe(32)
        with self._conn:
            cur = self._conn.execute(
                "UPDATE users SET token_hash = ? WHERE id = ?",
                (self.hash_token(raw), user_id),
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
            }
            for row in self._conn.execute(
                "SELECT id, role, created_at, disabled, note FROM users ORDER BY id"
            )
        ]

    def authenticate(self, token: str | None) -> str | None:
        """凭据 -> 用户 id。认不出来、被停用、或没给，都返回 ``None``。

        **这是身份的唯一起点**：调用方（API 层）只能从这里拿 actor，
        绝不允许把请求体里的名字当身份传进 ``_record_event``。
        """
        if not token:
            return None
        row = self._conn.execute(
            "SELECT id, disabled FROM users WHERE token_hash = ?",
            (self.hash_token(token),),
        ).fetchone()
        if row is None or row["disabled"]:
            return None
        return str(row["id"])

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
        if analysis_unit == "cluster" and estimator == "cuped":
            # 早失败：整簇路径需要**簇级**前置指标才能做 CUPED，
            # 而当前数仓/合成数据都还没有它。与其在分析时报错，不如创建时就拦住。
            raise RegistryError(
                "analysis_unit='cluster' 与 estimator='cuped' 不能同时选："
                "整簇随机化需要簇级的前置指标才能做 CUPED，当前数据源没有提供；"
                "请把 estimator 设为 'post_only'"
            )

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
             created_at, version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
