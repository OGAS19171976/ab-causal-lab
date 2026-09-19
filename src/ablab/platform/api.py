"""FastAPI 服务：把注册表与引擎暴露成 HTTP 接口。

接口设计上的三个取舍
--------------------
**1. 分析是 POST 而不是 GET。**
``POST /api/experiments/{id}/analyze`` 可以带参数（样本量、alpha），
而且**会返回一份体检报告而不是一个数**。把它做成 GET 会诱导调用方
"读一个效应值就完事"，而这恰恰是本项目从 M0 起就在反对的做法。

**2. 创建实验时把校验交给注册表。**
接口层只做类型转换，所有业务规则（权重和、重名、流量比例）
都在 ``ExperimentRegistry`` 里，而它又复用 ``ExperimentSpec``。
**校验逻辑自始至终只有一份。**

**3. 数据源由记录上的绑定决定，不由接口决定。**
``warehouse_experiment`` 留空 → 走合成数据；填了 → 按**只读**连接去 ADS + DWS 读。
接口层不提供"临时换个数据源算一次"的口子 —— 否则同一个实验会有两份口径不同的报告，
而报告里分不清哪份是哪个来源。数据源写在记录上，报告里也带 ``source`` 字段。

顺带：FastAPI 自动生成 ``/docs``（Swagger UI）与 ``/openapi.json``，
这两个是免费的，也让这套接口可以被别的客户端直接消费。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..inference import mde, required_n_per_arm
from .analysis import (
    analyse_experiment,
    analyse_experiment_from_warehouse,
    run_aa_validation,
)
from .datasource import list_warehouse_experiments
from .registry import (
    ROLES,
    STATUSES,
    ExperimentRegistry,
    RegistryConflict,
    RegistryError,
)

__all__ = ["create_app", "default_registry_path", "default_warehouse_path"]

_STATIC_DIR = Path(__file__).parent / "static"


class _Strict(BaseModel):
    """所有请求模型的基类：**多余字段直接报错**。

    默认行为是静默忽略未知字段，这在实验平台上特别危险 ——
    客户端写错一个字段名（或者用了还没上线的字段），请求照样 200，
    而配置被悄悄丢掉了。实测踩过一次：``analysis_unit="cluster"`` 被忽略，
    于是"整簇随机化"静默变成了"单元级"，而这两种做法的 I 类错误率差 10 倍以上。

    宁可 422 让调用方立刻发现，也不要 200 之后给出一个口径不对的结论。
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class VariantIn(_Strict):
    name: str
    weight: float


class ExperimentIn(_Strict):
    name: str = Field(..., min_length=1, description="实验唯一名，注册表的唯一键")
    variants: list[VariantIn] = Field(..., min_length=1)
    salt: str | None = Field(
        None, description="分流盐。留空则由实验名派生；**上线后不可更改**"
    )
    hypothesis: str = ""
    owner: str = ""
    layer: str | None = None
    unit: str = "user_id"
    traffic_ratio: float = 1.0
    primary_metric: str = "metric"
    guardrails: list[str] = []
    status: Literal["draft", "running", "stopped"] = "draft"
    start_ds: str | None = None
    end_ds: str | None = None
    true_lift: float = Field(
        0.0, description="仅演示用：分析时注入的真实效应。真实平台没有这个字段"
    )
    warehouse_experiment: str | None = Field(
        None,
        description=(
            "绑定的数仓实验名（ads_experiment_result.experiment）。"
            "留空 = 分析走合成数据；填了 = 读 ADS + DWS。**允许后置修改**"
        ),
    )
    estimator: Literal["cuped", "post_only"] = Field(
        "cuped",
        description=(
            "判定口径。序贯边界、显著性判定、always-valid p 都按它算 —— "
            "必须与头条结论同一个估计量，否则监控曲线会和结论卡互相打架"
        ),
    )
    analysis_unit: Literal["unit", "cluster"] = Field(
        "unit",
        description=(
            "分析单元。**整簇随机化必须显式声明 cluster**，否则平台会做单元级 t 检验，"
            "而那种误用的 I 类错误率在 60% 以上（实测 69.3%）—— 且 p 值看起来完全正常"
        ),
    )
    metric_type: Literal["mean", "ratio"] = Field(
        "mean",
        description=(
            "指标类型。ratio 表示比值指标 Σy/Σx（曝光做分母），会用 delta method —— "
            "它和人均比值不是同一个量，口径差可达 20% 以上"
        ),
    )


class EstimatorIn(_Strict):
    estimator: Literal["cuped", "post_only"]


class BindIn(_Strict):
    warehouse_experiment: str | None = Field(
        None, description="设为目标实验名即绑定，传 null 即解绑"
    )


class StatusIn(_Strict):
    status: Literal["draft", "running", "stopped"]


class AnalyzeIn(_Strict):
    n_users: int = Field(
        20_000,
        ge=200,
        le=200_000,
        description="仅**合成数据**路径使用；数仓路径下数据是既成的，该参数被忽略",
    )
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    n_looks: int = Field(5, ge=2, le=20)
    seed: int | None = Field(
        None,
        ge=0,
        description="合成数据种子。留空则由实验 salt 确定性派生（salt 不可变，故同一实验永远同一份数据）",
    )


class AAIn(_Strict):
    n_trials: int = Field(400, ge=50, le=5_000)
    n_units: int = Field(8_000, ge=500, le=100_000)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)


class DesignIn(_Strict):
    """实验前定量：要检出多大的相对提升，需要多少样本。"""

    baseline_mean: float = Field(..., gt=0.0, description="指标的基线均值")
    sd: float = Field(..., gt=0.0, description="单个观测的标准差（**不是**标准误）")
    relative_mde: float = Field(
        ..., gt=0.0, lt=1.0, description="想检出的相对提升，如 0.02 表示 2%"
    )
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    power: float = Field(0.8, gt=0.0, lt=1.0)
    treatment_ratio: float = Field(
        0.5, gt=0.0, lt=1.0, description="分给处理组的比例；不等权会明显降低效率"
    )
    pre_post_correlation: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description="前置指标与后置指标的相关系数。>0 表示用 CUPED，样本量需求按 1-ρ² 下降",
    )


# --------------------------------------------------------------------------- #
# 身份：凭据 -> 操作者
# --------------------------------------------------------------------------- #
#
# 三条不可动摇的规则（都在测试里钉着）：
#   1. **操作者只从凭据推导**，永远不从请求体里读 —— 客户端能填的名字不是身份，
#      只是声明。审计字段一旦可以被请求方指定，它就比没有更糟（看起来权威）。
#   2. **认不出来就是 401，角色不够就是 403**，而且被拒绝的操作**不写审计** ——
#      否则审计会被失败的尝试淹没，"谁改了什么"要翻十页才看得见。
#   3. **读接口匿名**。这是单机实验平台的取舍，不是遗漏；说出来，别让人以为
#      "做了鉴权 = 什么都挡住了"。
#: 不需要凭据的写路径。**目前为空** —— 将来加"登录接口"时放这里，
#: 并且测试会读同一个常量，不会出现"测试名单和实现名单不一致"。
PUBLIC_WRITE_PATHS: frozenset[str] = frozenset()

#: 哪些方法算"写"。读（GET/HEAD/OPTIONS）不受影响：本平台读接口是匿名的，
#: 这是单机实验平台的取舍，写在这里以免被当成遗漏。
WRITE_METHODS = ("POST", "PUT", "PATCH", "DELETE")

_BEARER_PREFIX = "bearer "
_READ_ROLES = {"viewer", "editor", "admin"}


def bearer_token(authorization: str | None) -> str | None:
    """从 `Authorization: Bearer <token>` 里取出 token；格式不对返回 None。"""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.strip().lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def expected_version_of(request: Request) -> int | None:
    """读 `If-Match` 头里的版本号（HTTP 原生语义）。

    为什么用 `If-Match` 而不是自定义头或请求体字段：这是 HTTP 里
    "只有当资源还是我以为的那一版时才执行"的**标准**表达，
    冲突时标准的状态码就是 **412 Precondition Failed** ——
    前端、代理、测试工具都认识它，不需要我们发明约定。

    没带这个头时返回 None，注册表按"后写覆盖"放行（默认行为，见已知边界）。
    格式不对（不是整数、负数）则直接 400：那是调用方写错了，不是冲突。
    """
    raw = request.headers.get("if-match")
    if raw is None or not raw.strip():
        return None
    text = raw.strip().strip('"').lstrip("W/").strip('"')
    try:
        version = int(text)
    except ValueError as exc:
        raise HTTPException(
            400, f"If-Match 必须是版本号（整数），收到 {raw!r}"
        ) from exc
    if version < 1:
        raise HTTPException(400, f"If-Match 必须 >= 1，收到 {version}")
    return version


def require_role(app: FastAPI, authorization: str | None, minimum: str) -> str:
    """校验凭据与角色，返回**用户 id**（而不是请求里声称的名字）。

    `app` 走参数而不是闭包：`create_app` 里定义的依赖解析不到注解（见那里的注释）。
    """
    registry: ExperimentRegistry = app.state.registry
    user_id = registry.authenticate(bearer_token(authorization))
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="需要凭据：Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = registry.role_of(user_id)
    if role not in _READ_ROLES or ROLES.index(role) < ROLES.index(minimum):
        raise HTTPException(
            status_code=403,
            detail=f"该操作需要 {minimum} 及以上角色，当前是 {role}",
        )
    return user_id


# --------------------------------------------------------------------------- #
# 应用
# --------------------------------------------------------------------------- #
def create_app(
    registry_path: str | Path,
    warehouse_path: str | Path | None = None,
) -> FastAPI:
    """构造应用。

    ``warehouse_path`` 指向 DuckDB 数仓（可选）。给了它，绑定了数仓实验的记录
    就能走真实链路；没给则所有分析都走合成数据。
    """
    app = FastAPI(
        title="ab-causal-lab 实验平台",
        description=(
            "确定性分流 + 实验分析 + 因果推断验证台的 HTTP 接口。\n\n"
            "分析接口返回的是**体检报告**（SRM、CUPED、效应分解、序贯监控），不只是一个效应值。"
        ),
        version="0.1.0",
    )
    registry = ExperimentRegistry(registry_path)
    app.state.registry = registry
    wh_path = Path(warehouse_path) if warehouse_path is not None else None
    app.state.warehouse_path = wh_path if wh_path and wh_path.exists() else None

    @contextlib.contextmanager
    def warehouse_connection() -> Iterator[Any]:
        """每次请求开一个**只读**连接。

        DuckDB 连接不适合跨线程共用（FastAPI 的同步端点跑在线程池里），
        而连接开销只有毫秒级 —— 按请求开一个新连接比加锁更简单也更安全。
        """
        if app.state.warehouse_path is None:
            raise HTTPException(
                400, "服务启动时没有提供数仓（--warehouse），无法读取真实链路数据"
            )
        import duckdb

        con = duckdb.connect(str(app.state.warehouse_path), read_only=True)
        try:
            yield con
        finally:
            con.close()

    def _handle(exc: RegistryError) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    def _conflict(exc: RegistryConflict) -> HTTPException:
        """版本冲突是 412，不是 400：调用方重读一遍再提交通常会成功。"""
        return HTTPException(status_code=412, detail=str(exc))

    # ---- 身份：**服务端从凭据推导，绝不相信请求体里的名字** -------------- #
    #
    # 这里刻意**不用** `Depends(...)` 注解：本模块有
    # `from __future__ import annotations`，而 FastAPI 求值注解时只用
    # **模块全局**命名空间 —— 定义在 `create_app` 里的依赖（闭包）解析不到，
    # 会被当成普通参数（实测：`actor` 变成了必填的 query 参数，直接 422）。
    # 所以身份校验写成"模块级函数 + 端点内显式调用"；"忘记加"这件事由
    # `tests/test_governance.py::TestAuthAndActor` 自动枚举所有写路由来兜底。
    def actor_of(request: Request) -> str:
        """取中间件已经校验并放好的操作者。

        中间件在**解析请求体之前**就完成了校验（见 `_guard_mutations`），
        所以这里不会再抛 401/403；但**失败要关闭**：状态上没有 actor 时仍然
        拒绝，而不是写一条没有操作者的审计（那正是这一列以前一直是空的原因）。
        """
        user_id = getattr(request.state, "actor", None)
        if not user_id:
            raise HTTPException(401, "这条写路径没有经过身份校验（内部错误，已拒绝）")
        return str(user_id)

    # ---- 身份中间件：在**请求体校验之前**拦下无凭据的写请求 -------------- #
    #
    # 为什么必须是中间件而不是"在端点里调用一次"：FastAPI 先校验请求体、
    # 再进入函数体，所以匿名调用者会先拿到 422（还附带了 schema 详情），
    # 而不是 401 —— 实测就是这样，被自动枚举路由的测试抓出来了。
    # 放在中间件里，顺序就变成"先认证、再校验"，也顺带保证**没有哪个写端点
    # 能绕过它**（包括将来新加的）。
    @app.middleware("http")
    async def _guard_mutations(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if request.method in WRITE_METHODS and path.startswith("/api") and path not in PUBLIC_WRITE_PATHS:
            minimum = "admin" if request.method == "DELETE" else "editor"
            try:
                request.state.actor = require_role(
                    app, request.headers.get("authorization"), minimum
                )
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code, content={"detail": exc.detail}
                )
        return await call_next(request)

    # ---- 基础 ------------------------------------------------------------ #
    @app.get("/healthz", tags=["基础"])
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "experiments": registry.count(), "version": "0.1.0"}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    # ---- 注册表 ---------------------------------------------------------- #
    @app.get("/api/experiments", tags=["注册表"])
    def list_experiments(status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in STATUSES:
            raise HTTPException(400, f"status 必须是 {STATUSES} 之一")
        return [r.to_dict() for r in registry.list(status=status)]

    @app.post("/api/experiments", status_code=201, tags=["注册表"])
    def create_experiment(payload: ExperimentIn, request: Request) -> dict[str, Any]:
        # 身份来自凭据；payload 里的 owner 只是业务字段，不参与审计
        actor = actor_of(request)
        try:
            record = registry.create(actor=actor, **payload.model_dump())
        except RegistryError as exc:
            raise _handle(exc) from exc
        return record.to_dict()

    @app.get("/api/experiments/{experiment_id}", tags=["注册表"])
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        try:
            return registry.get(experiment_id).to_dict()
        except RegistryError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.patch("/api/experiments/{experiment_id}/status", tags=["注册表"])
    def set_status(
        experiment_id: str, payload: StatusIn, request: Request
    ) -> dict[str, Any]:
        actor = actor_of(request)
        expected = expected_version_of(request)
        try:
            return registry.set_status(
                experiment_id, payload.status, actor=actor, expected_version=expected
            ).to_dict()
        except RegistryConflict as exc:
            raise _conflict(exc) from exc
        except RegistryError as exc:
            raise HTTPException(404 if "找不到" in str(exc) else 400, str(exc)) from exc

    @app.delete("/api/experiments/{experiment_id}", status_code=204, tags=["注册表"])
    def delete_experiment(experiment_id: str, request: Request) -> JSONResponse:
        actor = actor_of(request)
        try:
            registry.delete(
                experiment_id, actor=actor, expected_version=expected_version_of(request)
            )
        except RegistryConflict as exc:
            raise _conflict(exc) from exc
        except RegistryError as exc:
            raise HTTPException(404, str(exc)) from exc
        return JSONResponse(status_code=204, content=None)

    @app.post("/api/experiments/{experiment_id}/estimator", tags=["注册表"])
    def set_estimator(
        experiment_id: str, payload: EstimatorIn, request: Request
    ) -> dict[str, Any]:
        actor = actor_of(request)
        expected = expected_version_of(request)
        try:
            return registry.set_estimator(
                experiment_id, payload.estimator, actor=actor, expected_version=expected
            ).to_dict()
        except RegistryConflict as exc:
            raise _conflict(exc) from exc
        except RegistryError as exc:
            raise HTTPException(404 if "找不到" in str(exc) else 400, str(exc)) from exc

    # ---- 操作审计（只读） -------------------------------------------------- #
    @app.get("/api/experiments/{experiment_id}/events", tags=["审计"])
    def experiment_events(experiment_id: str, limit: int | None = None) -> dict[str, Any]:
        """某个实验的操作审计（append-only，按发生顺序）。

        **实验删掉之后这个接口仍然可用** —— 那正是最需要它的时候。
        所以这里**不**先校验实验是否存在（否则删掉之后永远 404，
        而"谁删的"恰恰是要回答的问题）。
        """
        events = registry.events(experiment_id, limit=limit)
        return {
            "experiment_id": experiment_id,
            "count": len(events),
            "events": [e.to_dict() for e in events],
        }

    @app.get("/api/events", tags=["审计"])
    def recent_events(limit: int = 50) -> dict[str, Any]:
        """最近的操作（跨实验，倒序）——用来回答"刚才谁动了什么"。"""
        events = registry.recent_events(limit=limit)
        return {"count": len(events), "events": [e.to_dict() for e in events]}

    # ---- 数仓绑定 -------------------------------------------------------- #
    @app.get("/api/warehouse/experiments", tags=["数仓"])
    def warehouse_experiments() -> dict[str, Any]:
        """列出数仓里可绑定的实验。没配数仓时返回空列表而不是报错。"""
        if app.state.warehouse_path is None:
            return {"available": False, "warehouse": None, "experiments": []}
        with warehouse_connection() as con:
            try:
                items = list_warehouse_experiments(con)
            except Exception as exc:  # DuckDB 表缺失等情况给出可读信息
                raise HTTPException(400, f"读取数仓失败：{exc}") from exc
        return {
            "available": True,
            "warehouse": str(app.state.warehouse_path),
            "experiments": items,
        }

    @app.post("/api/experiments/{experiment_id}/bind", tags=["数仓"])
    def bind_experiment(
        experiment_id: str, payload: BindIn, request: Request
    ) -> dict[str, Any]:
        actor = actor_of(request)
        try:
            record = registry.bind_warehouse(
                experiment_id,
                payload.warehouse_experiment,
                actor=actor,
                expected_version=expected_version_of(request),
            )
        except RegistryConflict as exc:
            raise _conflict(exc) from exc
        except RegistryError as exc:
            raise HTTPException(404 if "找不到" in str(exc) else 400, str(exc)) from exc
        # 绑定的目标必须真的存在，否则用户要到点"分析"时才发现 —— 那是更晚、更贵的反馈
        if record.warehouse_experiment is not None and app.state.warehouse_path is not None:
            with warehouse_connection() as con:
                known = {e["experiment"] for e in list_warehouse_experiments(con)}
            if record.warehouse_experiment not in known:
                # 回滚也用同一个 actor：这条审计记的是"同一个人绑定失败并回滚"
                registry.bind_warehouse(experiment_id, None, actor=actor)
                raise HTTPException(
                    400,
                    f"数仓里没有实验 {record.warehouse_experiment!r}；"
                    f"可选：{sorted(known)}（已回滚这次绑定）",
                )
        return record.to_dict()

    # ---- 分析 ------------------------------------------------------------ #
    @app.post("/api/experiments/{experiment_id}/analyze", tags=["分析"])
    def analyze(experiment_id: str, payload: AnalyzeIn) -> dict[str, Any]:
        try:
            record = registry.get(experiment_id)
        except RegistryError as exc:
            raise HTTPException(404, str(exc)) from exc

        try:
            if record.warehouse_experiment:
                # 真实链路：读数仓。n_users / seed 在这里没有意义（数据已经存在），
                # 所以不静默忽略它们 —— 那样只会让人以为参数生效了。
                if payload.seed is not None:
                    raise HTTPException(
                        400, "数仓链路的数据是既成的，seed 参数只对合成数据路径有意义"
                    )
                with warehouse_connection() as con:
                    report = analyse_experiment_from_warehouse(
                        record, con, alpha=payload.alpha, n_looks=payload.n_looks
                    )
            else:
                report = analyse_experiment(
                    record,
                    n_users=payload.n_users,
                    alpha=payload.alpha,
                    n_looks=payload.n_looks,
                    seed=payload.seed,
                )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return report.to_dict()

    @app.post("/api/validate/aa", tags=["分析"])
    def validate_aa(payload: AAIn) -> dict[str, Any]:
        return run_aa_validation(
            n_trials=payload.n_trials,
            n_units=payload.n_units,
            alpha=payload.alpha,
        )

    # ---- 实验前定量 ------------------------------------------------------ #
    @app.post("/api/design/power", tags=["分析"])
    def design_power(payload: DesignIn) -> dict[str, Any]:
        """要检出给定的相对提升，需要多少样本；用 CUPED 能省多少。"""
        target = abs(payload.baseline_mean) * payload.relative_mde
        # CUPED 把效应估计的标准误降到 sqrt(1-ρ²) 倍，等价于把"要压到的 SE"
        # 保持不变、而可用的 sd 变小 —— 这里用等效 sd 表达，避免复述公式。
        effective_sd = payload.sd * float(np.sqrt(1.0 - payload.pre_post_correlation**2))
        n_arm_cuped = required_n_per_arm(
            effective_sd,
            target,
            alpha=payload.alpha,
            power=payload.power,
            treatment_ratio=payload.treatment_ratio,
        )
        n_arm_plain = required_n_per_arm(
            payload.sd,
            target,
            alpha=payload.alpha,
            power=payload.power,
            treatment_ratio=payload.treatment_ratio,
        )
        return {
            "target_absolute_effect": target,
            "mde_per_unit_se": mde(1.0, payload.alpha, payload.power),
            "n_per_arm": n_arm_cuped,
            "n_total": n_arm_cuped * 2,
            "n_per_arm_post_only": n_arm_plain,
            "n_total_post_only": n_arm_plain * 2,
            "sample_saving": 1.0 - n_arm_cuped / n_arm_plain if n_arm_plain > 0 else None,
            "note": (
                "n_per_arm 是总样本量的每臂平均值（总量 = 2×）；"
                "post-only 那一列是同一目标下不用 CUPED 的需求"
            ),
        }

    return app


def default_registry_path() -> Path:
    """默认注册表位置：项目内的 ``build/platform/registry.db``。

    用项目内路径而不是系统目录，是为了让整个平台**自包含** ——
    删掉 ``build/`` 就等于重置，不会在用户机器上留下散落的状态。
    """
    return Path(__file__).resolve().parents[3] / "build" / "platform" / "registry.db"


def default_warehouse_path() -> Path:
    """默认数仓位置：``build/warehouse.duckdb``（由 ``run_warehouse.py`` 建）。

    不存在也没关系 —— 平台会退化成"只有合成数据路径"，
    而 ``GET /api/warehouse/experiments`` 会如实报 ``available: false``。
    """
    return Path(__file__).resolve().parents[3] / "build" / "warehouse.duckdb"
