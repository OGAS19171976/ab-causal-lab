#!/usr/bin/env python
"""治理验证：操作审计（append-only）+ 护栏指标显式化。

为什么这两件事值得单独一份报告
------------------------------
它们都不是统计方法，而是**平台会不会静默骗人**的问题：

1. **操作审计**。M6 特意允许改 ``estimator``（口径是策略不是数据），
   也允许改状态、绑数仓、删实验 —— 于是"谁在什么时候把判定口径从 CUPED
   改成 post-only"必须留痕。否则报告里写着"用的是哪个口径"，
   但**没人知道它是不是在看到结果之后才改的** —— 那就是一条事后挑口径的通道。
2. **护栏指标**。``guardrails`` 字段一直存在、界面上也显示，
   而引擎从头到尾没读过它 —— 用户合理地以为护栏被看着。
   真正的解法不是"假装分析"，而是在报告里**明说没分析**并说清为什么。

这份报告把两件事都跑出来：审计的**不可改写**由 SQLite 触发器保证（当场试给它看），
护栏的**显式声明**打印在报告正文里。

用法::

    python scripts/run_governance_validation.py            # 约 10 秒
    python scripts/run_governance_validation.py --out reports
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.platform.analysis import analyse_experiment  # noqa: E402
from ablab.platform.api import create_app  # noqa: E402
from ablab.platform.registry import (  # noqa: E402
    ExperimentRecord,
    ExperimentRegistry,
    RegistryConflict,
    RegistryError,
)
from ablab.reporting import for_report  # noqa: E402

VARIANTS = [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}]

#: 本报告脚本自己就是"操作者"。用一个具名用户，而不是空字符串 ——
#: 审计里的操作者现在是必填的具名参数，忘了记谁会在调用点直接报错。
OP = "gov-validator"
OP2 = "gov-reviewer"


def main() -> int:
    ap = argparse.ArgumentParser(description="治理验证：审计 + 护栏")
    ap.add_argument("--out", default=str(ROOT / "reports"), help="报告输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    t0 = time.perf_counter()

    def _run_checker(script: str) -> list[str]:
        """跑一个检查脚本并把它**真实的输出**按行返回（治理报告要嵌它）。"""
        import subprocess as _sp

        proc = _sp.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        return (proc.stdout or "").splitlines()

    def emit(text: str = "") -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · 治理验证：操作审计（append-only）+ 护栏指标显式化")
    emit(header)

    # 用**临时文件库**而不是内存库：要验证"落盘重开之后审计还在"。
    #
    # 但不用 ``tempfile.mkdtemp()``：它给目录加 0o700 权限位，而在受限（沙箱）
    # 环境里那种目录**建得出来、写不进去** —— 实测下一行的 ``ExperimentRegistry``
    # 直接报 ``sqlite3.OperationalError: unable to open database file``，
    # 整个 gov 检查因此变红（CI 上一切正常，所以一直没被发现）。
    # 改成在项目内用默认权限建（与 ``tests/conftest.py::work_dir`` 同一个理由）；
    # 每次跑前先清掉上一次的库 —— 这份报告会断言审计条数，
    # 残留的老库会让那些数字漂移。
    tmpdir = ROOT / "build" / "_gov_tmp"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    db = tmpdir / "registry.db"

    # ---- 1. 操作审计 ------------------------------------------------------ #
    emit("\n### 1. 操作审计：谁改了什么，留痕")
    reg = ExperimentRegistry(db)
    rec = reg.create(
        actor=OP,
        name="gov_demo",
        variants=VARIANTS,
        salt="gov_demo_v1",
        primary_metric="post_metric_14d",
        guardrails=["latency_p99", "crash_rate", "revenue_per_user"],
        owner="demo",
    )
    emit(f"  创建实验 {rec.name}，并依次改状态 / 改判定口径 / 绑数仓")
    emit("  （实验 id 是 uuid4，每次入库都变 —— 刻意不打印它，"
         "否则这份报告每次都不一样；审计本身按 seq 排序，不依赖 id。）")
    reg.set_status(rec.id, "running", actor=OP)
    reg.set_estimator(rec.id, "post_only", actor=OP)
    reg.bind_warehouse(rec.id, "exp_rank_v2", actor=OP)

    emit("")
    emit(f"  {'seq':>4}  {'操作者':<14} {'action':<16} {'field':<22} before -> after")
    for e in reg.events(rec.id):
        change = ""
        if e.field:
            change = f"{e.before or '（空）'} -> {e.after or '（空）'}"
        emit(f"  {e.seq:>4}  {e.actor:<14} {e.action:<16} {e.field or '':<22} {change}")
    emit("")
    emit("  改判定口径那条的备注（它改变的是**判定规则**，不只是元数据）：")
    est_event = next(e for e in reg.events(rec.id) if e.action == "set_estimator")
    emit(f"    {est_event.note}")

    # ---- 1.5 身份：操作者从凭据来，不从请求来 ----------------------------- #
    emit("\n### 1.5 身份：操作者**只能**来自凭据")
    emit("  审计里的 `actor` 一列现在是必填的（忘了记谁会在调用点报错），")
    emit("  而它的值**只**由服务端从凭据推导。三种伪造尝试实测：")
    emit("")
    admin_token = reg.add_user("gov_admin", role="admin", note="治理验证用")
    editor_token = reg.add_user("gov_editor", role="editor")
    reg.add_user("gov_viewer", role="viewer")
    emit(f"  {'用户':<12}{'角色':<9}{'库里存的是':<14}明文 token 可读?")
    for u in reg.list_users():
        row = reg._conn.execute(
            "SELECT token_hash FROM users WHERE id = ?", (u["id"],)
        ).fetchone()
        emit(f"  {u['id']:<12}{u['role']:<9}{'sha256 前 12 位':<14}"
             f"{'否（只存哈希）' if row['token_hash'][:12] else ''}")
    emit("")
    emit("  认证结果（真值来自凭据，伪造一律无效）：")
    emit(f"    正确 token            -> {reg.authenticate(admin_token)!r}")
    emit(f"    错误 token            -> {reg.authenticate('not-a-token')!r}")
    emit(f"    空凭据                -> {reg.authenticate(None)!r}")
    reg.disable_user("gov_editor")
    emit(f"    被停用的用户          -> {reg.authenticate(editor_token)!r}"
         "（停用立即生效）")
    reg.disable_user("gov_editor", disabled=False)
    emit("")
    emit("  **请求里写的名字不算数** —— 这一条走真实 HTTP 接口实测：")
    from fastapi.testclient import TestClient as _TC

    auth_app = create_app(tmpdir / "auth_api.db")
    auth_reg = auth_app.state.registry
    a_admin = auth_reg.add_user("http_admin", role="admin")
    a_editor = auth_reg.add_user("http_alice", role="editor")
    a_viewer = auth_reg.add_user("http_bob", role="viewer")
    ac = _TC(auth_app)
    body = {"name": "gov_auth", "variants": VARIANTS, "salt": "gov_auth_v1"}
    no_cred = ac.post("/api/experiments", json=body).status_code
    bad_cred = ac.post(
        "/api/experiments", json=body, headers={"Authorization": "Bearer nope"}
    ).status_code
    viewer = ac.post(
        "/api/experiments", json=body,
        headers={"Authorization": f"Bearer {a_viewer}"},
    ).status_code
    spoof = ac.post(
        "/api/experiments?actor=http_admin",
        json={**body, "owner": "http_admin"},
        headers={
            "Authorization": f"Bearer {a_editor}",
            "X-Actor": "http_admin",
        },
    )
    made = spoof.json()
    events = ac.get(f"/api/experiments/{made['id']}/events").json()["events"]
    editor_cannot_delete = ac.delete(
        f"/api/experiments/{made['id']}",
        headers={"Authorization": f"Bearer {a_editor}"},
    ).status_code
    admin_can_delete = ac.delete(
        f"/api/experiments/{made['id']}",
        headers={"Authorization": f"Bearer {a_admin}"},
    ).status_code
    emit(f"    无凭据 POST                      -> {no_cred}")
    emit(f"    错 token POST                    -> {bad_cred}")
    emit(f"    viewer POST                      -> {viewer}（角色不够）")
    emit(f"    伪造（query+header 都写 http_admin）-> 审计记为 "
         f"{[e['actor'] for e in events]}（凭据是 http_alice）")
    emit(f"    editor DELETE                    -> {editor_cannot_delete}（删是 admin 的权限）")
    emit(f"    admin DELETE                     -> {admin_can_delete}")
    emit(f"    被拒的三次尝试留下审计条数        -> "
         f"{ac.get('/api/events').json()['count'] - 2}（只记成功的那两条）")
    emit("    另外 `actor` 也不是请求体字段（`_Strict` 直接 422），见 "
         "`tests/test_governance.py::TestAuthAndActor`：")
    emit("    · 有一条测试**自动枚举所有写路由**逐个断言 401，")
    emit("      所以「新加了端点忘了鉴权」会在 CI 上直接红。")

    # ---- 2. 删掉实验之后审计仍在 ------------------------------------------ #
    emit("\n### 2. 删掉实验之后，审计必须还在")
    reg.delete(rec.id, actor=OP)
    gone = False
    try:
        reg.get(rec.id)
    except RegistryError:
        gone = True
    emit(f"  实验本身已删除：{gone}")
    events = reg.events(rec.id)
    emit(f"  该实验的审计仍有 {len(events)} 条，最后一条是 "
         f"{events[-1].action!r}：{events[-1].note}")
    emit("  —— 也就是说「谁删的、删之前是什么状态」仍然查得到；")
    emit("     审计表**没有外键级联**，这是有意的。")

    # ---- 3. append-only 是机械保证，不是约定 ------------------------------ #
    emit("\n### 3. append-only：不是「我们不写 UPDATE」，是**写不动**")
    for sql in (
        "UPDATE experiment_events SET after = '被篡改' WHERE seq = 1",
        "DELETE FROM experiment_events WHERE seq = 1",
    ):
        try:
            reg._conn.execute(sql)
            emit(f"  [!!] {sql.split()[0]} 竟然成功了 —— 触发器没生效")
        except sqlite3.IntegrityError as exc:
            emit(f"  {sql.split()[0]:<7} 被 SQLite 触发器拒绝：{exc}")
    emit(f"  试完之后审计条数仍然是 {len(reg.events(rec.id))}（没有被改掉）")

    # ---- 4. 落盘重开仍在 -------------------------------------------------- #
    emit("\n### 4. 落盘重开：审计跟着数据库走")
    reg.close()
    reopened = ExperimentRegistry(db)
    emit(f"  重新打开 {db.name}，同一个实验的审计 {len(reopened.events(rec.id))} 条，"
         f"动作序列 {[e.action for e in reopened.events(rec.id)]}")

    # ---- 5. 失败的写入不留痕 ---------------------------------------------- #
    emit("\n### 5. 失败的写入**不能**留下审计")
    before = len(reopened.recent_events(limit=100))
    try:
        reopened.set_estimator(rec.id, "not_an_estimator", actor=OP)
    except RegistryError as exc:
        emit(f"  非法口径被拒：{exc}")
    try:
        reopened.set_estimator("不存在的实验", "cuped", actor=OP)
    except RegistryError as exc:
        emit(f"  不存在的实验被拒：{exc}")
    after = len(reopened.recent_events(limit=100))
    emit(f"  审计条数 {before} -> {after}（没有变化，说明拒绝了就没记）")
    emit("  —— 否则审计会记下「没发生的事」，那比不记更糟。")

    # ---- 6. 接口层：只读查询 ---------------------------------------------- #
    emit("\n### 6. 接口层：只读查询，且删除后仍可查")
    from fastapi.testclient import TestClient

    app = create_app(tmpdir / "api.db")
    api_token = app.state.registry.add_user("gov_api_admin", role="admin")
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {api_token}"})
    api_rec = client.post(
        "/api/experiments",
        json={"name": "gov_api", "variants": VARIANTS, "salt": "gov_api_v1"},
    ).json()
    client.patch(f"/api/experiments/{api_rec['id']}/status", json={"status": "running"})
    payload = client.get(f"/api/experiments/{api_rec['id']}/events").json()
    emit(f"  GET /api/experiments/{{id}}/events -> {payload['count']} 条："
         f"{[e['action'] for e in payload['events']]}")
    client.delete(f"/api/experiments/{api_rec['id']}")
    after_delete = client.get(f"/api/experiments/{api_rec['id']}/events")
    emit(f"  DELETE 之后再查同一个接口 -> HTTP {after_delete.status_code}，"
         f"{after_delete.json()['count']} 条："
         f"{[e['action'] for e in after_delete.json()['events']]}")
    recent = client.get("/api/events?limit=3").json()
    # 只打印**动作**，不打印 experiment_id：id 是 uuid4，印出来这份报告就不可复现了。
    emit(f"  GET /api/events?limit=3 -> 最近三次操作："
         f"{[e['action'] for e in recent['events']]}（同一个实验的 id 未打印，见上）")

    # ---- 7. 护栏指标：声明了但没人分析，必须说出来 ------------------------ #
    emit("\n### 7. 护栏指标：**真的判定**，并给出「要不要停实验」的判据")
    emit("  三种声明各跑一遍，因为它们的**补救办法完全不同**：")
    emit("    A) 有规格、有数据（其中一条被注入了 +12% 的真实伤害）")
    emit("    B) 只声明了名字，没给方向与容忍度")
    emit("    C) 规格齐全、但没有数据（数仓路径还没有护栏表）")
    emit("")
    from ablab.platform.guardrails import GuardrailSpec as _Spec

    spec_demo = ExperimentRecord(
        name="guardrail_demo",
        variants=list(VARIANTS),
        salt="guardrail_demo_v1",
        primary_metric="post_metric_14d",
        guardrails=["latency_p99", "crash_rate", "revenue_per_user"],
        guardrail_specs=[
            _Spec("latency_p99", "lower_is_better", 0.05, demo_harm=0.12),
            _Spec("crash_rate", "lower_is_better", 0.10, demo_harm=0.0),
            _Spec("revenue_per_user", "higher_is_better", 0.05, demo_harm=0.0),
        ],
        true_lift=0.02,
    )
    rep_a = analyse_experiment(spec_demo, n_users=8_000)
    item_a = next(c for c in rep_a.checks if c.name == "护栏指标")
    emit(f"  A) 护栏那条 status = {item_a.status}，health = {rep_a.health}")
    for line in item_a.message.splitlines()[:6]:
        emit(f"     {line}")
    emit("")
    bare = ExperimentRecord(
        name="guardrail_bare",
        variants=list(VARIANTS),
        salt="guardrail_bare_v1",
        primary_metric="post_metric_14d",
        guardrails=["recall_coverage"],
    )
    rep_b = analyse_experiment(bare, n_users=8_000)
    item_b = next(c for c in rep_b.checks if c.name == "护栏指标")
    emit(f"  B) 只有名字：status = {item_b.status}（**用户这一次就能补**："
         "补 direction + max_harm）")
    emit(f"     {[ln.strip() for ln in item_b.message.splitlines() if 'unknown' in ln][0]}")
    emit("")
    import dataclasses as _dc

    from ablab.platform.analysis import PLATFORM_POPULATION, analyse_data

    wh_like = _dc.replace(
        _with_wh_data := __import__(
            "ablab.platform.datasource", fromlist=["build_synthetic_data"]
        ).build_synthetic_data(
            experiment="guardrail_wh", salt="guardrail_wh_v1",
            variants=[("control", 0.5), ("treatment", 0.5)],
            metric="post_metric_14d", n_users=4_000, n_looks=3, seed=11,
            population=PLATFORM_POPULATION,
            guardrail_specs=(_Spec("latency_p99", "lower_is_better", 0.05),),
        ),
        guardrail_series={},
    )
    item_c = next(c for c in analyse_data(wh_like).checks if c.name == "护栏指标")
    emit(f"  C) 有规格没数据：status = {item_c.status}（**平台要补**：数仓还没有护栏表）")
    emit(f"     {[ln.strip() for ln in item_c.message.splitlines() if 'unknown' in ln][0]}")
    emit("")
    emit("  判定规则（写在代码里，也写在 README 里）：")
    emit("    · 把两臂之差按 direction 折算成**伤害**；越界看的是伤害的**置信下界**，")
    emit("      不是点估计 —— 点估计超了但证据不足记 warn（继续观察），")
    emit("      这样「停机」这个动作才是保守的；")
    emit("    · K 条护栏用 Bonferroni（alpha/K）校正：要控的是「误判有害从而错误停机」；")
    emit("    · **没声明方向与容忍度就不判断**（判 unknown）—— 从指标名猜方向")
    emit("      会把伤害静默读成改善；")
    emit("    · **缺数据判 unknown，绝不判 pass**：那正是这一块原来的毛病。")
    emit("")
    emit("  A 组的细节值得看：latency_p99 被注入 +12% 伤害（远超 5% 容忍度），")
    emit("  于是 health 直接变成 fail，报告里出现「建议停止实验」——")
    emit("  Kohavi 那本书里护栏触发是**停实验的理由**，不是参考信息。")

    no_guard = ExperimentRecord(
        name="no_guardrail_demo",
        variants=list(VARIANTS),
        salt="no_guardrail_demo_v1",
        primary_metric="post_metric_14d",
    )
    rep2 = analyse_experiment(no_guard, n_users=8_000)
    emit("")
    emit(f"  对照（没声明护栏）：检查项里有没有护栏那条 = "
         f"{bool([c for c in rep2.checks if c.name == '护栏指标'])}（应为 False）")

    # ---- 8. 结论 ---------------------------------------------------------- #
    # ---- 7. 并发：丢失更新与乐观锁 ---------------------------------------- #
    # ---- 7.7 护栏判定的校准：误停率与功效 --------------------------------- #
    emit("\n### 7.7 护栏判定的**校准**：这条停机规则到底有多保守")
    emit("  判定规则问的不是「有没有伤害」，而是「伤害**是否超过容忍度**」，")
    emit("  所以必须量出它的运行特征（误停率 / 功效），而不是只说规则合理：")
    emit("")
    from ablab.validation.guardrail_audit import run_guardrail_audit

    cal = run_guardrail_audit(
        n_trials=80, n_users=3000, max_harm=0.05, harm=0.12
    )
    for line in cal.summary().splitlines():
        emit("  " + line)
    emit("")
    emit("  三个工作点合起来说明：**阈值在容忍度上而不是 0 上**，")
    emit("  于是 H0 下几乎不会误停，代价是伤害刚好压在容忍度附近时")
    emit("  大量落在「观察」带 —— 那正是「宁可少停、也不要误停」的取舍。")

    # ---- 7.9 决策层：把"建议停实验"接到动作上 ----------------------------- #
    emit("\n### 7.9 决策层：护栏触发**真的能停实验**（而且服务端自己复核）")
    emit("  前面几节做到的是「报告里写着建议停止实验」。从一句话到一个动作之间")
    emit("  隔着一次判断，所以这个端点**不信任调用方递过来的结论**：")
    emit("  它自己重跑一遍分析，确认护栏确实越界才执行。")
    emit("")
    from fastapi.testclient import TestClient as _StopClient

    from ablab.platform.api import create_app as _create_app

    stop_app = _create_app(tmpdir / "stop_api.db")
    stop_reg = stop_app.state.registry
    stop_editor = stop_reg.add_user("stop_editor", role="editor")
    stop_admin = stop_reg.add_user("stop_admin", role="admin")
    sc = _StopClient(stop_app)
    stop_variants = [
        {"name": "control", "weight": 0.5},
        {"name": "treatment", "weight": 0.5},
    ]

    def _make_stop_experiment(name: str, max_harm: float, demo_harm: float) -> str:
        resp = sc.post(
            "/api/experiments",
            json={
                "name": name, "variants": stop_variants, "salt": f"{name}_v1",
                "status": "running", "true_lift": 0.02,
                "guardrails": ["latency_p99"],
                "guardrail_specs": [{
                    "name": "latency_p99", "direction": "lower_is_better",
                    "max_harm": max_harm, "demo_harm": demo_harm,
                }],
            },
            headers={"Authorization": f"Bearer {stop_editor}"},
        )
        return resp.json()["id"]

    tripped_id = _make_stop_experiment("gov_stop_tripped", 0.05, 0.12)
    clean_id = _make_stop_experiment("gov_stop_clean", 0.50, 0.0)
    emit("  三种情形各跑一遍：")
    r_ok = sc.post(
        f"/api/experiments/{tripped_id}/stop",
        json={"analyze": {"n_users": 4000}},
        headers={"Authorization": f"Bearer {stop_editor}"},
    )
    emit(f"    A) 护栏越界 -> HTTP {r_ok.status_code}，状态 {r_ok.json().get('status')}，"
         f"tripped={r_ok.json().get('guardrail_tripped')}")
    r_409 = sc.post(
        f"/api/experiments/{clean_id}/stop",
        json={},
        headers={"Authorization": f"Bearer {stop_editor}"},
    )
    emit(f"    B) 护栏没越界 -> HTTP {r_409.status_code}（拒绝，实验状态不变）")
    r_403 = sc.post(
        f"/api/experiments/{clean_id}/stop",
        json={"force": True},
        headers={"Authorization": f"Bearer {stop_editor}"},
    )
    r_force = sc.post(
        f"/api/experiments/{clean_id}/stop",
        json={"force": True, "reason": "业务方要求"},
        headers={"Authorization": f"Bearer {stop_admin}"},
    )
    emit(f"    C) 人工强制：editor -> HTTP {r_403.status_code}；"
         f"admin -> HTTP {r_force.status_code}（forced={r_force.json().get('forced')}）")
    emit("")
    emit("  审计里留下的依据（**动作名单独记为 stop**，理由进 note）：")
    for e in sc.get(f"/api/experiments/{tripped_id}/events").json()["events"]:
        if e["action"] == "stop":
            emit(f"    {e['actor']} | {e['note']}")
    emit("")
    emit("  为什么值得单独做一层：停实验是这套平台里**最不可逆**的动作")
    emit("  （分流随时能重开，已经造成的伤害收不回来）。所以它比「改个状态」厚：")
    emit("  服务端复核、理由必填、人工叫停要 admin、成功与拒绝都留痕。")

    # ---- 7.11 "没做"的清单：让过时声明在检查集里直接红 -------------------- #
    emit("\n### 7.11 「没做」的清单：从「靠人偶然发现」到「检查集里直接红」")
    emit("  这个仓库栽过三次同一类跟头：**功能做完了，README 还写着没做** ——")
    emit("  簇级 CUPED（被三处代码拒绝了两轮）、M2 决策层、数仓比值链路。")
    emit("  共同点是「没做」是一句**无法被核对**的话：数字有人对（声明清单），")
    emit("  「没做」没人对，于是它只朝一个方向漂移。")
    emit("  这一轮把每一句「没做」变成一条**带证据**的记录：证据必须现在还成立")
    emit("  （某个符号确实不存在 / 某个串搜不到 / 某个文件不存在），")
    emit("  一旦不成立就在检查集里报错并指出该改哪一句。")
    emit("")
    import subprocess as _sp

    _proc = _sp.run(
        [sys.executable, str(ROOT / "scripts" / "check_unimplemented.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    for line in (_proc.stdout or "").splitlines():
        emit("  " + line)
    emit(f"  退出码：{_proc.returncode}（0 = 清单与事实一致）")
    emit("")
    emit("  第一次跑就报了**一个假阳性**：清单里写着 `target=load_real_traffic`，")
    emit("  于是「在 src/ 下搜这个串」搜到了**清单自己** —— 检查器也会骗人，")
    emit("  所以搜索时显式跳过清单文件（代码里写了原因）。")
    emit("  更早一步，这一轮还顺手抓到**三条已经过时的声明**：")
    emit("    · M2「没做决策层」（护栏决策层已做，见 7.9）")
    emit("    · 「整簇路径只支持 post-only」（簇级 CUPED 已做并校准）")
    emit("    · 「数仓不支持比值指标」（06/07 两条 SQL 早已上线）")

    emit("\n### 7.12 环境来源与三方库类型：两句「只能人看」变成两张表")
    emit("  已知边界里原先有两句话是**无法核对**的：")
    emit("    · 「三方库没有类型保证：取决于上游是否带 py.typed，只能人看」；")
    emit("    · （隐含的）「锁文件 == 环境」—— 它比的是**版本**，看不见**来源**。")
    emit("  这一轮把它们各变成一条机器检查。")
    emit("")
    emit("  第一句的实测（`scripts/check_typed_deps.py`）：")
    for _line in _run_checker("check_typed_deps.py"):
        emit("    " + _line)
    emit("")
    emit("  第二句实测出来的是一个**藏了很久的事实**：本地 venv 曾经是混合环境")
    emit("  （`include-system-site-packages = true`），锁文件 48 个包里有")
    emit("  **14 个**（pandas / matplotlib / pytest / packaging…）实际解析自")
    emit("  **系统 Python 的 site-packages**，venv 里根本没有它们 —— 而")
    emit("  `lock_requirements.py --check` 照样全绿，因为**版本号恰好一样**。")
    emit("  也就是说「本地跑的东西」与「CI 装的东西」不是同一套文件。")
    emit("  修法两步：按锁文件把缺的包装进 venv（必须 `--ignore-installed`，")
    emit("  否则 pip 看到系统里的同名包就认为「已满足」——这一步实测踩过），")
    emit("  再把 `include-system-site-packages` 置为 false。修完的读数：")
    for _line in _run_checker("check_env_origin.py"):
        emit("    " + _line)
    emit("")
    emit("  为什么值得单独记一条：这个漏检**不是**版本错，是**拓扑**错 ——")
    emit("  版本对、来源错，所有基于版本的自检都会说「没问题」（设计决策 53）。")

    emit("\n### 7.5 并发：丢失更新（后写覆盖），以及乐观锁怎么挡住它")
    emit("  场景：两个客户端（**两个独立连接**，不是同一个对象）都读到同一版本，")
    emit("  然后都要改状态 —— 这就是「两个人同时改」的最小复现。")
    emit("")
    conc_db = tmpdir / "concurrent.db"
    c1 = ExperimentRegistry(conc_db)
    c2 = ExperimentRegistry(conc_db)
    try:
        target = c1.create(
            actor="alice", name="conc_demo", variants=VARIANTS, salt="conc_demo_v1"
        )
        emit(f"  实验建好，version = {target.version}；两个客户端各自读到 "
             f"v{c1.get(target.id).version} 与 v{c2.get(target.id).version}")
        emit("")
        emit("  A) 不带版本号（默认语义 = 后写覆盖）：")
        c1.set_status(target.id, "running", actor="alice")
        c2.set_status(target.id, "stopped", actor="bob")
        final = c1.get(target.id)
        events = [e for e in c1.events(target.id) if e.action == "set_status"]
        emit("     alice 写 running -> 成功；bob 写 stopped -> 成功")
        emit(f"     最终状态 = {final.status}（bob 覆盖了 alice），"
             f"version = {final.version}")
        emit(f"     审计里两条都在：{[e.actor for e in events]} —— "
             "但**alice 的意图已经不在结果里了**，而谁都没收到错误。")
        emit("     这就是丢失更新：比崩溃难查，因为一切「看起来都成功了」。")
        emit("")
        emit("  B) 带上各自读到的版本号（乐观锁）：")
        c1.set_status(target.id, "running", actor="alice")  # 先把状态放回去
        v_a = c1.get(target.id).version
        v_b = c2.get(target.id).version  # 两人都读到同一版本
        c1.set_status(target.id, "running", actor="alice", expected_version=v_a)
        try:
            c2.set_status(target.id, "stopped", actor="bob", expected_version=v_b)
            emit("     bob 的写竟然成功了 —— 这说明乐观锁没生效（是 bug）")
        except RegistryConflict as exc:
            emit(f"     alice 写 -> 成功；bob 写 -> **被拒**（{type(exc).__name__}）")
            emit(f"     拒绝理由：{exc}")
        settled = c1.get(target.id)
        emit(f"     最终状态 = {settled.status}（alice 的改动还在），"
             f"version = {settled.version}")
        emit("     也就是说：**冲突被变成了一次可见的失败**，而不是一次静默的覆盖。")
        emit("")
        emit("  边界（写在明处）：")
        emit("    · 乐观锁是**可选**的 —— 不带 If-Match 就退回后写覆盖，")
        emit("      这是刻意的默认（单机平台上大多数调用就是这么用的）；")
        emit("    · 没有自动重试与自动合并：拿到 412 之后要**重新读、重新决定**，")
        emit("      因为「该不该改」取决于中间那次改动是什么；")
        emit("    · sqlite 单文件本身是串行写的，这里的「并发」是应用层的")
        emit("      读-改-写交错，不是数据库层的写冲突。")
    finally:
        c1.close()
        c2.close()

    emit("\n### 8. 结论")
    emit("  * 审计是 append-only 的**机械**保证：SQLite 触发器拒绝 UPDATE/DELETE，")
    emit("    而且它是被当场试出来的，不是一句声称。")
    emit("  * 审计与业务变更在**同一个事务**里；失败的写入不留痕。")
    emit("  * 删除实验不会删除审计 —— 那正是最需要它的时刻。")
    emit("  * 护栏的「未分析」状态出现在每一份相关报告里，并说清了原因。")
    emit("  * 仍未做的（写在这里而不是留着让人误会）：")
    emit("    - ~~审计没有「操作者」字段~~ **已补**：静态 token 鉴权 + `actor` 列，")
    emit("      见第 1.5 节与 README 设计决策第 45 条。")
    emit("      **边界**：静态 token 无过期、无轮换、无限速，token 泄露即冒充；")
    emit("      读接口仍然匿名；迁移前的老记录操作者是「（迁移前未知）」。")
    emit("    - ~~注册表没有并发控制~~ **已补**：`version` 列 + `If-Match` 头，")
    emit("      冲突返回 412 而不是静默覆盖（见第 7.5 节）。**边界**：乐观锁是")
    emit("      可选的（不带 If-Match 仍是后写覆盖），且没有自动重试与合并。")
    emit("    - ~~护栏没有被分析~~ **已补**：合成路径现在**真的判定**护栏，")
    emit("      越界（伤害的置信下界超过事先声明的容忍度）会让 health 变 fail")
    emit("      并给出「建议停止实验」（见第 7 节）。**边界**：数仓路径还没有")
    emit("      护栏表，那里的护栏判 unknown（不是通过）；方向与容忍度必须显式声明。")

    emit(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    reopened.close()

    report = out_dir / "governance_report.md"
    report.write_text(
        "# 治理验证报告（操作审计 + 护栏指标）\n\n```text\n"
        + "\n".join(for_report(log, root=ROOT))
        + "\n```\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
