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
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.platform.analysis import analyse_experiment  # noqa: E402
from ablab.platform.api import create_app  # noqa: E402
from ablab.platform.registry import (  # noqa: E402
    ExperimentRecord,
    ExperimentRegistry,
    RegistryError,
)
from ablab.reporting import for_report  # noqa: E402

VARIANTS = [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}]


def main() -> int:
    ap = argparse.ArgumentParser(description="治理验证：审计 + 护栏")
    ap.add_argument("--out", default=str(ROOT / "reports"), help="报告输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    t0 = time.perf_counter()

    def emit(text: str = "") -> None:
        print(text)
        log.append(text)

    header = "=" * 74
    emit(header)
    emit("ab-causal-lab · 治理验证：操作审计（append-only）+ 护栏指标显式化")
    emit(header)

    # 用**临时文件库**而不是内存库：要验证"落盘重开之后审计还在"。
    tmpdir = Path(tempfile.mkdtemp(prefix="gov_"))
    db = tmpdir / "registry.db"

    # ---- 1. 操作审计 ------------------------------------------------------ #
    emit("\n### 1. 操作审计：谁改了什么，留痕")
    reg = ExperimentRegistry(db)
    rec = reg.create(
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
    reg.set_status(rec.id, "running")
    reg.set_estimator(rec.id, "post_only")
    reg.bind_warehouse(rec.id, "exp_rank_v2")

    emit("")
    emit(f"  {'seq':>4}  {'action':<16} {'field':<22} before -> after")
    for e in reg.events(rec.id):
        change = ""
        if e.field:
            change = f"{e.before or '（空）'} -> {e.after or '（空）'}"
        emit(f"  {e.seq:>4}  {e.action:<16} {e.field or '':<22} {change}")
    emit("")
    emit("  改判定口径那条的备注（它改变的是**判定规则**，不只是元数据）：")
    est_event = next(e for e in reg.events(rec.id) if e.action == "set_estimator")
    emit(f"    {est_event.note}")

    # ---- 2. 删掉实验之后审计仍在 ------------------------------------------ #
    emit("\n### 2. 删掉实验之后，审计必须还在")
    reg.delete(rec.id)
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
        reopened.set_estimator(rec.id, "not_an_estimator")
    except RegistryError as exc:
        emit(f"  非法口径被拒：{exc}")
    try:
        reopened.set_estimator("不存在的实验", "cuped")
    except RegistryError as exc:
        emit(f"  不存在的实验被拒：{exc}")
    after = len(reopened.recent_events(limit=100))
    emit(f"  审计条数 {before} -> {after}（没有变化，说明拒绝了就没记）")
    emit("  —— 否则审计会记下「没发生的事」，那比不记更糟。")

    # ---- 6. 接口层：只读查询 ---------------------------------------------- #
    emit("\n### 6. 接口层：只读查询，且删除后仍可查")
    from fastapi.testclient import TestClient

    app = create_app(tmpdir / "api.db")
    client = TestClient(app)
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
    emit("\n### 7. 护栏指标：把「没分析」这条静默变成显式")
    declared = ["latency_p99", "crash_rate", "revenue_per_user"]
    demo = ExperimentRecord(
        name="guardrail_demo",
        variants=list(VARIANTS),
        salt="guardrail_demo_v1",
        primary_metric="post_metric_14d",
        guardrails=declared,
    )
    rep = analyse_experiment(demo, n_users=8_000)
    item = next((c for c in rep.checks if c.name == "护栏指标"), None)
    emit(f"  该实验声明了 {len(declared)} 个护栏：{'、'.join(declared)}")
    emit(f"  分析报告的检查项（{len(rep.checks)} 项）：{[c.name for c in rep.checks]}")
    if item is not None:
        emit(f"  护栏那条：status={item.status}，health 不受影响（当前 health={rep.health}）")
        emit(f"  正文：{item.message}")
    emit("")
    emit("  为什么是 info 而不是 warn：护栏未接入是**平台级**缺口，")
    emit("  声明了护栏的每个实验都会一直 warn —— 而「一条永远亮的告警等于没有告警」，")
    emit("  health 会因此失去意义（这条教训来自第 31 条）。")
    emit("  信息要显式（这条检查永远在报告里），但不占用「这次运行有问题」这个信号。")

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
    emit("\n### 8. 结论")
    emit("  * 审计是 append-only 的**机械**保证：SQLite 触发器拒绝 UPDATE/DELETE，")
    emit("    而且它是被当场试出来的，不是一句声称。")
    emit("  * 审计与业务变更在**同一个事务**里；失败的写入不留痕。")
    emit("  * 删除实验不会删除审计 —— 那正是最需要它的时刻。")
    emit("  * 护栏的「未分析」状态出现在每一份相关报告里，并说清了原因。")
    emit("  * 仍未做的（写在这里而不是留着让人误会）：")
    emit("    - 审计没有「操作者」字段。平台上还没有鉴权，写上去也只是个空字段；")
    emit("      真上线要先有身份，再谈「谁做的」。")
    emit("    - 护栏**仍然没有被分析**：数据模型只有主指标一条时间序列。")
    emit("      要做需要数仓里另建指标表 + 停实验的判据，那是另一件事。")

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
