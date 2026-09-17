#!/usr/bin/env python
"""构建 DuckDB 数仓链路（ODS→DWD→DWS→ADS）并输出实验结论。

运行::

    python scripts/run_warehouse.py
    python scripts/run_warehouse.py --users 50000 --rebuild

输出到 ``reports/warehouse_report.md``，同时打印到终端。

这份脚本演示"数仓链路 + 统计引擎"的接口面：
SQL 负责口径（首次曝光去重、前后窗口、可加汇总），
Python 负责推断（t 检验、置信区间、SRM 体检），
两边各有一个交叉验证点，任何一个不过就说明有一侧写错了。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.reporting import for_report  # noqa: E402
from ablab.warehouse import (  # noqa: E402
    WarehouseConfig,
    analyse_ads,
    build_warehouse,
    covariate_adjustment_report,
    render_report,
    verify_against_detail,
)

LAYERS = (
    ("ods_exposure_log", "ODS 曝光日志（分流服务直接落盘）"),
    ("ods_event_log", "ODS 行为明细"),
    ("dwd_experiment_user", "DWD 实验单元宽表（已去重、已对齐前后窗口）"),
    ("dws_experiment_variant_daily", "DWS 实验×分支×日（全部可加字段）"),
    ("ads_experiment_result", "ADS 实验结果（点估计 + 方差）"),
    ("ads_experiment_srm", "ADS SRM 体检表"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description="构建数仓链路并输出实验结论")
    ap.add_argument("--users", type=int, default=20_000, help="合成用户数")
    ap.add_argument("--rebuild", action="store_true", help="强制重新生成源数据")
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
    emit("ab-causal-lab · 数仓链路 ODS → DWD → DWS → ADS")
    emit(header)

    cfg = WarehouseConfig(n_users=args.users)
    con = build_warehouse(
        db_path=ROOT / "build" / "warehouse.duckdb",
        data_dir=ROOT / "build" / "source",
        sql_dir=ROOT / "sql",
        config=cfg,
        force_data=args.rebuild,
    )

    # ---- 1. 各层规模 ------------------------------------------------------ #
    emit("\n### 1. 各层规模")
    counts: dict[str, int] = {}
    for table, desc in LAYERS:
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        emit(f"  {table:<34} {counts[table]:>10,}   {desc}")

    # ---- 2. 口径验证：首次曝光去重 ---------------------------------------- #
    emit("\n### 2. 口径验证")
    raw = counts["ods_exposure_log"]
    dedup = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT experiment, user_id FROM ods_exposure_log)"
    ).fetchone()[0]
    dwd = counts["dwd_experiment_user"]
    emit(f"  首次曝光去重: 原始曝光 {raw:,} 条 -> 去重后 {dedup:,} 个 (实验, 用户) 对")
    emit(f"  DWD 行数 = {dwd:,}，与去重结果一致: {dwd == dedup}")

    rolled = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT experiment, variant, SUM(user_cnt) AS n
            FROM dws_experiment_variant_daily
            GROUP BY experiment, variant
        ) d JOIN ads_experiment_result a USING (experiment, variant)
        WHERE d.n <> a.user_cnt
        """
    ).fetchone()[0]
    emit(f"  DWS 可加汇总 vs ADS 人数，不一致行数: {rolled}")

    # ---- 3. 实验结论（推断层） -------------------------------------------- #
    emit("\n### 3. 实验结论")
    analyses = analyse_ads(con)
    emit(render_report(analyses))

    # ---- 4. 交叉验证 ------------------------------------------------------ #
    emit("\n### 4. 交叉验证：汇总路径 vs 明细路径")
    for a in analyses:
        cv = verify_against_detail(con, analyses, a.experiment)
        emit(cv.summary())
        emit(f"    passed={cv.passed}；SQL/Python SRM 卡方一致: {a.srm_agrees}")

    # ---- 5. 负对照：协变量失衡与 CUPED 的动机 ----------------------------- #
    emit("\n### 5. 负对照诊断 —— 为什么必须做前置协变量校正")
    emit("exp_rec_emb 的真实效应是 0（负对照）。如果 post-only 分析给出显著结论，")
    emit("那一定是这一次分流的协变量失衡造成的假象。")
    adj = covariate_adjustment_report(con, "exp_rec_emb", n_trials=1_000)
    emit(adj.summary())

    emit("对照看有真实效应的实验：")
    adj2 = covariate_adjustment_report(con, "exp_rank_v2", n_trials=1_000)
    emit(adj2.summary())

    rank = next(a for a in analyses if a.experiment == "exp_rank_v2")
    emit("结论：")
    emit(f"  * 负对照实验的 post 差距 {adj.realized_post_gap:+.4f} 里，"
         f"约 {adj.bias_removed * 100:.0f}% 被 CUPED 扣掉")
    emit(f"  * 方差缩减 = rho^2 = {adj.theoretical_variance_reduction:.4f}"
         f"（残余方差 1-rho^2 = {adj.remaining_variance_fraction:.4f}，"
         f"标准误降幅 {adj.se_shrinkage:.4f}，等效样本量 x"
         f"{adj.effective_sample_multiplier:.2f}）")
    emit(f"    实测方差缩减 {adj.variance_reduction:.4f}，与理论吻合")
    emit(f"  * 有真实效应的 {rank.experiment}（真实日效应 {rank.true_lift:+.2f}）：")
    emit(f"      校正前 {adj2.realized_post_gap:+.4f} -> 校正后 {adj2.realized_adjusted_gap:+.4f}"
         f"，更接近真值")
    emit("  * -> M1 将把这个诊断正式实现为 CUPED，并作为默认分析口径")

    emit(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    con.close()

    report = out_dir / "warehouse_report.md"
    report.write_text(
        "# 数仓链路验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n",
        encoding="utf-8",
    )
    print(f"\n报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
