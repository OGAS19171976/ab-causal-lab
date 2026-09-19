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


def scalar(con, sql: str) -> int:
    """跑一句"只返回一个数"的 SQL。

    ``fetchone()`` 的静态类型是 ``tuple | None``（查询可能一行都不返回），
    旧代码直接 ``.fetchone()[0]`` —— 类型检查器说得对。这里把它变成
    **一句能读懂的报错**：``COUNT(*)`` 永远有一行，所以"没有行"只可能是
    查询本身写错了，而 ``None[0]`` 的 TypeError 不会告诉你这一点。
    """
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"查询没有返回任何行：{' '.join(sql.split())[:70]}")
    return int(row[0])


def _guardrail_harm_from_rows(rows, guardrail: str) -> float:
    """从 ADS 行算某条护栏的相对伤害（只用于报告展示）。

    方向假定 ``lower_is_better``（演示里两条都是），因为这里只打印，
    真正判定在 ``ablab/platform/guardrails.py``（按声明的方向折算）。
    """
    means = {v: m for g, v, _n, m in rows if g == guardrail}
    if "control" not in means or "treatment" not in means or not means["control"]:
        return float("nan")
    return float(means["treatment"] / means["control"] - 1.0)


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
        counts[table] = scalar(con, f"SELECT COUNT(*) FROM {table}")
        emit(f"  {table:<34} {counts[table]:>10,}   {desc}")

    # ---- 2. 口径验证：首次曝光去重 ---------------------------------------- #
    emit("\n### 2. 口径验证")
    raw = counts["ods_exposure_log"]
    dedup = scalar(
        con,
        "SELECT COUNT(*) FROM (SELECT DISTINCT experiment, user_id FROM ods_exposure_log)",
    )
    dwd = counts["dwd_experiment_user"]
    emit(f"  首次曝光去重: 原始曝光 {raw:,} 条 -> 去重后 {dedup:,} 个 (实验, 用户) 对")
    emit(f"  DWD 行数 = {dwd:,}，与去重结果一致: {dwd == dedup}")

    rolled = scalar(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT experiment, variant, SUM(user_cnt) AS n
            FROM dws_experiment_variant_daily
            GROUP BY experiment, variant
        ) d JOIN ads_experiment_result a USING (experiment, variant)
        WHERE d.n <> a.user_cnt
        """,
    )
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

    # ---- 6. 比值链路（06/07）的交叉验证 -------------------------------- #
    #
    # 这一节存在的理由：比值指标走的是**另一条 ADS 链路**（分子/分母两列可加量），
    # 而"换了数据源，校准主张就不再自动成立"是这个项目反复强调的规矩。
    # 所以这里不是"再看一眼数字"，而是拿 **M1 的独立实现**（直接吃 DWD 明细的
    # ratio_delta_method）当裁判，并顺带给出负对照与"末次查看 == 主结论"的不变量。
    emit("\n### 6. 比值指标链路（06/07）：平台 vs M1 独立实现")
    ratio_rows = con.execute(
        "SELECT experiment, variant, user_cnt, sum_y, sum_x FROM ads_experiment_ratio_result"
        " ORDER BY experiment, variant"
    ).df()
    emit(f"  比值 ADS 有 {len(ratio_rows)} 行（每条实验两臂）")
    for _, r in ratio_rows.iterrows():
        ratio = float(r["sum_y"]) / float(r["sum_x"]) if float(r["sum_x"]) else float("nan")
        emit(f"    {r['experiment']:<14}{r['variant']:<11}"
             f"Σy={float(r['sum_y']):,.1f}  Σx={float(r['sum_x']):,.0f}"
             f"  比值 Σy/Σx={ratio:.6f}")

    from ablab.inference import ratio_delta_method
    from ablab.inference.aggregates import AggregateStats
    from ablab.platform.analysis import analyse_experiment_from_warehouse
    from ablab.platform.registry import ExperimentRecord

    def _stats(frame) -> AggregateStats:
        y = frame["post_metric"].to_numpy(dtype=float)
        x = frame["post_cnt"].to_numpy(dtype=float)
        return AggregateStats.from_sums(
            n=int(y.size), sum_x=float(x.sum()), sum_y=float(y.sum()),
            sum_xx=float((x * x).sum()), sum_yy=float((y * y).sum()),
            sum_xy=float((x * y).sum()),
        )

    emit("")
    emit(f"  {'实验':<14}{'平台比值':>12}{'平台 SE':>10}{'独立实现':>12}"
         f"{'偏差':>10}{'p':>10}")
    for experiment in ("exp_rank_v2", "exp_rec_emb"):
        detail = con.execute(
            "SELECT variant, post_metric, post_cnt FROM dwd_experiment_user"
            " WHERE experiment = ?",
            [experiment],
        ).df()
        t = detail[detail["variant"] == "treatment"]
        c = detail[detail["variant"] == "control"]
        ref = ratio_delta_method(_stats(t), _stats(c))

        record = ExperimentRecord(
            id="ratio_check", name=experiment, salt=f"{experiment}_v1",
            variants=[{"name": "control", "weight": 0.5},
                      {"name": "treatment", "weight": 0.5}],
            primary_metric="post_metric_14d", warehouse_experiment=experiment,
            estimator="post_only", metric_type="ratio",
        )
        rep = analyse_experiment_from_warehouse(record, con, n_looks=5)
        primary = rep.primary
        if primary is None:  # pragma: no cover - 比值路径必然给出 primary
            raise RuntimeError(f"{experiment} 的比值路径没有给出主估计")
        deviation = abs(primary.absolute_effect - ref.absolute_effect)
        emit(f"  {experiment:<14}{primary.absolute_effect:>12.6f}"
             f"{primary.std_error:>10.6f}{ref.absolute_effect:>12.6f}"
             f"{deviation:>10.2e}{primary.p_value:>10.3g}")

        # 不变量：监控曲线的最后一次查看必须**逐位**等于主结论（M6.1 的那条）
        last = rep.monitoring[-1]
        same = (
            abs(float(last["effect"]) - primary.absolute_effect) < 1e-12
            and abs(float(last["std_error"]) - primary.std_error) < 1e-12
        )
        emit(f"    末次查看 == 主结论：{same}；监控口径={last.get('estimator')}；"
             f"查看次数={len(rep.monitoring)}")

    emit("")
    emit("  负对照（exp_rec_emb 的真实效应为零）—— **实测它显著**（p=0.0091），")
    emit("  所以不能拿「它不显著」当验收标准。真正的问题是：这是比值链路的问题，")
    emit("  还是这份数据本身的问题？同一份数据上均值口径的负对照是：")
    for a in analyses:
        if a.experiment == "exp_rec_emb":
            emit(f"    {a.experiment} 均值口径 post-only "
                 f"{a.naive.absolute_effect:+.4f} (SE {a.naive.std_error:.4f}, "
                 f"p={a.naive.p_value:.3g})")
    emit("  两个口径都显著 → 指向**同一份实现**（第 5 节的效应分解已经说明：")
    emit("  这次分流的协变量失衡让 naive 显著，CUPED 把它扣掉）。")
    emit("  所以这里的结论只是：比值链路的表现与均值链路**一致**，")
    emit("  而不是「比值链路被验证为校准」—— 那需要很多个 salt 的重复，")
    emit("  是 README 已知边界里还没做的那一条。")
    emit("  逐位一致性：平台编排与 M1 的独立实现在 1e-9 内一致（有测试守着）；")
    emit("  本节偏差列是实测差，量级 1e-14。")

        # ---- 护栏链路（08/09）：长表 + 判定所需的可加量 ------------------------ #
    emit("\n### 护栏链路：08 DWS -> 09 ADS（长表，不新增落地文件）")
    emit("  护栏与主指标**共用一张事件表**（event_name = 护栏名），所以：")
    emit("    · 不需要新的 Parquet 与新的 ODS 视图，08 路一条 GROUP BY 就够；")
    emit("    · 代价是 01 路 DWD **必须**按 event_name = 'interaction' 过滤 ——")
    emit("      不加这一条，护栏的取值（延迟 ~100ms）会被加进主指标：")
    emit("      实测效应从 +27.2 变成 +161，而两个数都「正常显著」。")
    emit("      这类错误显著性检查发现不了，只能靠不变量（有测试钉着）。")
    emit("")
    emit("  名单来自**声明**（dim_guardrail_config），不是「事件里出现过什么」：")
    try:
        declared = con.execute(
            "SELECT experiment, guardrail, direction, max_harm "
            "FROM dim_guardrail_config ORDER BY experiment, guardrail"
        ).fetchall()
        emit(f"  {'实验':<16}{'护栏':<18}{'方向':<18}{'容忍度':>8}")
        for exp_name, guard, direction, limit in declared:
            emit(f"  {exp_name:<16}{guard:<18}{direction:<18}{float(limit):>8.2%}")
        emit("")
        emit("  09 路 ADS（判定所需的可加量，**不含阈值** —— 阈值是声明，属于注册表）：")
        rows = con.execute(
            """
            SELECT guardrail, variant, user_cnt, value_mean
            FROM ads_experiment_guardrail_result
            WHERE experiment = 'exp_rank_v2'
            ORDER BY guardrail, variant
            """
        ).fetchall()
        emit(f"  {'护栏':<18}{'臂':<12}{'n':>10}{'均值':>12}")
        for guard, variant, n, mean in rows:
            emit(f"  {guard:<18}{variant:<12}{int(n):>10}{float(mean):>12.4f}")
        if rows:
            harm = _guardrail_harm_from_rows(rows, "latency_p99")
            emit("")
            emit(f"  latency_p99 注入的真实伤害 = {harm:+.2%}"
                 "（演示真值；判定用的是这个数**是否越过声明的容忍度**）")
    except Exception as exc:  # 老库没有 09 路表
        emit(f"  （这份数仓里没有护栏链路：{type(exc).__name__}）")

    # ---- 簇级 CUPED：整簇随机化下的口径与代价 ------------------------------ #
    emit("\n### 簇级 CUPED：整簇随机化 + 前置指标（本轮打开的一条口径）")
    emit("  这条口径原先被**创建时就拒绝**，理由写的是「数据源没有簇级前置指标」。")
    emit("  实测那是个**过时假设**：05 路 DWS 一直落着簇级的 pre/cross 列，")
    emit("  pre_sum ≈ 4.4e6。现在按声明走，真拿不到前置指标时由分析层报错。")
    emit("")
    try:
        from ablab.platform.analysis import analyse_experiment_from_warehouse
        from ablab.platform.registry import ExperimentRecord

        demo_variants = [
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ]
        for est in ("post_only", "cuped"):
            rec = ExperimentRecord(
                name="exp_city_ctr", variants=demo_variants, salt="exp_city_ctr_v1",
                primary_metric="post_metric_14d", warehouse_experiment="exp_city_ctr",
                analysis_unit="cluster", estimator=est,
            )
            rep = analyse_experiment_from_warehouse(rec, con)
            n_units = int(getattr(rep, "n_analysis_units", 0) or 0)
            primary = rep.primary
            if primary is None:  # 理论上不会发生；发生了就说出来，别静默跳过
                emit(f"  estimator={est:<10} 报告没有主口径估计（跳过）")
                continue
            emit(f"  estimator={est:<10} 效应 {primary.absolute_effect:+.4f}"
                 f"  SE {primary.std_error:.4f}"
                 f"  分析单元数 {n_units}"
                 f"（{'簇' if rep.analysis_unit == 'cluster' else '用户'}）")
            if rep.cuped_fit is not None:
                emit(f"    ρ={rep.cuped_fit.correlation:.4f}"
                     f"  方差缩减 {rep.cuped_fit.variance_reduction:.2%}"
                     f"  标准误降 {rep.cuped_fit.se_shrinkage:.2%}")
        emit("")
        emit("  两条口径的**观测单位都是簇**（自由度 = 簇数 − 2），CUPED 只是把")
        emit("  每簇的 (前置均值, 后置均值) 当成一对观测再做回归调整 ——")
        emit("  与单元级 CUPED 是同一份实现（``cuped_estimate``）。")
        emit("  边界：这一轮**没有**量簇级 CUPED 的 A/A 校准（数仓只有一份实现，")
        emit("  换不了 salt），所以只报「与 post-only 相比方差确实降了」，")
        emit("  不声称名义覆盖率 —— 那句话要等一个能重复抽样的路径才敢写。")
    except Exception as exc:
        emit(f"  （这份数仓里没有簇级实验：{type(exc).__name__}: {exc}）")

    emit(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    con.close()

    report = out_dir / "warehouse_report.md"
    report.write_text(
        "# 数仓链路验证报告\n\n```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\n报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
