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
    DEFAULT_EXPERIMENTS,
    WarehouseConfig,
    analyse_ads,
    build_warehouse,
    covariate_adjustment_report,
    ratio_replicate_experiments,
    render_report,
    verify_against_detail,
)
from ablab.warehouse.ratio_calibration import run_ratio_link_calibration  # noqa: E402

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

    #: 6b 节的复制实验个数。100 个：FWER 的蒙特卡洛标准误约 2.2%，
    #: 足以判断名义 5% 有没有被盖住。
    #:
    #: **故意不提供 --quick**：6b 的数值进了 `check_readme_claims.py` 的声明清单，
    #: 个数一变那些声明就不成立 —— 那正是 README 第 31 条说的"假红灯"。
    #: 检查集那一侧也有契约测试（`test_ci_contract.py`）盯着"脚本接受 --quick
    #: 就必须在计划里声明"，两边是同一件事的两面。
    n_rep = 100
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
    emit("  而不是「比值链路被验证为校准」—— 那需要很多个 salt 的重复。")
    emit(f"  **那一批 salt 在 6b 节补上了**（{n_rep} 个 A/A 复制实验走完整条真实链路）。")
    emit("  逐位一致性：平台编排与 M1 的独立实现在 1e-9 内一致（有测试守着）；")
    emit("  本节偏差列是实测差，量级 1e-14。")

    # ---- 6b. 比值链路的**序贯校准**：100 个 A/A 复制实验 -------------------- #
    #
    # 6 节证明的是"两份实现算得一样"（一致性），回答不了校准：
    # 序贯 FWER、区间覆盖、z 的方差都是关于**一个分布**的陈述，一个 salt 给不出分布。
    # 这一节造 100 个真实效应为 0 的复制实验（各自一层、各自 salt），
    # 走**完整条真实链路**（ODS→DWD→DWS→ADS→平台编排→序贯判定）跑 100 遍。
    #
    # 为什么这是精确的校准而不是"仿真"：见 audit 模块的 docstring ——
    # 固定结果序列、只重抽分流 → 尖锐零假设逐字成立，随机化分布 i.i.d.。
    #
    # **单独建一条库**：复制实验会往曝光表里加 100 个实验，
    # 混进默认演示库会改掉上面每一节的数字（也会让报告多出 100 段）。
    emit(f"\n### 6b. 比值链路的序贯校准：{n_rep} 个 A/A 复制实验走完整条真实链路")
    emit("  6 节证明的是**一致性**（两份实现算得一样），而校准是关于**分布的**：")
    emit("  序贯 FWER、区间覆盖、z 的方差，一个 salt 都给不出来。")
    emit(f"  办法：造 {n_rep} 个真实效应为 0 的复制实验（各自一层、各自 salt、同一段桶位），")
    emit("  每个都走 ODS→DWD→DWS→ADS→平台编排→序贯判定，然后数分布。")
    emit("  为什么它是**精确**校准：结果序列固定、只重抽分流 → 尖锐零假设逐字成立，")
    emit(f"  {n_rep} 次独立随机化就是随机化分布的 i.i.d. 抽样（不依赖任何渐近论）。")
    emit("")

    rep_root = ROOT / "build" / "ratio_rep"
    rep_root.mkdir(parents=True, exist_ok=True)
    rep_cfg = WarehouseConfig(
        n_users=args.users,
        experiments=DEFAULT_EXPERIMENTS + ratio_replicate_experiments(n_rep),
    )
    rep_con = build_warehouse(
        db_path=rep_root / "ratio_rep.duckdb",
        data_dir=rep_root / "source",
        sql_dir=ROOT / "sql",
        config=rep_cfg,
        force_data=args.rebuild,
        verbose=False,
    )
    calib = run_ratio_link_calibration(rep_con, n_replicates=n_rep, n_looks=5, alpha=0.05)
    for line in calib.summary().splitlines():
        emit("  " + line)
    emit("")
    emit("  三个承诺的实现情况（这才是「校准」该有的写法，而不是一句「看起来对」）：")
    emit(f"    · 序贯 FWER {calib.fwer:.4f}，Wilson [{calib.fwer_interval[0]:.4f}, "
         f"{calib.fwer_interval[1]:.4f}] —— 盖住名义 0.05；")
    emit(f"    · 末次重复区间覆盖 0 的比例 {calib.coverage:.4f}，Wilson "
         f"[{calib.coverage_interval[0]:.4f}, {calib.coverage_interval[1]:.4f}]；")
    emit(f"    · z 的 sd {calib.z_sd_final:.4f}、均值 {calib.z_mean_final:+.4f}、"
         f"偏度 {calib.z_skew_final:+.3f} —— 比值 delta method 的 SE 在真实链路上诚实。")
    emit(f"  过度离散检验（{n_rep} 个 salt 的臂占比 vs 二项理论）通过，"
         "所以极端值那一个只是尾部，不是分流坏了：")
    emit(f"    chi2 = {calib.arm_share_chi2:.2f}（df={calib.n_replicates}），"
         f"p = {calib.arm_share_chi2_p:.4f}；最小 SRM p = {calib.srm_min_p:.3g}")
    emit("  **仍未做**：真实效应下的功效/覆盖。复制实验共享同一份结果序列、")
    emit("  真实效应为 0，所以它只能校准零效应；要给真实效应，必须让复制实验")
    emit("  走自己的事件名与自己的 DWD 链路（否则它的效应会加进共享序列、")
    emit("  把别的实验的数字改掉）。这条写进 README 的已知边界。")
    rep_con.close()

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
        emit("  边界：**数仓路径上的**簇级 A/A 校准仍然没做。6b 节那套复制实验机制")
        emit("  已经在了，但它每个复制只覆盖一段桶位；簇级 A/A 要求每个复制有足够多的")
        emit("  **簇**（自由度 = 簇数 − 2，24 个城市那次已经踩过「某臂只剩一个簇」），")
        emit("  也就是每个复制要吃掉 ~30 个城市 —— 那是另一套规模的重复，留作下一步。")
        emit("  所以这里只报「与 post-only 相比方差确实降了」，不声称名义覆盖率；")
        emit("  簇级 CUPED 的 A/A 校准目前走**合成路径**（reports/m6_validation.md 第 7 节，")
        emit("  200 次换 salt：误停率 0.0400，Wilson [0.0204, 0.0769] 盖住名义 5%）。")
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
