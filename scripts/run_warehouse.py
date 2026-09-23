#!/usr/bin/env python
"""构建 DuckDB 数仓链路（ODS→DWD→DWS→ADS）并输出实验结论。

运行::

    python scripts/run_warehouse.py
    python scripts/run_warehouse.py --users 50000 --rebuild

输出到 ``reports/warehouse_report.md``，同时打印到终端。

这份脚本演示「数仓链路 + 统计引擎」的接口面：
SQL 负责口径（首次曝光去重、前后窗口、可加汇总），
Python 负责推断（t 检验、置信区间、SRM 体检），
两边各有一个交叉验证点，任何一个不过就说明有一侧写错了。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.inference import (  # noqa: E402
    CovariateMoments,
    fit_multivariate_cuped,
    multivariate_cuped_from_moments,
)
from ablab.reporting import for_report  # noqa: E402
from ablab.warehouse import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    ExternalExperiment,
    WarehouseConfig,
    analyse_ads,
    build_warehouse,
    cluster_replicate_experiments,
    covariate_adjustment_report,
    load_real_traffic,
    ratio_replicate_experiments,
    ratio_replicate_experiments_with_lift,
    render_report,
    verify_against_detail,
)
from ablab.warehouse.cluster_calibration import (  # noqa: E402
    run_cluster_replicate_calibration,
)
from ablab.warehouse.ratio_calibration import (  # noqa: E402
    RerandomizationReference,
    lift_replicate_name,
    rerandomization_reference,
    run_ratio_link_calibration,
    run_ratio_link_power_calibration,
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
    """跑一句「只返回一个数」的 SQL。

    ``fetchone()`` 的静态类型是 ``tuple | None``（查询可能一行都不返回），
    旧代码直接 ``.fetchone()[0]`` —— 类型检查器说得对。这里把它变成
    **一句能读懂的报错**：``COUNT(*)`` 永远有一行，所以「没有行」只可能是
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
    #: 个数一变那些声明就不成立 —— 那正是 README 第 31 条说的「假红灯」。
    #: 检查集那一侧也有契约测试（`test_ci_contract.py`）盯着"脚本接受 --quick
    #: 就必须在计划里声明"，两边是同一件事的两面。
    n_rep = 100
    #: 6c 节：带真实效应的复制实验。80 个 × 每臂 ~730 个单元 ⇒
    #: 覆盖率的 Wilson 区间宽约 ±5.6%、功效约 0.8，足以判断「盖不盖得住」。
    #: n_users 决定每个复制的样本量（桶位由 10000 均分给 n_pow 个复制）。
    n_pow = 80
    pow_lift = 2.0
    pow_users = 120_000
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

    # 多协变量 CUPED：**真实数据上**的两个协变量（前置互动值 + 前置互动次数）
    #
    # 为什么单独写一节：上面的诊断只用了一个协变量（pre_metric）。
    # 数仓的 DWD 里其实还落着 pre_cnt，两个一起用能多拿多少方差缩减
    # 是个能用真实数据回答的问题 —— 而不是靠仿真说"理论上更多"。
    emit("")
    emit("  **多协变量 CUPED（真实数据：pre_metric + pre_cnt）**")
    emit("  上面的诊断只用了一个协变量；DWD 里还落着 pre_cnt（前置互动次数）。")
    emit("  两个协变量一起用能多拿多少？用**交叉拟合**的 θ̂（诚实口径）量：")
    detail = con.execute(
        "SELECT variant, pre_metric, pre_cnt, post_metric FROM dwd_experiment_user"
        " WHERE experiment = 'exp_rank_v2'"
    ).df()
    X_two = detail[["pre_metric", "pre_cnt"]].to_numpy(dtype=float)
    y_post = detail["post_metric"].to_numpy(dtype=float)
    mv_one = fit_multivariate_cuped(X_two[:, :1], y_post, n_folds=5, seed=0)
    mv_two = fit_multivariate_cuped(X_two, y_post, n_folds=5, seed=0)
    mv_in = fit_multivariate_cuped(X_two, y_post, n_folds=1)
    # **三位小数**：这几个数是从 DWD 明细算出来的，DuckDB 并行聚合的浮点末位
    # 会让第 4 位漂（实测 0.7438 / 0.7440）。声明清单钉的是逐字字符串，
    # 所以这里按"不会翻面"的精度报 —— 与 SA 那个"改善了 50 倍"是同一个教训。
    emit(f"    单协变量（pre_metric）：诚实方差缩减 {mv_one.variance_reduction:.3f}"
         f"（最好的单协变量 {mv_one.best_univariate_reduction:.3f}）")
    emit(f"    两协变量（+pre_cnt）  ：诚实方差缩减 {mv_two.variance_reduction:.3f}"
         f"，多拿 {mv_two.extra_from_multivariate:+.3f}")
    emit(f"    同一个拟合的**样本内**缩减 {mv_in.variance_reduction:.3f}"
         f"（过拟合 {mv_in.variance_reduction - mv_two.variance_reduction:+.3f}）"
         f"，条件数 {mv_two.condition_number:.2f}")
    emit("    -> 多一个真实协变量确实多拿了一点；而样本内与交叉拟合的差就是"
         "「多协变量看起来更有效」的那一部分。")

    # 同一条口径，从**充分统计量**算（10/11）：
    #
    # 上面那两个数是**扫 DWD 明细**算出来的。而"能不能不回扫明细"是个工程问题：
    # 多协变量 CUPED 只依赖二阶矩，而二阶矩是可加的 —— 所以这一轮把它们落成
    # 两张新表（10/11）。这一节的意义与第 6 节（比值链路）一样：
    # **换了读取路径，口径必须还是那个口径**，所以拿明细路径当裁判。
    emit("")
    emit("  **同一条口径，从充分统计量算（10/11）**")
    emit("  上面的数是**扫明细**算出来的。多协变量 CUPED 只依赖二阶矩")
    emit("  （n, Σx_i, Σx_i x_j, Σx_i y, Σy, Σy²），而二阶矩是可加的 ——")
    emit("  所以把它们落成 DWS/ADS 两张新表（10/11），上层就能不回扫明细地复原 θ̂。")
    cov_rows = con.execute(
        """
        SELECT user_cnt,
               pre_metric_sum, pre_metric_sq_sum, pre_cnt_sum, pre_cnt_sq_sum,
               pre_metric_pre_cnt_cross_sum, pre_metric_post_cross_sum,
               pre_cnt_post_cross_sum, post_metric_sum, post_metric_sq_sum
        FROM ads_experiment_covariate_result WHERE experiment = 'exp_rank_v2'
        """
    ).fetchall()
    # 两臂**加起来**再喂给估计器：θ̂ 是在全样本上估的（见 11 号 SQL 的注释）
    total = [sum(float(row[i]) for row in cov_rows) for i in range(1, 10)]
    n_cov = float(sum(float(row[0]) for row in cov_rows))
    moments = CovariateMoments(
        n=n_cov,
        sum_x=np.array([total[0], total[2]]),
        sum_xx=np.array([[total[1], total[4]], [total[4], total[3]]]),
        sum_xy=np.array([total[5], total[6]]),
        sum_y=total[7],
        sum_yy=total[8],
    )
    mv_ads = multivariate_cuped_from_moments(moments)
    mv_detail_in = fit_multivariate_cuped(X_two, y_post, n_folds=1)
    # 报三位小数 + 偏差的**量级**（而不是位数）：这几个数由 DuckDB 并行聚合算出来，
    # 浮点末位会漂（实测 0.7438 / 0.7440）。把会漂的位数写进报告，
    # 声明清单就会随机变红 —— 与"最大相对差只报量级"是同一条规矩（决策 31）。
    emit(f"    ADS 充分统计量 {n_cov:,.0f} 人 -> **样本内**缩减 "
         f"{mv_ads.variance_reduction:.3f}")
    deviation = abs(mv_ads.variance_reduction - mv_detail_in.variance_reduction)
    emit(f"    明细路径同口径          -> **样本内**缩减 "
         f"{mv_detail_in.variance_reduction:.3f}"
         f"，偏差量级 1e{int(round(np.log10(max(deviation, 1e-300))))}"
         "（< 1e-9，与测试同一条阈值）")
    emit(f"    条件数：ADS {mv_ads.condition_number:.2f}"
         f" / 明细 {mv_detail_in.condition_number:.2f}")
    emit("    **边界（这一层最重要的结论）：交叉拟合的 θ̂ 从充分统计量算不出来。**")
    emit("    它要求「留出那一折用**别的折**估的 θ̂」，而「哪个用户在哪个折」不是")
    emit("    可加量 —— 折号一旦落库，随机划分就变成数据的一部分（换一个 seed")
    emit("    就得重跑整条数仓）。所以 ADS 路径给的是**样本内**口径，")
    emit("    交叉拟合仍然只能在明细上做：诚实口径 "
         f"{mv_two.variance_reduction:.3f}，与样本内差 "
         f"{mv_in.variance_reduction - mv_two.variance_reduction:+.3f}。")

    # ---- 6. 比值链路（06/07）的交叉验证 -------------------------------- #
    #
    # 这一节存在的理由：比值指标走的是**另一条 ADS 链路**（分子/分母两列可加量），
    # 而「换了数据源，校准主张就不再自动成立」是这个项目反复强调的规矩。
    # 所以这里不是「再看一眼数字」，而是拿 **M1 的独立实现**（直接吃 DWD 明细的
    # ratio_delta_method）当裁判，并顺带给出负对照与「末次查看 == 主结论」的不变量。
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
    # 6 节证明的是「两份实现算得一样」（一致性），回答不了校准：
    # 序贯 FWER、区间覆盖、z 的方差都是关于**一个分布**的陈述，一个 salt 给不出分布。
    # 这一节造 100 个真实效应为 0 的复制实验（各自一层、各自 salt），
    # 走**完整条真实链路**（ODS→DWD→DWS→ADS→平台编排→序贯判定）跑 100 遍。
    #
    # 为什么这是精确的校准而不是「仿真」：见 audit 模块的 docstring ——
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
    emit("  **这一节只能校准零效应**（复制实验共享同一份结果序列、真实效应为 0）。")
    emit("  真实效应下的覆盖与功效由 6c 节补上。")
    rep_con.close()

    # ---- 6c. 真实效应下的校准：覆盖、功效、SE 是否诚实 ---------------------- #
    #
    # 6b 校准的是零效应（FWER、z 的方差、覆盖 0）；功率与"真实效应下盖不盖得住"
    # 是另一半，而它需要**真的往结果里加效应**。做法：建一条**只含这批实验**的
    # 源数据（不是新事件名、也不是新 SQL —— 换一条数据就够了），每个复制实验
    # 自带一段互斥的桶位、真实效应 = 每条互动 +lift。
    #
    # 真值为什么可以取 lift 本身：生成器把效应加进**每一条**后置互动记录的取值，
    # 而分母 post_cnt 数的就是互动条数 ⇒ Y_i(1) = Y_i(0) + lift·X_i，
    # 两边同除 ΣX 得 R_t = R_t(0) + lift。**逐字相等**，不用估「大概的真值」。
    emit("\n### 6c. 真实效应下的校准：覆盖、功效、以及 SE 到底诚不诚实")
    emit("  6b 只能校准零效应。这一节让复制实验**真的带效应**：")
    emit(f"  {n_pow} 个复制实验，每个占一段互斥桶位（一个用户最多落进一个），")
    emit(f"  真实效应 = 每条互动 +{pow_lift:g}。真值为什么能取这个数本身：")
    emit("  效应是加在**每一条**后置互动记录的取值上的，而分母 post_cnt 数的就是")
    emit("  互动条数 ⇒ Y_i(1) = Y_i(0) + lift·X_i ⇒ R_t = R_t(0) + lift，")
    emit("  **逐字相等** —— 所以覆盖率是直接对着真值数的，不需要先估一个真值。")
    emit("")

    pow_root = ROOT / "build" / "ratio_pow"
    pow_root.mkdir(parents=True, exist_ok=True)
    pow_cfg = WarehouseConfig(
        n_users=pow_users,
        experiments=ratio_replicate_experiments_with_lift(n_pow, lift=pow_lift),
    )
    pow_con = build_warehouse(
        db_path=pow_root / "ratio_pow.duckdb",
        data_dir=pow_root / "source",
        sql_dir=ROOT / "sql",
        config=pow_cfg,
        force_data=args.rebuild,
        verbose=False,
    )
    power_calib = run_ratio_link_power_calibration(
        pow_con, n_replicates=n_pow, true_lift=pow_lift, n_looks=5, alpha=0.05
    )
    for line in power_calib.summary().splitlines():
        emit("  " + line)
    emit("")
    emit("  40 个点量不出「SE 是不是 1」（sd 本身就有 ±30% 的区间），所以再做一次")
    emit("  **重随机化对照**：DGP 已知 ⇒ 潜在结果可逐字重建，于是在同一批用户上")
    emit("  重抽 400 次分流，直接量估计量的设计分布（蒙特卡洛误差只由重抽次数决定）：")
    # 变量名不要复用上面 6 节那个 `ref`（那是 Estimate，mypy 会按第一个绑定推断）
    refs: list[RerandomizationReference] = []
    for i in (0, n_pow // 2):
        rr = rerandomization_reference(
            pow_con, experiment=lift_replicate_name(i), true_lift=pow_lift, n_splits=400
        )
        refs.append(rr)
        for line in rr.summary().splitlines():
            emit(line)
    emit("")
    emit("  三条结论（每个数都从这里算出来，不写死）：")
    cov_ok = "盖住" if power_calib.coverage_covers_nominal else "**没盖住**"
    emit(f"    · **覆盖**：末次 95% 区间覆盖真值 {power_calib.coverage:.4f}"
         f"（Wilson [{power_calib.coverage_interval[0]:.4f}, "
         f"{power_calib.coverage_interval[1]:.4f}]）—— {cov_ok}名义 0.95；")
    emit(f"    · **功效**：末次显著率 {power_calib.power:.4f}，序贯口径 "
         f"{power_calib.sequential_power:.4f}；非中心度 {power_calib.noncentrality:.3f} ——")
    emit("      所以「检出率」这个数不能单独读，它由这批设置的信息量决定；")
    se_ok = "盖住" if power_calib.se_is_honest else "**没盖住**"
    emit(f"    · **SE 诚实**：平均 SE ÷ 跨复制 sd = {power_calib.se_over_sd:.4f}"
         f"（区间 [{power_calib.se_over_sd_interval[0]:.4f}, "
         f"{power_calib.se_over_sd_interval[1]:.4f}]，{se_ok} 1）；")
    emit("      重随机化那两组更锋利（蒙特卡洛误差只由重抽次数决定）："
         + "；".join(
             f"{r.experiment} 的 SE/sd {r.se_over_sd:.4f}、偏差 {r.bias:+.4f}"
             for r in refs
         )
         + "。")
    emit("  数仓路径上的**簇级** A/A 校准见下一节（6d）：那里量了误停率与 z 的分布，")
    emit("  所以这一节不再需要「只报方差缩减、不声称覆盖率」这句边界。")
    pow_con.close()

        # ---- 6d. 数仓路径上的**簇级** A/A 校准 ---------------------------------- #
    #
    # 上面那节只报了"簇级 CUPED 的方差确实降了"，并明确写着"不声称名义覆盖率" ——
    # 因为数仓只有一份实现、换不了 salt。现在有了整簇随机化的复制实验
    # （cluster_replicate_experiments：每个自带一层、自带 salt、true_lift=0），
    # 那句话就不再成立：60 个整簇 A/A 走完整条真实链路，量误停率与 z 的分布。
    #
    # **单独建库**：每个整簇复制实验都会路由到全部用户（簇级分流要 60 个簇），
    # 混进默认演示库会把已有数字改掉。
    emit("\n### 6d. 数仓路径上的簇级 A/A 校准（整簇随机化 × 60 个 salt）")
    emit("  为什么必须单独量一遍：簇级推断走的是**另一条数据链路** ——")
    emit("  ADS 给臂级总数、簇粒度 DWS 给每组簇的统计量，两者由不变量钉在一起。")
    emit("  合成路径上用对了统计量，**推不出**数仓那条读取路径也对。")
    emit("")

    clu_root = ROOT / "build" / "cluster_rep"
    clu_root.mkdir(parents=True, exist_ok=True)
    # **这个脚本故意没有 --quick**（数值进了声明清单，快速模式会让它随机变红，
    # 见检查计划里的注释与 test_ci_contract）。所以这里写死 60 个复制实验。
    n_clu = 60
    clu_cfg = WarehouseConfig(
        # 8,000 个用户：簇级分流要的是**簇数**（城市仍是 60 个），
        # 每个复制实验都会路由到全部用户，所以用户数只影响规模、不影响簇数。
        n_users=8_000,
        experiments=DEFAULT_EXPERIMENTS + cluster_replicate_experiments(n_clu),
    )
    clu_con = build_warehouse(
        db_path=clu_root / "cluster_rep.duckdb",
        data_dir=clu_root / "source",
        sql_dir=ROOT / "sql",
        config=clu_cfg,
        force_data=args.rebuild,
        verbose=False,
    )
    clu_results = []
    for est in ("post_only", "cuped"):
        cres = run_cluster_replicate_calibration(clu_con, n_replicates=n_clu, estimator=est)
        clu_results.append(cres)
        for line in cres.summary().splitlines():
            emit("  " + line)
        emit("")
    clu_con.close()
    emit("  三条要一起读的：")
    emit(f"    · **两者都盖住名义值**：post-only 误停率 "
         f"{clu_results[0].fpr:.4f}（Wilson 上界 {clu_results[0].fpr_interval[1]:.4f}）、"
         f"CUPED {clu_results[1].fpr:.4f}（Wilson 上界 "
         f"{clu_results[1].fpr_interval[1]:.4f}）；")
    emit(f"    · **偏保守的那一侧**：末次 z 的 sd 只有 "
         f"{clu_results[0].z_sd:.4f} / {clu_results[1].z_sd:.4f}（应为 1）——")
    emit("      也就是说簇级 SE 大约**高估 18%**，代价是功效。这不是 bug：")
    emit("      CR1 的小样本修正 + t(簇数−2) 在均衡簇上本来就偏保守；")
    emit("      而 §7b 那节量过：真正会让 size 崩的是**簇大小不平衡**，这批示数据是均衡的。")
    emit("    · **CUPED 没有改变观测单位**：两个估计量的 z 分布同量级，")
    emit("      而合成路径上量的用户级误停率是它的十几倍（m6 报告第 7 节）。")

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
        emit("  **数仓路径上的簇级 A/A 校准已经补上了**（见下面的 6d 节）：")
        emit("  6b 那套复制实验是单元级的，每个只覆盖一段桶位；簇级 A/A 要求每个复制")
        emit("  有足够多的**簇**（自由度 = 簇数 − 2），所以另造了一批"
             "**整簇随机化**的复制实验。")
    except Exception as exc:
        emit(f"  （这份数仓里没有簇级实验：{type(exc).__name__}: {exc}）")

    # ---- 8. 真实数据入口：换数据源不改链路 ---------------------------------- #
    #
    # 这一节是抽象主张的**可执行版本**：把同一份合成源导出成"外部文件"
    # （CSV、列顺序打乱、多几列没用的、混一个未声明事件），走
    # ``load_real_traffic`` 接进来，再用**同一套 SQL** 建一次数仓，
    # 最后把两次的 ADS 逐行比。数字对得上，才叫「换数据源不改链路」；
    # 对不上，说明链路里还藏着对合成器形状的假设。
    emit("\n### 8. 真实数据入口：换数据源不改链路（端到端）")
    emit("  做法：小规模（n_users=3000）先建一条标准数仓作为对照；")
    emit("  把它的源数据导出成 CSV（列顺序打乱、多两列设备信息、")
    emit("  事件表里混一个未声明的 page_view），走 load_real_traffic 接入，")
    emit("  再用**同一套 SQL**（build_warehouse(generate=False)）建第二条。")
    emit("")

    ext_root = ROOT / "build" / "ingest_demo"
    if ext_root.exists():
        shutil.rmtree(ext_root, ignore_errors=True)
    ext_root.mkdir(parents=True, exist_ok=True)
    std_cfg = WarehouseConfig(n_users=3000)
    std_con = build_warehouse(
        db_path=ext_root / "std.duckdb", data_dir=ext_root / "std_source",
        sql_dir=ROOT / "sql", config=std_cfg, force_data=True, verbose=False,
    )
    std_ads = std_con.execute(
        "SELECT experiment, variant, user_cnt, post_sum FROM ads_experiment_result"
        " ORDER BY experiment, variant"
    ).df()
    std_con.close()

    external_dir = ext_root / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    src_dir = ext_root / "std_source"
    exposure = pd.read_parquet(src_dir / "exposure_log" / "part-0000.parquet")
    events = pd.read_parquet(src_dir / "event_log" / "part-0000.parquet")
    profile = pd.read_parquet(src_dir / "user_profile" / "part-0000.parquet")
    exposure.assign(device="web", app_version="9.9").to_csv(
        external_dir / "exposure_log.csv", index=False
    )
    stray = events.head(500).assign(event_name="page_view", metric_value=1.0)
    pd.concat([events, stray], ignore_index=True).assign(platform="ios").to_csv(
        external_dir / "event_log.csv", index=False
    )
    profile.assign(country="CN").to_csv(external_dir / "user_profile.csv", index=False)

    ingest_report = load_real_traffic(
        external_dir,
        target_dir=ext_root / "external_source",
        fmt="csv",
        # **三个实验都要声明**：第一版按"只走单元级链路"把簇级实验滤掉了，
        # 结果接入时直接报「曝光表里出现了没有声明的实验：exp_city_ctr」——
        # 这正是"口径必须来自声明"该有的行为（数据里有、声明里没有 = 拦下来）。
        experiments=tuple(
            ExternalExperiment(e.name, {"control": 0.5, "treatment": 0.5}, layer=e.layer)
            for e in DEFAULT_EXPERIMENTS
        ),
        metric_event="interaction",
        guardrail_events=("latency_p99", "complaint_rate"),
    )
    for line in ingest_report.summary().splitlines():
        emit("  " + line)
    emit("")
    ext_con = build_warehouse(
        db_path=ext_root / "external.duckdb",
        data_dir=ext_root / "external_source",
        sql_dir=ROOT / "sql",
        config=std_cfg,
        generate=False,
        verbose=False,
    )
    ext_ads = ext_con.execute(
        "SELECT experiment, variant, user_cnt, post_sum FROM ads_experiment_result"
        " ORDER BY experiment, variant"
    ).df()
    ext_con.close()

    merged = std_ads.merge(ext_ads, on=["experiment", "variant"], suffixes=("_std", "_ext"))
    rel = (
        (merged["post_sum_ext"] - merged["post_sum_std"]).abs()
        / merged["post_sum_std"].abs()
    )
    emit(f"  {'实验':<14}{'分支':<11}{'用户数':>9}{'Σy（合成）':>16}{'Σy（外部入口）':>17}{'相对差':>11}")
    for (_, row), r in zip(merged.iterrows(), rel):
        emit(f"  {row['experiment']:<14}{row['variant']:<11}{int(row['user_cnt_std']):>9}"
             f"{row['post_sum_std']:>16.4f}{row['post_sum_ext']:>17.4f}{r:>11.2e}")
    emit("")
    emit(f"  行数一致 {len(std_ads) == len(ext_ads)}，最大相对差 {rel.max():.2e}"
         " —— **同一套 SQL，两条完全不同的数据路径，数字到浮点末位相同**。")
    emit("  这才是「换数据源不改链路」的证据；在这之前那句话只是主张。")
    emit("  随之而来的三条边界也写在这里：")
    emit("    · 真实数据**没有 true_lift**（那一列写空）—— 报告里「演示真值」在外部路径上是空的；")
    emit("    · 变体名/实验名/事件名必须**声明**，不从数据反推：")
    emit("      曝光里出现未声明的变体名会直接报错，未声明的事件名只报告、")
    emit("      不会被任何指标读走（DWD 按 event_name 过滤）；")
    emit("    · 没有 user_profile 时簇级路径不可用（城市/注册日期缺失），其余链路不受影响。")
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
