"""DuckDB 数仓链路：ODS → DWD → DWS → ADS，再接推断层。

全链路职责边界（这是这个项目能同时讲"数仓"和"统计"的关键）：

    SQL 侧  口径正确、可加、可回溯：首次曝光去重、前后窗口、方差汇总
    Python 侧  假设检验、区间估计、诊断

两个交叉验证点，任何一个不过就说明有一侧写错了：

1. SQL 算的 SRM 卡方统计量 必须等于 Python ``srm_check`` 算的
2. SQL 汇总出的 (n, mean, var) 喂给 ``welch_ttest_from_stats``
   必须等于直接拉明细跑 ``welch_ttest``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from ..assignment import ExperimentSpec, Randomizer, Variant
from ..hashing import KeyBatcher
from ..inference import (
    AggregateStats,
    CupedFit,
    Estimate,
    cuped_estimate,
    cuped_ttest,
    fit_cuped,
    srm_check,
    welch_ttest,
    welch_ttest_from_stats,
)
from .generate import WarehouseConfig, generate_source_data

__all__ = [
    "SQL_ORDER",
    "split_statements",
    "build_warehouse",
    "run_sql_files",
    "load_ads_result",
    "ExperimentAnalysis",
    "analyse_ads",
    "render_report",
    "CrossValidation",
    "verify_against_detail",
    "CovariateAdjustmentReport",
    "covariate_adjustment_report",
]

#: SQL 文件必须按层序执行，文件名前缀就是执行顺序
SQL_ORDER: tuple[str, ...] = (
    "00_ods.sql",
    "01_dwd_experiment_user.sql",
    "02_dws_experiment_variant_daily.sql",
    "03_ads_experiment_result.sql",
    "04_ads_experiment_srm.sql",
    # 簇粒度 DWS：与 02 出自同一张 DWD、同一组 SUM，只是分组键多了一个 cluster_id。
    # 排在 ADS 之后是有意的 —— 它不属于主链路，而是"换个分析单元"的旁路，
    # 谁需要谁读。这样主链路的依赖关系保持清晰。
    "05_dws_experiment_cluster_daily.sql",
    # 比值指标口径：与 02/03 并列的第二条链路（分子=值之和、分母=次数之和）。
    # 独立成表而不是给 02/03 加列 —— 已发布层的内容一改动，
    # README 里所有已引用的数仓数字就全变了。
    "06_dws_experiment_ratio_daily.sql",
    "07_ads_experiment_ratio_result.sql",
    # 护栏链路（08/09）：与 06/07 同一个理由独立成表 —— 它既不是主指标口径，
    # 也不是比值口径，而是"另一组指标"。名单来自声明（dim_guardrail_config）。
    "08_dws_experiment_guardrail_daily.sql",
    "09_ads_experiment_guardrail_result.sql",
)


def split_statements(sql: str) -> list[str]:
    """把一个 .sql 文件切成多条可执行语句。

    DuckDB 的 Python ``execute`` 一次只跑一条语句，所以要自己切。
    切分前先剥掉整行 ``--`` 注释，避免"只剩注释的分片"被当成语句执行。
    本项目的 SQL 里没有字符串常量含分号的情况，因此朴素按 ``;`` 切分是安全的。
    """
    statements: list[str] = []
    for chunk in sql.split(";"):
        body = "\n".join(
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        )
        if body.strip():
            statements.append(chunk.strip())
    return statements


def run_sql_files(
    con: duckdb.DuckDBPyConnection,
    sql_dir: str | Path,
    substitutions: dict[str, object],
    *,
    order: tuple[str, ...] = SQL_ORDER,
    verbose: bool = True,
) -> list[str]:
    """按层序执行 SQL 文件，并替换 ``{PLACEHOLDER}`` 占位符。"""
    sql_dir = Path(sql_dir)
    executed: list[str] = []

    for filename in order:
        path = sql_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少 SQL 文件: {path}")

        sql = path.read_text(encoding="utf-8")
        for key, value in substitutions.items():
            sql = sql.replace("{" + key + "}", str(value))

        statements = split_statements(sql)
        for stmt in statements:
            con.execute(stmt)
        executed.append(filename)
        if verbose:
            print(f"  [SQL] {filename}  ({len(statements)} 条语句)")

    return executed


def build_warehouse(
    db_path: str | Path,
    data_dir: str | Path,
    sql_dir: str | Path,
    *,
    config: WarehouseConfig | None = None,
    force_data: bool = False,
    verbose: bool = True,
) -> duckdb.DuckDBPyConnection:
    """建源数据 → 跑四层 SQL → 返回已就绪的 DuckDB 连接。

    所有 DDL 都用 ``CREATE OR REPLACE``，因此本函数**幂等**：
    重复调用会原地重建，不会出现"表已存在"的报错，也不会脏读上一次的残留。
    """
    cfg = config or WarehouseConfig()
    db_path, data_dir = Path(db_path), Path(data_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    counts = generate_source_data(data_dir, cfg, force=force_data)
    if verbose:
        if "__cached__" in counts:
            print(f"  [源数据] 复用已有 Parquet: {data_dir}")
        else:
            for name, cnt in counts.items():
                print(f"  [源数据] {name:<20} {cnt:>10,} 行")

    con = duckdb.connect(str(db_path))
    # threads=4 沿用原值 —— **曾经想改它，但对照实验不支持**。
    #
    # 起因：同一份代码、同一份 Parquet 重跑两次，ADS 侧的 CUPED 效应出现过
    # -3.9243251037 -> -3.9243251038（1 ulp），而 DWD 明细侧（numpy 按行算的同一个量）
    # 一位没变。DuckDB 的并行聚合是"各线程先算部分和、再合并"，合并顺序随调度变化，
    # 浮点加法又不满足结合律 —— 看起来很像是它。
    #
    # 于是做了单变量对照：只把这一行改成 threads=1 / 改回 threads=4，各重跑 8 次，
    # 比较报告里 4 个量的全精度值。**两臂都是 8/8 逐位稳定、ADS 与 DWD 逐位相同** ——
    # 那个差异在 16 次重跑里一次都没复现，对照**没有功效**，证不出任何因果。
    # 观测到的频率是"两次全量跑里出现过一次"，8 次一臂本来就抓不到。
    #
    # 所以这里保持原值：一个没被证实的原因，不值得为它改生产配置（单线程在大数据量下
    # 是真的会慢）。可复现性改在**比对那一侧**解决：允许末位在容差内、但把差异量化
    # 出来（见 ablab/reporting.py 的 Comparison 与 scripts/check_report_determinism.py）。
    con.execute("PRAGMA threads=4")

    run_sql_files(
        con,
        sql_dir,
        {
            "DATA_DIR": data_dir.as_posix(),
            "PRE_DAYS": cfg.pre_days,
            "POST_DAYS": cfg.post_days,
        },
        verbose=verbose,
    )
    return con


def load_ads_result(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """读取 ADS 结果表。"""
    return con.execute("SELECT * FROM ads_experiment_result ORDER BY experiment, variant").df()


@dataclass
class ExperimentAnalysis:
    """一个实验的完整分析结论。

    **``cuped`` 是默认口径**，``naive``（post-only）保留下来做对照 ——
    两者的差距本身就说明了前置协变量校正值不值。
    """

    experiment: str
    layer: str
    hypothesis: str
    true_lift: float
    naive: Estimate
    cuped: Estimate
    fit: CupedFit
    sql_chi2: float
    python_chi2: float
    srm_p_value: float
    srm_degrees_of_freedom: int

    @property
    def estimate(self) -> Estimate:
        """默认结论 —— CUPED 校正后的估计。"""
        return self.cuped

    @property
    def srm_agrees(self) -> bool:
        """SQL 侧与 Python 侧的卡方统计量是否一致（交叉验证）。"""
        return abs(self.sql_chi2 - self.python_chi2) < 1e-6

    @property
    def srm_triggered(self) -> bool:
        return self.srm_p_value < 1e-3

    @property
    def p_value_shift(self) -> float:
        """CUPED 把 p 值推高/推低了几个数量级（用于展示校正效果）。"""
        if self.naive.p_value <= 0 or self.cuped.p_value <= 0:
            return float("nan")
        return float(np.log10(self.cuped.p_value / self.naive.p_value))

    def summary(self) -> str:
        c, n = self.cuped, self.naive
        srm_tag = "失衡(FAIL)" if self.srm_triggered else "均衡(PASS)"
        agree = "一致" if self.srm_agrees else "不一致 <<< 检查链路"
        return (
            f"\n{self.experiment}  [{self.layer} 层]\n"
            f"  假设: {self.hypothesis}\n"
            f"  真实效应(仅仿真已知): {self.true_lift:+.2f}/天\n"
            f"  样本: {c.n_treatment:,} / {c.n_control:,}\n"
            f"  --- 默认口径：CUPED ---\n"
            f"    rho(pre,post)={self.fit.correlation:.4f}  theta={self.fit.theta:.4f}  "
            f"方差缩减 {self.fit.variance_reduction:.2%}（等效样本量 x"
            f"{self.fit.effective_sample_multiplier:.2f}）\n"
            f"    效应 {c.absolute_effect:+.4f} ({c.relative_effect * 100:+.2f}%)  "
            f"SE {c.std_error:.4f}\n"
            f"    95% CI [{c.ci_low:.4f}, {c.ci_high:.4f}]   p = {c.p_value:.4g}   "
            f"significant = {c.significant}\n"
            f"  --- 对照口径：post-only ---\n"
            f"    效应 {n.absolute_effect:+.4f} ({n.relative_effect * 100:+.2f}%)  "
            f"SE {n.std_error:.4f}\n"
            f"    95% CI [{n.ci_low:.4f}, {n.ci_high:.4f}]   p = {n.p_value:.4g}   "
            f"significant = {n.significant}\n"
            f"    p 值变化（CUPED 相对 post-only，log10 倍数）{self.p_value_shift:+.2f}\n"
            f"  SRM: chi2={self.sql_chi2:.4f} (df={self.srm_degrees_of_freedom}), "
            f"p={self.srm_p_value:.4g} -> {srm_tag}\n"
            f"  SQL/Python SRM 交叉验证: {agree}\n"
        )


def _stats_from_ads_row(row) -> AggregateStats:
    """把 ADS 的一行还原成可加统计量。

    这就是"数仓只输出充分统计量"的价值：下游不需要任何明细。
    """
    return AggregateStats.from_sums(
        n=int(row["user_cnt"]),
        sum_x=float(row["pre_sum"]),
        sum_y=float(row["post_sum"]),
        sum_xx=float(row["pre_sq_sum"]),
        sum_yy=float(row["post_sq_sum"]),
        sum_xy=float(row["pre_post_cross_sum"]),
    )


def analyse_ads(
    con: duckdb.DuckDBPyConnection,
    *,
    metric: str = "post_metric_14d",
    alpha: float = 0.05,
) -> list[ExperimentAnalysis]:
    """把 ADS 结果表接进推断层，产出每个实验的结论。

    **只读 ADS 的汇总统计量，不拉明细** —— 这正是数仓分层的目的。
    M1 之后默认输出 CUPED 校正后的结论，同时保留 post-only 做对照。
    """
    result = load_ads_result(con)
    srm = con.execute("SELECT * FROM ads_experiment_srm").df()

    analyses: list[ExperimentAnalysis] = []
    for experiment, group in result.groupby("experiment", sort=True):
        by_variant = {str(row["variant"]): row for _, row in group.iterrows()}
        control, treatment = "control", "treatment"
        if control not in by_variant or treatment not in by_variant:
            raise ValueError(
                f"实验 {experiment} 缺少 control/treatment 分支，实际有 {sorted(by_variant)}"
            )

        c_row, t_row = by_variant[control], by_variant[treatment]
        c_stats = _stats_from_ads_row(c_row)
        t_stats = _stats_from_ads_row(t_row)
        weights = {
            control: float(c_row["design_weight"]),
            treatment: float(t_row["design_weight"]),
        }

        # 对照组口径：不用前置指标
        naive = welch_ttest_from_stats(
            n_treatment=t_stats.n,
            mean_treatment=t_stats.mean_y,
            var_treatment=t_stats.var_y,
            n_control=c_stats.n,
            mean_control=c_stats.mean_y,
            var_control=c_stats.var_y,
            metric=metric,
            variant=treatment,
            control_name=control,
            alpha=alpha,
            expected_weights=weights,
        )

        # 默认口径：CUPED
        cuped, fit = cuped_estimate(
            t_stats,
            c_stats,
            metric=metric,
            variant=treatment,
            control_name=control,
            alpha=alpha,
            expected_weights=weights,
        )

        # 交叉验证：SQL 侧卡方 vs Python 侧卡方
        srm_rows = srm[srm["experiment"] == experiment]
        sql_chi2 = float(srm_rows["chi2_statistic"].iloc[0])
        df_ = int(srm_rows["degrees_of_freedom"].iloc[0])
        python_diag = srm_check(
            {control: c_stats.n, treatment: t_stats.n}, weights
        )

        analyses.append(
            ExperimentAnalysis(
                experiment=str(experiment),
                layer=str(c_row["layer"]),
                hypothesis=str(c_row["hypothesis"]),
                true_lift=float(c_row["true_lift"]),
                naive=naive,
                cuped=cuped,
                fit=fit,
                sql_chi2=sql_chi2,
                python_chi2=float(python_diag.statistic or 0.0),
                srm_p_value=float(stats.chi2.sf(sql_chi2, df_)),
                srm_degrees_of_freedom=df_,
            )
        )

    return analyses


@dataclass
class CrossValidation:
    """ADS 汇总路径 vs DWD 明细路径的交叉验证结果。

    两条路径算的是同一个量，判据是**效应与标准误的绝对差 < 1e-9**。
    不一致就说明有一层的口径写错了，而线上只会跑其中一条 ——
    这正是数仓分层最需要守住的东西。

    这里原本写的是"必须**完全一致**"，实测把它改掉了：同一份代码、同一份 Parquet，
    ADS 侧的 CUPED 效应出现过 ``-3.9243251037`` 与 ``-3.9243251038`` 两种值
    （DWD 侧两次都是 ``…037``）—— 差 1 ulp，来自两侧求和顺序不同，不是口径差异。
    所以判据是"在容差内一致"而不是"逐位相同"，报告也**只印到小数点后 6 位**：
    印 10 位、再按 1e-9 说"一致"，会让读者以为这句话自相矛盾。
    """

    experiment: str
    naive_summary: Estimate
    naive_detail: Estimate
    cuped_summary: Estimate
    cuped_detail: Estimate

    #: 判据的容差。**故意印进报告**（见 ``summary()`` 最后一行）：报告要能被独立读懂，
    #: 读者不该为了知道"一致"是什么意思而回来翻源码。
    TOLERANCE = 1e-9

    @property
    def naive_matches(self) -> bool:
        return self._close(self.naive_summary, self.naive_detail)

    @property
    def cuped_matches(self) -> bool:
        return self._close(self.cuped_summary, self.cuped_detail)

    @staticmethod
    def _close(a: Estimate, b: Estimate) -> bool:
        return (
            abs(a.absolute_effect - b.absolute_effect) < CrossValidation.TOLERANCE
            and abs(a.std_error - b.std_error) < CrossValidation.TOLERANCE
            and abs(a.p_value - b.p_value) < 1e-12
        )

    @property
    def passed(self) -> bool:
        return self.naive_matches and self.cuped_matches

    def summary(self) -> str:
        def row(name: str, s: Estimate, d: Estimate) -> str:
            return (
                f"    {name:<10} ADS汇总 effect={s.absolute_effect:>12.6f} "
                f"se={s.std_error:>10.6f} | DWD明细 effect={d.absolute_effect:>12.6f} "
                f"se={d.std_error:>10.6f} | {'一致' if self._close(s, d) else '不一致 <<<'}"
            )

        return "\n".join(
            [
                f"  {self.experiment}",
                row("post-only", self.naive_summary, self.naive_detail),
                row("CUPED", self.cuped_summary, self.cuped_detail),
                f"    判据：|Δ效应|、|Δ标准误| < {self.TOLERANCE:g}"
                f"（上面只印到小数点后 6 位，比判据粗 3 个数量级）",
            ]
        )


def verify_against_detail(
    con: duckdb.DuckDBPyConnection,
    analyses: list[ExperimentAnalysis],
    experiment: str,
) -> CrossValidation:
    """交叉验证：ADS 汇总路径 vs DWD 明细路径。

    post-only 与 CUPED 两条路都要对得上（判据与理由见 ``CrossValidation``）。
    CUPED 尤其关键 —— 它的 theta 来自 ADS 里的 ``pre_post_cross_sum``，
    如果那一列在 SQL 里写错，只有这条交叉验证能发现。
    """
    detail = con.execute(
        """
        SELECT variant, pre_metric, post_metric
        FROM dwd_experiment_user
        WHERE experiment = ?
        """,
        [experiment],
    ).df()

    t = detail[detail["variant"] == "treatment"]
    c = detail[detail["variant"] == "control"]
    if t.empty or c.empty:
        raise ValueError(f"实验 {experiment} 在 DWD 里缺少某一臂")

    analysis = next(a for a in analyses if a.experiment == experiment)

    naive_detail = welch_ttest(
        t["post_metric"].to_numpy(),
        c["post_metric"].to_numpy(),
        metric="post_metric_14d",
    )
    cuped_detail, _fit = cuped_ttest(
        t["pre_metric"].to_numpy(),
        t["post_metric"].to_numpy(),
        c["pre_metric"].to_numpy(),
        c["post_metric"].to_numpy(),
        metric="post_metric_14d",
    )

    return CrossValidation(
        experiment=experiment,
        naive_summary=analysis.naive,
        naive_detail=naive_detail,
        cuped_summary=analysis.cuped,
        cuped_detail=cuped_detail,
    )


def render_report(analyses: list[ExperimentAnalysis]) -> str:
    """输出人类可读的实验结论。"""
    header = "=" * 72
    body = [header, "数仓链路 → 推断层：实验结果", header]
    for a in analyses:
        body.append(a.summary().rstrip())
    return "\n".join(body)


# --------------------------------------------------------------------------- #
# 协变量失衡诊断（M0 → M1 的衔接点）
# --------------------------------------------------------------------------- #
@dataclass
class CovariateAdjustmentReport:
    """前置协变量失衡造成的偏置，以及 CUPED 能拿回多少。

    这是 M0 最重要的一条发现，也是 M1 把 CUPED 做成默认口径的理由。

    ``theta`` 用的是**合并样本**估出来的 CUPED 系数 —— 也就是线上真正会用的那个；
    ``theta_from_gaps`` 是从重新分流的差距分布里反推的系数，作为交叉验证。
    """

    experiment: str
    n_users: int
    n_trials: int
    correlation: float

    realized_pre_gap: float
    realized_post_gap: float
    realized_adjusted_gap: float

    theta: float
    theta_from_gaps: float

    sd_post_gap: float
    sd_adjusted_gap: float

    #: 实测去掉的方差比例（把真实的 theta 应用到重新分流分布上）
    variance_reduction: float
    #: 理论去掉的方差比例 = rho^2；**残余方差**才是 1 - rho^2
    theoretical_variance_reduction: float

    @property
    def remaining_variance_fraction(self) -> float:
        """校正后残余的方差比例 = 1 - rho^2。"""
        return 1.0 - self.correlation**2

    @property
    def se_shrinkage(self) -> float:
        """标准误的降幅 = 1 - sqrt(1 - rho^2)。别和方差缩减混用。"""
        return 1.0 - np.sqrt(self.remaining_variance_fraction)

    @property
    def effective_sample_multiplier(self) -> float:
        """等效样本量倍数 = 1 / (1 - rho^2)。"""
        return 1.0 / self.remaining_variance_fraction

    @property
    def bias_removed(self) -> float:
        """后置差距里，被前置协变量校正掉的比例。"""
        if self.realized_post_gap == 0:
            return float("nan")
        return (self.realized_post_gap - self.realized_adjusted_gap) / self.realized_post_gap

    def summary(self) -> str:
        sign = "+" if self.realized_post_gap >= 0 else ""
        return (
            f"\n协变量失衡诊断: {self.experiment}  (n={self.n_users:,})\n"
            f"  corr(pre, post) = {self.correlation:.4f}\n"
            f"    方差缩减 = rho^2 = {self.theoretical_variance_reduction:.4f}"
            f"   （残余方差 1-rho^2 = {self.remaining_variance_fraction:.4f}，"
            f"标准误降幅 {self.se_shrinkage:.4f}，等效样本量 x"
            f"{self.effective_sample_multiplier:.2f}）\n"
            f"  这一次实现的分流:\n"
            f"    前置组间差   = {self.realized_pre_gap:+.4f}\n"
            f"    后置组间差   = {sign}{self.realized_post_gap:.4f}   <- 未经校正的结论\n"
            f"    校正后差距   = {self.realized_adjusted_gap:+.4f}   "
            f"（偏置被消掉 {self.bias_removed:.1%}）\n"
            f"  重新分流 {self.n_trials} 次得到的分布:\n"
            f"    theta = {self.theta:.4f}（合并样本估计，线上实际使用的值）\n"
            f"    theta（参考） = {self.theta_from_gaps:.4f}"
            " —— 从差距分布反推，比合并样本估计更嘈杂，且系统性略低\n"
            "      （差距分布的相关系数比个体级相关略小，属于二阶效应）\n"
            f"    校正前 sd = {self.sd_post_gap:.4f}  ->  校正后 sd = {self.sd_adjusted_gap:.4f}\n"
            f"    实测方差缩减 = {self.variance_reduction:.4f}  "
            f"(理论 rho^2 = {self.theoretical_variance_reduction:.4f})\n"
        )


def covariate_adjustment_report(
    con: duckdb.DuckDBPyConnection,
    experiment: str,
    *,
    n_trials: int = 300,
    seed: int = 0,
) -> CovariateAdjustmentReport:
    """量化"前置协变量失衡"对结论的影响，以及 CUPED 的收益。

    做法：对同一批用户做 ``n_trials`` 次**重新分流**，看效应估计的分布；
    再用合并样本估出的 CUPED θ 去校正，对比校正前后的波动与偏置。

    ``realized_*`` 三项用的是 DWD 里**实际发生过的那一次**分流，
    也就是分析师真正会看到的那份数据。
    """
    rows = con.execute(
        """
        SELECT user_id, variant, pre_metric, post_metric
        FROM dwd_experiment_user WHERE experiment = ?
        """,
        [experiment],
    ).df()
    if rows.empty:
        raise ValueError(f"实验 {experiment} 在 dwd_experiment_user 中没有数据")

    ids = [str(u) for u in rows["user_id"]]
    pre = rows["pre_metric"].to_numpy(dtype=float)
    post = rows["post_metric"].to_numpy(dtype=float)
    variant = rows["variant"].to_numpy()

    # CUPED 实际使用的 theta：合并样本估计
    fit = fit_cuped(AggregateStats.from_arrays(post, pre))
    theta = fit.theta

    m_t, m_c = variant == "treatment", variant == "control"
    realized_pre = float(pre[m_t].mean() - pre[m_c].mean())
    realized_post = float(post[m_t].mean() - post[m_c].mean())

    rz = Randomizer()
    batcher = KeyBatcher(ids)
    pre_gaps = np.empty(n_trials)
    post_gaps = np.empty(n_trials)

    for k in range(n_trials):
        spec = ExperimentSpec(
            name="covariate_diagnostic",
            variants=(Variant("control", 0.5), Variant("treatment", 0.5)),
            salt=f"cov_diag_{seed}_{k}",
        )
        codes = rz.assign_codes(ids, spec, batcher)
        a, b = codes == 1, codes == 0
        pre_gaps[k] = pre[a].mean() - pre[b].mean()
        post_gaps[k] = post[a].mean() - post[b].mean()

    # 交叉验证：从差距分布反推的 theta（应当接近合并样本估出的值）
    theta_from_gaps = float(
        np.cov(post_gaps, pre_gaps, ddof=1)[0, 1] / np.var(pre_gaps, ddof=1)
    )

    # 用**真实的 theta** 去校正，而不是从差距分布反推的那个
    adjusted = post_gaps - theta * pre_gaps
    sd_post = float(post_gaps.std(ddof=1))
    sd_adj = float(adjusted.std(ddof=1))

    return CovariateAdjustmentReport(
        experiment=experiment,
        n_users=len(ids),
        n_trials=n_trials,
        correlation=fit.correlation,
        realized_pre_gap=realized_pre,
        realized_post_gap=realized_post,
        realized_adjusted_gap=float(realized_post - theta * realized_pre),
        theta=theta,
        theta_from_gaps=theta_from_gaps,
        sd_post_gap=sd_post,
        sd_adjusted_gap=sd_adj,
        variance_reduction=1.0 - (sd_adj / sd_post) ** 2,
        theoretical_variance_reduction=fit.variance_reduction,
    )

