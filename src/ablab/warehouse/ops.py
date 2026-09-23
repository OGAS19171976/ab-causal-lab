"""数仓的**运营层**：血缘、数据质量测试、新鲜度。

为什么补这一层
--------------
前面 11 个 SQL 文件把"算得对"解决了：口径可复算、三条读取路径逐位一致、
报告里每句话都有出处。但一个真实的数仓还要回答另外三个问题 ——
而它们在 dbt 那类工具里是内置的，本项目只有一个 DuckDB 文件加一叠 SQL：

  1. **这张表从哪来？**（血缘）—— 出了口径问题，第一个要问的就是它；
  2. **这一列现在干净吗？**（数据质量测试）—— 主键重复、枚举串味、
     计数为负，这些不会让 SQL 报错，只会让下游的数字悄悄错；
  3. **我读的这份数据，是用仓库里现在这份 SQL 建出来的吗？**（新鲜度）——
     改了 SQL 忘了重建，是最容易发生、也最难察觉的一类事故。

三条设计选择，与有没有 dbt 无关
--------------------------------
* **血缘靠解析，不靠手写**：手写的血缘会漂（这个仓库栽过的正是"声明没人对"）。
  解析只做一件事 —— 找出每个 ``CREATE OR REPLACE TABLE <t>`` 之后引用了哪些表；
  它不打算成为 SQL parser，它只打算**与人工声明的那张表对得上**（对不上就红）。
  只认"由本项目 SQL 创建的表"：Parquet 与 ``read_parquet(...)`` 这类函数不算边。
* **新鲜度就是指纹**：建库时把每个 SQL 文件的 sha256 与每张表的行数写进
  ``<db>.manifest.json``；读之前核对。于是"改了 SQL 没重建"与"重建后行数变了"
  都会被抓到 —— 这两种情况都会让人读到**不是他以为的那份数据**。
* **质量测试宁可少而硬**：主键唯一非空、必需列非空、枚举合法、计数非负，
  外加三条**跨层一致性**（DWD → DWS → ADS 逐层对得上）——
  后三条才是数仓里最值钱的测试，因为它们是"分层没写歪"的直接证据。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

__all__ = [
    "DECLARED_LINEAGE",
    "TABLE_SPECS",
    "BuildManifest",
    "LineageEdge",
    "TableSpec",
    "check_cross_layer",
    "check_lineage",
    "check_manifest",
    "check_quality",
    "check_table",
    "manifest_path",
    "parse_lineage",
    "write_manifest",
]


# --------------------------------------------------------------------------- #
# 1. 血缘
# --------------------------------------------------------------------------- #
#: 人工声明的血缘：``表 -> 它直接读的表``。**解析结果必须与它一致**。
#:
#: 为什么还要手写一份（既然有解析）：解析证明"SQL 里真是这么读的"，
#: 声明证明"这是**有意**的读法"。两边都要有 —— 只靠解析，
#: 一次不小心的 ``JOIN`` 会被当成设计；只靠声明，声明本身会漂。
#:
#: **这份声明第一次跑就被解析器纠正了四处**（留在这里当证据）：
#:   1. ODS 与 DIM 是 ``VIEW`` 不是表（00 路不落存储、直接读 Parquet）；
#:   2. ``ads_experiment_srm`` 读的是 **ADS 结果表**（设计权重已经在 03 里 join 过），
#:      不是我写的"读 DWD 明细"；
#:   3. ``dws_experiment_guardrail_daily`` 读的是 **ODS 三张视图**，
#:      不是 DWD —— 因为护栏是别的**事件**（延迟/崩溃），而 01 路的 metric_value
#:      只装了主指标事件。它自己重写了一遍"首次曝光"口径，
#:      这是有意为之（08 的注释写着"与 01 路同一套口径"），
#:      但**血缘上它是一条独立的入口**，这一点以前没人写下来过；
#:   4. ``ads_experiment_guardrail_result`` 不读 DIM —— 名单在 08 路已经 join 完了。
DECLARED_LINEAGE: dict[str, tuple[str, ...]] = {
    # ODS 与 DIM：源是 Parquet 文件（不是表），所以这些节点没有上游边
    "ods_exposure_log": (),
    "ods_event_log": (),
    "ods_user_profile": (),
    "dim_experiment_config": (),
    "dim_guardrail_config": (),
    # DWD：首次曝光 × 事件窗口
    "dwd_experiment_user": ("ods_exposure_log", "ods_event_log", "ods_user_profile"),
    # 主链路
    "dws_experiment_variant_daily": ("dwd_experiment_user",),
    "ads_experiment_result": ("dws_experiment_variant_daily", "dim_experiment_config"),
    "ads_experiment_srm": ("ads_experiment_result",),
    # 旁路一：换个分析单元（簇）
    "dws_experiment_cluster_daily": ("dwd_experiment_user",),
    # 旁路二：比值口径
    "dws_experiment_ratio_daily": ("dwd_experiment_user",),
    "ads_experiment_ratio_result": ("dws_experiment_ratio_daily",),
    # 旁路三：护栏（**从 ODS 起头的第二条入口**）
    "dws_experiment_guardrail_daily": (
        "ods_exposure_log",
        "ods_event_log",
        "dim_guardrail_config",
    ),
    "ads_experiment_guardrail_result": ("dws_experiment_guardrail_daily",),
    # 旁路四：多协变量 CUPED 的二阶矩
    "dws_experiment_covariate_daily": ("dwd_experiment_user",),
    "ads_experiment_covariate_result": ("dws_experiment_covariate_daily",),
}

_CREATE_RE = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+(TABLE|VIEW)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I
)
_SOURCE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)


def _strip_comments(sql: str) -> str:
    """去掉 ``--`` 行注释再做词法扫描。

    这一步不是洁癖：这些 SQL 的注释里大量出现"FROM xxx 会把 yyy 加进来"这类
    解释性句子，不剥掉就会**解析出并不存在的边**（而这个仓库的注释密度是最高的）。
    """
    return "\n".join(line.split("--")[0] for line in sql.splitlines())


@dataclass(frozen=True)
class LineageEdge:
    """一个 SQL 文件的产出节点与它直接读的表。"""

    sql_file: str
    table: str
    reads: tuple[str, ...]
    #: 是视图还是表。**ODS/DIM 是视图**（00 路不落存储、直接读 Parquet）——
    #: 这一点以前没写下来过，而它决定了"这一层能不能被回填"。
    is_view: bool = False


def parse_lineage(sql_dir: str | Path) -> tuple[LineageEdge, ...]:
    """把 ``sql/`` 解析成血缘边（按文件名排序，结果稳定）。

    只做两件事：认出 ``CREATE OR REPLACE TABLE|VIEW <t>``，再找它后面引用的表名。
    引用的名字**只有落在"本项目创建过的节点"里才算一条边** ——
    Parquet 路径与 ``read_parquet(...)`` 这类函数调用不是边，也不该被当成边。
    """
    files = sorted(Path(sql_dir).glob("*.sql"))
    created: dict[str, str] = {}
    raw: list[tuple[str, str, bool, list[str]]] = []
    for path in files:
        body = _strip_comments(path.read_text(encoding="utf-8"))
        # **一个文件里可能建多个节点**：00 路就建了 5 张视图。
        # 第一版只看第一个 CREATE，于是另外四个节点"没有被任何 SQL 创建"——
        # 血缘图缺一半，而缺的那一半恰好是 ODS 层。
        matches = list(_CREATE_RE.finditer(body))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            is_view = match.group(1).upper() == "VIEW"
            table = match.group(2).lower()
            created[table] = path.name
            raw.append((path.name, table, is_view, _SOURCE_RE.findall(body[match.end():end])))
    known = set(created)
    return tuple(
        LineageEdge(
            sql_file=name,
            table=table,
            reads=tuple(
                sorted(
                    {
                        s.lower()
                        for s in sources
                        if s.lower() in known and s.lower() != table
                    }
                )
            ),
            is_view=is_view,
        )
        for name, table, is_view, sources in raw
    )


def check_lineage(
    sql_dir: str | Path,
    order: Sequence[str],
    declared: Mapping[str, Sequence[str]] = DECLARED_LINEAGE,
) -> list[str]:
    """血缘三条判据：表齐、边对得上、**执行顺序满足依赖**（拓扑）。"""
    problems: list[str] = []
    edges = parse_lineage(sql_dir)
    parsed = {e.table: set(e.reads) for e in edges}
    if len(parsed) != len(edges):  # pragma: no cover - 需要两张同名表
        problems.append("同一个表名被多个 SQL 文件创建")

    for table in sorted(set(parsed) - set(declared)):
        problems.append(f"{table} 没有在 DECLARED_LINEAGE 里声明 —— 新表要显式写下来源")
    for table in sorted(set(declared) - set(parsed)):
        problems.append(f"{table} 在声明里，但没有 SQL 文件创建它 —— 是删了还是改了名？")
    for table in sorted(set(parsed) & set(declared)):
        if parsed[table] != set(declared[table]):
            problems.append(
                f"{table} 的血缘对不上：SQL 里读 {sorted(parsed[table])}，"
                f"声明说读 {sorted(declared[table])}"
            )

    # 拓扑：每个源表的**产出文件**必须排在读它的文件之前
    producer = {e.table: e.sql_file for e in edges}
    position = {name: i for i, name in enumerate(order)}
    for edge in edges:
        if edge.sql_file not in position:
            problems.append(f"{edge.sql_file} 不在 SQL_ORDER 里")
            continue
        for source in edge.reads:
            src_file = producer.get(source)
            if src_file is None:
                continue
            if position.get(src_file, 10**6) >= position[edge.sql_file]:
                problems.append(
                    f"执行顺序违反依赖：{edge.sql_file} 读 {source}，"
                    f"而 {source} 由它后面的 {src_file} 产出"
                )
    return problems


# --------------------------------------------------------------------------- #
# 2. 数据质量测试
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableSpec:
    """一张表的质量规格。**只放会让下游数字悄悄错的约束**。"""

    table: str
    #: 主键：既要唯一，也不能有 NULL
    key: tuple[str, ...] = ()
    #: 必需列：不允许 NULL（注意"0"与"缺失"是两件事，这里只挡缺失）
    required: tuple[str, ...] = ()
    #: 枚举列：取值必须落在集合里
    enums: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: 计数类列：不允许为负
    non_negative: tuple[str, ...] = ()


#: 全部规格。**刻意不追求列级全覆盖**：把每列都写一遍会变成一份没人维护的清单，
#: 而真正会咬人的是这几类（主键、缺失、枚举、负数）加上下面的跨层一致性。
#:
#: **非负只用在计数上** —— 这条判据被数据纠正过一次，值得写下来：
#: 第一版把 `pre_metric` / `post_metric` / `sum_y` / `pre_metric_sum` 也标成非负，
#: 一跑就红（最小 −98.9）。而这不是数据脏：主指标在 DGP 里是**高斯连续量**，
#: 值之和当然可以是负的。把它们从规格里去掉，比让门红着更诚实 ——
#: 一条"靠运气通过"的约束（这次恰好是正的）比没有约束更糟，因为它会让人以为
#: 自己被保护着（决策 31 的同一条）。
TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        table="dwd_experiment_user",
        key=("experiment", "user_id"),
        required=("variant", "expose_ds", "pre_metric", "pre_cnt", "post_metric", "post_cnt"),
        enums={"variant": ("control", "treatment")},
        # 只对**计数**要求非负；pre/post_metric 是连续量，可以为负
        non_negative=("pre_cnt", "post_cnt"),
    ),
    TableSpec(
        table="dws_experiment_variant_daily",
        key=("experiment", "variant", "ds"),
        required=("user_cnt", "post_sum", "pre_sum"),
        enums={"variant": ("control", "treatment")},
        non_negative=("user_cnt",),
    ),
    TableSpec(
        table="ads_experiment_result",
        key=("experiment", "variant"),
        required=("user_cnt", "post_sum", "pre_sum", "pre_post_cross_sum"),
        enums={"variant": ("control", "treatment")},
        non_negative=("user_cnt",),
    ),
    TableSpec(
        table="ads_experiment_srm",
        key=("experiment", "variant"),
        required=("experiment", "user_cnt", "chi2_statistic"),
        non_negative=("user_cnt", "chi2_statistic"),
    ),
    TableSpec(
        table="dws_experiment_ratio_daily",
        key=("experiment", "variant", "ds"),
        required=("user_cnt", "sum_y", "sum_x"),
        enums={"variant": ("control", "treatment")},
        non_negative=("user_cnt", "sum_x"),
    ),
    TableSpec(
        table="ads_experiment_ratio_result",
        key=("experiment", "variant"),
        required=("user_cnt", "sum_y", "sum_x"),
        non_negative=("user_cnt", "sum_x"),
    ),
    TableSpec(
        table="dws_experiment_covariate_daily",
        key=("experiment", "variant", "ds"),
        required=(
            "user_cnt",
            "pre_metric_sum",
            "pre_cnt_sum",
            "pre_metric_pre_cnt_cross_sum",
            "post_metric_sum",
            "post_metric_sq_sum",
        ),
        non_negative=("user_cnt", "pre_cnt_sum"),
    ),
    TableSpec(
        table="ads_experiment_covariate_result",
        key=("experiment", "variant"),
        required=("user_cnt", "pre_metric_sum", "pre_cnt_sum", "post_metric_sq_sum"),
        non_negative=("user_cnt", "pre_cnt_sum"),
    ),
    TableSpec(
        table="dws_experiment_cluster_daily",
        key=("experiment", "variant", "ds", "cluster_id"),
        non_negative=("user_cnt",),
    ),
    TableSpec(
        table="dws_experiment_guardrail_daily",
        # 长表：事件名在这一层叫 guardrail（08 路把它当"被声明的护栏名"用）
        key=("experiment", "variant", "ds", "guardrail"),
        required=("user_cnt", "value_sum"),
        non_negative=("user_cnt",),
    ),
    TableSpec(
        table="ads_experiment_guardrail_result",
        key=("experiment", "guardrail", "variant"),
        required=("user_cnt", "value_sum"),
        non_negative=("user_cnt",),
    ),
)

#: 跨层一致性：``(明细层, 汇总层, 汇总层的可加列 -> 明细层的可加列)``。
#:
#: 这三条是这一层最值钱的测试：它们直接证明"分层没写歪"。
#: 比"每个列都不为 NULL"强得多 —— 分层写歪时，列当然都非空。
_CONSISTENCY: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "dwd_experiment_user",
        "ads_experiment_result",
        (
            ("user_cnt", "COUNT(DISTINCT user_id)"),
            ("post_sum", "SUM(post_metric)"),
            ("pre_sum", "SUM(pre_metric)"),
        ),
    ),
    (
        "dws_experiment_variant_daily",
        "ads_experiment_result",
        (
            ("user_cnt", "SUM(user_cnt)"),
            ("post_sum", "SUM(post_sum)"),
            ("pre_post_cross_sum", "SUM(pre_post_cross_sum)"),
        ),
    ),
    (
        "dws_experiment_ratio_daily",
        "ads_experiment_ratio_result",
        (("sum_y", "SUM(sum_y)"), ("sum_x", "SUM(sum_x)")),
    ),
    (
        "dws_experiment_covariate_daily",
        "ads_experiment_covariate_result",
        (
            ("user_cnt", "SUM(user_cnt)"),
            ("pre_metric_post_cross_sum", "SUM(pre_metric_post_cross_sum)"),
            ("post_metric_sq_sum", "SUM(post_metric_sq_sum)"),
        ),
    ),
)

#: 浮点比较的相对容差。这些是"同一个量、两条路径算两遍"，
#: 差异只应来自求和的结合律（实测 1e-16 量级）。
_REL_TOL = 1e-9


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    row = con.execute(sql).fetchone()
    assert row is not None  # 聚合查询永远有一行
    return row[0]


def check_table(con: duckdb.DuckDBPyConnection, spec: TableSpec) -> tuple[list[str], int]:
    """一张表的四类检查：存在 / 主键 / 必需列 / 枚举 / 非负。

    单独一个函数（而不是塞进 ``check_quality`` 的循环里）是为了让**故障注入**
    能针对一张表做：测试建一张故意写坏的小表，喂一个规格进来，断言四类都被抓到。
    没有这种测试，检查器最可能的失效方式是"永远绿"（决策 54）。
    """
    problems: list[str] = []
    checks = 0
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if spec.table not in tables:
        return [f"{spec.table} 不存在 —— 血缘里声明了它，SQL 却没建出来"], 1

    if spec.key:
        keys = ", ".join(spec.key)
        nulls = " + ".join(f"CASE WHEN {k} IS NULL THEN 1 ELSE 0 END" for k in spec.key)
        row = con.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT ({keys})), SUM({nulls}) FROM {spec.table}"
        ).fetchone()
        # 聚合查询永远有一行；mypy 不认这件事，所以显式断言（与仓库其它地方一致）
        assert row is not None
        dup, null_cnt = row
        checks += 2
        if dup:
            problems.append(f"{spec.table} 主键 ({keys}) 有 {dup} 行重复")
        if null_cnt:
            problems.append(f"{spec.table} 主键 ({keys}) 有 {null_cnt} 行为 NULL")
    for column in spec.required:
        checks += 1
        missing = _scalar(con, f"SELECT COUNT(*) FROM {spec.table} WHERE {column} IS NULL")
        if missing:
            problems.append(f"{spec.table}.{column} 有 {missing} 行为 NULL")
    for column, allowed in spec.enums.items():
        checks += 1
        values = ", ".join(f"'{v}'" for v in allowed)
        bad = con.execute(
            f"SELECT DISTINCT {column} FROM {spec.table} "
            f"WHERE {column} IS NOT NULL AND {column} NOT IN ({values})"
        ).fetchall()
        if bad:
            problems.append(f"{spec.table}.{column} 出现未声明的取值 {[r[0] for r in bad]}")
    for column in spec.non_negative:
        checks += 1
        worst = _scalar(con, f"SELECT MIN({column}) FROM {spec.table}")
        if worst is not None and float(worst) < 0:
            problems.append(f"{spec.table}.{column} 出现负值（最小 {worst}）")
    return problems, checks


def check_cross_layer(con: duckdb.DuckDBPyConnection) -> tuple[list[str], int]:
    """跨层一致性：汇总层的每个可加量都必须等于明细层的对应汇总。

    这是整个运营层最值钱的检查：主键不重复只说明"键写对了"，
    而这一条说明**分层没写歪**。分层写歪时，所有列级约束都会通过。
    """
    problems: list[str] = []
    checks = 0
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for detail, summary, columns in _CONSISTENCY:
        if detail not in tables or summary not in tables:  # pragma: no cover - 结构测试挡着
            continue
        detail_sql = ", ".join(f"{expr} AS {name}" for name, expr in columns)
        detail_rows = {
            (str(r[0]), str(r[1])): r[2:]
            for r in con.execute(
                f"SELECT experiment, variant, {detail_sql} FROM {detail} GROUP BY 1, 2"
            ).fetchall()
        }
        summary_cols = ", ".join(name for name, _ in columns)
        for row in con.execute(
            f"SELECT experiment, variant, {summary_cols} FROM {summary}"
        ).fetchall():
            checks += 1
            key = (str(row[0]), str(row[1]))
            want = detail_rows.get(key)
            if want is None:
                problems.append(f"{summary} 里有 {key}，而 {detail} 里没有")
                continue
            for i, (name, _expr) in enumerate(columns):
                got, expected = float(row[2 + i]), float(want[i])
                scale = max(abs(got), abs(expected), 1.0)
                if abs(got - expected) > _REL_TOL * scale:
                    problems.append(
                        f"{summary}.{name} ({key}) = {got!r}，而 {detail} 汇总 = {expected!r}"
                    )
    return problems, checks


def check_quality(con: duckdb.DuckDBPyConnection) -> tuple[list[str], int]:
    """跑完所有质量测试，返回 ``(问题列表, 检查条数)``。

    检查条数要报出来 —— "0 个问题"在一个**根本没跑**的检查器上也成立，
    这与本仓库对"永远绿"的警惕是同一条（决策 54）。
    """
    problems: list[str] = []
    checks = 0
    for spec in TABLE_SPECS:
        spec_problems, spec_checks = check_table(con, spec)
        problems += spec_problems
        checks += spec_checks
    cross_problems, cross_checks = check_cross_layer(con)
    return problems + cross_problems, checks + cross_checks


# --------------------------------------------------------------------------- #
# 3. 新鲜度（指纹）
# --------------------------------------------------------------------------- #
def manifest_path(db_path: str | Path) -> Path:
    """清单文件放在**它描述的那个库旁边**（两个库：合成 / 真实）。"""
    return Path(str(db_path) + ".manifest.json")


@dataclass(frozen=True)
class BuildManifest:
    """一次建库留下的指纹：SQL 文件的 sha256 + 每张表的行数。"""

    sql_fingerprints: dict[str, str]
    row_counts: dict[str, int]
    written_at: str


def _fingerprints(sql_dir: str | Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(sql_dir).glob("*.sql"))
    }


def _row_counts(con: duckdb.DuckDBPyConnection, tables: Sequence[str]) -> dict[str, int]:
    present = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    return {
        name: int(_scalar(con, f"SELECT COUNT(*) FROM {name}"))
        for name in tables
        if name in present
    }


def write_manifest(
    db_path: str | Path,
    sql_dir: str | Path,
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[str],
) -> Path:
    """建库之后写清单。**放在 SQL 跑完之后**：它描述的是"这次建出来的是什么"。"""
    manifest = BuildManifest(
        sql_fingerprints=_fingerprints(sql_dir),
        row_counts=_row_counts(con, tables),
        # 时间戳只给人看：它进不了任何断言（否则报告就不可复现了）
        written_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    path = manifest_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sql_fingerprints": manifest.sql_fingerprints,
                "row_counts": manifest.row_counts,
                "written_at": manifest.written_at,
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def check_manifest(
    db_path: str | Path,
    sql_dir: str | Path,
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[str],
) -> list[str]:
    """新鲜度三条判据：清单在、SQL 没变、行数没变。"""
    path = manifest_path(db_path)
    if not path.exists():
        return [
            f"没有 {path.name} —— 这份库不是用「带清单」的流程建出来的，"
            "无法判断它对应哪一版 SQL。重跑一次 build_warehouse 即可"
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    current = _fingerprints(sql_dir)
    recorded = dict(data.get("sql_fingerprints", {}))
    changed = sorted(
        name
        for name in set(current) | set(recorded)
        if current.get(name) != recorded.get(name)
    )
    if changed:
        problems.append(
            f"SQL 在建库之后被改过（指纹不同）：{changed} —— "
            "现在读到的是**旧数据配新 SQL**，重跑 build_warehouse"
        )

    now_counts = _row_counts(con, tables)
    was_counts = {k: int(v) for k, v in dict(data.get("row_counts", {})).items()}
    for name in sorted(now_counts):
        if name not in was_counts:  # pragma: no cover - 新表会在更改指纹时一起报
            problems.append(f"{name} 在清单里没有行数记录")
        elif now_counts[name] != was_counts[name]:
            problems.append(
                f"{name} 行数变了：清单 {was_counts[name]} -> 现在 {now_counts[name]}"
            )
    return problems
