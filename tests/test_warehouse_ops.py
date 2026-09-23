"""数仓**运营层**的测试：血缘 / 数据质量 / 新鲜度。

这一层存在的理由：前 11 个 SQL 解决"算得对"，但不回答"这张表从哪来"、
"这一列现在干净吗"、"我读的是现在这份 SQL 建出来的吗"。

测试的重点不是"现在全绿"，而是**检查器真的抓得住**：
血缘解析错了会不会红、主键重复会不会红、SQL 改了没重建会不会红。
没有故障注入，检查器最可能的失效方式是永远绿（决策 54）。
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from ablab.warehouse.build import SQL_ORDER
from ablab.warehouse.ops import (
    DECLARED_LINEAGE,
    TableSpec,
    check_lineage,
    check_manifest,
    check_table,
    manifest_path,
    parse_lineage,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"


class TestLineage:
    def test_the_real_graph_matches_its_declaration(self):
        """真图与声明必须一致，而且执行顺序满足依赖（拓扑）。"""
        assert check_lineage(SQL_DIR, SQL_ORDER, DECLARED_LINEAGE) == []

    def test_views_are_nodes_too(self):
        """ODS/DIM 是**视图**，但它们也是血缘节点。

        第一版解析器只认 ``CREATE OR REPLACE TABLE``，于是 ODS 层整个消失、
        四条边凭空断掉 —— 而缺的恰好是最上游那一层。
        """
        edges = parse_lineage(SQL_DIR)
        views = {e.table for e in edges if e.is_view}
        assert views == {
            "ods_exposure_log",
            "ods_event_log",
            "ods_user_profile",
            "dim_experiment_config",
            "dim_guardrail_config",
        }

    def test_guardrail_layer_starts_from_ods(self):
        """护栏那条链是**从 ODS 起头的第二条入口**（不是读 DWD）。

        这是血缘核对第一次跑纠正我的三处之一：08 路自己重写了一遍"首次曝光"口径，
        因为护栏是别的**事件**，而 01 路的 metric_value 只装了主指标事件。
        """
        edges = {e.table: e for e in parse_lineage(SQL_DIR)}
        assert edges["dws_experiment_guardrail_daily"].reads == (
            "dim_guardrail_config",
            "ods_event_log",
            "ods_exposure_log",
        )

    def test_wrong_order_is_caught(self):
        """**故障注入**：把执行顺序倒过来，拓扑判据必须报出来。"""
        problems = check_lineage(SQL_DIR, tuple(reversed(SQL_ORDER)), DECLARED_LINEAGE)
        assert problems and any("执行顺序违反依赖" in p for p in problems), problems

    def test_missing_and_extra_declarations_are_caught(self):
        declared = dict(DECLARED_LINEAGE)
        declared.pop("ads_experiment_srm")
        declared["ads_does_not_exist"] = ()
        problems = check_lineage(SQL_DIR, SQL_ORDER, declared)
        assert any("没有在 DECLARED_LINEAGE 里声明" in p for p in problems), problems
        assert any("没有 SQL 文件创建它" in p for p in problems), problems

    def test_a_changed_edge_is_caught(self):
        """声明与 SQL 不一致（比如有人悄悄加了一个 JOIN）必须报出来。"""
        declared = dict(DECLARED_LINEAGE)
        declared["dwd_experiment_user"] = ("ods_exposure_log",)
        problems = check_lineage(SQL_DIR, SQL_ORDER, declared)
        assert any("血缘对不上" in p for p in problems), problems

    def test_comments_do_not_create_edges(self):
        """注释里出现的 ``FROM xxx`` 不能变成边 —— 这些 SQL 的注释密度极高。"""
        edges = {e.table: e for e in parse_lineage(SQL_DIR)}
        assert edges["ads_experiment_srm"].reads == ("ads_experiment_result",)


class TestQuality:
    @staticmethod
    def _planted() -> duckdb.DuckDBPyConnection:
        """一张故意写坏的表：重复主键、NULL 必需列、串味枚举、负计数。"""
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE demo (
                experiment VARCHAR, variant VARCHAR, user_cnt BIGINT, note VARCHAR
            )
            """
        )
        con.executemany(
            "INSERT INTO demo VALUES (?, ?, ?, ?)",
            [
                ("e1", "control", 3, "a"),
                ("e1", "control", 1, "重复主键"),
                ("e1", "treatmnet", 2, "枚举串味（拼错）"),
                ("e1", None, -5, "变体缺列 + 负计数"),
            ],
        )
        return con

    def test_each_violation_kind_is_caught(self):
        spec = TableSpec(
            table="demo",
            key=("experiment", "variant"),
            required=("variant", "user_cnt"),
            enums={"variant": ("control", "treatment")},
            non_negative=("user_cnt",),
        )
        con = self._planted()
        try:
            problems, checks = check_table(con, spec)
        finally:
            con.close()
        joined = "\n".join(problems)
        assert "主键 (experiment, variant) 有 1 行重复" in joined, problems
        assert "主键 (experiment, variant) 有 1 行为 NULL" in joined, problems
        assert "demo.variant 有 1 行为 NULL" in joined, problems
        assert "未声明的取值 ['treatmnet']" in joined, problems
        assert "user_cnt 出现负值" in joined, problems
        # 检查条数也要报对：4 类检查（主键算两条）
        assert checks == 6, checks

    def test_a_missing_table_is_reported_not_skipped(self):
        """表不见了要报出来 —— "跳过"会让 0 个问题看起来像"通过"。"""
        con = duckdb.connect(":memory:")
        try:
            problems, _checks = check_table(con, TableSpec(table="nope", key=("a",)))
        finally:
            con.close()
        assert problems and "不存在" in problems[0]

    def test_a_clean_table_passes(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE demo (experiment VARCHAR, variant VARCHAR, user_cnt BIGINT)")
        con.executemany(
            "INSERT INTO demo VALUES (?, ?, ?)",
            [("e1", "control", 3), ("e1", "treatment", 2)],
        )
        try:
            problems, checks = check_table(
                con,
                TableSpec(
                    table="demo",
                    key=("experiment", "variant"),
                    required=("variant", "user_cnt"),
                    enums={"variant": ("control", "treatment")},
                    non_negative=("user_cnt",),
                ),
            )
        finally:
            con.close()
        assert problems == []
        assert checks == 6


class TestFreshness:
    @staticmethod
    def _env(work_dir: Path, rows: int = 2) -> tuple[Path, Path, duckdb.DuckDBPyConnection]:
        """用 ``work_dir`` 而不是 pytest 的 ``tmp_path``。

        **这是被机检抓出来的**：两轮前加的 `tests/test_restricted_env.py` 扫描
        `tmp_path` 与 tempfile 的临时目录 API（受限环境里它们建不出来/清理会被拒），
        我写这个文件时顺手用了 `tmp_path`，于是它当场红了 —— 检查器成立的最好证据
        就是它抓住了写检查器的人。
        """
        sql_dir = work_dir / "sql"
        sql_dir.mkdir()
        (sql_dir / "00_demo.sql").write_text("CREATE OR REPLACE TABLE demo AS SELECT 1;\n", encoding="utf-8")
        db = work_dir / "demo.duckdb"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE demo (a INTEGER)")
        con.executemany("INSERT INTO demo VALUES (?)", [(i,) for i in range(rows)])
        return db, sql_dir, con

    def test_manifest_is_written_and_verified(self, work_dir: Path):
        db, sql_dir, con = self._env(work_dir)
        try:
            path = write_manifest(db, sql_dir, con, ["demo"])
            assert path == manifest_path(db) and path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["row_counts"] == {"demo": 2}
            assert set(data["sql_fingerprints"]) == {"00_demo.sql"}
            assert check_manifest(db, sql_dir, con, ["demo"]) == []
        finally:
            con.close()

    def test_sql_changed_after_the_build_is_caught(self, work_dir: Path):
        """**故障注入**：建完之后改 SQL —— 正是"改了忘重建"那一刻。"""
        db, sql_dir, con = self._env(work_dir)
        try:
            write_manifest(db, sql_dir, con, ["demo"])
            (sql_dir / "00_demo.sql").write_text(
                "CREATE OR REPLACE TABLE demo AS SELECT 2;\n", encoding="utf-8"
            )
            problems = check_manifest(db, sql_dir, con, ["demo"])
            assert any("被改过" in p for p in problems), problems
        finally:
            con.close()

    def test_row_count_drift_is_caught(self, work_dir: Path):
        db, sql_dir, con = self._env(work_dir)
        try:
            write_manifest(db, sql_dir, con, ["demo"])
            con.execute("INSERT INTO demo VALUES (99)")
            problems = check_manifest(db, sql_dir, con, ["demo"])
            assert any("行数变了" in p for p in problems), problems
        finally:
            con.close()

    def test_a_missing_manifest_is_reported(self, work_dir: Path):
        db, sql_dir, con = self._env(work_dir)
        try:
            problems = check_manifest(db, sql_dir, con, ["demo"])
            assert problems and "不是用「带清单」的流程建出来的" in problems[0]
        finally:
            con.close()


class TestRealWarehouse:
    """真库上三件都必须过 —— 前面几条证明"抓得住"，这条证明"现在是干净的"。"""

    @pytest.fixture(scope="class")
    def con(self, project_root: Path):
        db = project_root / "build" / "warehouse.duckdb"
        if not db.exists():  # pragma: no cover - 检查集里 warehouse 步骤先生成它
            pytest.skip("没有 build/warehouse.duckdb：先跑 scripts/run_warehouse.py")
        conn = duckdb.connect(str(db), read_only=True)
        yield conn
        conn.close()

    def test_quality_and_freshness_hold(self, con, project_root: Path):
        from ablab.warehouse.ops import check_quality

        problems, checks = check_quality(con)
        assert problems == [], problems
        assert checks > 90, f"只跑了 {checks} 条检查，规格可能被改坏了"
        edges = parse_lineage(project_root / "sql")
        assert check_manifest(
            project_root / "build" / "warehouse.duckdb",
            project_root / "sql",
            con,
            [e.table for e in edges],
        ) == []
