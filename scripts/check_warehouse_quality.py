#!/usr/bin/env python
"""**数仓运营层的门**：血缘 · 数据质量 · 新鲜度。

为什么要有这一步
----------------
前 11 个 SQL 文件解决的是"算得对"（口径可复算、三条读取路径逐位一致）。
但一个真实的数仓还要回答另外三个问题，而它们在 dbt 那类工具里是内置的：

  1. **这张表从哪来？**（血缘）—— 出了口径问题，第一个要问的就是它；
  2. **这一列现在干净吗？**（质量测试）—— 重复主键、串味的枚举、负的计数
     都不会让 SQL 报错，只会让下游的数字悄悄错；
  3. **我读的这份数据是用现在这份 SQL 建出来的吗？**（新鲜度）——
     "改了 SQL 忘重建"是最常见、也最难察觉的一类事故。

三件事的判据都写在 ``ablab/warehouse/ops.py`` 里，这里只负责跑 + 报读数 + 定退出码。

用法::

    python scripts/check_warehouse_quality.py
    python scripts/check_warehouse_quality.py --db build/warehouse_real.duckdb
    python scripts/check_warehouse_quality.py --list      # 只看规格与声明
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.warehouse.build import SQL_ORDER  # noqa: E402
from ablab.warehouse.ops import (  # noqa: E402
    DECLARED_LINEAGE,
    TABLE_SPECS,
    check_lineage,
    check_manifest,
    check_quality,
    parse_lineage,
)

DEFAULT_DB = ROOT / "build" / "warehouse.duckdb"
SQL_DIR = ROOT / "sql"


def main() -> int:
    ap = argparse.ArgumentParser(description="数仓运营层：血缘 / 质量 / 新鲜度")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="DuckDB 文件路径")
    ap.add_argument("--list", action="store_true", help="只列规格与血缘声明")
    args = ap.parse_args()

    edges = parse_lineage(SQL_DIR)
    if args.list:
        print(f"血缘声明 {len(DECLARED_LINEAGE)} 个节点、{len(edges)} 条边：")
        for edge in edges:
            kind = "VIEW " if edge.is_view else "TABLE"
            print(f"  {edge.sql_file:<46} {kind} {edge.table:<34} -> {list(edge.reads)}")
        print(f"\n质量规格 {len(TABLE_SPECS)} 张表：")
        for spec in TABLE_SPECS:
            print(
                f"  {spec.table:<36} 主键={list(spec.key)} "
                f"必需={len(spec.required)} 列 枚举={list(spec.enums)} "
                f"非负={list(spec.non_negative)}"
            )
        return 0

    db = Path(args.db)
    if not db.exists():
        print(f"**缺数仓文件 {db}** —— 先跑 scripts/run_warehouse.py"
              "（或 check_real_traffic.py 接外部数据）")
        return 1

    problems: list[str] = []

    # ---- 1. 血缘 --------------------------------------------------------- #
    lineage_problems = check_lineage(SQL_DIR, SQL_ORDER, DECLARED_LINEAGE)
    tables = len(edges)
    views = sum(1 for e in edges if e.is_view)
    print(f"一、血缘：{tables} 个节点（其中视图 {views} 个）、{sum(len(e.reads) for e in edges)} 条边")
    print(f"    声明 {len(DECLARED_LINEAGE)} 个节点；执行顺序满足依赖：{not lineage_problems}")
    problems += [f"血缘：{p}" for p in lineage_problems]

    # ---- 2. 质量 --------------------------------------------------------- #
    con = duckdb.connect(str(db), read_only=True)
    try:
        quality_problems, checks = check_quality(con)
        print(f"二、质量：跑了 {checks} 条检查，{len(quality_problems)} 条不成立")
        problems += [f"质量：{p}" for p in quality_problems]

        # ---- 3. 新鲜度 --------------------------------------------------- #
        fresh_problems = check_manifest(db, SQL_DIR, con, [e.table for e in edges])
        print(
            f"三、新鲜度：{'清单与库一致' if not fresh_problems else '对不上'}"
            f"（SQL 指纹 + 各节点行数）"
        )
        problems += [f"新鲜度：{p}" for p in fresh_problems]
    finally:
        con.close()

    if problems:
        print(f"\n**{len(problems)} 条不成立**：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\n血缘、质量、新鲜度三项都成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
