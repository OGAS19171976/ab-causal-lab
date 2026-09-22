#!/usr/bin/env python
"""**真实数据的门**：契约校验 + 反冒充 + 真接入。

这个脚本回答一个以前无法核对的问题：**到底有没有接入过外部数据？**
做法不是"看目录里有没有文件"（那太容易自欺），而是分三层查：

  1. **契约**：`data/real/provenance.json` 必须有 `source / exported_at /
     external_generator / experiments / tables`，且实验声明合法
     （权重和为 1 —— 口径必须来自声明，不能从观测数据反推）；
  2. **字节与声明一致**：每张表的 `sha256` 与行数必须和磁盘上的文件对得上；
  3. **反冒充**（重点）：`external_generator` 必须为 true，且源目录里
     **不许**出现本仓库合成器的痕迹（`.generated` 标记、`config_fingerprint` 列）。
     没有这一条，"合成数据 + 一份手写 provenance"就能把这句话骗过去。

三层都过了才**真接入**：调用 `warehouse.ingest.load_real_traffic` 归一化到与
合成器相同的落地布局，再调 `warehouse.build.build_warehouse(generate=False)`
跑**同一套 SQL**（下游一行不改 —— 这就是"换数据源不改链路"的可执行版本），
把行数、日期范围与实验清单打出来。

目录为空时它**不算失败**：那正是 `unimplemented.py` 里那条机检项
（`kind="file_absent"`，target 就是这个 provenance 文件）成立的条件，
脚本会把这句话显式打出来，而不是静默返回。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"
PROVENANCE = DATA / "provenance.json"
TARGET = ROOT / "build" / "real_traffic"
SQL_DIR = ROOT / "sql"
DB_PATH = ROOT / "build" / "warehouse_real.duckdb"

#: 每张表的**必需列**。
#:
#: 第一版契约把 ``reg_ds`` 写成"可选"——因为 ``_normalize_profile`` 对它是
#: `if "reg_ds" in out.columns` 的写法。结果端到端测试里 SQL 直接
#: `KeyError: "['reg_ds'] not in index"`：**归一化器认为可选，链路认为必需**。
#: 契约要从**链路**写，不是从某一段代码写 —— 所以现在在这里拦，报清楚哪张表缺哪列。
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "exposure_log": ("ds", "ts", "user_id", "experiment", "variant"),
    "event_log": ("ds", "user_id", "event_name", "metric_value"),
    "user_profile": ("user_id", "reg_ds"),
}

#: 本仓库合成器留下的痕迹 —— 出现任何一个就说明数据不是外部的
GENERATOR_MARKERS = (".generated",)
GENERATOR_COLUMNS = ("config_fingerprint",)
SYNTHETIC_SOURCE_HINTS = ("ablab", "generate.py", "synthetic", "合成器")


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(prov: dict, base: pathlib.Path) -> tuple[list[str], list[dict]]:
    """``(问题列表, 表清单)``。表清单只在契约通过时有意义。"""
    problems: list[str] = []
    for key in (
        "source",
        "exported_at",
        "external_generator",
        "experiments",
        "tables",
        "metric_event",
    ):
        if key not in prov:
            problems.append(f"provenance.json 缺字段 {key!r}")
    if problems:
        return problems, []

    source = str(prov["source"])
    if not source.strip():
        problems.append("source 是空的 —— 必须写清数据从哪来")
    low = source.lower()
    for hint in SYNTHETIC_SOURCE_HINTS:
        if hint in low:
            problems.append(
                f"source 里出现 {hint!r} —— 这看起来是合成数据，不是外部来源"
            )
    try:
        date.fromisoformat(str(prov["exported_at"]))
    except ValueError:
        problems.append(f"exported_at={prov['exported_at']!r} 不是 ISO 日期（YYYY-MM-DD）")

    if prov["external_generator"] is not True:
        problems.append(
            "external_generator 不是 true —— 必须显式声明『这份数据不由本仓库生成』"
        )

    experiments = prov["experiments"]
    if not isinstance(experiments, list) or not experiments:
        problems.append("experiments 必须是非空列表（口径要来自声明）")
    else:
        for exp in experiments:
            name = exp.get("name", "?")
            variants = exp.get("variants")
            if not isinstance(variants, dict) or not variants:
                problems.append(f"实验 {name!r} 没有 variants 声明")
                continue
            total = float(sum(variants.values()))
            if abs(total - 1.0) > 1e-6:
                problems.append(
                    f"实验 {name!r} 的设计权重和为 {total:g}，应当为 1"
                )

    tables = prov.get("tables")
    if not isinstance(tables, list) or not tables:
        return problems + ["tables 必须是非空列表"], []

    for table in tables:
        name = table.get("name", "?")
        rel = table.get("file")
        if not rel:
            problems.append(f"表 {name!r} 没有 file 字段")
            continue
        path = base / rel
        if not path.exists():
            problems.append(f"表 {name!r} 声明的文件不存在：{rel}")
            continue
        digest = sha256_of(path)
        if digest != table.get("sha256"):
            problems.append(
                f"表 {name!r} 的 sha256 与声明不一致"
                f"（磁盘 {digest[:12]}… vs 声明 {str(table.get('sha256'))[:12]}…）"
                "—— 声明与数据对不上，说明中间被改过"
            )
        if path.parent.joinpath(*GENERATOR_MARKERS).exists():
            problems.append(
                f"表 {name!r} 所在目录里有 {GENERATOR_MARKERS[0]} 标记"
                "—— **合成数据不能冒充外部数据**"
            )
    return problems, tables


def check_generator_traces(base: pathlib.Path, tables: list[dict]) -> list[str]:
    """翻一遍真实列名：必需列齐不齐、有没有合成器特有的列。"""
    import pandas as pd

    problems: list[str] = []
    for table in tables:
        path = base / str(table.get("file"))
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        required = REQUIRED_COLUMNS.get(str(table.get("name")), ())
        missing = [c for c in required if c not in frame.columns]
        if missing:
            problems.append(
                f"表 {table.get('name')!r} 缺必需列 {missing}（契约见 data/real/README.md）"
                "—— 缺列会在链路的 SQL 里炸，不如在这里说清"
            )
        hits = [c for c in GENERATOR_COLUMNS if c in frame.columns]
        if hits:
            problems.append(
                f"表 {table.get('name')!r} 里出现合成器特有的列 {hits}"
                "—— **合成数据不能冒充外部数据**"
            )
        rows = int(table.get("rows", -1))
        if rows >= 0 and rows != len(frame):
            problems.append(
                f"表 {table.get('name')!r} 声明 {rows} 行，实际 {len(frame)} 行"
            )
    return problems


def run_ingest(prov: dict) -> dict:
    """契约通过后的真接入：归一化 + 跑同一套 SQL。"""
    sys.path.insert(0, str(ROOT / "src"))
    from ablab.warehouse.build import build_warehouse
    from ablab.warehouse.ingest import ExternalExperiment, load_real_traffic

    specs = []
    for exp in prov["experiments"]:
        guardrails = tuple(
            (str(g[0]), str(g[1]), float(g[2])) for g in exp.get("guardrails", [])
        )
        specs.append(
            ExternalExperiment(
                name=str(exp["name"]),
                variants={str(k): float(v) for k, v in exp["variants"].items()},
                control=str(exp.get("control", "control")),
                treatment=str(exp.get("treatment", "treatment")),
                guardrails=guardrails,
            )
        )
    # 度量事件名与护栏事件名必须从 provenance 读：
    # 上一轮的版本写死用 load_real_traffic 的默认值（"interaction"），
    # 那等于"接进来的数据被迫改名"—— 真实数据的度量就叫它自己的名字。
    report = load_real_traffic(
        DATA,
        target_dir=TARGET,
        experiments=specs,
        metric_event=str(prov.get("metric_event", "interaction")),
        guardrail_events=tuple(str(e) for e in prov.get("guardrail_events", ())),
    )
    build_warehouse(DB_PATH, TARGET, SQL_DIR, generate=False, verbose=False)
    return {
        "experiments": [s.name for s in specs],
        "target": str(TARGET),
        "db": str(DB_PATH),
        "ingest_report": report,
    }


def main() -> int:
    print("真实数据的门（契约 + 反冒充 + 真接入）")
    print(f"  目录：{DATA}")
    if not PROVENANCE.exists():
        print()
        print("  当前状态：**没有接入真实数据** —— 目录里没有 provenance.json。")
        print("  这不是失败：unimplemented.py 里那条机检项（kind=file_absent，")
        print("  target=data/real/provenance.json）正是靠『它不存在』成立的；")
        print("  一旦有人接进来，那条检查会红，并逼着 README 改口径。")
        print()
        print("  接进来的做法见 data/real/README.md（三张表的列 + provenance 契约）。")
        return 0

    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    problems, tables = validate(prov, DATA)
    problems += check_generator_traces(DATA, tables)
    if problems:
        print(f"\n**{len(problems)} 条不成立**（契约没过，不会去跑接入）：")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"\n  契约通过：source={prov['source']!r}，"
          f"exported_at={prov['exported_at']}，{len(tables)} 张表，"
          f"{len(prov['experiments'])} 个实验声明")
    result = run_ingest(prov)
    print(f"  真接入完成：归一化 → {result['target']}；SQL → {result['db']}")
    print(f"  实验：{', '.join(result['experiments'])}")
    print("  注意口径：这只能叫「**接入过**外部数据」，不等于在生产流量上验证过 ——")
    print("  差异审计（哪些数字变了、哪些没变）才是这一步真正买到的东西。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
