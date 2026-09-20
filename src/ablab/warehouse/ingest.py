"""真实数据入口：把**外部文件**接进同一条 ODS→DWD→DWS→ADS 链路。

为什么需要它
------------
这个仓库的每一层都在主张一件事：**换数据源不该改链路**。
在它之前，这句话只被"合成数据写得像真实数据"支持着 —— 那是主张，不是证据。
真实流量的字段名、类型、多余列、事件名集合都与合成器不同，
所以必须有一个**真实的入口**：读用户给的文件 → 校验 schema → 归一化成
数仓期望的落地布局 → 后面一行 SQL 都不用改。

三件必须做的事
--------------
1. **schema 校验要吵**。少一列就报错，并指名道姓说缺哪一列；
   多余列**丢**掉但记进报告（真实导出总是多几列，静默带着走会让
   下游不知道自己在读什么）。
2. **口径要显式**。主指标事件名、护栏事件名、变体名、设计权重
   都由调用方**声明**，不从数据里反推 —— 反推出来的设计权重会让
   SRM 卡方恒等于 0（这条坑在 00_ods.sql 里写着），而"事件里出现过什么"
   也不等于"哪些是护栏"。
3. **真实数据没有 ``true_lift``**。那一列是仿真专用，外部路径写 NULL：
   报告里"演示真值"这一栏在真实数据上必须是空的，不能假装知道。

质量检查只**报告**不拦路的有：重复曝光（DWD 按 ``MIN(ts)`` 去重）、
未曝光用户的事件（LEFT JOIN 自然处理）、窗口覆盖范围、未声明的事件名。
真正拦路的只有两类：**缺列**与**变体名不在声明里** ——
后者几乎总是数据侧的 bug，静默放行会让实验被"分析"成一个不存在的分支。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "ExternalExperiment",
    "IngestReport",
    "load_real_traffic",
]

#: 每张源表**必须**有的列（下游 SQL 直接按名字取用）。
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "exposure_log": ("ds", "ts", "user_id", "experiment", "variant"),
    "event_log": ("ds", "user_id", "event_name", "metric_value"),
    "user_profile": ("user_id",),
}

#: 每张表我们**会用到**的列；其余列会被丢掉（但记进报告）。
USED_COLUMNS: dict[str, tuple[str, ...]] = {
    "exposure_log": ("ds", "ts", "user_id", "experiment", "variant", "layer"),
    "event_log": ("ds", "user_id", "event_name", "metric_value"),
    "user_profile": ("user_id", "reg_ds", "city"),
}


@dataclass(frozen=True)
class ExternalExperiment:
    """外部数据里一个实验的**声明**（不是从数据里反推的）。"""

    name: str
    #: 变体名 → 设计权重。必须显式给，且权重和应为 1。
    variants: dict[str, float]
    control: str = "control"
    treatment: str = "treatment"
    layer: str = "external"
    hypothesis: str = "外部数据（真实流量）：没有事前声明的假设"
    #: 声明的护栏：(名字, 方向, 最大容忍伤害)。空表示没有护栏。
    guardrails: tuple[tuple[str, str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(f"实验 {self.name!r} 没有声明任何变体")
        total = float(sum(self.variants.values()))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"实验 {self.name!r} 的设计权重之和为 {total:g}，应当为 1 —— "
                "权重必须来自声明，不能从观测数据反推"
            )
        for guard, direction, harm in self.guardrails:
            if direction not in ("lower_is_better", "higher_is_better"):
                raise ValueError(f"护栏 {guard!r} 的方向非法：{direction!r}")
            if harm < 0:
                raise ValueError(f"护栏 {guard!r} 的容忍度不能为负")


@dataclass
class IngestReport:
    """接进来之后到底发生了什么 —— 每一行都要能被复核。"""

    source_dir: str
    target_dir: str
    fmt: str
    row_counts: dict[str, int] = field(default_factory=dict)
    #: 被丢掉的列（真实导出总是多几列）
    dropped_columns: dict[str, list[str]] = field(default_factory=dict)
    #: 数据里出现过、但**没有声明**的事件名 —— 不会被当成指标，但要报出来
    undeclared_events: list[str] = field(default_factory=list)
    declared_events: list[str] = field(default_factory=list)
    #: 重复曝光行数（DWD 会按 MIN(ts) 去重，这里只是告诉你数据长什么样）
    duplicate_exposures: int = 0
    #: 有事件但没有任何曝光的用户数（LEFT JOIN 会自然处理）
    events_without_exposure: int = 0
    date_range: dict[str, tuple[str, str]] = field(default_factory=dict)
    experiments: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"真实数据入口：{self.source_dir} → {self.target_dir}（{self.fmt}）",
        ]
        for table, n in self.row_counts.items():
            rng = self.date_range.get(table)
            span = f"，ds {rng[0]} ~ {rng[1]}" if rng else ""
            lines.append(f"  {table:<14}{n:>10,} 行{span}")
        lines.append(f"  实验（来自声明）：{', '.join(self.experiments)}")
        lines.append(f"  指标事件：{', '.join(self.declared_events)}")
        if self.undeclared_events:
            lines.append(
                f"  **未声明的事件名**（不会被算进任何指标，但要说出来）："
                f"{', '.join(self.undeclared_events)}"
            )
        if self.duplicate_exposures:
            lines.append(
                f"  重复曝光 {self.duplicate_exposures:,} 行"
                "（DWD 按首次曝光去重，这里只是报告数据长什么样）"
            )
        if self.events_without_exposure:
            lines.append(
                f"  有事件但无曝光的用户 {self.events_without_exposure:,} 个"
                "（LEFT JOIN 自然处理）"
            )
        for table, cols in self.dropped_columns.items():
            if cols:
                lines.append(f"  {table} 丢掉的多余列：{', '.join(cols)}")
        for note in self.notes:
            lines.append(f"  · {note}")
        return "\n".join(lines)


def _read(source: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(source)
    if fmt == "csv":
        return pd.read_csv(source)
    raise ValueError(f"fmt 只能是 parquet / csv，收到 {fmt!r}")


def _pick(source_dir: Path, stem: str, fmt: str, explicit: Path | None) -> Path:
    """找到一个源文件：优先用显式给的，否则在目录里按后缀找。"""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"找不到文件：{explicit}")
        return explicit
    cands = sorted(source_dir.glob(f"{stem}*.{fmt}"))
    if not cands:
        raise FileNotFoundError(
            f"在 {source_dir} 里找不到 {stem}*.{fmt} —— 真实数据入口至少要"
            " exposure_log 与 event_log 两张表"
        )
    return cands[0]


def _normalize_exposure(frame: pd.DataFrame, experiments: dict[str, ExternalExperiment]) -> pd.DataFrame:
    out = frame.copy()
    # 未声明的事件名在这里不适用；变体名必须落在声明里
    declared = {v for e in experiments.values() for v in e.variants}
    bad = sorted(set(out["variant"].astype(str)) - declared)
    if bad:
        raise ValueError(
            f"曝光表里出现了没有声明的变体名：{bad}；"
            f"已声明的是 {sorted(declared)} —— "
            "静默放行会让实验被分析成一个不存在的分支"
        )
    unknown_exp = sorted(set(out["experiment"].astype(str)) - set(experiments))
    if unknown_exp:
        raise ValueError(
            f"曝光表里出现了没有声明的实验：{unknown_exp}；"
            "实验口径（设计权重、护栏）必须来自声明"
        )
    out["ds"] = pd.to_datetime(out["ds"]).dt.date
    out["ts"] = pd.to_datetime(out["ts"])
    out["user_id"] = out["user_id"].astype(str)
    out["experiment"] = out["experiment"].astype(str)
    out["variant"] = out["variant"].astype(str)
    if "layer" not in out.columns:
        out["layer"] = [
            experiments[name].layer for name in out["experiment"].astype(str)
        ]
    out["layer"] = out["layer"].astype(str)
    return out


def _normalize_event(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.date
    out["user_id"] = out["user_id"].astype(str)
    out["event_name"] = out["event_name"].astype(str)
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="raise").astype(float)
    return out


def _normalize_profile(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["user_id"] = out["user_id"].astype(str)
    if "reg_ds" in out.columns:
        out["reg_ds"] = pd.to_datetime(out["reg_ds"]).dt.date
    if "city" not in out.columns:
        # 没有城市列 ⇒ 簇级路径不可用。显式补一列占位而不是让下游报 KeyError。
        out["city"] = ""
    return out


def load_real_traffic(
    source_dir: str | Path,
    *,
    target_dir: str | Path,
    experiments: tuple[ExternalExperiment, ...] | list[ExternalExperiment],
    metric_event: str = "interaction",
    guardrail_events: tuple[str, ...] = (),
    fmt: str = "parquet",
    event_file: str | Path | None = None,
    exposure_file: str | Path | None = None,
    profile_file: str | Path | None = None,
) -> IngestReport:
    """把外部文件接成数仓能读的落地布局，返回一份可复核的接入报告。

    归一化后的文件写到 ``target_dir``，目录结构与 ``generate_source_data``
    **完全一致** —— 后面 ``build_warehouse(generate=False)`` 跑同一套 SQL，
    一行都不改。这就是"换数据源不改链路"的可执行版本。
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在：{source_dir}")
    if not experiments:
        raise ValueError("至少要声明一个实验（设计权重与护栏必须来自声明）")
    exp_by_name = {e.name: e for e in experiments}
    if len(exp_by_name) != len(experiments):
        raise ValueError("实验名重复")

    report = IngestReport(
        source_dir=str(source_dir), target_dir=str(target_dir), fmt=fmt
    )
    report.declared_events = [metric_event, *guardrail_events]
    report.experiments = sorted(exp_by_name)

    frames: dict[str, pd.DataFrame] = {}
    for table in ("exposure_log", "event_log", "user_profile"):
        explicit = {
            "exposure_log": exposure_file,
            "event_log": event_file,
            "user_profile": profile_file,
        }[table]
        if table == "user_profile" and explicit is None:
            found = sorted(source_dir.glob(f"user_profile*.{fmt}"))
            if not found:
                report.notes.append(
                    "没有 user_profile（用户维表）—— 城市/注册日期缺失会让"
                    "簇级路径不可用，其余链路不受影响"
                )
                continue
            explicit = found[0]
        path = _pick(source_dir, table, fmt, Path(explicit) if explicit else None)
        raw = _read(path, fmt)

        missing = [c for c in REQUIRED_COLUMNS[table] if c not in raw.columns]
        if missing:
            raise ValueError(
                f"{table}（{path.name}）缺少必需的列：{missing}；"
                f"实际列：{list(raw.columns)}"
            )
        dropped = [c for c in raw.columns if c not in USED_COLUMNS[table]]
        if dropped:
            report.dropped_columns[table] = dropped

        if table == "exposure_log":
            frame = _normalize_exposure(raw, exp_by_name)
        elif table == "event_log":
            frame = _normalize_event(raw)
        else:
            frame = _normalize_profile(raw)
        frames[table] = frame

    if "exposure_log" not in frames or "event_log" not in frames:
        raise ValueError("exposure_log 与 event_log 都是必需的")

    # ---- 质量检查（只报告，不拦路） --------------------------------------- #
    exposure = frames["exposure_log"]
    event = frames["event_log"]
    dup = int(exposure.duplicated(subset=["experiment", "user_id"]).sum())
    report.duplicate_exposures = dup

    declared = set(report.declared_events)
    seen_events = sorted(set(event["event_name"]))
    report.undeclared_events = [name for name in seen_events if name not in declared]
    if report.undeclared_events:
        report.notes.append(
            "未声明的事件名**不会**被任何指标读走（DWD 按 event_name 过滤），"
            "列在这里是为了让你核对它们该不该被声明"
        )
    exposed_users = set(exposure["user_id"])
    report.events_without_exposure = len(set(event["user_id"]) - exposed_users)

    for table, frame in frames.items():
        report.row_counts[table] = int(frame.shape[0])
        if "ds" in frame.columns and frame.shape[0]:
            ds = pd.to_datetime(frame["ds"])
            report.date_range[table] = (str(ds.min().date()), str(ds.max().date()))

    # ---- 落盘：与合成器**完全一致**的目录结构 ----------------------------- #
    for table, frame in frames.items():
        out_dir = target_dir / table
        out_dir.mkdir(parents=True, exist_ok=True)
        frame[list(USED_COLUMNS[table])].to_parquet(
            out_dir / "part-0000.parquet", index=False
        )

    # ---- 维表：设计权重/护栏来自**声明**，true_lift 一律为空 ------------- #
    config_rows = [
        {
            "experiment": exp.name,
            "variant": variant,
            "design_weight": float(weight),
            "layer": exp.layer,
            "true_lift": float("nan"),  # 真实数据没有"演示真值"
            "hypothesis": exp.hypothesis,
        }
        for exp in experiments
        for variant, weight in exp.variants.items()
    ]
    # 目录要先建出来：pandas 不会替你建父目录，而报错信息
    # （"Cannot save file into a non-existent directory"）指向的是路径而不是
    # "你忘了 mkdir" —— 上面三张表用的是同一个模式，这里别漏。
    cfg_dir = target_dir / "experiment_config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(config_rows).to_parquet(cfg_dir / "part-0000.parquet", index=False)
    guard_rows = [
        {
            "experiment": exp.name,
            "guardrail": guard,
            "direction": direction,
            "max_harm": float(harm),
        }
        for exp in experiments
        for guard, direction, harm in exp.guardrails
    ]
    gdir = target_dir / "guardrail_config"
    gdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        guard_rows, columns=["experiment", "guardrail", "direction", "max_harm"]
    ).to_parquet(gdir / "part-0000.parquet", index=False)

    # 外部数据**不写** .generated 指纹：那一层是给合成器判缓存用的。
    # 留一个说明文件，免得下次有人以为这里忘了写。
    (target_dir / "SOURCE.json").write_text(
        json.dumps(
            {
                "kind": "external",
                "source_dir": str(source_dir),
                "fmt": fmt,
                "metric_event": metric_event,
                "guardrail_events": list(guardrail_events),
                "experiments": report.experiments,
                "true_lift": "unavailable（真实数据没有演示真值）",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report.notes.append(
        "写成与合成器一致的落地布局 ⇒ 下游 SQL 一行不用改；"
        "true_lift 全部为空（真实数据没有演示真值）"
    )
    return report


def external_summary_frame(report: IngestReport) -> pd.DataFrame:
    """把报告压成一张小表（给报告脚本用）。"""
    rows = [
        {"表": table, "行数": n} for table, n in report.row_counts.items()
    ]
    for table, cols in report.dropped_columns.items():
        rows.append({"表": f"{table}（丢掉的列）", "行数": len(cols)})
    return pd.DataFrame(rows)
