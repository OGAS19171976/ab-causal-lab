"""生成数仓链路用的仿真源数据，落地成 Parquet。

**为什么要先生成再走 SQL，而不是直接在 SQL 里造数**
    这样"埋点日志 → ODS → DWD → DWS → ADS"的每一层都处理真实形态的数据，
    SQL 里的口径逻辑（首次曝光去重、前后窗口条件聚合、可加性汇总）
    才是真的在解决问题，而不是在表演。

生成数据的结构
--------------
* 每个用户有一个"进入实验系统的日期" ``expose_ds``（同一天进入，便于对齐窗口；
  真实平台上这通常对应一次客户端发版）。
* 用户落在**两个正交层**里的若干个实验中：``ranking`` 层 80% 流量、
  ``recall`` 层 60% 流量。同一个用户可能同时在 0、1、2 个实验里 ——
  这正是分层正交的意义。
* 实验后指标 = 用户潜在水平 + 日噪声 + 真实效应。
  ``exp_rank_v2`` 有 +2.0 的真实效应，``exp_rec_emb`` 真实效应为 0
  （作为"负对照"，用来检验整条链路是否会凭空造出显著结果）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..assignment import ExperimentSpec, Layer, LayerSlot, Randomizer, Variant
from ..hashing import KeyBatcher, murmur3_32

__all__ = [
    "ExperimentDef",
    "WarehouseConfig",
    "DEFAULT_EXPERIMENTS",
    "generate_source_data",
]


@dataclass(frozen=True)
class GuardrailDef:
    """数仓里的一条**护栏声明**（演示真值的一部分）。

    ``harm`` 是注入的**相对伤害**：处置组在这条护栏上真的劣化多少。
    它与 ``ExperimentDef.true_lift`` 同一个性质 —— **仅演示用**：
    真实数仓里没有"真值"这一列，护栏的值来自埋点，不需要注入。
    有了它，"护栏触发 -> 建议停实验"这条链路才能在真实数仓路径上被跑到。
    """

    name: str
    #: ``lower_is_better``（延迟/崩溃率）或 ``higher_is_better``（收入/留存）
    direction: str = "lower_is_better"
    #: 允许的最大相对劣化（如 0.05 = 5%）
    max_harm: float = 0.05
    #: 基线量纲（例如延迟 100ms）。判定只看相对伤害，所以量纲是装饰性的
    baseline: float = 100.0
    #: **注入的相对伤害**（仅演示）。0 表示这条护栏没有劣化
    harm: float = 0.0
    #: 护栏自身的日噪声（相对标准差）
    noise: float = 0.02


@dataclass(frozen=True)
class ExperimentDef:
    """一个实验的分层与分流定义。"""

    name: str
    layer: str
    layer_salt: str
    bucket_start: int
    bucket_end: int
    true_lift: float
    hypothesis: str
    #: 该实验声明的护栏（含**仅演示用**的注入伤害）。空元组表示没有护栏。
    guardrails: tuple[GuardrailDef, ...] = ()
    control: str = "control"
    treatment: str = "treatment"
    #: 设为列名（如 ``"city"``）表示**整簇随机化**：分流在簇级别做，
    #: 同一个簇里的所有用户拿到同一个分支。
    #:
    #: 别小看这一个字段：它决定了"分析单元"是什么。
    #: 若数据其实是人级随机化，却按簇去做簇级检验，虽然算得出一个数，
    #: 却答的是另一个问题（平台侧的 ``_verify_clusters_are_randomized`` 会拦住它）。
    cluster_key: str | None = None


DEFAULT_EXPERIMENTS: tuple[ExperimentDef, ...] = (
    ExperimentDef(
        name="exp_rank_v2",
        layer="ranking",
        layer_salt="layer_ranking",
        bucket_start=0,
        bucket_end=8000,  # 占 80% 流量
        true_lift=2.0,
        hypothesis="新排序模型提升人均互动次数",
        # 排序模型最常见的代价就是延迟：这条护栏注入 +12% 的真实伤害，
        # 于是数仓路径也能演示"护栏触发 -> 建议停实验"（容忍度 5%）
        guardrails=(
            GuardrailDef("latency_p99", "lower_is_better", 0.05,
                         baseline=100.0, harm=0.12),
            GuardrailDef("complaint_rate", "lower_is_better", 0.10,
                         baseline=1.0, harm=0.0, noise=0.05),
        ),
    ),
    ExperimentDef(
        # **簇随机化**实验：分流在**城市**级别做（同一个城市里的用户拿到同一分支）。
        # 为什么仓库里必须有这么一个：没有它，`dws_experiment_cluster_daily`、
        # "簇必须是整簇落在一臂"的闸门、以及簇级 CUPED 这三条链路
        # 在演示里**根本走不到** —— 而走不到的代码等于没有验证过的代码。
        name="exp_city_ctr",
        layer="geo",
        layer_salt="layer_geo",
        bucket_start=0,
        bucket_end=10_000,
        true_lift=1.5,
        hypothesis="城市级投放策略提升人均互动次数（整簇随机化）",
        cluster_key="city",
        # **不在这里声明护栏**，尽管它本来也可以有。原因是一处正在踩到的坑：
        # ODS 的事件表是 (ds, user_id, event_name, metric_value)，
        # **没有 experiment 这一列**。于是"同名护栏由两个实验各生成一遍"
        # 会在同一 (用户, 日, 名字) 上留下两行：08 路的 `SUM(value)` 把它们都加上，
        # 而 `COUNT(DISTINCT user)` 只算一个 —— 均值被**静默放大一倍**
        # （实测 latency_p99 从 100 变成 200，伤害从 12% 变成 6%，
        # 两个数看起来都很正常）。
        # 这条约束写在这里，免得下次有人顺手给新实验也加一条同名护栏。
        guardrails=(),
    ),
    ExperimentDef(
        name="exp_rec_emb",
        layer="recall",
        layer_salt="layer_recall",
        bucket_start=0,
        bucket_end=6000,  # 占 60% 流量
        true_lift=0.0,
        hypothesis="新召回向量对互动次数无影响（负对照）",
    ),
)


@dataclass(frozen=True)
class WarehouseConfig:
    """仿真源数据的超参数。"""

    n_users: int = 20_000
    seed: int = 20260301
    start_ds: date = date(2026, 3, 1)
    entry_span_days: int = 21
    pre_days: int = 14
    post_days: int = 14
    daily_active_p: float = 0.80
    user_level_mean: float = 50.0
    user_level_sd: float = 15.0
    daily_noise_sd: float = 10.0
    # 24 个城市而不是 5 个：簇级检验的自由度是**簇数**减 2，
    # 5 个城市在 50/50 分流下每臂可能只剩 1~2 个簇，簇级方差直接无定义
    # （实测 5 个城市时某个臂只有 1 个簇，簇级路径直接报错）。
    # 城市只出现在用户维表与簇粒度 DWS 里，不影响单元级链路的任何数字。
    # 60 个而不是 5 个：簇级检验的自由度是**簇数**减 2。
    # 5 个城市时某个臂可能只剩 1 个簇（簇级方差直接无定义）；
    # 24 个时 50/50 的哈希分流仍常出现 9/15 这类失衡（SRM 会触发，
    # 而那是**真的**不平衡，不是误报 —— 只是演示数据不该自找麻烦）。
    # 60 个簇让每臂大约 30 个，SRM 与簇级方差都稳。
    #
    # **名字必须等宽**（全是两个字）：分流用的 KeyBatcher 要求 unit_id 的
    # UTF-8 字节长度一致，混进"哈尔滨""石家庄"这种三字名会直接报错。
    cities: tuple[str, ...] = (
        "深圳", "杭州", "北京", "上海", "成都", "广州", "武汉", "西安",
        "南京", "重庆", "苏州", "天津", "长沙", "青岛", "郑州", "东莞",
        "宁波", "佛山", "合肥", "福州", "厦门", "济南", "大连", "昆明",
        "沈阳", "长春", "太原", "南昌", "贵阳", "南宁", "兰州", "银川",
        "西宁", "海口", "三亚", "珠海", "中山", "惠州", "温州", "绍兴",
        "嘉兴", "台州", "金华", "泉州", "烟台", "潍坊", "徐州", "常州",
        "南通", "扬州", "芜湖", "洛阳", "襄阳", "宜昌", "株洲", "柳州",
        "无锡", "汕头", "湛江", "江门",
    )
    experiments: tuple[ExperimentDef, ...] = DEFAULT_EXPERIMENTS


def generate_source_data(
    data_dir: str | Path,
    config: WarehouseConfig | None = None,
    *,
    force: bool = False,
) -> dict[str, int]:
    """生成并写出四张源表，返回各表行数。

    写出目录结构::

        {data_dir}/exposure_log/*.parquet
        {data_dir}/event_log/*.parquet
        {data_dir}/user_profile/*.parquet
        {data_dir}/experiment_config/*.parquet
    """
    cfg = config or WarehouseConfig()
    out = Path(data_dir)
    marker = out / ".generated"
    if marker.exists() and not force:
        return {"__cached__": 1}

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    # ---- 用户维表 -------------------------------------------------------- #
    user_ids = [f"u{i:07d}" for i in range(n)]
    level = rng.normal(cfg.user_level_mean, cfg.user_level_sd, n)
    expose_ds = [
        cfg.start_ds + timedelta(days=int(d)) for d in rng.integers(0, cfg.entry_span_days, n)
    ]
    reg_ds = [
        cfg.start_ds - timedelta(days=int(d)) for d in rng.integers(120, 900, n)
    ]
    city = rng.choice(np.array(cfg.cities, dtype=object), n)

    profile = pd.DataFrame(
        {"user_id": user_ids, "reg_ds": reg_ds, "city": list(city)}
    )

    # ---- 分流 + 曝光日志 -------------------------------------------------- #
    rz = Randomizer()
    batcher = KeyBatcher(user_ids)
    post_effect = np.zeros(n)  # 每个用户的累积真实效应（只有处理后窗口吃得到）

    exposure_frames: list[pd.DataFrame] = []
    guard_frames: list[pd.DataFrame] = []
    guard_plans: list[tuple[ExperimentDef, np.ndarray, np.ndarray, Any]] = []
    # 每个实验一个**独立随机流**：加一个新实验时，别的实验的数字不该跟着变。
    # 第一版所有实验共用一条 rng，于是「多加一个实验」会移动整条随机流，
    # 全部实验的数据、以及 README 里引用过的每一个数字都会变 ——
    # 实测就是这么发生的（加簇级实验时比值链路从 2.3508 变成 2.3308）。
    # 种子用项目自己的 murmur3 挂在实验名上，所以它也与「是第几个」无关。
    config_rows: list[dict] = []

    for exp in cfg.experiments:
        layer = Layer(
            name=exp.layer,
            salt=exp.layer_salt,
            slots=(LayerSlot(exp.name, exp.bucket_start, exp.bucket_end),),
        )

        spec = ExperimentSpec(
            name=exp.name,
            variants=(
                Variant(exp.control, 0.5),
                Variant(exp.treatment, 0.5),
            ),
            salt=f"{exp.name}_v1",
            layer=exp.layer,
            # 整簇随机化时 unit 就是簇键：分流器本身不关心"单元"是用户还是城市，
            # 它只把单元标识拼进哈希键。这一点正是 unit 字段该有的语义。
            unit=exp.cluster_key or "user_id",
        )

        if exp.cluster_key:
            # ---- 整簇随机化：层路由与分流都在簇级别做 ----
            if exp.cluster_key != "city":
                raise ValueError(
                    f"cluster_key 目前只支持 'city'（收到的 {exp.cluster_key!r}）；"
                    "DWD 里带出来的簇键只有 city"
                )
            cities = np.array(sorted(set(city)), dtype=object)
            city_batcher = KeyBatcher([str(c) for c in cities])
            routed_city = np.array(
                [
                    r == exp.name
                    for r in layer.route_many([str(c) for c in cities], city_batcher)
                ],
                dtype=bool,
            )
            city_codes = rz.assign_codes([str(c) for c in cities], spec, city_batcher)
            lookup = {c: int(code) for c, code in zip(cities, city_codes)}
            codes = np.array([lookup[c] for c in city], dtype=int)
            routed = np.array([routed_city[list(cities).index(c)] for c in city], dtype=bool)
        else:
            routed = np.array(
                [r == exp.name for r in layer.route_many(user_ids, batcher)], dtype=bool
            )
            codes = rz.assign_codes(user_ids, spec, batcher)

        is_treatment = routed & (codes == 1)
        post_effect += np.where(is_treatment, exp.true_lift, 0.0)

        exp_rng = np.random.default_rng(
            (cfg.seed + murmur3_32(exp.name.encode("utf-8"))) % (2**32)
        )
        idx = np.flatnonzero(routed)
        exposure_frames.append(
            pd.DataFrame(
                {
                    "ds": [expose_ds[i] for i in idx],
                    # 曝光时刻打散在当天，用于验证"首次曝光去重"口径
                    "ts": [
                        datetime.combine(expose_ds[i], datetime.min.time())
                        + timedelta(minutes=int(m))
                        for i, m in zip(idx, exp_rng.integers(0, 1440, idx.size))
                    ],
                    "user_id": [user_ids[i] for i in idx],
                    "experiment": exp.name,
                    "variant": np.where(is_treatment[idx], exp.treatment, exp.control),
                    "layer": exp.layer,
                }
            )
        )

        config_rows.extend(
            [
                {
                    "experiment": exp.name,
                    "variant": exp.control,
                    "design_weight": 0.5,
                    "layer": exp.layer,
                    "true_lift": exp.true_lift,
                    "hypothesis": exp.hypothesis,
                },
                {
                    "experiment": exp.name,
                    "variant": exp.treatment,
                    "design_weight": 0.5,
                    "layer": exp.layer,
                    "true_lift": exp.true_lift,
                    "hypothesis": exp.hypothesis,
                },
            ]
        )

        # 护栏事件的生成挪到后面：它要用到 offsets/active 这些**之后才定义**的量
        # （第一版直接写在这里，跑起来就是 UnboundLocalError: offsets）。
        guard_plans.append((exp, routed, is_treatment, exp_rng))

    # 同一用户重复曝光（用于验证 DWD 层的去重口径）
    exposures = pd.concat(exposure_frames, ignore_index=True)
    dups = exposures.sample(frac=0.10, random_state=cfg.seed).copy()
    dups["ts"] = dups["ts"] + pd.Timedelta(minutes=30)
    exposures = pd.concat([exposures, dups], ignore_index=True)

    exp_config = pd.DataFrame(config_rows)

    # 护栏声明（长表）：08 路只认这张表给的名单 —— 声明是护栏存在的**前提**，
    # 不是"事件里出现过什么"。方向与容忍度也在这里，但它们只是给数仓一份
    # 可追溯的副本；**判定用的是注册表里的声明**（见 09 路的注释）。
    guardrail_rows = [
        {
            "experiment": exp.name,
            "guardrail": guard.name,
            "direction": guard.direction,
            "max_harm": guard.max_harm,
        }
        for exp in cfg.experiments
        for guard in exp.guardrails
    ]
    guardrail_config = pd.DataFrame(
        guardrail_rows,
        columns=["experiment", "guardrail", "direction", "max_harm"],
    )

    # ---- 行为明细 -------------------------------------------------------- #
    offsets = np.arange(-cfg.pre_days, cfg.post_days)
    is_post = (offsets >= 0).astype(float)

    # 形状 (n_users, n_days)
    active = rng.random((n, offsets.size)) < cfg.daily_active_p
    noise = rng.normal(0.0, cfg.daily_noise_sd, (n, offsets.size))
    values = level[:, None] + noise + post_effect[:, None] * is_post

    # ---- 护栏事件：与主指标同一批用户、同一个日窗 ------------------------ #
    #
    # 建模成**长表**（event_name = 护栏名）而不是给主指标表加列：
    # 每个实验声明几个护栏是业务决定的，列式建模会逼着人每加一个护栏
    # 就改一次已发布层的表结构 —— 而按本仓库的规矩，那会让所有引用过的数字全变。
    for exp, routed, is_treatment, exp_rng in guard_plans:
        for guard in exp.guardrails:
            # 伤害只作用于**处置组**，而且只在处置之后（is_post）
            boost = np.where(is_treatment, guard.harm, 0.0)
            sign = 1.0 if guard.direction == "lower_is_better" else -1.0
            g_noise = exp_rng.normal(1.0, guard.noise, (n, offsets.size))
            g_values = guard.baseline * g_noise * (
                1.0 + sign * boost[:, None] * is_post[None, :]
            )
            gu, gd = np.nonzero(active & routed[:, None])
            guard_frames.append(
                pd.DataFrame(
                    {
                        "ds": [
                            expose_ds[i] + timedelta(days=int(offsets[j]))
                            for i, j in zip(gu, gd)
                        ],
                        "user_id": np.array(user_ids, dtype=object)[gu],
                        "event_name": guard.name,
                        "metric_value": g_values[gu, gd],
                    }
                )
            )

    u_idx, d_idx = np.nonzero(active)
    event = pd.DataFrame(
        {
            "ds": [expose_ds[i] + timedelta(days=int(offsets[j])) for i, j in zip(u_idx, d_idx)],
            "user_id": np.array(user_ids, dtype=object)[u_idx],
            "event_name": "interaction",
            "metric_value": values[u_idx, d_idx],
        }
    )

    # 护栏事件与主指标共表（ODS 的 event_log 本来就是"按天按用户的事件"）。
    # 用 concat 而不是另写一张 Parquet：这样 08 路 SQL 只需按 event_name 过滤。
    if guard_frames:
        event = pd.concat([event, *guard_frames], ignore_index=True)

    # ---- 落盘 ------------------------------------------------------------ #
    for name, frame in (
        ("exposure_log", exposures),
        ("event_log", event),
        ("user_profile", profile),
        ("experiment_config", exp_config),
        ("guardrail_config", guardrail_config),
    ):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "part-0000.parquet", index=False)

    marker.write_text("ok", encoding="utf-8")
    return {
        "exposure_log": len(exposures),
        "event_log": len(event),
        "user_profile": len(profile),
        "experiment_config": len(exp_config),
        "guardrail_config": len(guardrail_config),
    }
