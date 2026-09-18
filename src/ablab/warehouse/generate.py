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

import numpy as np
import pandas as pd

from ..assignment import ExperimentSpec, Layer, LayerSlot, Randomizer, Variant
from ..hashing import KeyBatcher

__all__ = [
    "ExperimentDef",
    "WarehouseConfig",
    "DEFAULT_EXPERIMENTS",
    "generate_source_data",
]


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
    cities: tuple[str, ...] = ("深圳", "杭州", "北京", "上海", "成都")
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

        idx = np.flatnonzero(routed)
        exposure_frames.append(
            pd.DataFrame(
                {
                    "ds": [expose_ds[i] for i in idx],
                    # 曝光时刻打散在当天，用于验证"首次曝光去重"口径
                    "ts": [
                        datetime.combine(expose_ds[i], datetime.min.time())
                        + timedelta(minutes=int(m))
                        for i, m in zip(idx, rng.integers(0, 1440, idx.size))
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

    # 同一用户重复曝光（用于验证 DWD 层的去重口径）
    exposures = pd.concat(exposure_frames, ignore_index=True)
    dups = exposures.sample(frac=0.10, random_state=cfg.seed).copy()
    dups["ts"] = dups["ts"] + pd.Timedelta(minutes=30)
    exposures = pd.concat([exposures, dups], ignore_index=True)

    exp_config = pd.DataFrame(config_rows)

    # ---- 行为明细 -------------------------------------------------------- #
    offsets = np.arange(-cfg.pre_days, cfg.post_days)
    is_post = (offsets >= 0).astype(float)

    # 形状 (n_users, n_days)
    active = rng.random((n, offsets.size)) < cfg.daily_active_p
    noise = rng.normal(0.0, cfg.daily_noise_sd, (n, offsets.size))
    values = level[:, None] + noise + post_effect[:, None] * is_post

    u_idx, d_idx = np.nonzero(active)
    event = pd.DataFrame(
        {
            "ds": [expose_ds[i] + timedelta(days=int(offsets[j])) for i, j in zip(u_idx, d_idx)],
            "user_id": np.array(user_ids, dtype=object)[u_idx],
            "event_name": "interaction",
            "metric_value": values[u_idx, d_idx],
        }
    )

    # ---- 落盘 ------------------------------------------------------------ #
    for name, frame in (
        ("exposure_log", exposures),
        ("event_log", event),
        ("user_profile", profile),
        ("experiment_config", exp_config),
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
    }
