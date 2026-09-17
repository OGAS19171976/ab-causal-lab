"""演示数据：让平台一启动就有东西可看。

三条演示实验分别盯住三件事：

1. ``exp_rank_v2`` **正向对照**：注入 ``true_lift=0.35``，CUPED 应当检出显著。
2. ``exp_rec_emb`` **负对照**：``true_lift=0``。它最有教育意义，
   但**不是**"naive 显著而 CUPED 不显著"—— 在干净随机化下那是反的：
   两者点估计几乎相同，而 CUPED 的标准误更小，所以同一份数据里
   CUPED 只会**更**显著。naive 显著、CUPED 不显著这个现象只在
   **处置前协变量失衡**时出现（M1 的偏误分解实验）。
3. ``exp_ui_density`` **90/10 非均分 + 低流量**：用来暴露流量口径与
   SRM 体检（分支不等权时，一条看总数的直觉是会出错的）。

salt 是**选出来的**，规则公开
----------------------------
一份确定种子的合成数据只对应**一个**实现；如果这个实现恰好落在分布的尾巴上，
演示就会把罕见现象当成常态来展示。所以演示 salt 按下面的**机械规则**挑选，
任何人都可以复算：

    对每条实验，在 salt 后缀 ``_v1, _v2, _v3, ...`` 上**顺序搜索**，取第一个满足：
      正向对照  : CUPED z >= 2.5
      负对照    : |CUPED z| < 1 且 |naive z| < 1 且 |ΔX̄ z| < 1 且协变量平衡 p > 0.1
      护栏实验  : |CUPED z| < 1 且 |naive z| < 1

复算：``python scripts/pick_demo_salts.py``（会打印每个候选和被拒原因）。
当前结果：exp_rank_v2_v1 / exp_rec_emb_v7 / exp_ui_density_v3。

**这是"演示数据挑选"，不是"分析结果挑选"。** 界限在这里：
用于**评估**系统的数据必须从预定分布无挑选地抽 —— 那是
``run_platform_aa_audit`` 的 400 个 salt，一个都没挑；
用于**演示**系统的数据是产物的一部分，可以被挑选，但规则必须公开。
报告里把 `exp_rec_emb_v1`（被规则拒掉的那个）的分解也如实记下来了。
"""

from __future__ import annotations

from typing import Any

from .registry import ExperimentRegistry

__all__ = ["DEMO_EXPERIMENTS", "WAREHOUSE_DEMO", "seed_demo"]

DEMO_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "exp_rank_v2",
        "hypothesis": "新排序模型提升人均互动次数",
        "owner": "search-ranking",
        "layer": "ranking",
        "variants": [
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ],
        "salt": "exp_rank_v2_v1",
        "traffic_ratio": 0.8,
        "primary_metric": "interaction_per_user_14d",
        "guardrails": ["latency_p99", "complaint_rate"],
        "status": "running",
        "start_ds": "2026-03-01",
        "true_lift": 0.35,
    },
    {
        "name": "exp_rec_emb",
        "hypothesis": "新召回向量对互动次数无影响（负对照）",
        "owner": "recall",
        "layer": "recall",
        "variants": [
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ],
        "salt": "exp_rec_emb_v7",
        "traffic_ratio": 0.6,
        "primary_metric": "interaction_per_user_14d",
        "guardrails": ["recall_coverage"],
        "status": "running",
        "start_ds": "2026-03-01",
        # 真实效应为零 —— 用来演示"负对照会怎样被协变量失衡污染"
        "true_lift": 0.0,
    },
    {
        "name": "exp_ui_density",
        "hypothesis": "提高信息流密度不改变人均互动次数（预期零效应，检验护栏）",
        "owner": "growth",
        "layer": "ui",
        "variants": [
            {"name": "control", "weight": 0.9},
            {"name": "treatment", "weight": 0.1},
        ],
        "salt": "exp_ui_density_v3",
        "traffic_ratio": 0.3,
        "primary_metric": "interaction_per_user_14d",
        "guardrails": ["scroll_depth"],
        "status": "draft",
        "true_lift": -0.1,
    },
)


#: 绑定**真实数仓**的演示记录（仅在 ``build/warehouse.duckdb`` 存在时写入）。
#:
#: 它和上面三条的区别不是"名字不同"，而是数据来源不同：
#: 这里的数字来自 M0 那套 ODS→DWD→DWS→ADS 链路，平台只读 ADS 的充分统计量。
#: ``true_lift=2.0`` 是数仓生成时注入的真值（``WarehouseConfig``），
#: 在数仓路径下这个字段只用于展示 —— 平台不会拿它去造数据。
WAREHOUSE_DEMO: dict[str, Any] = {
    "name": "wh_exp_rank_v2",
    "hypothesis": "【数仓链路】新排序模型提升人均互动次数（读 ADS + DWS）",
    "owner": "search-ranking",
    "layer": "ranking",
    "variants": [
        {"name": "control", "weight": 0.5},
        {"name": "treatment", "weight": 0.5},
    ],
    "salt": "wh_exp_rank_v2_v1",
    "primary_metric": "post_metric_14d",
    "guardrails": ["latency_p99"],
    "status": "running",
    "start_ds": "2026-03-01",
    "true_lift": 2.0,
    "warehouse_experiment": "exp_rank_v2",
}


def seed_demo(
    registry: ExperimentRegistry,
    *,
    force: bool = False,
    warehouse_available: bool = False,
) -> int:
    """写入演示实验，返回新增条数。

    ``force=False``（默认）时**已存在的实验不会被覆盖** ——
    保证重复启动服务不会把用户的改动冲掉。
    ``force=True`` 时先删后建：名字是注册表的唯一键，直接 ``create`` 会撞键，
    所以"覆盖"只能通过删除旧记录实现（也顺带换掉旧的 salt）。

    ``warehouse_available=True`` 时额外写入一条**绑定数仓**的实验，
    于是页面上能同时看到两条数据源：合成数据（三条）与真实链路（一条）。
    数仓不存在时不写 —— 否则那条记录一点"分析"就会报"没有数仓"。
    """
    todo = list(DEMO_EXPERIMENTS)
    if warehouse_available:
        todo.append(WAREHOUSE_DEMO)

    added = 0
    for spec in todo:
        existing = registry.get_by_name(spec["name"])
        if existing is not None:
            if not force:
                continue
            registry.delete(existing.id)
        registry.create(**spec)
        added += 1
    return added
