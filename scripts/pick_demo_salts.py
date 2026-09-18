"""按公开规则挑选演示 salt。

规则（写进 README，任何人可复算）
--------------------------------
对每条演示实验，保持实验名不变，在 salt 后缀 ``_v1, _v2, _v3, ...`` 上**顺序搜索**，
取第一个满足下列条件的后缀：

  正向对照（true_lift=0.35）: CUPED z >= 2.5（真实效应被检出）
  负对照  （true_lift=0.0） : |CUPED z| < 1 且 |naive z| < 1 且 |ΔX z| < 1 且平衡 p > 0.1
  护栏实验（true_lift=-0.1）: |CUPED z| < 1 且 |naive z| < 1

这是"演示数据挑选"，不是"分析结果挑选"：
**评估**用的是 ``run_platform_aa_audit`` 的 400 个 salt，那一份**没有任何挑选**。
两者的分工必须在报告里写清楚，否则挑选就变成了粉饰。
"""

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.platform import run_demo_decomposition  # noqa: E402

N = 20_000
MAX_K = 40

#: 注解写 ``Any`` 是**故意**的：每一条都是"手写的实验配置"，值有 str / float / list
#: 好几种类型（``variants`` 是列表、``traffic_ratio`` 是数）。
#: 不给注解时 mypy 会把 ``cfg["name"]`` 推成 ``object``，于是它没法当字典的键用 ——
#: 报错是真的，但根因是"这个结构本来就是异构的"，而不是代码写错了。
CONFIGS: list[dict[str, Any]] = [
    {
        "name": "exp_rank_v2",
        "hypothesis": "新排序模型提升人均互动次数",
        "owner": "search-ranking",
        "layer": "ranking",
        "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
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
        "variants": [{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
        "traffic_ratio": 0.6,
        "primary_metric": "interaction_per_user_14d",
        "guardrails": ["recall_coverage"],
        "status": "running",
        "start_ds": "2026-03-01",
        "true_lift": 0.0,
    },
    {
        "name": "exp_ui_density",
        "hypothesis": "提高信息流密度不改变人均互动次数（预期零效应，检验护栏）",
        "owner": "growth",
        "layer": "ui",
        "variants": [{"name": "control", "weight": 0.9}, {"name": "treatment", "weight": 0.1}],
        "traffic_ratio": 0.3,
        "primary_metric": "interaction_per_user_14d",
        "guardrails": ["scroll_depth"],
        "status": "draft",
        "true_lift": -0.1,
    },
]


def ok(cfg: dict, d) -> tuple[bool, str]:
    name = cfg["name"]
    if name == "exp_rank_v2":
        good = d.cuped_z >= 2.5
        return good, f"CUPED z={d.cuped_z:+.2f} (需>=2.5)"
    if cfg["true_lift"] == 0.0:
        good = (
            abs(d.cuped_z) < 1.0
            and abs(d.naive_z) < 1.0
            and abs(d.split_z) < 1.0
            and d.balance_p_value > 0.1
        )
        return good, (
            f"CUPED z={d.cuped_z:+.2f} naive z={d.naive_z:+.2f} "
            f"ΔX z={d.split_z:+.2f} 平衡 p={d.balance_p_value:.3f}"
        )
    good = abs(d.cuped_z) < 1.0 and abs(d.naive_z) < 1.0
    return good, f"CUPED z={d.cuped_z:+.2f} naive z={d.naive_z:+.2f}"


def main() -> int:
    chosen: dict[str, str] = {}
    for cfg in CONFIGS:
        print("=" * 78)
        print(f"{cfg['name']}  (true_lift={cfg['true_lift']}, traffic={cfg['traffic_ratio']})")
        print("=" * 78)
        found = None
        for k in range(1, MAX_K + 1):
            salt = f"{cfg['name']}_v{k}"
            demo = dict(cfg, salt=salt)
            try:
                d = run_demo_decomposition(n_users=N, demos=(demo,))[0]
            except ValueError as exc:
                print(f"  k={k:<3} salt={salt:<24} 跳过：{exc}")
                continue
            good, why = ok(cfg, d)
            mark = "  <-- 选中" if good else ""
            print(f"  k={k:<3} salt={salt:<24} {why}{mark}")
            if good:
                found = salt
                break
        if found is None:
            print(f"  [失败] {MAX_K} 个后缀内没有满足条件的 salt")
            return 1
        chosen[cfg["name"]] = found

    print()
    print("=" * 78)
    print("选定的 salt（写进 demo.py）")
    print("=" * 78)
    for name, salt in chosen.items():
        print(f'  {name:<16} "{salt}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
