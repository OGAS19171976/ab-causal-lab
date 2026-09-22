#!/usr/bin/env python
"""把 **MovieLens（GroupLens）** 转成这个仓库的"外部数据"三表 + provenance。

为什么选它
----------
要压测的是一条 A/B 链路，而外部数据**没有 `true_lift`**（那一列在外部路径上写空），
所以它不能回答"我估的效应准不准"。它能回答的是另一个更基础的问题：
**在真实分布上，我的校准还成立吗？** 回答这个问题只需要一个 A/A —— 不需要真值。
MovieLens 满足三件关键事：

  * **真实分布**：10 万条评分、610 个用户，评分严重偏向 4 分，
    用户活跃度差两个数量级 —— 合成器给不出这种零膨胀 + 长尾；
  * **真实时间戳**：1996–2018，可以切出"前段/后段"（CUPED 要的正是处置前指标）；
  * **体积小**（~1 MB）：随仓库走，于是**门在 CI 里每天都被真的跑一遍**，
    而不是只在某人手动接入时跑一次。

切分方式（被数据逼出来的，写在这里而不是留在注释外）
----------------------------------------------------
第一版写死一个全局分界日 2016-01-01 —— 全库只有 **24 个用户**两侧都有评分；
改成"由数据选"（候选里两侧都有评分的用户最多的那天）也只有 **27 个**：
MovieLens 用户的活跃期很**短促**，大多数人在几周内把想评的评完。
现在改成**一人一条分界线**（该用户自己的中位评分时间），**610 个用户全部可用**。
这不是伪造实验窗口：A/A 里没有真实处置，"前段"只用来当 CUPED 的协变量。
但这一点必须写清楚，否则"处置前"会被读成真实实验窗口。

诚实的边界（同时写进 provenance 与报告）
----------------------------------------
  * **分流是我们做的 A/A**：按 ``sha256(user_id)`` 奇偶分 control/treatment；
    能校准**口径与方差**，不能验证效应；
  * `ds/ts` 直接用真实评分时间，**未编造**；
  * `reg_ds` 是派生列（该用户最早评分日），`city` 缺失 ⇒ 簇级路径不可用；
  * 电影评分不是业务 KPI：它压的是**分布性质**（零膨胀、长尾、真实行为节奏），
    不是"这个指标在我们业务里长什么样"。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real"
DEFAULT_ZIP = ROOT / "build" / "realdata_dl" / "ml.zip"
SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

EXPERIMENT = "ml_aa"
METRIC_EVENT = "rating"
GUARDRAIL_EVENT = "rating_low"
GUARDRAIL_TOLERANCE = 0.02


def load_ratings(zip_path: pathlib.Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("ratings.csv"))
        frame = pd.read_csv(io.BytesIO(z.read(name)))
    frame["ts"] = pd.to_datetime(frame["timestamp"], unit="s")
    frame["ds"] = frame["ts"].dt.date
    frame["user_id"] = frame["userId"].astype(str)
    return frame


def assign_variant(user_ids: pd.Series) -> pd.Series:
    """A/A 分流：``sha256(user_id)`` 的奇偶。

    刻意用最朴素的确定性规则并**写下来**：它只需要"确定 + 均衡"，
    与用哪个 hash 无关 —— 这一轮压的是分布与口径，分流实现由 M0 的审计覆盖。
    """
    def one(uid: str) -> str:
        digest = hashlib.sha256(str(uid).encode("utf-8")).hexdigest()
        return "treatment" if int(digest[-1], 16) % 2 == 0 else "control"

    return user_ids.map(one)


def per_user_split(ratings: pd.DataFrame) -> pd.DataFrame:
    """每个用户一行：他自己的中位评分时间 + 两侧都有评分的用户才留下。

    只保留前/后两段**都有**评分的用户：没有前段就没有 CUPED 可言。
    """
    ordered = ratings.sort_values(["user_id", "ts"], kind="stable")
    cut = ordered.groupby("user_id")["ts"].transform("median")
    pre_users = set(ordered.loc[ordered["ts"] < cut, "user_id"])
    post_users = set(ordered.loc[ordered["ts"] >= cut, "user_id"])
    usable = sorted(pre_users & post_users)
    exposure = (
        ordered[ordered["user_id"].isin(usable)]
        .groupby("user_id", as_index=False)["ts"]
        .median()
        .rename(columns={"ts": "exposure_ts"})
    )
    exposure["ds"] = exposure["exposure_ts"].dt.date
    exposure["variant"] = assign_variant(exposure["user_id"]).to_numpy()
    return exposure


def build(zip_path: pathlib.Path) -> dict:
    ratings = load_ratings(zip_path)
    exposure_users = per_user_split(ratings)
    users = sorted(exposure_users["user_id"])
    frame = ratings[ratings["user_id"].isin(users)].copy()

    exposure = pd.DataFrame(
        {
            "ds": exposure_users["ds"],
            "ts": exposure_users["exposure_ts"],
            "user_id": exposure_users["user_id"],
            "experiment": [EXPERIMENT] * len(exposure_users),
            "variant": exposure_users["variant"],
        }
    )
    events = pd.concat(
        [
            pd.DataFrame(
                {
                    "ds": frame["ds"],
                    "user_id": frame["user_id"],
                    "event_name": METRIC_EVENT,
                    "metric_value": frame["rating"].astype(float),
                }
            ),
            pd.DataFrame(
                {
                    "ds": frame["ds"],
                    "user_id": frame["user_id"],
                    "event_name": GUARDRAIL_EVENT,
                    "metric_value": (frame["rating"] <= 2.0).astype(float),
                }
            ),
        ],
        ignore_index=True,
    )
    first_seen = frame.groupby("user_id")["ds"].min().rename("reg_ds").reset_index()
    profile = pd.DataFrame({"user_id": users}).merge(first_seen, on="user_id")

    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    for name, table in (
        ("exposure_log", exposure),
        ("event_log", events),
        ("user_profile", profile),
    ):
        path = OUT / f"{name}.parquet"
        table.to_parquet(path, index=False)
        tables.append(
            {
                "name": name,
                "file": path.name,
                "rows": int(len(table)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    median_ds = exposure_users.set_index("user_id")["ds"]
    pre_rows = int((frame["ds"] < frame["user_id"].map(median_ds)).sum())
    provenance = {
        "source": "MovieLens ml-latest-small（GroupLens Research，外部分发）",
        "exported_at": str(pd.Timestamp.now(tz="UTC").date()),
        "external_generator": True,
        "metric_event": METRIC_EVENT,
        "guardrail_events": [GUARDRAIL_EVENT],
        "notes": (
            "评分与时间戳来自 MovieLens（真实），ds/ts 直接用评分时间，未编造。"
            "切分方式是**一人一条分界线**（该用户自己的中位评分时间）："
            "前段当处置前指标、后段当结果 —— A/A 里没有真实处置，"
            "所以这不是一个真实的实验窗口，前段只用来做 CUPED 的协变量。"
            "分流由 build_real_aa_dataset.py 按 sha256(user_id) 奇偶决定："
            "这是**我们做的 A/A**，只能校准口径与方差，不能验证效应。"
            "reg_ds 是派生列（该用户最早评分日）；city 缺失 ⇒ 簇级路径不可用。"
            f"原始文件 {SOURCE_URL}"
        ),
        "experiments": [
            {
                "name": EXPERIMENT,
                "variants": {"control": 0.5, "treatment": 0.5},
                "control": "control",
                "treatment": "treatment",
                "guardrails": [
                    [GUARDRAIL_EVENT, "lower_is_better", GUARDRAIL_TOLERANCE]
                ],
            }
        ],
        "tables": tables,
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "users": len(users),
        "ratings": int(len(frame)),
        "pre_rows": pre_rows,
        "date_range": (str(frame["ds"].min()), str(frame["ds"].max())),
        "tables": tables,
        "source_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="把 MovieLens 转成本仓库的外部数据三表")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP), help="ml-latest-small.zip 的路径")
    args = ap.parse_args()
    zip_path = pathlib.Path(args.zip)
    if not zip_path.exists():
        print(f"找不到 {zip_path}")
        print(f"先下载：curl -sSL -o {zip_path} {SOURCE_URL}")
        return 1
    info = build(zip_path)
    print("外部数据已生成：data/real/（三表 + provenance.json）")
    print(f"  用户 {info['users']}；评分 {info['ratings']}（前段 {info['pre_rows']}）")
    print(f"  日期范围 {info['date_range'][0]} → {info['date_range'][1]}；"
          "切分：每人自己的中位评分时间")
    print(f"  原始 zip sha256 {info['source_zip_sha256'][:16]}…")
    for t in info["tables"]:
        print(f"  {t['name']:<14}{t['rows']:>8} 行  sha256 {t['sha256'][:12]}…")
    print("  注意：分流是我们做的 A/A —— 能校准口径与方差，不能验证效应。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
