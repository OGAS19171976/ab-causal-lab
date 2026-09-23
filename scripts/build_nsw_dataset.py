#!/usr/bin/env python
"""把 **LaLonde NSW 实验** 接成契约里的三张表（真实处置 + 公开基准）。

为什么是这份数据
----------------
前面接进来的 MovieLens 只能做 A/A **校准**：分流是我们自己做的奇偶切分，
所以它能校准口径与方差，**不能验证效应**（谁也不知道真效应是多少）。
LaLonde 的 NSW 子样本（Dehejia–Wahba）补的正是这一块：

  * **真的随机化** —— 1970 年代美国"国家支持性就业示范项目"把合格申请人
    随机分到处置组/对照组（不是我们切的）；
  * **公开的实验基准** —— 用实验组的两个臂直接比 re78（1978 年收入）：
    Dehejia–Wahba 子样本（185 处置 / 260 对照）给出 **+1794.34**（SE 671.0）。
    这个数就是"已知真值"：任何声称能在 NSW 上估效应的实现，都该复现它；
  * **天然的前置/后置** —— re74 / re75（1974/75 年收入）是处置前，
    re78 是结果。于是 CUPED 的协变量、DWD 的前后窗口都不用造。

口径（把年度数据放进"曝光前后窗口"的模型里）
--------------------------------------------
    1974-01-01  re74   —— 前置
    1975-01-01  re75   —— 前置
    1976-01-01  曝光（项目开始）      {PRE_DAYS}=800 覆盖前两条
    1978-01-01  re78   —— 后置      {POST_DAYS}=1200 覆盖它
事件名统一叫 ``earnings``（metric_event）。**这不是伪造时间维度**：
它是对"每个单元在一个时间点上被处置、前后各有观测"这一模型的忠实映射，
而窗口宽度是 `WarehouseConfig` 的参数（年度数据当然不能沿用合成的 14 天）。

用法::

    python scripts/build_nsw_dataset.py            # 下载 + 校验基准 + 落盘
    python scripts/build_nsw_dataset.py --offline  # 复用已下载的原始文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "build" / "_nsw_raw"
OUT_DIR = ROOT / "data" / "real_nsw"

#: 上游文件与它们**下载当时**的 sha256。
#:
#: 记下来是为了下次能发现"上游改了数据"：源站是可变的 URL，而我们的结论
#: 挂在具体某一份字节上（与 `data/real/provenance.json` 同一个规矩）。
UPSTREAM = {
    "nswre74_treated.txt": (
        "https://users.nber.org/~rdehejia/data/nswre74_treated.txt",
        "e7b742fe0ff07a0f45e129b4ff108bb9611cd83d53604732c48a8a0a3e20eda3",
    ),
    "nswre74_control.txt": (
        "https://users.nber.org/~rdehejia/data/nswre74_control.txt",
        "a1364cea459d953dc691a667d99194b4ad335d6d550354fe23a5d2dc58d729b5",
    ),
}

#: 列序由源站页面给出（见 provenance 里的 citation）
COLUMNS = (
    "treat", "age", "education", "black", "hispanic", "married",
    "nodegree", "re74", "re75", "re78",
)

#: Dehejia–Wahba 子样本的**发表值**（处置组 re78 均值 − 对照组 re78 均值）。
#: 容差取 0.01 美元：它是一串精确到分的数据的均值差，不该有舍入空间。
PUBLISHED_ATT = 1794.34
PUBLISHED_ATT_TOL = 0.01

EXPERIMENT = "nsw_dw"
EXPOSE_DS = date(1976, 1, 1)
PRE_DATES = {"re74": date(1974, 1, 1), "re75": date(1975, 1, 1)}
POST_DATE = date(1978, 1, 1)
METRIC_EVENT = "earnings"


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: pathlib.Path, *, attempts: int = 4) -> None:
    """带重试、**分块读取**的下载。

    源站（NBER）在慢网络上会"读到一半断掉"（``http.client.IncompleteRead``），
    而那不是"数据有问题"。第一版直接 ``resp.read()`` 读整块、也没接异常，
    于是它伪装成了一次构建失败 —— 与"临时目录写不进去让整个检查变红"是同一类：
    **环境噪音伪装成结论**。分块写还有个好处：断了只丢最后一块，
    下一次重试直接覆盖重写（``open("wb")``）。
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ab-causal-lab/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
            return
        except Exception as exc:  # 只用来重试；真实错误在最后一次抛出去
            last = exc
            print(f"    第 {attempt} 次下载失败（{type(exc).__name__}），重试…")
            time.sleep(2.0 * attempt)
    raise SystemExit(f"下载 {url} 失败（{attempts} 次）：{last}")


def fetch(offline: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in UPSTREAM.items():
        path = RAW_DIR / name
        if not (offline and path.exists()):
            _download(url, path)
        actual = sha256_of(path)
        if actual != expected:
            raise SystemExit(
                f"**上游数据变了**：{name} 的 sha256 = {actual[:12]}…，"
                f"而记录的是 {expected[:12]}…\n"
                "两种可能：源站更新了文件，或者我们记错了。两条都要人看一眼，"
                "不能静默继续 —— 结论挂在具体某一份字节上。"
            )
        print(f"  [原始] {name:<26} {path.stat().st_size:>7} 字节  sha256 {actual[:12]}…")


def load() -> pd.DataFrame:
    frames = []
    for name, variant in (("nswre74_treated.txt", "treatment"), ("nswre74_control.txt", "control")):
        arr = np.loadtxt(RAW_DIR / name)
        frame = pd.DataFrame(arr, columns=list(COLUMNS))
        frame["variant"] = variant
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    # 源文件里的处理指示列必须与文件名一致 —— 对不上说明我们读错了列序
    mismatched = int((data["treat"].astype(int).astype(bool) != (data["variant"] == "treatment")).sum())
    if mismatched:
        raise SystemExit(f"处理指示列与文件名不一致：{mismatched} 行 —— 列序可能读错了")
    return data


def benchmark(data: pd.DataFrame) -> dict[str, float]:
    treated = data.loc[data["variant"] == "treatment", "re78"].to_numpy(float)
    control = data.loc[data["variant"] == "control", "re78"].to_numpy(float)
    diff = float(treated.mean() - control.mean())
    se = math.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    return {
        "att": diff,
        "se": se,
        "t": diff / se,
        "n_treated": float(treated.size),
        "n_control": float(control.size),
    }


def build_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """三张表：完全按契约的列（profile 多带协变量，留给将来的观察性对照）。"""
    user_id = [f"nsw{i:04d}" for i in range(len(data))]

    exposure = pd.DataFrame(
        {
            "ds": [EXPOSE_DS] * len(data),
            "ts": [f"{EXPOSE_DS.isoformat()}T00:00:00"] * len(data),
            "user_id": user_id,
            "experiment": EXPERIMENT,
            "variant": data["variant"].to_numpy(),
        }
    )

    # 向量化拼 event_log（而不是逐行 iterrows：既快，也避开 Series 的 object 类型）
    frames = [
        pd.DataFrame(
            {
                "ds": at,
                "user_id": user_id,
                "event_name": METRIC_EVENT,
                "metric_value": data[column].to_numpy(float),
            }
        )
        for column, at in PRE_DATES.items()
    ]
    frames.append(
        pd.DataFrame(
            {
                "ds": POST_DATE,
                "user_id": user_id,
                "event_name": METRIC_EVENT,
                "metric_value": data["re78"].to_numpy(float),
            }
        )
    )
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["ds", "user_id"], kind="stable").reset_index(drop=True)

    # reg_ds 在这个数据集里**不存在**（没有"注册日"这个概念）。
    # 与其编一个像真的日期，不如用一个显式的占位值并写清它是占位：
    # 契约要求这一列，但真实含义是"我们不知道"。
    profile = pd.DataFrame(
        {
            "user_id": user_id,
            "reg_ds": [date(1970, 1, 1)] * len(data),
            "age": data["age"].to_numpy(),
            "education": data["education"].to_numpy(),
            "black": data["black"].to_numpy(),
            "hispanic": data["hispanic"].to_numpy(),
            "married": data["married"].to_numpy(),
            "nodegree": data["nodegree"].to_numpy(),
        }
    )
    return {"exposure_log": exposure, "event_log": events, "user_profile": profile}


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 LaLonde NSW 真实实验数据集")
    ap.add_argument("--offline", action="store_true", help="复用已下载的原始文件")
    args = ap.parse_args()

    print("一、取原始文件（源：users.nber.org/~rdehejia/data，CC BY-NC 2.0）")
    fetch(args.offline)
    data = load()
    stats = benchmark(data)

    print("\n二、复现发表的实验基准（这一步是『数据没接错』的证据）")
    print(f"  处置组 re78 均值 = {data.loc[data.variant == 'treatment', 're78'].mean():.4f}"
          f"（n={int(stats['n_treated'])}）")
    print(f"  对照组 re78 均值 = {data.loc[data.variant == 'control', 're78'].mean():.4f}"
          f"（n={int(stats['n_control'])}）")
    print(f"  ATT = {stats['att']:+.4f}  SE = {stats['se']:.4f}  t = {stats['t']:+.3f}")
    print(f"  发表值 {PUBLISHED_ATT:+.2f}，偏差 {abs(stats['att'] - PUBLISHED_ATT):.4f}"
          f"（容差 {PUBLISHED_ATT_TOL}）")
    if abs(stats["att"] - PUBLISHED_ATT) > PUBLISHED_ATT_TOL:
        raise SystemExit("**复现不出发表值** —— 先别往下走，这通常意味着列序或子样本选错了")

    tables = build_tables(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_meta = []
    print("\n三、落成契约里的三张表")
    for name, frame in tables.items():
        path = OUT_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        digest = sha256_of(path)
        table_meta.append(
            {"name": name, "file": path.name, "rows": int(len(frame)), "sha256": digest}
        )
        print(f"  {name:<14} {len(frame):>5} 行  sha256 {digest[:12]}…")

    treated_share = stats["n_treated"] / (stats["n_treated"] + stats["n_control"])
    provenance = {
        "source": (
            "National Supported Work Demonstration (LaLonde 1986) 的 "
            "Dehejia–Wahba 子样本；取自 Rajeev Dehejia 在 NBER 的公开页面 "
            "https://users.nber.org/~rdehejia/nswdata2.html"
        ),
        "exported_at": date.today().isoformat(),
        "external_generator": True,
        "license": "CC BY-NC 2.0（署名、非商业）",
        "citation": [
            "Dehejia & Wahba (1999), JASA 94(448): 1053-1062",
            "Dehejia & Wahba (2002), Review of Economics and Statistics 84: 151-161",
            "LaLonde (1986), American Economic Review 76: 604-620",
        ],
        "upstream_files": [
            {"name": name, "url": url, "sha256": digest}
            for name, (url, digest) in UPSTREAM.items()
        ],
        "metric_event": METRIC_EVENT,
        "experiments": [
            {
                "name": EXPERIMENT,
                "variants": {
                    "control": round(1.0 - treated_share, 6),
                    "treatment": round(treated_share, 6),
                },
                "control": "control",
                "treatment": "treatment",
                # 这份数据**没有护栏事件**：源文件只有收入，没有延迟/崩溃那类量。
                # 声明为空比编一个像真的护栏更诚实。
                "guardrails": [],
            }
        ],
        "tables": table_meta,
        # 下面是本仓库自己的口径说明（契约不要求，但读的人需要）
        "notes": {
            "benchmark": {
                "att": round(stats["att"], 4),
                "se": round(stats["se"], 4),
                "published": PUBLISHED_ATT,
                "source": "Dehejia–Wahba 子样本的两个实验臂直接比较 re78",
            },
            "weights_are_realized": (
                "variants 用的是**实测份额**（185/260），不是设计值 —— 源站没有给出"
                "分配比例，而按实测份额算 SRM 恒为 0。这一点写在报告里。"
            ),
            "reg_ds_is_a_placeholder": (
                "user_profile.reg_ds 是 1970-01-01 的**占位值**：这份数据里没有"
                "『注册日』这个概念，契约要求这一列。"
            ),
            "time_mapping": (
                "re74/re75 落在 1974/75-01-01（前置），曝光 1976-01-01，"
                "re78 落在 1978-01-01（后置）；窗口宽度由 WarehouseConfig 给"
                "（pre_days=800 / post_days=1200），年度数据不能沿用合成的 14 天。"
            ),
        },
    }
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n四、provenance.json 已写入 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
