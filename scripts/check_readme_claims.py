#!/usr/bin/env python
"""核对 README 里的**关键数字**是否真的能在 ``reports/`` 里找到。

为什么值得做成脚本
------------------
README 的第一句话是"每一个数字都能在 ``reports/`` 里找到对应的图和原始输出"。
但在这之前，**没有任何东西在核对它** —— 报告重跑时数字会变，README 不会跟着变，
于是那句话会慢慢变成不准确的声明（这个仓库已经在"统一换行符"和"CI 跑过没有"
两件事上各吃过一次同样的亏）。

做法刻意保守：不试图解析 README 的每个数（那会误报一片），
而是维护一张**声明清单** ``CLAIMS``：(人读的说法, 应当在哪个报告里出现的字符串)。
每个数字都必须能在**指定的那一份**报告里逐字找到 —— 找不到就是漂移，
要么改 README，要么补证据，两种情况都必须有人看一眼。

用法::

    python scripts/check_readme_claims.py          # 核对，缺一个就 exit 1
    python scripts/check_readme_claims.py --list    # 只看清单
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

#: ``(人读的说法, 报告文件, 必须逐字出现的子串)``
#:
#: 只放**有明确出处**的数字。三类东西刻意不放：
#:   * 依赖 / 工具版本（出处是 requirements.lock，不是 reports/）
#:   * 测试个数（出处是 pytest 的输出，见 README 自己写的那条说明）
#:   * 派生出来的百分比（比如"降 49%"是算出来的，报告里是原始的两个数）
CLAIMS: list[tuple[str, str, str]] = [
    ("A/A 校准：naive I 类错误", "m0_validation.md", "0.0520"),
    ("A/A 校准：覆盖率", "m0_validation.md", "0.9480"),
    ("CUPED 方差缩减（实测）", "m1_validation.md", "0.4938"),
    ("CUPED 等效样本量", "m1_validation.md", "2.01"),
    ("比值口径差", "m1_validation.md", "-12.61%"),
    ("群序贯：5 次查看的 FWER", "m2_validation.md", "0.0500"),
    ("交错处置：TWFE 符号翻转率", "m3_validation.md", "100%"),
    ("聚合方差：独立合成的越界率", "m3_validation.md", "0.220"),
    ("聚合方差：影响函数合成的越界率", "m3_validation.md", "0.065"),
    ("聚合方差：SE 低估幅度", "m3_validation.md", "43.9%"),
    ("M4：森林 vs 常数基线的对照", "m4_validation.md", "因果森林 vs 常数基线"),
    ("M5：平台 A/A 的 CUPED I 类错误", "m5_validation.md", "0.0625"),
    # 这一条刻意用"最大偏差"而不是那个 1e-13 的数：**那个数本身随运行环境漂移**
    # （见 README 14 节，两轮实测取过 1.1e-14 ~ 4.5e-13）。把它当声明，
    # 检查器就会随机变红 —— 那正是本仓库不想要的"假红灯"。
    ("M5：三路径最大偏差（量级）", "m5_validation.md", "最大偏差"),
    ("M6：单元级 I 类错误（错的那个）", "m6_validation.md", "0.6933"),
    ("M6：簇级 I 类错误（对的那个）", "m6_validation.md", "0.0633"),
    ("M6：MDE 与功效的自洽性", "m6_validation.md", "z_power(mde)="),
    ("数仓：CUPED 与 DWD 明细一致", "warehouse_report.md", "一致"),
    ("数仓：ADS 判据写在报告里", "warehouse_report.md", "判据"),
    ("治理：审计删不掉", "governance_report.md", "append-only"),
    ("治理：护栏未分析", "governance_report.md", "尚不分析护栏指标"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="核对 README 里的数字能否在 reports/ 里找到")
    ap.add_argument("--list", action="store_true", help="只列清单")
    args = ap.parse_args()

    if args.list:
        for label, report, needle in CLAIMS:
            print(f"  {label:<32} {report:<24} {needle!r}")
        return 0

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing_report: list[str] = []
    not_in_report: list[str] = []
    not_in_readme: list[str] = []

    for label, report_name, needle in CLAIMS:
        path = REPORTS / report_name
        if not path.exists():
            missing_report.append(f"{label}: 报告不存在 {report_name}")
            continue
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            not_in_report.append(f"{label}: {needle!r} 不在 {report_name} 里")
            continue
        # 报告里有，README 里也应当能看到同一个数字（否则是"报告变了、README 没跟"）
        if needle not in readme:
            not_in_readme.append(f"{label}: {needle!r} 在报告里，但 README 没提")

    print(f"核对 {len(CLAIMS)} 条声明：报告存在且有该数字 "
          f"{len(CLAIMS) - len(missing_report) - len(not_in_report)} 条")
    if not_in_readme:
        print(f"\n{len(not_in_readme)} 条**报告里有、README 没写**"
              "（不一定是错，但要么补上，要么把这条从清单里删掉）：")
        for line in not_in_readme:
            print(f"  - {line}")
    if missing_report or not_in_report:
        print(f"\n**{len(missing_report) + len(not_in_report)} 条对不上**"
              "（声明漂移了，必须有人看一眼）：")
        for line in missing_report + not_in_report:
            print(f"  - {line}")
        return 1
    print("\n所有声明都能在指定的报告里逐字找到")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
