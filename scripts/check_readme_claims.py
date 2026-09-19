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
    # 比值口径的 A/A 校准（"换了口径就要重跑审计"那一条的执行）
    ("M6·比值口径：序贯 FWER", "m6_validation.md", "0.0700"),
    ("M6·比值口径：95% 覆盖率", "m6_validation.md", "0.9250"),
    # CATE 区间：**只加文本类声明**。这份报告的数值会随 --quick 变，
    # 把数字写进清单就会让检查器在快速模式下随机变红（第 31 条那个坑）。
    ("M4·CATE 区间：结论是「校准不了」", "cate_interval_report.md", "但它校准不了"),
    ("M4·CATE 区间：结论并进了 M4 报告", "m4_validation.md", "CATE 的区间：两条路线的覆盖率"),
    ("M4·CATE 区间：原因是偏差不是方差", "cate_interval_report.md", "而是**点估计有偏**"),
    # 组级校准（BLP/GATES）：同样只加文本类声明 —— 覆盖率与斜率会随
    # --quick（分裂次数 6 vs 30）变，把数字写进清单就是第 31 条那个假红灯。
    # SA 回归版：最后队列当基准（数值随面板固定，稳定）
    ("M3·SA：最后队列当基准", "m3_validation.md", "最后队列当基准"),
    ("M3·SA：改善倍数", "m3_validation.md", "改善了 50 倍"),
    ("M4·组级校准：换推断对象", "cate_interval_report.md", "组级路线：BLP 与 GATES"),
    ("M4·组级校准：验收结论", "cate_interval_report.md", "对象不同，结论不同"),
    ("M4·组级校准：瓶颈是信号重尾", "cate_interval_report.md", "重尾到不实用"),
    ("M4·组级校准：下一步换 AIPW 信号", "cate_interval_report.md", "AIPW 信号"),
    ("M4·信号选择：裁剪与 AIPW 的分工", "cate_interval_report.md", "裁剪管尾巴"),
    ("M4·信号选择：这是一次自我纠正", "cate_interval_report.md", "自我纠正"),
    ("M4·信号选择：四版本对照表的列", "cate_interval_report.md", "信号 sd"),
    # 默认路径（AIPW + 自动裁剪）：仍然只加文本类声明（数值随 --quick 变）
    ("M4·默认路径：已接成默认", "cate_interval_report.md", "默认路径"),
    ("M4·默认路径：代价是估计目标变了", "cate_interval_report.md", "重叠总体"),
    ("M4·默认路径：阈值有文献依据", "cate_interval_report.md", "Crump"),
    ("数仓：CUPED 与 DWD 明细一致", "warehouse_report.md", "一致"),
    ("数仓：ADS 判据写在报告里", "warehouse_report.md", "判据"),
    # 比值链路（06/07）的数字 —— 它们是"接通了"这件事唯一的实测证据
    ("数仓·比值：正对照实验的效应", "warehouse_report.md", "2.297438"),
    ("数仓·比值：负对照也显著（诚实记下）", "warehouse_report.md", "0.671170"),
    ("数仓·比值：末次查看 == 主结论", "warehouse_report.md", "末次查看 == 主结论：True"),
    ("治理：审计删不掉", "governance_report.md", "append-only"),
    # 原"治理：护栏未分析"这条已删除：护栏现在**真的**会被判定，
    # 报告里不再有"尚不分析"这句话。改成钉"判定规则"本身。
    ("治理：护栏判定含停实验建议", "governance_report.md", "判定规则"),
    # 身份与审计操作者（静态 token）：用户 id 与角色是稳定的字符串，可以进清单
    ("治理·身份：静态 token + 角色", "governance_report.md", "静态 token"),
    ("治理·身份：凭据只存哈希", "governance_report.md", "sha256"),
    ("治理·身份：伪造无效", "governance_report.md", "http_alice"),
    # 并发（乐观锁）：文本类声明 —— 版本号随操作次数变，不把数字写进清单
    ("治理·并发：丢失更新已可见", "governance_report.md", "丢失更新"),
    ("治理·并发：冲突返回 412", "governance_report.md", "412"),
    # 护栏：真的判定（文本类；数值随 n_users 变，不写进清单）
    ("治理·护栏：越界建议停实验", "governance_report.md", "建议停止实验"),
    ("治理·护栏：缺数据不等于通过", "governance_report.md", "不是通过"),
    ("治理·护栏：方向不从名字猜", "governance_report.md", "不从指标名猜"),
    ("治理·护栏：判定已校准", "governance_report.md", "误停率"),
    ("治理·护栏：H0 不误停", "governance_report.md", "宁可少停"),
    ("治理·护栏：边界点是真正的工作点", "governance_report.md", "边界点"),
    # 数仓护栏链路：两臂均值与"事件名过滤"是稳定的
    ("数仓·护栏：长表链路", "warehouse_report.md", "护栏链路"),
    ("数仓·护栏：注入的伤害可见", "warehouse_report.md", "latency_p99"),
    # 簇级 CUPED（文本类：数值随数据重生成会变）
    ("数仓·簇级 CUPED：口径已打开", "warehouse_report.md", "簇级 CUPED"),
    ("数仓·簇级 CUPED：观测单位是簇", "warehouse_report.md", "观测单位都是簇"),
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
