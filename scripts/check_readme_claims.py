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
    ("M6·簇级 CUPED：校准后的误停率", "m6_validation.md", "簇级 CUPED 校准"),
    # 比值口径的 A/A 校准（"换了口径就要重跑审计"那一条的执行）
    ("M6·比值口径：序贯 FWER", "m6_validation.md", "0.0700"),
    ("M6·比值口径：95% 覆盖率", "m6_validation.md", "0.9250"),
    # CATE 区间：**只加文本类声明**。这份报告的数值会随 --quick 变，
    # 把数字写进清单就会让检查器在快速模式下随机变红（第 31 条那个坑）。
    ("M4·CATE 区间：结论是「校准不了」", "cate_interval_report.md", "但它校准不了"),
    ("M4·CATE 区间：结论并进了 M4 报告", "m4_validation.md", "CATE 的区间：两条路线的覆盖率"),
    # 数仓比值链路的**序贯校准**（100 个 A/A 复制实验走真实链路）。
    # 这一份报告不带 --quick 以外的开关，数字稳定，所以可以直接钉数值。
    ("数仓·比值链路：序贯 FWER（100 个 A/A 复制实验）",
     "warehouse_report.md", "序贯 FWER = 0.0400"),
    ("数仓·比值链路：末次区间覆盖 0",
     "warehouse_report.md", "末次区间覆盖 0 的比例 0.9600"),
    ("数仓·比值链路：末次 z 的 sd（SE 诚实的证据）",
     "warehouse_report.md", "sd 1.0487"),
    ("数仓·比值链路：salt 独立性（同臂一致率）",
     "warehouse_report.md", "同臂一致率 0.500004"),
    ("数仓·比值链路：100 个 salt 的过度离散检验",
     "warehouse_report.md", "chi2 = 105.36"),
    ("数仓·比值链路：零效应的校准边界",
     "warehouse_report.md", "这一节只能校准零效应"),
    # 6c：真实效应下的覆盖/功效（数值稳定，非 --quick 路径）
    ("数仓·比值链路：真实效应下的 95% 覆盖真值",
     "warehouse_report.md", "末次 95% 区间覆盖真值的比例 0.9375"),
    ("数仓·比值链路：真实效应下的功效",
     "warehouse_report.md", "末次显著率 0.8000"),
    ("数仓·比值链路：平均 SE ÷ 跨复制 sd（诚实性）",
     "warehouse_report.md", "平均 SE ÷ 跨复制 sd = 1.0327"),
    ("数仓·比值链路：重随机化把 SE/sd 钉住（0.9968 与 0.9924）",
     "warehouse_report.md", "SE/sd = 0.9968"),
    ("数仓·比值链路：真值与注入的 lift 逐字相等",
     "warehouse_report.md", "所以覆盖率是直接对着真值数的"),
    ("M4·CATE 区间：主因是点估计有偏（方差另有一处，见下）",
     "cate_interval_report.md", "而是**点估计有偏**"),
    # 森林的跨树方差（第 19 节）。同样只加文本类声明：8c 节的数值随
    # --quick 变（重抽 8 vs 30 次、bootstrap 2×6 vs 6×15）。
    ("M4·森林 SE：跨树协方差是第四处同族错误",
     "cate_interval_report.md", "跨树协方差（第四处同族错误）"),
    ("M4·森林 SE：改法是 GRF 的森林权重",
     "cate_interval_report.md", "GRF 那套森林权重的写法"),
    ("M4·森林 SE：只覆盖「给定树结构」的方差",
     "cate_interval_report.md", "给定树结构**的方差"),
    ("M4·森林 SE：inf 点等于没有区间",
     "cate_interval_report.md", "其余只拿到无穷区间，等于没有区间"),
    # 组级校准（BLP/GATES）：同样只加文本类声明 —— 覆盖率与斜率会随
    # --quick（分裂次数 6 vs 30）变，把数字写进清单就是第 31 条那个假红灯。
    # SA 回归版：最后队列当基准（数值随面板固定，稳定）
    ("M3·SA：最后队列当基准", "m3_validation.md", "最后队列当基准"),
    ("M3·SA：改善倍数", "m3_validation.md", "改善了 50 倍"),
    ("M3·BJS：插补估计量已补", "m3_validation.md", "BJS 插补"),
    ("M3·BJS：差异不是效应异质", "m3_validation.md", "逐位相同"),
    ("M3·dCDH：换手估计量已补", "m3_validation.md", "dCDH 换手估计量"),
    ("M3·dCDH：逐队列与 CS 相同", "m3_validation.md", "逐队列核对"),
    ("M3·dCDH：安慰剂会报警", "m3_validation.md", "报警"),
    ("M4·组级校准：换推断对象", "cate_interval_report.md", "组级路线：BLP 与 GATES"),
    ("M4·组级校准：验收结论", "cate_interval_report.md", "对象不同，结论不同"),
    ("M4·组级校准：瓶颈是信号重尾", "cate_interval_report.md", "重尾到不实用"),
    ("M4·组级校准：下一步换 AIPW 信号", "cate_interval_report.md", "AIPW 信号"),
    ("M4·信号选择：裁剪与 AIPW 的分工", "cate_interval_report.md", "裁剪管尾巴"),
    ("M4·信号选择：这是一次自我纠正", "cate_interval_report.md", "自我纠正"),
    ("M4·信号选择：四版本对照表的列", "cate_interval_report.md", "信号 sd"),
    # 保形个体效应区间（数值随 --quick 变，只钉文本）
    ("M4·保形：对象不同", "cate_interval_report.md", "个体效应：解析区间做不到"),
    ("M4·保形：边际覆盖与条件均值的区分", "cate_interval_report.md", "预测区间"),
    ("M4·保形：分组覆盖两端最弱", "cate_interval_report.md", "两端最弱"),
    ("M4·保形：决策相关的数", "cate_interval_report.md", "决策相关"),
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
    ("治理·决策层：护栏能停实验", "governance_report.md", "决策层：护栏触发"),
    ("治理·决策层：服务端复核", "governance_report.md", "服务端自己复核"),
    ("治理·没做清单：检查器会抓过时声明", "governance_report.md", "没做」的清单"),
    ("治理·没做清单：9 条机检", "governance_report.md", "机检 9 条"),
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
