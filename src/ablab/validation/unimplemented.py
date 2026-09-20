"""**"没做"的清单**：每条未做事项都必须带一份机器可核对的证据。

为什么要这个东西
----------------
这个仓库已经栽过三次同一类跟头：**"没做"的声明在功能做完之后没人回来改**。

  * 簇级 CUPED 被三处代码拒绝，理由是"数据源没有簇级前置指标" ——
    而 05 路 DWS 早就落了那一列（那条注释甚至写着"将来要做时不必改这一层"）；
  * M2"没做决策层"在护栏决策层做完之后还挂在已知边界里；
  * "数仓不支持比值指标"在 06/07 两条 SQL 上线之后还挂着。

共同点：**"没做"是一句无法被核对的话**。README 里每个**数字**都有人对
（``scripts/check_readme_claims.py``），但"没做"没人对 —— 于是它只会朝
一个方向漂移：**越来越不准**（功能做了，话还留着）。

这个模块把每一句"没做"变成一条带**证据**的记录：
证据必须**现在还成立**（例如某个符号确实不存在）；一旦它不成立了，
说明功能已经做出来，而 README 没改 —— ``scripts/check_unimplemented.py``
会在检查集里直接红，并告诉你该改哪一条。

三类证据
--------
* ``symbol_absent``：点分路径**必须不存在**。用于"某个类/函数没有实现"。
  同时用 ``anchor_present`` 钉住"这个模块还在" —— 否则改个名就能骗过检查。
* ``text_absent``：某个字面串**必须不在**指定目录里。用于"没有这条技术路线"。
* ``file_absent``：某个文件**必须不存在**。用于"没有这张表/这个脚本"。

诚实的边界：**这份清单只覆盖能机检的那些"没做"**。像"没有真实流量"这种
无法用符号表达的，仍然只能靠人读 —— 清单不会假装覆盖了它们（见
``check_unimplemented.py`` 的输出里那句"人工核对的条数"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvidenceKind = Literal["symbol_absent", "text_absent", "file_absent"]


@dataclass(frozen=True)
class UnimplementedItem:
    """一条"未做事项" + 它的可核对证据。"""

    id: str
    #: README 里那句"没做"的**逐字片段**（清单与文档必须对得上）
    readme_phrase: str
    #: 证据种类
    kind: EvidenceKind
    #: symbol_absent: 点分路径；text_absent: 要搜的字面串；file_absent: 相对仓库根的路径
    target: str
    #: text_absent 时的搜索目录（相对仓库根）
    scope: str = "src"
    #: 钉住"这个模块还在"，避免改名骗过检查（symbol_absent 时用）
    anchor_present: str = ""
    #: 做出来之后 README 大概要改成什么（写给人看的一句话）
    when_done: str = ""
    notes: list[str] = field(default_factory=list)


#: 清单本体。**只放能机检的**；无法机检的归到人工核对，不混进来充数。
ITEMS: tuple[UnimplementedItem, ...] = (
    UnimplementedItem(
        id="rdd",
        readme_phrase="断点回归还没有",
        kind="symbol_absent",
        target="ablab.causal.rdd",
        # IV 那一半已经做出来了（ablab.causal.iv，见 m3 报告 2.8 节），
        # 所以这条从 "IV/RDD" 收窄成 "RDD"；锚点换成 iv 是为了钉住
        # "模块还在、只是 RDD 那一半没做"，避免改名骗过检查。
        anchor_present="ablab.causal.iv",
        when_done="README 的 M3 那条要改成「已实现 RDD」，并补验收证据",
    ),
    UnimplementedItem(
        id="policy_optimization",
        readme_phrase="策略学习只做到 CATE 排序",
        kind="symbol_absent",
        target="ablab.causal.policy.aipw_policy_learner",
        # R/DR-learner 这一轮做完了（``causal.hte.r_learner`` / ``dr_learner``，
        # 见 m4 报告 3c 节），所以条目收窄成"直接优化策略价值"那一层。
        # 锚点钉在 hte 上：模块还在、只是更进一步的策略优化没做。
        anchor_present="ablab.causal.hte",
        when_done=(
            "README 要改成「实现了直接优化策略价值的策略学习」，"
            "并给出策略价值（AIPW 意义下）相对 CATE 排序策略的实测对照"
        ),
    ),
    UnimplementedItem(
        id="msprt_tau_choice",
        readme_phrase="mSPRT 的 `tau` 没有自动选择",
        kind="symbol_absent",
        target="ablab.inference.tests.choose_tau",
        anchor_present="ablab.inference.tests",
        when_done="README 要说明 tau 的选择规则与它对功效的影响实测",
    ),
    UnimplementedItem(
        id="scm_time_placebo",
        readme_phrase="合成控制只做了空间安慰剂",
        kind="symbol_absent",
        target="ablab.causal.synthetic.time_placebo",
        anchor_present="ablab.causal.synthetic",
        when_done="README 要补时间安慰剂/留一法的实测",
    ),
    UnimplementedItem(
        id="sensitivity_smoothness",
        readme_phrase="敏感性分析只做了线性违背",
        kind="symbol_absent",
        target="ablab.causal.sensitivity.rambachan_roth_smoothness",
        anchor_present="ablab.causal.sensitivity",
        when_done="README 要补相对幅度/平滑约束版本的实测",
    ),
    UnimplementedItem(
        id="warehouse_cluster_aa",
        readme_phrase="数仓路径上还没有簇级的 A/A 校准",
        kind="symbol_absent",
        target="ablab.warehouse.ratio_calibration.run_cluster_replicate_calibration",
        anchor_present="ablab.warehouse.ratio_calibration.run_ratio_link_power_calibration",
        when_done=(
            "README 要改成「数仓路径上的簇级 A/A 也量过了」，"
            "并给出误停率/覆盖的实测（那时每个复制实验要吃下 ~30 个城市）"
        ),
        notes=[
            "只钉「数仓路径上没有簇级重复实现」，不声称「簇级 CUPED 没校准」——"
            "合成路径上量过（reports/m6_validation.md 第 7 节，误停率 0.0400）",
        ],
    ),
)


def human_reviewed_notes() -> tuple[str, ...]:
    """**无法机检**的"没做"（清单不假装覆盖它们，但要列出来提醒人去读）。"""
    return (
        "没有真实流量：清单只能钉住「没有加载真实流量的代码路径」，"
        "数据里到底有没有真实流量，得人看。",
        "前端没有构建步骤、没有测试：那是工程取舍，不是「没做」，"
        "但也没有东西能自动核对它。",
        "scipy / pandas / sklearn 没有类型保证：取决于上游是否带 py.typed，"
        "只能人看 mypy 的输出。",
    )


__all__ = ["ITEMS", "EvidenceKind", "UnimplementedItem", "human_reviewed_notes"]
