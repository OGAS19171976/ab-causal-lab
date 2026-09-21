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
#:
#: 现在的状态：**空**。曾经挂在里面的五条（簇级 CUPED、M2 决策层、数仓比值链路、
#: mSPRT 的 tau、时间安慰剂、非线性敏感性、策略学习、RDD）都被做出来了，
#: 每一次都是这条检查先把 README 顶红、再改文档 —— 清单空着不是"没有未做事项"，
#: 而是"所有**能机检**的未做事项都被清掉了"；剩下三条无法机检的见
#: ``human_reviewed_notes()``。清单空着也意味着检查器现在抓不到新的过时声明，
#: 所以新增"没做"时**照旧要往这里加一条**，否则那句话又回到没人管的状态。
ITEMS: tuple[UnimplementedItem, ...] = (
)


def human_reviewed_notes() -> tuple[str, ...]:
    """**无法机检**的"没做"（清单不假装覆盖它们，但要列出来提醒人去读）。"""
    return (
        "没有真实流量：清单只能钉住「没有加载真实流量的代码路径」，"
        "数据里到底有没有真实流量，得人看。",
        "前端**没有行为测试**：契约（端点/选择器/语法）已经由 "
        "scripts/check_frontend.py 机检，但「点了按钮会不会真的做对」仍然"
        "只由 HTTP 层测试与人工看一眼覆盖 —— 端到端（Playwright）与已有 HTTP "
        "测试重叠度高，是**有意不做**，不是漏了。",
        "类型只覆盖**接口形状**，不覆盖**数值语义**：scipy 的 optimize 收不收敛、"
        "sklearn 的随机性、pandas 的隐式类型转换都不在类型系统里 —— "
        "「哪些包没带 py.typed」已经被 scripts/check_typed_deps.py 机检了，"
        "剩下这一句只能人读。",
    )


__all__ = ["ITEMS", "EvidenceKind", "UnimplementedItem", "human_reviewed_notes"]
