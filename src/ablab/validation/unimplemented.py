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

诚实的边界，以及**这份清单自己漂移过**（这一轮补的第二层）
----------------------------------------------------------
上面那三类证据只能收"能用符号/文件/串表达的"未做事项。而**大多数"没做"是
正文里的句子**，它们没有这种证据 —— 于是它们绕过了 `ITEMS`，照样漂：
这一轮就抓到三处（路线图里"第二种交错处置估计量（Sun-Abraham 或 BJS）"其实
早已实现、比值数仓的施工口径标题还写着"下一次开工照着做"而它上一段就写着
"已经接通了"、"README 里还没有比值数仓的数字"也不成立）。

所以补了第二层：``NOT_DONE_STATEMENTS`` —— **正文里每一句被标记的"没做"
都必须在这里挂号，挂了号的句子也必须在 README 里还在**。它不检查"做没做"
（那做不到），它检查**有没有人管**：两头对不上就在检查集里红。
写法约定见 ``NOT_DONE_MARKERS``。

于是三层各管一段，别混：
* ``ITEMS``：能机检证据的未做事项（现在**空**）；
* ``NOT_DONE_STATEMENTS``：正文里带标记、逐字登记的"没做"（覆盖性检查）；
* ``human_reviewed_notes()``：**取舍声明**（有意不做，不是欠账），只能人读。
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
#: 现在**又是空的**：`real_traffic` 那条被"做完了"——真的接进了外部数据
#: （MovieLens，610 用户 / 10 万条评分），所以它按规矩离开了这份清单。
#: 净效果不是"少了一条约束"，而是**约束变强了**：
#:   之前：机检"provenance.json 不存在"（证明*没有*真数据）；
#:   现在：`realdata` 检查步骤每个 push 都跑一遍**契约 + 反冒充 + 真接入**
#:         （证明"接进来的确实是外部数据，而且链路真的能跑"）。
#: 前者只能证明"没做"，后者能证明"做对了"——这也是这份清单该有的归宿：
#: 它管"没做"，做完了就换成正面检查。
#:
#: 现在的状态：**空**（``()``）。曾经挂在这里的每一条（簇级 CUPED、M2 决策层、
#: 数仓比值链路、mSPRT 的 tau、时间安慰剂、Rambachan-Roth 三档限制、策略学习、
#: RDD、真实流量）都被做出来了，每一次都是这条检查先把 README 顶红、再改文档。
#: 清单空着不是"没有未做事项"，而是"所有**能机检**的未做事项都被清掉了"；
#: 剩下两条无法机检的见 ``human_reviewed_notes()``。
#: 最后一条 `real_traffic` 的去向值得记：它做完之后**不是消失，而是换了方向** ——
#: 过去它机检"provenance.json 不存在"（证明*没有*真数据），现在由 `realdata`
#: 检查步骤每个 push 跑"契约 + 反冒充 + 真接入"（证明*接进来的确实是外部数据*）。
#: 所以新增"没做"时照旧往这里加一条；而做完了，就该想想能不能换成正面检查。
ITEMS: tuple[UnimplementedItem, ...] = ()


#: 活跃"没做"的**写法约定**：粗体标记，且粗体**以这三个词开头**。
#:
#: 为什么要有这个约定、而不是直接扫"未做"三个字：README 里还有大量**引用与
#: 回顾**（"当时留了一句『仍未做：……』"、"~~仍未做：断点回归~~ —— 后来补上了"）。
#: 把叙述当成活跃声明，检查器第一天就会被假阳性淹没，然后被人关掉 ——
#: 所以用"粗体标记"这个**作者显式作出的声明**当判据，被划掉的（``~~``）一律算历史。
#:
#: 判据取**前缀**（``**仍未做``）而不是全形（``**仍未做**``）：后者会漏掉
#: ``**仍未做：真实效应下的功效/覆盖**`` 这种"整句加粗"的写法 —— 而**漏掉**
#: 恰恰是最危险的失效方向（没人管的那句悄悄滑过检查）。这一条是写检查器的同一轮
#: 实测出来的：登记表建好之后，扫描器只认全形，于是那条句子一条都没被看见。
NOT_DONE_MARKERS: tuple[str, ...] = ("**未做", "**仍未做", "**没有做")


@dataclass(frozen=True)
class NotDoneStatement:
    """正文里一句**活跃的**"没做"，逐字登记。

    ``phrase`` 必须出现在那条被标记的句子里 —— 比对前会把 markdown 的
    ``*`` 与反引号去掉（否则 ``**业务损失函数**`` 这种加粗会让逐字比对白白失败）。
    判据与声明清单同一个套路：**用"逐字能搜到"换掉"语义相似"**，
    因为后者的判据会随人变。
    """

    key: str
    #: 必须出现在那条被标记句子里的**逐字片段**
    phrase: str
    #: 为什么它还是"没做"（人读的一句话）
    why: str


#: 活跃的"没做"（本仓库目前的全部）。
#:
#: 加一条时问自己两件事：它**真的**还没做吗？以及——它能不能升级成 ``ITEMS``
#: 里那种有证据的检查（能的话就别放在这里，那里更强）。
NOT_DONE_STATEMENTS: tuple[NotDoneStatement, ...] = (
    NotDoneStatement(
        key="monitor_looks_not_comparable",
        phrase="各次查看之间不再可比",
        why="监控曲线跨查看点不可比是另一个（更麻烦的）问题，本仓库不做",
    ),
    NotDoneStatement(
        key="business_loss_function",
        phrase="按业务损失函数自动选择上线与否",
        why="要先把收益/损失的效用函数写成声明，平台现在只有统计判据",
    ),
    NotDoneStatement(
        key="donor_pool_common_shocks",
        phrase="捐赠池本身被共同冲击污染",
        why="合成控制需要交互固定效应一类的方法，本仓库的 SCM 没做这一档",
    ),
    NotDoneStatement(
        key="second_covariate_in_warehouse",
        phrase="把第二个协变量接进 DWS/ADS",
        why="ADS 只落了一个 pre_sum；多协变量目前只能在明细/合成路径上算",
    ),
    NotDoneStatement(
        key="cluster_pairing",
        phrase="簇级配对/协变量调整",
        why="整簇随机化下设计层面的补救（配对、协变量调整）没有做",
    ),
    NotDoneStatement(
        key="token_lifecycle_rest",
        phrase="刷新令牌/撤销列表",
        why="静态 token 补了有效期/轮换/限速，但刷新令牌、撤销列表、多设备管理、"
        "自助轮换端点与限速共享存储都没有",
    ),
    NotDoneStatement(
        key="ratio_real_effect_power",
        phrase="真实效应下的功效/覆盖",
        why="比值链路的复制实验共享同一份结果序列、真实效应为 0，所以只能校准零效应",
    ),
)


def human_reviewed_notes() -> tuple[str, ...]:
    """**无法机检**的"没做"（清单不假装覆盖它们，但要列出来提醒人去读）。"""
    return (

        "前端**没有行为测试**：契约（端点/选择器/语法）已经由 "
        "scripts/check_frontend.py 机检，但「点了按钮会不会真的做对」仍然"
        "只由 HTTP 层测试与人工看一眼覆盖 —— 端到端（Playwright）与已有 HTTP "
        "测试重叠度高，是**有意不做**，不是漏了。",
        "类型只覆盖**接口形状**，不覆盖**数值语义**：scipy 的 optimize 收不收敛、"
        "sklearn 的随机性、pandas 的隐式类型转换都不在类型系统里 —— "
        "「哪些包没带 py.typed」已经被 scripts/check_typed_deps.py 机检了，"
        "剩下这一句只能人读。",
    )


__all__ = [
    "ITEMS",
    "NOT_DONE_MARKERS",
    "NOT_DONE_STATEMENTS",
    "EvidenceKind",
    "NotDoneStatement",
    "UnimplementedItem",
    "human_reviewed_notes",
]
