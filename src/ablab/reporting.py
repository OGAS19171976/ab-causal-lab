"""报告的写出规则：``reports/*.md`` 里只留**在别的机器上也能复现**的内容。

为什么需要这条规则
------------------
``reports/`` 是这个项目"数字可信"的**证据**，而证据有两个硬要求：

1. **换台机器跑，报告必须一模一样。** 一旦里面有耗时、绝对路径这类东西，
   "重跑一次、报告没变"这句话就永远无法验证 —— 而那恰恰是最强的一条
   可复现性证据（比任何声称都硬）。实测过：同一份代码重跑，
   ``git diff reports/`` 只应该因为**数字变了**而变化；加 CI 之前它每次都有噪声，
   16 增 14 删全是 "耗时 61s -> 62s"，真正的数字变化会被淹掉。

2. **别把本地路径写进版本库。** 一条 ``D:\\deep seek\\...\\build\\warehouse.duckdb``
   对别人毫无用处，还顺手泄露了工作目录。

所以规则是：

    耗时与绝对路径 → 只打到控制台（以及 ``build/checks/*.log``，那些不进版本库）
    能被复现的东西（样本量、参数、统计量、结论）→ 进 reports/

做法是**在写出这一处统一过滤**，而不是逐个改 ``say(...)`` 调用。理由是：
将来有人加一行带耗时的输出时，不需要记得"这条不能进报告"——
规则已经在那儿了。

这个过滤器的**已知弱点**（实测踩到过）
----------------------------------------
它按**措辞**匹配（"耗时/总耗时"），而措辞是开放集合。``run_m6_validation.py``
里有一行写成 ``（400 个 salt，{t:.0f}s）``，没带"耗时"两个字，于是整行绕过过滤器，
m6 报告每次重跑都 diff 一行 —— 而且第一次全量实测时它被淹在"36 个文件一致"里。

修法不是把正则加宽到"任何 ``数字+s``"：那会把 ``（400 个 salt，71s）`` 削成
``（400 个 salt，）``，产出更难看，且**掩盖**了调用点的问题。正确做法是
* 过滤器只管措辞（它要产出干净正文），
* 不变量交给 ``tests/test_reporting.py`` 的 ``_DURATION_TOKEN``（按**形状**断言），

一旦那个断言响了，就去改调用点让它带上"耗时"两个字。**宁可报错，不要静默改坏。**
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Comparison",
    "NumericDiff",
    "compare_report_texts",
    "for_report",
    "relativize_paths",
    "report_text",
    "strip_timings",
]

#: "耗时 61s" / "，耗时 6s" / "总耗时 178s" / "；总耗时 12.3s"
_TIMING = re.compile(r"[，；、,]?\s*(?:总)?耗时\s*[\d.]+s")

#: Windows 盘符路径与类 Unix 的家目录路径。
#: 字符类里**不含空格** —— 这个限制是有意的：让正则去猜"路径到哪儿结束"不可靠
#: （本项目的工作目录恰好叫 ``D:\deep seek``，一含空格就会被截成 ``D:\deep``）。
#: 项目内的路径改用**已知根做字面替换**处理，这个正则只兜底项目外的路径。
_ABS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s，；、）)」』\]]*|/(?:home|Users)/[^\s，；、）)\]]*")

#: 字面替换掉根之后，`<root>\build\x.duckdb` 里剩下的那截
_ROOT_TAIL = re.compile(r"<root>[\\/]([^\s，；、）)」』\]]*)")

#: 行首残留的标点（剥掉耗时之后可能留下"，图：xxx"）
_LEADING_PUNCT = re.compile(r"^[，；、,）)」』\]\s]+")


def strip_timings(line: str) -> str:
    """去掉一行里的耗时片段，并清理因此留下的孤立标点。"""
    cleaned = _TIMING.sub("", line)
    return _LEADING_PUNCT.sub("", cleaned).rstrip() if "耗时" in line else cleaned.rstrip()


def relativize_paths(line: str, root: Path | None = None) -> str:
    """把项目内的绝对路径换成**相对路径**；项目外的换成占位符。

    实现上不用正则去匹配"一个完整路径" —— 那是猜，且遇到含空格的目录名必错
    （本项目自己的路径里就有空格）。做法是先拿**已知的根**做字面替换，
    再把紧跟其后的那一段路径转成正斜杠；剩下匹配不到根的绝对路径才用正则兜底。
    """
    out = line
    if root is not None:
        base = str(Path(root).resolve())
        for variant in (base, base.replace("\\", "/"), base.replace("/", "\\")):
            out = out.replace(variant, "<root>")
        out = _ROOT_TAIL.sub(lambda m: m.group(1).replace("\\", "/"), out)
        out = out.replace("<root>", ".")

    def _sub(match: re.Match[str]) -> str:
        raw = match.group(0)
        if root is not None:
            try:
                return str(Path(raw).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
            except (ValueError, OSError):
                pass
        return "<绝对路径>"

    return _ABS_PATH.sub(_sub, out)


def for_report(lines, *, root: Path | None = None) -> list[str]:
    """把控制台日志转成报告正文：剥掉机器相关信息，并丢掉**因此**变空的行。

    注意"因此"两个字。第一版写成"结果为空就丢掉"，于是把日志里**本来就有的空行**
    也一并删了 —— 报告的段落结构被压平，m5 的 diff 一下变成 38 行，
    而真正的原因只是少了个判断。空行是排版，不是噪声。
    """
    out: list[str] = []
    for raw in lines:
        text = str(raw)
        line = relativize_paths(strip_timings(text), root=root)
        # 原本就空的行（排版）保留；只有"剥完之后变空"的才丢
        if line.strip() or not text.strip():
            out.append(line)
    return out


def report_text(lines, *, root: Path | None = None) -> str:
    return "\n".join(for_report(lines, root=root))


# --------------------------------------------------------------------------- #
# 报告比对的第二个判据：正文不变时，数值末位允许差多少
# --------------------------------------------------------------------------- #
#: 一个数：整数、小数、科学计数法都算。故意**不**要求前后是分隔符 ——
#: 报告里 `effect=27.2482063215`、`(0.0400, 0.0700)`、`1.0e-13` 都得能切出来。
#:
#: **这个捕获组是承重的，不是排版。** ``re.split`` 只在模式里出现**显式捕获组**时
#: 才把匹配到的分隔符留在结果里，否则当成普通分隔符**丢掉**。第一版把整条模式
#: 写成非捕获组（``(?:...)``），于是 ``"…effect=-3.9243251037\n"`` 被切成
#: ``['…effect=', '\n']`` —— 数字凭空消失、交替结构塌成"文本、文本"，
#: 后面就拿着 ``'\n'`` 去 ``float()``。测试当场炸了，这就是它的价值。
_NUMBER = re.compile(r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")

#: 数值比对容差。**和审计自己用的容差同源**（``CrossValidation._close``、
#: ``SourceEquivalence.agree`` 都是 1e-9）—— 这样"报告没变"和"结论一致"
#: 在同一个尺度上说话，而不是各定一套。
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


@dataclass(frozen=True)
class NumericDiff:
    """一处数值差异（同一个位置上的两个数）。"""

    before: str
    after: str
    rel_diff: float


@dataclass(frozen=True)
class Comparison:
    """两份报告正文的比对结果。

    * ``identical``：逐字节相同 —— 最强的结论；
    * ``text_identical``：**非数值内容**完全一样（措辞、排版、结构、样本量都没动）；
    * ``worst``：**实测**最大的那处数值差异（用来量化"到底差多少"）；
    * ``exceeds``：唯一能判失败的东西 —— 超出容差的最大那处差异。

    为什么要分两级：浮点求和的末位**依赖执行环境**（求和顺序、BLAS 选到的
    内核、有没有 FMA），这是任何"重跑两次"实验都消不掉的。实测到过同一个量
    在两次独立建仓之间差 1 ulp（``-3.9243251037`` → ``…038``），而它在报告里
    印到小数点后 6 位，读者根本看不见 —— 也就是说**逐字节判据会把一个
    纯环境噪声报成失败**，而失败次数多了，这条检查就会被当成"反正它老是红的"。
    所以判据做成两级：先是"正文一个字都没变"，再把数值差异**量化**出来。

    容差是 ``atol + rtol * max(|before|, |after|)``，两个常数都**照抄审计自己
    用的尺度**（``CrossValidation._close`` / ``SourceEquivalence.agree`` 都是
    1e-9），这样"报告没变"和"结论一致"说的是同一个话。``atol`` 那一项是必须的：
    报告里有一类量**本身就是舍入噪声**（三条路径的"最大偏差"就是两个几乎相等的
    数相减），它在两次运行间能从 ``2.274e-13`` 变成 ``1.137e-13`` —— 相对差 50%，
    绝对差 1e-13。只比相对值会把它判失败，而它比任何结论都小十来个数量级。
    """

    identical: bool
    text_identical: bool
    worst: NumericDiff | None
    exceeds: NumericDiff | None
    #: 数值位置上的差异处数（含容差内的）
    n_numeric_diffs: int
    #: 第一处差异的可读描述，用于定位
    first_diff: str | None

    @property
    def within_tolerance(self) -> bool:
        return self.text_identical and self.exceeds is None

    @property
    def ok(self) -> bool:
        """放行条件：逐字节相同，或者"正文没变 + 数值只在容差内"。"""
        return self.identical or self.within_tolerance


def _tokenize(text: str) -> list[str]:
    """把正文切成"文本 / 数字"交替的 token；数字带标记。"""
    parts = _NUMBER.split(text)
    # re.split 带捕获组：偶数下标是文本，奇数下标是数字
    tokens: list[str] = []
    for i, part in enumerate(parts):
        tokens.append(("#" if i % 2 else "T") + part)
    return tokens


def _relative_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else abs(a - b)


def compare_report_texts(
    before: str,
    after: str,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Comparison:
    """比对两份报告正文，允许数值在 ``rtol``/``atol`` 内不同。

    只有当**同样位置的 token 类型也对得上**时才逐位置比较；
    一旦结构不同（比如一边多出一个数、少了一行），就直接判 ``text_identical=False``，
    因为那时逐位置比较已经失去意义 —— 与其猜对齐，不如报"正文变了"。
    """
    if before == after:
        return Comparison(True, True, None, None, 0, None)

    tb, ta = _tokenize(before), _tokenize(after)
    if len(tb) != len(ta):
        return Comparison(
            False,
            False,
            None,
            None,
            0,
            f"token 数不同（{len(tb)} vs {len(ta)}）—— 结构或行数变了",
        )

    worst: NumericDiff | None = None
    exceeds: NumericDiff | None = None
    n_diffs = 0
    first: str | None = None
    for i, (b, a) in enumerate(zip(tb, ta)):
        if b == a:
            continue
        b_num, a_num = b.startswith("#"), a.startswith("#")
        if not (b_num and a_num):
            return Comparison(
                False,
                False,
                worst,
                exceeds,
                n_diffs,
                f"token {i} 不是同一类（{b[:40]!r} vs {a[:40]!r}）—— 正文变了",
            )
        bv, av = float(b[1:]), float(a[1:])
        rel = _relative_diff(bv, av)
        n_diffs += 1
        diff = NumericDiff(b[1:], a[1:], rel)
        if first is None:
            first = f"{b[1:]} -> {a[1:]}（相对偏差 {rel:.3e}）"
        if worst is None or rel > worst.rel_diff:
            worst = diff
        if abs(bv - av) > atol + rtol * max(abs(bv), abs(av)) and (
            exceeds is None or rel > exceeds.rel_diff
        ):
            exceeds = diff

    return Comparison(False, True, worst, exceeds, n_diffs, first)
