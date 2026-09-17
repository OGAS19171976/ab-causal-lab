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
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["for_report", "report_text", "strip_timings", "relativize_paths"]

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
