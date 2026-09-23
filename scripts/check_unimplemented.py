#!/usr/bin/env python
"""核对 README 里每一句"没做"是否**仍然成立**。

为什么需要它：README 里每个**数字**都有人对（``check_readme_claims.py``），
但"没做"没人对 —— 于是它只会朝一个方向漂移：功能做了、话还留着。
这个仓库已经栽过三次（簇级 CUPED、M2 决策层、数仓比值链路），
每次都是靠人偶然发现。这个脚本把它变成一次机械检查。

判据（三条同时成立才算通过）：

1. README 里那句"没做"**还在**（``readme_phrase`` 逐字可见）；
2. 它的**证据仍然成立**（符号不存在 / 字面串搜不到 / 文件不存在）；
3. 用来钉住上下文的锚点**还在**（防止改个名就骗过检查）。

证据不成立 -> 说明**已经做出来了**：报错并告诉人该改 README 的哪一句。

用法::

    python scripts/check_unimplemented.py            # 检查
    python scripts/check_unimplemented.py --list     # 只列清单
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.validation.unimplemented import (  # noqa: E402
    ITEMS,
    NOT_DONE_MARKERS,
    NOT_DONE_STATEMENTS,
    UnimplementedItem,
    human_reviewed_notes,
)


def _symbol_exists(dotted: str) -> bool:
    """点分路径是否可解析（``ablab.causal.did`` 或 ``ablab.causal.did.foo``）。"""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        module_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


#: 搜索时要跳过的文件：**清单自己**。
#:
#: 为什么必须跳过：清单里写着 ``target="load_real_traffic"``，
#: 于是"在 src/ 下搜这个串"会搜到清单本身 —— 第一次跑就报了假阳性
#: （"看起来功能做出来了"）。检查器自己也会骗人，所以这条写在代码里。
_SELF: frozenset[Path] = frozenset(
    {Path("src/ablab/validation/unimplemented.py").resolve()}
)


def _text_present(scope: str, needle: str) -> bool:
    directory = ROOT / scope
    for path in directory.rglob("*.py"):
        try:
            if path.resolve() in _SELF:
                continue
            if needle in path.read_text(encoding="utf-8"):
                return True
        except (UnicodeDecodeError, OSError):  # pragma: no cover - 读不了就跳过
            continue
    return False


def check_item(item: UnimplementedItem, readme: str) -> tuple[bool, str]:
    """返回 ``(通过?, 说明)``。"""
    if item.readme_phrase not in readme:
        return False, (
            f"README 里找不到这句「{item.readme_phrase}」——"
            "是被人删了，还是措辞改了？清单与文档必须对得上"
        )

    if item.kind == "symbol_absent":
        if item.anchor_present and not _symbol_exists(item.anchor_present):
            return False, (
                f"锚点 {item.anchor_present} 也不存在了 —— 可能整个模块被改名/删除，"
                "请人工确认这条『没做』还成不成立"
            )
        if _symbol_exists(item.target):
            return False, (
                f"**{item.target} 已经存在** —— 看起来功能做出来了，"
                f"而 README 还写着「{item.readme_phrase}」。{item.when_done}"
            )
        return True, f"符号 {item.target} 仍不存在 ✓"

    if item.kind == "text_absent":
        if _text_present(item.scope, item.target):
            return False, (
                f"**在 {item.scope}/ 下搜到了「{item.target}」** ——"
                f"看起来功能做出来了。{item.when_done}"
            )
        return True, f"{item.scope}/ 下没有「{item.target}」✓"

    if item.kind == "file_absent":
        if (ROOT / item.target).exists():
            return False, (
                f"**{item.target} 已经存在** —— 看起来做出来了。{item.when_done}"
            )
        return True, f"{item.target} 仍不存在 ✓"

    raise ValueError(f"未知的证据种类：{item.kind!r}")


def _plain(line: str) -> str:
    """去掉 markdown 的强调与反引号，供**逐字**比对用。

    不做这一步，"``**业务损失函数**`` 自动选择上线与否"这种加粗会让登记表里的
    普通写法（"业务损失函数自动选择上线与否"）白白搜不到。
    """
    for token in ("**", "`", "~~", "*"):
        line = line.replace(token, "")
    return line


def active_not_done_lines(readme: str) -> list[tuple[int, str]]:
    """README 里**活跃**的"没做"句子：``(行号, 原文)``。

    两个判据，都是为了把**叙述**排除掉：
    * 必须带 ``NOT_DONE_MARKERS`` 里的粗体标记（作者显式声明）；
    * 被 ``~~`` 划掉的一律算历史（"后来补上了"）。
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(readme.splitlines(), start=1):
        if "~~" in line:
            continue
        if any(marker in line for marker in NOT_DONE_MARKERS):
            found.append((number, line))
    return found


def check_not_done_coverage(readme: str) -> list[str]:
    """**覆盖面**：被标记的句子 ↔ 登记表，两头都要对得上。

    这一层不检查"做没做"（做不到），它检查**有没有人管**：
    新写一句"没做"却没挂号 → 红；挂了号的句子从 README 里消失了（做完了、
    或改了措辞）→ 也红。两头都红，是因为这两件事都需要人回去改文档。
    """
    problems: list[str] = []
    matched: set[str] = set()
    for number, line in active_not_done_lines(readme):
        hits = [s.key for s in NOT_DONE_STATEMENTS if s.phrase in _plain(line)]
        if not hits:
            problems.append(
                f"README:{number} 是一句被标记的『没做』，但登记表（NOT_DONE_STATEMENTS）"
                f"里没有对应条目 —— 要么给它挂个 key，要么说明它其实已经做完："
                f"{_plain(line).strip()[:70]}"
            )
        elif len(hits) > 1:
            problems.append(
                f"README:{number} 同时匹配了 {hits} —— 一句『没做』只该挂一个 key"
            )
        else:
            matched.add(hits[0])
    for statement in NOT_DONE_STATEMENTS:
        if statement.key not in matched:
            problems.append(
                f"登记表里的 {statement.key}（{statement.phrase!r}）在 README 里找不到了 "
                "—— 是做完了，还是那句话被改了措辞？两条路都要改文档"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="核对『没做』的清单")
    ap.add_argument("--list", action="store_true", help="只列清单，不检查")
    args = ap.parse_args()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    if args.list:
        print(f"机检的『没做』共 {len(ITEMS)} 条：")
        for item in ITEMS:
            print(f"  [{item.kind}] {item.id}: {item.readme_phrase}")
        print(f"\n正文里逐字登记的『没做』共 {len(NOT_DONE_STATEMENTS)} 条：")
        for statement in NOT_DONE_STATEMENTS:
            print(f"  [{statement.key}] {statement.phrase} —— {statement.why}")
        print(f"\n人工核对的『没做』共 {len(human_reviewed_notes())} 条（清单不覆盖）：")
        for note in human_reviewed_notes():
            print(f"  - {note}")
        return 0

    failed: list[str] = []
    for item in ITEMS:
        ok, detail = check_item(item, readme)
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {item.id}: {detail}")
        if not ok:
            failed.append(item.id)

    coverage = check_not_done_coverage(readme)
    marked = active_not_done_lines(readme)
    print(
        f"[{'OK  ' if not coverage else 'FAIL'}] 正文『没做』的覆盖面："
        f"被标记的句子 {len(marked)} 条，登记表 {len(NOT_DONE_STATEMENTS)} 条"
        + ("（两头顶得上）" if not coverage else "（有对不上的，见下）")
    )
    for problem in coverage:
        print(f"       - {problem}")
    item_failures = [f for f in failed if f != "not_done_coverage"]
    if coverage:
        failed.append("not_done_coverage")

    print()
    print(
        f"机检 {len(ITEMS)} 条『没做』：{len(ITEMS) - len(item_failures)} 条仍成立"
        f"，{len(item_failures)} 条已经过时。"
    )
    print(
        f"另有 {len(NOT_DONE_STATEMENTS)} 条**无法机检**、只能在正文里逐字登记 ——"
        "「有没有人管」是机检的（两头顶不上就红），「做没做」还是只能人读。"
    )
    print(f"以及 {len(human_reviewed_notes())} 条**取舍声明**（有意不做，不是欠账）。")
    if failed:
        print()
        print("**失败：下面这些『没做』的登记与 README 对不上，请更新 README 与该清单**：")
        for item_id in failed:
            print(f"  - {item_id}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
