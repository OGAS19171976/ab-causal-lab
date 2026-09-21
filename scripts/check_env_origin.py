#!/usr/bin/env python
"""**环境来源检查**：锁文件不决定环境，来源也要核对。

为什么需要它
------------
``scripts/lock_requirements.py --check`` 比的是**版本**（从已装的包里读
``dist.version`` 再与锁文件对），所以它看不见一件事：**包是从哪来的**。

这个仓库的本地 venv 曾经是**混合环境**：``.venv/pyvenv.cfg`` 里
``include-system-site-packages = true``，于是 ``requirements.lock`` 里
48 个包中有 **14 个**（pandas、matplotlib、pytest、packaging…）实际解析自
**系统 Python 的 site-packages**，venv 里根本没有它们 —— 而 ``--check``
照样全绿，因为版本号恰好一样。CI 是干净的（``setup-python`` + 全新的
``pip install -r requirements.lock``），也就是说**本地与 CI 跑的不是同一套文件**。

判据（三条，任一条不成立就红）
------------------------------
  1. ``include-system-site-packages`` 必须是 ``false``；
  2. 锁文件里每个**发行版**的安装位置必须在 venv 内；
  3. 仓库源码里 import 的每个顶层模块，必须是标准库 / 本项目 / 锁文件里的包
     —— 第 3 条抓的是"能 import 但没人锁"的东西（它今天能跑，明天在 CI 上就没了）。

诚实的边界：这一条检查"装在哪"，不检查"装的东西对不对"（那是锁文件的哈希/
版本的事）。两者都要有，缺一个都会漏。
"""

from __future__ import annotations

import ast
import importlib.metadata as md
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock"
VENV = ROOT / ".venv"
SITE = VENV / "Lib" / "site-packages"
SRC = ROOT / "src"

#: 发行版名 → 实际 import 的顶层模块名（两者不同的那几种）
DIST_TO_MODULE = {
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "python-multipart": "multipart",
    "typing-extensions": "typing_extensions",
    "annotated-types": "annotated_types",
    "pygments": "pygments",
}

#: 源码里允许出现的"非标准库、非本项目、非锁文件"的顶层模块（**必须带理由**）
ALLOWED_EXTRA: dict[str, str] = {}


def lock_pins() -> list[str]:
    """锁文件里的发行版名（去掉 marker 与注释）。"""
    names = []
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        names.append(line.split("==")[0].strip())
    return names


def module_of(dist_name: str) -> str:
    return DIST_TO_MODULE.get(dist_name.lower(), dist_name.replace("-", "_").lower())


def check_flag() -> list[str]:
    cfg = (VENV / "pyvenv.cfg").read_text(encoding="utf-8")
    problems = []
    for line in cfg.splitlines():
        if line.startswith("include-system-site-packages"):
            value = line.split("=", 1)[1].strip().lower()
            if value != "false":
                problems.append(
                    f"include-system-site-packages = {value}（必须是 false —— "
                    "否则系统 Python 的包会漏进这个环境，锁文件就管不住它）"
                )
    return problems


def check_origins() -> tuple[list[str], list[tuple[str, str]]]:
    problems, rows = [], []
    for name in lock_pins():
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            problems.append(f"{name}: 锁文件里有，但这个环境里没装")
            continue
        loc = pathlib.Path(str(dist.locate_file(""))).resolve()
        inside = SITE.resolve() in loc.parents or loc == SITE.resolve()
        rows.append((name, "venv" if inside else f"**外面** {loc}"))
        if not inside:
            problems.append(
                f"{name}: 来自 venv 之外（{loc}）—— 本地与 CI 跑的不是同一套文件"
            )
    return problems, rows


def _stdlib_names() -> set[str]:
    return set(sys.stdlib_module_names)


def source_imports() -> dict[str, list[str]]:
    """源码里出现的顶层 import → 出现在哪些文件（相对路径）。"""
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - 语法错由 ruff/mypy 抓
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # 相对 import：本项目内部
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for n in names:
                if n:
                    found.setdefault(n, []).append(str(path.relative_to(ROOT)))
    return found


def check_imports() -> tuple[list[str], list[tuple[str, str]]]:
    locked = {module_of(n) for n in lock_pins()}
    problems, rows = [], []
    stdlib = _stdlib_names()
    for module, files in sorted(source_imports().items()):
        if module in stdlib:
            where = "标准库"
        elif module == "ablab" or module in {"tests", "build", "scripts"}:
            where = "本项目"
        elif module in locked:
            where = "锁文件"
        elif module in ALLOWED_EXTRA:
            where = f"白名单（{ALLOWED_EXTRA[module]}）"
        else:
            where = "**没人锁**"
            problems.append(
                f"{module}: 源码里 import 了它，但它既不是标准库、也不是本项目、"
                f"也不在 requirements.lock 里（出现在 {', '.join(sorted(set(files))[:3])}）"
            )
        rows.append((module, where))
    return problems, rows


def main() -> int:
    problems = check_flag()
    origin_problems, origin_rows = check_origins()
    import_problems, import_rows = check_imports()
    problems += origin_problems + import_problems

    print("环境来源检查（锁文件管版本，这一条管**来源**）")
    print(f"  venv: {VENV}")
    print(f"  site-packages: {SITE}")
    print()
    print(f"  一、锁文件里的 {len(origin_rows)} 个发行版装在哪")
    outside = [name for name, where in origin_rows if where != "venv"]
    print(f"    venv 内 {len(origin_rows) - len(outside)} 个 / 外面 {len(outside)} 个")
    for name in outside:
        print(f"      ** {name} 不在 venv 里")
    print()
    untyped = [r for r in import_rows if r[1] == "**没人锁**"]
    print(f"  二、源码 import 的顶层模块：共 {len(import_rows)} 个，"
          f"没人锁的 {len(untyped)} 个")
    categories: dict[str, int] = {}
    for _, where in import_rows:
        categories[where] = categories.get(where, 0) + 1
    for where, n in sorted(categories.items()):
        print(f"    {where:<12}{n}")
    print()
    if problems:
        print(f"**{len(problems)} 条不成立**（环境来源与锁文件对不上）：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("环境来源与锁文件一致：所有包都来自 venv，源码没有未锁的 import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
