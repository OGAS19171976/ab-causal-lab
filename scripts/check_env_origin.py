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
  1. 若仓库目录下有 ``.venv``，它的 ``include-system-site-packages`` 必须是 ``false``
     （CI 用裸解释器时这一条不适用 —— 但会**显式打印**"不适用"，不静默跳过）；
  2. 锁文件里每个**发行版**的安装位置必须在**当前解释器的 purelib** 内；
  3. 仓库源码里 import 的每个顶层模块，必须是标准库 / 本项目 / 锁文件里的包
     —— 第 3 条抓的是"能 import 但没人锁"的东西（它今天能跑，明天在 CI 上就没了）。

诚实的边界：这一条检查"装在哪"，不检查"装的东西对不对"（那是锁文件的哈希/
版本的事）。两者都要有，缺一个都会漏。
"""

from __future__ import annotations

import ast
import importlib.metadata as md
import pathlib
import site
import sys
import sysconfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock"
VENV = ROOT / ".venv"
SRC = ROOT / "src"

#: "当前这套解释器把包装在哪" —— 用 sysconfig 而不是拼 .venv 路径。
#: 第一版写死了 ``ROOT/.venv/Lib/site-packages``，本地是对的，**CI 上直接红**：
#: GitHub Actions 的 setup-python 装的是一个**裸解释器**（没有 .venv，
#: 包进 tool cache 的 site-packages）。检查的判据应该是"跑测试的这套包
#: 必须都来自当前解释器的 purelib"，而不是"必须有一个 .venv 目录" ——
#: 后者是把**本地布局**当成了不变量。
PURELIB = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
IS_VENV = sys.prefix != sys.base_prefix
USER_SITE = pathlib.Path(site.getusersitepackages()).resolve() if hasattr(site, "getusersitepackages") else None

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


def lock_pins() -> tuple[list[str], list[str]]:
    """``(在当前平台上适用的发行版名, 因 marker 不适用的)``。

    **marker 必须判**：锁文件里有 ``colorama ; sys_platform == "win32"``
    这类平台条件依赖，在 Linux 上它们**本来就不该被安装**。第一版直接把
    锁文件每一行都当成"必须装"，于是本地（Windows）全绿、**CI（Linux）红** ——
    报的是"锁文件里有，但这个环境里没装"。判据本身没错，是漏了一步
    "这条在当前平台适用吗"。

    复用 ``lock_requirements.parse_lock`` / ``_marker_applies`` 而不是自己
    再写一遍：同一个规则实现两次，迟早会分叉（这个仓库已经栽过）。
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import lock_requirements as lr

    entries = lr.parse_lock(LOCK)
    applicable: list[str] = []
    skipped: list[str] = []
    for name, (_version, marker) in entries.items():
        (applicable if lr._marker_applies(marker) else skipped).append(name)
    return sorted(applicable), sorted(skipped)


def module_of(dist_name: str) -> str:
    return DIST_TO_MODULE.get(dist_name.lower(), dist_name.replace("-", "_").lower())


def check_flag() -> tuple[list[str], str]:
    """``(问题列表, 这一条是否适用)``。

    ``include-system-site-packages`` 只在"解释器跑在一个 venv 里、而且那个
    venv 就在仓库目录下"时才是一个可核对的开关。CI 用的是裸解释器，
    这一条**不适用** —— 但要说出来，不能默默跳过（静默跳过就是这个仓库
    反复栽过的那种错）。
    """
    cfg = VENV / "pyvenv.cfg"
    if not cfg.exists():
        return [], f"不适用：{VENV} 不存在（本次跑在非 venv 环境：{sys.prefix}）"
    value = "（没有这一行）"
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.startswith("include-system-site-packages"):
            value = line.split("=", 1)[1].strip().lower()
    if value != "false":
        return [
            f"include-system-site-packages = {value}（必须是 false —— 否则系统 "
            "Python 的包会漏进这个环境，锁文件就管不住它）"
        ], f"适用：{cfg} 里读到 {value}"
    return [], f"适用：{cfg} 里读到 false"


def check_origins() -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """每个锁文件发行版的安装位置必须在**当前解释器的 purelib** 里。

    这一条同时抓两种情形：本地那种"venv 看得见系统 site-packages"的混合环境，
    以及任何"包从别的解释器漏进来"的情形 —— 判据是 purelib，与有没有 .venv 无关。
    """
    problems: list[str] = []
    rows: list[tuple[str, str]] = []
    applicable, skipped = lock_pins()
    for name in applicable:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            problems.append(f"{name}: 锁文件里有，但这个环境里没装")
            continue
        loc = pathlib.Path(str(dist.locate_file(""))).resolve()
        inside = PURELIB in loc.parents or loc == PURELIB
        where = "当前解释器" if inside else f"**外面** {loc}"
        rows.append((name, where))
        if not inside:
            hint = ""
            if USER_SITE is not None and parents_or_self(loc, USER_SITE):
                hint = "（这是 **user site**：pip install --user 装的，不属于这套环境）"
            problems.append(
                f"{name}: 来自当前解释器之外（{loc}）{hint} —— "
                "跑测试的这套包与锁文件不是同一份"
            )
    return problems, rows, skipped


def parents_or_self(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or root in path.parents


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
    applicable, _skipped = lock_pins()
    locked = {module_of(n) for n in applicable}
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
    flag_problems, flag_note = check_flag()
    origin_problems, origin_rows, skipped = check_origins()
    import_problems, import_rows = check_imports()
    problems = flag_problems + origin_problems + import_problems

    print("环境来源检查（锁文件管版本，这一条管**来源**）")
    print(f"  解释器: {sys.executable}")
    print(f"  当前解释器的 purelib: {PURELIB}"
          f"{'（venv）' if IS_VENV else '（裸解释器，CI 就是这样）'}")
    print(f"  开关 include-system-site-packages：{flag_note}")
    print()
    print(f"  一、锁文件里的 {len(origin_rows)} 个发行版装在哪")
    outside = [name for name, where in origin_rows if where != "当前解释器"]
    print(f"    当前解释器内 {len(origin_rows) - len(outside)} 个 / 外面 {len(outside)} 个")
    for name in outside:
        print(f"      ** {name} 不在 venv 里")
    print()
    untyped = [r for r in import_rows if r[1] == "**没人锁**"]
    print(f"    另有 {len(skipped)} 个因平台 marker 不适用（本平台不该装）："
          f"{', '.join(skipped) if skipped else '无'}")
    print()
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
    print("环境来源与锁文件一致：所有包都来自当前解释器，源码没有未锁的 import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
