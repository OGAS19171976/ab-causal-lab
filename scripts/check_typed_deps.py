#!/usr/bin/env python
"""**三方库类型清单**：把"三方库没有类型保证"从一句无法核对的话，变成一张表。

为什么需要它
------------
README 的已知边界里原先写着「scipy / pandas / sklearn 没有类型保证：取决于上游
是否带 ``py.typed``，只能人看」。这句话的问题是它**既不精确也不可核对**：
"没有保证"到底指哪些包、我们到底用了它们的多少、mypy 是真的检查了还是当成
``Any`` 放过去了 —— 全靠人去读。

这个脚本把三件事量出来：

  1. 锁文件里的每个发行版**有没有** ``py.typed``（以及有没有官方 stub 包）；
  2. 本项目**实际 import 了哪些**包 —— 按"被多少个源码文件用到"排序，
     因为一个没类型的包只有被用到才是风险；
  3. 哪些**没带类型**的包被用到了 —— 那就是要么写本地最小 stub、
     要么接受 ``Any`` 的那一小撮。

判据（任一条不成立就红）
------------------------
  * 锁文件里的每个包都必须能在这个环境里找到（找不到 = 环境不完整）；
  * 清单必须**覆盖 100% 的锁文件条目**（不许挑着report）；
  * 每个"没带 py.typed 但被用到"的包，必须在下面的 ``UNTYPED_DECISIONS``
    里有明确处置（写本地 stub / 接受 Any 并写清代价）—— 于是"没有类型保证"
    这句话变成**逐包的决定**，而不是一句笼统的免责声明。

诚实的边界：类型只覆盖**接口形状**，不覆盖**数值语义**
（``optimize.minimize`` 收不收敛、``sklearn`` 的随机性、``pandas`` 的隐式
类型转换都不在类型系统里）。所以 mypy 全绿不等于数值正确 —— 这句话仍然
只能靠人读，它留在了 ``unimplemented.human_reviewed_notes()`` 里。
"""

from __future__ import annotations

import ast
import importlib.metadata as md
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock"
SRC = ROOT / "src"
STUB_SUFFIXES = ("-stubs", "_stubs")

#: 发行版名 → 实际 import 的顶层模块名（与 check_env_origin 保持同一张表）
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

#: **每个"没带类型且被本项目用到"的包都要在这里有一条决定**。
#: 值 = (处置, 理由)。处置只能是 ``stub``（我们写最小 stub 钉住用法）
#: 或 ``accept-any``（接受 Any，并写清代价）。
UNTYPED_DECISIONS: dict[str, tuple[str, str]] = {
    "scipy": (
        "stub",
        "用到的是 stats.norm / optimize.minimize / spatial 的几个函数；"
        "本地最小 stub 只声明这些签名，上游改名会在 mypy 里报",
    ),
    "pandas": (
        "accept-any",
        "DataFrame 的类型在无 stub 时基本退化成 Any；"
        "本仓库对它的用法集中在数仓 IO 与列选择，靠测试与 schema 检查兜底",
    ),
    "scikit-learn": (
        "accept-any",
        "只作为 nuisance 学习器（Ridge / RandomForest）出现，"
        "接口窄且被测试覆盖；上游一旦补上顶层 py.typed，本条会被判过时",
    ),
}


def lock_pins() -> list[tuple[str, str]]:
    """``(发行版名, 版本)``；marker 与注释都跳过。"""
    pins = []
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, rest = line.split("==", 1)
        version = rest.split(";")[0].strip()
        pins.append((name.strip(), version))
    return pins


def module_of(dist_name: str) -> str:
    return DIST_TO_MODULE.get(dist_name.lower(), dist_name.replace("-", "_").lower())


def has_py_typed(dist: md.Distribution) -> bool:
    """发行版里有没有 ``py.typed``。

    判据是**顶层包目录里**有 ``py.typed`` —— mypy 只认这一层。
    这个函数被抓错过两次，两次都是"看起来更省事"的写法：

      * 第一版遍历 ``dist.locate_file("")`` 下的子目录，而 wheel 安装时
        它返回的是 **site-packages 根目录** —— 于是遍历了整个 site-packages，
        找到 numpy 的 py.typed 就认为**每个包**都有类型（48/48 全绿）；
      * 第二版按 RECORD 里"任意一层有 py.typed"判，于是 ``sklearn`` 被算成
        有类型 —— 因为它的某个**子包**里有 py.typed，而顶层没有；
        mypy 对顶层没有标记的包一律当 ``Any``，所以那是假绿。

    现在的写法：先看被 import 的那个顶层包目录里有没有 py.typed；
    不能 import 时（例如只装了 stub 包）退回按 RECORD 比对顶层那一条路径。
    """
    import importlib.util

    spec = importlib.util.find_spec(module_of(dist.metadata["Name"]))
    if spec is not None and spec.origin:
        return (pathlib.Path(spec.origin).parent / "py.typed").exists()
    files = dist.files or []
    top = module_of(dist.metadata["Name"]) + "/py.typed"
    return any(str(f).replace("\\", "/") == top for f in files)


def stub_package(dist_name: str) -> str | None:
    """有没有装官方/社区的 stub 包（``X-stubs`` / ``types-X``）。"""
    for candidate in (f"{dist_name}-stubs", f"types-{dist_name}",
                      f"{dist_name.replace('-', '_')}-stubs"):
        try:
            md.distribution(candidate)
            return candidate
        except md.PackageNotFoundError:
            continue
    return None


def usage_counts() -> dict[str, int]:
    """每个顶层模块被多少个文件 import（本项目的真实用量）。

    统计范围是 ``src`` + ``scripts`` + ``tests``：只看 ``src`` 会把
    "只在脚本与测试里用到"的包（pyarrow、pytest 这类）漏掉，
    于是处置表会被误判成过时 —— 这正是第一版踩到的那个假阳性。
    """
    counts: dict[str, int] = {}
    paths: list[pathlib.Path] = []
    for base in (SRC, ROOT / "scripts", ROOT / "tests"):
        paths.extend(sorted(base.rglob("*.py")))
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        seen: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                seen.add((node.module or "").split(".")[0])
        for module in seen:
            if module:
                counts[module] = counts.get(module, 0) + 1
    return counts


def main() -> int:
    pins = lock_pins()
    usage = usage_counts()
    problems: list[str] = []
    rows: list[tuple[str, str, str, str, int]] = []
    missing: list[str] = []

    for name, version in pins:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            missing.append(name)
            continue
        typed_mark = "有" if has_py_typed(dist) else "**没有**"
        stub = stub_package(name)
        rows.append(
            (name, version, typed_mark, stub or "-", usage.get(module_of(name), 0))
        )

    if missing:
        problems.append(f"锁文件里有 {len(missing)} 个包在这个环境里找不到：{missing}")

    used_untyped = [
        (name, n) for name, _v, typed, _s, n in rows if typed == "**没有**" and n > 0
    ]
    for name, n in used_untyped:
        if name not in UNTYPED_DECISIONS:
            problems.append(
                f"{name}: 没带 py.typed、且被 {n} 个源码文件用到，"
                "但 UNTYPED_DECISIONS 里没有处置（写 stub 还是接受 Any？）"
            )
    # 反向：处置表里的条目如果已经不需要（上游带了类型、或本项目不再用它），
    # 那也是一句**过时声明** —— 与"没做"清单同一个道理，必须红。
    need = {name for name, _n in used_untyped}
    typed_names = {name for name, _v, typed, _s, _n in rows if typed == "有"}
    for name in sorted(set(UNTYPED_DECISIONS) - need):
        why = "上游已经带 py.typed" if name in typed_names else "本项目没有 import 它"
        problems.append(
            f"{name}: UNTYPED_DECISIONS 里的处置已经过时（{why}）—— 删掉这一条"
        )

    typed_rows = [r for r in rows if str(r[2]) == "有"]
    print("三方库类型清单（判据：覆盖 100% 的锁文件条目 + 每个无类型的包都有处置）")
    print(f"  锁文件条目 {len(pins)} 个；带 py.typed 的 {len(typed_rows)} 个；"
          f"本项目实际 import 的 {sum(1 for r in rows if r[4] > 0)} 个")
    print()
    print(f"  {'包':<22}{'版本':<12}{'py.typed':<10}{'stub 包':<16}{'用到它的源码文件':>8}")
    for name, version, typed, stub, n in sorted(rows, key=lambda r: (-r[4], r[0])):
        if n == 0 and typed == "有":
            continue  # 没用到又有类型的：不必占版面
        print(f"  {name:<22}{version:<12}{typed:<10}{stub:<16}{n:>8}")
    print()
    print("  没带类型、但被本项目用到的（逐包处置）：")
    if not used_untyped:
        print("    没有 —— 所有被用到的包都带类型")
    for name, n in sorted(used_untyped, key=lambda x: -x[1]):
        action, why = UNTYPED_DECISIONS.get(name, ("**缺处置**", ""))
        print(f"    {name:<18}{n:>3} 个文件   处置：{action}")
        print(f"      {why}")
    print()
    if problems:
        print(f"**{len(problems)} 条不成立**：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("清单覆盖锁文件全部条目，且每个无类型的包都有明确处置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
