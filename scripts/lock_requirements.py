#!/usr/bin/env python
"""生成 ``requirements.lock``：直接依赖 + 它们的**传递闭包**，全部钉死版本。

为什么需要这个脚本而不是一句 ``uv pip freeze``
----------------------------------------------
1. 本项目的 venv 是 ``uv venv --system-site-packages`` 建的，所以
   "当前环境里装着什么" 包含了使用者全局装的所有东西（jupyter、torch……）。
   直接 freeze 会把它们全写进锁文件，于是锁文件既不真实也不最小。
   这里只从**声明的直接依赖**出发做闭包。
2. 不依赖任何外部工具（``packaging`` 可选），因为它要能在受限环境里跑 ——
   ``uv`` 自己的缓存目录默认在 ``~/.cache/uv``，沙箱里会被拒。
   真要跑 uv，记得 ``UV_CACHE_DIR`` 指到项目内。

用法::

    python scripts/lock_requirements.py            # 写 requirements.lock
    python scripts/lock_requirements.py --check     # 只校验当前环境是否满足锁定
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 直接依赖 —— 与 pyproject.toml 的 dependencies / optional-dependencies 对应。
#: 分组只是给人看的；``requirements.lock`` 里是扁平的传递闭包。
#:
#: **这里是第三份"直接依赖"清单**（另两份是 requirements.txt 与 pyproject）。
#: 实测被它坑过一次：给 requirements.txt 加了 mypy 之后重新生成锁文件，
#: 锁里**没有 mypy** —— 因为生成器不知道它。而 ``--check`` 只比对
#: "锁文件 vs 当前环境"，对"两边都没有 mypy"这件事是瞎的。
#: 抓住它的是 ``tests/test_dependencies.py`` 里的
#: ``test_lock_covers_everything_required``（锁必须覆盖 requirements.txt 的每一项）；
#: 这里再加一条反方向的（DIRECT 里的每一项都要在 requirements.txt 里），
#: 这样三份清单就只能**同时**正确，不能各自漂移。
DIRECT: dict[str, tuple[str, ...]] = {
    "运行时": ("numpy", "scipy", "pandas", "duckdb", "pyarrow", "matplotlib"),
    "M4 机器学习": ("scikit-learn",),
    "M5/M6 接口": ("fastapi", "uvicorn", "pydantic"),
    "测试 / 工具": ("pytest", "httpx", "packaging", "ruff", "mypy"),
}


def _marker_applies(marker: str | None, environment: dict | None = None) -> bool:
    """判断依赖的 environment marker 在当前环境是否成立。

    拿不到 ``packaging`` 时保守地返回 True（宁可多写一行，不要漏一个依赖）。

    ``environment`` 传 None 表示用真实环境；传一个字典可以**模拟别的平台** ——
    这让"在 Linux 上 --check 会不会因为 Windows 专属包而失败"这件事
    能在 Windows 上被测试覆盖（见 tests/test_dependencies.py）。
    """
    if not marker:
        return True
    try:
        from packaging.markers import Marker
    except Exception:
        return True
    try:
        return bool(Marker(marker).evaluate(environment))
    except Exception:
        return True


def _requirement_name(req: str) -> str | None:
    """从一条 Requires-Dist 里取出**包名**（剥掉版本、extra、marker）。

    ``Requires-Dist`` 的实际形态很杂：
        ``httpcore==1.*`` ｜ ``typing_extensions>=4.16.0``
        ``numpy (>=1.24) ; extra == 'dev'`` ｜ ``pillow>=9``
    第一版直接按空格切，于是 ``httpcore==1.*`` 整串被当成包名、
    ``md.distribution()`` 找不到 —— 闭包因此漏了 starlette / pydantic-core / h11 等一大批。
    正经做法是交给 ``packaging`` 解析；它不可用时退回正则。
    """
    try:
        from packaging.requirements import Requirement

        return Requirement(req).name
    except Exception:
        pass
    import re

    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return m.group(1) if m else None


def closure(direct: tuple[str, ...]) -> dict[str, tuple[str, str | None]]:
    """从直接依赖出发，递归收集传递闭包 ``{分发名: (版本, marker)}``。

    **为什么要带着 marker 走**：闭包是在**当前平台**上算的，而依赖里有一类是
    平台条件依赖 —— 实测这份锁里有 ``colorama``（pytest / click 只在 Windows 上要）
    与 ``tzdata``（pandas 只在 Windows / emscripten 上要）。它们在 Ubuntu 上
    **不会被安装**，而 ``--check`` 要求"每个锁定包都已安装" → CI 第一次跑就红。

    所以把"这个包是在什么条件下需要的"记进锁文件（标准 requirements 语法
    ``name==version ; marker``）：
    * 别的平台 ``--check`` 时能正确跳过（pip 装锁文件时也会跳过）；
    * 这份锁仍然只描述**生成平台上真实存在的那套环境**，不做跨平台推断。

    marker 用**到达路径的合取**累积：A 依 B（m1）、B 依 C（m2）→ C 的条件是
    ``m1 and m2``。同一个包被多条路径到达时：任一路径无条件 → 无条件；
    否则取析取 ``(m1) or (m2)``（这是合法的 PEP 508 marker，pip 认）。
    """
    seen: dict[str, tuple[str, str | None]] = {}
    queue: list[tuple[str, str | None]] = [(name, None) for name in direct]
    while queue:
        name, accumulated = queue.pop()
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            print(f"  [警告] 依赖 {name!r} 在当前环境里找不到，锁文件会缺它", file=sys.stderr)
            continue
        key = dist.metadata["Name"]
        if key in seen:
            seen[key] = (seen[key][0], _or_markers(seen[key][1], accumulated))
            continue
        seen[key] = (dist.version, accumulated)
        for req in dist.requires or ():
            _, _, marker = req.partition(";")
            # 跳过 optional extra（它们不是本项目的直接需要）
            if 'extra ==' in marker or 'extra==' in marker:
                continue
            own = marker.strip() or None
            if not _marker_applies(own):
                continue
            pkg = _requirement_name(req)
            if pkg and pkg.lower() != "python":
                queue.append((pkg, _and_markers(accumulated, own)))
    return seen


def _and_markers(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return f"({a}) and ({b})"


def _or_markers(a: str | None, b: str | None) -> str | None:
    if a is None or b is None:
        return None  # 有一条路径无条件需要 → 这个包就是无条件需要
    if a == b:
        return a
    return f"({a}) or ({b})"


def select_for_platform(
    locked: dict[str, tuple[str, str | None]], environment: dict | None = None
) -> tuple[dict[str, tuple[str, str | None]], list[str]]:
    """把锁文件里的包分成「本平台适用」与「本平台不适用」。

    这是 ``--check`` 与"换了平台会不会红"这两件事**共用的唯一判定**。
    抽成纯函数是为了能测：传一个模拟的 Linux 环境字典，就能在 Windows 上
    验证"colorama / tzdata 会被正确跳过" —— 而不是等到 CI 上才发现。
    """
    applicable: dict[str, tuple[str, str | None]] = {}
    skipped: list[str] = []
    for name, (version, marker) in locked.items():
        if _marker_applies(marker, environment):
            applicable[name] = (version, marker)
        else:
            skipped.append(name)
    return applicable, sorted(skipped)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 / 校验 requirements.lock")
    ap.add_argument("--out", default=str(ROOT / "requirements.lock"))
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    everything = tuple(n for group in DIRECT.values() for n in group)
    locked = closure(everything)
    # 规范名 -> 版本，便于按声明名查（分发名的大小写与连字符未必一致，如 scikit-learn）
    by_lower = {name.lower(): (name, version) for name, version in locked.items()}
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if args.check:
        applicable, skipped = select_for_platform(locked)
        bad: list[str] = []
        for name, (version, _marker) in sorted(applicable.items()):
            try:
                actual = md.version(name)
            except md.PackageNotFoundError:
                bad.append(f"{name} 未安装（锁 {version}）")
                continue
            if actual != version:
                bad.append(f"{name} 装了 {actual}，锁的是 {version}")
        if bad:
            print("当前环境与 requirements.lock 不一致：")
            for line in bad:
                print(f"  - {line}")
            if skipped:
                print(f"  （另有 {len(skipped)} 个包在当前平台不适用，已跳过："
                      f"{', '.join(skipped)}）")
            return 1
        note = (
            f"；{len(skipped)} 个在当前平台不适用已跳过（{', '.join(skipped)}）"
            if skipped
            else ""
        )
        print(f"当前环境与 requirements.lock 一致（{len(applicable)} 个包，"
              f"Python {py}{note}）")
        return 0

    lines = [
        "# 由 scripts/lock_requirements.py 生成 —— 不要手改。",
        f"# 生成环境：CPython {py} / {sys.platform}",
        "# 内容 = 直接依赖 + 传递闭包，全部钉死版本。",
        "# 带 `; marker` 的行是**平台条件依赖**：换个平台它可能根本不该被安装",
        "#   （实测：colorama / tzdata 只在 Windows 上需要），--check 会先判 marker 再查。",
        "# 注意这份闭包是在**生成平台**上算的：换平台只保证「已锁的都在」，",
        "#   不保证「该平台需要、而生成平台上不需要」的包也被锁进来。",
        "# 校验：python scripts/lock_requirements.py --check",
        "",
    ]
    for group, names in DIRECT.items():
        lines.append(f"# ---- {group} ----")
        for name in names:
            hit = by_lower.get(name.lower())
            lines.append(f"#   {name}=={hit[1] if hit else '???'}")
        lines.append("")
    lines.append("# ---- 传递依赖 ----")
    for name, (version, marker) in sorted(locked.items(), key=lambda kv: kv[0].lower()):
        lines.append(f"{name}=={version}" + (f" ; {marker}" if marker else ""))

    path = Path(args.out)
    # newline="\n" 不是可有可无的：默认（newline=None）在 Windows 上会把 \n 翻成 \r\n，
    # 于是**同一个脚本在 Windows 与 Linux 上写出不同的字节** —— 而 requirements.lock
    # 是被提交的文件，这种平台差异会变成别人 diff 里的整文件重写。
    # 实测到过：工作区里 requirements.lock 是 CRLF，而索引里是 LF
    # （`.gitattributes` 声明了 eol=lf，所以 git 侧看不出来）。
    # 报告那几个写出口早就带了 newline="\n"，这里是漏掉的一个。
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"已写入 {path}（{len(locked)} 个包，Python {py}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
