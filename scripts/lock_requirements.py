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
DIRECT: dict[str, tuple[str, ...]] = {
    "运行时": ("numpy", "scipy", "pandas", "duckdb", "pyarrow", "matplotlib"),
    "M4 机器学习": ("scikit-learn",),
    "M5/M6 接口": ("fastapi", "uvicorn", "pydantic"),
    "测试 / 工具": ("pytest", "httpx", "packaging", "ruff"),
}


def _marker_applies(marker: str | None) -> bool:
    """判断依赖的 environment marker 在当前环境是否成立。

    拿不到 ``packaging`` 时保守地返回 True（宁可多写一行，不要漏一个依赖）。
    """
    if not marker:
        return True
    try:
        from packaging.markers import Marker
    except Exception:
        return True
    try:
        return bool(Marker(marker).evaluate())
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


def closure(direct: tuple[str, ...]) -> dict[str, str]:
    """从直接依赖出发，递归收集传递闭包 {分发名: 版本}。"""
    seen: dict[str, str] = {}
    queue = list(direct)
    while queue:
        name = queue.pop()
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            print(f"  [警告] 依赖 {name!r} 在当前环境里找不到，锁文件会缺它", file=sys.stderr)
            continue
        key = dist.metadata["Name"]
        if key in seen:
            continue
        seen[key] = dist.version
        for req in dist.requires or ():
            _, _, marker = req.partition(";")
            # 跳过 optional extra（它们不是本项目的直接需要）
            if 'extra ==' in marker or 'extra==' in marker:
                continue
            if not _marker_applies(marker.strip() or None):
                continue
            pkg = _requirement_name(req)
            if pkg and pkg.lower() != "python":
                queue.append(pkg)
    return seen


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
        bad: list[str] = []
        for name, version in sorted(locked.items()):
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
            return 1
        print(f"当前环境与 requirements.lock 一致（{len(locked)} 个包，Python {py}）")
        return 0

    lines = [
        "# 由 scripts/lock_requirements.py 生成 —— 不要手改。",
        f"# 生成环境：CPython {py} / {sys.platform}",
        "# 内容 = 直接依赖 + 传递闭包，全部钉死版本。",
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
    for name, version in sorted(locked.items(), key=lambda kv: kv[0].lower()):
        lines.append(f"{name}=={version}")

    path = Path(args.out)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {path}（{len(locked)} 个包，Python {py}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
