"""依赖声明的一致性测试。

为什么值得单独写一个测试
------------------------
M5/M6 的依赖（scikit-learn / fastapi / uvicorn / httpx / pydantic）
曾经**全都没写进 `requirements.txt`**，而 README 的"快速开始"又让人照那个文件装。
本机能跑，是因为 venv 用了 `--system-site-packages` 恰好蹭到了全局装好的包 ——
换句话说，这个缺口在本机永远不会暴露，只在别人 clone 下来时炸。

这类"本机永远看不见"的问题正好适合用静态检查钉住：不 import 任何东西，
只用 ``ast`` 扫源码里的顶层 import，逐个对着 `pyproject.toml` 的声明查。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

#: import 名 → 分发名。两者不一致的都得在这里登记，
#: 否则测试会把"装了却查不到"误判成缺依赖。
_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
    "pkg_resources": "setuptools",
}

#: 本项目自己的包（不是第三方依赖）+ 仅测试期用的本地工具
_OWN = {"ablab"}


def _pyproject() -> dict:
    return tomllib.load(open(ROOT / "pyproject.toml", "rb"))


def _dist_name(spec: str) -> str:
    return re.split(r"[<>=!~\[]", spec)[0].strip().lower()


def _declared() -> set[str]:
    cfg = _pyproject()
    names = {_dist_name(d) for d in cfg["project"]["dependencies"]}
    for items in cfg["project"]["optional-dependencies"].values():
        names |= {_dist_name(d) for d in items}
    return names


def _third_party_imports() -> dict[str, set[str]]:
    """扫 src/ 与 scripts/，返回 {分发名: {出现位置}}。"""
    found: dict[str, set[str]] = {}
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    assert files, "没扫到任何源码文件，测试本身失效了"

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module] if node.module and node.level == 0 else []
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                if not top or top in _OWN or top in sys.stdlib_module_names:
                    continue
                dist = _IMPORT_TO_DIST.get(top, top).lower()
                found.setdefault(dist, set()).add(f"{path.relative_to(ROOT)}")
    return found


class TestDependencyDeclarations:
    def test_every_third_party_import_is_declared(self):
        """源码里 import 的每个第三方包，都必须在 pyproject 里声明。

        这正是当初漏掉 sklearn / fastapi 的那道口子：代码能跑（本机装了），
        但声明里没有，于是"照 README 装一遍"得到的是一个跑不起来的环。
        """
        declared = _declared()
        used = _third_party_imports()
        missing = {name: sorted(where) for name, where in used.items() if name not in declared}
        assert not missing, (
            "以下第三方包被源码 import 了，却没在 pyproject.toml 里声明：\n"
            + "\n".join(f"  - {k}  （出现在 {', '.join(v[:3])}）" for k, v in missing.items())
        )

    def test_requirements_are_covered_by_pyproject(self):
        """requirements.txt 的每一项都要能在 pyproject 里找到对应声明。"""
        declared = _declared()
        required = set()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                required.add(_dist_name(line))
        missing = sorted(required - declared)
        assert not missing, f"requirements.txt 里有而 pyproject 没声明的：{missing}"

    def test_requirements_are_pinned(self):
        """直接依赖必须钉死版本 —— 否则"这些数字可复现"这句话不成立。

        numpy / pandas / scipy 都是很新的大版本，放宽到 ``>=`` 就等于
        把小数点后几位交给运气。
        """
        unpinned = [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#") and "==" not in line
        ]
        assert not unpinned, f"requirements.txt 里没有钉死版本的项：{unpinned}"

    def test_lock_covers_everything_required(self):
        """锁文件必须至少覆盖 requirements.txt 的直接依赖。"""
        lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        locked = {
            _dist_name(line)
            for line in lock.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        required = {
            _dist_name(line)
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        missing = sorted(required - locked)
        assert not missing, f"requirements.lock 没覆盖：{missing}"
        # 传递依赖也得在（当初第一版闭包就漏了 starlette / pydantic_core / h11）
        for transitive in ("starlette", "pydantic_core", "h11", "joblib"):
            assert transitive in locked, f"锁文件缺传递依赖 {transitive}"

    def test_lock_check_passes(self):
        """锁文件与当前环境必须逐项一致（这把"可复现"变成可执行的检查）。"""
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "lock_requirements.py"), "--check"],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "一致" in proc.stdout

    def test_python_version_is_documented(self):
        """README 里要写清实测过的 Python 版本。"""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "3.14" in readme, "README 没写实测的 Python 版本"
        assert "requirements.lock" in readme, "README 没提锁文件"

    def test_scan_actually_finds_third_party(self):
        """反向检查：如果扫描逻辑坏了（比如全被当成标准库），上面几条会假通过。"""
        used = _third_party_imports()
        # 注意用**分发名**：扫描会把 import 名映射过去（sklearn -> scikit-learn）
        for expected in ("numpy", "scipy", "scikit-learn", "fastapi", "duckdb"):
            assert expected in used, f"扫描没发现 {expected}，说明扫描逻辑失真"
