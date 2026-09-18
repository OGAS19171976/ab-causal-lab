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
import importlib.util
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


class TestCrossPlatformLock:
    """锁文件必须在**另一个平台**上也能通过校验 —— 这是 CI 第一次跑就会撞上的事。

    背景（实测）：`requirements.lock` 是在 Windows 上生成的闭包，里面必然带上
    Windows 专属的传递依赖 —— `colorama`（pytest / click 只在 Windows 上要）、
    `tzdata`（pandas 只在 Windows / emscripten 上要）。而 `lock_requirements.py
    --check` 原本要求"每个锁定包都已安装"，于是在 Ubuntu runner 上会报
    "colorama 未安装" → **CI 第一步就红**，而本机永远看不到。

    修法是把"这个包在什么条件下需要"（PEP 508 marker）写进锁文件，校验时先判
    marker。下面这些测试**在 Windows 上模拟 Linux**，所以这条 CI-only 的失败路径
    在本机就能被覆盖 —— 而不是等推上去才发现。
    """

    LINUX = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix"}
    WINDOWS = {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt"}

    @staticmethod
    def _load():
        spec = importlib.util.spec_from_file_location(
            "lock_requirements_under_test2", ROOT / "scripts" / "lock_requirements.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_windows_only_packages_are_marked(self):
        """平台专属依赖必须带 marker，否则别的平台会把它当成"必须已安装"。"""
        text = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        marked = {
            line.split("==")[0].strip(): line.split(";", 1)[1].strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#") and ";" in line
        }
        assert "colorama" in marked, "colorama 是 Windows 专属依赖，却没写 marker"
        assert "win32" in marked["colorama"], marked["colorama"]
        assert "tzdata" in marked, "tzdata（pandas 只在 Windows/emscripten 要）也没写 marker"

    def test_check_skips_them_on_linux(self):
        """模拟 Linux：这两个包必须被跳过，而其余包照旧检查。

        **用注入的 entries，不要从当前环境重算闭包。** 第一版是从 `closure()`
        拿的条目，于是在 CI 的 Linux 上失败：那台机器根本没装 colorama / tzdata
        （它们是 Windows 专属依赖），**压根不会出现在闭包里**，
        自然既谈不上"被跳过"、也谈不上"被保留" —— 测试写成了只在本机成立的形状。
        这也正是"测试必须自己把输入条件写死"的又一例。
        """
        module = self._load()
        entries = {
            "numpy": ("2.5.3", None),
            "librt": ("0.15.0", 'platform_python_implementation != "PyPy"'),
            "colorama": ("0.4.6", 'sys_platform == "win32"'),
            "tzdata": ("2026.2", 'sys_platform == "win32"'),
        }
        applicable, skipped = module.select_for_platform(entries, self.LINUX)

        assert "colorama" in skipped and "tzdata" in skipped, skipped
        assert "colorama" not in applicable
        # librt（mypy 的依赖）条件是非 PyPy —— Linux 上照样需要，不能被误跳
        assert "librt" in applicable, "librt 在 Linux 上也需要，不该被跳过"
        # 无条件依赖一个都不能被跳过
        assert "numpy" in applicable

    def test_check_keeps_them_on_windows(self):
        """模拟 Windows：它们就该被要求存在。"""
        module = self._load()
        entries = {
            "numpy": ("2.5.3", None),
            "colorama": ("0.4.6", 'sys_platform == "win32"'),
            "tzdata": ("2026.2", 'sys_platform == "win32"'),
        }
        applicable, skipped = module.select_for_platform(entries, self.WINDOWS)
        assert "colorama" in applicable and "tzdata" in applicable
        assert skipped == [], f"Windows 上不该跳过任何包：{skipped}"

    def test_real_lock_file_loses_nothing_on_linux(self):
        """真锁文件在 Linux 上：该跳的跳、该留的留 —— 用真文件而不是真环境。

        这条读的是 `requirements.lock` 本身，所以**在任何平台上结论都一样**；
        它保证"锁文件对 Linux 是有意义的"，而不是"本机恰好测出这个结果"。
        """
        module = self._load()
        entries = module.parse_lock(ROOT / "requirements.lock")
        applicable, skipped = module.select_for_platform(entries, self.LINUX)
        assert set(skipped) == {"colorama", "tzdata"}, skipped
        # 直接依赖一个都不能被跳过
        direct = {module.normalize(n) for g in module.DIRECT.values() for n in g}
        assert not (direct & set(skipped)), f"直接依赖被误判为不适用：{direct & set(skipped)}"
        assert "librt" in applicable
        assert len(applicable) == len(entries) - 2

    def test_every_lock_line_is_a_valid_requirement(self):
        """锁文件必须能被标准 PEP 508 解析器读 —— pip 就是那么读的。

        带 marker 的写法（``name==1.0 ; sys_platform == "win32"``）是标准
        requirements 语法，不是我们自创的；这条测试把它钉住，
        否则 ``pip install -r requirements.lock`` 会在别的平台上解析失败。
        """
        from packaging.requirements import Requirement

        text = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = Requirement(line)  # 解析失败会直接抛异常
            assert parsed.specifier, f"锁文件里这一行没有钉死版本：{line}"


class TestLockCheckActuallyChecks:
    """``--check`` 必须**真的读锁文件** —— 它曾经不是。

    实测到的问题：第一版 ``--check`` 是从当前环境重算一遍闭包（``dist.version``
    来自已装的包），再拿 ``md.version()`` 去比自己 —— **恒等**，而且从头到尾没打开过
    ``requirements.lock``。于是这一步**必然绿**，却挂着"依赖声明与锁文件一致"的牌子。

    它是被 CI 当场演示的：第一次在 Ubuntu runner 上跑时，锁文件把 ``colorama``
    （pytest / click 的 Windows 专属依赖）写成硬依赖，那台机器上根本没有它，
    这一步照样 ``OK``。**假绿灯比红灯危险**，因为它不产生任何信号。

    这三条测试用**注入数据**构造"锁写错了"的三种情形，所以它们检查的是
    "这个 check 有没有能力发现问题"，而不是"当前环境恰好是对的"。
    """

    @staticmethod
    def _load():
        spec = importlib.util.spec_from_file_location(
            "lock_requirements_under_test3", ROOT / "scripts" / "lock_requirements.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_parses_the_real_lock_file(self):
        """先把真锁文件读进来：条目数、版本、以及 marker 都要解析出来。"""
        module = self._load()
        entries = module.parse_lock(ROOT / "requirements.lock")
        assert len(entries) >= 40, f"锁文件只解析出 {len(entries)} 条，太小了"
        assert entries["numpy"][0] == "2.5.3"
        assert entries["colorama"][1] is not None, "colorama 的 marker 没被解析出来"
        assert "win32" in entries["colorama"][1]
        # 规范化：锁里写 ast_serialize，键应当是 ast-serialize
        assert "ast-serialize" in entries

    def test_detects_a_wrong_version_in_the_lock(self):
        """锁里版本写错必须报出来 —— 旧实现连这个都做不到（自比恒等）。"""
        module = self._load()
        entries = {"numpy": ("1.0.0", None)}
        bad, skipped = module.check_against(
            entries,
            installed={"numpy": "2.5.3"},
            needed={"numpy": "2.5.3"},
        )
        assert any("numpy" in b and "1.0.0" in b for b in bad), bad
        assert skipped == []

    def test_detects_a_package_missing_from_the_lock(self):
        """环境需要、锁里没有 → 报出来。这正是"加了 mypy 却忘了重新生成锁"那一幕。"""
        module = self._load()
        entries = {"numpy": ("2.5.3", None)}
        bad, _ = module.check_against(
            entries,
            installed={"numpy": "2.5.3", "mypy": "2.3.1"},
            needed={"numpy": "2.5.3", "mypy": "2.3.1"},
        )
        assert any("mypy" in b and "锁文件里没有" in b for b in bad), bad

    def test_detects_a_missing_installation(self):
        module = self._load()
        entries = {"numpy": ("2.5.3", None), "mypy": ("2.3.1", None)}
        bad, _ = module.check_against(
            entries, installed={"numpy": "2.5.3"}, needed={"numpy": "2.5.3"}
        )
        assert any("mypy 未安装" in b for b in bad), bad

    def test_clean_case_has_nothing_to_report(self):
        module = self._load()
        entries = {"numpy": ("2.5.3", None), "colorama": ("0.4.6", 'sys_platform == "win32"')}
        bad, skipped = module.check_against(
            entries,
            installed={"numpy": "2.5.3"},
            needed={"numpy": "2.5.3"},
            environment={"sys_platform": "linux", "platform_system": "Linux"},
        )
        assert bad == [], bad
        assert skipped == ["colorama"], skipped


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

    def test_lock_generator_agrees_with_requirements(self):
        """锁生成器的 ``DIRECT`` 不能漏掉 requirements.txt 里的任何一项。

        这是"直接依赖"的**第三份清单**（另两份是 requirements.txt 与 pyproject），
        而它是最容易被忘掉的一份：实测给 requirements.txt 加了 mypy 之后重新生成
        锁文件，锁里**没有 mypy** —— 生成器根本不知道它。``--check`` 抓不到
        （它比对的是"锁 vs 当前环境"，对"两边都没有"是瞎的），
        上一条 ``test_lock_covers_everything_required`` 抓到了；
        这一条补的是反方向：DIRECT 里有、requirements.txt 里没有的也会漂移。
        """
        spec = importlib.util.spec_from_file_location(
            "lock_requirements_under_test", ROOT / "scripts" / "lock_requirements.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        in_generator = {name.lower() for group in module.DIRECT.values() for name in group}
        in_requirements = {
            _dist_name(line).lower()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert in_generator == in_requirements, (
            "锁生成器的 DIRECT 与 requirements.txt 不一致：\n"
            f"  只在生成器里：{sorted(in_generator - in_requirements)}\n"
            f"  只在 requirements.txt 里：{sorted(in_requirements - in_generator)}"
        )

    def test_lock_check_passes(self):
        """锁文件与当前环境必须逐项一致（这把"可复现"变成可执行的检查）。

        **必须显式告诉子进程按 UTF-8 写。** 这一条是实测踩出来的：脚本的 stdout 一旦
        不是控制台（被 pytest 抓走就是管道），Python 就用 locale 编码 —— 简体中文
        机器上是 GBK；而这里按 UTF-8 解，于是 ``UnicodeDecodeError: 'utf-8' codec
        can't decode byte 0xb5``，``proc.stdout`` 直接是 ``None``，报错还长成
        ``TypeError: argument of type 'NoneType'``，完全看不出真正的原因。

        它藏在"每次全量检查都过"后面：``run_all_checks.py`` 会给 pytest 传
        ``PYTHONIOENCODING=utf-8``，子进程一路继承 —— 于是**只有单独跑
        ``pytest tests`` 的人**才会撞上。这就是"测试依赖了调用者的环境"：
        测试必须自己把输入条件写死，而不是碰运气继承。
        """
        import os
        import subprocess

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "lock_requirements.py"), "--check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
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
