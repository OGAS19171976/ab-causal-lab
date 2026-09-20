"""CI 与本地"检查清单"之间的契约。

为什么值得单独测
----------------
"CI 跑哪些、本地跑哪些"如果各维护一份清单，迟早漂移：某天本地加了个脚本、
CI 没加，于是**绿灯的 CI 其实没检查那一步**。所以全部检查的定义只有一处
（``scripts/run_all_checks.py``），CI 与 ``tasks.ps1`` 都调它。

这个文件把那条约定的三个具体后果钉住：

1. **顺序依赖**：``warehouse`` 必须排在 ``m5`` / ``m6`` 之前。
   这两段的数仓内容需要 ``build/warehouse.duckdb`` 存在，否则会**静默跳过**
   ——报告里只剩一句"跳过"，而汇总仍然全绿。这种"因为缺前置条件而少测一段"
   是本项目最该防的行为，所以用测试钉死顺序。
2. **``--quick`` 只发给真正接受它的脚本**。第一版把"接受 --quick"和
   "快速模式下是否运行"合成一个开关，结果 ``warehouse`` 被跳过，
   恰好触发了上面那条静默跳过。
3. **CI 确实调用同一份定义**，而不是在 YAML 里另写一串命令。
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _load_runner():
    """按路径加载 scripts/run_all_checks.py（它不是包的一部分，不能 import）。"""
    path = ROOT / "scripts" / "run_all_checks.py"
    spec = importlib.util.spec_from_file_location("run_all_checks_under_test", path)
    assert spec and spec.loader, f"加载不了 {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _accepts_quick(script: Path) -> bool:
    return '"--quick"' in script.read_text(encoding="utf-8")


def _script_steps(runner) -> list:
    """只挑"直接跑某个 .py 脚本"的步骤。

    新增的 lint 步骤走的是 ``python -m ruff``，``argv[1]`` 是 ``-m`` 而不是脚本路径 ——
    按原样去 ``ROOT / "-m"`` 会直接 FileNotFoundError。所以这里统一过滤一次，
    两条依赖"argv[1] 是脚本"的测试都用它。
    """
    return [s for s in runner.steps() if str(s.argv[1]).endswith(".py")]


class TestCheckPlan:
    def test_scripts_exist(self, runner):
        missing = [s.key for s in _script_steps(runner) if not (ROOT / s.argv[1]).exists()]
        assert not missing, f"检查计划里引用了不存在的脚本：{missing}"

    def test_warehouse_precedes_its_consumers(self, runner):
        """数仓必须排在 m5/m6 之前 —— 否则那两段的数仓内容会被静默跳过。"""
        keys = [s.key for s in runner.steps()]
        assert "warehouse" in keys
        idx = keys.index("warehouse")
        for consumer in ("m5", "m6"):
            assert consumer in keys
            assert idx < keys.index(consumer), (
                f"{consumer} 排在 warehouse 之前：它的数仓段落会因为 "
                "build/warehouse.duckdb 不存在而静默跳过（汇总却仍然全绿）"
            )

    def test_lock_runs_first(self, runner):
        assert [s.key for s in runner.steps()][0] == "lock"

    def test_fast_checks_run_first(self, runner):
        """秒级的检查（锁文件、lint、类型检查、"没做"清单）与前置条件排在前面。

        早失败就早反馈 —— 不用等五分钟的 pytest 跑完才发现少了个导入。
        **声明核对（claims）不在这里**：它要读的 reports/ 是后面那些步骤写的，
        所以它必须收尾（见下一条测试）。
        """
        keys = [s.key for s in runner.steps()]
        assert keys[:7] == [
            "lock", "lint", "types", "unimplemented", "warehouse", "gov", "cate"
        ], keys[:7]

    def test_claims_runs_last(self, runner):
        """**声明核对必须排在最后**：它检查的"README 数字能否在报告里找到"
        依赖前面所有会重写 reports/ 的步骤。

        第一版把它放在最前面，于是"改了报告内容 + 同时加声明"时
        **第一次跑必红、第二次才绿** —— 那是最难查的一类假红灯：
        看起来像声明写错了，其实只是顺序。
        """
        keys = [s.key for s in runner.steps()]
        assert keys[-1] == "claims", keys[-3:]

    def test_unimplemented_step_exists(self, runner):
        """**"没做"的清单也必须在检查集里** —— 与声明核对同一个理由：
        没人跑的核对等于没有核对。这个仓库栽过三次"功能做完了 README 还写着
        没做"（簇级 CUPED、M2 决策层、数仓比值链路），每次都靠人偶然发现。"""
        step = next(s for s in runner.steps() if s.key == "unimplemented")
        assert any("check_unimplemented.py" in a for a in step.argv), step.argv

    def test_cate_interval_step_exists(self, runner):
        """CATE 区间的验证必须在检查集里 —— 它的结论是"校准不了"，
        而**结论是负面的时候更需要有人在跑它**：否则下一个人会以为只是没做。"""
        cate = next(s for s in runner.steps() if s.key == "cate")
        assert any("run_cate_interval_validation.py" in a for a in cate.argv), cate.argv

    def test_readme_claims_step_exists(self, runner):
        """README 的数字要有东西核对 —— 否则"每个数字都能在 reports/ 里找到"
        只是一句没人执行的声明（这个仓库已经在两处吃过同样的亏）。"""
        claims = next(s for s in runner.steps() if s.key == "claims")
        assert any("check_readme_claims.py" in a for a in claims.argv), claims.argv

    def test_governance_step_exists(self, runner):
        """治理验证（审计 + 护栏）必须在检查集里，而不是"我本地跑过一次"。"""
        gov = next(s for s in runner.steps() if s.key == "gov")
        assert any("run_governance_validation.py" in a for a in gov.argv), gov.argv

    def test_type_check_step_uses_module_invocation(self, runner):
        """``python -m mypy``，理由同上条（跨平台，且不拼平台相关的可执行文件名）。"""
        types_step = next(s for s in runner.steps() if s.key == "types")
        assert types_step.argv[1] == "-m", types_step.argv
        assert types_step.argv[2] == "mypy", types_step.argv

    def test_lint_step_uses_module_invocation(self, runner):
        """``python -m ruff`` 而不是直接调可执行文件。

        前者跨平台（Windows 上叫 ruff.exe、Linux 上叫 ruff），
        后者要在代码里拼平台相关的名字，容易写错。
        """
        lint = next(s for s in runner.steps() if s.key == "lint")
        assert lint.argv[1:3] == ("-m", "ruff"), lint.argv
        assert "src" in lint.argv and "tests" in lint.argv

    def test_ruff_config_is_deliberate(self):
        """ruff 的配置必须是**显式**的。

        未配置时 ruff 0.16 的默认规则集比 E4/E7/E9/F 宽得多（实测在这份代码上
        扫出 196 条），其中大量是风格性改写（UP / SIM / RUF / FURB）。
        配上显式的 select 才能保证"lint 绿"意味着"没有真问题"，
        而不是"碰巧没触发风格规则"—— 也避免 lint 结果随 ruff 版本漂移。
        """
        import tomllib

        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        ruff = cfg.get("tool", {}).get("ruff")
        assert ruff, "pyproject.toml 里没有 [tool.ruff]"
        selected = ruff["lint"]["select"]
        for must in ("F", "I"):
            assert any(s.startswith(must) for s in selected), f"select 里缺 {must}"

    def test_ruff_is_declared_and_locked(self):
        """ruff 必须在依赖里声明并出现在锁文件中（否则 CI 上 lint 会直接失败）。"""
        assert "ruff" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "ruff==" in (ROOT / "requirements.lock").read_text(encoding="utf-8")

    def test_quick_flag_matches_what_scripts_accept(self, runner):
        """``accepts_quick`` 必须与脚本自己的 argparse 一致，两个方向都要查。"""
        for step in _script_steps(runner):
            script = ROOT / step.argv[1]
            actual = _accepts_quick(script)
            assert actual == step.accepts_quick, (
                f"{step.key}（{script.name}）声明 accepts_quick={step.accepts_quick}，"
                f"但脚本里{'有' if actual else '没有'} --quick："
                "两边不一致时，快速模式要么报 argparse 错，要么该跑的被跳过"
            )

    def test_quick_argv_is_derived_not_duplicated(self, runner):
        """快速模式只在原 argv 后追加 ``--quick``，不另写一份。

        手写第二份就会出"完整版也带着 --quick"这种错（第一版 m0–m3 就是）。
        """
        for step in runner.steps():
            quick = step.argv_for(quick=True)
            full = step.argv_for(quick=False)
            if quick is None:
                continue
            if step.accepts_quick:
                assert quick == full + ("--quick",), f"{step.key} 的快速 argv 不是派生出来的"
            else:
                assert quick == full, f"{step.key} 不该拿到 --quick"

    def test_nothing_is_silently_skipped_in_quick_mode(self, runner):
        """快速模式不允许有 ""None""（跳过）的项 —— 宁可跑完整版也别装作跑过。"""
        skipped = [s.key for s in runner.steps() if s.argv_for(quick=True) is None]
        assert not skipped, f"快速模式下会被跳过的步骤：{skipped}"


class TestCIContract:
    def test_ci_exists(self):
        assert CI.exists(), "还没有 CI 工作流"

    def test_ci_uses_the_shared_definition(self):
        """CI 必须调用 run_all_checks.py，而不是在 YAML 里另写一串命令。"""
        text = CI.read_text(encoding="utf-8")
        assert "scripts/run_all_checks.py" in text, (
            "CI 没有调用 scripts/run_all_checks.py —— 那意味着它自己维护了一份命令清单，"
            "迟早与本地漂移（出现'本地绿、CI 没跑那一步'的假绿灯）"
        )

    def test_ci_python_version_matches_local(self):
        """CI 的 Python 版本要与本仓库实测的版本一致，而且**只有一个来源**。

        版本写在 ``.python-version`` 里（uv / pyenv / setup-python 都认它），
        YAML 用 ``python-version-file`` 去读 —— 两处各写一份就会出现
        "CI 在 3.13 上绿、本机是 3.14"这种漂移，而绿灯的含义就此含糊。
        """
        declared = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
        local = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert declared == local, f".python-version 写 {declared}，本机是 {local}"

        text = CI.read_text(encoding="utf-8")
        assert "python-version-file: .python-version" in text, "CI 没有从 .python-version 读版本"
        # 常见的坑：既写了 python-version-file 又留了一个硬编码的 python-version
        assert not re.search(r"^\s*python-version:\s", text, re.M), (
            "CI 里同时存在硬编码的 python-version —— 两个来源会打架，"
            "而 setup-python 取哪个是它的实现细节，不该由我们来赌"
        )

    def test_ci_publishes_the_evidence(self):
        """报告与图是这个项目的**交付物**，CI 不该只报"绿了"，该把东西交出来。"""
        text = CI.read_text(encoding="utf-8")
        assert "upload-artifact" in text
        assert "reports/" in text, "CI 没有把 reports/ 作为产物上传"

    def test_ci_runs_lock_check(self):
        assert "lock_requirements.py" in CI.read_text(encoding="utf-8")

    def test_tasks_script_uses_the_same_entry_points(self):
        tasks = (ROOT / "tasks.ps1").read_text(encoding="utf-8")
        assert "run_all_checks.py" in tasks
        assert "lock_requirements.py" in tasks
        # 不要再让 -q 出现两次（会吞掉汇总行）
        assert "-m pytest tests/" in tasks
        assert "-m pytest tests/ -q" not in tasks


class TestLineEndings:
    """工作区里不该有 CRLF 的文本文件。

    ``.gitattributes`` 声明了 ``* text=auto eol=lf``，所以**仓库里存的一定是 LF**；
    但工作区是另一回事：Windows 上的 ``Path.write_text()``（文本模式 newline=None）
    与 PowerShell 的 ``[System.IO.File]::WriteAllLines`` 都会写出 CRLF，而
    ``core.autocrlf=false`` 不会替你转回来。

    这条测试是**被实测打脸之后**补的：做完"统一换行符"那一轮，复查发现工作区里
    还有 7 个 .py 是 CRLF（其中 ``platform/datasource.py``、``platform/analysis.py``
    都是核心模块）—— 也就是说 README 里那句"统一换行符"当时**是不准确的**。
    ``git ls-files --eol`` 显示 ``i/lf w/crlf``，索引是干净的，所以只查 git 看不出来；
    要查的是工作区字节。
    """

    #: `.gitattributes` 里声明为 LF 的后缀。``*.ps1`` 刻意不在其中 —— 它按平台走 CRLF。
    LF_SUFFIXES = (".py", ".md", ".sql", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini")
    #: 没有后缀但同样是 LF 的文件
    LF_NAMES = (".gitattributes", ".python-version", "requirements.lock", "Makefile")
    SKIP_DIRS = {
        ".git",
        ".venv",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
    }

    def test_gitattributes_declares_lf(self):
        text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        assert "eol=lf" in text, ".gitattributes 没有声明 eol=lf"

    def test_no_crlf_in_working_tree(self):
        offenders: list[str] = []
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            if set(path.relative_to(ROOT).parts[:-1]) & self.SKIP_DIRS:
                continue
            if not (path.suffix in self.LF_SUFFIXES or path.name in self.LF_NAMES):
                continue
            data = path.read_bytes()
            if b"\r\n" in data:
                offenders.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        assert not offenders, (
            "这些文件在工作区里是 CRLF（仓库里存的是 LF，但工作区没跟上）：\n  - "
            + "\n  - ".join(offenders)
            + "\n修法（在仓库根执行）：把文件里的 \\r\\n 换成 \\n，然后 git status 应当没有额外变化"
        )


class TestTypeCheckContract:
    """类型检查的接入契约（第 (4) 项：ruff + **类型检查**）。

    为什么要有测试盯着"配置"而不只是"跑得通"：``mypy`` 的**档位**决定了
    "通过"这句话有多少信息量。第一版配置里写了 ``python_version = "3.10"``
    （照的是 ``requires-python`` 下界），结果 mypy 用 3.10 的语法去读 numpy 的
    stub，直接报 ``Type statement is only supported in Python 3.12 and greater
    [syntax]`` **并且中断后续所有检查** —— 于是"mypy 通过"实际是"什么都没查"。
    那是个**假绿灯**，比红色危险得多。
    """

    def test_mypy_is_configured(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        mypy = cfg["tool"]["mypy"]
        # 只查函数体（签名允许不写注解）：这是刻意的档位，理由写在配置里
        assert mypy["check_untyped_defs"] is True
        assert mypy["files"] == ["src/ablab", "scripts"]
        assert mypy["warn_unused_ignores"] is True, "无用的 type: ignore 必须报出来"

    def test_mypy_config_does_not_hardcode_python_version(self):
        """Python 版本只有一个来源（``.python-version``）。

        写进 mypy 配置就是第二个来源，而且实测会以"中断检查"的方式造成假绿灯。
        不写时 mypy 用**正在运行的解释器**，那个版本正是 ``.python-version`` 决定的。
        """
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert "python_version" not in cfg["tool"]["mypy"], (
            "mypy 配置里硬编码了 python_version —— 它必须跟着 .python-version 走"
        )

    def test_mypy_is_a_check_step(self):
        """类型检查必须是检查集里的一步，而不是"我本地跑过一次"。"""
        plan = _load_runner().steps()
        keys = {s.key: s for s in plan}
        assert "types" in keys, "run_all_checks.py 里没有类型检查这一步"
        argv = keys["types"].argv
        assert "mypy" in argv, f"types 步骤跑的不是 mypy：{argv}"
        assert "src" not in argv and "scripts" not in argv, (
            "命令行不该再写一遍路径 —— 检查范围由 pyproject 的 [tool.mypy] files 决定，"
            "否则就会出现两份范围、迟早漂移"
        )

    def test_mypy_is_declared_and_locked(self):
        declared = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        locked = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        assert "mypy==" in declared
        assert "mypy==" in locked
        assert "mypy_extensions==" in locked, "锁文件缺 mypy 的传递依赖"


    """检查集在**管道**下也必须能跑 —— 这条是实测崩过一次之后补的。

    现象：``python scripts/run_all_checks.py --only m2 | Select-Object -Last 20``
    直接 ``UnicodeEncodeError: 'gbk' codec can't encode character '\\u25b6'`` ——
    一步都没跑，崩在打印进度标记上。以前没暴露是因为 ``tasks.ps1`` 和 CI 都替它设了
    ``PYTHONIOENCODING=utf-8``，而 ``check_report_determinism.py`` 恰恰绕过了那两个入口。

    这类 bug 的危险在于**它看起来像"检查失败"**：报错发生在子进程里，汇总只会说
    "预热运行失败，先修好再谈可复现性" —— 读到的人会去查被检查的东西，
    而真正坏的是检查器自己。
    """

    def test_runner_survives_a_pipe_without_encoding_env(self):
        """按路径起一次真进程：stdout 是管道，环境里**没有**任何 PYTHON* 变量。

        端到端复现"有人直接调用"那条路径，而不是断言实现细节。
        """
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        proc = subprocess.run(
            [sys.executable, "scripts/run_all_checks.py", "--list"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.returncode == 0, f"管道下退出了 {proc.returncode}：{proc.stderr[-800:]}"
        assert "UnicodeEncodeError" not in proc.stderr
        # 中文必须**完整**地穿过管道。这一条同时证明了两件事：
        # 子进程没在编码上崩，而且它写的是 UTF-8（若写了 GBK，父进程按 UTF-8 解
        # 会得到乱码，`errors="replace"` 不会报错，但这里就找不到了）。
        assert "计划（完整）" in proc.stdout
        assert "锁文件与当前环境一致" in proc.stdout

    def test_children_are_told_to_write_utf8(self):
        """子进程的 stdout 是按 UTF-8 抓的，所以它们也必须按 UTF-8 写。

        不设那个环境变量时，Windows 上子进程按 GBK 写中文、父进程按 UTF-8 读，
        而 ``errors="replace"`` 让这件事**不报错** —— 只是 ``build/checks/*.log``
        里的中文静默变成乱码。又是一个"没有信号的降级"。
        """
        text = (ROOT / "scripts" / "run_all_checks.py").read_text(encoding="utf-8")
        assert "PYTHONIOENCODING" in text
        assert re.search(r"subprocess\.run\(\s*[^)]*env=child_env\(\)", text, re.S), (
            "run_all_checks.py 起子进程时没有传 child_env()，日志编码会依赖调用者环境"
        )


class TestUnimplementedRegistry:
    """「没做」的清单：每一句都要有**仍然成立**的机器可核对证据。

    这一组测试要钉的不是"清单里有 8 条"，而是**检查器真的会抓过时声明**：
    否则它只是一段会打印 OK 的代码，而那段代码挡不住这个仓库栽过三次的跟头
    （簇级 CUPED、M2 决策层、数仓比值链路 —— 都是功能做完了、README 没改）。
    """

    def test_all_registered_items_still_hold(self):
        """清单本体：每条的证据都仍然成立（否则 README 该改了）。"""
        import importlib

        from ablab.validation.unimplemented import ITEMS

        sys.path.insert(0, str(ROOT / "scripts"))
        checker = importlib.import_module("check_unimplemented")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        stale = []
        for item in ITEMS:
            ok, detail = checker.check_item(item, readme)
            if not ok:
                stale.append(f"{item.id}: {detail}")
        assert not stale, stale

    def test_checker_detects_a_stale_claim(self):
        """**核心断言**：证据不成立时必须报错，并指出该改哪一条。"""
        import importlib

        from ablab.validation.unimplemented import UnimplementedItem

        sys.path.insert(0, str(ROOT / "scripts"))
        checker = importlib.import_module("check_unimplemented")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        # 一条"没做：没有 CausalForest"——而它显然存在
        stale = UnimplementedItem(
            id="fake",
            readme_phrase="CausalForest",  # README 里确实有这个词
            kind="symbol_absent",
            target="ablab.causal.forest.CausalForest",
            anchor_present="ablab.causal.forest",
            when_done="把这条从清单里删掉",
        )
        ok, detail = checker.check_item(stale, readme)
        assert not ok
        assert "已经存在" in detail and "把这条从清单里删掉" in detail

    def test_checker_detects_a_phrase_that_left_the_readme(self):
        """README 里那句被删/改词了 -> 也要报，否则清单会与文档脱节。"""
        import importlib

        from ablab.validation.unimplemented import UnimplementedItem

        sys.path.insert(0, str(ROOT / "scripts"))
        checker = importlib.import_module("check_unimplemented")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        item = UnimplementedItem(
            id="fake2",
            readme_phrase="这句话在 README 里绝对不存在（测试用）",
            kind="symbol_absent",
            target="ablab.causal.iv",
        )
        ok, detail = checker.check_item(item, readme)
        assert not ok and "找不到这句" in detail

    def test_registry_does_not_cover_uncheckable_claims_silently(self):
        """无法机检的"没做"要**显式列出来**，不能混进清单充数。"""
        from ablab.validation.unimplemented import ITEMS, human_reviewed_notes

        notes = human_reviewed_notes()
        assert notes, "至少要把无法机检的几条列出来"
        # 清单里的每一条都必须是可机检的三种证据之一
        assert {i.kind for i in ITEMS} <= {"symbol_absent", "text_absent", "file_absent"}
        # 而且不能把人工那条伪装成机检项
        assert not any("真实流量" in i.readme_phrase and i.kind == "symbol_absent" for i in ITEMS)
