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
import re
import sys
from pathlib import Path

import pytest

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
        """秒级的检查（锁文件、lint）与前置条件（数仓）必须排在最前面。

        早失败就早反馈 —— 不用等五分钟的 pytest 跑完才发现少了个导入。
        """
        keys = [s.key for s in runner.steps()]
        assert keys[:3] == ["lock", "lint", "warehouse"], keys[:3]

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
        """CI 的 Python 版本要与本仓库实测的版本一致（3.14）。"""
        text = CI.read_text(encoding="utf-8")
        versions = set(re.findall(r'python-version:\s*"([\d.]+)"', text))
        assert versions, "CI 里没有声明 python-version"
        assert versions == {f"{sys.version_info.major}.{sys.version_info.minor}"}, (
            f"CI 用 {versions}，本机是 {sys.version_info.major}.{sys.version_info.minor} —— "
            "版本不一致时 CI 的绿灯不代表本机验证过的那套"
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
