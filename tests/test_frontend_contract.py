"""前端契约检查的测试。

这一组测试分两层：

  * **纯函数**（路径归一化、调用点抽取、选择器抽取）—— 契约检查的"眼睛"，
    它们抽错了，后面所有判据都是空转；
  * **故障注入** —— 拿一份**故意写坏**的页面（调用不存在的端点、引用不存在的
    id）跑一遍检查器，断言它**确实报红并指出位置**。检查器最容易骗人的地方
    就是"永远绿"，所以必须有一条测试证明它会红。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_check_frontend", ROOT / "scripts" / "check_frontend.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fe():
    return _load()


class TestFrontendHelpers:
    def test_normalize_path_unifies_templates(self, fe):
        assert fe.normalize_path("/api/experiments/${id}/analyze") == (
            "/api/experiments/{param}/analyze"
        )
        assert fe.normalize_path("/api/experiments/{experiment_id}") == (
            "/api/experiments/{param}"
        )
        assert fe.normalize_path("/api/experiments?limit=3") == "/api/experiments"

    def test_frontend_calls_reads_method_and_position(self, fe):
        js = """
        const a = await api("/api/experiments");
        const b = await api(`/api/experiments/${id}/analyze`, {
          method: "POST", headers: {"Content-Type": "application/json"},
        });
        """
        calls = fe.frontend_calls(js)
        assert [(m, p) for m, p, _l in calls] == [
            ("GET", "/api/experiments"),
            ("POST", "/api/experiments/{param}/analyze"),
        ]
        assert calls[1][2] > calls[0][2]  # 行号是递增的

    def test_selectors_and_pools(self, fe):
        js = '$("#a").onclick = 1; document.querySelector(".b"); getElementById("c");'
        found = {(kind, name) for kind, name, _l in fe.selectors(js)}
        assert found == {("id", "a"), ("class", "b"), ("id", "c")}

    def test_js_created_ids_are_used_as_a_pool(self, fe):
        """JS 自己渲染出来的 id 也要算数，否则会误报（页面是动态拼表的）。"""
        js = 'row.innerHTML = `<td id="cell-1">x</td>`; $("#cell-1").textContent = "y";'
        assert "cell-1" in fe.js_ids(js)

    def test_real_page_passes(self, fe):
        """仓库里现在这一版页面必须通过（否则 CI 就是红的）。"""
        assert fe.main() == 0


class TestFrontendFailureInjection:
    """**核心断言**：页面写错时必须报红，而且要说清错在哪一行。

    这里用 ``work_dir``（项目内的 ``build/_test_tmp``）而不是 pytest 自带的
    ``tmp_path``：后者落在系统 TEMP 下，会话结束时还要 rmtree + chmod，
    而受限（沙箱）环境里那两步都可能被拒 —— 于是**测试本身**变成 15 条 error，
    看着像"页面坏了"，其实只是临时目录写不进去。理由与
    ``tests/conftest.py::work_dir`` 的 docstring 同一条，
    现在由 ``tests/test_restricted_env.py`` 机检守着。
    """

    def _run_with(self, fe, work_dir: Path, html: str) -> tuple[int, str, str]:
        index = work_dir / "index.html"
        index.write_text(html, encoding="utf-8", newline="\n")
        original = fe.INDEX
        fe.INDEX = index
        try:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = fe.main()
            return code, buf.getvalue(), ""
        finally:
            fe.INDEX = original

    def test_unknown_endpoint_is_caught(self, fe, work_dir, capsys):
        html = """
        <html><body><div id="x"></div>
        <script>
        async function go() { return api("/api/does-not-exist"); }
        </script></body></html>
        """
        code, out, _ = self._run_with(fe, work_dir, html)
        capsys.readouterr()
        assert code == 1
        assert "不存在的端点" in out and "/api/does-not-exist" in out

    def test_wrong_method_is_caught(self, fe, work_dir, capsys):
        html = """
        <html><body><div id="x"></div>
        <script>
        async function go() {
          return api("/api/experiments", { method: "DELETE" });
        }
        </script></body></html>
        """
        code, out, _ = self._run_with(fe, work_dir, html)
        capsys.readouterr()
        assert code == 1
        assert "DELETE" in out and "只有" in out

    def test_broken_selector_is_caught(self, fe, work_dir, capsys):
        html = """
        <html><body><div id="x"></div>
        <script>
        $("#not-here").onclick = () => 1;
        </script></body></html>
        """
        code, out, _ = self._run_with(fe, work_dir, html)
        capsys.readouterr()
        assert code == 1
        assert "not-here" in out and "不存在" in out

    def test_syntax_error_is_caught(self, fe, work_dir, capsys):
        html = """
        <html><body><div id="x"></div>
        <script>
        function broken( { return 1;
        </script></body></html>
        """
        code, out, _ = self._run_with(fe, work_dir, html)
        capsys.readouterr()
        assert code == 1
        assert "语法检查失败" in out or "node --check" in out
