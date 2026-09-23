"""**受限环境契约**：检查集不许假设"系统临时目录可写"。

为什么值得单独测
----------------
同一类事故在这个仓库里出现过**三次**，前两次都只修了当时踩到的那一处：

1. ``tests/conftest.py``：pytest 自带的 ``tmp_path`` 落在系统 TEMP 下，
   会话结束还要 rmtree + chmod，而受限环境里两步都可能被拒绝 ——
   所以这里的测试统一改用项目内的 ``work_dir``；
2. ``scripts/run_governance_validation.py``：``tempfile.mkdtemp()`` 建的目录
   带 0o700，"**建得出来、写不进去**"，整个 gov 检查因此变红（CI 上正常，
   所以一直没被发现）；
3. **第三次（本轮）**：``scripts/check_frontend.py`` 的 node 语法检查还在用
   ``tempfile.TemporaryDirectory``，``tests/test_frontend_contract.py`` 与
   ``tests/test_real_traffic_contract.py`` 还在用 ``tmp_path``。

第 3 次的形态最坏，值得记下来：语法检查**本身**没跑成，异常却发生在**清理**阶段，
于是前端检查变成一段 traceback → 治理报告里少了"页面与接口的契约一致" →
下游 ``claims`` 报"声明漂移了"。**一条与环境有关的 traceback，最后伪装成了一条
文档漂移**。这正是设计决策 31 说的假红灯。

前两次靠"记得"没修住，所以这里把它变成一条机检：
``scripts/`` 与 ``tests/`` 下不许再出现 ``tmp_path`` 或 ``tempfile`` 的临时目录 API。

判据走 **AST**（不看注释与 docstring）—— 解释这件事的那些注释本身不该误报，
而"import 了但没用"由 ruff 管，这里只管**用**。

诚实的边界：这条只管**检查集**（``scripts/`` + ``tests/``）。库代码将来若真的
需要临时文件（例如"同目录原子写"必须先写临时文件再 rename），那是另一个判断；
而检查集必须能在收紧的环境里跑出与 CI 相同的结论。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 这些 API 的位置与权限由**系统 TEMP**决定，落点不受本仓库控制。
_TEMP_API = frozenset({
    "TemporaryDirectory",
    "TemporaryFile",
    "NamedTemporaryFile",
    "mkdtemp",
    "mkstemp",
    "gettempdir",
    "gettempdirb",
})

#: 扫描范围：检查集本身。
_SCANNED = (ROOT / "scripts", ROOT / "tests")


def scan_source(text: str, name: str) -> list[str]:
    """返回这段源码里违反约定的位置（``文件名:行号 + 说明``）。"""
    hits: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.arg) and node.arg.startswith("tmp_path"):
            hits.append(
                f"{name}:{node.lineno} 形参 {node.arg!r} —— pytest 的 tmp_path "
                "落在系统 TEMP 下，改用 conftest 的 work_dir"
            )
        elif isinstance(node, ast.Name) and node.id in _TEMP_API:
            hits.append(f"{name}:{node.lineno} 用了 {node.id}（系统临时目录 API）")
        elif isinstance(node, ast.Attribute) and node.attr in _TEMP_API:
            hits.append(f"{name}:{node.lineno} 用了 {node.attr}（系统临时目录 API）")
    return hits


def scan_tree() -> tuple[list[Path], list[str]]:
    """扫一遍检查集，返回 ``(读到的文件, 违规说明)``。"""
    files: list[Path] = []
    problems: list[str] = []
    for base in _SCANNED:
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
            problems += scan_source(
                # 报位置时统一用 POSIX 分隔符：这里断言的是"哪个文件"，
                # 而不该随操作系统变（Windows 上是反斜杠）。
                path.read_text(encoding="utf-8"),
                path.relative_to(ROOT).as_posix(),
            )
    return files, problems


class TestNoSystemTempAssumption:
    def test_no_system_temp_assumptions(self):
        _files, problems = scan_tree()
        assert not problems, (
            "检查集里又出现了对系统临时目录的依赖：\n  "
            + "\n  ".join(problems)
            + "\n\n受限环境里这会变成 traceback 或一片 error（前两次的形态见本文件"
            "开头的记录）。改法：脚本里在 build/ 下用默认权限建目录"
            "（见 run_governance_validation.py），测试里用 conftest 的 work_dir。"
        )

    def test_scan_actually_reads_the_tree(self):
        """扫描必须真的读到文件 —— 否则它最可能的失效方式是**永远绿**。"""
        files, _problems = scan_tree()
        rel = {p.relative_to(ROOT).as_posix() for p in files}
        assert len(files) > 20, f"只扫到 {len(files)} 个文件，扫描范围可能写错了"
        for must in (
            "tests/conftest.py",
            "tests/test_frontend_contract.py",
            "tests/test_real_traffic_contract.py",
            "scripts/check_frontend.py",
            "scripts/run_governance_validation.py",
        ):
            assert must in rel, f"{must} 没被扫到 —— 扫描范围漏了它"


class TestScannerItself:
    """故障注入：给扫描器喂一段**故意写坏**的源码，断言它抓得住。"""

    PLANTED = (
        "import tempfile\n"
        "from tempfile import mkdtemp\n"
        "\n"
        "def test_x(tmp_path):\n"
        "    with tempfile.TemporaryDirectory() as d:\n"
        "        open(d + '/a', 'w')\n"
        "    mkdtemp()\n"
    )

    def test_each_planted_shape_is_caught(self):
        hits = scan_source(self.PLANTED, "planted.py")
        joined = "\n".join(hits)
        assert "tmp_path" in joined, hits
        assert "TemporaryDirectory" in joined, hits
        assert "mkdtemp" in joined, hits

    def test_comments_and_docstrings_do_not_trip_it(self):
        """解释这件事的注释/docstring 不该误报（判据走 AST 的理由）。"""
        text = (
            '"""禁止 tmp_path 与 tempfile.mkdtemp()（系统 TEMP 不可写）。"""\n'
            "# 也别用 tempfile.TemporaryDirectory：清理阶段要 chmod\n"
            "VALUE = 1\n"
        )
        assert scan_source(text, "comments.py") == []
