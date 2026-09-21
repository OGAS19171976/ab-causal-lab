#!/usr/bin/env python
"""**前端契约检查**：单文件页面与 HTTP 接口之间的四件事。

为什么值得做成检查
------------------
``src/ablab/platform/static/index.html`` 是 495 行**无构建步骤**的单文件页面
（一个 ``api(path, opts)`` helper 包住 ``fetch``）。它没有任何测试，
而它最容易出的两类事故都不报错：

  * **后端改了路径、前端静默 404** —— 页面只是转圈或什么都不显示；
  * **改了某个控件的 id、JS 静默失效** —— 按钮点了没反应，控制台一片安静。

这两类都是"能跑的坏"，只有静态契约检查能提前抓住。四件事：

  1. **路由契约**：页面调用的每条端点都必须在服务端路由表里（含方法）；
  2. **反向契约**：路由表里没有出现在页面上的端点，必须有一条**写下来的决定**
     （有意不做，而不是忘了）—— 这一条把"UI 只覆盖 3/15"从不可见的事实
     变成一张决定表；
  3. **选择器契约**：``$("#x")`` / ``querySelector(".y")`` 指向的 id/class
     必须真的存在（页面里或 JS 自己渲染出来的）；
  4. **语法**：``node --check`` 对页面里的 JS 做一次真正的语法解析
     （括号/引号/模板字符串的错在提交前就暴露，而不是等浏览器）。

路由表走**两条独立路径**取：静态解析 ``api.py`` 里的 ``@app.<method>("...")``
装饰器，以及真的 ``create_app(临时注册表)`` 之后读 ``app.routes`` —— 两条
算出来不一样就报错（例如以后有人改用 ``include_router``，静态那一条会漏）。

诚实的边界：这一条检查的是**契约**（路径/方法/选择器/语法），不是**行为**。
"点了按钮会不会真的做对"要靠 HTTP 层测试（`tests/test_platform_*.py` 已覆盖
状态码、乐观锁 412、审计留痕）与人工看一眼；Playwright 那类端到端测试
与本仓库已有的 HTTP 测试重叠度高，性价比低，这里**刻意不做**。
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = ROOT / "src" / "ablab" / "platform" / "api.py"
INDEX = ROOT / "src" / "ablab" / "platform" / "static" / "index.html"

#: FastAPI **自带**的路由（文档与 OpenAPI）。它们不是应用的接口，
#: 不参与"UI 用没用"的判据，但要在输出里**列出来** —— 第一版把它们当成
#: "界面没用到且没有决定"报了 4 条假阳性。
FRAMEWORK_ROUTES: frozenset[tuple[str, str]] = frozenset({
    ("GET", "/docs"),
    ("GET", "/docs/oauth2-redirect"),
    ("GET", "/openapi.json"),
    ("GET", "/redoc"),
})

#: 路由表里**有意**没有出现在页面上的端点 → 理由。
#: 反向也会查：表里的端点如果后来被 UI 用上了、或者路由没了，同样报错
#: （与 `check_typed_deps.UNTYPED_DECISIONS` 同一个规矩：决定不能烂在表里）。
UNUSED_API_DECISIONS: dict[str, str] = {
    "GET /healthz": "探活接口，给运维与 CI 用，页面不需要",
    "GET /": "静态页面本身（服务端把 index.html 发出来），不是页面要调的 API",
    "GET /api/experiments/{experiment_id}": (
        "详情页用列表返回的字段直接渲染；这条是给脚本/curl 单取一条用的"
    ),
    "POST /api/experiments/{experiment_id}/estimator": (
        "改判定口径会**改变结论的解释**，页面刻意不提供入口（只留带 token 的脚本调用）"
    ),
    "GET /api/experiments/{experiment_id}/events": "审计留痕：页面没有审计页，运维用 curl 查",
    "GET /api/events": "同上（全局审计流）",
    "GET /api/warehouse/experiments": "数仓里有哪几条实验：属于运维探索，页面不做",
    "POST /api/experiments/{experiment_id}/bind": "绑定数仓需要选表与确认，交给脚本",
    "POST /api/experiments/{experiment_id}/stop": (
        "停实验是不可逆动作：宁可不在页面上放一个容易被误点的按钮"
    ),
    "DELETE /api/experiments/{experiment_id}": "删除是破坏性操作，只留 admin 的 curl 路径",
    "POST /api/design/power": "设计期算功效：设计期用 Python API，页面只管在跑的实验",
    "PATCH /api/experiments/{experiment_id}/status": (
        "改实验状态（draft/running/...）：页面只做创建与查看，状态流转留给脚本"
    ),
}


def read_index() -> str:
    if not INDEX.exists():  # pragma: no cover - 文件是这个仓库的一部分
        raise FileNotFoundError(f"找不到前端页面：{INDEX}")
    return INDEX.read_text(encoding="utf-8")


def script_text(html: str) -> str:
    """页面里 ``<script>`` 的内容（无构建步骤，全部内联）。"""
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    if not blocks:  # pragma: no cover
        raise ValueError("页面里没有 <script> 块")
    return "\n".join(blocks)


def normalize_path(raw: str) -> str:
    """前端模板字面量 → 服务端路径模板。

    ``/api/experiments/${id}/analyze`` → ``/api/experiments/{id}/analyze``，
    再把参数名统一成 ``{param}`` —— 参数叫 id 还是 experiment_id 与契约无关。
    """
    path = re.sub(r"\$\{[^}]*\}", "{param}", raw.strip())
    path = re.sub(r"\{[^}]*\}", "{param}", path)
    return path.split("?")[0].rstrip("/") or "/"


def call_span(js: str, open_paren: int) -> str:
    """从 ``api(`` 的左括号开始，扫到与它配对的那个右括号。

    为什么要扫而不是"取后面 400 个字符"：第一版用固定窗口，
    于是 ``api("/api/experiments")`` 的窗口吃到了**下一条**调用的
    ``method: "POST"``，把它读成 POST —— 这是一个**假阴性**：
    方法写错也可能被放过去（测试里就是这么抓出来的）。
    扫描时要知道自己在不在字符串/模板字符串里，否则括号计数会被
    ``"("`` 之类的字面量带偏。
    """
    depth = 0
    quote: str | None = None
    i = open_paren
    while i < len(js):
        ch = js[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return js[open_paren : i + 1]
        i += 1
    return js[open_paren : open_paren + 400]


def frontend_calls(js: str) -> list[tuple[str, str, int]]:
    """``[(方法, 归一化路径, 行号)]`` —— 从 ``api(...)`` 的调用点抽。"""
    calls: list[tuple[str, str, int]] = []
    for m in re.finditer(r"api\(\s*[`'\"]([^`'\"]+)[`'\"]", js):
        span = call_span(js, js.index("(", m.start()))
        method = "GET"
        mm = re.search(r"method\s*:\s*[\"'](\w+)[\"']", span)
        if mm:
            method = mm.group(1).upper()
        line = js[: m.start()].count("\n") + 1
        calls.append((method, normalize_path(m.group(1)), line))
    return calls


def static_routes() -> set[tuple[str, str]]:
    """静态解析：``@app.<method>("path")`` 装饰器。"""
    tree = ast.parse(API.read_text(encoding="utf-8"))
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if not isinstance(dec.func.value, ast.Name) or dec.func.value.id != "app":
                continue
            if dec.args and isinstance(dec.args[0], ast.Constant):
                routes.add((dec.func.attr.upper(), normalize_path(str(dec.args[0].value))))
    return routes


def live_routes() -> set[tuple[str, str]]:
    """动态取：真的构造一次 app 再读 ``app.routes``（注册表写在临时目录）。"""
    sys.path.insert(0, str(ROOT / "src"))
    from ablab.platform.api import create_app

    # 注册表写在这块临时目录里，**不能**用 tempfile.TemporaryDirectory：
    # Windows 上 sqlite 连接还握着文件句柄，清理时会 WinError 32
    # （"另一个程序正在使用此文件"）。build/ 已经是这个仓库的临时区。
    tmp = ROOT / "build" / "_frontend_check"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    app = create_app(registry_path=tmp / "registry.sqlite")
    out: set[tuple[str, str]] = set()
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        if not path:
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            out.add((method.upper(), normalize_path(str(path))))
    return out


def js_ids(js: str) -> set[str]:
    """JS 自己渲染出来的 id（模板字符串里的 ``id="x"``）—— 否则会误报。"""
    return set(re.findall(r"""id\s*=\s*["']([A-Za-z0-9_-]+)["']""", js))


def selectors(js: str) -> list[tuple[str, str, int]]:
    """``[(类型, 名字, 行号)]``；类型是 ``id`` 或 ``class``。"""
    found: list[tuple[str, str, int]] = []
    patterns = [
        r"""\$\(\s*["']#([A-Za-z0-9_-]+)["']\s*\)""",
        r"""querySelector(?:All)?\(\s*["']#([A-Za-z0-9_-]+)["']\s*\)""",
        r"""getElementById\(\s*["']([A-Za-z0-9_-]+)["']\s*\)""",
    ]
    for pat in patterns:
        for m in re.finditer(pat, js):
            found.append(("id", m.group(1), js[: m.start()].count("\n") + 1))
    for pat in (
        r"""\$\(\s*["']\.([A-Za-z0-9_-]+)["']\s*\)""",
        r"""querySelector(?:All)?\(\s*["']\.([A-Za-z0-9_-]+)["']\s*\)""",
    ):
        for m in re.finditer(pat, js):
            found.append(("class", m.group(1), js[: m.start()].count("\n") + 1))
    return found


def html_ids_classes(html: str) -> tuple[set[str], set[str]]:
    body = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
    ids = set(re.findall(r"""id\s*=\s*["']([A-Za-z0-9_-]+)["']""", body))
    classes: set[str] = set()
    for attr in re.findall(r"""class\s*=\s*["']([^"']*)["']""", body):
        classes.update(attr.split())
    return ids, classes


def node_syntax_check(js: str) -> tuple[bool, str]:
    node = shutil.which("node")
    if node is None:
        return False, "没有找到 node —— 语法检查未执行（装 node ≥ 20，或说明为什么这台机器不该有）"
    with tempfile.TemporaryDirectory(prefix="frontend-js-") as tmp:
        path = pathlib.Path(tmp) / "page.js"
        path.write_text(js, encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [node, "--check", str(path)], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    if proc.returncode == 0:
        return True, "node --check 通过"
    return False, (proc.stderr or proc.stdout or "").strip().splitlines()[-1] if (
        proc.stderr or proc.stdout
    ) else "node --check 失败"


def main() -> int:
    html = read_index()
    js = script_text(html)
    problems: list[str] = []

    calls = frontend_calls(js)
    used = {(method, path) for method, path, _line in calls}

    # ---- 1/2. 路由契约（两条独立路径取路由表，先互相对账）----
    static = static_routes()
    live = live_routes()
    live_app = live - FRAMEWORK_ROUTES
    if static != live_app:
        only_static = sorted(static - live_app)
        only_live = sorted(live_app - static)
        problems.append(
            "静态解析与运行时路由表不一致："
            f"只有静态有 {only_static}；只有运行时才有 {only_live}"
            "（改了注册方式？静态解析要跟着改）"
        )
    routes = live_app

    known = {path for _m, path in routes}
    for method, path, line in calls:
        if path not in known:
            problems.append(f"页面第 {line} 行调用了不存在的端点：{method} {path}")
        elif (method, path) not in routes:
            actual = sorted(m for m, p in routes if p == path)
            problems.append(
                f"页面第 {line} 行用 {method} 调 {path}，但服务端只有 {actual}"
            )

    used_paths = {path for _m, path in used}
    stale_decisions = []
    for key, reason in UNUSED_API_DECISIONS.items():
        method, _, path = key.partition(" ")
        path = normalize_path(path)
        if (method.upper(), path) in used:
            stale_decisions.append(f"{key}: UI 已经在用它了 —— 从决定表里删掉（理由：{reason}）")
        elif (method.upper(), path) not in routes and (
            (method.upper(), path) not in FRAMEWORK_ROUTES
        ):
            stale_decisions.append(f"{key}: 这条路由不存在了 —— 决定表要跟着删")
    problems.extend(stale_decisions)

    decisions_normalized = {
        f"{k.partition(' ')[0].upper()} {normalize_path(k.partition(' ')[2])}": reason
        for k, reason in UNUSED_API_DECISIONS.items()
    }
    decided = {k.partition(" ")[2] for k in decisions_normalized}
    undecided = sorted(
        f"{method} {path}"
        for method, path in routes
        if path not in used_paths and path not in decided
    )
    for item in undecided:
        problems.append(f"{item}: 界面没用到，也没有在 UNUSED_API_DECISIONS 里写明决定")

    # ---- 3. 选择器契约 ----
    ids_in_html, classes_in_html = html_ids_classes(html)
    ids_available = ids_in_html | js_ids(js)
    bad_selectors = []
    for kind, name, line in selectors(js):
        pool = ids_available if kind == "id" else classes_in_html
        if name not in pool:
            bad_selectors.append(f"第 {line} 行选择器 {kind}={name!r} 在页面里不存在")
    problems.extend(bad_selectors)

    # ---- 4. 语法 ----
    ok_syntax, syntax_note = node_syntax_check(js)
    if not ok_syntax:
        problems.append(f"JS 语法检查失败：{syntax_note}")

    # ---- 输出 ----
    print("前端契约检查（页面：src/ablab/platform/static/index.html，无构建步骤）")
    print(f"  页面 {len(html.splitlines())} 行，内联 JS {len(js.splitlines())} 行；"
          f"应用路由 {len(routes)} 条（静态解析 {len(static)} 条，两条路径一致）；"
          f"另有 {len(FRAMEWORK_ROUTES)} 条 FastAPI 自带（文档/OpenAPI，不参与判据）")
    print()
    print("  一、UI → API：页面调用的端点")
    print(f"    {'方法':<7}{'端点':<42}{'行'}")
    for method, path, line in sorted(calls, key=lambda c: c[2]):
        print(f"    {method:<7}{path:<42}L{line}")
    print()
    print("  二、API → UI：路由表里没有出现在页面上的端点（每条都有决定）")
    for method, path in sorted(routes):
        if path in used_paths:
            continue
        key = f"{method} {path}"
        # 变量名不复用 `reason`：上面那个循环把它绑成了 str，
        # 这里 `.get()` 返回 str | None，复用会让 mypy 报 arg-type
        decision = decisions_normalized.get(key)
        mark = "有决定" if decision else "**没有决定**"
        print(f"    {method:<7}{path:<42}{mark}")
        if decision:
            print(f"           └ {decision}")
    print()
    print(f"  三、选择器：页面里引用 {len(selectors(js))} 处，"
          f"其中指向不存在的 {len(bad_selectors)} 处")
    print(f"  四、语法：{syntax_note}")
    print()
    if problems:
        print(f"**{len(problems)} 条不成立**：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("页面与接口的契约一致：端点都在、决定都齐、选择器都能落地、JS 语法通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
