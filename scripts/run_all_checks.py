#!/usr/bin/env python
"""跑完整的检查集：锁文件校验 + 测试 + 全部验证脚本。

为什么要有这个脚本，而不是在 CI 和本地各写一遍
----------------------------------------------
"CI 跑哪些、本地跑哪些"如果各维护一份，迟早漂移：某天本地加了个脚本、
CI 没加，于是绿灯的 CI 其实没检查那个脚本。这里把**全部检查的定义**放在一处，
CI 与 `tasks.ps1` 都调它，两边不可能不一致。

有一条容易被忽略的**顺序依赖**：``run_m5_validation.py`` 与 ``run_m6_validation.py``
里的数仓段落需要 ``build/warehouse.duckdb`` 已经存在，否则它们会打印"跳过"。
所以 ``run_warehouse.py`` 必须排在前面 —— 第一版按 M0→M6 顺序跑，
数仓三条路径等价性那一段就静默跳过了，报告里只剩一句"跳过"，而汇总仍然全绿。
这种"因为缺前置条件而静默少测一段"的行为，正是本项目最该防的。

用法::

    python scripts/run_all_checks.py                 # 全部（约 15 分钟）
    python scripts/run_all_checks.py --quick         # 快速（跳过 M4 的完整版）
    python scripts/run_all_checks.py --list          # 只列计划
    python scripts/run_all_checks.py --only m5,m6    # 只跑指定项
    python scripts/run_all_checks.py --skip-tests    # 只跑验证脚本
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def force_utf8_output() -> None:
    """固定本进程 stdout / stderr 的编码，让它在**管道和重定向**下也能跑。

    为什么需要：Windows 上 stdout 一旦不是控制台（被管道接走、重定向到文件 ——
    也就是 CI、``Select-Object``、``> log.txt`` 这些最常见用法），Python 就用 locale
    编码（简体中文机器上是 GBK），而本脚本的进度标记 ``▶`` 在 GBK 里没有码位，
    于是**检查一步都没跑就崩在打印标题上**：

        UnicodeEncodeError: 'gbk' codec can't encode character '\\u25b6'

    实测踩到过：``python scripts/run_all_checks.py --only m2 | Select-Object -Last 20``。
    以前没暴露，是因为 ``tasks.ps1``（``chcp 65001`` + ``PYTHONIOENCODING=utf-8``）
    和 CI 的 ``env:`` 都替它设好了 —— 也就是说这个脚本**只在有人绕过那两个入口时**坏掉，
    而 ``scripts/check_report_determinism.py`` 恰好就是绕过它直接调的。

    控制台与管道**分开处理**，这不是洁癖：在 GBK 控制台上强行改成 UTF-8 会让中文
    全部变乱码，那是拿一个 bug 换另一个。所以控制台只加 ``errors="replace"``
    （符号降级成 ``?``，中文照旧），管道才换成 UTF-8（父进程按 UTF-8 读它）。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        # ``reconfigure`` 只存在于 ``io.TextIOWrapper``，而 ``sys.stdout`` 的静态
        # 类型是 ``TextIO`` —— 用 getattr 取一次，取不到就跳过。
        # 这不是为了哄类型检查器：被换成 ``StringIO``（测试里常见）时确实没有它。
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def child_env() -> dict[str, str]:
    """子进程的环境：强制 UTF-8 输出。

    本脚本用 ``encoding="utf-8"`` 抓子进程的 stdout 并写进 ``build/checks/*.log``，
    所以子进程也必须**按 UTF-8 写**，否则日志里的中文全是乱码（抓的时候
    ``errors="replace"`` 不会报错，只会静默变成 ``?`` —— 又是一个"没有信号的降级"）。
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    argv: tuple[str, ...]
    #: 该脚本是否接受 ``--quick`` 这个**参数**。
    #: 注意它与"快速模式下要不要跑"是两件事 —— 第一版把两者合成一个开关，
    #: 结果 ``warehouse`` 在快速模式下被跳过，而 m5/m6 的数仓段落依赖它，
    #: 于是快速模式恰好触发了本项目最该防的行为：**因为缺前置条件而静默少测一段**。
    accepts_quick: bool = False
    #: 仅在快速模式下跳过（目前没有这样的项 —— 保留给将来真正很慢的步骤）
    skip_in_quick: bool = False

    def argv_for(self, *, quick: bool) -> tuple[str, ...] | None:
        """返回本次该跑的 argv；``None`` 表示快速模式下跳过这一项。"""
        if quick and self.skip_in_quick:
            return None
        if quick and self.accepts_quick:
            return self.argv + ("--quick",)
        return self.argv


#: 执行顺序。``warehouse`` 必须排在任何读它的脚本之前 ——
#: m5/m6 的数仓段落要 ``build/warehouse.duckdb`` 存在，否则会**静默跳过**那一段
#: （报告里只剩一句"跳过"，汇总却仍然全绿）。这种"因为缺前置条件而少测一段"
#: 是本项目最该防的行为，所以顺序在这里写死并加了断言。
#: ``gov``（治理：审计 + 护栏）不依赖数仓，但它是秒级的、且属于"平台自己会不会
#: 静默骗人"那一类，所以跟静态检查一起放在前面，早失败早反馈。
#
# ``claims`` **故意排在最后**（在所有会重写 reports/ 的步骤之后）：
# 它检查的是"README 里的数字能不能在 reports/ 里逐字找到"，
# 而那些报告正是后面这些步骤生成的。第一版把 claims 放在最前面，
# 于是"改了报告内容 + 同时加声明"时**第一次跑必红、第二次才绿** ——
# 本轮又踩了两次（治理清单的条数、m6 的新节）。那是最难查的一类假红灯：
# 看起来像声明写错了，其实只是顺序。
_ORDER_HEAD = (
    "lock",
    "env",
    "typed",
    "frontend",
    "realdata",
    "lint",
    "types",
    "unimplemented",
    "warehouse",
    # 运营层紧跟数仓链路：它读的是上一步建出来的库，秒级、且失败要早报
    "whquality",
    "gov",
    "cate",
)


def steps() -> list[Step]:
    py = sys.executable
    every = [
        Step("lock", "锁文件与当前环境一致",
             (py, "scripts/lock_requirements.py", "--check")),
        # 紧跟着 lock：`lock_requirements.py --check` 比的是**版本**，
        # 看不见"包是从哪来的" —— 这两条补的正是那一半（见 7.12 节）。
        Step("env", "环境来源：锁文件里的每个包都必须来自 venv（不只是版本对）",
             (py, "scripts/check_env_origin.py")),
        Step("typed", "三方库类型清单：覆盖 100% 的锁文件条目，逐包给处置",
             (py, "scripts/check_typed_deps.py")),
        # 前端契约：页面 495 行、无构建步骤，靠静态契约挡住"后端改路径前端静默 404"
        # 与"改了 id 控件静默失效"这两类不报错的坏。node 只用来做语法检查。
        Step("frontend", "前端契约：UI 调用的端点/选择器必须存在（含 node 语法检查）",
             (py, "scripts/check_frontend.py")),
        # 真实数据的门：目录空着不算失败（那正是机检项 real_traffic 成立的条件），
        # 但一旦有人放了 provenance.json，契约、反冒充与真接入都要过。
        Step("realdata", "真实数据：契约 + 反冒充（合成数据不许冒充外部数据）",
             (py, "scripts/check_real_traffic.py")),
        # ruff 走 `python -m ruff` 而不是直接调可执行文件 —— 后者在 Windows 上叫
        # ruff.exe、在 Linux 上叫 ruff，路径拼接容易写错，而 python -m 是跨平台的。
        # 下面 mypy 同理。
        Step("lint", "ruff 静态检查（配置见 pyproject，刻意只查真问题）",
             (py, "-m", "ruff", "check", "src", "scripts", "tests")),
        Step("types", "mypy 类型检查（刻意不 --strict，见 pyproject 的说明）",
             (py, "-m", "mypy")),
        Step(
            "unimplemented",
            "README 里每一句『没做』是否仍然成立（防止过时声明）",
            (py, "scripts/check_unimplemented.py"),
        ),
        Step("claims", "README 里的关键数字能否在 reports/ 里找到",
             (py, "scripts/check_readme_claims.py")),
        # 数仓这一步**故意不接受 --quick**：6b 节的比值链路校准（100 个 A/A
        # 复制实验）有数值进了声明清单，快速模式下个数变少、那些数就不成立了
        # —— 那正是第 31 条说的"假红灯"。脚本自己留了 --quick 供人工冒烟。
        Step("warehouse", "数仓链路（m5/m6 的前提）",
             (py, "scripts/run_warehouse.py")),
        # 数仓的**运营层**：血缘 / 数据质量 / 新鲜度。它不重建库，只读上一步的产物 ——
        # 所以必须排在 warehouse 之后（排前面会读到上一次的库）。
        Step("whquality", "数仓运营层：血缘 / 质量测试 / 新鲜度指纹",
             (py, "scripts/check_warehouse_quality.py")),
        Step("gov", "治理：操作审计（append-only）+ 护栏指标显式化",
             (py, "scripts/run_governance_validation.py")),
        Step("cate", "M4 CATE 区间：两条路线的覆盖率（含「校准不了」这个结论）",
             (py, "scripts/run_cate_interval_validation.py"), accepts_quick=True),
        Step("m0", "M0 分流层 + 推断层",
             (py, "scripts/run_m0_validation.py"), accepts_quick=True),
        Step("m1", "M1 CUPED / 比值 / 聚类",
             (py, "scripts/run_m1_validation.py"), accepts_quick=True),
        Step("m2", "M2 序贯与贝叶斯",
             (py, "scripts/run_m2_validation.py"), accepts_quick=True),
        Step("m3", "M3 观察数据因果",
             (py, "scripts/run_m3_validation.py"), accepts_quick=True),
        Step("m4", "M4 异质效应（完整版约 5 分钟）",
             (py, "scripts/run_m4_validation.py"), accepts_quick=True),
        Step("m5", "M5 平台管道 + 数仓三路径",
             (py, "scripts/run_m5_validation.py"), accepts_quick=True),
        Step("m6", "M6 生产口径（口径一致 / 分析单元 / MDE）",
             (py, "scripts/run_m6_validation.py"), accepts_quick=True),
        # 差异审计：合成 vs 真实外部数据（MovieLens）。它要两个仓库都在，
        # 所以排在 warehouse 与 realdata 之后（属于 rest 组，claims 仍收尾）。
        Step("realdiff", "差异审计：合成 vs 真实外部数据（哪些数字变了）",
             (py, "scripts/run_real_vs_synthetic.py")),
    ]
    head = [s for k in _ORDER_HEAD for s in every if s.key == k]
    # claims 收尾：它要读的 reports/ 由前面所有步骤生成
    tail = [s for s in every if s.key == "claims"]
    rest = [s for s in every if s.key not in _ORDER_HEAD and s.key != "claims"]
    ordered = head + rest + tail
    assert [s.key for s in ordered[: len(_ORDER_HEAD)]] == list(_ORDER_HEAD), "前置顺序被破坏"
    assert ordered[-1].key == "claims", "claims 必须是最后一步（它读别的步骤写的报告）"
    return ordered





def run_step(step: Step, *, log_dir: Path, quick: bool) -> tuple[int, float]:
    argv = step.argv_for(quick=quick)
    if argv is None:
        print(f"  [跳过] {step.key}：快速模式不支持它")
        return 0, 0.0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.key}.log"
    print(f"  ▶ {step.key:<10} {step.title}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env(),
    )
    elapsed = time.time() - t0
    log_path.write_text(
        f"$ {' '.join(argv)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        encoding="utf-8",
    )
    status = "OK " if proc.returncode == 0 else "FAIL"
    print(f"    [{status}] {elapsed:6.1f}s   日志 {log_path.relative_to(ROOT)}", flush=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        for line in tail:
            print(f"      | {line}")
    return proc.returncode, elapsed


#: 上一次每步耗时的落点。它的用处只有一个：**把"这次为什么慢"从猜测变成读数** ——
#: 2026-09-21 那一次整套跑了 94.5 分钟（历史 27 分钟，3.4×），当时只能靠人肉对比
#: 日志去判断"是机器慢还是代码变慢"。现在汇总里直接给出比值与核数。
TIMINGS = ROOT / "build" / "checks" / ".timings.json"


def previous_timings() -> dict[str, float]:
    import json

    if not TIMINGS.exists():
        return {}
    try:
        data = json.loads(TIMINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - 文件坏了就当没有
        return {}
    return {str(k): float(v) for k, v in data.items()}


def remember_timings(results: list[tuple[str, int, float]]) -> None:
    """把这次每步耗时并进基线。

    **必须合并而不是覆盖**：`--only m5` 这种单步运行如果直接覆盖，
    下次整套跑的时候基线里就只剩一步，比值列全是"-" ——
    我第一版就是这么写的，跑了一次 `--only realdata` 之后基线被清成一行。
    合并规则：这次跑过的步骤更新，没跑的保留旧值。
    """
    import json

    merged = previous_timings()
    merged.update({k: round(v, 2) for k, _c, v in results})
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    # 放在第一行：下面每一句 print 都可能带中文或 ▶，包括 --help 之后的计划列表。
    force_utf8_output()
    ap = argparse.ArgumentParser(description="跑完整检查集")
    ap.add_argument("--quick", action="store_true", help="快速模式（各脚本的 --quick）")
    ap.add_argument("--list", action="store_true", help="只列计划，不执行")
    ap.add_argument("--only", default=None, help="只跑指定验证项（逗号分隔，如 m5,m6）；pytest 用 --skip-tests 控制")
    ap.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    ap.add_argument("--skip-lock", action="store_true", help="跳过锁文件校验")
    ap.add_argument("--skip-lint", action="store_true", help="跳过 ruff")
    ap.add_argument("--log-dir", default=str(ROOT / "build" / "checks"))
    args = ap.parse_args()

    plan = steps()
    if args.only:
        wanted = {k.strip() for k in args.only.split(",") if k.strip()}
        plan = [s for s in plan if s.key in wanted]
    if args.skip_lock:
        plan = [s for s in plan if s.key != "lock"]
    if args.skip_lint:
        plan = [s for s in plan if s.key != "lint"]

    if args.list:
        print(f"计划（{'快速' if args.quick else '完整'}）：")
        if not args.skip_tests:
            print(f"  {'tests':<10} pytest tests/")
        for s in plan:
            argv = s.argv_for(quick=args.quick)
            if argv is None:
                print(f"  {s.key:<10} {s.title}   [快速模式下跳过]")
            else:
                # 从**真实 argv** 派生展示，不要从 quick 开关猜 ——
                # 猜的话会给不接受 --quick 的脚本（lock / warehouse）也标上它
                extra = "  --quick" if "--quick" in argv else ""
                print(f"  {s.key:<10} {s.title}{extra}")
        return 0

    log_dir = Path(args.log_dir)
    started = time.time()
    results: list[tuple[str, int, float]] = []

    if not args.skip_tests:
        print(f"  ▶ {'tests':<10} pytest tests/", flush=True)
        t0 = time.time()
        log_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=child_env(),
        )
        (log_dir / "tests.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        elapsed = time.time() - t0
        # pytest 的 -q 会吞掉汇总行，所以从输出里抠 "N passed"；抠不到就只看退出码
        summary = next(
            (ln.strip() for ln in reversed(proc.stdout.splitlines()) if "passed" in ln), ""
        )
        print(f"    [{'OK ' if proc.returncode == 0 else 'FAIL'}] {elapsed:6.1f}s   {summary}",
              flush=True)
        results.append(("tests", proc.returncode, elapsed))

    for step in plan:
        code, elapsed = run_step(step, log_dir=log_dir, quick=args.quick)
        results.append((step.key, code, elapsed))

    total = time.time() - started
    failed = [k for k, code, _ in results if code != 0]
    print()
    print("=" * 74)
    prev = previous_timings()
    print(f"{'项':<12}{'结果':<8}{'耗时':>10}{'上次':>10}{'倍数':>7}")
    for key, code, elapsed in results:
        before = prev.get(key)
        ratio = f"{elapsed / before:>6.1f}x" if before and before > 0.5 else "      -"
        shown = f"{before:>9.1f}s" if before else "        -"
        print(f"{key:<12}{'OK' if code == 0 else 'FAIL':<8}{elapsed:>9.1f}s{shown}{ratio}")
    if prev:
        # 只对**秒级以上**的步骤提示：0.1s → 0.6s 也是 6 倍，
        # 但那说明不了机器负载（第一版就报过一次"typed 慢 5.7 倍"，
        # 而它总共只花了 1.1 秒 —— 这种提示会教人忽略提示）。
        slowest = max(
            (
                (k, v / prev[k])
                for k, _c, v in results
                if prev.get(k, 0) > 0.5 and v >= 5.0
            ),
            key=lambda kv: kv[1],
            default=None,
        )
        if slowest and slowest[1] >= 1.5:
            print()
            print(f"  本次整体比上一次慢：最慢的一步是 {slowest[0]}（{slowest[1]:.1f}×）。")
            print(f"  机器：CPU {os.cpu_count()} 核；如果每步都同比例变慢，"
                  "先怀疑机器负载（内存/其它进程），不是代码回归 ——")
            print("  代码回归通常只让**少数几步**变慢，而负载会让所有步骤一起变慢。")
    print("-" * 74)
    print(f"合计 {len(results)} 项，用时 {total:.0f}s（{total / 60:.1f} 分钟）")
    print("全部通过" if not failed else f"**失败：{', '.join(failed)}**")
    remember_timings(results)
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
