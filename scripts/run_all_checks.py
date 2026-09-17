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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
_ORDER_HEAD = ("lock", "warehouse")


def steps() -> list[Step]:
    py = sys.executable
    every = [
        Step("lock", "锁文件与当前环境一致",
             (py, "scripts/lock_requirements.py", "--check")),
        Step("warehouse", "数仓链路（m5/m6 的前提）",
             (py, "scripts/run_warehouse.py")),
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
    ]
    head = [s for k in _ORDER_HEAD for s in every if s.key == k]
    rest = [s for s in every if s.key not in _ORDER_HEAD]
    ordered = head + rest
    assert [s.key for s in ordered[:2]] == list(_ORDER_HEAD), "前置顺序被破坏"
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
        argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
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


def main() -> int:
    ap = argparse.ArgumentParser(description="跑完整检查集")
    ap.add_argument("--quick", action="store_true", help="快速模式（各脚本的 --quick）")
    ap.add_argument("--list", action="store_true", help="只列计划，不执行")
    ap.add_argument("--only", default=None, help="只跑指定验证项（逗号分隔，如 m5,m6）；pytest 用 --skip-tests 控制")
    ap.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    ap.add_argument("--skip-lock", action="store_true", help="跳过锁文件校验")
    ap.add_argument("--log-dir", default=str(ROOT / "build" / "checks"))
    args = ap.parse_args()

    plan = steps()
    if args.only:
        wanted = {k.strip() for k in args.only.split(",") if k.strip()}
        plan = [s for s in plan if s.key in wanted]
    if args.skip_lock:
        plan = [s for s in plan if s.key != "lock"]

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
    print(f"{'项':<12}{'结果':<8}{'耗时':>10}")
    for key, code, elapsed in results:
        print(f"{key:<12}{'OK' if code == 0 else 'FAIL':<8}{elapsed:>9.1f}s")
    print("-" * 74)
    print(f"合计 {len(results)} 项，用时 {total:.0f}s（{total / 60:.1f} 分钟）")
    print("全部通过" if not failed else f"**失败：{', '.join(failed)}**")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
