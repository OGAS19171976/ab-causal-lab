#!/usr/bin/env python
"""实测 ``reports/`` 的**逐字节稳定性**：同一份代码跑两遍，产物必须一模一样。

为什么必须有这个脚本
--------------------
"重跑一次，报告没变"是这个项目**最硬的一条可复现性证据** ——
它比任何关于"我们很严谨"的声称都实在。但它只有在被真跑出来之后才算证据，
靠推理是不算的（README 里那条边界就是这么写的）。

**它会跑两遍全部验证脚本，约 32 分钟**，所以不进 CI（CI 只要 16 分钟就能绿；
把 32 分钟塞进去会让"快速反馈"这条优点消失）。它属于"发版前 / 改完报告逻辑后
手动跑一次"的那类检查。

实测覆盖范围
------------
* ``.md`` 报告：靠 ``ablab/reporting.py`` 的过滤，耗时与绝对路径不进报告
* ``.png`` 图：matplotlib 的 PNG 输出**是可复现的**（只有 ``Software`` 字段带版本号，
  没有时间戳）。所以图也一起查 —— 换 matplotlib 大版本时它们会变，那是应该变。

判据是**两级**的（``ablab.reporting.Comparison``）
-------------------------------------------------
1. 逐字节相同 —— 最强；
2. 否则看"正文（非数值部分）是否一个字没变 + 数值差异是否在容差内"。

为什么必须有第二级：浮点求和的末位**依赖执行环境**，实测到同一个量在两次独立
建仓之间差 1 ulp（``-3.9243251037`` → ``…038``），而报告里印到小数点后 6 位 ——
读者看不见，逐字节判据却会把它报成失败。**一条会经常假红的检查等于没有检查。**
第二级不是"放宽"，是把差异**量化**出来（打印实测最大相对偏差），
而且它照样会抓住真变化：样本量 30741→30742、耗时 69s→71s 都远超容差。

想要"就是必须逐字节"的场合，加 ``--strict``。

用法::

    python scripts/check_report_determinism.py                 # 全部（约 32 分钟）
    python scripts/check_report_determinism.py --only m2,m3    # 快速自检（约 1 分钟）
    python scripts/check_report_determinism.py --skip-warmup   # 复用已有产物直接对照
    python scripts/check_report_determinism.py --strict        # 数值末位差异也算失败

**必须用项目 venv 的 python 运行**（``.venv\\Scripts\\python.exe``）：这个脚本用
``sys.executable`` 起子进程，用系统 python 起会得到一堆 ModuleNotFoundError。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.reporting import (  # noqa: E402
    DEFAULT_RTOL,
    Comparison,
    compare_report_texts,
)

REPORTS = ROOT / "reports"

#: 数值比对容差。默认值与 ``ablab.reporting`` 里审计自己用的尺度同源（1e-9），
#: 命令行留了 ``--strict`` 给"就是要逐字节"的场合。
rtol = DEFAULT_RTOL


def say(line: str = "") -> None:
    """带 flush 的输出。

    子进程（run_all_checks → 各脚本）直接往同一个控制台写，而本脚本被管道接走时
    stdout 是块缓冲的 —— 不 flush 就会出现"子进程的汇总先出来、我这边的标题后出来"
    的错位，读起来像是脚本跑乱了。
    """
    # 注意这行**必须**是 print：用批量替换把 print( 换成 say( 时，
    # 函数体内部这一处也被换掉了，于是 say 变成自我递归 ——
    # 一个 TypeError 都算不上，直接 RecursionError/参数不匹配。
    # 这是"批量替换后必须重跑一次"的活教材。
    print(line, flush=True)


def snapshot() -> dict[str, str]:
    """reports/ 下每个文件的 SHA256。"""
    out: dict[str, str] = {}
    for path in sorted(REPORTS.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(REPORTS)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


#: 对照运行之前，把 reports/ 整个存一份。
#:
#: 为什么必须存：只留哈希的话，**失败时无从查起** —— 对照运行会把文件覆盖掉，
#: 你只知道"变了"，不知道"哪一行变了"。第一次真跑就吃了这个亏：
#: 36 个文件一致、m6 变了，而 m6 的内容已经被覆盖，无法定位。
#: 3 MB 的复制换"可诊断"，这个交换很划算。
_ARCHIVE = ROOT / "build" / "determinism"


def archive_reports() -> None:
    import shutil

    if _ARCHIVE.exists():
        shutil.rmtree(_ARCHIVE, ignore_errors=True)
    shutil.copytree(REPORTS, _ARCHIVE / "before")
    say(f"      已存档对照前的内容：{_ARCHIVE.relative_to(ROOT)}/before/")


def run_suite(*, only: str | None) -> int:
    """跑一轮全部验证脚本。

    ``env`` 必须显式给：本脚本用 ``sys.executable`` 起子进程并让它们**继承本进程的
    stdout**（也就是被管道接走的那一个）。Windows 上管道会让 Python 回落到 locale
    编码（GBK），而脚本里有 ``▶`` 这类 GBK 编不出来的符号 —— 于是子进程会
    ``UnicodeEncodeError`` 崩在打印标题上，看起来像是"预热运行失败"。
    ``run_all_checks.py`` 自己也会设一遍（它要能在被直接调用时也活下来），
    这里设是为了不依赖调用者的环境。
    """
    argv = [sys.executable, "scripts/run_all_checks.py", "--skip-tests", "--skip-lock"]
    if only:
        argv += ["--only", only]
    proc = subprocess.run(
        argv, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    )
    return proc.returncode


def classify(key: str) -> tuple[Comparison | None, str | None]:
    """对一处变化做定性：只有 ``.md`` 能进一步判断，图只能报"哈希不同"。

    返回 ``(comparison, 不可比的原因)``。
    """
    before = _ARCHIVE / "before" / key
    after = REPORTS / key
    if after.suffix != ".md":
        return None, "二进制产物（只看哈希）"
    if not before.exists():
        return None, "对照前的副本不存在"
    try:
        return (
            compare_report_texts(
                before.read_text(encoding="utf-8"), after.read_text(encoding="utf-8")
            ),
            None,
        )
    except Exception as exc:  # 读不动就老实说读不动，不要当成"通过"
        return None, f"比对失败：{exc}"


def describe(key: str, cmp: Comparison | None, reason: str | None) -> None:
    """把一处差异讲到"能定位"的粒度。"""
    say(f"  **变化** {key}")
    if cmp is None:
        say(f"      {reason}")
    else:
        say(f"      正文（非数值部分）是否相同：{cmp.text_identical}")
        say(f"      数值差异 {cmp.n_numeric_diffs} 处；首处：{cmp.first_diff}")
        if cmp.worst is not None:
            say(f"      最大相对偏差 {cmp.worst.rel_diff:.3e}"
                f"（{cmp.worst.before} -> {cmp.worst.after}）"
                f"，容差 {rtol:g}")
        if cmp.exceeds is not None:
            say(f"      **超出容差**：{cmp.exceeds.before} -> {cmp.exceeds.after}"
                f"（相对偏差 {cmp.exceeds.rel_diff:.3e}）")
    say(f"      diff 命令： git diff --no-index --text "
        f"\"{_ARCHIVE / 'before' / key}\" \"{REPORTS / key}\"")


def main() -> int:
    ap = argparse.ArgumentParser(description="实测 reports/ 的重跑稳定性")
    ap.add_argument("--only", default=None, help="只跑指定验证项（逗号分隔），用于快速自检")
    ap.add_argument("--skip-warmup", action="store_true", help="跳过第一次（预热）运行")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="要求逐字节相同（默认允许数值在容差内变化，但会打印实测最大偏差）",
    )
    args = ap.parse_args()

    scope = args.only or "全部 M0–M6 + 数仓"
    t_start = time.time()

    if not args.skip_warmup:
        say(f"[1/3] 预热运行（{scope}）—— 让产物处于「同一份代码写出来的」状态")
        if run_suite(only=args.only) != 0:
            say("预热运行失败，先修好再谈可复现性")
            return 1
    else:
        say("[1/3] 跳过预热运行（按 --skip-warmup）")

    before = snapshot()
    say(f"      快照：{len(before)} 个文件")
    archive_reports()

    say(f"[2/3] 对照运行（{scope}）")
    if run_suite(only=args.only) != 0:
        say("对照运行失败")
        return 1

    after = snapshot()
    say(f"      快照：{len(after)} 个文件")

    say("[3/3] 对比")
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])

    if added:
        say(f"  新增 {len(added)} 个：{added[:5]}")
    if removed:
        say(f"  消失 {len(removed)} 个：{removed[:5]}")

    #: 正文没变、数值只在容差内的文件；真正判失败的文件
    numeric_only: list[tuple[str, Comparison]] = []
    real_fail: list[str] = []
    for key in changed:
        cmp, reason = classify(key)
        describe(key, cmp, reason)
        if cmp is not None and cmp.ok:
            numeric_only.append((key, cmp))
        else:
            real_fail.append(key)

    worst = max(
        (c.worst.rel_diff for _, c in numeric_only if c.worst is not None), default=0.0
    )
    identical = len(after) - len(changed)
    say()
    say("=" * 74)
    if not (added or removed or real_fail):
        if not numeric_only:
            say(f"[OK] {len(after)} 个产物在两次独立运行之间**逐字节相同**")
            say("     —— README 里那条「重跑后报告没变」由推断升级为实测")
        else:
            say(f"[OK*] {identical}/{len(after)} 个产物逐字节相同；"
                f"{len(numeric_only)} 个只在**印刷精度之外**的末位有差")
            say(f"     最大相对偏差 {worst:.3e}（容差 {rtol:g}）—— "
                f"指的是 {', '.join(k for k, _ in numeric_only)}")
            say("     正文、结构、样本量全部一致；差异小于任何报告里印出来的位数。")
            say("     **这不是逐字节结论**：浮点求和的末位依赖执行环境，")
            say("     要更强的主张就得让数值本身可复现，或者改用定点求和。")
    else:
        say(f"[FAIL] 逐字节变了 {len(changed)} 个 / 新增 {len(added)} / 消失 {len(removed)}；"
            f"其中 {len(real_fail)} 个是**真差异**：{real_fail[:5]}")
        say("     真差异 = 正文变了，或数值变化超出容差（数据变了/口径变了/耗时辰进来了）")
    if args.strict and numeric_only:
        say("     （--strict：数值末位差异也算失败）")
    say(f"     用时 {time.time() - t_start:.0f}s")
    say("=" * 74)

    if added or removed or real_fail:
        return 1
    return 1 if (args.strict and numeric_only) else 0


if __name__ == "__main__":
    raise SystemExit(main())
