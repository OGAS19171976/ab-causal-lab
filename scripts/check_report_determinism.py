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

用法::

    python scripts/check_report_determinism.py                 # 全部（约 32 分钟）
    python scripts/check_report_determinism.py --only m2,m3    # 快速自检（约 1 分钟）
    python scripts/check_report_determinism.py --skip-warmup   # 复用已有产物直接对照
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def say(line: str = "") -> None:
    """带 flush 的输出。

    子进程（run_all_checks → 各脚本）直接往同一个控制台写，而本脚本被管道接走时
    stdout 是块缓冲的 —— 不 flush 就会出现"子进程的汇总先出来、我这边的标题后出来"
    的错位，读起来像是脚本跑乱了。
    """
    say(line, flush=True)


def snapshot() -> dict[str, str]:
    """reports/ 下每个文件的 SHA256。"""
    out: dict[str, str] = {}
    for path in sorted(REPORTS.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(REPORTS)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


def run_suite(*, only: str | None) -> int:
    argv = [sys.executable, "scripts/run_all_checks.py", "--skip-tests", "--skip-lock"]
    if only:
        argv += ["--only", only]
    proc = subprocess.run(argv, cwd=ROOT)
    return proc.returncode


def first_differences(path: Path, other_sha: str) -> list[str]:
    """文本文件给出头几处不同的行，便于定位（图给不出，就只报哈希不同）。"""
    if path.suffix != ".md":
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    # 这里只能看到"当前"内容，拿不到上一次的正文；所以只提示最可能的原因
    suspects = [ln for ln in lines if "耗时" in ln or ":\\" in ln]
    return [f"（当前内容里仍含可疑行：{s.strip()[:70]}）" for s in suspects[:3]]


def main() -> int:
    ap = argparse.ArgumentParser(description="实测 reports/ 的重跑稳定性")
    ap.add_argument("--only", default=None, help="只跑指定验证项（逗号分隔），用于快速自检")
    ap.add_argument("--skip-warmup", action="store_true", help="跳过第一次（预热）运行")
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
    for key in changed:
        where = REPORTS / key
        say(f"  **变化** {key}")
        say(f"      before {before[key][:16]}…  after {after[key][:16]}…")
        for note in first_differences(where, before[key]):
            say(f"      {note}")

    ok = not (added or removed or changed)
    say()
    say("=" * 74)
    if ok:
        say(f"[OK] {len(after)} 个产物在两次独立运行之间**逐字节相同**")
        say("     —— README 里那条「重跑后报告没变」由推断升级为实测")
    else:
        say(
            f"[FAIL] {len(changed)} 个变了 / 新增 {len(added)} / 消失 {len(removed)}"
            " —— 说明仍有机器相关信息混进产物，或者某处输出本身不确定"
        )
    say(f"     用时 {time.time() - t_start:.0f}s")
    say("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
