#!/usr/bin/env python
"""用户与凭据管理：`add` / `list` / `rotate` / `disable` / `enable`。

为什么需要它：写接口现在要求凭据（``Authorization: Bearer <token>``），
而库里存的是 **token 的 sha256**，不是 token 本身 —— 所以明文只在
``add`` / ``rotate`` 的那一刻打印一次，之后再也取不回来（这是有意的：
库文件泄露不应该等于凭据泄露）。

用法::

    python scripts/manage_users.py add --id ogas --role admin --note "本机管理员"
    python scripts/manage_users.py list
    python scripts/manage_users.py rotate --id ogas      # 旧 token 立即失效
    python scripts/manage_users.py disable --id ogas

默认操作 ``run_platform.py`` 用的那个注册表库；用 ``--db`` 可以指定别的。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.platform.registry import ROLES, ExperimentRegistry, RegistryError  # noqa: E402


def _utf8_output() -> None:
    """管道下 Windows 默认是 GBK，中文 token 提示会乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:  # pragma: no cover - 只在极端环境下失败
                pass


def default_registry_path() -> Path:
    """与 ``run_platform.py`` 保持同一个默认值（否则会建到两个库上）。"""
    from ablab.platform.api import default_registry_path as _default

    return Path(_default())


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    ap = argparse.ArgumentParser(description="用户与凭据管理")
    ap.add_argument("action", choices=("add", "list", "rotate", "disable", "enable"))
    ap.add_argument("--id", dest="user_id", help="用户 id（add/rotate/disable/enable 必填）")
    ap.add_argument("--role", default="editor", choices=ROLES, help="角色（仅 add）")
    ap.add_argument("--note", default="", help="备注（仅 add）")
    ap.add_argument("--db", default=None, help="注册表库路径（默认与 run_platform.py 一致）")
    args = ap.parse_args(argv)

    path = Path(args.db) if args.db else default_registry_path()
    registry = ExperimentRegistry(path)
    try:
        if args.action == "list":
            users = registry.list_users()
            print(f"库：{path}")
            if not users:
                print("  （没有用户 —— 写接口会全部返回 401，先 add 一个）")
            for u in users:
                flag = "已停用" if u["disabled"] else "启用"
                print(f"  {u['id']:<16} {u['role']:<8} {flag:<6} {u['note']}")
            return 0

        if not args.user_id:
            print("[错误] 这个操作需要 --id", file=sys.stderr)
            return 2

        if args.action == "add":
            token = registry.add_user(
                args.user_id, role=args.role, note=args.note
            )
            print(f"已建用户 {args.user_id}（角色 {args.role}），库：{path}")
            print("")
            print(f"  token: {token}")
            print("")
            print("**这个 token 只显示这一次**（库里只存 sha256，取不回来）。")
            print("用法：请求头加上  Authorization: Bearer <token>")
            print(f"丢了就换发：python scripts/manage_users.py rotate --id {args.user_id}")
            return 0

        if args.action == "rotate":
            token = registry.rotate_token(args.user_id)
            print(f"已换发 {args.user_id} 的 token（旧的立即失效）：")
            print("")
            print(f"  token: {token}")
            return 0

        registry.disable_user(args.user_id, disabled=args.action == "disable")
        verb = "停用" if args.action == "disable" else "启用"
        print(f"已{verb} {args.user_id}")
        return 0
    except RegistryError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    finally:
        registry.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
