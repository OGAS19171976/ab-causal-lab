#!/usr/bin/env python
"""用户与凭据管理：`add` / `list` / `rotate` / `disable` / `enable`。

为什么需要它：读写接口都要求凭据（``Authorization: Bearer <token>``），
而库里存的是 **token 的 sha256**，不是 token 本身 —— 所以明文只在
``add`` / ``rotate`` 的那一刻打印一次，之后再也取不回来（这是有意的：
库文件泄露不应该等于凭据泄露）。

用法::

    python scripts/manage_users.py add --id ogas --role admin --note "本机管理员"
    python scripts/manage_users.py list
    python scripts/manage_users.py rotate --id ogas --grace-minutes 30
    python scripts/manage_users.py disable --id ogas

凭据的**生命周期**（这一轮补上）：

  * ``--ttl-days``：有效期，默认 90 天；``0`` = 永不过期（要显式选）。
    老库里已有的 token 迁移后是"永不过期"，所以加这一列不会把人踢下线。
  * ``--grace-minutes``（仅 rotate）：换发之后**旧 token 还能用多久**。
    默认 0（立即失效，与之前一致）。给一段窗口，换发就从"停机动作"变成
    "滚动动作" —— 代价是那一小段时间里两个 token 同时有效。

默认操作 ``run_platform.py`` 用的那个注册表库；用 ``--db`` 可以指定别的。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.platform.registry import (  # noqa: E402
    DEFAULT_TOKEN_TTL_DAYS,
    ROLES,
    ExperimentRegistry,
    RegistryError,
)


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
    ap.add_argument(
        "--ttl-days", type=int, default=DEFAULT_TOKEN_TTL_DAYS,
        help=f"有效期天数（add/rotate；默认 {DEFAULT_TOKEN_TTL_DAYS}，0 = 永不过期）",
    )
    ap.add_argument(
        "--grace-minutes", type=int, default=0,
        help="换发后旧 token 还能用多久（仅 rotate；默认 0 = 立即失效）",
    )
    ap.add_argument("--db", default=None, help="注册表库路径（默认与 run_platform.py 一致）")
    args = ap.parse_args(argv)

    path = Path(args.db) if args.db else default_registry_path()
    registry = ExperimentRegistry(path)
    try:
        if args.action == "list":
            users = registry.list_users()
            print(f"库：{path}")
            if not users:
                print("  （没有用户 —— 读写接口会全部返回 401，先 add 一个）")
            for u in users:
                flag = "已停用" if u["disabled"] else "启用"
                expires = u["expires_at"] or "永不过期"
                grace = f" 宽限至 {u['grace_until']}" if u["grace_until"] else ""
                print(f"  {u['id']:<16} {u['role']:<8} {flag:<6} {u['note']}")
                print(f"  {'':<16} 到期：{expires}{grace}")
            return 0

        if not args.user_id:
            print("[错误] 这个操作需要 --id", file=sys.stderr)
            return 2

        if args.action == "add":
            token = registry.add_user(
                args.user_id, role=args.role, note=args.note, ttl_days=args.ttl_days or None
            )
            print(f"已建用户 {args.user_id}（角色 {args.role}），库：{path}")
            print("")
            print(f"  token: {token}")
            print("")
            print("**这个 token 只显示这一次**（库里只存 sha256，取不回来）。")
            print("用法：请求头加上  Authorization: Bearer <token>")
            print(f"丢了就换发：python scripts/manage_users.py rotate --id {args.user_id}")
            ttl = "永不过期" if not args.ttl_days else f"{args.ttl_days} 天后到期"
            print(f"有效期：{ttl}")
            return 0

        if args.action == "rotate":
            token = registry.rotate_token(
                args.user_id,
                grace_minutes=args.grace_minutes,
                ttl_days=args.ttl_days or None,
            )
            if args.grace_minutes > 0:
                print(
                    f"已换发 {args.user_id} 的 token"
                    f"（旧的还能用 {args.grace_minutes} 分钟）："
                )
            else:
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
