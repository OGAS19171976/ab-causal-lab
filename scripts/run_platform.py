#!/usr/bin/env python
"""启动实验平台服务。

运行::

    python scripts/run_platform.py                    # 127.0.0.1:8077
    python scripts/run_platform.py --port 9000        # 换端口
    python scripts/run_platform.py --reset            # 清空注册表重来
    python scripts/run_platform.py --no-warehouse     # 只用合成数据，不接数仓

这是**本项目自带的演示服务**，与你在用的 DSH Web GUI（默认 3080）互不相干，
监听独立的端口。首次启动会写入三条合成演示实验（不会覆盖已存在的）；
若 ``build/warehouse.duckdb`` 存在（由 ``run_warehouse.py`` 建），
还会多写一条**绑定数仓**的实验，于是页面上能同时看到两条数据源。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablab.platform.api import (  # noqa: E402
    create_app,
    default_registry_path,
    default_warehouse_path,
)
from ablab.platform.demo import seed_demo  # noqa: E402
from ablab.platform.registry import ExperimentRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="启动 ab-causal-lab 实验平台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--registry", default=None, help="注册表路径（默认项目内 build/platform）")
    ap.add_argument("--warehouse", default=None, help="DuckDB 数仓路径（默认 build/warehouse.duckdb）")
    ap.add_argument("--no-warehouse", action="store_true", help="不接数仓，只跑合成数据")
    ap.add_argument("--reset", action="store_true", help="先删掉已有注册表")
    ap.add_argument("--no-seed", action="store_true", help="不写演示数据")
    args = ap.parse_args()

    path = Path(args.registry) if args.registry else default_registry_path()
    if args.reset and path.exists():
        path.unlink()
        print(f"已删除注册表 {path}")

    wh_path = None if args.no_warehouse else (
        Path(args.warehouse) if args.warehouse else default_warehouse_path()
    )
    wh_ok = wh_path is not None and wh_path.exists()

    # 先探端口：uvicorn.run 之前就把网址打印出来是不诚实的 ——
    # 端口被占时进程会直接崩，而用户已经看到"服务地址"了。
    import socket

    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((args.host, args.port)) == 0:
            print(f"[错误] {args.host}:{args.port} 已被占用，请用 --port 换一个端口。")
            return 1

    registry = ExperimentRegistry(path)
    existing = registry.count()
    added = 0 if args.no_seed else seed_demo(registry, warehouse_available=wh_ok)
    total = registry.count()
    registry.close()

    print(f"注册表: {path}")
    print(f"实验数: {existing} -> {total}（本次新增 {added} 条演示数据）")
    if wh_ok:
        print(f"数仓:   {wh_path}  -> 已绑定数仓的实验会走真实链路")
    elif wh_path is not None:
        print(f"数仓:   {wh_path} 不存在 -> 全部走合成数据（先跑 run_warehouse.py 即可）")
    else:
        print("数仓:   已禁用（--no-warehouse）")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"接口文档: http://{args.host}:{args.port}/docs")
    print("（这是本项目自带的演示服务，与 DSH Web GUI 的 3080 端口无关）")

    import uvicorn

    uvicorn.run(
        create_app(path, warehouse_path=wh_path if wh_ok else None),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
