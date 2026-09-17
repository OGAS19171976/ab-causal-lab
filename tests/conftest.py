"""共享 fixture。"""

import re
import shutil
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def sql_dir(project_root: Path) -> Path:
    return project_root / "sql"


@pytest.fixture
def work_dir(project_root: Path, request) -> Path:
    """可写的工作目录。

    不用 pytest 自带的 ``tmp_path``：它落在系统 TEMP 下，
    并且在会话结束时会 ``rmtree`` 整个 basetemp —— 在受限（沙箱）环境里
    这两步都可能被拒绝，导致测试在 teardown 阶段莫名失败。
    这里把目录放在项目内的 ``build/_test_tmp``，用完自己清理。
    """
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", request.node.nodeid)[:60]
    path = project_root / "build" / "_test_tmp" / f"{safe}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
