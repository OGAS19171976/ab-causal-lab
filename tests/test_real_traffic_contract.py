"""真实数据「门」的测试：契约要拦得住，反冒充要真的拦得住。

这一组测试**不依赖任何真实数据**：它在一个临时目录里造出"外部数据"该有的样子
（三张小 Parquet + 一份 provenance），然后逐个把契约的每一条打坏，
断言门**确实报红**。最后一组让契约**完全成立**，断言门真的把数据接进数仓
（跑通 `load_real_traffic` + `build_warehouse(generate=False)`）——
即这条链路本身是可执行的，不只是几行注释。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_check_real_traffic", ROOT / "scripts" / "check_real_traffic.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate(work_dir: Path, monkeypatch):
    """把门的三个路径都指到临时目录（绝不动仓库里的 data/real）。

    用 ``work_dir``（项目内）而不是 pytest 的 ``tmp_path``：后者落在系统 TEMP 下，
    受限（沙箱）环境里建目录/清理会被拒，整组测试变成 error —— 与
    ``tests/conftest.py::work_dir`` 同一个理由，由 ``tests/test_restricted_env.py`` 机检守着。
    """
    module = _load()
    data = work_dir / "real"
    data.mkdir()
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "PROVENANCE", data / "provenance.json")
    monkeypatch.setattr(module, "TARGET", work_dir / "target")
    monkeypatch.setattr(module, "DB_PATH", work_dir / "wh.duckdb")
    return module, data


def _write_tables(base: Path) -> list[dict]:
    """三张最小但**列名与合成器一致**的表。"""
    exposure = pd.DataFrame(
        {
            "ds": ["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02"],
            "ts": pd.to_datetime(
                ["2026-09-01 10:00", "2026-09-01 11:00",
                 "2026-09-02 10:00", "2026-09-02 11:00"]
            ),
            "user_id": ["u1", "u2", "u1", "u2"],
            "experiment": ["exp_real"] * 4,
            "variant": ["control", "treatment", "control", "treatment"],
        }
    )
    events = pd.DataFrame(
        {
            "ds": ["2026-09-01"] * 4,
            "user_id": ["u1", "u2", "u1", "u2"],
            "event_name": ["interaction"] * 4,
            "metric_value": [1.0, 0.0, 2.0, 1.0],
        }
    )
    profile = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "city": ["成都", "北京"],
            # reg_ds 是**必需列**：归一化器把它当可选，链路的 SQL 会选它
            # （这一条是被端到端测试抓出来的，见 data/real/README.md）
            "reg_ds": ["2026-08-01", "2026-08-02"],
        }
    )
    tables = []
    for name, frame in (
        ("exposure_log", exposure),
        ("event_log", events),
        ("user_profile", profile),
    ):
        path = base / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        tables.append(
            {
                "name": name,
                "file": path.name,
                "rows": len(frame),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return tables


def _provenance(base: Path, **overrides) -> dict:
    prov = {
        "source": "某业务线导出（2026-09-01 ~ 2026-09-02）",
        "exported_at": "2026-09-03",
        "external_generator": True,
        # 度量事件名与护栏事件名是**契约的一部分**（门会从 provenance 读它们，
        # 不许写死默认值 —— 真实数据的度量就叫它自己的名字）
        "metric_event": "interaction",
        "guardrail_events": [],
        "notes": "外部系统导出",
        "experiments": [
            {
                "name": "exp_real",
                "variants": {"control": 0.5, "treatment": 0.5},
                "guardrails": [["latency_p99", "lower_is_better", 0.05]],
            }
        ],
        "tables": _write_tables(base),
    }
    prov.update(overrides)
    return prov


def _dump(base: Path, prov: dict) -> None:
    (base / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class TestNoRealData:
    def test_empty_directory_is_not_a_failure(self, gate, capsys):
        """目录空着不算失败，而且要把这句话**说出来**（不静默返回）。"""
        module, _data = gate
        assert module.main() == 0
        out = capsys.readouterr().out
        assert "没有接入真实数据" in out
        assert "file_absent" in out


class TestContractFailures:
    def test_sha_mismatch_is_caught(self, gate):
        module, data = gate
        prov = _provenance(data)
        prov["tables"][0]["sha256"] = "0" * 64
        _dump(data, prov)
        problems, _tables = module.validate(prov, data)
        assert any("sha256" in p for p in problems)

    def test_row_count_mismatch_is_caught(self, gate):
        module, data = gate
        prov = _provenance(data)
        _dump(data, prov)
        _problems, tables = module.validate(prov, data)
        prov["tables"][0]["rows"] = 999
        assert any("行" in p for p in module.check_generator_traces(data, prov["tables"]))
        assert tables  # 契约本身是过的

    def test_not_declared_external_is_caught(self, gate):
        module, data = gate
        prov = _provenance(data, external_generator=False)
        _dump(data, prov)
        problems, _tables = module.validate(prov, data)
        assert any("external_generator" in p for p in problems)

    def test_synthetic_source_hint_is_caught(self, gate):
        module, data = gate
        prov = _provenance(data, source="ablab 合成器导出")
        _dump(data, prov)
        problems, _tables = module.validate(prov, data)
        assert any("合成数据" in p for p in problems)

    def test_weights_must_sum_to_one(self, gate):
        module, data = gate
        prov = _provenance(data)
        prov["experiments"][0]["variants"] = {"control": 0.7, "treatment": 0.5}
        _dump(data, prov)
        problems, _tables = module.validate(prov, data)
        assert any("权重和" in p for p in problems)

    def test_missing_key_is_caught(self, gate):
        module, data = gate
        prov = _provenance(data)
        prov.pop("tables")
        _dump(data, prov)
        problems, _tables = module.validate(prov, data)
        assert any("tables" in p for p in problems)

    def test_generator_marker_blocks_masquerade(self, gate):
        """**核心断言**：合成器的 .generated 标记必须让"真实数据"进不来。"""
        module, data = gate
        prov = _provenance(data)
        _dump(data, prov)
        (data / ".generated").write_text("ablab-generator", encoding="utf-8")
        problems, _tables = module.validate(prov, data)
        assert any("冒充" in p for p in problems)

    def test_generator_column_blocks_masquerade(self, gate):
        """第二道：表里带 config_fingerprint 列也不行。"""
        module, data = gate
        prov = _provenance(data)
        frame = pd.read_parquet(data / "exposure_log.parquet")
        frame["config_fingerprint"] = "deadbeef"
        frame.to_parquet(data / "exposure_log.parquet", index=False)
        prov["tables"][0]["sha256"] = hashlib.sha256(
            (data / "exposure_log.parquet").read_bytes()
        ).hexdigest()
        _dump(data, prov)
        _problems, tables = module.validate(prov, data)
        assert any(
            "冒充" in p for p in module.check_generator_traces(data, tables)
        )


class TestRequiredColumns:
    def test_missing_required_column_is_caught(self, gate):
        """缺列要在门这里说清，而不是等 SQL 深处 KeyError。"""
        module, data = gate
        prov = _provenance(data)
        frame = pd.read_parquet(data / "user_profile.parquet")
        frame = frame.drop(columns=["reg_ds"])
        frame.to_parquet(data / "user_profile.parquet", index=False)
        prov["tables"][2]["sha256"] = hashlib.sha256(
            (data / "user_profile.parquet").read_bytes()
        ).hexdigest()
        _dump(data, prov)
        _problems, tables = module.validate(prov, data)
        problems = module.check_generator_traces(data, tables)
        assert any("缺必需列" in p and "reg_ds" in p for p in problems)


class TestValidDataRunsTheChain:
    def test_contract_passes_and_ingest_runs(self, gate, capsys):
        """契约成立时真的跑通：归一化 → 同一套 SQL → 数仓里出现这三张表。"""
        module, data = gate
        prov = _provenance(data)
        _dump(data, prov)
        assert module.main() == 0
        out = capsys.readouterr().out
        assert "契约通过" in out and "真接入完成" in out
        assert module.TARGET.exists() and any(module.TARGET.rglob("*.parquet"))
        assert module.DB_PATH.exists()
