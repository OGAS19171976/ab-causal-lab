"""LaLonde NSW 真实处置数据集的测试：契约、基准、以及"两条路径同一个数"。

这份数据的价值是**别的那份给不了的**：MovieLens 是自造分流的 A/A，只能校准；
NSW 是真的随机化 + 有公开的实验基准（Dehejia–Wahba：+1794.34）。
所以这里钉三件事：

1. **契约**：provenance 的字段齐、sha256 与磁盘一致（契约函数与门共用一份实现）；
2. **基准**：两臂的 re78 均值差必须复现发表值 —— 这是"数据没接错"的证据。
   对不上就是列序/子样本错了，不是"差一点"；
3. **两条路径**：DWD 明细与 ADS 充分统计量算出的 z 必须逐位一致
   （与仓库其它地方同一条不变量）。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_nsw"
PUBLISHED_ATT = 1794.34


def _gate():
    spec = importlib.util.spec_from_file_location(
        "_check_real_traffic_for_nsw", ROOT / "scripts" / "check_real_traffic.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def provenance() -> dict:
    path = DATA / "provenance.json"
    if not path.exists():  # pragma: no cover - 数据已提交，正常不会走到
        pytest.skip("没有 data/real_nsw/provenance.json：先跑 scripts/build_nsw_dataset.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw(provenance) -> dict:
    import pandas as pd

    return {
        table["name"]: pd.read_parquet(DATA / str(table["file"]))
        for table in provenance["tables"]
    }


class TestContract:
    def test_provenance_passes_the_shared_gate(self, provenance):
        """契约校验用的是门自己那份实现（`check_real_traffic.validate`）。"""
        gate = _gate()
        problems, tables = gate.validate(provenance, DATA)
        problems += gate.check_generator_traces(DATA, tables)
        assert problems == [], problems

    def test_upstream_bytes_are_recorded(self, provenance):
        """上游文件的 sha256 要留在 provenance 里。

        源站（NBER）是**可变 URL**，而结论挂在具体某一份字节上：
        记下来才能在下次发现"上游改了文件"。
        """
        files = provenance["upstream_files"]
        assert {f["name"] for f in files} == {
            "nswre74_treated.txt",
            "nswre74_control.txt",
        }
        for entry in files:
            assert len(str(entry["sha256"])) == 64
            assert str(entry["url"]).startswith("https://users.nber.org/")

    def test_citation_and_license_are_present(self, provenance):
        """CC BY-NC 的数据必须带着出处（三篇原文）与许可。"""
        assert "CC BY-NC" in provenance["license"]
        assert len(provenance["citation"]) == 3
        assert any("LaLonde" in c for c in provenance["citation"])

    def test_weights_sum_to_one_and_are_labelled_realized(self, provenance):
        variants = provenance["experiments"][0]["variants"]
        assert abs(sum(variants.values()) - 1.0) < 1e-6
        # 权重是**实测份额**，不是设计值 —— 这一点必须写在 provenance 里，
        # 否则读的人会以为 SRM 那个 0 是"分流没问题"的证据。
        assert "实测份额" in provenance["notes"]["weights_are_realized"]


class TestShape:
    def test_arm_sizes(self, raw):
        exposure = raw["exposure_log"]
        assert len(exposure) == 445
        counts = exposure["variant"].value_counts().to_dict()
        assert counts == {"control": 260, "treatment": 185}

    def test_every_unit_has_two_pre_and_one_post_event(self, raw):
        events = raw["event_log"]
        assert len(events) == 445 * 3
        assert set(events["event_name"]) == {"earnings"}
        per_unit = events.groupby("user_id").size()
        assert set(per_unit.unique()) == {3}

    def test_covariates_are_in_the_source_file_but_dropped_by_ingest(self, raw):
        """**实测到的事实**：源表带协变量，归一化只保留契约列。

        所以"源表里有"不等于"链路里能用" —— 下一轮做倾向得分对照时，
        要么扩契约、要么直接读源表。这条测试把这个差别钉住，
        免得有人以为 DWS 里已经有 age/education 了。
        """
        profile = raw["user_profile"]
        for column in ("age", "education", "black", "married", "nodegree"):
            assert column in profile.columns
        # 归一化后的布局与合成器一致：**每个逻辑表是一个目录**，
        # 里面是 parquet 分片（00_ods.sql 读的是 `.../user_profile/*.parquet`）。
        parts = sorted((ROOT / "build" / "nsw_traffic" / "user_profile").glob("*.parquet"))
        if not parts:  # pragma: no cover - 检查集里 nsw 步骤会生成它
            pytest.skip("还没有 build/nsw_traffic：先跑 scripts/run_nsw_validation.py")
        import pandas as pd

        ingested = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        assert "age" not in ingested.columns


class TestBenchmark:
    def test_att_reproduces_the_published_value(self, raw):
        """**这是"已知真值"**：两臂 re78 均值差 = 发表值 ± 0.05。"""
        import pandas as pd

        detail = raw["event_log"].merge(
            raw["exposure_log"][["user_id", "variant"]], on="user_id"
        )
        post = detail[detail["ds"] == pd.Timestamp("1978-01-01").date()]
        treated = post.loc[post["variant"] == "treatment", "metric_value"].to_numpy(float)
        control = post.loc[post["variant"] == "control", "metric_value"].to_numpy(float)
        att = treated.mean() - control.mean()
        assert abs(att - PUBLISHED_ATT) < 0.05, att
        se = math.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
        assert abs(se - 671.0) < 1.0, se

    def test_pre_covariate_barely_predicts_the_outcome(self, raw):
        """真实数据教的那件事：corr(re75, re78) 很低，所以 CUPED 几乎不省样本。

        这个断言是**方向性**的（低相关），不是"某个具体数字" ——
        写成具体数字会随子样本漂，而它要钉的是"合成器上的 0.63 不能外推"。
        """
        events = raw["event_log"]
        wide = events.pivot_table(index="user_id", columns="ds", values="metric_value")
        columns = sorted(wide.columns)
        pre = wide[columns[0]] + wide[columns[1]]
        post = wide[columns[2]]
        corr = float(np.corrcoef(pre, post)[0, 1])
        assert abs(corr) < 0.2, corr


class TestPipelineOnRealData:
    """跑在真实数据上：契约 -> 接入 -> 数仓 -> 两条路径同一个数。"""

    @pytest.fixture(scope="class")
    def con(self, project_root: Path):
        db = project_root / "build" / "warehouse_nsw.duckdb"
        if not db.exists():  # pragma: no cover - 检查集里 nsw 步骤会生成它
            pytest.skip("没有 build/warehouse_nsw.duckdb：先跑 scripts/run_nsw_validation.py")
        conn = duckdb.connect(str(db), read_only=True)
        yield conn
        conn.close()

    def test_detail_and_ads_agree(self, con):
        """明细 vs 充分统计量：**同一个数**（与仓库其它地方同一条不变量）。"""
        detail = con.execute(
            "SELECT variant, post_metric FROM dwd_experiment_user WHERE experiment = 'nsw_dw'"
        ).df()
        treated = detail.loc[detail["variant"] == "treatment", "post_metric"].to_numpy(float)
        control = detail.loc[detail["variant"] == "control", "post_metric"].to_numpy(float)
        z_detail = (treated.mean() - control.mean()) / math.sqrt(
            treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size
        )
        rows = {
            str(r[0]): r[1:]
            for r in con.execute(
                "SELECT variant, user_cnt, post_sum, post_sq_sum FROM ads_experiment_result"
                " WHERE experiment = 'nsw_dw'"
            ).fetchall()
        }

        def _z(arm):
            n, s, sq = (float(v) for v in arm)
            return n, s / n, (sq - s * s / n) / (n - 1)

        n_t, m_t, v_t = _z(rows["treatment"])
        n_c, m_c, v_c = _z(rows["control"])
        z_ads = (m_t - m_c) / math.sqrt(v_t / n_t + v_c / n_c)
        assert abs(z_ads - z_detail) < 1e-9, (z_ads, z_detail)
        # 真实随机化 + t=2.674：这个 z 可以读成效应检验
        assert 2.0 < z_ads < 3.5
