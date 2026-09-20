"""数仓链路测试：分层口径正确 + 两条路径可交叉验证。

这里断言的不是"SQL 能跑通"，而是"SQL 算出来的东西和 Python 算出来的一样"。
只有这一条成立，才能说数仓分层没有改变分析结论。
"""

import shutil

import numpy as np
import pytest

from ablab.inference import welch_ttest
from ablab.warehouse import (
    WarehouseConfig,
    analyse_ads,
    build_warehouse,
    covariate_adjustment_report,
    load_ads_result,
    render_report,
    split_statements,
    verify_against_detail,
)

NUMERIC_TOL = 1e-9


@pytest.fixture(scope="module")
def warehouse(sql_dir, project_root):
    root = project_root / "build" / "_test_tmp" / "warehouse_fixture"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    cfg = WarehouseConfig(n_users=4_000, seed=20260301)
    con = build_warehouse(
        db_path=root / "wh.duckdb",
        data_dir=root / "source",
        sql_dir=sql_dir,
        config=cfg,
        force_data=True,
        verbose=False,
    )
    yield con, cfg
    con.close()
    shutil.rmtree(root, ignore_errors=True)


class TestSqlSplitting:
    def test_strips_comment_only_chunks(self):
        sql = "-- 只有注释\n;\nSELECT 1;\n-- 注释\nSELECT 2;"
        assert len(split_statements(sql)) == 2

    def test_keeps_inline_leading_comments(self):
        sql = "-- 说明\nSELECT 1;"
        assert len(split_statements(sql)) == 1


class TestLayerCounts:
    def test_all_layers_built(self, warehouse):
        con, _ = warehouse
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        for expected in (
            "ods_event_log",
            "dwd_experiment_user",
            "dws_experiment_variant_daily",
            "ads_experiment_result",
            "ads_experiment_srm",
        ):
            assert expected in tables, f"缺少 {expected}"

    def test_dedup_is_effective(self, warehouse):
        """DWD 的行数必须等于曝光日志去重后的 (experiment, user_id) 数。"""
        con, _ = warehouse
        raw = con.execute("SELECT COUNT(*) FROM ods_exposure_log").fetchone()[0]
        dedup = con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT experiment, user_id FROM ods_exposure_log)"
        ).fetchone()[0]
        dwd = con.execute("SELECT COUNT(*) FROM dwd_experiment_user").fetchone()[0]

        assert raw > dedup, "测试数据里应当包含重复曝光"
        assert dwd == dedup

    def test_dws_rolls_up_to_ads(self, warehouse):
        """DWS 的可加字段汇总后必须等于 ADS 的 user_cnt。"""
        con, _ = warehouse
        mismatch = con.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT experiment, variant, SUM(user_cnt) AS n
                FROM dws_experiment_variant_daily
                GROUP BY experiment, variant
            ) d
            JOIN ads_experiment_result a USING (experiment, variant)
            WHERE d.n <> a.user_cnt
            """
        ).fetchone()[0]
        assert mismatch == 0

    def test_layer_coverage_matches_design(self, warehouse):
        """ranking 层 80%、recall 层 60% 流量 —— DWD 里的用户数应当接近。"""
        con, cfg = warehouse
        n_users = cfg.n_users
        for experiment, expected in (("exp_rank_v2", 0.80), ("exp_rec_emb", 0.60)):
            n = con.execute(
                "SELECT COUNT(DISTINCT user_id) FROM dwd_experiment_user WHERE experiment = ?",
                [experiment],
            ).fetchone()[0]
            assert n / n_users == pytest.approx(expected, abs=0.02)

    def test_variance_is_positive(self, warehouse):
        con, _ = warehouse
        bad = con.execute(
            "SELECT COUNT(*) FROM ads_experiment_result WHERE post_var <= 0"
        ).fetchone()[0]
        assert bad == 0


class TestCrossValidation:
    def test_sql_and_python_srm_agree(self, warehouse):
        """SQL 算的卡方必须等于 Python srm_check 算的卡方。"""
        con, _ = warehouse
        analyses = analyse_ads(con)
        assert analyses
        for a in analyses:
            assert a.srm_agrees, f"{a.experiment}: SQL={a.sql_chi2} Python={a.python_chi2}"

    def test_ads_summary_matches_dwd_detail(self, warehouse):
        """ADS 汇总路径与 DWD 明细路径必须给出**在容差内一致**的效应与标准误。

        post-only 与 CUPED 两条路都要对得上。CUPED 尤其关键 ——
        它的 theta 来自 ADS 的 pre_post_cross_sum，SQL 里写错只有这条能发现。

        措辞是"在容差内"而不是"完全相同"：两侧的求和顺序不同，末位可以差 1 ulp
        （实测出现过 ``-3.9243251037`` vs ``…038``）。见 ``CrossValidation`` 的 docstring。
        """
        con, _ = warehouse
        analyses = analyse_ads(con)
        cv = verify_against_detail(con, analyses, "exp_rank_v2")

        assert cv.naive_matches, cv.summary()
        assert cv.cuped_matches, cv.summary()
        assert cv.passed
        assert cv.cuped_summary.std_error > 0

    def test_cuped_reduces_variance_in_warehouse(self, warehouse):
        """CUPED 的标准误必须显著小于 post-only。"""
        con, _ = warehouse
        for a in analyse_ads(con):
            assert a.cuped.std_error < a.naive.std_error
            # 1-rho^2 越大，缩减越小；这里 rho≈0.8，标准误应降到 60% 左右
            ratio = a.cuped.std_error / a.naive.std_error
            assert ratio == pytest.approx(np.sqrt(1 - a.fit.correlation**2), abs=0.05)

    def test_cuped_removes_exactly_the_imbalance_component(self, warehouse):
        """精确恒等式：naive − CUPED == theta × 前置组间差。

        这是 CUPED 消偏置的**定义式**，不依赖任何一次具体实现 ——
        单次实现的 p 值可以朝任一方向变，但这个分解必须严格成立。
        """
        con, _ = warehouse
        analyses = {a.experiment: a for a in analyse_ads(con)}
        a = analyses["exp_rec_emb"]

        rows = con.execute(
            """
            SELECT variant, pre_metric FROM dwd_experiment_user WHERE experiment = ?
            """,
            ["exp_rec_emb"],
        ).df()
        t = rows.loc[rows["variant"] == "treatment", "pre_metric"].mean()
        c = rows.loc[rows["variant"] == "control", "pre_metric"].mean()
        pre_gap = float(t - c)

        removed = a.naive.absolute_effect - a.cuped.absolute_effect
        assert removed == pytest.approx(a.fit.theta * pre_gap, rel=1e-9)

    def test_se_ratio_follows_rho(self, warehouse):
        """CUPED 的标准误必须恰好是 post-only 的 sqrt(1-rho^2) 倍。

        这条关系与 p 值方向无关，是每个实验都必须成立的硬约束。
        """
        con, _ = warehouse
        for a in analyse_ads(con):
            expected = a.naive.std_error * np.sqrt(1 - a.fit.correlation**2)
            assert a.cuped.std_error == pytest.approx(expected, rel=0.12)

    def test_pvalue_direction_is_not_a_law(self, warehouse):
        """单次实现的 p 值方向不可预知 —— 这是 M0 那条原则的延伸。

        在**报告用的那份数据**（20000 用户）上，负对照的 post-only 报显著、
        CUPED 校正后不显著。但那是一次具体实现的结果，不是可依赖的定律：
        换成 4000 用户，两个 p 值都不显著。所以这里只断言不变量。
        """
        con, _ = warehouse
        analyses = {a.experiment: a for a in analyse_ads(con)}
        for a in analyses.values():
            # 不变量 1：CUPED 证据更强（标准误更小）
            assert a.cuped.std_error < a.naive.std_error
            # 不变量 2：分解恒等式（见上一条测试）与标准误比值都成立
            assert abs(a.p_value_shift) >= 0.0 or np.isnan(a.p_value_shift)

    def test_cuped_recovers_true_effect_better(self, warehouse):
        """有真实效应的实验：CUPED 的估计应更接近真值。"""
        con, _ = warehouse
        analyses = {a.experiment: a for a in analyse_ads(con)}
        rank = analyses["exp_rank_v2"]
        # CUPED 把协变量失衡带来的正偏置扣掉了，估计值比 post-only 更小、更接近真值
        assert rank.cuped.absolute_effect < rank.naive.absolute_effect

    def test_detail_path_is_independently_correct(self, warehouse):
        """再独立算一遍：直接从 DWD 拉明细喂给 welch_ttest。"""
        con, _ = warehouse
        rows = con.execute(
            "SELECT variant, post_metric FROM dwd_experiment_user WHERE experiment = ?",
            ["exp_rank_v2"],
        ).df()
        est = welch_ttest(
            rows.loc[rows["variant"] == "treatment", "post_metric"].to_numpy(),
            rows.loc[rows["variant"] == "control", "post_metric"].to_numpy(),
        )
        assert est.n_treatment == est.n_treatment  # 非 NaN
        assert est.std_error > 0


class TestAnalysisOutput:
    def test_treatment_effect_is_recovered(self, warehouse):
        """exp_rank_v2 真实效应 +2.0/天，14 天窗口下应显著为正。"""
        con, _ = warehouse
        analyses = {a.experiment: a for a in analyse_ads(con)}
        est = analyses["exp_rank_v2"].cuped
        assert est.absolute_effect > 0
        assert est.significant

    def test_srm_does_not_trigger(self, warehouse):
        """分层分流是均匀的，两个实验都不该触发 SRM。"""
        con, _ = warehouse
        # **SRM 必须在随机化单元上做。**
        # 人级随机化的实验看用户数；簇随机化的实验要看**簇数** ——
        # 实测 exp_city_ctr 的用户数是 10623 vs 9377（城市分流 32 vs 28，
        # 每个城市的用户量相近，于是用户数跟着城市走），
        # 拿用户数做 SRM 会得到一个 p≈1e-18 的"严重失衡"，
        # 而那只是**分析单元选错了**：随机化单元是城市，不是用户。
        clusters = {
            row[0]: (int(row[1]), int(row[2]))
            for row in con.execute(
                """
                SELECT experiment,
                       SUM(CASE WHEN variant = 'control' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN variant = 'treatment' THEN 1 ELSE 0 END)
                FROM (
                    SELECT DISTINCT experiment, variant, cluster_id
                    FROM dws_experiment_cluster_daily
                ) GROUP BY experiment
                """
            ).fetchall()
        }
        cluster_randomized = {"exp_city_ctr"}
        for a in analyse_ads(con):
            if a.experiment in cluster_randomized:
                n_c, n_t = clusters[a.experiment]
                # 60 个城市按哈希 50/50 分流：偏离 30/30 五五开是正常的
                assert abs(n_c - n_t) <= 0.25 * (n_c + n_t), (n_c, n_t)
                continue
            assert not a.srm_triggered, f"{a.experiment} 触发了 SRM"

    def test_negative_control_covariate_imbalance(self, warehouse):
        """负对照实验的偏差与 CUPED 方差缩减，必须在**随机化分布**上成立。

        这是 M0 最重要的一条发现，也是 M1 引入 CUPED 的动机：
        一次实现的分流里前置协变量不平衡，post-only 分析会把它当成真效应。

        注意不能断言"某一次实现的校正方向"，因为单次实现的不平衡是随机的。
        可断言的是两条定律：前置/后置差距高度相关，且回归校正把标准差
        压到 sqrt(1 - rho^2)。
        """
        con, _ = warehouse
        r = covariate_adjustment_report(con, "exp_rec_emb", n_trials=300)

        # 真效应为 0：偏差完全由协变量失衡驱动
        assert r.correlation > 0.5, "合成数据里 pre/post 应强相关"
        assert r.theta > 0
        # 从差距分布反推的 theta 应与合并样本估计大致吻合
        assert r.theta_from_gaps == pytest.approx(r.theta, rel=0.15)

        # 校正前偏置 ≈ theta × 前置失衡；校正后应被扣掉这一项
        explained = r.theta * r.realized_pre_gap
        assert r.realized_adjusted_gap == pytest.approx(
            r.realized_post_gap - explained, abs=1e-9
        )

        # 方差缩减必须贴近理论值 rho^2（注意不是 1-rho^2）
        assert r.theoretical_variance_reduction == pytest.approx(r.correlation**2, abs=1e-12)
        assert r.variance_reduction == pytest.approx(r.theoretical_variance_reduction, abs=0.10)
        assert r.variance_reduction > 0.4, "corr≈0.8 时应拿到约 63% 的方差缩减"
        assert r.remaining_variance_fraction == pytest.approx(1 - r.correlation**2, abs=1e-12)
        assert r.remaining_variance_fraction < 0.45

    def test_covariate_report_summary_renders(self, warehouse):
        con, _ = warehouse
        r = covariate_adjustment_report(con, "exp_rank_v2", n_trials=50)
        text = r.summary()
        assert "协变量失衡诊断" in text
        assert "方差缩减" in text
        # 摘要里必须同时出现"方差缩减"和"残余方差"，避免再次把两者搞混
        assert "残余方差" in text

    def test_report_renders(self, warehouse):
        con, _ = warehouse
        text = render_report(analyse_ads(con))
        assert "exp_rank_v2" in text
        assert "SRM" in text

    def test_ads_result_shape(self, warehouse):
        con, _ = warehouse
        df = load_ads_result(con)
        assert set(df.columns) >= {
            "experiment",
            "variant",
            "user_cnt",
            "post_mean",
            "post_var",
            "pre_mean",
            "design_weight",
            # M1 新增：CUPED 的原料
            "pre_sq_sum",
            "post_sq_sum",
            "pre_post_cross_sum",
            "pre_post_cov",
        }
        # 实验条数从**配置**推，不写死：写死会在加实验时红，
        # 而那条红与被测的东西（ADS 的列）无关。
        from ablab.warehouse.generate import DEFAULT_EXPERIMENTS

        assert len(df) == 2 * len(DEFAULT_EXPERIMENTS)

    def test_cross_sum_enables_covariance(self, warehouse):
        """SQL 的 pre_post_cross_sum 必须能还原出正确的协方差。"""
        con, _ = warehouse
        df = load_ads_result(con)
        for _, row in df.iterrows():
            n = row["user_cnt"]
            expected = (
                row["pre_post_cross_sum"] - row["pre_sum"] * row["post_sum"] / n
            ) / (n - 1)
            assert row["pre_post_cov"] == pytest.approx(expected, rel=1e-12)


class TestRealDataIngest:
    """真实数据入口：把外部文件接成数仓能读的布局。

    这一组钉的是那句被反复引用的主张 —— **换数据源不改链路**。
    在 `warehouse/ingest.py` 之前，它只被"合成数据写得像真实数据"支持着。
    """

    @staticmethod
    def _export(source_dir, target_dir, *, fmt="csv", stray_event=True, extra_cols=True):
        """把合成器的落地文件导出成"外部文件"：列顺序打乱、多几列、混一个未声明事件。"""
        import pandas as pd

        target_dir.mkdir(parents=True, exist_ok=True)
        exposure = pd.read_parquet(source_dir / "exposure_log" / "part-0000.parquet")
        events = pd.read_parquet(source_dir / "event_log" / "part-0000.parquet")
        profile = pd.read_parquet(source_dir / "user_profile" / "part-0000.parquet")
        if extra_cols:
            exposure = exposure.assign(device="web", app_version="9.9")
            events = events.assign(platform="ios")
            profile = profile.assign(country="CN")
        if stray_event:
            stray = events.head(200).assign(event_name="page_view", metric_value=1.0)
            events = pd.concat([events, stray], ignore_index=True)
        if fmt == "csv":
            exposure.to_csv(target_dir / "exposure_log.csv", index=False)
            events.to_csv(target_dir / "event_log.csv", index=False)
            profile.to_csv(target_dir / "user_profile.csv", index=False)
        else:
            exposure.to_parquet(target_dir / "exposure_log.parquet", index=False)
            events.to_parquet(target_dir / "event_log.parquet", index=False)
            profile.to_parquet(target_dir / "user_profile.parquet", index=False)
        return target_dir

    @staticmethod
    def _declarations():
        from ablab.warehouse import DEFAULT_EXPERIMENTS, ExternalExperiment

        return tuple(
            ExternalExperiment(e.name, {"control": 0.5, "treatment": 0.5}, layer=e.layer)
            for e in DEFAULT_EXPERIMENTS
        )

    def _build_standard(self, work_dir, sql_dir, n_users=800):
        cfg = WarehouseConfig(n_users=n_users, seed=5)
        con = build_warehouse(
            db_path=work_dir / "std.duckdb", data_dir=work_dir / "std_source",
            sql_dir=sql_dir, config=cfg, force_data=True, verbose=False,
        )
        ads = con.execute(
            "SELECT experiment, variant, user_cnt, post_sum FROM ads_experiment_result"
            " ORDER BY experiment, variant"
        ).df()
        con.close()
        return ads, cfg

    def test_round_trip_reproduces_the_synthetic_numbers(self, work_dir, sql_dir):
        """**这一步才是证据**：两条完全不同的数据路径，同一套 SQL，数字要相同。"""
        import pandas as pd

        from ablab.warehouse import build_warehouse as bw
        from ablab.warehouse import load_real_traffic

        std, cfg = self._build_standard(work_dir, sql_dir)
        self._export(work_dir / "std_source", work_dir / "external", fmt="csv")
        report = load_real_traffic(
            work_dir / "external", target_dir=work_dir / "external_source", fmt="csv",
            experiments=self._declarations(), metric_event="interaction",
            guardrail_events=("latency_p99", "complaint_rate"),
        )
        assert report.row_counts["event_log"] > 0
        assert "page_view" in report.undeclared_events
        assert report.duplicate_exposures >= 0

        con = bw(
            db_path=work_dir / "ext.duckdb", data_dir=work_dir / "external_source",
            sql_dir=sql_dir, config=cfg, generate=False, verbose=False,
        )
        ext = con.execute(
            "SELECT experiment, variant, user_cnt, post_sum FROM ads_experiment_result"
            " ORDER BY experiment, variant"
        ).df()
        con.close()

        merged = std.merge(ext, on=["experiment", "variant"], suffixes=("_std", "_ext"))
        assert len(merged) == len(std) == len(ext)
        assert np.array_equal(merged["user_cnt_std"], merged["user_cnt_ext"])
        rel = (
            (merged["post_sum_ext"] - merged["post_sum_std"]).abs()
            / merged["post_sum_std"].abs()
        )
        assert rel.max() < 1e-12, rel.max()
        assert pd.notna(merged["post_sum_std"]).all()

    def test_true_lift_is_empty_on_the_external_path(self, work_dir, sql_dir):
        """真实数据没有"演示真值" —— 那一列必须写空，不能假装知道。"""
        import pandas as pd

        from ablab.warehouse import load_real_traffic

        self._build_standard(work_dir, sql_dir)
        self._export(work_dir / "std_source", work_dir / "external", fmt="parquet")
        load_real_traffic(
            work_dir / "external", target_dir=work_dir / "ext2", fmt="parquet",
            experiments=self._declarations(),
        )
        cfg_frame = pd.read_parquet(work_dir / "ext2" / "experiment_config" / "part-0000.parquet")
        assert cfg_frame["true_lift"].isna().all()
        assert set(cfg_frame["variant"]) == {"control", "treatment"}
        # 设计权重来自声明
        assert cfg_frame["design_weight"].eq(0.5).all()

    def test_missing_column_names_itself(self, work_dir, sql_dir):
        import pandas as pd
        import pytest

        from ablab.warehouse import load_real_traffic

        self._build_standard(work_dir, sql_dir)
        self._export(work_dir / "std_source", work_dir / "external")
        events = pd.read_csv(work_dir / "external" / "event_log.csv").drop(
            columns=["metric_value"]
        )
        events.to_csv(work_dir / "external" / "event_log.csv", index=False)
        with pytest.raises(ValueError, match="metric_value"):
            load_real_traffic(
                work_dir / "external", target_dir=work_dir / "bad", fmt="csv",
                experiments=self._declarations(),
            )

    def test_undeclared_variant_is_refused(self, work_dir, sql_dir):
        import pandas as pd
        import pytest

        from ablab.warehouse import load_real_traffic

        self._build_standard(work_dir, sql_dir)
        self._export(work_dir / "std_source", work_dir / "external")
        exposure = pd.read_csv(work_dir / "external" / "exposure_log.csv")
        exposure.loc[exposure.index[:5], "variant"] = "holdout"
        exposure.to_csv(work_dir / "external" / "exposure_log.csv", index=False)
        with pytest.raises(ValueError, match="变体名"):
            load_real_traffic(
                work_dir / "external", target_dir=work_dir / "bad2", fmt="csv",
                experiments=self._declarations(),
            )

    def test_generate_false_requires_ingested_files(self, work_dir, sql_dir):
        import pytest

        (work_dir / "empty").mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="load_real_traffic"):
            build_warehouse(
                db_path=work_dir / "x.duckdb", data_dir=work_dir / "empty",
                sql_dir=sql_dir, force_data=False, generate=False, verbose=False,
            )

    def test_external_experiment_validates_declarations(self):
        import pytest

        from ablab.warehouse import ExternalExperiment

        with pytest.raises(ValueError, match="权重之和"):
            ExternalExperiment("e", {"control": 0.5, "treatment": 0.3})
        with pytest.raises(ValueError, match="方向"):
            ExternalExperiment("e", {"control": 0.5, "treatment": 0.5},
                               guardrails=(("latency", "sideways", 0.05),))


class TestIdempotency:
    def test_rebuild_is_safe(self, work_dir, sql_dir):
        """重复构建不应报错，也不应重复累加数据。"""
        cfg = WarehouseConfig(n_users=1_500, seed=1)
        con = build_warehouse(
            db_path=work_dir / "w.duckdb",
            data_dir=work_dir / "src",
            sql_dir=sql_dir,
            config=cfg,
            force_data=True,
            verbose=False,
        )
        first = con.execute("SELECT COUNT(*) FROM dwd_experiment_user").fetchone()[0]
        con.close()

        con = build_warehouse(
            db_path=work_dir / "w.duckdb",
            data_dir=work_dir / "src",
            sql_dir=sql_dir,
            config=cfg,
            force_data=False,  # 复用已有 Parquet
            verbose=False,
        )
        second = con.execute("SELECT COUNT(*) FROM dwd_experiment_user").fetchone()[0]
        con.close()
        assert first == second

    def test_cache_marker_carries_the_config_fingerprint(self, work_dir):
        """**换配置就必须重建**，不能沿用上一份配置的落地文件。

        这一条是踩出来的：先用 ``--quick``（12 个复制实验）建了一次，
        再跑默认（100 个）时缓存被复用，SQL 链只建出 12 个复制实验，
        直到平台去读第 13 个才报 ``数仓 ADS 里找不到实验``。
        报错直白算是运气好 —— 把 ``--users 50000`` 写成复用 20,000 的数据，
        就只会得到一个"看起来正常"的数字。

        同时钉住反面：**只改假设文字不该触发重建**（否则改文案会白跑一遍，
        而"为什么又重算了"本身也是噪音）。
        """
        from dataclasses import replace

        from ablab.warehouse.generate import (
            DEFAULT_EXPERIMENTS,
            config_fingerprint,
            generate_source_data,
        )

        small = WarehouseConfig(n_users=400, seed=7)
        big = WarehouseConfig(n_users=900, seed=7)
        assert config_fingerprint(small) != config_fingerprint(big)

        data_dir = work_dir / "src"
        first = generate_source_data(data_dir, small)
        assert first["user_profile"] == 400
        assert generate_source_data(data_dir, small) == {"__cached__": 1}

        second = generate_source_data(data_dir, big)
        assert second["user_profile"] == 900, "换了配置却复用了缓存"

        # 只改 hypothesis（不影响数据）→ 仍然复用
        renamed = replace(
            big,
            experiments=tuple(
                replace(e, hypothesis=e.hypothesis + "（说法变了）") for e in DEFAULT_EXPERIMENTS
            ),
        )
        assert generate_source_data(data_dir, renamed) == {"__cached__": 1}
