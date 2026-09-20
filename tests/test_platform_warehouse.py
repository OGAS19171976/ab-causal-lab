"""平台 × 数仓的集成测试。

与 ``test_platform.py`` 的分工：那边测的是**平台自己**（注册表、编排、接口），
这边测的是**接上真实数仓之后**才出现的东西 —— 数据源等价性、绑定语义、结构迁移。

一个会话级的小数仓 fixture
-------------------------
``WarehouseConfig(n_users=3000)`` 走完整的 ODS→DWD→DWS→ADS 链路，
几秒钟就能建好。**刻意不用生产那份 20000 人的数仓** ——
测试要的是链路口径，不是样本量。
"""

import sqlite3

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ablab.inference import cluster_level_ttest
from ablab.platform import (
    ExperimentRegistry,
    RegistryError,
    analyse_data,
    analyse_experiment,
    run_source_equivalence_audit,
)
from ablab.platform.api import create_app
from ablab.platform.datasource import build_warehouse_data, list_warehouse_experiments
from ablab.platform.demo import WAREHOUSE_DEMO, seed_demo

#: 审计里的操作者（现在是必填的具名参数）。
ACTOR = "tester"

TWO_ARM = [
    {"name": "control", "weight": 0.5},
    {"name": "treatment", "weight": 0.5},
]
NINE_ONE = [
    {"name": "control", "weight": 0.9},
    {"name": "treatment", "weight": 0.1},
]


#: 写接口现在需要凭据。测试里统一用这个助手：建一个 admin 用户，
#: 把 token 挂到 client 上（`client.headers`），这样各测试的调用点不用逐个改。
def authed_client(app):
    """给 TestClient 装上 admin 凭据。返回 client 本身。"""
    token = app.state.registry.add_user("test_admin", role="admin")
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture(scope="session")
def warehouse_path(project_root):
    """建一个小数仓，返回**文件路径**（不在 fixture 里持有连接）。

    为什么不顺手把连接 yield 出去：DuckDB 在同一进程内**不允许对同一个文件
    混用不同配置的连接** —— 建仓的连接是读写，平台按请求开的是只读，
    两者共存会直接抛 ``Can't open a connection to same database file with
    a different configuration``。所以建完就关，把文件交出去，
    后续谁要读谁自己开只读连接。
    """
    from ablab.warehouse import WarehouseConfig, build_warehouse

    base = project_root / "build" / "_test_tmp" / "warehouse_fixture"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "wh.duckdb"
    con = build_warehouse(
        path,
        base / "source",
        project_root / "sql",
        config=WarehouseConfig(n_users=3000),
        force_data=True,
        verbose=False,
    )
    con.close()
    return path


@pytest.fixture
def warehouse_con(warehouse_path):
    """只读连接 —— 与平台按请求开的连接配置一致。"""
    con = duckdb.connect(str(warehouse_path), read_only=True)
    yield con
    con.close()


@pytest.fixture
def registry(work_dir):
    reg = ExperimentRegistry(work_dir / "reg.db")
    yield reg
    reg.close()


@pytest.fixture(scope="session")
def replicate_warehouse_path(project_root):
    """再建一条**含复制实验**的小数仓（8 个 A/A 复制实验）。

    与 ``warehouse_path`` 一样的规模（``n_users=3000``），这样"复制实验对
    已有数字零影响"这条声明可以直接把两张 ADS 表逐行比对 —— 声明要有测试。
    """
    from ablab.warehouse import (
        DEFAULT_EXPERIMENTS,
        WarehouseConfig,
        build_warehouse,
        ratio_replicate_experiments,
    )

    base = project_root / "build" / "_test_tmp" / "warehouse_replicates"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "wh_rep.duckdb"
    config = WarehouseConfig(
        n_users=3000,
        experiments=DEFAULT_EXPERIMENTS + ratio_replicate_experiments(8),
    )
    con = build_warehouse(
        path, base / "source", project_root / "sql", config=config,
        force_data=True, verbose=False,
    )
    con.close()
    return path


@pytest.fixture(scope="session")
def lifted_warehouse_path(project_root):
    """再建一条**只含带真实效应复制实验**的小数仓（6 个，每条互动 +2）。

    为什么单独一条库：它们会真的往 ``post_effect`` 里加东西，
    与默认演示实验混在一起会把已有数字改掉（README 里引用过的那些）。
    """
    from ablab.warehouse import (
        WarehouseConfig,
        build_warehouse,
        ratio_replicate_experiments_with_lift,
    )

    base = project_root / "build" / "_test_tmp" / "warehouse_lifted"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "wh_pow.duckdb"
    config = WarehouseConfig(
        n_users=4000, experiments=ratio_replicate_experiments_with_lift(6, lift=2.0)
    )
    con = build_warehouse(
        path, base / "source", project_root / "sql", config=config,
        force_data=True, verbose=False,
    )
    con.close()
    return path


class TestRatioLinkPowerCalibration:
    """真实效应下的校准：覆盖、功效、以及"SE 到底诚不诚实"。

    零效应那一半（``TestRatioReplicateCalibration``）回答 FWER 与 z 的方差；
    这一半回答功效与真实效应下的覆盖 —— 而它需要**真的往结果里加效应**。
    """

    def test_lift_replicates_are_mutually_exclusive(self, lifted_warehouse_path):
        """互斥是"真实效应恰好等于 lift"的前提：一个用户最多落进一个复制实验。

        若允许一个用户同时进两个，别的复制的效应会加进来 —— 它仍然是被平衡掉的
        噪声，但目标就不再**逐字**相等了，而"真值 = 注入的 lift"正是这一节的立身之本。
        """
        import duckdb

        con = duckdb.connect(str(lifted_warehouse_path), read_only=True)
        try:
            dup = con.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT user_id FROM ods_exposure_log
                    WHERE experiment LIKE 'exp_ratio_pow%'
                    GROUP BY user_id HAVING COUNT(DISTINCT experiment) > 1
                )
                """
            ).fetchone()[0]
            n_lift = con.execute(
                "SELECT COUNT(DISTINCT true_lift) FROM dim_experiment_config"
                " WHERE experiment LIKE 'exp_ratio_pow%'"
            ).fetchone()[0]
        finally:
            con.close()
        assert dup == 0, f"{dup} 个用户同时落进多个复制实验 —— 互斥性不成立"
        assert n_lift == 1, "这批复制实验的真实效应应当是同一个值"

    def test_power_calibration_runs_on_the_real_chain(self, lifted_warehouse_path):
        import duckdb

        from ablab.warehouse.ratio_calibration import run_ratio_link_power_calibration

        con = duckdb.connect(str(lifted_warehouse_path), read_only=True)
        try:
            res = run_ratio_link_power_calibration(
                con, n_replicates=6, true_lift=2.0, n_looks=5, alpha=0.05
            )
        finally:
            con.close()

        assert res.n_replicates == 6
        assert res.true_lift == 2.0
        for rate, interval in (
            (res.coverage, res.coverage_interval),
            (res.power, res.power_interval),
        ):
            assert 0.0 <= rate <= 1.0
            assert interval[0] <= rate <= interval[1]
        # SE 诚实性：点估计必须落在自己的区间里，且非中心度与真值/平均 SE 一致
        assert res.se_over_sd_interval[0] <= res.se_over_sd <= res.se_over_sd_interval[1]
        assert res.noncentrality == pytest.approx(res.true_lift / res.se_mean, rel=1e-9)
        assert res.se_mean > 0 and res.effect_sd > 0
        assert res.sequential_power >= 0.0

    def test_rerandomization_is_deterministic_and_unbiased(self, lifted_warehouse_path):
        """两个断言，各自钉一件事。

        1. **可复算**：同一份数据跑两次必须逐位相同。第一版没写 ``ORDER BY``，
           DuckDB 并行扫描的行序一变、同一个 seed 抽到的用户组合就变了 ——
           实测 SE/sd 在 1.01 ~ 1.06 之间跳，而汇总量一位不差。这类"数字会漂"
           正是 README 第 42 条要防的东西。
        2. **无偏 + 真值逐字相等**：``Y_i(1) = Y_i(0) + lift·X_i`` ⇒ 全处置 vs
           全对照的真实效应**恰好**是 lift；重抽分布的均值应当在蒙特卡洛误差内。
        """
        import duckdb

        from ablab.warehouse.ratio_calibration import rerandomization_reference

        con = duckdb.connect(str(lifted_warehouse_path), read_only=True)
        try:
            a = rerandomization_reference(
                con, experiment="exp_ratio_pow000", true_lift=2.0, n_splits=200
            )
            b = rerandomization_reference(
                con, experiment="exp_ratio_pow000", true_lift=2.0, n_splits=200
            )
        finally:
            con.close()

        assert (a.estimate_mean, a.sd, a.se_mean) == (b.estimate_mean, b.sd, b.se_mean), (
            "重随机化不可复算 —— 先检查取数有没有 ORDER BY"
        )
        assert a.truth == pytest.approx(2.0, abs=1e-9), a.truth
        # 偏差应当在重抽的蒙特卡洛误差内（4 倍标准误的宽带，避免假红灯）
        mc_se = a.sd / np.sqrt(a.n_splits)
        assert abs(a.bias) <= 4 * mc_se, (a.bias, mc_se)


@pytest.fixture(scope="session")
def cluster_replicate_warehouse_path(project_root):
    """再建一条**整簇随机化复制实验**的小数仓（6 个 A/A，60 个城市）。"""
    from ablab.warehouse import (
        DEFAULT_EXPERIMENTS,
        WarehouseConfig,
        build_warehouse,
        cluster_replicate_experiments,
    )

    base = project_root / "build" / "_test_tmp" / "warehouse_cluster_reps"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "wh_clu.duckdb"
    config = WarehouseConfig(
        n_users=3000,
        experiments=DEFAULT_EXPERIMENTS + cluster_replicate_experiments(6),
    )
    con = build_warehouse(
        path, base / "source", project_root / "sql", config=config,
        force_data=True, verbose=False,
    )
    con.close()
    return path


class TestClusterReplicateCalibration:
    """数仓路径上的**簇级** A/A 校准：整簇随机化 × 多个 salt。

    这一组钉的是"M6 只报了方差缩减、不声称覆盖率"那条边界被补上的证据：
    簇级推断走的是另一条数据链路（ADS 臂级 + DWS 簇级），
    合成路径上对，推不出数仓路径上也对。
    """

    def test_replicate_definitions_are_cluster_randomized(self):
        from ablab.warehouse import cluster_replicate_experiments

        reps = cluster_replicate_experiments(6)
        assert len({e.name for e in reps}) == 6
        assert len({e.layer for e in reps}) == 6, "每个复制实验必须独占一层"
        assert all(e.cluster_key == "city" for e in reps), "必须是整簇随机化"
        assert all(e.true_lift == 0.0 for e in reps), "A/A：真实效应必须是 0"
        assert all(e.bucket_end == 10_000 for e in reps), "取满桶位才有 60 个簇"

    def test_every_replicate_keeps_all_clusters(self, cluster_replicate_warehouse_path):
        """每个复制实验都要拿到全部 60 个簇 —— 少了就不叫"够多簇"的重复。"""
        import duckdb

        con = duckdb.connect(str(cluster_replicate_warehouse_path), read_only=True)
        try:
            frame = con.execute(
                "SELECT experiment, COUNT(DISTINCT city) AS n_clusters"
                " FROM dwd_experiment_user WHERE experiment LIKE 'exp_cluster_rep%'"
                " GROUP BY experiment"
            ).df()
        finally:
            con.close()
        assert len(frame) == 6
        assert (frame["n_clusters"] == 60).all(), frame.to_dict("records")

    def test_calibration_on_the_real_chain(self, cluster_replicate_warehouse_path):
        import duckdb

        from ablab.warehouse.cluster_calibration import run_cluster_replicate_calibration

        con = duckdb.connect(str(cluster_replicate_warehouse_path), read_only=True)
        try:
            res = run_cluster_replicate_calibration(con, n_replicates=6, estimator="cuped")
        finally:
            con.close()

        assert res.n_replicates == 6
        assert 0.0 <= res.fpr <= 1.0
        assert res.fpr_interval[0] <= res.fpr <= res.fpr_interval[1]
        assert res.n_clusters_min == 60
        assert len(res.final_z) == 6
        assert res.z_sd > 0
        # 簇级 A/A 的误停率必须**不高于**名义值太多（Wilson 上界是宽区间，
        # 6 个复制实验量不出精确的 size —— 所以这里只钉"没有明显崩"）
        assert res.fpr_interval[1] < 0.6, res.fpr_interval

    def test_degenerate_input_is_refused(self, cluster_replicate_warehouse_path):
        import duckdb
        import pytest as _pytest

        from ablab.warehouse.cluster_calibration import run_cluster_replicate_calibration

        con = duckdb.connect(str(cluster_replicate_warehouse_path), read_only=True)
        try:
            with _pytest.raises(ValueError, match="至少"):
                run_cluster_replicate_calibration(con, n_replicates=3)
        finally:
            con.close()


class TestRatioReplicateCalibration:
    """比值链路的**序贯校准**：重复实现才是校准，一致性不是。

    6 节那条链路只被证明过"两份实现算得一样"。这一组钉住补上的那一半：
    100 个（测试里 8 个）真实效应为 0 的复制实验，走完整条真实链路，
    然后数 FWER / 覆盖 / z 的分布。
    """

    @staticmethod
    def _replicates(n: int = 8):
        from ablab.warehouse import ratio_replicate_experiments

        return ratio_replicate_experiments(n)

    def test_replicate_definitions_are_orthogonal_and_null(self):
        """复制实验的定义：真实效应为 0、各自独占一层、层与 salt 都不重名。"""
        reps = self._replicates(8)
        assert len({e.name for e in reps}) == 8
        assert len({e.layer for e in reps}) == 8, "每个复制实验必须独占一层"
        assert len({e.layer_salt for e in reps}) == 8
        assert all(e.true_lift == 0.0 for e in reps), "A/A 复制实验的真实效应必须是 0"
        assert all(0 <= e.bucket_start < e.bucket_end <= 10_000 for e in reps)

    def test_replicates_do_not_move_existing_numbers(self, warehouse_path, replicate_warehouse_path):
        """**零影响声明要有测试**：加了 8 个复制实验，已有实验的 ADS 一行都不能变。

        这条声明有两个支柱（``true_lift=0`` 与"每个复制实验独占一层"），
        任何一个塌了都会改掉 README 里引用过的数仓数字。所以这里直接
        逐行比对两边的 ADS 结果表。
        """
        import duckdb

        def ads(path):
            con = duckdb.connect(str(path), read_only=True)
            try:
                return con.execute(
                    "SELECT * FROM ads_experiment_result ORDER BY experiment, variant"
                ).df()
            finally:
                con.close()

        base = ads(warehouse_path)
        with_rep = ads(replicate_warehouse_path)
        with_rep = with_rep[~with_rep["experiment"].str.startswith("exp_ratio_rep")]
        assert list(base.columns) == list(with_rep.columns)
        base = base.reset_index(drop=True)
        with_rep = with_rep.reset_index(drop=True)

        # 结构列与计数必须**逐字**相同：它们变了一点，就说明复制实验真的动了别人。
        for col in ("experiment", "variant", "layer", "hypothesis"):
            assert (base[col].astype(str) == with_rep[col].astype(str)).all(), col
        for col in ("user_cnt", "true_lift", "design_weight"):
            assert np.array_equal(
                base[col].to_numpy(dtype=float), with_rep[col].to_numpy(dtype=float)
            ), f"{col} 变了 —— 复制实验影响了已有实验"

        # 可加量的**数值**允许末位差：两份库的表的行数不同，
        # DuckDB 的并行聚合（threads=4）合并顺序随之不同，而浮点加法不满足结合律。
        # 这是已知现象（见 build.py 里 threads 那段注释），实测最大相对差 ~5.5e-15。
        # 卡在 1e-12 上：真出了问题（比如复制实验的曝光混进了别人）差的是**量级**，
        # 不是末位。**不允许**因此放宽结构列与计数那一半。
        for col in ("pre_sum", "post_sum", "pre_sq_sum", "post_sq_sum",
                    "pre_post_cross_sum", "post_mean", "pre_mean", "post_var",
                    "pre_var", "pre_post_cov"):
            a = base[col].to_numpy(dtype=float)
            b = with_rep[col].to_numpy(dtype=float)
            rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-12)
            assert rel.max() < 1e-12, f"{col} 相对差 {rel.max():.3e} 超过浮点噪声量级"

    def test_calibration_on_the_real_chain(self, replicate_warehouse_path):
        """走完整条真实链路跑 8 个 A/A 复制实验，数三个承诺。"""
        import duckdb

        from ablab.warehouse.ratio_calibration import run_ratio_link_calibration

        con = duckdb.connect(str(replicate_warehouse_path), read_only=True)
        try:
            res = run_ratio_link_calibration(con, n_replicates=8, n_looks=5, alpha=0.05)
        finally:
            con.close()

        assert res.n_replicates == 8
        assert 0.0 <= res.fwer <= 1.0
        assert res.fwer_interval[0] <= res.fwer <= res.fwer_interval[1]
        assert res.coverage == pytest.approx(1.0 - res.final_rate)
        assert len(res.final_z) == 8 and len(res.per_look_rate) == 5

        # 信息分数是累计信息量之比：必须单调、末次为 1
        info = res.information_fractions
        assert info[-1] == pytest.approx(1.0)
        assert all(b > a for a, b in zip(info, info[1:])), info

        # **M6.1 的不变量**：末次查看 = 主结论，逐位相同，一个复制实验都不能漏
        assert res.last_look_matches_primary == 1.0

        # z 的诊断要有意义（8 个点量不出 1.0，但必须有限且不是常数）
        assert 0.0 < res.z_sd_final < 5.0
        assert abs(res.salt_agreement - 0.5) < 0.05, res.salt_agreement
        assert res.salt_pairs == 28, "8 个复制实验应有 C(8,2)=28 对"
        assert res.arm_share_chi2 > 0

    def test_calibration_rejects_degenerate_inputs(self, replicate_warehouse_path):
        import duckdb
        import pytest as _pytest

        from ablab.warehouse.ratio_calibration import run_ratio_link_calibration

        con = duckdb.connect(str(replicate_warehouse_path), read_only=True)
        try:
            with _pytest.raises(ValueError, match="至少"):
                run_ratio_link_calibration(con, n_replicates=1)
        finally:
            con.close()

    def test_replicate_prefix_is_what_the_loader_looks_for(self, replicate_warehouse_path):
        """校准按前缀认人 —— 前缀写错就会**静默**只看一个实验。"""
        import duckdb

        from ablab.warehouse import RATIO_REPLICATE_PREFIX

        con = duckdb.connect(str(replicate_warehouse_path), read_only=True)
        try:
            n = con.execute(
                "SELECT COUNT(DISTINCT experiment) FROM ads_experiment_result"
                " WHERE experiment LIKE ?",
                [f"{RATIO_REPLICATE_PREFIX}%"],
            ).fetchone()[0]
        finally:
            con.close()
        assert n == 8


# --------------------------------------------------------------------------- #
# 数据源层：合成路径
# --------------------------------------------------------------------------- #
class TestSyntheticSource:
    def test_last_look_equals_the_main_analysis(self, registry):
        """**最重要的一条不变量**：监控曲线的右端点必须与主结论逐位相同。

        这是 M6.1 的核心：判定的估计量必须与头条结论是同一个。
        默认口径是 CUPED，所以最后一次查看要等于 ``primary``（CUPED），
        而 ``alt``（post-only）是另一个口径、不该被拿来比。
        """
        rec = registry.create(actor=ACTOR, name="ds1", variants=TWO_ARM, true_lift=0.3)
        rep = analyse_experiment(rec, n_users=6000, seed=1)
        last = rep.monitoring[-1]
        assert rep.primary_estimator_name == "cuped"
        assert last["estimator"] == "cuped"
        assert last["effect"] == rep.primary.absolute_effect
        assert last["std_error"] == rep.primary.std_error
        assert last["information_fraction"] == 1.0
        # 对照口径是另一回事，绝不能顺手相等
        assert last["alt_estimator"] == "post_only"
        assert last["alt_effect"] == rep.alt.absolute_effect

    def test_last_look_is_full_even_for_unequal_arms(self, registry):
        """90/10 分支下最容易写错的地方。

        如果按"两臂都取 min(n_t, n_c) 的前缀"来构造查看，小臂永远取不到全量，
        最后一次查看就不等于全量分析了。正确做法是**每臂内部**各自按比例取前缀。
        """
        rec = registry.create(actor=ACTOR, name="ds2", variants=NINE_ONE)
        rep = analyse_experiment(rec, n_users=8000, seed=2)
        last = rep.monitoring[-1]
        assert last["n_treatment"] == rep.primary.n_treatment
        assert last["n_control"] == rep.primary.n_control
        assert last["std_error"] == rep.primary.std_error

    def test_information_fractions_are_uniform(self, registry):
        rec = registry.create(actor=ACTOR, name="ds3", variants=TWO_ARM)
        rep = analyse_experiment(rec, n_users=4000, n_looks=8, seed=3)
        fracs = [m["information_fraction"] for m in rep.monitoring]
        assert fracs == [round(i / 8, 10) for i in range(1, 9)]

    def test_n_per_arm_is_monotone(self, registry):
        rec = registry.create(actor=ACTOR, name="ds4", variants=NINE_ONE)
        rep = analyse_experiment(rec, n_users=6000, n_looks=5, seed=4)
        sizes = [m["n_per_arm"] for m in rep.monitoring]
        assert sizes == sorted(sizes)
        assert all(s >= 2 for s in sizes)

    def test_source_is_marked(self, registry):
        rec = registry.create(actor=ACTOR, name="ds5", variants=TWO_ARM)
        rep = analyse_experiment(rec, n_users=2000, seed=5)
        assert rep.source == "synthetic"
        assert rep.population_size == 2000


# --------------------------------------------------------------------------- #
# 注册表：绑定语义与结构迁移
# --------------------------------------------------------------------------- #
class TestBinding:
    def test_create_with_binding(self, registry):
        rec = registry.create(actor=ACTOR, name="b1", variants=TWO_ARM, warehouse_experiment="exp_rank_v2")
        assert rec.warehouse_experiment == "exp_rank_v2"
        assert registry.get(rec.id).warehouse_experiment == "exp_rank_v2"

    def test_bind_and_unbind(self, registry):
        rec = registry.create(actor=ACTOR, name="b2", variants=TWO_ARM)
        assert rec.warehouse_experiment is None
        assert registry.bind_warehouse(rec.id, "  exp_rec_emb  ", actor=ACTOR).warehouse_experiment == "exp_rec_emb"
        assert registry.bind_warehouse(rec.id, None, actor=ACTOR).warehouse_experiment is None

    def test_blank_binding_rejected(self, registry):
        rec = registry.create(actor=ACTOR, name="b3", variants=TWO_ARM)
        with pytest.raises(RegistryError, match="空白"):
            registry.bind_warehouse(rec.id, "   ", actor=ACTOR)
        with pytest.raises(RegistryError, match="空白"):
            registry.create(actor=ACTOR, name="b3b", variants=TWO_ARM, warehouse_experiment="  ")

    def test_binding_is_mutable_but_salt_is_not(self, registry):
        """绑定可变、salt 不可变 —— 两者的区别必须清晰。

        salt 决定了每个用户的分组，改它等于让已有数据报废；
        ``warehouse_experiment`` 只决定"从哪里读数"，不影响任何分组。
        """
        rec = registry.create(actor=ACTOR, name="b4", variants=TWO_ARM)
        salt_before = rec.salt
        registry.bind_warehouse(rec.id, "exp_rank_v2", actor=ACTOR)
        registry.bind_warehouse(rec.id, "exp_rec_emb", actor=ACTOR)
        after = registry.get(rec.id)
        assert after.salt == salt_before, "绑定不该碰到 salt"
        assert after.warehouse_experiment == "exp_rec_emb"

    def test_migration_adds_column_to_old_db(self, work_dir):
        """老库（没有 warehouse_experiment 列）必须能被自动补齐。

        ``CREATE TABLE IF NOT EXISTS`` 对已存在的表什么都不做，
        缺了迁移就会在第一次写入时报 no such column。
        """
        legacy = work_dir / "legacy.db"
        con = sqlite3.connect(legacy)
        con.execute(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                hypothesis TEXT NOT NULL DEFAULT '', owner TEXT NOT NULL DEFAULT '',
                layer TEXT, unit TEXT NOT NULL DEFAULT 'user_id', salt TEXT NOT NULL,
                traffic_ratio REAL NOT NULL DEFAULT 1.0, variants TEXT NOT NULL,
                primary_metric TEXT NOT NULL DEFAULT 'metric',
                guardrails TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'draft', start_ds TEXT, end_ds TEXT,
                true_lift REAL NOT NULL DEFAULT 0.0, created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO experiments (id,name,salt,variants,created_at) "
            "VALUES ('old1','legacy','legacy_v1','[]','2026-01-01T00:00:00+00:00')"
        )
        con.commit()
        con.close()

        reg = ExperimentRegistry(legacy)
        try:
            # 老行读得出来，新列取默认值
            assert reg.get("old1").warehouse_experiment is None
            # 新行写得进去
            rec = reg.create(actor=ACTOR, name="new_after_migration", variants=TWO_ARM)
            assert rec.warehouse_experiment is None
            assert reg.bind_warehouse(rec.id, "exp_rank_v2", actor=ACTOR).warehouse_experiment == "exp_rank_v2"
        finally:
            reg.close()


# --------------------------------------------------------------------------- #
# 数仓路径
# --------------------------------------------------------------------------- #
class TestWarehouseSource:
    def test_lists_bindable_experiments(self, warehouse_con):
        items = list_warehouse_experiments(warehouse_con)
        names = {i["experiment"] for i in items}
        assert {"exp_rank_v2", "exp_rec_emb"} <= names
        for i in items:
            assert i["n_variants"] == 2 and i["n_users"] > 0

    def test_looks_are_daily_cumulative(self, warehouse_con):
        data = build_warehouse_data(warehouse_con, "exp_rank_v2", n_looks=5)
        assert data.source == "warehouse"
        fracs = [lk.information_fraction for lk in data.looks]
        assert fracs == sorted(fracs) and fracs[-1] == 1.0
        # 数仓路径的查看标签是**日期**，不是 "look k"
        assert all("（累计）" in lk.label for lk in data.looks)
        # 累计样本量单调增
        sizes = [lk.n_per_arm for lk in data.looks]
        assert sizes == sorted(sizes)

    def test_last_look_equals_total(self, warehouse_con):
        data = build_warehouse_data(warehouse_con, "exp_rec_emb", n_looks=5)
        assert data.total.treatment.n == data.counts["treatment"]
        assert data.total.control.n == data.counts["control"]

    def test_information_fractions_are_actual_not_calendar(self, warehouse_con):
        """信息比例必须取自实际累计样本量，不是日历天数。

        每天进入实验的人数并不相等，用天数比会高估早期信息量、让早期边界偏松。
        """
        data = build_warehouse_data(warehouse_con, "exp_rank_v2", n_looks=5)
        fracs = [lk.information_fraction for lk in data.looks]
        # 与等距目标有偏差（这就是"实际"的含义），但偏差不会大到离谱
        assert any(abs(f - i / 5) > 1e-6 for i, f in enumerate(fracs, start=1))
        assert all(abs(f - i / 5) < 0.12 for i, f in enumerate(fracs, start=1))

    def test_unknown_experiment_raises(self, warehouse_con):
        with pytest.raises(ValueError, match="找不到"):
            build_warehouse_data(warehouse_con, "no_such_experiment")

    def test_three_paths_agree(self, warehouse_con):
        """**头条证据**：三条读取路径给同一个答案。

        1. DWD 明细（``welch_ttest`` / ``cuped_ttest``）
        2. ADS 汇总（``ablab.warehouse.analyse_ads`` —— 平台存在之前就写好的独立实现）
        3. 平台编排（``analyse_experiment_from_warehouse``）

        1 和 2 是 M0 写的，拿它们当参照才有意义。
        """
        for experiment in ("exp_rank_v2", "exp_rec_emb"):
            r = run_source_equivalence_audit(warehouse_con, experiment, n_looks=5)
            assert r.agree, f"{experiment} 三条路径不一致，最大偏差 {r.max_deviation:.3e}"
            assert r.max_deviation < 1e-9
            assert r.last_look_matches
            # 换算成"相对偏差"看更直观：效应量级远大于 1e-9
            assert abs(r.from_platform[0]) > 1.0


# --------------------------------------------------------------------------- #
# 数仓路径：HTTP 接口
# --------------------------------------------------------------------------- #
@pytest.fixture
def wh_client(work_dir, warehouse_path):
    app = create_app(work_dir / "wh_api.db", warehouse_path=warehouse_path)
    seed_demo(app.state.registry, warehouse_available=True)
    with authed_client(app) as c:
        yield c
    app.state.registry.close()


@pytest.fixture
def plain_client(work_dir):
    """没配数仓的平台 —— 必须优雅退化，不是崩掉。"""
    app = create_app(work_dir / "plain_api.db")
    seed_demo(app.state.registry)
    with authed_client(app) as c:
        yield c
    app.state.registry.close()


# --------------------------------------------------------------------------- #
# 分析单元：数仓侧
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def cluster_warehouse_path(project_root):
    """一个**真正整簇随机化**的小数仓（分流在城市级别做）。

    为什么不把它加进默认的 ``DEFAULT_EXPERIMENTS``：
    ① 那会改变 ODS 曝光行数，而 README 引用了那个数字 —— 改掉真数字要连带改文档，
       而且很容易忘；
    ② 默认那份数仓是**人级随机化**的，它的价值恰恰在于"平台会正确拒绝把它当簇级做"。
       两份数据各司其职，比混在一起清楚。

    城市数提到 40（默认只有 5 个）：簇太少时"每臂至少 2 个簇"都凑不齐，
    簇级检验根本算不出来 —— 那测的就不是链路，是运气。
    """
    from ablab.warehouse import WarehouseConfig, build_warehouse
    from ablab.warehouse.generate import ExperimentDef

    base = project_root / "build" / "_test_tmp" / "cluster_warehouse"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "wh.duckdb"
    cluster_exp = ExperimentDef(
        name="exp_city_rollout",
        layer="city_rollout",
        layer_salt="layer_city_rollout",
        bucket_start=0,
        bucket_end=8000,  # 80% 的城市进入实验
        true_lift=0.0,    # 真效应为零：不影响共享的事件数据，也就不动已有数字
        hypothesis="【整簇随机化】城市级灰度，预期零效应",
        cluster_key="city",
    )
    con = build_warehouse(
        path,
        base / "source",
        project_root / "sql",
        config=WarehouseConfig(
            n_users=2000,
            cities=tuple(f"city{i:02d}" for i in range(40)),
            experiments=(cluster_exp,),
        ),
        force_data=True,
        verbose=False,
    )
    con.close()
    return path


@pytest.fixture
def cluster_con(cluster_warehouse_path):
    con = duckdb.connect(str(cluster_warehouse_path), read_only=True)
    yield con
    con.close()


class TestWarehouseAnalysisUnit:
    def test_mixed_clusters_are_refused(self, warehouse_con):
        """**误用必须被拒绝，而不是给出一个看起来正常的数。**

        默认那份数仓是人级随机化的，用城市当簇去做簇级分析会算出
        ``效应 +27.27、SE 3.38、p=5.5e-5`` —— 一个完全正常的外观。
        而它答的是另一个问题：簇级检验假设处理在**簇**级别分配。

        M1 写的独立实现 ``cluster_level_ttest`` 会直接拒绝这种输入；
        平台侧也必须拒绝，否则"能算"就会被当成"该算"。
        """
        for experiment in ("exp_rank_v2", "exp_rec_emb"):
            with pytest.raises(ValueError, match="不是整簇随机化"):
                build_warehouse_data(warehouse_con, experiment, analysis_unit="cluster")

    def test_cluster_randomized_experiment_is_accepted(self, cluster_con):
        # 口径按**声明**走：这里是"没显式声明"的情况，于是用默认的 cuped。
        # （这条断言原先写死 post_only —— 那是硬编码时代的产物。）
        data = build_warehouse_data(cluster_con, "exp_city_rollout", analysis_unit="cluster")
        assert data.analysis_unit == "cluster"
        assert data.primary_estimator == "cuped"
        assert data.total.has_clusters
        assert data.total.clusters_consistent()
        # SRM 的对象是随机化单元 —— 簇数，不是用户数
        assert sum(data.counts.values()) < 0.1 * data.extra["n_users"]
        assert data.counts["treatment"] >= 2 and data.counts["control"] >= 2

    def test_every_cluster_is_wholly_in_one_arm(self, cluster_con):
        """整簇随机化的定义就是这一条：簇内不混臂。"""
        rows = cluster_con.execute(
            """
            SELECT cluster_id, COUNT(DISTINCT variant) AS n_variants
            FROM dws_experiment_cluster_daily
            WHERE experiment = 'exp_city_rollout'
            GROUP BY cluster_id
            """
        ).df()
        assert (rows["n_variants"] == 1).all(), "有簇内部混臂，那就不是整簇随机化"

    def test_cluster_level_matches_the_independent_implementation(self, cluster_con):
        """平台簇级结论 vs M1 的 ``cluster_level_ttest``（吃 DWD 明细）。"""
        # 这条测的是**簇级 Welch** 那条路径，所以显式声明 post_only ——
        # 否则会走默认的簇级 CUPED，答的就是另一个问题了。
        data = build_warehouse_data(
            cluster_con, "exp_city_rollout", analysis_unit="cluster",
            primary_estimator="post_only",
        )
        rep = analyse_data(data)
        assert rep.primary_estimator_name == "cluster_level"
        assert rep.alt_estimator_name == "unit_level"

        detail = cluster_con.execute(
            """
            SELECT d.variant, d.post_metric, p.city
            FROM dwd_experiment_user d
            LEFT JOIN ods_user_profile p ON p.user_id = d.user_id
            WHERE d.experiment = 'exp_city_rollout'
            """
        ).df()
        ref = cluster_level_ttest(
            detail["city"].fillna("UNKNOWN").to_numpy(),
            (detail["variant"] == "treatment").to_numpy(),
            detail["post_metric"].to_numpy(),
        )
        assert rep.primary.absolute_effect == pytest.approx(ref.absolute_effect, abs=1e-9)
        assert rep.primary.std_error == pytest.approx(ref.std_error, abs=1e-9)
        assert rep.primary.n_treatment == ref.n_treatment
        assert rep.primary.n_control == ref.n_control

    def test_clusters_barely_accrue_so_monitoring_shrinks(self, cluster_con):
        """簇不分批进入时，簇级序贯监控**几乎没有可用的中间查看点**。

        这份 DGP 里几乎所有城市第一天就有用户进来，于是"累计簇数之比"从第一个
        可用的查看点起就已经接近 1.0 —— 请求 5 次查看只能给出 1~2 个点。
        正确行为是**照实减少并在报告里说清**，而不是硬凑出几个信息比例相同的假查看点
        （那样 BoundarySolver 会拿到非递增的网格，给出无意义的边界）。

        真实的簇级序贯监控需要"簇分批上线"（城市/门店分批开城），这份数据不建模这件事 ——
        这是个数据边界，不是实现缺陷。
        """
        data = build_warehouse_data(cluster_con, "exp_city_rollout", analysis_unit="cluster")
        assert data.n_looks < 5, "请求了 5 次查看，但簇不累积时应给出更少的点"
        assert data.total.information_fraction == 1.0
        # 第一个可用点就已经接近全量 —— 这就是"簇几乎不随时间累积"的量化说法
        assert data.looks[0].information_fraction > 0.8
        assert "可用查看点" in data.extra["look_note"]
        assert "请求 5 次" in data.extra["look_note"]
        rep = analyse_data(data)
        msg = next(c for c in rep.checks if c.name == "序贯监控")
        assert "可用查看点" in msg.message
        # 每一次查看都必须是合法的簇级输入
        assert all(lk.n_clusters[0] >= 2 and lk.n_clusters[1] >= 2 for lk in data.looks)

    def test_api_returns_400_for_cluster_misuse(self, wh_client):
        """接口层也要把误用挡在 400，而不是 200 给一份错口径的报告。"""
        created = wh_client.post("/api/experiments", json={
            "name": "wh_cluster_misuse",
            "variants": [{"name": "control", "weight": 0.5},
                         {"name": "treatment", "weight": 0.5}],
            "analysis_unit": "cluster",
            "estimator": "post_only",
            "warehouse_experiment": "exp_rank_v2",
        })
        assert created.status_code == 201
        r = wh_client.post(f"/api/experiments/{created.json()['id']}/analyze",
                           json={"n_looks": 5})
        assert r.status_code == 400
        assert "不是整簇随机化" in r.json()["detail"]


class TestClusterCuped:
    """簇级 CUPED：打开它，并且**不静默忽略声明**。

    这一块原先有两处**过时假设**写死在代码里：
      * 创建时拒绝 `analysis_unit=cluster` + `estimator=cuped`（理由是"数据源没有簇级前置指标"）；
      * `ExperimentData.validate()` 里同样的拒绝；
      * 以及分析层 `_headline_path` 把簇级路径写死成 post-only ——
        于是就算前两处放行了，声明也会被**静默忽略**（报告里连 CUPED 那一项都没有）。
    而 05 路 DWS 一直落着簇级的 pre_sum / pre_sq_sum / pre_post_cross_sum
    （当时的注释就写着"将来做簇级 CUPED 时不必改这一层"），实测 pre_sum ≈ 4.4e6。
    """

    @staticmethod
    def _cluster_record(estimator: str = "cuped"):
        from ablab.platform.registry import ExperimentRecord

        return ExperimentRecord(
            name="exp_city_ctr",
            variants=list(TWO_ARM),
            salt="exp_city_ctr_v1",
            primary_metric="post_metric_14d",
            warehouse_experiment="exp_city_ctr",
            analysis_unit="cluster",
            estimator=estimator,
        )

    def test_cluster_randomized_experiment_exists_in_the_warehouse(self, warehouse_con):
        """数仓里必须**真有**一个簇随机化实验，否则这条链路在演示里走不到。

        实测（24 个城市）：处理组 9 个簇、对照组 15 个簇 —— 每臂 ≥ 2 个簇，
        簇级方差的自由度（簇数 − 2）才有定义。
        """
        rows = dict(
            warehouse_con.execute(
                """
                SELECT variant, COUNT(DISTINCT cluster_id)
                FROM dws_experiment_cluster_daily
                WHERE experiment = 'exp_city_ctr'
                GROUP BY variant
                """
            ).fetchall()
        )
        assert rows, "没有簇随机化实验的簇表 —— 簇级链路在演示里走不到"
        assert rows["control"] >= 2 and rows["treatment"] >= 2, rows

    def test_unit_randomized_experiment_still_refuses_cluster_analysis(
        self, warehouse_con
    ):
        """闸门没被放松：人级随机化的实验按城市分析必须**被拒**。

        簇级检验假设"处理在簇级别分配"。同一个城市里既有对照又有处置时，
        按城市算的簇级方差量到的不是随机化带来的变异 —— 实测把 I 类错误率
        从 6% 抬到 69%。这条闸门是 M1 留下的，不能被这一轮的放行顺手拆掉。
        """
        import pytest

        from ablab.platform.analysis import analyse_experiment_from_warehouse
        from ablab.platform.registry import ExperimentRecord

        rec = ExperimentRecord(
            name="wh_exp_rank_v2",
            variants=list(TWO_ARM),
            salt="wh_exp_rank_v2_v1",
            primary_metric="post_metric_14d",
            warehouse_experiment="exp_rank_v2",  # 人级随机化
            analysis_unit="cluster",
            estimator="post_only",
        )
        with pytest.raises(ValueError, match="簇"):
            analyse_experiment_from_warehouse(rec, warehouse_con)

    def test_cluster_cuped_reduces_variance_and_keeps_cluster_units(
        self, warehouse_con
    ):
        """簇级 CUPED 真的在起作用：方差缩减 > 0，且观测单位仍是**簇**。

        实测：方差缩减 ~31%，标准误降 ~17%（簇级前置指标与后置指标的相关
        在簇之间是真实存在的）。自由度按"簇数 − 2"，不是"用户数 − 2"。
        """
        from ablab.platform.analysis import analyse_experiment_from_warehouse

        rec = self._cluster_record("cuped")
        report = analyse_experiment_from_warehouse(rec, warehouse_con)
        assert report.cuped is not None and report.cuped_fit is not None
        assert report.cuped_fit.variance_reduction > 0.05, report.cuped_fit
        cuped_item = next(c for c in report.checks if c.name == "CUPED 收益")
        assert cuped_item.status == "pass"
        # 主口径就是簇级 CUPED（不是被静默忽略）
        assert report.primary_estimator_name == "cluster_cuped", (
            report.primary_estimator_name
        )
        # 观测单位是簇：n_treatment 应当等于簇数，而不是用户数
        n_clusters_t = warehouse_con.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM dws_experiment_cluster_daily "
            "WHERE experiment = 'exp_city_ctr' AND variant = 'treatment'"
        ).fetchone()[0]
        assert report.cuped.n_treatment == int(n_clusters_t)

    def test_post_only_cluster_path_still_works_and_reports_unit_level_contrast(
        self, warehouse_con
    ):
        """post-only 那条路径不受影响，而且**照旧报单元级对照**。

        那个对照是 M1 里 I 类错误率 69% 的错误做法，摆在旁边用来说明
        "分析单元必须与随机化单元对齐"；放行 CUPED 不该把它弄丢。
        """
        from ablab.platform.analysis import analyse_experiment_from_warehouse

        rec = self._cluster_record("post_only")
        report = analyse_experiment_from_warehouse(rec, warehouse_con)
        item = next(c for c in report.checks if c.name == "分析单元")
        assert item.status == "info"
        assert "单元级" in item.message, item.message

    def test_cluster_cuped_without_pre_metric_raises_instead_of_falling_back(self):
        """拿不到簇级前置指标时**报错**，而不是静默退回 post-only。

        "声明了 CUPED 却按 post-only 出结论"正是 M6 那条老毛病；
        这里用一份人为把 sum_x 清零的簇级统计量来验证它。
        """
        import numpy as np
        import pytest

        from ablab.inference.aggregates import AggregateStats
        from ablab.platform.datasource import LookData

        def arm() -> tuple[AggregateStats, ...]:
            return tuple(
                AggregateStats(n=100, sum_x=0.0, sum_y=float(i + 1) * 100.0)
                for i in range(4)
            )

        look = LookData(
            label="look 1",
            information_fraction=1.0,
            treatment=AggregateStats(n=400, sum_y=600.0),
            control=AggregateStats(n=400, sum_y=600.0),
            cluster_treatment=arm(),
            cluster_control=arm(),
        )
        assert not look.clusters_have_pre_metric
        with pytest.raises(ValueError, match="前置指标"):
            look.cluster_cuped()
        # 有前置指标时同一份数据就能算
        with_pre = tuple(
            AggregateStats(n=100, sum_x=float(i + 1) * 10.0, sum_y=float(i + 1) * 12.0)
            for i in range(4)
        )
        ok = LookData(
            label="look 1",
            information_fraction=1.0,
            treatment=AggregateStats(n=400, sum_y=600.0),
            control=AggregateStats(n=400, sum_y=600.0),
            cluster_treatment=with_pre,
            cluster_control=with_pre,
        )
        assert ok.clusters_have_pre_metric
        estimate, _fit = ok.cluster_cuped()
        assert np.isfinite(estimate.absolute_effect)


class TestWarehouseGuardrails:
    """数仓护栏链路（08 DWS -> 09 ADS -> 判定）：**接上了，而且没有污染主指标**。

    这一组里最重要的一条不是"护栏能读出来"，而是
    ``test_guardrail_events_do_not_pollute_the_main_metric``：
    护栏事件与主指标共用一张 ODS 事件表（长表），而 01 路 DWD 一开始**没有**
    按 event_name 过滤 —— 于是护栏的取值（延迟 ~100ms）被加进了主指标，
    效应从 +27.2 变成 +161，而一切看起来都"正常显著"。
    这条测试把那个过滤条件钉住。
    """

    def test_guardrail_tables_exist_and_are_declared_only(self, warehouse_con):
        """08/09 两张表存在；且只包含**被声明**的护栏（配置维表说了算）。"""
        declared = {
            r[0]
            for r in warehouse_con.execute(
                "SELECT DISTINCT guardrail FROM dim_guardrail_config"
            ).fetchall()
        }
        assert declared, "演示配置里应当声明了护栏"
        produced = {
            r[0]
            for r in warehouse_con.execute(
                "SELECT DISTINCT guardrail FROM ads_experiment_guardrail_result"
            ).fetchall()
        }
        assert produced <= declared, (produced, declared)
        assert produced, "ADS 里应当有护栏数据"

    def test_guardrail_events_do_not_pollute_the_main_metric(self, warehouse_con):
        """主指标只算 ``interaction`` 事件 —— 否则护栏取值会混进来。

        实测（不加过滤时）：效应 +27.2 -> +161。两个数都"显著"，
        所以这类错误不会被显著性检查发现，只能靠这条不变量。
        """
        rows = warehouse_con.execute(
            "SELECT DISTINCT event_name FROM ods_event_log ORDER BY 1"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "interaction" in names
        assert len(names) > 1, "护栏事件应当也在事件表里（长表），否则这条测试没意义"
        # DWD 的主指标必须与"只算 interaction"完全一致
        total = warehouse_con.execute(
            "SELECT SUM(post_metric) FROM dwd_experiment_user"
        ).fetchone()[0]
        only_interaction = warehouse_con.execute(
            """
            SELECT SUM(metric_value) FROM ods_event_log
            WHERE event_name = 'interaction' AND metric_value > 0
            """
        ).fetchone()[0]
        # 粗粒度数量级校验：两者必须同量级；一旦护栏混入，量级会翻几倍
        assert total < only_interaction * 1.5, (total, only_interaction)

    def test_guardrail_harm_is_visible_in_the_ads(self, warehouse_con):
        """注入的 +12% 伤害必须在 ADS 的均值上看得出来（可核对）。"""
        rows = dict(
            (
                (r[0], r[1]),
                (int(r[2]), float(r[3])),
            )
            for r in warehouse_con.execute(
                """
                SELECT variant, guardrail, user_cnt, value_mean
                FROM ads_experiment_guardrail_result
                WHERE experiment = 'exp_rank_v2' AND guardrail = 'latency_p99'
                """
            ).fetchall()
        )
        c_n, c_mean = rows[("control", "latency_p99")]
        t_n, t_mean = rows[("treatment", "latency_p99")]
        assert c_n > 1000 and t_n > 1000
        assert 0.10 < (t_mean - c_mean) / c_mean < 0.14, (c_mean, t_mean)

    def test_warehouse_path_judges_guardrails_and_recommends_stopping(self):
        """走真实数仓路径：判定 fail，并给出"建议停止实验"。"""
        import duckdb

        from ablab.platform.analysis import analyse_experiment_from_warehouse
        from ablab.platform.api import default_warehouse_path
        from ablab.platform.guardrails import GuardrailSpec
        from ablab.platform.registry import ExperimentRecord

        path = default_warehouse_path()
        if not path.exists():
            import pytest

            pytest.skip("没有数仓文件")
        con = duckdb.connect(str(path), read_only=True)
        try:
            rec = ExperimentRecord(
                name="exp_rank_v2",
                variants=list(TWO_ARM),
                salt="exp_rank_v2_v1",
                primary_metric="interaction_per_user_14d",
                warehouse_experiment="exp_rank_v2",
                guardrails=["latency_p99", "complaint_rate"],
                guardrail_specs=[
                    GuardrailSpec("latency_p99", "lower_is_better", 0.05),
                    GuardrailSpec("complaint_rate", "lower_is_better", 0.10),
                ],
            )
            report = analyse_experiment_from_warehouse(rec, con)
        finally:
            con.close()
        item = next(c for c in report.checks if c.name == "护栏指标")
        assert item.status == "fail", item.message
        assert report.health == "fail"
        assert "停止实验" in item.message
        assert "latency_p99" in item.message

    def test_missing_guardrail_table_degrades_to_unknown(self):
        """读不到护栏数据时返回空字典 -> 判 unknown（不是通过）。"""
        import duckdb

        from ablab.platform.datasource import _warehouse_guardrail_data

        con = duckdb.connect(":memory:")
        try:
            # 内存库里没有 09 路表：这不是错误，而是"这份数据源还没有护栏"
            series = _warehouse_guardrail_data(con, "exp_x", "control", "treatment")
        finally:
            con.close()
        assert series == {}


class TestWarehouseRatioMetric:
    """比值指标在**数仓侧**也要走对口径（M1 的独立实现当裁判）。

    比值链路的 ADS（07）落的是分子/分母两列可加量，与均值口径那张 ADS（03）
    的六个可加量语义不同。所以这一组测试的判据不是"平台自己前后一致"，
    而是**平台读数仓 == M1 的 `ratio_delta_method` 直接吃明细** ——
    前者是 M5 写的编排，后者是 M1 写的独立实现。
    """

    @staticmethod
    def _stats(y, x):
        import numpy as np

        from ablab.inference.aggregates import AggregateStats

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        return AggregateStats.from_sums(
            n=int(y.size),
            sum_x=float(x.sum()),
            sum_y=float(y.sum()),
            sum_xx=float((x * x).sum()),
            sum_yy=float((y * y).sum()),
            sum_xy=float((x * y).sum()),
        )

    def test_ratio_path_matches_detail(self, warehouse_con):
        from ablab.inference import ratio_delta_method
        from ablab.platform.analysis import analyse_experiment_from_warehouse
        from ablab.platform.registry import ExperimentRecord

        con = warehouse_con
        detail = con.execute(
            """
            SELECT variant, post_metric, post_cnt FROM dwd_experiment_user
            WHERE experiment = 'exp_rank_v2'
            """
        ).df()
        t = detail[detail["variant"] == "treatment"]
        c = detail[detail["variant"] == "control"]
        # 分子 = 互动值之和，分母 = 互动次数之和 —— 与 06/07 两张表的口径一致
        ref = ratio_delta_method(
            self._stats(t["post_metric"], t["post_cnt"]),
            self._stats(c["post_metric"], c["post_cnt"]),
        )

        rec = ExperimentRecord(
            name="wh_ratio",
            variants=[{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
            salt="wh_ratio_v1",
            primary_metric="post_metric_14d",
            warehouse_experiment="exp_rank_v2",
            estimator="post_only",
            metric_type="ratio",
        )
        rep = analyse_experiment_from_warehouse(rec, con, n_looks=5)

        assert rep.primary is not None
        assert rep.primary.absolute_effect == pytest.approx(
            ref.absolute_effect, abs=1e-9
        ), f"平台 {rep.primary.absolute_effect} vs 明细 {ref.absolute_effect}"
        assert rep.primary.std_error == pytest.approx(ref.std_error, abs=1e-9)

    def test_ratio_and_cuped_are_refused_at_creation(self, work_dir):
        """CUPED 需要前置协变量，而比值链路里只有分子/分母 —— 创建时就拦住。"""
        from ablab.platform.registry import ExperimentRegistry, RegistryError

        reg = ExperimentRegistry(work_dir / "ratio_gate.db")
        with pytest.raises(RegistryError, match="ratio"):
            reg.create(actor=ACTOR, 
                name="bad_ratio",
                variants=[{"name": "control", "weight": 0.5}, {"name": "treatment", "weight": 0.5}],
                salt="bad_ratio_v1",
                metric_type="ratio",
                estimator="cuped",
            )
        reg.close()

    def test_ratio_link_is_additive_not_a_column(self, warehouse_con):
        """比值链路必须是**新表**，不能往均值链路里加列。

        这是硬约束的守卫：README 里所有已引用的数仓数字都挂在 02/03 上，
        给它们加列（哪怕不改值）也会让下游按位置取列的代码错位。
        断言只用**结构不变式** —— 第一版这里写了两个我现编的数字
        （16033 之类），一跑就红：**测试里不能出现没量过的数**，
        这和 README 里"每个数字都能在 reports/ 里找到"是同一条纪律。
        """
        con = warehouse_con
        mean_cols = {
            r[1] for r in con.execute("PRAGMA table_info(dws_experiment_variant_daily)").fetchall()
        }
        assert "sum_y" not in mean_cols and "sum_x" not in mean_cols, mean_cols

        ratio_cols = {
            r[1] for r in con.execute("PRAGMA table_info(dws_experiment_ratio_daily)").fetchall()
        }
        assert {"sum_y", "sum_x", "sum_xx", "sum_yy", "sum_xy"} <= ratio_cols, ratio_cols
        assert "user_cnt" in ratio_cols

        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert {"dws_experiment_ratio_daily", "ads_experiment_ratio_result"} <= tables

        # 两张 DWS 出自同一张 DWD、同一组分组键 → 行数必须相同
        mean_rows = con.execute("SELECT COUNT(*) FROM dws_experiment_variant_daily").fetchone()[0]
        ratio_rows = con.execute("SELECT COUNT(*) FROM dws_experiment_ratio_daily").fetchone()[0]
        assert mean_rows == ratio_rows, (mean_rows, ratio_rows)


class TestWarehouseAPI:
    def test_lists_warehouse_experiments(self, wh_client):
        body = wh_client.get("/api/warehouse/experiments").json()
        assert body["available"] is True
        names = {e["experiment"] for e in body["experiments"]}
        assert "exp_rank_v2" in names

    def test_without_warehouse_degrades_gracefully(self, plain_client):
        body = plain_client.get("/api/warehouse/experiments").json()
        assert body["available"] is False and body["experiments"] == []
        # 合成路径照常工作
        eid = plain_client.get("/api/experiments").json()[0]["id"]
        r = plain_client.post(f"/api/experiments/{eid}/analyze", json={"n_users": 3000})
        assert r.status_code == 200 and r.json()["source"] == "synthetic"

    def test_seed_adds_a_warehouse_bound_demo(self, wh_client):
        items = {i["name"]: i for i in wh_client.get("/api/experiments").json()}
        assert WAREHOUSE_DEMO["name"] in items
        assert items[WAREHOUSE_DEMO["name"]]["warehouse_experiment"] == "exp_rank_v2"

    def test_analyze_bound_record_reads_warehouse(self, wh_client):
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        r = wh_client.post(
            f"/api/experiments/{items[WAREHOUSE_DEMO['name']]}/analyze",
            json={"alpha": 0.05, "n_looks": 5},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "warehouse"
        assert body["population_size"] is None
        # 与 M0 数仓报告里的数字同源（真值 +2.0/天 × 14 天）
        assert body["cuped"]["absolute_effect"] > 0
        assert body["checks"][0]["name"] == "SRM"
        # 监控的查看标签是日期
        assert "（累计）" in body["monitoring"][0]["label"]

    def test_analyze_bound_record_rejects_seed(self, wh_client):
        """数仓数据是既成的，seed 没有意义 —— 明确报错，不静默忽略。"""
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        r = wh_client.post(
            f"/api/experiments/{items[WAREHOUSE_DEMO['name']]}/analyze",
            json={"seed": 7},
        )
        assert r.status_code == 400
        assert "seed" in r.json()["detail"]

    def test_bind_valid(self, wh_client):
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        r = wh_client.post(
            f"/api/experiments/{items['exp_rec_emb']}/bind",
            json={"warehouse_experiment": "exp_rec_emb"},
        )
        assert r.status_code == 200
        assert r.json()["warehouse_experiment"] == "exp_rec_emb"

    def test_bind_unknown_rolls_back(self, wh_client):
        """绑一个不存在的数仓实验必须当场失败并回滚。

        否则用户要到点"分析"时才发现 —— 那是更晚、更贵的反馈。
        """
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        eid = items["exp_rec_emb"]
        r = wh_client.post(
            f"/api/experiments/{eid}/bind", json={"warehouse_experiment": "nope"}
        )
        assert r.status_code == 400
        assert "没有实验" in r.json()["detail"]
        # 回滚检查：仍然是未绑定状态
        assert wh_client.get(f"/api/experiments/{eid}").json()["warehouse_experiment"] is None

    def test_unbind(self, wh_client):
        items = {i["name"]: i["id"] for i in wh_client.get("/api/experiments").json()}
        eid = items[WAREHOUSE_DEMO["name"]]
        r = wh_client.post(f"/api/experiments/{eid}/bind", json={"warehouse_experiment": None})
        assert r.status_code == 200 and r.json()["warehouse_experiment"] is None
        # 解绑后回到合成路径，此时 seed 又生效了
        body = wh_client.post(
            f"/api/experiments/{eid}/analyze", json={"n_users": 3000, "seed": 1}
        ).json()
        assert body["source"] == "synthetic"

    def test_bind_without_warehouse_still_records(self, plain_client):
        """没配数仓时允许先记下绑定 —— 但分析会明确报错，而不是悄悄走合成数据。

        悄悄降级是最坏的选择：报告里写着"数仓链路"，数字其实是平台编的。
        """
        items = {i["name"]: i["id"] for i in plain_client.get("/api/experiments").json()}
        eid = items["exp_rec_emb"]
        assert plain_client.post(
            f"/api/experiments/{eid}/bind", json={"warehouse_experiment": "exp_rec_emb"}
        ).status_code == 200
        r = plain_client.post(f"/api/experiments/{eid}/analyze", json={})
        assert r.status_code == 400
        assert "数仓" in r.json()["detail"]
