#!/usr/bin/env python
"""M5 验证：平台层的全部证据。

运行::

    python scripts/run_m5_validation.py             # 完整版，约 3 分钟
    python scripts/run_m5_validation.py --quick     # 快速版

M5 不引入新方法，所以这里验的不是分布性质，而是**工程不变量** + 一件必须在这里测的事：

1. **整条管道的 A/A 校准**（换 400 个 salt）。平台比 M0 的验证台多用了几路随机数
   （合成协变量、结果噪声、查看顺序），流一旦串了零效应实验就会超发。
2. **随机流独立性**：``default_rng(S)`` 与 ``default_rng(S+1)`` 到底独立不独立。
3. **两个成分跨 salt 的分布**：ΔX̄ 与 Δeps 各自是不是标准正态。
4. **效应分解恒等式**：naive − CUPED 必须恰好等于 θ̂·ΔX̄。
5. **演示实验的分解**，包括被挑选规则拒掉的那个 salt（如实留档，不隐藏）。
6. **接口契约**：用 TestClient 把每个端点跑一遍。
7. **数仓链路等价性**：同一批数据的三条读取路径（DWD 明细 / ADS 汇总 / 平台编排）
   必须给同一个答案，并画出真实链路上的**按日累计**监控路径。

输出到 ``reports/``：

    fig25_platform_aa.png        A/A 下 z 的分布 vs 标准正态，以及 FPR 的 Wilson 区间
    fig26_effect_decomposition.png  演示实验的效应分解（失衡 vs 残余）
    fig27_warehouse_monitoring.png  真实链路上的按日累计监控（数仓路径独有）
    m5_validation.md             全部数字
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy import stats  # noqa: E402

from ablab.platform import (  # noqa: E402
    run_demo_decomposition,
    run_noise_component_audit,
    run_platform_aa_audit,
    run_source_equivalence_audit,
    run_stream_independence_audit,
)
from ablab.platform.analysis import (  # noqa: E402
    ExperimentReport,
    analyse_experiment_from_warehouse,
)
from ablab.platform.api import default_warehouse_path  # noqa: E402
from ablab.platform.datasource import (  # noqa: E402
    build_warehouse_data,
    list_warehouse_experiments,
)
from ablab.platform.demo import DEMO_EXPERIMENTS  # noqa: E402
from ablab.platform.registry import ExperimentRecord  # noqa: E402
from ablab.plotting import label, plt, save, setup_style  # noqa: E402
from ablab.reporting import for_report  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN, PURPLE = (
    "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c", "#9467bd",
)


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def fig_aa(aa, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    grid = np.linspace(-4.5, 4.5, 300)
    bins = np.linspace(-4.5, 4.5, 40)
    ax.hist(aa.naive_z, bins=bins, density=True, alpha=0.45, color=GREY,
            label=label(f"naive z（n={aa.n_salts}）", f"naive z (n={aa.n_salts})"))
    ax.hist(aa.cuped_z, bins=bins, density=True, alpha=0.45, color=BLUE,
            label=label(f"CUPED z（n={aa.n_salts}）", f"CUPED z (n={aa.n_salts})"))
    ax.plot(grid, stats.norm.pdf(grid), color=RED, lw=2.0, label="N(0,1)")
    for s in (1.959963985, -1.959963985):
        ax.axvline(s, color=RED, ls=":", lw=1.2)
    ax.set_title(label("A/A 下 z 的分布（400 个 salt）", "z distribution under A/A (400 salts)"))
    ax.set_xlabel(label("z 统计量", "z statistic"))
    ax.set_ylabel(label("密度", "density"))
    ax.legend(fontsize=8)

    ax = axes[1]
    names = ["naive", "CUPED"]
    fprs = [aa.naive_fpr, aa.cuped_fpr]
    los = [aa.naive_fpr_interval[0], aa.cuped_fpr_interval[0]]
    his = [aa.naive_fpr_interval[1], aa.cuped_fpr_interval[1]]
    xs = np.arange(2)
    ax.bar(xs, fprs, width=0.5, color=[GREY, BLUE], alpha=0.8)
    ax.errorbar(xs, fprs, yerr=[np.array(fprs) - np.array(los), np.array(his) - np.array(fprs)],
                fmt="none", ecolor="black", capsize=5, lw=1.4)
    ax.axhline(aa.alpha, color=RED, ls="--", lw=1.5, label=label("α = 0.05", "alpha = 0.05"))
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(0.12, max(his) * 1.25))
    ax.set_title(label("零效应下的显著率（Wilson 95% 区间）", "FPR under null (Wilson 95% CI)"))
    ax.set_ylabel(label("经验 FPR", "empirical FPR"))
    ax.legend(fontsize=8)

    fig.suptitle(
        label(
            f"平台管道 A/A 校准：覆盖 {aa.coverage:.3f}（真值 0），z 标准差 "
            f"{aa.naive_z_sd:.3f} / {aa.cuped_z_sd:.3f}",
            f"Pipeline A/A calibration: coverage {aa.coverage:.3f}, z sd "
            f"{aa.naive_z_sd:.3f} / {aa.cuped_z_sd:.3f}",
        ),
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out / "fig25_platform_aa.png")


def fig_decomposition(decomps, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    # 同名的那条是被挑选规则拒掉的 salt（留档对照），标出来避免误读
    names = [d.name for d in decomps]
    for i in range(len(names)):
        if names[i] in names[:i]:
            names[i] = f"{names[i]}\n(被拒 salt)"
    xs = np.arange(len(decomps))
    imb = np.array([d.imbalance_component for d in decomps])
    res = np.array([d.residual_component for d in decomps])

    ax.bar(xs, imb, width=0.55, color=ORANGE, alpha=0.85,
           label=label("前置协变量失衡（CUPED 扣掉）", "pre-treatment imbalance (removed by CUPED)"))
    ax.bar(xs, res, width=0.55, bottom=imb, color=BLUE, alpha=0.85,
           label=label("校正后残余（真实效应 + 结果噪声）", "residual (true effect + noise)"))
    ax.axhline(0.0, color="black", lw=0.8)
    for x, d in zip(xs, decomps):
        ax.annotate(
            f"naive z={d.naive_z:+.2f}\nCUPED z={d.cuped_z:+.2f}",
            (x, d.naive_effect), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=12)
    ax.set_ylabel(label("效应（指标单位）", "effect (metric units)"))
    ax.set_title(
        label(
            "演示实验的效应分解：naive = 失衡贡献 + 校正后残余",
            "Effect decomposition: naive = imbalance + residual",
        )
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "fig26_effect_decomposition.png")


def fig_warehouse_monitoring(paths: list[tuple[str, ExperimentReport]], out: Path) -> None:
    """真实链路上的按日累计监控。

    这是数仓路径独有的东西：合成数据的"查看"是按用户进入顺序取前缀，
    数仓的"查看"是**按天累计**（DWS 天然按 ds 汇总）。
    信息比例因此是实际累计样本量之比，而不是日历天数之比。
    """
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    colors = [BLUE, ORANGE]
    for (name, rep), color in zip(paths, colors):
        zs = [m["z"] for m in rep.monitoring]
        xs = list(range(1, len(zs) + 1))
        ax.plot(xs, zs, "o-", color=color, lw=1.8, ms=6, label=f"{name}（z）")
        ax.plot(
            xs, [m["boundary"] for m in rep.monitoring], "--", color=color, lw=1.2,
            alpha=0.75, label=f"{name}（OBF 边界）",
        )
        for x, m in zip(xs, rep.monitoring):
            if m["crossed"]:
                ax.plot([x], [m["z"]], "o", mfc="none", mec=RED, ms=14, mew=2.2)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(range(1, len(paths[0][1].monitoring) + 1))
    ax.set_xticklabels(
        [str(m["label"]).replace("（累计）", "") for m in paths[0][1].monitoring],
        fontsize=8, rotation=12,
    )
    ax.set_xlabel(label("查看日期（累计）", "look date (cumulative)"))
    ax.set_ylabel(label("z 统计量", "z statistic"))
    ax.set_title(
        label(
            "真实数仓链路的按日累计监控：z 与 OBF 边界（红圈 = 触及边界）",
            "Daily cumulative monitoring on the warehouse path",
        )
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, out / "fig27_warehouse_monitoring.png")
def main() -> int:
    ap = argparse.ArgumentParser(description="M5 平台层验证")
    ap.add_argument("--quick", action="store_true", help="快速版（数字更粗）")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()

    n_salts = 120 if args.quick else 400
    n_component = 600 if args.quick else 2_000
    n_stream = 500 if args.quick else 2_000

    log: list[str] = []
    t_start = time.time()

    def say(line: str = "") -> None:
        print(line, flush=True)
        log.append(line)

    # ---- 1. 整条管道的 A/A 校准 ------------------------------------------ #
    say("=" * 78)
    say(f"1. 整条 analyse_experiment 管道：{n_salts} 次换 salt 的零效应实验")
    say("=" * 78)
    t0 = time.time()
    aa = run_platform_aa_audit(n_salts=n_salts, n_units=20_000)
    say(f"耗时 {time.time() - t0:.0f}s，每次 {aa.n_units:,} 用户")
    say("")
    say(f"{'口径':<8}{'FPR':>9}{'Wilson 95% CI':>24}{'z 均值':>11}{'z 标准差':>11}")
    say(f"{'naive':<8}{aa.naive_fpr:>9.4f}"
        f"{f'({aa.naive_fpr_interval[0]:.4f}, {aa.naive_fpr_interval[1]:.4f})':>24}"
        f"{aa.naive_z_mean:>11.4f}{aa.naive_z_sd:>11.4f}")
    say(f"{'CUPED':<8}{aa.cuped_fpr:>9.4f}"
        f"{f'({aa.cuped_fpr_interval[0]:.4f}, {aa.cuped_fpr_interval[1]:.4f})':>24}"
        f"{aa.cuped_z_mean:>11.4f}{aa.cuped_z_sd:>11.4f}")
    say("")
    say(f"CUPED 区间对真值 0 的覆盖率 = {aa.coverage:.4f} "
        f"CI=({aa.coverage_interval[0]:.4f}, {aa.coverage_interval[1]:.4f})")
    say(f"判定 calibrated = {aa.calibrated}")
    say("")
    say("判据：两个 FPR 的 Wilson 区间都要盖住 alpha=0.05，且 z 的标准差都在 1±0.15 内。")
    say("注意这一轮**没有任何挑选** —— 400 个 salt 都是按编号顺序生成的。")

    # ---- 2. 随机流独立性 -------------------------------------------------- #
    say("")
    say("=" * 78)
    say(f"2. 随机流独立性：default_rng(S) 与 default_rng(S+1)（{n_stream} 组）")
    say("=" * 78)
    t0 = time.time()
    si = run_stream_independence_audit(n_reps=n_stream)
    say(f"耗时 {time.time() - t0:.0f}s（固定同一个分组掩码，只改种子构造方式）")
    say("")
    say(f"{'种子构造':<34}{'corr(ΔX̄, Δeps)':>18}")
    say(f"{'相邻 S / S+1（管道实际用法）':<34}{si.adjacent_corr:>+18.4f}")
    say(f"{'远离 S / S+1e6':<34}{si.far_corr:>+18.4f}")
    say(f"{'SeedSequence.spawn（理想）':<34}{si.spawn_corr:>+18.4f}")
    say("")
    say(f"抽样误差 ±{si.tolerance:.4f}（3/sqrt(n)）；ΔX̄ 的 z 标准差 {si.adjacent_sd_pre:.3f}，"
        f"Δeps 的 z 标准差 {si.adjacent_sd_eps:.3f}")
    say(f"判定 independent = {si.independent}")
    say("")
    say("结论：管道用相邻整数种子分离协变量流与结果流，实测相关在抽样误差内为 0。")

    # ---- 3. 两个成分跨 salt 的分布 ---------------------------------------- #
    say("")
    say("=" * 78)
    say(f"3. 两个成分在**不同 salt 之间**的分布（{n_component} 个 salt）")
    say("=" * 78)
    t0 = time.time()
    nc = run_noise_component_audit(n_salts=n_component, n_units=20_000)
    say(f"耗时 {time.time() - t0:.0f}s")
    say("")
    say(f"{'成分':<10}{'均值':>10}{'标准差':>10}{'最大|z|':>10}{'|z|>3 个数':>12}")
    say(f"{'ΔX̄':<10}{nc.d_pre_z_mean:>10.4f}{nc.d_pre_z_sd:>10.4f}"
        f"{nc.d_pre_max_abs:>10.3f}{nc.d_pre_over3:>12d}")
    say(f"{'Δeps':<10}{nc.d_noise_z_mean:>10.4f}{nc.d_noise_z_sd:>10.4f}"
        f"{nc.d_noise_max_abs:>10.3f}{nc.d_noise_over3:>12d}")
    say("")
    b = np.array(nc.d_noise_z)
    say("Δeps 的经验分位 vs 理论分位：")
    for q in (0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995):
        say(f"  {q:>6.3f}   经验 {np.quantile(b, q):+.4f}   理论 {stats.norm.ppf(q):+.4f}")
    say("")
    say(f"|z|>3 的经验比例 {float((np.abs(b) > 3).mean()):.5f}（理论 0.00270）")
    say(f"判定 looks_standard_normal = {nc.looks_standard_normal}")

    # ---- 4. 演示实验的效应分解 -------------------------------------------- #
    say("")
    say("=" * 78)
    say("4. 演示实验的效应分解（含被挑选规则拒掉的 salt，如实留档）")
    say("=" * 78)
    rejected = dict(DEMO_EXPERIMENTS[1], salt="exp_rec_emb_v1")
    t0 = time.time()
    decomps = run_demo_decomposition(n_users=20_000)
    rejected_d = run_demo_decomposition(n_users=20_000, demos=(rejected,))[0]
    say(f"耗时 {time.time() - t0:.0f}s，每次 20,000 用户")
    say("")
    say(f"{'实验':<17}{'salt':<21}{'true_lift':>10}{'naive':>10}{'naive z':>9}"
        f"{'CUPED':>10}{'CUPED z':>9}{'失衡':>10}")
    for d, salt in zip(decomps, [x["salt"] for x in DEMO_EXPERIMENTS]):
        say(f"{d.name:<17}{salt:<21}{d.true_lift:>+10.2f}{d.naive_effect:>+10.4f}"
            f"{d.naive_z:>+9.2f}{d.cuped_effect:>+10.4f}{d.cuped_z:>+9.2f}"
            f"{d.imbalance_component:>+10.4f}")
    say(f"{'[被拒] ' + rejected_d.name:<17}{'exp_rec_emb_v1':<21}{rejected_d.true_lift:>+10.2f}"
        f"{rejected_d.naive_effect:>+10.4f}{rejected_d.naive_z:>+9.2f}"
        f"{rejected_d.cuped_effect:>+10.4f}{rejected_d.cuped_z:>+9.2f}"
        f"{rejected_d.imbalance_component:>+10.4f}")
    say("")
    say("分解恒等式逐条核对（naive − CUPED 是否恰等于 θ̂·ΔX̄）：")
    for d in (*decomps, rejected_d):
        lhs = d.naive_effect - d.cuped_effect
        rhs = d.imbalance_component
        say(f"  {d.name:<16} 左 {lhs:+.10f}   右 {rhs:+.10f}   差 {abs(lhs - rhs):.2e}")
    say("")
    say("被拒掉的那个 salt 的成因（这就是「演示数据为什么要挑」的证据）：")
    say(f"  ΔX̄ z={rejected_d.split_z:+.3f}  Δeps z={rejected_d.noise_z:+.3f}"
        f"  → 两个成分都在 3σ 附近，且难得地同号")
    p_split = float((np.abs(np.array(nc.d_pre_z)) >= abs(rejected_d.split_z)).mean())
    p_noise = float((np.abs(b) >= abs(rejected_d.noise_z)).mean())
    say(f"  按第 3 节的分布：ΔX̄ 双尾经验 p={p_split:.5f}，Δeps 双尾经验 p={p_noise:.5f}")
    say(f"  同一次实现里两者同时这么极端 → 约 {p_split * p_noise:.2e} 量级。"
        "机制没有错，是这份实现落在了尾部。")

    # ---- 5. 接口契约 ------------------------------------------------------- #
    say("")
    say("=" * 78)
    say("5. 接口契约（TestClient，逐端点）")
    say("=" * 78)
    from fastapi.testclient import TestClient

    from ablab.platform.api import create_app
    from ablab.platform.demo import seed_demo

    tmp = ROOT / "build" / "_m5_api_check.db"
    if tmp.exists():
        tmp.unlink()
    # 带上数仓一起测 —— 否则绑定相关的端点在这份契约里根本没被跑到
    api_wh = default_warehouse_path()
    app = create_app(tmp, warehouse_path=api_wh if api_wh.exists() else None)
    seeded = seed_demo(app.state.registry, warehouse_available=api_wh.exists())
    with TestClient(app) as c:
        hz = c.get("/healthz").json()
        say(f"GET  /healthz                        -> {hz}")
        say(f"GET  /                               -> {c.get('/').status_code} "
            f"{c.get('/').headers['content-type']}")
        items = c.get("/api/experiments").json()
        by_name = {i["name"]: i["id"] for i in items}
        say(f"GET  /api/experiments                -> {len(items)} 条（本次 seed 新增 {seeded}）")
        say(f"GET  /api/experiments?status=running -> "
            f"{len(c.get('/api/experiments?status=running').json())} 条")
        say(f"GET  /api/experiments?status=nope    -> "
            f"{c.get('/api/experiments?status=nope').status_code}")
        bad = c.post("/api/experiments",
                     json={"name": "bad", "variants": [{"name": "a", "weight": 0.3}]})
        say(f"POST /api/experiments（权重和!=1）    -> {bad.status_code} {bad.json()['detail'][:28]}…")
        dup = c.post("/api/experiments",
                     json={"name": DEMO_EXPERIMENTS[0]["name"], "variants": [
                         {"name": "control", "weight": 0.5},
                         {"name": "treatment", "weight": 0.5}]})
        say(f"POST /api/experiments（重名）         -> {dup.status_code}")
        eid = items[0]["id"]
        rep = c.post(f"/api/experiments/{eid}/analyze", json={"n_users": 20_000}).json()
        say(f"POST /api/experiments/{{id}}/analyze   -> health={rep['health']}，"
            f"体检项 {len(rep['checks'])} 条：{', '.join(x['name'] for x in rep['checks'])}")
        say(f"     效应分解字段 imbalance={rep['imbalance_component']:+.4f} "
            f"residual={rep['residual_component']:+.4f}")
        say(f"GET  /api/experiments/{{id}}           -> {c.get(f'/api/experiments/{eid}').status_code}")
        say(f"GET  /api/experiments/nope           -> "
            f"{c.get('/api/experiments/nope').status_code}")
        say(f"PATCH /api/experiments/{{id}}/status   -> "
            f"{c.patch(f'/api/experiments/{eid}/status', json={'status': 'stopped'}).status_code}")
        # 守卫要求每组至少 2·n_looks 个样本。这个配置在 n_users=200 时每组只有 10 个，
        # 所以把 n_looks 提到 20（需要 40 个）就明确越界；直接用 5 次会正好压在边界上。
        small = c.post(
            f"/api/experiments/{by_name['exp_ui_density']}/analyze",
            json={"n_users": 200, "n_looks": 20},
        )
        say(f"POST analyze（90/10 + 30% 流量，每组仅 10 样本却要 20 次查看） -> "
            f"{small.status_code} {small.json().get('detail', '')[:30]}…")
        aa_live = c.post("/api/validate/aa", json={"n_trials": 200, "n_units": 3_000}).json()
        say(f"POST /api/validate/aa                -> FPR={aa_live['empirical_fpr']:.4f} "
            f"CI=({aa_live['fpr_interval'][0]:.4f},{aa_live['fpr_interval'][1]:.4f}) "
            f"calibrated={aa_live['calibrated']}")
        whl = c.get("/api/warehouse/experiments").json()
        say(f"GET  /api/warehouse/experiments      -> available={whl['available']}，"
            f"{len(whl['experiments'])} 个可选："
            f"{[e['experiment'] for e in whl['experiments']]}")
        ev = c.post(f"/api/experiments/{by_name['exp_rec_emb']}/bind",
                    json={"warehouse_experiment": "nope"})
        say(f"POST /bind（数仓里没有这个实验）       -> {ev.status_code} "
            f"{ev.json()['detail'][:26]}…")
        ok_bind = c.post(f"/api/experiments/{by_name['exp_rec_emb']}/bind",
                         json={"warehouse_experiment": "exp_rec_emb"})
        say(f"POST /bind（合法）                    -> {ok_bind.status_code} "
            f"bound={ok_bind.json()['warehouse_experiment']}")
        wh_rep = c.post(f"/api/experiments/{by_name['exp_rec_emb']}/analyze",
                        json={"n_looks": 5}).json()
        say(f"POST analyze（绑定后）                -> source={wh_rep['source']} "
            f"n={wh_rep['n_users']:,} CUPED={wh_rep['cuped']['absolute_effect']:.4f} "
            f"p={wh_rep['cuped']['p_value']:.4g}")
        seed_reject = c.post(f"/api/experiments/{by_name['exp_rec_emb']}/analyze",
                             json={"seed": 7})
        say(f"POST analyze（数仓路径带 seed）        -> {seed_reject.status_code} "
            f"{seed_reject.json()['detail'][:26]}…")
        say(f"DELETE /api/experiments/{{id}}         -> "
            f"{c.delete(f'/api/experiments/{eid}').status_code}")
        say(f"GET  /api/experiments/{{id}}（删除后） -> "
            f"{c.get(f'/api/experiments/{eid}').status_code}")
    app.state.registry.close()
    if tmp.exists():
        tmp.unlink()

    # ---- 6. 数仓链路等价性 ------------------------------------------------- #
    say("")
    say("=" * 78)
    say("6. 数仓链路：三条读取路径必须给同一个答案")
    say("=" * 78)
    wh_path = default_warehouse_path()
    wh_reports: list[tuple[str, ExperimentReport]] = []
    if not wh_path.exists():
        say(f"跳过：{wh_path} 不存在。先跑 `python scripts/run_warehouse.py` 再执行本脚本。")
    else:
        import duckdb

        con = duckdb.connect(str(wh_path), read_only=True)
        say(f"数仓: {wh_path}")
        say("")
        say("可绑定的数仓实验：")
        for item in list_warehouse_experiments(con):
            say(f"  {item['experiment']:<16} 层={item['layer']:<10} "
                f"分支={item['n_variants']}  人数={item['n_users']:,}")
        say("")
        say("三条路径：① DWD 明细  ② ADS 汇总（M0 的独立实现）③ 平台编排（M5）")
        say("")
        say(f"{'实验':<15}{'路径':<10}{'naive 效应':>14}{'naive SE':>12}"
            f"{'CUPED 效应':>14}{'CUPED SE':>12}")
        for experiment in ("exp_rank_v2", "exp_rec_emb"):
            eq = run_source_equivalence_audit(con, experiment, n_looks=5)
            for tag, g in (
                ("① DWD 明细", eq.from_detail),
                ("② ADS 汇总", eq.from_ads),
                ("③ 平台编排", eq.from_platform),
            ):
                say(f"{experiment:<15}{tag:<10}{g[0]:>14.6f}{g[1]:>12.6f}"
                    f"{g[2]:>14.6f}{g[3]:>12.6f}")
            say(f"{'':<15}{'最大偏差':<10}{eq.max_deviation:>14.3e}"
                f"{'':>12}{'一致' if eq.agree else '不一致 <<<':>14}")
            say(f"{'':<15}最后一次查看 == 主结论：{eq.last_look_matches}")
            say(f"{'':<15}监控信息比例（实际累计，不是日历天数）："
                f"{[round(x, 4) for x in eq.information_fractions]}")
            say("")

            record = ExperimentRecord(
                id="wh_check", name=experiment, salt=f"{experiment}_v1",
                variants=[{"name": "control", "weight": 0.5},
                          {"name": "treatment", "weight": 0.5}],
                primary_metric="post_metric_14d", warehouse_experiment=experiment,
            )
            wr = analyse_experiment_from_warehouse(record, con, n_looks=5)
            wh_reports.append((experiment, wr))
            say(f"平台读该实验：source={wr.source} 进入分析 {wr.n_users:,} 人  "
                f"health={wr.health}")
            say(f"  监控查看（按日累计）："
                f"{[(m['label'][:10], m['n_per_arm']) for m in wr.monitoring]}")
            design = wr.sequential
            if design is None:
                raise RuntimeError("数仓路径的报告里应当有序贯设计（n_looks>1）")
            say(f"  末次边界={design.final_boundary:.4f} "
                f"reliable={design.reliable}")
            say("")
        data = build_warehouse_data(con, "exp_rank_v2", n_looks=5)
        say(f"数仓路径的查看标签形如 {data.looks[0].label!r} —— "
            "合成路径是 'look k'，数仓路径是**日期**，两者共用同一段监控代码。")
        con.close()

    # ---- 7. 图与报告 ------------------------------------------------------- #
    fig_aa(aa, out)
    fig_decomposition([*decomps, rejected_d], out)

    say("")
    say("=" * 78)
    say(f"总耗时 {time.time() - t_start:.0f}s；图：fig25_platform_aa.png、"
        "fig26_effect_decomposition.png"
        + ("、fig27_warehouse_monitoring.png" if wh_reports else ""))
    say("=" * 78)

    if wh_reports:
        fig_warehouse_monitoring(wh_reports, out)

    report = out / "m5_validation.md"
    report.write_text(
        "# M5 验证报告：平台层\n\n"
        "> 由 `python scripts/run_m5_validation.py` 生成，全部数字可复现。\n"
        "> M5 不引入新方法，验证重点从「分布性质」转为「工程不变量」+ 整条管道的 A/A 校准。\n\n"
        "```text\n" + "\n".join(for_report(log, root=ROOT)) + "\n```\n",
        encoding="utf-8", newline="\n",
    )
    print(f"报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
