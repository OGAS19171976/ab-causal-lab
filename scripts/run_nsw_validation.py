#!/usr/bin/env python
"""**真实处置的验证**：LaLonde NSW 实验走完整条链路，并与公开基准对照。

与 MovieLens 那份的分工
-----------------------
MovieLens 是**自造分流的 A/A**：它能校准口径与方差，但谁也不知道真效应是多少，
所以它**验证不了效应**。LaLonde NSW 补的正是这一块：真实随机化 + 公开的实验基准
（Dehejia–Wahba 子样本 ATT = +1794.34，SE 671.0）。

这个脚本做四件事：
  1. **契约**：按 `data/real_nsw/provenance.json` 逐张核对 sha256 与行数
     （直接复用 `check_real_traffic.validate` —— 契约只有一份实现）；
  2. **接入**：`load_real_traffic` + `build_warehouse`，与 MovieLens 走**同一套 SQL**；
  3. **对照基准**：实验组两臂的 re78 均值差 vs 发表值，容差写死；
  4. **两条路径同一个数**：DWD 明细与 ADS 充分统计量各算一遍 post-only z，
     对不上就红（这是"换了数据源口径没写歪"的直接证据）。

另外把三件"这份数据教我的事"印在报告里：设计权重只能用实测份额（于是 SRM 恒为 0）、
CUPED 在这份数据上几乎不省样本（corr(re75, re78) = 0.085）、以及观察性对照没做。

用法::

    python scripts/run_nsw_validation.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ablab.reporting import for_report  # noqa: E402
from ablab.warehouse import (  # noqa: E402
    ExternalExperiment,
    WarehouseConfig,
    build_warehouse,
    load_real_traffic,
)

DATA = ROOT / "data" / "real_nsw"
TARGET = ROOT / "build" / "nsw_traffic"
DB = ROOT / "build" / "warehouse_nsw.duckdb"
REPORT = ROOT / "reports" / "nsw_lalonde.md"
EXPERIMENT = "nsw_dw"

#: 年度数据：窗口按年给，不能沿用合成的 14 天。
CONFIG = WarehouseConfig(pre_days=800, post_days=1200)

#: 发表值（Dehejia–Wahba 子样本）与容差。
PUBLISHED_ATT = 1794.34
TOL = 0.05


def _load_gate():
    """按路径加载 `check_real_traffic.py`，复用它的契约校验（只有一份实现）。"""
    spec = importlib.util.spec_from_file_location(
        "_check_real_traffic", ROOT / "scripts" / "check_real_traffic.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    log: list[str] = []

    def emit(line: str = "") -> None:
        log.append(line)
        print(line)

    emit("# 真实处置的验证：LaLonde NSW（Dehejia–Wahba 子样本）")
    emit()
    emit("这份数据与 `reports/real_vs_synthetic.md` 里那份（MovieLens）**分工不同**：")
    emit("MovieLens 是自造分流的 A/A，只能校准口径与方差；这一份是**真的随机化**，")
    emit("而且有**公开的实验基准**，所以它能验证效应。")

    # ---- 1. 契约 --------------------------------------------------------- #
    emit("\n## 1. 契约：字节与声明一致")
    prov = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    gate = _load_gate()
    problems, tables = gate.validate(prov, DATA)
    problems += gate.check_generator_traces(DATA, tables)
    if problems:
        emit(f"**契约不成立（{len(problems)} 条）**：")
        for problem in problems:
            emit(f"  - {problem}")
        return 1
    emit(f"  source：{prov['source']}")
    emit(f"  许可：{prov['license']}；引用：{prov['citation'][0]}")
    for table in prov["tables"]:
        emit(f"  {table['name']:<14} {table['rows']:>5} 行  sha256 {table['sha256'][:12]}… ✓")
    emit("  上游原始文件的 sha256 也记在 provenance 里 —— 源站是可变 URL，")
    emit("  而结论挂在具体某一份字节上。")

    # ---- 2. 接入（与 MovieLens 同一套 SQL） ------------------------------- #
    emit("\n## 2. 接入：与 MovieLens 走同一套 SQL")
    exp = prov["experiments"][0]
    report = load_real_traffic(
        DATA,
        target_dir=TARGET,
        experiments=(
            ExternalExperiment(
                name=str(exp["name"]),
                variants={str(k): float(v) for k, v in exp["variants"].items()},
                control=str(exp.get("control", "control")),
                treatment=str(exp.get("treatment", "treatment")),
                layer="nsw_experiment",
            ),
        ),
        metric_event=str(prov["metric_event"]),
        guardrail_events=(),
    )
    for line in report.summary().splitlines():
        emit("  " + line)
    con = build_warehouse(
        DB, TARGET, ROOT / "sql", config=CONFIG, generate=False, verbose=False,
        metric_event=str(prov["metric_event"]),
    )
    n_dwd_row = con.execute(
        "SELECT COUNT(*) FROM dwd_experiment_user WHERE experiment = ?", [EXPERIMENT]
    ).fetchone()
    assert n_dwd_row is not None  # 聚合查询永远有一行；mypy 要这句
    n_dwd = n_dwd_row[0]
    emit(f"  窗口：pre_days={CONFIG.pre_days} / post_days={CONFIG.post_days}"
         "（年度数据不能沿用合成的 14 天）")
    emit(f"  DWD 落 {n_dwd} 个单元；三张源表 -> 12 张数仓表，一条 SQL 都没改")

    # ---- 3. 复现公开基准（明细路径） ------------------------------------- #
    emit("\n## 3. 复现公开基准（明细路径，手算）")
    detail = con.execute(
        "SELECT variant, post_metric FROM dwd_experiment_user WHERE experiment = ?",
        [EXPERIMENT],
    ).df()
    treated = detail.loc[detail["variant"] == "treatment", "post_metric"].to_numpy(float)
    control = detail.loc[detail["variant"] == "control", "post_metric"].to_numpy(float)
    att = float(treated.mean() - control.mean())
    se = math.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    emit(f"  处置组 re78 均值 = {treated.mean():.4f}（n={treated.size}）")
    emit(f"  对照组 re78 均值 = {control.mean():.4f}（n={control.size}）")
    emit(f"  **ATT = {att:+.2f}**（SE {se:.2f}, t = {att / se:+.3f}）")
    emit(f"  发表值 {PUBLISHED_ATT:+.2f} —— 偏差 {abs(att - PUBLISHED_ATT):.4f}"
         f"（容差 {TOL}）")
    if abs(att - PUBLISHED_ATT) > TOL:
        emit("\n**复现不出发表值** —— 数仓侧的列序/窗口或子样本选错了")
        return 1

    # ---- 4. 两条路径同一个数（明细 vs 充分统计量） ----------------------- #
    emit("\n## 4. 两条路径同一个数：明细 vs ADS 充分统计量")
    ads = con.execute(
        """
        SELECT variant, user_cnt, post_sum, post_sq_sum, pre_sum, pre_sq_sum,
               pre_post_cross_sum
        FROM ads_experiment_result WHERE experiment = ?
        """,
        [EXPERIMENT],
    ).fetchall()
    stats = {str(r[0]): r[1:] for r in ads}
    tc, cc = stats["treatment"], stats["control"]

    def _z(arm_t, arm_c):
        nt, st, sqt = float(arm_t[0]), float(arm_t[1]), float(arm_t[2])
        nc, sc, sqc = float(arm_c[0]), float(arm_c[1]), float(arm_c[2])
        mt, mc = st / nt, sc / nc
        vt = (sqt - st * st / nt) / (nt - 1)
        vc = (sqc - sc * sc / nc) / (nc - 1)
        return (mt - mc) / math.sqrt(vt / nt + vc / nc), mt - mc

    z_ads, tau_ads = _z(tc, cc)
    z_detail = att / se
    emit(f"  明细：τ̂ = {att:+.4f}，z = {z_detail:+.6f}")
    emit(f"  ADS ：τ̂ = {tau_ads:+.4f}，z = {z_ads:+.6f}")
    emit(f"  偏差 {abs(z_ads - z_detail):.2e} —— 两条路径用的是同一批充分统计量")
    if abs(z_ads - z_detail) > 1e-9:
        emit("\n**两条路径对不上** —— 这正是要用不变量抓的那类错")
        return 1

    # ---- 5. 这份数据教的三件事 ------------------------------------------- #
    emit("\n## 5. 这份数据教的三件事")
    srm_row = con.execute(
        "SELECT max(chi2_statistic) FROM ads_experiment_srm WHERE experiment = ?",
        [EXPERIMENT],
    ).fetchone()
    assert srm_row is not None
    srm = srm_row
    n_t, n_c = float(tc[0]), float(cc[0])
    total = n_t + n_c
    chi2_equal = (n_t - total / 2) ** 2 / (total / 2) + (n_c - total / 2) ** 2 / (total / 2)
    emit(f"  ① **SRM 在这份数据上不能作为证据**：实测份额 {n_t / total:.4f} / "
         f"{n_c / total:.4f}")
    emit(f"     按声明（= 实测）权重算 χ² = {float(srm[0]):.4f} —— 恒为 0，"
         "因为权重就是从同一批数据反推的；")
    emit(f"     若按 1:1 这个『想当然』的权重算，χ² = {chi2_equal:.2f}"
         " —— 会得出『分流坏了』的错误结论。")
    emit("     源站没有给出分配比例，所以**这份数据上 SRM 无解**："
         "设计权重必须来自声明，而声明我们没有。")

    n = float(sum(float(stats[arm][0]) for arm in ("treatment", "control")))
    post_sum = sum(float(stats[arm][1]) for arm in ("treatment", "control"))
    post_sq = sum(float(stats[arm][2]) for arm in ("treatment", "control"))
    pre_sum = sum(float(stats[arm][3]) for arm in ("treatment", "control"))
    pre_sq = sum(float(stats[arm][4]) for arm in ("treatment", "control"))
    cross = sum(float(stats[arm][5]) for arm in ("treatment", "control"))
    var_pre = (pre_sq - pre_sum * pre_sum / n) / (n - 1)
    var_post = (post_sq - post_sum * post_sum / n) / (n - 1)
    cov = (cross - pre_sum * post_sum / n) / (n - 1)
    corr = cov / math.sqrt(var_pre * var_post)
    vr = corr * corr
    emit(f"  ② **CUPED 在这份真实数据上几乎不省样本**：前置（re74+re75）与 re78 的"
         f"相关只有 {corr:+.4f}，")
    emit(f"     于是方差缩减 ≈ **{vr:.5f}**（合成器上是 0.63，MovieLens 上是 0.04）。")
    emit("     这不是实现问题，是**收入数据的真实性质**：去年的收入几乎预测不了"
         "两年后的收入。")
    emit("     这也说明合成数据上的 CUPED 结论不能外推 —— 正是接真数据买到的东西。")
    emit("  ③ **观察性对照没做**（LaLonde 那道名题的正面部分）：用 PSID/CPS 当对照组")
    emit("     会得到偏得很远的估计（原论文里甚至出现过负号）。本仓库这一步只验证了")
    emit("     **实验臂**。顺带记一个实测到的事实：源表 `data/real_nsw/user_profile.parquet`")
    emit("     里带着 age/education/black 等协变量，但**归一化把它们丢掉了** ——")
    emit("     它只保留契约列（上面接入报告里印着『丢掉的多余列』）。")
    emit("     所以下一轮做倾向得分对照，要么扩契约、要么直接读源表。")

    emit("\n## 6. 边界（如实写）")
    emit("  * 这份数据**没进 `check_real_traffic.py` 那扇门**：门写死了 `data/real/`。")
    emit("    这个脚本直接调它的 `validate`（契约只有一份实现），但"
         "『多个数据集各自成门』是另一件事，留着；")
    emit("  * 两个臂是**真实随机化**，所以这里的 z 可以读成效应检验；但 n=445、t=2.67，")
    emit("    『显著』与『估得准』仍是两件事（发表值本身也有 SE 671）；")
    emit("  * 时间维度是**映射**出来的（re74/re75/曝光/re78 落在四个日期上），"
         "不是原始数据里的观测时间；")
    emit("  * **契约之外的列会被静默丢掉**（归一化只保留契约列）——"
         "这次丢的是协变量，")
    emit("    报告里印出来了，所以不算静默；但它意味着"
         "『源表里有的列』不等于『链路里能用的列』。")

    con.close()
    REPORT.write_text(
        "# LaLonde NSW 真实处置验证\n\n```text\n"
        + "\n".join(for_report(log, root=ROOT))
        + "\n```\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n报告已写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
