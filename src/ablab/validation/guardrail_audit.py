"""护栏判定的**校准**：误停率与功效。

为什么单列一个审计
------------------
护栏的判定规则不是"显著就报警"，而是"伤害的**置信下界**越过一个**事先声明的
容忍度**"。这意味着它比"检验有没有伤害"更保守：H0 下（真实伤害为 0）
只有在估计**明显超过容忍度**时才会喊停。

保守到什么程度、以及真有害时能不能喊出来，这两件事都必须**量**出来，
不能靠推：
  * 太保守 -> 真正有害的实验也不会被拦（护栏形同虚设）；
  * 太激进 -> 没有伤害却被停（"一条永远亮的告警等于没有告警"的另一种形态）。

这正是"口径变了就重跑审计"那条规矩：新加了一个判定规则，就得给出它的
运行特征（false-stop rate / power），而不是只说"规则很合理"。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardrailCalibration:
    """护栏判定的运行特征。"""

    n_trials: int
    #: 名义 alpha（用于对比：误停率**不该**接近它，见下）
    alpha: float
    max_harm: float
    #: H0（真实伤害 = 0）下判 ``stop`` 的比例 —— **误停率**
    false_stop_rate: float
    #: H0 下判 ``watch`` 的比例（点估计越界但证据不足）
    watch_rate: float
    #: 真有害（注入 ``harm``）时判 ``stop`` 的比例 —— **功效**
    power: float
    harm: float
    n_users: int
    #: 每条护栏的判定都记下来，便于看分布而不是只看汇总
    statuses_h0: dict[str, int] = field(default_factory=dict)
    statuses_h1: dict[str, int] = field(default_factory=dict)
    #: 边界点：注入的伤害**正好等于容忍度**时的判定分布 ——
    #: 这才是这个规则真正的工作点（H0 与 H1 都太极端，看不出转变在哪）
    boundary_statuses: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"护栏判定校准（{self.n_trials} 次/场景，n={self.n_users}，"
            f"容忍度 {self.max_harm:.2%}，名义 alpha={self.alpha}）",
            f"  H0（真实伤害 0）：误停率 **{self.false_stop_rate:.4f}**"
            f"、watch {self.watch_rate:.4f}",
            f"    判定分布：{self.statuses_h0}",
            f"  H1（注入伤害 {self.harm:.2%}）：停实验的比例（功效）"
            f"**{self.power:.4f}**",
            f"    判定分布：{self.statuses_h1}",
            f"  边界点（伤害正好 = 容忍度 {self.max_harm:.2%}）："
            f"{self.boundary_statuses}",
            "    读法：规则问的是「伤害**是否超过容忍度**」，所以边界点上应当"
            "大约一半喊停（估计以真值 5% 为中心、左右各一半）——",
            "    实测偏保守（点估计要越界、且置信下界也要越界才停），这是有意的。",
            "  读法：判定阈值是**容忍度**而不是 0，所以 H0 下误停率必然**远低于** alpha ——",
            "  这是「宁可少停」的取舍；代价是功效要靠更大的伤害或更多样本来补。",
        ]
        return "\n".join(lines)


def run_guardrail_audit(
    *,
    n_trials: int = 300,
    n_users: int = 4000,
    max_harm: float = 0.05,
    harm: float = 0.12,
    alpha: float = 0.05,
    seed: int = 0,
) -> GuardrailCalibration:
    """跑两组场景（H0 / H1），统计误停率与功效。

    走的入口是**产品里那一套**（``analyse_experiment`` -> 护栏分析），
    不是自己重写一遍判定 —— 否则量的是测试代码，不是产品行为。

    每次都换一个 salt（从而换一批数据），这是"重复抽样"在该数据源上的实现方式：
    合成路径的种子挂在 salt 上，所以换 salt 等于换一次实验实现。
    """
    from ..platform.analysis import analyse_experiment
    from ..platform.guardrails import GuardrailSpec
    from ..platform.registry import ExperimentRecord

    variants = [
        {"name": "control", "weight": 0.5},
        {"name": "treatment", "weight": 0.5},
    ]

    def one(trial: int, injected: float) -> str:
        rec = ExperimentRecord(
            name="guardrail_calibration",
            variants=list(variants),
            salt=f"gr_cal_{injected}_{trial}",
            primary_metric="post_metric_14d",
            guardrails=["latency_p99"],
            guardrail_specs=[
                GuardrailSpec(
                    "latency_p99", "lower_is_better", max_harm, demo_harm=injected
                )
            ],
        )
        report = analyse_experiment(rec, n_users=n_users, alpha=alpha)
        item = next(c for c in report.checks if c.name == "护栏指标")
        # 把检查项状态映射回判定：fail=stop、warn=watch、pass=ok、info=unknown
        return {"fail": "stop", "warn": "watch", "pass": "ok", "info": "unknown"}[
            item.status
        ]

    def run_group(injected: float) -> dict[str, int]:
        counts: dict[str, int] = {"stop": 0, "watch": 0, "ok": 0, "unknown": 0}
        for i in range(n_trials):
            counts[one(i, injected)] += 1
        return counts

    h0 = run_group(0.0)
    h1 = run_group(harm)
    boundary = run_group(max_harm)
    return GuardrailCalibration(
        n_trials=n_trials,
        alpha=alpha,
        max_harm=max_harm,
        false_stop_rate=h0["stop"] / n_trials,
        watch_rate=h0["watch"] / n_trials,
        power=h1["stop"] / n_trials,
        harm=harm,
        n_users=n_users,
        statuses_h0=h0,
        statuses_h1=h1,
        boundary_statuses=boundary,
    )


__all__: list[str] = ["GuardrailCalibration", "run_guardrail_audit"]
