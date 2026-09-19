"""护栏指标：**真正分析**它们，并给出"要不要停实验"的判据。

为什么单独一个模块
------------------
护栏与主指标的判定语义**不一样**，混在一起写迟早会串味：

* 主指标问"有没有效果"（双侧、显著就行）；
* 护栏问"**有没有造成超过容忍度的伤害**"（单侧、对着一个**事先声明的阈值**）。

Kohavi 那本书里护栏触发是**停实验的理由**，不是参考信息。所以这里的输出不是
"一个 p 值"，而是每条护栏一个判定 + 一句能不能继续的建议。

三条不可动摇的规则
------------------
1. **没有事先声明方向与阈值，就不判断**。``direction`` 与 ``max_harm`` 必须显式给出：
   从指标名猜"latency 越低越好"看着很聪明，但同一个名字在不同业务里含义可以相反
   （"延迟"是伤害，"停留时长"是收益），猜错的方向会**静默地把伤害读成改善**。
   没声明就返回 ``unknown``（理由写清楚），**绝不返回 pass** ——
   "没数据/没声明"与"检查通过"是两件事，混起来就是这一整块最危险的地方。
2. **多重比较要校正**。同时看 K 个护栏就是 K 次检验；不校正时"至少一个误报"的
   概率随 K 上升。这里用 Bonferroni（``alpha/K``），因为要控的是
   "**误判某个护栏有害从而错误停机**"这一类错误。
3. **判定看的是"有把握地超出容忍度"**，不是点估计超了就叫停：
   ``fail`` 要求伤害的**置信下界**已经超过 ``max_harm``。
   点估计超了但证据不足记 ``warn``（继续观察），这也让"停机"这个动作更保守。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..inference.aggregates import AggregateStats
from ..inference.welch import welch_inference

#: 方向：``lower_is_better`` 表示**越低越好**（延迟、崩溃率），
#: 变高就是伤害；``higher_is_better`` 反之（收入、留存）。
Direction = Literal["lower_is_better", "higher_is_better"]
DIRECTIONS: tuple[str, ...] = ("lower_is_better", "higher_is_better")

#: 判定。
#: * ``pass``    在容忍度内
#: * ``warn``    点估计超出容忍度，但证据不足以断言（继续观察）
#: * ``fail``    有把握地超出容忍度 —— **建议停止实验**
#: * ``unknown`` 没数据、或没声明方向/阈值 —— **既不是通过也不是失败**
GuardrailStatus = Literal["pass", "warn", "fail", "unknown"]


@dataclass(frozen=True)
class GuardrailSpec:
    """护栏的**声明**：名字、方向、容忍度。

    这三个都必须在看到结果**之前**定下来 —— 事后挑阈值等于没有护栏。
    """

    name: str
    #: 越低越好还是越高越好。**必填**，不从名字猜。
    direction: str = ""
    #: 允许的最大相对劣化（如 0.05 = 5%）。**必填**。
    max_harm: float | None = None
    #: **仅演示数据用**：合成时注入的真实伤害（与 ``true_lift`` 同一个性质）。
    #: 真实数仓没有这一列。
    demo_harm: float = 0.0

    def is_declared(self) -> bool:
        return self.direction in DIRECTIONS and self.max_harm is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "max_harm": self.max_harm,
            "demo_harm": self.demo_harm,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GuardrailSpec":
        return cls(
            name=str(raw.get("name", "")),
            direction=str(raw.get("direction", "")),
            max_harm=None if raw.get("max_harm") is None else float(raw["max_harm"]),
            demo_harm=float(raw.get("demo_harm", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class GuardrailOutcome:
    """一条护栏的判定。``harm`` 已经按方向折算成"**伤害**"（正数=变差）。"""

    name: str
    status: GuardrailStatus
    reason: str
    direction: str = ""
    max_harm: float | None = None
    treated_mean: float = float("nan")
    control_mean: float = float("nan")
    harm: float = float("nan")
    relative_harm: float = float("nan")
    harm_ci_low: float = float("nan")
    harm_ci_high: float = float("nan")
    p_value: float = float("nan")
    n_treated: int = 0
    n_control: int = 0
    #: 校正后的显著性水平（K 个护栏用 Bonferroni）
    alpha_adjusted: float = float("nan")

    def line(self) -> str:
        if self.status == "unknown":
            return f"  {self.name:<18} unknown  {self.reason}"
        return (
            f"  {self.name:<18} {self.status:<7} 伤害 {self.harm:+.4f}"
            f"（相对 {self.relative_harm:+.2%}，容忍 {self.max_harm:.2%}）"
            f"  CI[{self.harm_ci_low:+.4f}, {self.harm_ci_high:+.4f}]"
        )


@dataclass
class GuardrailReport:
    """所有护栏的判定汇总，以及"能不能继续"的结论。"""

    outcomes: list[GuardrailOutcome] = field(default_factory=list)
    alpha: float = 0.05

    @property
    def failed(self) -> list[GuardrailOutcome]:
        return [o for o in self.outcomes if o.status == "fail"]

    @property
    def warned(self) -> list[GuardrailOutcome]:
        return [o for o in self.outcomes if o.status == "warn"]

    @property
    def unknown(self) -> list[GuardrailOutcome]:
        return [o for o in self.outcomes if o.status == "unknown"]

    @property
    def missing_specs(self) -> list[GuardrailOutcome]:
        """**用户要补的**：声明了名字但没给方向/容忍度。"""
        return [o for o in self.unknown if not o.direction or o.max_harm is None]

    @property
    def missing_data(self) -> list[GuardrailOutcome]:
        """**平台要补的**：规格齐了但没有数据（例如数仓还没有护栏表）。"""
        return [o for o in self.unknown if o.direction and o.max_harm is not None]

    @property
    def n_declared(self) -> int:
        return len(self.outcomes)

    @property
    def n_analysed(self) -> int:
        return sum(1 for o in self.outcomes if o.status != "unknown")

    @property
    def verdict(self) -> str:
        """``stop`` / ``watch`` / ``ok`` / ``unknown``。"""
        if self.failed:
            return "stop"
        if self.warned:
            return "watch"
        if self.n_analysed == 0:
            return "unknown"
        return "ok"

    @property
    def check_status(self) -> str:
        """映射到报告里的检查项状态（``pass`` / ``warn`` / ``fail`` / ``info``）。

        ``unknown`` 映射成 ``info`` 而不是 ``pass``：没有数据时"检查通过"
        是一句没有依据的话（而 ``info`` 会让它出现在报告里、不改变 health）。
        """
        if self.verdict != "unknown":
            return {"stop": "fail", "watch": "warn", "ok": "pass"}[self.verdict]
        # 两种"未知"要分开：
        #   * 没声明规格 = **这一次**就能补的事（补 direction + max_harm）-> warn
        #   * 有规格没数据 = 平台级缺口（数仓还没有护栏表）-> info
        #     它对每个实验都一样，永远 warn 就等于没有告警（第 31 条那条教训）。
        return "warn" if self.missing_specs else "info"

    @property
    def recommendation(self) -> str:
        if self.verdict == "stop":
            names = "、".join(o.name for o in self.failed)
            return f"**建议停止实验**：{names} 的伤害已越过事先声明的容忍度"
        if self.verdict == "watch":
            names = "、".join(o.name for o in self.warned)
            return f"继续观察：{names} 的点估计超出容忍度，但证据不足以断言"
        if self.verdict == "unknown":
            if self.missing_specs:
                names = "、".join(o.name for o in self.missing_specs)
                return (
                    f"**无法判断**：{names} 只声明了名字、没给 direction + max_harm ——"
                    "补上声明就能判定（这不等于通过）"
                )
            names = "、".join(o.name for o in self.missing_data)
            return (
                f"**无法判断**：{names} 声明齐全但没有数据（数仓路径还没有护栏表）——"
                "这不等于通过"
            )
        return "全部护栏在容忍度内"

    def summary(self) -> str:
        lines = [
            f"护栏分析（{self.n_declared} 条声明，{self.n_analysed} 条可判定，"
            f"名义 alpha={self.alpha}）"
        ]
        lines += [o.line() for o in self.outcomes]
        lines.append(f"  判定：{self.verdict} —— {self.recommendation}")
        return "\n".join(lines)


def analyse_guardrails(
    specs: Sequence[GuardrailSpec],
    series: dict[str, dict[str, AggregateStats]],
    *,
    treated: str,
    control: str,
    alpha: float = 0.05,
) -> GuardrailReport:
    """逐条判定护栏。

    ``series[name][variant]`` 是该护栏在该臂上的**可加充分统计量**
    （与主指标同一套 ``AggregateStats``，所以护栏可以走相同的推断路径）。

    判定规则（写在这里，也写在 README 里）：

    1. 把两臂之差按 ``direction`` 折算成**伤害** ``harm``；
    2. ``fail``：伤害的**置信下界** > ``max_harm``；
    3. ``warn``：点估计 > ``max_harm`` 但下界没超过；
    4. ``pass``：点估计 ≤ ``max_harm``；
    5. 没数据 / 没声明方向或阈值 → ``unknown``（**不是** pass）。

    多重比较：K 条可判定的护栏用 Bonferroni（``alpha/K``）。
    """
    declared = list(specs)
    analysable = [
        s for s in declared if s.is_declared() and s.name in series
        and treated in series[s.name] and control in series[s.name]
    ]
    k = max(len(analysable), 1)
    adjusted = alpha / k

    outcomes: list[GuardrailOutcome] = []
    for spec in declared:
        if not spec.is_declared():
            outcomes.append(
                GuardrailOutcome(
                    name=spec.name,
                    status="unknown",
                    direction=spec.direction,
                    max_harm=spec.max_harm,
                    reason=(
                        "没有声明方向或容忍度（`direction` + `max_harm` 必须显式给出，"
                        "不从指标名猜）—— 无法判断，**不等于通过**"
                    ),
                )
            )
            continue
        arms = series.get(spec.name)
        if not arms or treated not in arms or control not in arms:
            outcomes.append(
                GuardrailOutcome(
                    name=spec.name,
                    status="unknown",
                    direction=spec.direction,
                    max_harm=spec.max_harm,
                    reason="没有该护栏的数据 —— 无法判断，**不等于通过**",
                )
            )
            continue

        t_stats = arms[treated]
        c_stats = arms[control]
        t_mean = _mean(t_stats)
        c_mean = _mean(c_stats)
        delta = t_mean - c_mean  # 处置组 - 对照组
        # 折算成"伤害"：越低越好的指标，变高才是伤害
        sign = 1.0 if spec.direction == "lower_is_better" else -1.0
        harm = sign * delta

        var_t = _var(t_stats)
        var_c = _var(c_stats)
        inf = welch_inference(
            harm,
            n_treatment=t_stats.n,
            var_treatment=var_t,
            n_control=c_stats.n,
            var_control=var_c,
            alpha=adjusted,
        )
        lo, hi = inf.interval(harm)
        base = abs(c_mean)
        relative = harm / base if base > 1e-12 else float("nan")

        assert spec.max_harm is not None  # is_declared() 已保证
        if lo > spec.max_harm:
            status: GuardrailStatus = "fail"
            reason = (
                f"伤害的 {1 - adjusted:.1%} 置信下界 {lo:+.4f} 已超过容忍度 "
                f"{spec.max_harm:+.4f}"
            )
        elif harm > spec.max_harm:
            status = "warn"
            reason = (
                f"点估计 {harm:+.4f} 超过容忍度 {spec.max_harm:+.4f}，"
                f"但置信下界 {lo:+.4f} 还没超过 —— 证据不足以断言"
            )
        else:
            status = "pass"
            reason = f"点估计 {harm:+.4f} 在容忍度 {spec.max_harm:+.4f} 之内"

        outcomes.append(
            GuardrailOutcome(
                name=spec.name,
                status=status,
                reason=reason,
                direction=spec.direction,
                max_harm=spec.max_harm,
                treated_mean=t_mean,
                control_mean=c_mean,
                harm=harm,
                relative_harm=relative,
                harm_ci_low=float(lo),
                harm_ci_high=float(hi),
                p_value=inf.p_value,
                n_treated=t_stats.n,
                n_control=c_stats.n,
                alpha_adjusted=adjusted,
            )
        )

    return GuardrailReport(outcomes=outcomes, alpha=alpha)


def _mean(stats: AggregateStats) -> float:
    return float(stats.sum_y / stats.n) if stats.n else float("nan")


def _var(stats: AggregateStats) -> float:
    """样本方差（ddof=1），从可加量还原。"""
    if stats.n < 2:
        return 0.0
    mean = stats.sum_y / stats.n
    return float(max(stats.sum_yy / stats.n - mean * mean, 0.0) * stats.n / (stats.n - 1))
