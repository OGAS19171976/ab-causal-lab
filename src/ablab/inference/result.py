"""推断层的统一返回结构。

设计原则：**任何一次分析都必须连诊断一起返回**。

只给一个 p 值的分析结果是不可信的 —— 分组比例对不对（SRM）、
样本量够不够（功效）、前置协变量平不平衡，这些决定了那个 p 值
能不能被解读。把 ``diagnostics`` 做成一等公民而不是"可选日志"，
是这个框架和"随手跑个 t 检验"的分界线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Diagnostic", "Estimate", "Status"]

Status = Literal["pass", "warn", "fail", "info"]

_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}


@dataclass(frozen=True)
class Diagnostic:
    """单项体检结果。"""

    name: str
    status: Status
    message: str
    statistic: float | None = None
    p_value: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """只有 ``fail`` 会让整个实验结论不可用。"""
        return self.status != "fail"

    def __str__(self) -> str:  # pragma: no cover - 展示用
        bits = [f"[{_ICON[self.status]}] {self.name}: {self.message}"]
        if self.statistic is not None:
            bits.append(f"stat={self.statistic:.4f}")
        if self.p_value is not None:
            bits.append(f"p={self.p_value:.4g}")
        return "  ".join(bits)


@dataclass(frozen=True)
class Estimate:
    """一次实验对比的完整结果。"""

    metric: str
    variant: str
    control: str
    method: str

    absolute_effect: float
    relative_effect: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float

    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    alpha: float = 0.05

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def is_healthy(self) -> bool:
        """所有诊断都没有 fail 才算可信。"""
        return all(d.ok for d in self.diagnostics)

    def diagnostics_of(self, name: str) -> Diagnostic | None:
        return next((d for d in self.diagnostics if d.name == name), None)

    def summary(self) -> str:
        lo, hi = self.ci_low, self.ci_high
        sign = "+" if self.absolute_effect >= 0 else ""
        return (
            f"{self.metric} | {self.variant} vs {self.control} | {self.method}\n"
            f"  n = {self.n_treatment:,} / {self.n_control:,}\n"
            f"  mean = {self.mean_treatment:.4f} / {self.mean_control:.4f}\n"
            f"  effect = {sign}{self.absolute_effect:.4f} "
            f"({sign}{self.relative_effect * 100:.2f}%)\n"
            f"  {int((1 - self.alpha) * 100)}% CI = [{lo:.4f}, {hi:.4f}]\n"
            f"  p = {self.p_value:.4g}   significant = {self.significant}\n"
        )

    def report(self) -> str:
        lines = [self.summary(), "  diagnostics:"]
        if not self.diagnostics:
            lines.append("    (none)")
        for d in self.diagnostics:
            lines.append(f"    {d}")
        return "\n".join(lines)
