"""Alpha 消耗函数（alpha spending function）。

问题
----
固定时点检验的 5% 只在"只看一次"时成立。M0 用仿真标定了一个**常数边界**，
它确实能把 I 类错误压回 5%，但那个数字是凑出来的，换一个观察次数就得重标定。

M2 换成有解析保证的做法：预先声明一个**消耗函数** ``alpha*(t)`` ——
到信息量比例 ``t`` 为止，最多允许累计花掉多少 I 类错误。
各次查看的边界由它反解出来（见 ``boundaries`` 模块）。

任何满足 ``alpha*(0) = 0``、``alpha*(1) = alpha`` 且单调不减的函数都合法。
常用的三种：

``obrien_fleming``
    ``alpha*(t) = 2(1 - Phi(z_{1-alpha/2} / sqrt(t)))``
    前期极其保守（几乎不可能提前停），末期边界接近名义水平。
    医药行业默认选择：它保住了最终那次检验，代价是很难提前结束。

``pocock``
    ``alpha*(t) = alpha * ln(1 + (e-1) t)``
    各次查看的边界近似相等，容易提前停；但如果最终才出结论，
    临界值要比 OBF 严得多（K=5 时约 2.41 vs 2.04）。

``kim_demets(rho)``
    ``alpha*(t) = alpha * t**rho``，``rho=1`` 是线性（最激进），
    ``rho=3`` 已很接近 OBF。用它可以连续地在两者之间调。

**信息量比例 ``t`` 不一定等于时间比例。** 对均值类指标，
``t`` 应该是累计样本量之比；若指标方差随时间变化，要按信息量
（1/方差）来算。这里默认按样本量线性推进。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

__all__ = [
    "SpendingFunction",
    "obrien_fleming",
    "pocock",
    "linear",
    "kim_demets",
    "get_spending",
    "SPENDING_FUNCTIONS",
]


@dataclass(frozen=True)
class SpendingFunction:
    """一个 alpha 消耗函数及其名字。"""

    name: str
    func: Callable[[np.ndarray, float], np.ndarray]

    def __call__(self, t, alpha: float) -> np.ndarray:
        return np.asarray(self.func(np.asarray(t, dtype=float), float(alpha)), dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        return f"SpendingFunction({self.name!r})"


def obrien_fleming(t, alpha: float) -> np.ndarray:
    """O'Brien-Fleming 型消耗函数（Lan-DeMets 形式）。"""
    t = np.clip(np.asarray(t, dtype=float), 1e-12, 1.0)
    z = stats.norm.isf(alpha / 2)
    return 2.0 * stats.norm.sf(z / np.sqrt(t))


def pocock(t, alpha: float) -> np.ndarray:
    """Pocock 型消耗函数。"""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return alpha * np.log1p((np.e - 1.0) * t)


def linear(t, alpha: float) -> np.ndarray:
    """线性消耗（Kim-DeMets rho=1），最激进的一种。"""
    return alpha * np.clip(np.asarray(t, dtype=float), 0.0, 1.0)


def kim_demets(rho: float) -> SpendingFunction:
    """Kim-DeMets 幂族：``rho=1`` 线性，``rho=3`` 接近 OBF。"""
    if rho <= 0:
        raise ValueError(f"rho 必须为正，收到 {rho}")

    def _func(t, alpha: float) -> np.ndarray:
        return alpha * np.clip(np.asarray(t, dtype=float), 0.0, 1.0) ** rho

    return SpendingFunction(f"kim-demets(rho={rho})", _func)


SPENDING_FUNCTIONS: dict[str, SpendingFunction] = {
    "obf": SpendingFunction("O'Brien-Fleming", obrien_fleming),
    "pocock": SpendingFunction("Pocock", pocock),
    "linear": SpendingFunction("linear (Kim-DeMets rho=1)", linear),
    "kim-demets-2": kim_demets(2.0),
    "kim-demets-3": kim_demets(3.0),
}


def get_spending(name: str | SpendingFunction) -> SpendingFunction:
    """按名字取消耗函数；直接传对象则原样返回。"""
    if isinstance(name, SpendingFunction):
        return name
    try:
        return SPENDING_FUNCTIONS[name]
    except KeyError:
        raise KeyError(
            f"未知的消耗函数 {name!r}；可选 {sorted(SPENDING_FUNCTIONS)}，"
            "或直接传入 SpendingFunction 对象"
        ) from None
