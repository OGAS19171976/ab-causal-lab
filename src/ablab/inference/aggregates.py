"""可加统计量：足以还原均值、方差、协方差的充分统计量。

一个数据结构同时支撑 M1 的三个新方法，因为它们要的原料是同一批：

============  ==================  ==================
方法          x（协变量/分母）      y（结果/分子）
============  ==================  ==================
post-only     —                   后置指标
CUPED         前置指标             后置指标
比值指标 delta 分母（曝光/会话）     分子（点击/转化）
============  ==================  ==================

``merge`` 的可加性正是数仓分层能成立的原因：DWS 按天汇总的六个量
可以安全地再汇总成实验级，不需要回扫明细。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AggregateStats"]


@dataclass(frozen=True)
class AggregateStats:
    """六个可加量：n 与 Σx, Σy, Σx², Σy², Σxy。

    ``ddof=1`` 的样本方差/协方差可以直接从这六个量还原，
    因此这一层不需要落任何不可加的字段。
    """

    n: int
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def from_arrays(cls, y, x=None) -> "AggregateStats":
        """从逐单元明细构造。``x`` 省略时视为 post-only（x 全置 0）。

        两列都按**成对有效**过滤：任一列为 NaN/Inf 的单元整条丢弃，
        避免两列样本量不一致从而把协方差算错。
        """
        yv = np.asarray(y, dtype=float).ravel()
        if x is None:
            xv = np.zeros_like(yv)
            keep = np.isfinite(yv)
        else:
            xv = np.asarray(x, dtype=float).ravel()
            if xv.size != yv.size:
                raise ValueError(f"x 与 y 长度不一致：{xv.size} vs {yv.size}")
            keep = np.isfinite(yv) & np.isfinite(xv)

        yv, xv = yv[keep], xv[keep]
        n = int(yv.size)
        if n == 0:
            raise ValueError("过滤 NaN 后有效样本为空")

        return cls(
            n=n,
            sum_x=float(xv.sum()),
            sum_y=float(yv.sum()),
            sum_xx=float((xv * xv).sum()),
            sum_yy=float((yv * yv).sum()),
            sum_xy=float((xv * yv).sum()),
        )

    @classmethod
    def from_outcomes(cls, y) -> "AggregateStats":
        return cls.from_arrays(y, None)

    @classmethod
    def from_sums(
        cls, n: int, sum_x: float, sum_y: float, sum_xx: float, sum_yy: float, sum_xy: float
    ) -> "AggregateStats":
        """从数仓 ADS 表直接构造（列名一一对应）。"""
        return cls(int(n), float(sum_x), float(sum_y), float(sum_xx), float(sum_yy), float(sum_xy))

    # -- 可加性 ------------------------------------------------------------ #
    def merge(self, *others: "AggregateStats") -> "AggregateStats":
        """合并若干组统计量 —— 这就是"可加"的含义。"""
        n = self.n
        sx, sy, sxx, syy, sxy = self.sum_x, self.sum_y, self.sum_xx, self.sum_yy, self.sum_xy
        for o in others:
            n += o.n
            sx += o.sum_x
            sy += o.sum_y
            sxx += o.sum_xx
            syy += o.sum_yy
            sxy += o.sum_xy
        return AggregateStats(n, sx, sy, sxx, syy, sxy)

    def __add__(self, other: "AggregateStats") -> "AggregateStats":
        return self.merge(other)

    # -- 还原 -------------------------------------------------------------- #
    @property
    def mean_x(self) -> float:
        return self.sum_x / self.n if self.n else float("nan")

    @property
    def mean_y(self) -> float:
        return self.sum_y / self.n if self.n else float("nan")

    @property
    def var_x(self) -> float:
        """样本方差 (ddof=1)。n < 2 时为 nan。"""
        if self.n < 2:
            return float("nan")
        return (self.sum_xx - self.sum_x**2 / self.n) / (self.n - 1)

    @property
    def var_y(self) -> float:
        if self.n < 2:
            return float("nan")
        return (self.sum_yy - self.sum_y**2 / self.n) / (self.n - 1)

    @property
    def cov_xy(self) -> float:
        """样本协方差 (ddof=1)。"""
        if self.n < 2:
            return float("nan")
        return (self.sum_xy - self.sum_x * self.sum_y / self.n) / (self.n - 1)

    @property
    def corr_xy(self) -> float:
        vx, vy = self.var_x, self.var_y
        if not np.isfinite(vx) or not np.isfinite(vy) or vx <= 0 or vy <= 0:
            return float("nan")
        return self.cov_xy / np.sqrt(vx * vy)

    @property
    def ratio(self) -> float:
        """合并比值 Σy / Σx —— 业务口径的 CTR / 人均订单额等。"""
        if self.sum_x == 0:
            return float("nan")
        return self.sum_y / self.sum_x

    # -- 派生：比值指标的 delta method 方差 --------------------------------- #
    def ratio_variance(self) -> float:
        """``Var(Σy/Σx)`` 的 delta method 渐近估计。

        R = Ȳ/X̄，由 delta method：

            Var(R) ≈ Var(y_i - R·x_i) / (n · X̄²)

        关键点在分子里的 ``Var(y_i - R x_i)``：它**同时**用到了 x 和 y 的方差
        以及二者的协方差。忽略协方差项（也就是"先算每人比值再求平均"的做法）
        会得到完全不同的量。
        """
        if self.n < 2 or self.mean_x == 0:
            return float("nan")
        r = self.ratio
        residual_var = self.var_y - 2.0 * r * self.cov_xy + r * r * self.var_x
        # 数值保护：理论上非负
        residual_var = max(residual_var, 0.0)
        return residual_var / (self.n * self.mean_x**2)

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        return (
            f"AggregateStats(n={self.n}, mean_x={self.mean_x:.4f}, "
            f"mean_y={self.mean_y:.4f}, corr={self.corr_xy:.4f})"
        )
