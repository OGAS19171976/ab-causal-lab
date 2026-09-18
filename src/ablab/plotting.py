"""图表统一样式与中文字体探测。

matplotlib 默认字体不含 CJK，直接写中文标题会渲染成一个个方块（tofu）。
这里在导入时探测系统里可用的中文字体；探测不到就自动切到英文标签，
保证图表在任何机器上都能跑出可读的结果。

另外把 matplotlib 的**缓存目录改到项目内**（``build/.mplconfig``）。
默认它写在 ``~/.matplotlib``，在受限（沙箱）环境里那里常常不可写，
每次导入都会刷一串 ``Could not save font_manager cache ... Permission denied``。
放到项目内既安静，也让整个仓库自包含（删掉 build/ 就等于重置）。
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / "build" / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 无头环境：只出文件，不弹窗

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

__all__ = ["setup_style", "label", "HAS_CJK", "save", "plt", "bin_edges"]

_CJK_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)

HAS_CJK = False


def setup_style() -> bool:
    """探测中文字体并设置全局样式，返回是否成功启用中文。"""
    global HAS_CJK

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_CANDIDATES if name in available), None)

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.frameon": False,
        }
    )

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号不要变方块

    HAS_CJK = chosen is not None
    return HAS_CJK


def label(zh: str, en: str) -> str:
    """有中文字体就用中文，否则退回英文。"""
    return zh if HAS_CJK else en


def bin_edges(values) -> list[float]:
    """给 ``Axes.hist`` 用的分箱边界（``np.linspace(...)`` 的结果）。

    为什么要在这里转一次 ``list``：matplotlib 的类型桩把 ``bins`` 声明成
    ``int | Sequence[float] | str | None``，而 numpy 的 ``ndarray`` 在类型系统里
    **不算 ``Sequence``**（运行时当然完全可以用）。第一版在 6 个调用点各写一个
    ``# type: ignore`` —— 那等于在 6 个地方重复同一句"这个桩不准"。
    收在出图这一层，既只说一次，也真的表达了意图：``.tolist()`` 得到的
    就是"一串边界值"，而这正是 ``bins`` 要的东西。
    """
    return np.asarray(values, dtype=float).tolist()


def save(fig, path) -> None:
    """保存并关闭，避免长时间循环里积累 figure 造成内存泄漏。"""
    fig.savefig(path)
    plt.close(fig)
