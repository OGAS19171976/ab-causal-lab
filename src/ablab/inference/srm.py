"""SRM（Sample Ratio Mismatch，样本比例失衡）检验。

这是实验可信度的**第一道体检**，也是大厂实验平台上线新实验时的硬性闸门。

为什么它排第一：分流是按固定权重做的，所以各组人数的比例是**已知的**。
如果观测到的比例和设计值显著不符，说明数据链路上出了系统性问题 ——
曝光日志丢包、跳转重定向丢失、爬虫过滤不均、缓存命中差异、分流代码 bug……
此时无论 p 值多漂亮，结论都不可信。业界经验：**SRM 报警的实验，
绝大多数都藏着真实的 bug，而不是运气不好。**

为什么阈值用 0.001 而不是 0.05：SRM 是一个每天每实验都要跑的常规检验，
多重比较下 0.05 会天天误报；且样本量通常很大，真实失衡会轻易击穿 0.001。
"""

from __future__ import annotations

from typing import Mapping

from scipy import stats

from .result import Diagnostic

__all__ = ["srm_check", "srm_from_weights"]

DEFAULT_ALPHA = 1e-3


def srm_check(
    counts: Mapping[str, int],
    expected_weights: Mapping[str, float],
    alpha: float = DEFAULT_ALPHA,
) -> Diagnostic:
    """卡方拟合优度检验：观测分组数 vs 设计权重。

    Parameters
    ----------
    counts:
        ``{分支名: 观测人数}``。
    expected_weights:
        ``{分支名: 设计权重}``，需要与 ``counts`` 的键完全一致。
    alpha:
        判定为失衡的显著性水平，默认 1e-3。
    """
    keys = list(counts.keys())
    missing = set(expected_weights) - set(keys)
    if missing:
        raise ValueError(f"expected_weights 中的分支在 counts 中缺失: {sorted(missing)}")

    observed = [int(counts[k]) for k in keys]
    total = sum(observed)
    if total == 0:
        return Diagnostic(
            name="SRM",
            status="fail",
            message="没有任何分流样本，无法检验（检查曝光日志是否为空）",
            statistic=None,
            p_value=None,
            detail={"counts": dict(counts)},
        )

    weight_sum = sum(expected_weights[k] for k in keys)
    expected = [expected_weights[k] / weight_sum * total for k in keys]

    # 期望频数太小时卡方近似不可靠
    if min(expected) < 5:
        return Diagnostic(
            name="SRM",
            status="warn",
            message=(
                f"存在期望频数 < 5 的分支（最小 {min(expected):.1f}），"
                "卡方近似不可靠，建议改用精确检验或继续积累样本"
            ),
            statistic=None,
            p_value=None,
            detail={"observed": dict(zip(keys, observed)), "expected": dict(zip(keys, expected))},
        )

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # p 值极小（< 1e-300）时浮点会下溢为 0，改用 chi2 分位点判断
    if p_value == 0.0:
        p_value = float(stats.chi2.sf(chi2, df=len(keys) - 1))

    max_dev = max(
        abs(o - e) / e for o, e in zip(observed, expected)
    )
    status = "fail" if p_value < alpha else "pass"
    if status == "fail":
        message = (
            f"样本比例失衡：卡方={chi2:.1f}，p={p_value:.3g} < {alpha:g}，"
            f"最大相对偏差 {max_dev:.2%}。**该实验结果不可信，先查数据链路，不要解读指标。**"
        )
    else:
        message = (
            f"分组比例与设计一致（最大相对偏差 {max_dev:.2%}，p={p_value:.3g}）"
        )

    return Diagnostic(
        name="SRM",
        status=status,
        message=message,
        statistic=float(chi2),
        p_value=float(p_value),
        detail={
            "observed": dict(zip(keys, observed)),
            "expected": {k: round(e, 2) for k, e in zip(keys, expected)},
            "max_relative_deviation": max_dev,
        },
    )


def srm_from_weights(
    counts: Mapping[str, int], alpha: float = DEFAULT_ALPHA
) -> Diagnostic:
    """权重由观测总数反推的版本，仅用于无法拿到设计权重的历史数据排查。

    注意：这**不是**真正的 SRM 检验。真正的 SRM 必须拿设计权重比，
    因为反推权重后卡方恒等于 0，什么都检验不出来。这里仅保留接口占位，
    永远返回 ``info``，提醒调用方去补设计权重。
    """
    total = sum(counts.values())
    uniform = {k: 1.0 / len(counts) for k in counts} if counts else {}
    diag = srm_check(counts, uniform, alpha=alpha) if counts else None
    return Diagnostic(
        name="SRM(占位)",
        status="info",
        message=(
            "缺少设计权重，无法做真正的 SRM 检验。请从实验配置里取权重后调用 srm_check()。"
        ),
        statistic=None,
        p_value=None,
        detail={"observed": dict(counts), "total": total, "uniform_baseline": diag.p_value if diag else None},
    )
