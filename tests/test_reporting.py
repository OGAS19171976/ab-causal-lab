"""``ablab.reporting``：报告里只留**在别的机器上也能复现**的内容。

这条规则看起来很琐碎，但它决定了一件大事能不能被验证：
**"重跑一次，报告没变"** 是最硬的一条可复现性证据，
而只要报告里带上耗时或绝对路径，这句话就永远无法用 git diff 来核对。
实测过：加这个过滤器之前，同一份代码重跑产生的 diff 是 16 增 14 删，
全是 "耗时 61s -> 62s"，真正的数字变化会被噪声淹掉。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ablab.reporting import (
    compare_report_texts,
    for_report,
    relativize_paths,
    report_text,
    strip_timings,
)

ROOT = Path(__file__).resolve().parents[1]

#: 耗时的**形状**：数字（可带小数点、后可跟一个空格）紧跟一个 s，且两侧不能再粘字母/点。
#: 用 ``(?<![\w.])`` / ``(?![\w])`` 卡住边界，避免把 ``1.5e-3``、
#: ``5 samples``、``fig28_...`` 这类真数字判成耗时。
#: 允许一个空格是因为 ``71 s`` 这种写法同样会进报告 —— 它误报的概率极低
#: （要恰好是"数字 空格 s"且后面接边界），漏掉它的代价却是每次重跑 diff 一行。
_DURATION_TOKEN = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?s(?![\w])")


class TestStripTimings:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("总耗时 66.7s", ""),
            ("零效应 400 个 salt，耗时 62s", "零效应 400 个 salt"),
            ("300 个 salt，真实效应为零，耗时 49s", "300 个 salt，真实效应为零"),
            ("耗时 6s（固定同一个分组掩码）", "（固定同一个分组掩码）"),
            ("总耗时 178s；图：fig28.png", "图：fig28.png"),
            ("口径        FPR", "口径        FPR"),
        ],
    )
    def test_patterns(self, raw, expected):
        assert strip_timings(raw) == expected

    def test_is_idempotent(self):
        once = strip_timings("零效应 400 个 salt，耗时 62s")
        assert strip_timings(once) == once

    def test_keeps_the_meaningful_part(self):
        """剥掉耗时不能顺手把同一行的**内容**也剥掉。"""
        line = "10 个 salt，耗时 3s，覆盖率 0.9480"
        assert "10 个 salt" in strip_timings(line)
        assert "覆盖率 0.9480" in strip_timings(line)


class TestRelativizePaths:
    def test_project_internal_path_becomes_relative(self):
        raw = str(ROOT / "build" / "warehouse.duckdb")
        assert relativize_paths(f"数仓: {raw}", root=ROOT) == "数仓: build/warehouse.duckdb"

    def test_forward_slashes_too(self):
        raw = str(ROOT).replace("\\", "/") + "/reports/x.md"
        assert relativize_paths(raw, root=ROOT) == "reports/x.md"

    def test_outside_path_becomes_placeholder(self):
        out = relativize_paths("缓存在 C:\\Users\\someone\\.cache\\uv", root=ROOT)
        assert "someone" not in out
        assert "<绝对路径>" in out

    def test_plain_text_untouched(self):
        assert relativize_paths("没有任何路径的一行", root=ROOT) == "没有任何路径的一行"

    def test_posix_semantics_would_leak_a_windows_path(self):
        """把"CI 上那次失败"的**机制**钉住（本机是 Windows，复现不出那个现象）。

        现象（CI / Ubuntu 上真实发生）：
        ``relativize_paths(r"C:\\Users\\someone\\.cache\\uv", root=...)`` 原样返回了，
        用户名照旧留在正文里 —— ``<绝对路径>`` 兜底没生效。

        原因就在这两行：在 POSIX 上，``C:\\Users\\...`` 不是绝对路径，
        于是 ``Path(raw).resolve()`` 把它接到 cwd 上，
        ``.relative_to(root)`` 竟然**成功**（因为 cwd 就是 root）。
        修法是先问 ``is_absolute()`` 再比 —— 这两条断言就是那个前提。
        """
        import pathlib

        raw = r"C:\Users\someone\.cache\uv"
        assert not pathlib.PurePosixPath(raw).is_absolute(), (
            "POSIX 上 C:\\... 必须是相对路径 —— 这正是当初 relative_to 会成功的原因"
        )
        assert pathlib.PureWindowsPath(raw).is_absolute()

    def test_sub_step_guards_with_is_absolute(self):
        """并且实现里**确实**用了那道闸门（否则上面的机制仍会咬人）。

        源码级断言不算优雅，但这里它检查的是一条**本机跑不到的分支**（POSIX 语义）：
        没有它，这个 bug 只会在 CI 上复现，而"只在 CI 上红"是最难查的一类。
        """
        import inspect

        import ablab.reporting as reporting

        source = inspect.getsource(reporting.relativize_paths)
        assert "is_absolute()" in source, (
            "relativize_paths 的兜底分支必须先确认候选路径在本平台上是绝对路径，"
            "否则在 POSIX 上 Windows 路径会被原样放回正文（用户名泄露）"
        )


class TestForReport:
    def test_drops_lines_that_become_empty(self):
        out = for_report(["总耗时 12.3s", "", "口径  FPR"], root=ROOT)
        assert out == ["", "口径  FPR"]

    def test_keeps_blank_lines_that_were_always_blank(self):
        """空行是排版，不是噪声。

        第一版写成"结果为空就丢掉"，把日志里本来就有的空行也删了 ——
        报告的段落结构被压平（m5 的 diff 一下变成 38 行），
        而根因只是少了一个"原本是否为空"的判断。
        """
        # "耗时 1s" 被丢弃后，前后两个原本就有的空行都保留 → 连续两个空行
        assert for_report(["标题", "", "耗时 1s", "", "口径  FPR"], root=ROOT) == [
            "标题",
            "",
            "",
            "口径  FPR",
        ]
        # 完全没有耗时行时，空行必须原样穿过
        assert report_text(["a", "", "b"], root=ROOT) == "a\n\nb"

    def test_only_drops_lines_emptied_by_the_filter(self):
        out = for_report(["", "总耗时 5s", "", "内容"], root=ROOT)
        assert out == ["", "", "内容"]

    def test_realistic_report_snippet_is_stable(self):
        """同一段日志跑两次（只有耗时不同）必须产出同一份报告正文。"""
        a = [
            "1. 整条管道：400 个 salt",
            "耗时 61s，每次 20,000 用户",
            "naive FPR = 0.0400",
            "总耗时 178s；图：fig28.png",
        ]
        b = [
            "1. 整条管道：400 个 salt",
            "耗时 63s，每次 20,000 用户",
            "naive FPR = 0.0400",
            "总耗时 181s；图：fig28.png",
        ]
        assert report_text(a, root=ROOT) == report_text(b, root=ROOT)

    def test_real_reports_have_no_timings_or_abs_paths(self):
        """把规则应用到真报告上：reports/*.md 里不该有耗时或绝对路径。

        这条是端到端的 —— 它检查的是**已经写出来的产物**，
        所以万一某个脚本绕过了过滤器（比如自己 write_text），这里会立刻暴露。
        """
        offenders: list[str] = []
        for path in sorted((ROOT / "reports").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if "耗时" in line:
                    offenders.append(f"{path.name}:{lineno} 有耗时：{line.strip()[:60]}")
                if _DURATION_TOKEN.search(line):
                    offenders.append(
                        f"{path.name}:{lineno} 有裸耗时（数字+秒）：{line.strip()[:60]}"
                    )
                if relativize_paths(line, root=ROOT) != line:
                    offenders.append(f"{path.name}:{lineno} 有绝对路径：{line.strip()[:60]}")
        assert not offenders, "报告里混进了机器相关信息（重跑就会变成 diff 噪声）：\n" + "\n".join(
            f"  - {o}" for o in offenders[:10]
        )


class TestCompareReportTexts:
    """第二级判据：正文不变时，数值末位允许差多少。

    为什么需要它，以及为什么不能只有它 —— 都在 ``Comparison`` 的 docstring 里。
    这组测试要守住的是**两侧**：
    * 末位浮点差不能被报成失败（否则真失败会被噪声淹掉）；
    * 真变化不能被放过（耗时从 69s 变 71s 就是真变化，哪怕它也是"数字变了"）。
    """

    def test_byte_identical(self):
        text = "effect=27.2482063215\n"
        c = compare_report_texts(text, text)
        assert c.identical and c.ok and c.worst is None

    def test_last_digit_float_change_is_tolerated_but_quantified(self):
        """实测到的那一次：-3.9243251037 -> -3.9243251038（1 ulp）。"""
        c = compare_report_texts(
            "CUPED ADS汇总 effect=-3.9243251037\n", "CUPED ADS汇总 effect=-3.9243251038\n"
        )
        assert not c.identical
        assert c.text_identical
        assert c.ok, "1 ulp 的差不该判失败"
        assert c.exceeds is None
        assert c.n_numeric_diffs == 1
        # worst 是"实测最大偏差"，不区分是否超容差 —— 它的用途是**量化**，
        # 判断用途的是 exceeds。这两个字段混在一起，报告就没法既说"通过"
        # 又说"最大偏差 2.5e-11"，而后者恰恰是这条检查真正想产出的信息。
        assert c.worst is not None and c.worst.rel_diff < 1e-9
        assert "1037" in (c.first_diff or "")

    def test_scientific_notation_deviation_is_tolerated(self):
        """本身就是舍入噪声的量：相对差 50%，绝对差 1e-13 —— 必须放过。

        三条路径的"最大偏差"就是这类量（两个几乎相等的数相减）。
        只比相对值会把它判失败；``atol=1e-12`` 那一项就是为它准备的。
        worst 仍然要把这 50% 记下来 —— 因为**看的人有权知道**它不稳定。
        """
        c = compare_report_texts("最大偏差 2.274e-13\n", "最大偏差 1.137e-13\n")
        assert not c.identical and c.ok
        assert c.exceeds is None
        assert c.worst is not None and c.worst.rel_diff > 0.4

    def test_timing_leak_still_fails(self):
        """耗时从 69s 变 71s：同样是"数字变了"，但**必须失败**。

        这正是第一版逐字节判据抓到的那个漏网 —— 换成带容差的判据之后，
        它不能因为"只是数字"就被放过：相对差 2.8e-2 远大于容差 1e-9。
        """
        c = compare_report_texts(
            "真实效应 0.25 下（400 个 salt，69s）\n", "真实效应 0.25 下（400 个 salt，71s）\n"
        )
        assert not c.identical
        assert not c.ok, "耗时变化必须判失败"
        assert c.worst is not None and c.worst.rel_diff > 1e-2

    def test_sample_count_change_fails(self):
        """样本量 30,741 -> 30,742 必须失败：这不是浮点噪声，是数据变了。"""
        c = compare_report_texts("ODS 行数 = 30741\n", "ODS 行数 = 30742\n")
        assert not c.ok
        assert c.worst is not None and c.worst.rel_diff > 1e-5

    def test_wording_change_is_text_difference(self):
        c = compare_report_texts("结论：守住 5%\n", "结论：守不住 5%\n")
        assert not c.ok
        assert not c.text_identical

    def test_extra_line_is_text_difference(self):
        c = compare_report_texts("a\nFPR 0.0625\n", "a\nFPR 0.0625\nb\n")
        assert not c.ok and not c.text_identical

    def test_number_in_a_filename_is_not_a_false_alarm(self):
        c = compare_report_texts(
            "图：fig28_monitoring_estimator.png\n", "图：fig28_monitoring_estimator.png\n"
        )
        assert c.identical

    def test_integers_are_not_compared_with_float_tolerance(self):
        """整数不能被"相对 1e-9"放过 —— 1 和 2 的相对差是 0.5，本来就过得去；
        但这里更想钉住的是：**不能因为都是数字就按同一套宽松规则比**。"""
        c = compare_report_texts("n=1000000\n", "n=1000001\n")
        assert not c.ok, "整数差 1 也必须失败"
        assert c.worst is not None and c.worst.rel_diff > 1e-9

    def test_format_only_change_is_recorded_with_zero_deviation(self):
        """只改印刷位数（1.0 -> 1.000）不改变数值。

        这时它**不是**逐字节相同，但相对偏差是 0 —— 所以放行，同时把
        ``1.0 -> 1.000`` 原样打印出来。把它算成"失败"会让格式调整寸步难行；
        把它算成"逐字节相同"又是撒谎。当前行为是第三条：记为数值差异、偏差 0。
        """
        c = compare_report_texts("effect=1.0\n", "effect=1.000\n")
        assert not c.identical
        assert c.ok
        assert c.worst is not None and c.worst.rel_diff == 0.0
        assert c.worst.before == "1.0" and c.worst.after == "1.000"


class TestDurationShapeRule:
    """耗时未必带"耗时"两个字 —— 这条规则是靠实测的漏网换来的。

    过滤器只认措辞（``耗时 71s``），于是 ``run_m6_validation.py`` 里
    ``真实效应 0.25 下（400 个 salt，{t:.0f}s）`` 这种写法整行绕过它，
    m6 报告每次重跑都 diff 一行。措辞是**开放集合**（还能写成"用时 71s"
    "花了 71 秒"），形状才是封闭的：**数字紧跟秒**。
    所以断言按形状写，``_DURATION_TOKEN`` 就是那个形状。
    """

    @pytest.mark.parametrize(
        "line",
        [
            "真实效应 0.25 下（400 个 salt，71s）",
            "用时 71s",
            "花了 71 秒".replace("秒", "s"),
            "3.5s 完成",
        ],
    )
    def test_shape_is_caught(self, line):
        assert _DURATION_TOKEN.search(line)

    @pytest.mark.parametrize(
        "line",
        [
            "naive FPR = 0.0400",  # 数字后面跟的不是 s
            "n=40 个簇",  # 有空格
            "5 samples",  # s 后面还跟着字母
            "eff=1.5e-3",  # 科学计数法
            "fig28_monitoring_estimator.png",  # 像但不以数字开头
            "覆盖率 0.9480，SE 之比 0.1590",
        ],
    )
    def test_statistical_values_are_not_false_positives(self, line):
        """误报比漏报更糟：一个会把真数字判成耗时的规则没人会留着。"""
        assert not _DURATION_TOKEN.search(line)

