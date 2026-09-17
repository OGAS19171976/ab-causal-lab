"""``ablab.reporting``：报告里只留**在别的机器上也能复现**的内容。

这条规则看起来很琐碎，但它决定了一件大事能不能被验证：
**"重跑一次，报告没变"** 是最硬的一条可复现性证据，
而只要报告里带上耗时或绝对路径，这句话就永远无法用 git diff 来核对。
实测过：加这个过滤器之前，同一份代码重跑产生的 diff 是 16 增 14 删，
全是 "耗时 61s -> 62s"，真正的数字变化会被噪声淹掉。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ablab.reporting import for_report, relativize_paths, report_text, strip_timings

ROOT = Path(__file__).resolve().parents[1]


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
                if relativize_paths(line, root=ROOT) != line:
                    offenders.append(f"{path.name}:{lineno} 有绝对路径：{line.strip()[:60]}")
        assert not offenders, "报告里混进了机器相关信息（重跑就会变成 diff 噪声）：\n" + "\n".join(
            f"  - {o}" for o in offenders[:10]
        )
