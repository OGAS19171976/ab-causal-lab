-- ============================================================================
-- 09 路 ADS：实验 × 护栏 —— 判定所需的**数据**（不含判定本身）
-- ============================================================================
-- 边界刻意划在这里：
--   * 这一层只给"每臂的 n / Σx / Σx²"（可加量），
--   * **方向与容忍度不在这里** —— 它们是**声明**，属于注册表
--     （``ExperimentRecord.guardrail_specs``），属于"看结果之前定下来的东西"。
-- 把阈值放进 ADS 是个诱人的偷懒：数仓是数据层，它不该持有业务口径；
-- 而且一旦落进表里，改阈值就得重跑数仓 —— 那会让"事后挑阈值"变得很容易。
--
-- 平台侧拿到这三个量之后走与主指标同一套 Welch 推断，判定规则见
-- ``ablab/platform/guardrails.py``。
-- ============================================================================

CREATE OR REPLACE TABLE ads_experiment_guardrail_result AS
SELECT
    d.experiment,
    d.guardrail,
    d.variant,
    SUM(d.user_cnt)      AS user_cnt,
    SUM(d.value_sum)     AS value_sum,
    SUM(d.value_sq_sum)  AS value_sq_sum,
    SUM(d.value_sum) / NULLIF(SUM(d.user_cnt), 0) AS value_mean
FROM dws_experiment_guardrail_daily d
GROUP BY d.experiment, d.guardrail, d.variant;
