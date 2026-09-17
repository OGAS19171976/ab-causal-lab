-- ============================================================================
-- 04 · ADS SRM 体检表
-- ============================================================================
-- SRM（样本比例失衡）是实验可信度的第一道闸门，必须在数仓侧就自动算出来，
-- 而不是等分析师想起来才手动跑。
--
-- 卡方统计量 = Σ (obs - exp)² / exp，其中 exp 来自**实验配置的设计权重**。
-- 自由度 = 分支数 - 1。多项卡方分量可加，所以这里输出每个分支的分量，
-- 检验本身放在 Python 侧完成（分位点与 p 值不适合写在 SQL 里）。
--
-- 阈值用 0.001 而不是 0.05：这是每天每个实验都跑的常规检验，
-- 多重比较下 0.05 会天天误报。
-- ============================================================================

CREATE OR REPLACE TABLE ads_experiment_srm AS
WITH expected AS (
    SELECT
        r.experiment,
        r.variant,
        r.user_cnt,
        r.design_weight,
        SUM(r.user_cnt)    OVER (PARTITION BY r.experiment) AS total_users,
        SUM(r.design_weight) OVER (PARTITION BY r.experiment) AS total_weight
    FROM ads_experiment_result r
),
scored AS (
    SELECT
        e.*,
        -- 归一化设计权重后乘以总人数，得到期望人数
        e.total_users * e.design_weight / NULLIF(e.total_weight, 0) AS expected_cnt
    FROM expected e
)
SELECT
    experiment,
    variant,
    user_cnt,
    expected_cnt,
    (user_cnt - expected_cnt) / NULLIF(expected_cnt, 0)  AS relative_deviation,
    POWER(user_cnt - expected_cnt, 2) / NULLIF(expected_cnt, 0) AS chi2_component,
    -- 自由度：分支数 - 1
    COUNT(*) OVER (PARTITION BY experiment) - 1 AS degrees_of_freedom,
    SUM(POWER(user_cnt - expected_cnt, 2) / NULLIF(expected_cnt, 0))
        OVER (PARTITION BY experiment) AS chi2_statistic
FROM scored
ORDER BY experiment, variant;
