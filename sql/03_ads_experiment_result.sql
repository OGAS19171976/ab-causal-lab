-- ============================================================================
-- 03 · ADS 实验结果表（充分统计量）
-- ============================================================================
-- 粒度：一个 (experiment, variant) 一行。
--
-- 从 DWS 的可加字段直接汇总，**不再回扫明细**。
--
-- 输出分两部分：
--
-- 1) **原始可加量**（user_cnt 与五个 SUM）
--    它们是"充分统计量"：只要给出这六个数，下游就能还原出均值、方差、
--    **协方差**，进而实现 post-only 检验、CUPED、甚至比值指标的 delta method。
--    这是 M1 把 CUPED 变成默认口径的前提 —— 加一个 SUM(pre×post) 就够了。
--
-- 2) **派生指标**（mean / var / cov）
--    给人和 BI 直接看。它们是可加量的确定性函数，不引入新信息。
--
-- 为什么两样都要留：只留派生值，下游想换个方法就得回扫明细；
-- 只留原始量，BI 那边每次都要重复写公式。多存几列的代价远小于两者之一。
--
-- 这正是"数仓链路"与"统计引擎"的接口面：
--   SQL 负责口径正确、可加、可回溯；
--   Python 负责假设检验、区间和诊断。
-- 两者都不越界，才不会出现"在 SQL 里算 p 值"这种无法复算的写法。
-- ============================================================================

CREATE OR REPLACE TABLE ads_experiment_result AS
WITH agg AS (
    SELECT
        experiment,
        variant,
        SUM(user_cnt)             AS user_cnt,
        SUM(post_sum)             AS post_sum,
        SUM(post_sq_sum)          AS post_sq_sum,
        SUM(pre_sum)              AS pre_sum,
        SUM(pre_sq_sum)           AS pre_sq_sum,
        SUM(pre_post_cross_sum)   AS pre_post_cross_sum
    FROM dws_experiment_variant_daily
    GROUP BY experiment, variant
),
joined AS (
    SELECT
        a.*,
        c.design_weight,
        c.layer,
        c.hypothesis,
        c.true_lift
    FROM agg a
    LEFT JOIN dim_experiment_config c
           ON c.experiment = a.experiment AND c.variant = a.variant
)
SELECT
    experiment,
    variant,
    layer,
    hypothesis,
    true_lift,
    design_weight,

    -- ---- 1. 原始可加量（充分统计量） ---- ---------------------------------
    user_cnt,
    pre_sum,
    post_sum,
    pre_sq_sum,
    post_sq_sum,
    pre_post_cross_sum,

    -- ---- 2. 派生指标 ---- -------------------------------------------------
    post_sum / user_cnt AS post_mean,
    pre_sum  / user_cnt AS pre_mean,

    -- 样本方差 (ddof=1)；user_cnt <= 1 时无定义，用 NULL 而不是 0，
    -- 避免下游把"无方差"误当成"零方差"
    CASE WHEN user_cnt > 1
         THEN (post_sq_sum - post_sum * post_sum / user_cnt) / (user_cnt - 1)
         ELSE NULL END AS post_var,
    CASE WHEN user_cnt > 1
         THEN (pre_sq_sum - pre_sum * pre_sum / user_cnt) / (user_cnt - 1)
         ELSE NULL END AS pre_var,

    -- CUPED 的 theta 只需要这一个协方差
    CASE WHEN user_cnt > 1
         THEN (pre_post_cross_sum - pre_sum * post_sum / user_cnt) / (user_cnt - 1)
         ELSE NULL END AS pre_post_cov
FROM joined
ORDER BY experiment, variant;
