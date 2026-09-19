-- ============================================================================
-- 07 · ADS 实验结果（**比值指标**口径）
-- ============================================================================
-- 粒度：一个 (experiment, variant) 一行。
--
-- 与 03（均值口径的 ADS）并列，而不是替换它：
-- 两条口径回答的是不同的问题，而且**必须能被分别复算** ——
-- 03 的六个数（n 与五个 SUM）对应 CUPED / Welch，
-- 这里的六个数对应 delta method。混在一张表里，
-- 将来任何一次口径调整都会同时动到两条链路的数字。
--
-- 只落可加量，点估计与标准误都在推断层算（与 03 同一条纪律）：
--     R   = sum_y / sum_x                          （比值本身）
--     Var(R) ≈ Var(y_i - R·x_i) / (n · X̄²)         （delta method）
-- 后者由 sum_yy / sum_xx / sum_xy 还原，不需要明细。
-- ============================================================================

CREATE OR REPLACE TABLE ads_experiment_ratio_result AS
SELECT
    experiment,
    variant,
    SUM(user_cnt)   AS user_cnt,
    SUM(sum_y)      AS sum_y,
    SUM(sum_x)      AS sum_x,
    SUM(sum_yy)     AS sum_yy,
    SUM(sum_xx)     AS sum_xx,
    SUM(sum_xy)     AS sum_xy
FROM dws_experiment_ratio_daily
GROUP BY experiment, variant;
