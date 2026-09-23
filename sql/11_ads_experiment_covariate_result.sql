-- ============================================================================
-- 11 · ADS 多协变量 CUPED 结果表（充分统计量）
-- ============================================================================
-- 粒度：一个 (experiment, variant) 一行，从 10 的可加字段直接汇总。
--
-- 与 03 一样，输出分两部分：
--
-- 1) **原始可加量**（十个 SUM）—— 它们才是"充分统计量"：
--    给出这十个数，下游能还原两个协变量的均值/方差、协变量之间的协方差、
--    协变量与结果的协方差，进而复原 θ̂ = Σ_X⁻¹ Σ_XY 与方差缩减。
--    **协变量之间的交叉项 `pre_metric_pre_cnt_cross_sum` 是这里唯一的新东西** ——
--    单协变量的 ADS（03）里根本不存在这一项，因为 p=1 时 Σ_X 是标量。
--
-- 2) **派生指标**（var / cov，ddof=1，与 03 的口径一致）
--    给人和 BI 直接看；它们是可加量的确定性函数，不引入新信息。
--
-- **两臂一起汇总**（不按 variant 分开算 θ̂）：这是口径，不是疏漏 ——
-- 多协变量 CUPED 的 θ̂ 在全样本上估（M1 的 `fit_multivariate_cuped` 就是这么做的，
-- Hsu 等人的原始形式也是），然后拿同一个 θ̂ 去校正两臂。
-- 逐臂各估一个 θ̂ 是另一种做法（嵌套更"灵活"、但方差更大且两臂的校正不等价），
-- 本仓库选前者 —— 报告里的明细路径与这条 ADS 路径必须用同一个口径，
-- 否则"两条路径对得上"就变成了一句没有意义的话。
-- 所以下游读这张表时，**两个 variant 的行要加起来**再喂给估计器。
-- ============================================================================

CREATE OR REPLACE TABLE ads_experiment_covariate_result AS
WITH agg AS (
    SELECT
        experiment,
        variant,
        SUM(user_cnt)                            AS user_cnt,
        SUM(pre_metric_sum)                      AS pre_metric_sum,
        SUM(pre_metric_sq_sum)                   AS pre_metric_sq_sum,
        SUM(pre_cnt_sum)                         AS pre_cnt_sum,
        SUM(pre_cnt_sq_sum)                      AS pre_cnt_sq_sum,
        SUM(pre_metric_pre_cnt_cross_sum)        AS pre_metric_pre_cnt_cross_sum,
        SUM(pre_metric_post_cross_sum)           AS pre_metric_post_cross_sum,
        SUM(pre_cnt_post_cross_sum)              AS pre_cnt_post_cross_sum,
        SUM(post_metric_sum)                     AS post_metric_sum,
        SUM(post_metric_sq_sum)                  AS post_metric_sq_sum
    FROM dws_experiment_covariate_daily
    GROUP BY experiment, variant
)
SELECT
    experiment,
    variant,

    -- ---- 1. 原始可加量（充分统计量） ---- ---------------------------------
    user_cnt,
    pre_metric_sum,
    pre_metric_sq_sum,
    pre_cnt_sum,
    pre_cnt_sq_sum,
    pre_metric_pre_cnt_cross_sum,
    pre_metric_post_cross_sum,
    pre_cnt_post_cross_sum,
    post_metric_sum,
    post_metric_sq_sum,

    -- ---- 2. 派生指标（样本二阶中心矩，ddof=1） ---- ------------------------
    -- user_cnt <= 1 时无定义，用 NULL 而不是 0（与 03 同一条理由：
    -- 别让下游把"无方差"读成"零方差"）
    CASE WHEN user_cnt > 1 THEN
        (pre_metric_sq_sum - pre_metric_sum * pre_metric_sum / user_cnt)
        / (user_cnt - 1) END                                  AS pre_metric_var,
    CASE WHEN user_cnt > 1 THEN
        (pre_cnt_sq_sum - pre_cnt_sum * pre_cnt_sum / user_cnt)
        / (user_cnt - 1) END                                  AS pre_cnt_var,
    CASE WHEN user_cnt > 1 THEN
        (post_metric_sq_sum - post_metric_sum * post_metric_sum / user_cnt)
        / (user_cnt - 1) END                                  AS post_metric_var,
    CASE WHEN user_cnt > 1 THEN
        (pre_metric_pre_cnt_cross_sum
         - pre_metric_sum * pre_cnt_sum / user_cnt)
        / (user_cnt - 1) END                                  AS pre_metric_pre_cnt_cov,
    CASE WHEN user_cnt > 1 THEN
        (pre_metric_post_cross_sum
         - pre_metric_sum * post_metric_sum / user_cnt)
        / (user_cnt - 1) END                                  AS pre_metric_post_cov,
    CASE WHEN user_cnt > 1 THEN
        (pre_cnt_post_cross_sum
         - pre_cnt_sum * post_metric_sum / user_cnt)
        / (user_cnt - 1) END                                  AS pre_cnt_post_cov
FROM agg
ORDER BY experiment, variant;
