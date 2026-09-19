-- ============================================================================
-- 01 · DWD 实验单元宽表
-- ============================================================================
-- 粒度：一个 (experiment, user_id) 一行。
--
-- 三个口径要点，每一个都是数仓面试和实验平台面试的真实考点：
--
-- 1) **只取首次曝光**
--    同一用户可能被反复曝光（刷新页面、多端登录）。如果用全量曝光 join 明细，
--    同一用户的行为会被重复计算，指标直接虚高。这里用 MIN(ts) 收敛成一次。
--
-- 2) **前后窗口用条件聚合一次算出**
--    不要 join 事实表两遍（pre 一次、post 一次）—— 那是两大表 join，
--    而且窗口边界容易写不一致。一个 LEFT JOIN + CASE 就够。
--
-- 3) **保留 pre 指标而不只是 post**
--    pre 指标是 CUPED 方差缩减的输入，也是"分流是否平衡"的体检依据，
--    必须在 DWD 就带上，不能等到分析时再回表捞。
-- ============================================================================

CREATE OR REPLACE TABLE dwd_experiment_user AS
WITH first_expose AS (
    -- 首次曝光：同一 (experiment, user_id) 只保留最早一次
    SELECT
        experiment,
        variant,
        user_id,
        MIN(ts) AS expose_ts
    FROM ods_exposure_log
    GROUP BY experiment, variant, user_id
),
expo AS (
    SELECT
        experiment,
        variant,
        user_id,
        expose_ts,
        CAST(expose_ts AS DATE) AS expose_ds
    FROM first_expose
),
windowed AS (
    -- 一次性算出曝光前 14 天 / 曝光后 14 天的可加聚合量
    SELECT
        e.experiment,
        e.variant,
        e.user_id,
        e.expose_ds,
        e.expose_ts,
        SUM(CASE WHEN ev.ds <  e.expose_ds THEN ev.metric_value ELSE 0 END) AS pre_metric,
        SUM(CASE WHEN ev.ds <  e.expose_ds THEN 1               ELSE 0 END) AS pre_cnt,
        SUM(CASE WHEN ev.ds >= e.expose_ds THEN ev.metric_value ELSE 0 END) AS post_metric,
        SUM(CASE WHEN ev.ds >= e.expose_ds THEN 1               ELSE 0 END) AS post_cnt
    FROM expo e
    LEFT JOIN ods_event_log ev
           ON ev.user_id = e.user_id
          -- **必须按事件名过滤**：event_log 是长表，除了主指标还有护栏事件
          -- （08 路要用）。不过滤就会把护栏的取值（延迟 ~100ms）加进主指标 ——
          -- 实测这会把效应从 +27.2 抬到 +161，而一切看起来都"正常显著"。
          AND ev.event_name = 'interaction'
          AND ev.ds >= e.expose_ds - INTERVAL {PRE_DAYS} DAY
          AND ev.ds <  e.expose_ds + INTERVAL {POST_DAYS} DAY
    GROUP BY
        e.experiment, e.variant, e.user_id, e.expose_ds, e.expose_ts
)
SELECT
    w.experiment,
    w.variant,
    w.user_id,
    w.expose_ds,
    w.expose_ts,
    p.city,
    p.reg_ds,
    w.pre_metric,
    w.pre_cnt,
    w.post_metric,
    w.post_cnt
FROM windowed w
LEFT JOIN ods_user_profile p
       ON p.user_id = w.user_id;
