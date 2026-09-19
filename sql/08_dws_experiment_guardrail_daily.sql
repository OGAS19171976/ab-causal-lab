-- ============================================================================
-- 08 路 DWS：实验 × 变体 × 护栏 × 天 —— **长表**的聚合
-- ============================================================================
-- 为什么是长表：护栏是"每个实验自己声明几个"，名字由业务定。
-- 列式建模（latency_p99 一列、crash_rate 一列）会逼着人每加一个护栏
-- 就改一次已发布层的表结构 —— 而按本仓库的规矩，已发布层一改动，
-- README 里所有引用过的数字就全变了（06/07 那条注释说的就是这件事）。
--
-- 长表的来源是 ODS 的 event_log：那里 event_name 既有主指标（interaction），
-- 也有护栏名。**不需要新的落地文件** —— 护栏本来就是"同一批用户身上多测几个量"。
--
-- 窗口口径与 01 路 DWD **完全一致**（`ds >= expose_ds` 且落在 +{POST_DAYS} 天内），
-- 否则主指标与护栏会看不同的时间窗，最后在报告里打架。
--
-- 只落**可加量**（n / Σx / Σx²）：均值与方差都能还原，
-- 上游要看"最近 7 天"时不必重扫明细（与 02 路同一条规矩）。
-- ============================================================================

CREATE OR REPLACE TABLE dws_experiment_guardrail_daily AS
SELECT
    e.experiment,
    e.variant,
    e.ds,
    e.event_name                        AS guardrail,
    COUNT(DISTINCT e.user_id)           AS user_cnt,
    SUM(e.metric_value)                 AS value_sum,
    SUM(e.metric_value * e.metric_value) AS value_sq_sum
FROM (
    SELECT
        x.experiment,
        x.variant,
        ev.ds,
        ev.user_id,
        ev.event_name,
        ev.metric_value
    FROM (
        -- 与 01 路同一套"首次曝光"口径：同一实验同一用户只算第一次
        SELECT
            experiment,
            variant,
            user_id,
            MIN(CAST(ts AS TIMESTAMP)) AS expose_ts
        FROM ods_exposure_log
        GROUP BY experiment, variant, user_id
    ) x
    JOIN ods_event_log ev
      ON ev.user_id = x.user_id
     -- 只看**处置后**的窗口：护栏关心的是"上线之后有没有变差"
     AND ev.ds >= CAST(x.expose_ts AS DATE)
     AND ev.ds <  CAST(x.expose_ts AS DATE) + INTERVAL {POST_DAYS} DAY
    -- 只留**被声明的**护栏。名单来自配置维表，而不是"事件里出现过什么" ——
    -- 后者会让一个误埋的事件名悄悄变成一条护栏（护栏的名单必须是声明出来的）。
    JOIN dim_guardrail_config g
      ON g.experiment = x.experiment
     AND g.guardrail = ev.event_name
) e
GROUP BY e.experiment, e.variant, e.ds, e.event_name;
