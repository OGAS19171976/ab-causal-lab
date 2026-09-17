-- ============================================================================
-- 02 · DWS 实验×分支×日 汇总层
-- ============================================================================
-- 粒度：一个 (experiment, variant, ds) 一行。
--
-- **这一层的全部意义在于"可加性"。**
-- 只落 SUM / SUM_OF_SQUARES / COUNT 这三类可加字段，
-- 于是：
--   * 想换时间窗口，直接 SUM 就行，不用回扫明细
--   * 想补算某一天，只需重跑那一天的 DWD 分区
--   * 均值、方差、协方差、标准误全部可以在上层用公式还原
--
-- 反面教材：在这一层直接存 AVG 和 STDDEV。
-- 平均值不可加（AVG(AVG) ≠ AVG），一旦上游要看"最近 7 天"就得重新扫明细。
--
-- **M1 新增 `pre_post_cross_sum`。** 没有它，ADS 只能算出 pre 与 post 各自的
-- 方差，算不出二者的协方差 —— 而 CUPED 的 theta 恰恰是
--   theta = Cov(pre, post) / Var(pre)
-- 少了这一项，方差缩减就没有原料。一行 SQL 换来 ~63% 的方差缩减。
--
-- 偏移量 SUM((x-μ)²) = SUM(x²) - (SUM(x))²/n 是数值不稳定的经典写法，
-- 这里样本量在万级、量级在千级，双精度足够；生产上若量级极端，
-- 应改用 Welford 或存 (x - K)² 的平移版本。
-- ============================================================================

CREATE OR REPLACE TABLE dws_experiment_variant_daily AS
SELECT
    d.experiment,
    d.variant,
    d.expose_ds                            AS ds,
    COUNT(DISTINCT d.user_id)              AS user_cnt,
    SUM(d.post_metric)                     AS post_sum,
    SUM(d.post_metric * d.post_metric)     AS post_sq_sum,
    SUM(d.pre_metric)                      AS pre_sum,
    SUM(d.pre_metric * d.pre_metric)       AS pre_sq_sum,
    -- CUPED 的原料：Σ(pre × post)
    SUM(d.pre_metric * d.post_metric)      AS pre_post_cross_sum,
    SUM(d.post_cnt)                        AS post_event_cnt,
    SUM(d.pre_cnt)                         AS pre_event_cnt
FROM dwd_experiment_user d
GROUP BY d.experiment, d.variant, d.expose_ds;
