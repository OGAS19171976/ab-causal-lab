-- ============================================================================
-- 06 · DWS 实验×分支×日 汇总层（**比值指标**口径）
-- ============================================================================
-- 粒度：一个 (experiment, variant, ds) 一行。
--
-- 为什么不能复用 02 那张 DWS
-- ------------------------
-- 02 里的 `post_sum` 是**分子**（互动值的和），而比值指标要的是
-- `Σy / Σx` —— 分子是值、分母是**次数**。两者语义不同：
-- 把 02 的 `post_sum` 当成比值指标的 `x` 用，算出来的东西没有意义
-- （那等于"互动值的和 ÷ 前置指标的和"）。
--
-- 更不能"顺手往 02 里加一列"：02 是现有 ADS 的输入，
-- 它的内容一改动，README 里所有已引用的数仓数字（27.248206 / 23.342905 /
-- 30,741 / 27,946）就全变了。**DGP 与已发布的层只做加法。**
--
-- 所以这一层是**独立的一张表**，与 05（簇粒度）同样的思路：
-- 同一张 DWD、同一组 SUM，只是换了个口径的目的。
--
-- 分母从哪来
-- ----------
-- **不需要改源数据**：DWD 里 `post_cnt`（互动次数）与 `post_metric`（互动值之和）
-- 都已经落下来了，两者都是可加的。于是
--     比值 = Σ post_metric / Σ post_cnt = "整体平均每次互动的价值"
-- 这是一个**真正的比值指标**（Σy/Σx），而不是"人均比值的均值"
-- mean(y_i/x_i) —— M1 实测两者口径差 12.6%。
--
-- 为什么还要落 sum_xx / sum_yy / sum_xy
-- ------------------------------------
-- delta method 的方差是 `Var(y_i - R·x_i) / (n·X̄²)`，
-- 它需要 Σy²、Σx²、Σxy 三个可加量。少落一个，上层就只能退回
-- "把两个均值相除"那种没有方差解释的算法。
--
-- 与 02 / 05 一样只落可加字段：换时间窗口直接 SUM，均值/方差在上层还原。
-- ============================================================================

CREATE OR REPLACE TABLE dws_experiment_ratio_daily AS
SELECT
    d.experiment,
    d.variant,
    d.expose_ds                            AS ds,
    COUNT(DISTINCT d.user_id)              AS user_cnt,
    SUM(d.post_metric)                     AS sum_y,
    SUM(d.post_cnt)                        AS sum_x,
    SUM(d.post_metric * d.post_metric)     AS sum_yy,
    SUM(d.post_cnt * d.post_cnt)           AS sum_xx,
    SUM(d.post_metric * d.post_cnt)        AS sum_xy
FROM dwd_experiment_user d
GROUP BY d.experiment, d.variant, d.expose_ds;
