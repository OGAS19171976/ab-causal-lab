-- ============================================================================
-- 05 · DWS 实验×分支×**簇**×日 汇总层
-- ============================================================================
-- 粒度：一个 (experiment, variant, cluster_id, ds) 一行。
--
-- 为什么在 DWS 已经有 (experiment, variant, ds) 之后还要再来一张
-- ------------------------------------------------------------------
-- 因为**随机化单元与分析单元可能不是一回事**。
-- 整簇随机化（城市/门店/学校级分流）时，正确的检验是把**簇**当观测单位；
-- 用用户级 t 检验会把标准误低估 80% 以上，I 类错误率实测 69.3%
-- ——而它给出的 p 值看起来完全正常（M6 的 02 节）。
--
-- 关键点是：**这张表不是"明细"，而是换个 GROUP BY 的同一批可加量。**
-- 05 与 02 出自同一张 DWD、同一组 SUM；区别只在分组键多了一个 cluster_id。
-- 换句话说，M0 就定下来的"只落 SUM / COUNT，不落 AVG / STDDEV"这条纪律，
-- 让"支持簇级分析"变成了加一个 GROUP BY，而不是加一条数据链路。
--
-- cluster_id 用什么
-- -----------------
-- 这里用用户维表里的 `city`（DWD 已经把 city 带出来了）。
-- 真实场景通常是门店号 / 学校号 / 城市号，由分流键决定 ——
-- 重要的是它**必须与分流时用的那个键完全一致**，否则簇级检验检的不是同一件事。
--
-- 空值必须显式处理：`COALESCE(city, 'UNKNOWN')`。
-- 若放任 NULL，SQL 的 GROUP BY 会把所有 NULL 归成一组，
-- 于是"一批身份不明的用户"会伪装成一个超大的簇，把簇间方差算错。
-- 这是数仓里很典型的一个坑：**聚合键上的空值要单独决定语义，不能默认放任**。
--
-- 与 02 一样只落可加字段：换时间窗口直接 SUM，均值/方差/协方差在上层还原。
-- ============================================================================

CREATE OR REPLACE TABLE dws_experiment_cluster_daily AS
SELECT
    d.experiment,
    d.variant,
    COALESCE(d.city, 'UNKNOWN')            AS cluster_id,
    d.expose_ds                            AS ds,
    COUNT(DISTINCT d.user_id)              AS user_cnt,
    SUM(d.post_metric)                     AS post_sum,
    SUM(d.post_metric * d.post_metric)     AS post_sq_sum,
    SUM(d.pre_metric)                      AS pre_sum,
    SUM(d.pre_metric * d.pre_metric)       AS pre_sq_sum,
    -- CUPED 的原料。整簇路径暂时用不到（该 DGP 没有前置指标），
    -- 但一起落下来，将来要做"簇级 CUPED"时不必改这一层。
    SUM(d.pre_metric * d.post_metric)      AS pre_post_cross_sum,
    SUM(d.post_cnt)                        AS post_event_cnt,
    SUM(d.pre_cnt)                         AS pre_event_cnt
FROM dwd_experiment_user d
GROUP BY d.experiment, d.variant, COALESCE(d.city, 'UNKNOWN'), d.expose_ds;
