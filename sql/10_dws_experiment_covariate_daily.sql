-- ============================================================================
-- 10 · DWS 多协变量 CUPED 的充分统计量（日粒度）
-- ============================================================================
-- 粒度：一个 (experiment, variant, ds) 一行。
--
-- **为什么独立成表**（而不是给 02 加列）：与 06/07 同一个理由 —— 02/03 的内容
-- 一格都不能变，README 里所有已引用的数仓数字都挂在它们上面。
--
-- 它要回答的问题：**"用两个协变量校正"能不能不回扫明细就做出来？**
-- 多协变量 CUPED 是 θ̂ = Σ_X⁻¹ Σ_XY，其中
--     Σ_X  = Cov([x1, x2])        （2×2）
--     Σ_XY = Cov([x1, x2], y)     （2×1）
-- 这两块**只由二阶矩决定**，而二阶矩是可加的：
--     n, Σx1, Σx2, Σx1², Σx2², Σx1x2, Σx1y, Σx2y, Σy, Σy²
-- 把这十个可加量落进 DWS，上层就能复原 θ̂、条件数与方差缩减 —— **不需要明细**。
--
-- 两个协变量都**已经在 01 里落着**：pre_metric（前置互动值之和）与
-- pre_cnt（前置互动次数之和）。所以这一层既没动 DGP、也没动 DWD ——
-- 与 06/07 那次同样的判断：**原料本来就在，缺的只是把二阶矩搬上来。**
--
-- 诚实边界（写在最前面，因为它是这一层最重要的结论）：
-- **交叉拟合的 θ̂（"诚实口径"）算不出来。** 它要求"留出那一折用别的折估的 θ̂"，
-- 而"哪个用户在哪个折"不是可加量 —— 把折号落库等于把随机划分变成数据的一部分
-- （换一个 seed 就得重跑整条数仓）。所以这一层给的是**样本内**口径；
-- 交叉拟合仍然只能在明细路径上做。报告里两个数并排给，并量出差额
-- （实测差额很小，但"很小"是量出来的，不是假设的）。
--
-- 条件数的一处坑（从 M1 那边学来的，这里照做）：条件数必须算在**真正用于求解的
-- 那个矩阵**上。加 ridge 之后要算 Σ_X + ridge·I 的条件数，否则"ridge 有没有
-- 改善条件数"这条断言会得到两个一模一样的数（M1 第一版就报了 9.9e17 vs 9.9e17）。
-- 本层不加 ridge（可加量里没有"正则化强度"这种东西，它属于分析时的选择）。
-- ============================================================================

CREATE OR REPLACE TABLE dws_experiment_covariate_daily AS
SELECT
    d.experiment,
    d.variant,
    d.expose_ds                              AS ds,
    COUNT(DISTINCT d.user_id)                AS user_cnt,
    -- 协变量 1：前置互动值
    SUM(d.pre_metric)                        AS pre_metric_sum,
    SUM(d.pre_metric * d.pre_metric)         AS pre_metric_sq_sum,
    -- 协变量 2：前置互动次数
    SUM(d.pre_cnt)                           AS pre_cnt_sum,
    SUM(d.pre_cnt * d.pre_cnt)               AS pre_cnt_sq_sum,
    -- 两个协变量之间的交叉项：**没有它就算不出 2×2 的 Σ_X**
    SUM(d.pre_metric * d.pre_cnt)            AS pre_metric_pre_cnt_cross_sum,
    -- 协变量与结果指标的交叉项：Σ_XY
    SUM(d.pre_metric * d.post_metric)        AS pre_metric_post_cross_sum,
    SUM(d.pre_cnt * d.post_metric)           AS pre_cnt_post_cross_sum,
    -- 结果指标的二阶矩
    SUM(d.post_metric)                       AS post_metric_sum,
    SUM(d.post_metric * d.post_metric)       AS post_metric_sq_sum
FROM dwd_experiment_user d
GROUP BY d.experiment, d.variant, d.expose_ds;
