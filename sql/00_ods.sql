-- ============================================================================
-- 00 · ODS 贴源层
-- ============================================================================
-- 贴源层只做两件事：把落地文件映射成表、统一命名。**不做任何清洗和业务逻辑**。
-- 好处是上游文件重跑时，这一层永远可以零成本重建，问题也能定位到具体是哪一层。
--
-- 用 view 而不是 table：ODS 不落存储，直接读 Parquet，
-- 既省一份存储，也保证"任何时候读到的都是最新落地文件"。
-- {DATA_DIR} 由 build.py 在执行前替换成实际路径。
-- ============================================================================

CREATE OR REPLACE VIEW ods_exposure_log AS
SELECT
    CAST(ds AS DATE)        AS ds,
    CAST(ts AS TIMESTAMP)   AS ts,
    CAST(user_id AS VARCHAR) AS user_id,
    CAST(experiment AS VARCHAR) AS experiment,
    CAST(variant AS VARCHAR)    AS variant,
    CAST(layer AS VARCHAR)      AS layer
FROM read_parquet('{DATA_DIR}/exposure_log/*.parquet');

-- 用户行为明细（已按天聚合到用户粒度，真实场景下这里是埋点事件流）
CREATE OR REPLACE VIEW ods_event_log AS
SELECT
    CAST(ds AS DATE)          AS ds,
    CAST(user_id AS VARCHAR)  AS user_id,
    CAST(event_name AS VARCHAR) AS event_name,
    CAST(metric_value AS DOUBLE) AS metric_value
FROM read_parquet('{DATA_DIR}/event_log/*.parquet');

-- 用户维表
CREATE OR REPLACE VIEW ods_user_profile AS
SELECT
    CAST(user_id AS VARCHAR) AS user_id,
    CAST(reg_ds AS DATE)     AS reg_ds,
    CAST(city AS VARCHAR)    AS city
FROM read_parquet('{DATA_DIR}/user_profile/*.parquet');

-- ============================================================================
-- DIM · 实验配置维表
-- ============================================================================
-- 设计权重必须来自**实验配置**，而不是从观测数据反推 ——
-- 反推出来的权重会让 SRM 卡方恒等于 0，什么也检验不出来。
-- ============================================================================
-- 护栏声明维表：名字由业务定，所以是长表；方向与容忍度**必须显式声明**。
-- 这张表回答的是"哪些护栏是被声明的"——08 路只认它给的名单，
-- 而不是"事件里出现过什么"，免得一个误埋的事件名悄悄变成一条护栏。
CREATE OR REPLACE VIEW dim_guardrail_config AS
SELECT
    CAST(experiment AS VARCHAR)  AS experiment,
    CAST(guardrail AS VARCHAR)   AS guardrail,
    CAST(direction AS VARCHAR)   AS direction,
    CAST(max_harm AS DOUBLE)     AS max_harm
FROM read_parquet('{DATA_DIR}/guardrail_config/*.parquet');

CREATE OR REPLACE VIEW dim_experiment_config AS
SELECT
    CAST(experiment AS VARCHAR)     AS experiment,
    CAST(variant AS VARCHAR)        AS variant,
    CAST(design_weight AS DOUBLE)   AS design_weight,
    CAST(layer AS VARCHAR)          AS layer,
    CAST(true_lift AS DOUBLE)       AS true_lift,   -- 仅仿真用；真实场景没有这一列
    CAST(hypothesis AS VARCHAR)     AS hypothesis
FROM read_parquet('{DATA_DIR}/experiment_config/*.parquet');
