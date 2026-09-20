"""数仓链路：用 DuckDB 实现 ODS → DWD → DWS → ADS，并接上推断层。

这一半负责回答"指标算得对不对、口径可不可复算"；
``ablab.validation`` 那一半负责回答"p 值算得对不对"。
"""

from .build import (
    SQL_ORDER,
    CovariateAdjustmentReport,
    CrossValidation,
    ExperimentAnalysis,
    analyse_ads,
    build_warehouse,
    covariate_adjustment_report,
    load_ads_result,
    render_report,
    run_sql_files,
    split_statements,
    verify_against_detail,
)
from .generate import (
    DEFAULT_EXPERIMENTS,
    LIFT_REPLICATE_LAYER,
    LIFT_REPLICATE_PREFIX,
    RATIO_REPLICATE_PREFIX,
    ExperimentDef,
    WarehouseConfig,
    config_fingerprint,
    generate_source_data,
    ratio_replicate_experiments,
    ratio_replicate_experiments_with_lift,
)
from .ingest import (
    ExternalExperiment,
    IngestReport,
    load_real_traffic,
)

__all__ = [
    "ExternalExperiment",
    "IngestReport",
    "load_real_traffic",
    "SQL_ORDER",
    "CrossValidation",
    "CovariateAdjustmentReport",
    "ExperimentAnalysis",
    "analyse_ads",
    "build_warehouse",
    "covariate_adjustment_report",
    "load_ads_result",
    "render_report",
    "run_sql_files",
    "split_statements",
    "verify_against_detail",
    "DEFAULT_EXPERIMENTS",
    "LIFT_REPLICATE_LAYER",
    "LIFT_REPLICATE_PREFIX",
    "RATIO_REPLICATE_PREFIX",
    "config_fingerprint",
    "ratio_replicate_experiments",
    "ratio_replicate_experiments_with_lift",
    "ExperimentDef",
    "WarehouseConfig",
    "generate_source_data",
]
