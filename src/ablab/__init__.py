"""ablab：确定性分流 + 实验分析 + 因果推断验证台。

模块地图
--------
``ablab.hashing``      纯 Python MurmurHash3，跨进程可复现的分流基石
``ablab.assignment``   分流引擎：两把哈希放量、分层正交
``ablab.inference``    推断层：Welch t / CUPED / 比值 delta / 聚类稳健 / SRM
``ablab.sequential``   序贯检验：alpha 消耗、群序贯边界、mSPRT、贝叶斯决策
``ablab.causal``       观察数据因果：DiD（含交错处置）、合成控制、敏感性分析
``ablab.causal.hte``   异质效应：DML、元学习器、honest 因果森林、Qini/AUUC
``ablab.sim``          合成数据生成器（已知 ground truth）
``ablab.validation``   仿真验证台：分流审计、方法校准、序贯审计
``ablab.warehouse``    DuckDB 数仓链路 ODS→DWD→DWS→ADS
``ablab.plotting``     图表样式与中文字体
"""

from .assignment import (
    N_BUCKETS,
    ExperimentSpec,
    Layer,
    LayerSlot,
    Randomizer,
    Variant,
)
from .causal import (
    CausalForest,
    ForestConfig,
    HTEConfig,
    Panel,
    StaggeredPanelConfig,
    callaway_santanna,
    dml_partial_linear,
    generate_hte_data,
    generate_staggered_panel,
    placebo_inference,
    pretrend_test,
    qini_coefficient,
    synthetic_control,
    trend_sensitivity,
    twfe,
    twfe_decomposition,
)
from .hashing import KeyBatcher, murmur3_32, murmur3_32_str
from .inference import (
    Diagnostic,
    Estimate,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_stats,
)
from .sequential import (
    BoundarySolver,
    NormalPrior,
    Posterior,
    SequentialDesign,
    adjusted_p_value,
    always_valid_path,
    build_design,
    decide,
    expected_loss,
    msprt_p_value,
    posterior,
    probability_better,
    repeated_ci,
)
from .sim import Population, PopulationConfig, generate_population, two_arm_spec
from .validation import (
    AAResult,
    AssignmentAudit,
    PeekResult,
    PowerResult,
    power_curve,
    run_aa_trials,
    run_assignment_audit,
    run_peeking_simulation,
    run_power_trials,
)

__version__ = "0.1.0"

__all__ = [
    "N_BUCKETS",
    "ExperimentSpec",
    "Layer",
    "LayerSlot",
    "Randomizer",
    "Variant",
    "KeyBatcher",
    "murmur3_32",
    "murmur3_32_str",
    "Diagnostic",
    "Estimate",
    "srm_check",
    "welch_ttest",
    "welch_ttest_from_stats",
    "two_proportion_ztest",
    "Population",
    "PopulationConfig",
    "generate_population",
    "two_arm_spec",
    "Panel",
    "StaggeredPanelConfig",
    "generate_staggered_panel",
    "twfe",
    "twfe_decomposition",
    "callaway_santanna",
    "pretrend_test",
    "synthetic_control",
    "placebo_inference",
    "trend_sensitivity",
    "CausalForest",
    "ForestConfig",
    "HTEConfig",
    "dml_partial_linear",
    "generate_hte_data",
    "qini_coefficient",
    "BoundarySolver",
    "SequentialDesign",
    "adjusted_p_value",
    "build_design",
    "repeated_ci",
    "always_valid_path",
    "msprt_p_value",
    "NormalPrior",
    "Posterior",
    "decide",
    "expected_loss",
    "posterior",
    "probability_better",
    "AAResult",
    "AssignmentAudit",
    "PeekResult",
    "PowerResult",
    "power_curve",
    "run_aa_trials",
    "run_assignment_audit",
    "run_peeking_simulation",
    "run_power_trials",
    "__version__",
]
