"""合成数据生成：为仿真台提供已知真值的数据。"""

from .generator import (
    ExperimentSample,
    Population,
    PopulationConfig,
    assign_codes,
    assign_variants,
    generate_population,
    make_experiment_sample,
    simulate_outcomes,
    simulate_outcomes_treated,
    two_arm_spec,
)
from .scenarios import (
    ClusterSample,
    ClusterScenarioConfig,
    IVSample,
    IVScenarioConfig,
    RatioSample,
    RatioScenarioConfig,
    generate_cluster_scenario,
    generate_iv_scenario,
    generate_ratio_scenario,
)
from .sequential import (
    CanonicalSequences,
    LookSequence,
    default_information_fractions,
    simulate_canonical_sequences,
    simulate_experiment_sequence,
)

__all__ = [
    "ExperimentSample",
    "Population",
    "PopulationConfig",
    "assign_codes",
    "assign_variants",
    "generate_population",
    "make_experiment_sample",
    "simulate_outcomes",
    "simulate_outcomes_treated",
    "two_arm_spec",
    "ClusterSample",
    "IVSample",
    "IVScenarioConfig",
    "ClusterScenarioConfig",
    "RatioSample",
    "RatioScenarioConfig",
    "generate_cluster_scenario",
    "generate_iv_scenario",
    "generate_ratio_scenario",
    "CanonicalSequences",
    "LookSequence",
    "default_information_fractions",
    "simulate_canonical_sequences",
    "simulate_experiment_sequence",
]
