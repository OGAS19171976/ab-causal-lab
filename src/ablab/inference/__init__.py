"""推断层：实验分析、体检诊断、区间估计。"""

from .aggregates import AggregateStats
from .clustered import (
    ClusterDiagnostics,
    WildBootstrapResult,
    cluster_level_ttest,
    cluster_robust_ttest,
    estimate_icc,
    wild_cluster_bootstrap,
)
from .cuped import (
    CovariateMoments,
    CupedFit,
    MultivariateCupedFit,
    cuped_estimate,
    cuped_ttest,
    fit_cuped,
    fit_multivariate_cuped,
    multivariate_cuped_from_moments,
    multivariate_theta,
)
from .power import mde, required_n_per_arm, se_of_mean_diff, z_power
from .ratio import NaiveRatioResult, naive_unit_ratio_ttest, ratio_delta_method
from .result import Diagnostic, Estimate, Status
from .srm import srm_check, srm_from_weights
from .tests import two_proportion_ztest, welch_ttest, welch_ttest_from_stats
from .welch import (
    WelchInference,
    t_inference,
    welch_inference,
    welch_inference_from_components,
)

__all__ = [
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
    "AggregateStats",
    "ClusterDiagnostics",
    "cluster_level_ttest",
    "cluster_robust_ttest",
    "estimate_icc",
    "CupedFit",
    "MultivariateCupedFit",
    "CovariateMoments",
    "multivariate_cuped_from_moments",
    "cuped_estimate",
    "cuped_ttest",
    "fit_cuped",
    "fit_multivariate_cuped",
    "multivariate_theta",
    "mde",
    "required_n_per_arm",
    "se_of_mean_diff",
    "z_power",
    "NaiveRatioResult",
    "naive_unit_ratio_ttest",
    "ratio_delta_method",
    "Diagnostic",
    "Estimate",
    "Status",
    "srm_check",
    "srm_from_weights",
    "welch_ttest",
    "welch_ttest_from_stats",
    "two_proportion_ztest",
    "WelchInference",
    "t_inference",
    "welch_inference",
    "welch_inference_from_components",
]
