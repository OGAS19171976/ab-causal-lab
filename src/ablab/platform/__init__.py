"""平台层：把 M0–M4 的引擎接成一个可演示的服务。

``registry``    实验注册表（sqlite3）：有状态、可校验、salt 不可变
``analysis``    分析编排：SRM → CUPED → 朴素对照 → 效应分解 → 序贯监控 → 健康判定
``api``         FastAPI 接口：注册表 CRUD + 分析 + 在线 A/A 验证
``demo``        演示数据：让平台一启动就有东西可看
``audit``       平台审计：整条管道的 A/A 校准、随机流独立性、演示效应分解
``static``      单文件前端（无构建步骤）

注意 ``api`` 不在包初始化里导入：它要 fastapi，而注册表和编排层本身
不依赖 web 框架。想用接口的人显式 ``from ablab.platform.api import create_app``。
"""

from .analysis import (
    PLATFORM_POPULATION,
    CheckItem,
    ExperimentReport,
    analyse_data,
    analyse_experiment,
    analyse_experiment_from_warehouse,
    run_aa_validation,
)
from .audit import (
    DemoDecomposition,
    MonitoringAuditResult,
    NoiseComponentResult,
    PlatformAAResult,
    RatioCalibrationAudit,
    SourceEquivalence,
    StreamIndependenceResult,
    UnitAwarenessResult,
    run_demo_decomposition,
    run_monitoring_fwer_audit,
    run_noise_component_audit,
    run_platform_aa_audit,
    run_ratio_calibration_audit,
    run_source_equivalence_audit,
    run_stream_independence_audit,
    run_unit_awareness_audit,
)
from .datasource import (
    ExperimentData,
    LookData,
    build_synthetic_data,
    build_warehouse_data,
    list_warehouse_experiments,
)
from .registry import (
    STATUSES,
    ExperimentRecord,
    ExperimentRegistry,
    RegistryError,
)

__all__ = [
    "CheckItem",
    "ExperimentReport",
    "PLATFORM_POPULATION",
    "analyse_data",
    "analyse_experiment",
    "analyse_experiment_from_warehouse",
    "run_aa_validation",
    "DemoDecomposition",
    "MonitoringAuditResult",
    "NoiseComponentResult",
    "PlatformAAResult",
    "SourceEquivalence",
    "StreamIndependenceResult",
    "UnitAwarenessResult",
    "RatioCalibrationAudit",
    "run_ratio_calibration_audit",
    "run_demo_decomposition",
    "run_monitoring_fwer_audit",
    "run_noise_component_audit",
    "run_platform_aa_audit",
    "run_source_equivalence_audit",
    "run_stream_independence_audit",
    "run_unit_awareness_audit",
    "ExperimentData",
    "LookData",
    "build_synthetic_data",
    "build_warehouse_data",
    "list_warehouse_experiments",
    "STATUSES",
    "ExperimentRecord",
    "ExperimentRegistry",
    "RegistryError",
]
