"""Benchmarks package comparing Conventional and Autonomous Multi-Agent ML workflows."""

from benchmarks.autonomous_workflow import AutonomousBenchmarkWrapper, AutonomousResult
from benchmarks.benchmark_config import (
    BENCHMARK_DATASETS,
    CONVENTIONAL_SEARCH_CONFIGS,
    BenchmarkConfig,
    BenchmarkDatasetSpec,
    get_dataset_hash,
)
from benchmarks.comparison import (
    BenchmarkRunner,
    DatasetComparisonResult,
    compute_dataset_comparison,
)
from benchmarks.conventional_workflow import (
    ConventionalExperimentRecord,
    ConventionalMLWorkflow,
    ConventionalResult,
)

__all__ = [
    "BENCHMARK_DATASETS",
    "CONVENTIONAL_SEARCH_CONFIGS",
    "BenchmarkConfig",
    "BenchmarkDatasetSpec",
    "get_dataset_hash",
    "ConventionalMLWorkflow",
    "ConventionalResult",
    "ConventionalExperimentRecord",
    "AutonomousBenchmarkWrapper",
    "AutonomousResult",
    "DatasetComparisonResult",
    "BenchmarkRunner",
    "compute_dataset_comparison",
]
