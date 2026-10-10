"""Shared configuration for the Phase 11 conventional-vs-autonomous benchmark.

The two workflows are benchmarked on the exact same committed datasets, so the
dataset selection is expressed as stable *keys* (``classification`` /
``multiclass``) that resolve to the fixture files the whole project uses. The
conventional grid below is derived from the Phase 6 Optuna search-space bounds
in :mod:`utils.config` so the two workflows explore overlapping regions of the
same configuration space; the grid is a small, hand-picked, bounded selection
rather than a copy of the search machinery, which is what makes it a
*conventional manual* experiment plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils import config

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_KEY_BINARY = "classification"
DATASET_KEY_MULTICLASS = "multiclass"
ALL_DATASET_KEYS: List[str] = [DATASET_KEY_BINARY, DATASET_KEY_MULTICLASS]

#: The committed fixtures the benchmark runs on. The hashes are validated by the
#: runner before any training happens, so a silently modified fixture cannot
#: poison a comparison.
DATASET_FILES: Dict[str, Path] = {
    DATASET_KEY_BINARY: DATASETS_DIR / "sample_classification.csv",
    DATASET_KEY_MULTICLASS: DATASETS_DIR / "sample_multiclass.csv",
}

#: Expected SHA-256 hashes of the committed fixtures (must never change).
EXPECTED_DATASET_SHA256: Dict[str, str] = {
    DATASET_KEY_BINARY: (
        "49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c"
    ),
    DATASET_KEY_MULTICLASS: (
        "0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8"
    ),
}


def resolve_dataset_path(key: str) -> Path:
    """Resolve a benchmark dataset key to its fixture path.

    Args:
        key: ``classification``, ``multiclass`` or an existing CSV path.

    Returns:
        The resolved :class:`pathlib.Path`.

    Raises:
        ValueError: The key is unknown.
    """
    if key in DATASET_FILES:
        return DATASET_FILES[key]
    candidate = Path(key).expanduser()
    if candidate.is_file():
        return candidate
    raise ValueError(
        f"Unknown benchmark dataset '{key}'. Choose from "
        f"{', '.join(ALL_DATASET_KEYS)} or pass a path to an existing CSV."
    )


def describe_dataset(key: str) -> str:
    """Return a short human readable label for a dataset key."""
    return {
        DATASET_KEY_BINARY: "binary classification (sample_classification.csv)",
        DATASET_KEY_MULTICLASS: "multiclass classification (sample_multiclass.csv)",
    }.get(key, key)


# --------------------------------------------------------------------------- #
# Experiment definition
# --------------------------------------------------------------------------- #
#: One experiment = one unique model configuration trained and evaluated once on
#: the hold-out split. Cross-validation folds are a within-experiment
#: measurement (as in the autonomous baseline stage) and are never counted as
#: experiments themselves. This is the counting rule applied on both sides.
EXPERIMENT_DEFINITION = (
    "one experiment is one unique model configuration trained and evaluated on "
    "the hold-out split; CV folds inside an experiment are not counted as "
    "experiments"
)


# --------------------------------------------------------------------------- #
# The conventional manual grid
# --------------------------------------------------------------------------- #
def _logistic_configs() -> List[Dict[str, object]]:
    """Hand-picked C values from the Phase 6 logistic range (1e-3 .. 1e2).

    The three values are the decade points ``10^-1, 10^0, 10^1``, all of which
    fall strictly inside the Phase 6 log-uniform range.
    """
    low, high = config.OPTUNA_LOGISTIC_C_RANGE
    return [
        {
            "C": c,
            "max_iter": config.OPTUNA_LOGISTIC_MAX_ITER,
            "random_state": config.RANDOM_STATE,
        }
        for c in (0.1, 1.0, 10.0)
        if low <= c <= high
    ]


def _tree_configs() -> List[Dict[str, object]]:
    """Decision-tree depths from the Phase 6 depth range (1 .. 20)."""
    low, high = config.OPTUNA_TREE_MAX_DEPTH_RANGE
    return [
        {"max_depth": depth, "random_state": config.RANDOM_STATE}
        for depth in (2, 4, 8)
        if low <= depth <= high
    ]


def _forest_configs() -> List[Dict[str, object]]:
    """Forest sizes and depths from the Phase 6 ranges/choices (10..200, depth choices)."""
    n_min, _ = config.OPTUNA_FOREST_N_ESTIMATORS_RANGE
    depths = [d for d in (None, 5, 8) if d in config.OPTUNA_FOREST_MAX_DEPTH_CHOICES]
    return [
        {"n_estimators": n, "max_depth": depth, "random_state": config.RANDOM_STATE}
        for n in (50, 100)
        if n_min <= n
        for depth in depths
    ]


def _knn_configs() -> List[Dict[str, object]]:
    """Neighbour counts from the Phase 6 range (1 .. 30)."""
    low, high = config.OPTUNA_KNN_N_NEIGHBORS_RANGE
    return [{"n_neighbors": k} for k in (3, 5, 7) if low <= k <= high]


def _svm_configs() -> List[Dict[str, object]]:
    """SVM C values and kernels from the Phase 6 range (1e-2 .. 1e2) and kernel choices."""
    low, high = config.OPTUNA_SVM_C_RANGE
    kernels = [k for k in ("linear", "rbf") if k in config.OPTUNA_SVM_KERNEL_CHOICES]
    return [
        {
            "C": c,
            "kernel": kernel,
            "probability": False,
            "random_state": config.RANDOM_STATE,
        }
        for c in (0.1, 1.0, 10.0)
        if low <= c <= high
        for kernel in kernels
    ]


def build_conventional_grid() -> Dict[str, List[Dict[str, object]]]:
    """Return the predefined conventional configuration grid.

    The grid keeps the exact registry names of the candidate families so the two
    workflows provably share the same model set. Every entry is a complete
    parameter set for :func:`tools.training_tools.build_model_pipeline`.

    Returns:
        A mapping of model name to a bounded list of parameter dictionaries.
    """
    return {
        "Logistic Regression": _logistic_configs(),
        "Decision Tree": _tree_configs(),
        "Random Forest": _forest_configs(),
        "K-Nearest Neighbors": _knn_configs(),
        "Support Vector Machine": _svm_configs(),
    }


def conventional_grid_size(grid: Optional[Dict[str, List[Dict[str, object]]]] = None) -> int:
    """Return the total number of configurations (i.e. experiments) in a grid."""
    return sum(len(v) for v in (grid or build_conventional_grid()).values())


# --------------------------------------------------------------------------- #
# Autonomous workflow options
# --------------------------------------------------------------------------- #
#: The autonomous side reuses the existing Optuna search with the project's own
#: budget, so the comparison measures the real pipeline instead of a test stub.
AUTONOMOUS_N_TRIALS: int = config.OPTUNA_N_TRIALS


# --------------------------------------------------------------------------- #
# Benchmark MLflow store (kept strictly separate from the canonical store)
# --------------------------------------------------------------------------- #
#: The benchmark records one run per dataset in its own store. This must never
#: point at ``experiments/tracking/mlflow.db``: that store holds the two
#: canonical Phase 10 runs and is a protected, read-only-for-tests resource.
BENCHMARK_MLFLOW_TRACKING_DIR = PROJECT_ROOT / "experiments" / "tracking_benchmark"
BENCHMARK_MLFLOW_TRACKING_URI = (
    f"sqlite:///{(BENCHMARK_MLFLOW_TRACKING_DIR / 'mlflow.db').as_posix()}"
)
BENCHMARK_MLFLOW_EXPERIMENT_NAME = "benchmark-phase11"
#: Tags applied to each benchmark run so the store stays self-describing.
BENCHMARK_MLFLOW_TAGS = {
    "phase": "11",
    "execution_mode": "benchmark",
    "workflow_scope": "conventional_vs_autonomous",
}


# --------------------------------------------------------------------------- #
# Report artifacts
# --------------------------------------------------------------------------- #
COMPARISON_JSON = REPORTS_DIR / "phase11_comparison.json"
COMPARISON_CSV = REPORTS_DIR / "phase11_comparison.csv"
CHART_F1 = REPORTS_DIR / "phase11_f1_comparison.png"
CHART_RUNTIME = REPORTS_DIR / "phase11_runtime_comparison.png"
CHART_EXPERIMENTS = REPORTS_DIR / "phase11_experiment_comparison.png"


@dataclass(frozen=True)
class BenchmarkResultFiles:
    """The files one benchmark run produces."""

    json: Path = COMPARISON_JSON
    csv: Path = COMPARISON_CSV
    chart_f1: Path = CHART_F1
    chart_runtime: Path = CHART_RUNTIME
    chart_experiments: Path = CHART_EXPERIMENTS
    charts: List[Path] = field(
        default_factory=lambda: [CHART_F1, CHART_RUNTIME, CHART_EXPERIMENTS]
    )