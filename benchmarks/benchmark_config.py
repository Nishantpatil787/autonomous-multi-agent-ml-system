"""Configuration and hyperparameter search space definitions for Phase 11 benchmarks.

Defines fixed, deterministic candidate configurations for the Conventional ML workflow
derived from the supported Phase 6 search spaces. Also provides dataset hash verification
and benchmark constants.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import Config


# ---------------------------------------------------------------------------
# Predefined Conventional Hyperparameter Search Configurations
# ---------------------------------------------------------------------------
# Derived from Phase 6 Optuna search spaces (tools/optimization_tools.py).
# Fixed, bounded, and deterministic to represent a standard manual/grid search.

CONVENTIONAL_SEARCH_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "logistic_regression": [
        {"C": 0.01, "solver": "lbfgs", "max_iter": 1000},
        {"C": 0.1, "solver": "lbfgs", "max_iter": 1000},
        {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        {"C": 10.0, "solver": "lbfgs", "max_iter": 1000},
    ],
    "decision_tree": [
        {"max_depth": 3, "min_samples_split": 2, "criterion": "gini"},
        {"max_depth": 5, "min_samples_split": 2, "criterion": "gini"},
        {"max_depth": 5, "min_samples_split": 5, "criterion": "entropy"},
        {"max_depth": 10, "min_samples_split": 4, "criterion": "entropy"},
    ],
    "random_forest": [
        {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
        {"n_estimators": 200, "max_depth": 5, "min_samples_split": 2},
    ],
    "k_nearest_neighbors": [
        {"n_neighbors": 3, "weights": "uniform", "metric": "euclidean"},
        {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
        {"n_neighbors": 7, "weights": "distance", "metric": "manhattan"},
        {"n_neighbors": 9, "weights": "distance", "metric": "euclidean"},
    ],
    "support_vector_machine": [
        {"C": 0.1, "kernel": "linear", "gamma": "scale"},
        {"C": 1.0, "kernel": "linear", "gamma": "scale"},
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        {"C": 10.0, "kernel": "rbf", "gamma": "scale"},
    ],
}


def get_dataset_hash(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a dataset file for integrity checking."""
    p = Path(file_path)
    if not p.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BenchmarkDatasetSpec:
    """Specification of a benchmark dataset."""

    name: str
    file_path: Path
    target_column: Optional[str]
    problem_type: str
    expected_sha256: str


# Standard project benchmark datasets
BENCHMARK_DATASETS: Dict[str, BenchmarkDatasetSpec] = {
    "binary_classification": BenchmarkDatasetSpec(
        name="sample_classification.csv",
        file_path=Config.DATASETS_DIR / "sample_classification.csv",
        target_column="Outcome",
        problem_type="binary_classification",
        expected_sha256="b9d52e83593ba8aab9c29c50d80649c1562543fec81b3dc4830d8e1b60d3275e",
    ),
    "multiclass_classification": BenchmarkDatasetSpec(
        name="sample_multiclass.csv",
        file_path=Config.DATASETS_DIR / "sample_multiclass.csv",
        target_column="class_label",
        problem_type="multiclass_classification",
        expected_sha256="ac49da0401a64ac9c9e7e7298f6a9611541e1214d4cfea06382b7f3ecc423c62",
    ),
}


@dataclass
class BenchmarkConfig:
    """Configuration options for a benchmark execution."""

    random_state: int = Config.RANDOM_STATE
    test_size: float = Config.TEST_SIZE
    cv_folds: int = Config.CV_FOLDS
    enable_mlflow: bool = Config.MLFLOW_ENABLED
    reports_dir: Path = field(default_factory=lambda: Config.REPORTS_DIR)
    json_output_path: Path = field(
        default_factory=lambda: Config.REPORTS_DIR / "phase11_comparison.json"
    )
    csv_output_path: Path = field(
        default_factory=lambda: Config.REPORTS_DIR / "phase11_comparison.csv"
    )
    chart_f1_path: Path = field(
        default_factory=lambda: Config.REPORTS_DIR / "phase11_f1_comparison.png"
    )
    chart_runtime_path: Path = field(
        default_factory=lambda: Config.REPORTS_DIR / "phase11_runtime_comparison.png"
    )
    chart_experiments_path: Path = field(
        default_factory=lambda: Config.REPORTS_DIR / "phase11_experiment_comparison.png"
    )
