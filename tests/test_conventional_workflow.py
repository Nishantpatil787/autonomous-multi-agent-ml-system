"""Tests for Phase 11 — Conventional ML Workflow benchmark implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_config import (
    BENCHMARK_DATASETS,
    CONVENTIONAL_SEARCH_CONFIGS,
    BenchmarkConfig,
)
from benchmarks.conventional_workflow import (
    ConventionalExperimentRecord,
    ConventionalMLWorkflow,
    ConventionalResult,
)
from utils.config import Config


class TestConventionalWorkflowBasics:
    """Basic validation of ConventionalMLWorkflow structure and configuration."""

    def test_search_configs_exist_for_all_supported_models(self):
        """All 5 baseline model families must have defined conventional search configs."""
        expected_models = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "k_nearest_neighbors",
            "support_vector_machine",
        ]
        for m in expected_models:
            assert m in CONVENTIONAL_SEARCH_CONFIGS
            assert len(CONVENTIONAL_SEARCH_CONFIGS[m]) > 0
            assert isinstance(CONVENTIONAL_SEARCH_CONFIGS[m], list)

    def test_conventional_workflow_initialization(self):
        """Workflow initializes with default or custom hyperparameters."""
        wf = ConventionalMLWorkflow(random_state=123, test_size=0.25, cv_folds=3)
        assert wf.random_state == 123
        assert wf.test_size == 0.25
        assert wf.cv_folds == 3


class TestConventionalWorkflowExecution:
    """Execution tests for ConventionalMLWorkflow on synthetic and sample data."""

    @pytest.fixture
    def mini_dataset(self, tmp_path: Path) -> Path:
        """Create a tiny deterministic binary classification CSV dataset.

        Note: rows are intentionally distinct (no duplicates) so that
        duplicate-removal does not reduce the dataset size unexpectedly.
        """
        rng = np.random.default_rng(42)
        n = 30
        df = pd.DataFrame(
            {
                "num_feat1": rng.uniform(1.0, 10.0, n).round(2),
                "num_feat2": rng.uniform(0.5, 9.5, n).round(2),
                "cat_feat": rng.choice(["A", "B"], n).tolist(),
                "target": (np.arange(n) % 2).tolist(),
            }
        )
        csv_file = tmp_path / "mini_binary.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_run_on_mini_dataset_bounded_search(self, mini_dataset: Path):
        """Conventional workflow runs end-to-end with bounded search configs."""
        # Use small custom search config to keep unit test fast
        fast_configs = {
            "logistic_regression": [{"C": 1.0, "solver": "lbfgs", "max_iter": 500}],
            "decision_tree": [{"max_depth": 3, "min_samples_split": 2, "criterion": "gini"}],
        }

        wf = ConventionalMLWorkflow(cv_folds=3, random_state=42)
        res = wf.run(
            dataset_path=mini_dataset,
            target_column="target",
            custom_search_configs=fast_configs,
        )

        assert isinstance(res, ConventionalResult)
        assert res.selected_model in fast_configs
        assert 0.0 <= res.holdout_accuracy <= 1.0
        assert 0.0 <= res.holdout_f1 <= 1.0
        assert res.experiment_count == 2
        assert len(res.evaluated_configurations) == 2
        assert res.total_runtime_seconds > 0.0
        # 30 distinct rows (no duplicates) → 80% train = 24, 20% test = 6
        assert res.train_samples == 24
        assert res.test_samples == 6

    def test_result_to_dict_serialization(self, mini_dataset: Path):
        """ConventionalResult converts cleanly to a JSON-serializable dictionary."""
        fast_configs = {
            "logistic_regression": [{"C": 1.0, "solver": "lbfgs", "max_iter": 500}],
        }
        wf = ConventionalMLWorkflow(cv_folds=2, random_state=42)
        res = wf.run(
            dataset_path=mini_dataset,
            target_column="target",
            custom_search_configs=fast_configs,
        )
        d = res.to_dict()
        assert isinstance(d, dict)
        assert "holdout_f1" in d
        assert "cv_f1_mean" in d
        assert "experiment_count" in d
        assert d["experiment_count"] == 1
        assert "selected_model" in d
        assert d["selected_model"] == "logistic_regression"

    def test_run_on_sample_classification_binary(self):
        """Conventional workflow evaluates sample_classification.csv correctly."""
        csv_path = Config.DATASETS_DIR / "sample_classification.csv"
        assert csv_path.exists()

        wf = ConventionalMLWorkflow(cv_folds=5, random_state=42)
        res = wf.run(dataset_path=csv_path, target_column="Outcome")

        assert isinstance(res, ConventionalResult)
        assert res.problem_type == "binary_classification"
        assert res.target_column == "Outcome"
        # 5 models * 4 configs each = 20 configurations
        assert res.experiment_count == 20
        assert 0.0 <= res.holdout_f1 <= 1.0
        assert 0.0 <= res.cv_f1_mean <= 1.0
        assert res.selected_model in CONVENTIONAL_SEARCH_CONFIGS
        assert len(res.confusion_matrix) == 2
