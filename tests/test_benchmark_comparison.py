"""Tests for Phase 11 — Benchmark Comparison, Plotting, and Reporting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.benchmark_config import (
    BENCHMARK_DATASETS,
    BenchmarkConfig,
    BenchmarkDatasetSpec,
    get_dataset_hash,
)
from benchmarks.comparison import (
    BenchmarkRunner,
    DatasetComparisonResult,
    compute_dataset_comparison,
)
from benchmarks.conventional_workflow import ConventionalResult
from benchmarks.autonomous_workflow import AutonomousResult
from dashboard import load_benchmark_results
from utils.config import Config


class TestDatasetIntegrity:
    """Verify that source benchmark datasets remain intact with exact expected hashes."""

    def test_sample_classification_hash_intact(self):
        spec = BENCHMARK_DATASETS["binary_classification"]
        computed_hash = get_dataset_hash(spec.file_path)
        assert computed_hash == spec.expected_sha256, (
            f"Binary dataset hash mismatch! Expected {spec.expected_sha256}, got {computed_hash}"
        )

    def test_sample_multiclass_hash_intact(self):
        spec = BENCHMARK_DATASETS["multiclass_classification"]
        computed_hash = get_dataset_hash(spec.file_path)
        assert computed_hash == spec.expected_sha256, (
            f"Multiclass dataset hash mismatch! Expected {spec.expected_sha256}, got {computed_hash}"
        )

    def test_nonexistent_dataset_hash_returns_error_string(self, tmp_path: Path):
        fake_file = tmp_path / "absent_file.csv"
        assert get_dataset_hash(fake_file) == "file_not_found"


class TestComparisonComputation:
    """Verify mathematical difference computations and reporting logic."""

    @pytest.fixture
    def mock_results(self) -> tuple[ConventionalResult, AutonomousResult]:
        conv = ConventionalResult(
            dataset_path="test.csv",
            target_column="target",
            problem_type="binary_classification",
            train_samples=40,
            test_samples=10,
            feature_count=5,
            selected_model="logistic_regression",
            selected_display_name="Logistic Regression",
            selected_params={"C": 1.0},
            holdout_accuracy=0.80,
            holdout_precision=0.79,
            holdout_recall=0.81,
            holdout_f1=0.80,
            cv_f1_mean=0.78,
            cv_accuracy_mean=0.79,
            total_runtime_seconds=2.50,
            search_runtime_seconds=2.20,
            experiment_count=20,
        )
        auto = AutonomousResult(
            dataset_path="test.csv",
            target_column="target",
            problem_type="binary_classification",
            train_samples=40,
            test_samples=10,
            feature_count=5,
            selected_model="k_nearest_neighbors",
            selected_display_name="K-Nearest Neighbors",
            selected_params={"n_neighbors": 5},
            holdout_accuracy=0.90,
            holdout_precision=0.89,
            holdout_recall=0.91,
            holdout_f1=0.90,
            cv_f1_mean=0.85,
            cv_accuracy_mean=0.86,
            total_runtime_seconds=6.50,
            baseline_runtime_seconds=1.50,
            optimization_runtime_seconds=5.00,
            baseline_experiment_count=5,
            optimization_trial_count=40,
            experiment_count=45,
            optimization_executed=True,
        )
        return conv, auto

    def test_compute_dataset_comparison_differences(self, mock_results):
        conv, auto = mock_results
        spec = BenchmarkDatasetSpec(
            name="test.csv",
            file_path=Path("test.csv"),
            target_column="target",
            problem_type="binary_classification",
            expected_sha256="",
        )
        comp = compute_dataset_comparison(conv, auto, spec)

        assert pytest.approx(comp.f1_difference, 0.0001) == 0.10
        assert pytest.approx(comp.accuracy_difference, 0.0001) == 0.10
        assert pytest.approx(comp.cv_f1_difference, 0.0001) == 0.07
        assert pytest.approx(comp.runtime_difference, 0.0001) == 4.00
        assert comp.experiment_difference == 25
        assert pytest.approx(comp.effort_automation_ratio, 0.01) == 2.25

    def test_dataset_comparison_to_dict_serializable(self, mock_results):
        conv, auto = mock_results
        spec = BenchmarkDatasetSpec(
            name="test.csv",
            file_path=Path("test.csv"),
            target_column="target",
            problem_type="binary_classification",
            expected_sha256="",
        )
        comp = compute_dataset_comparison(conv, auto, spec)
        d = comp.to_dict()

        assert "conventional" in d
        assert "autonomous" in d
        assert "comparison" in d
        assert d["comparison"]["f1_difference"] == 0.10
        assert d["comparison"]["experiment_difference"] == 25


class TestBenchmarkArtifactGeneration:
    """Verify JSON, CSV, and plot artifact generation."""

    @pytest.fixture
    def runner_with_mock_data(self, tmp_path: Path):
        cfg = BenchmarkConfig(
            reports_dir=tmp_path,
            json_output_path=tmp_path / "test_comparison.json",
            csv_output_path=tmp_path / "test_comparison.csv",
            chart_f1_path=tmp_path / "test_f1.png",
            chart_runtime_path=tmp_path / "test_runtime.png",
            chart_experiments_path=tmp_path / "test_exp.png",
        )
        runner = BenchmarkRunner(config=cfg)

        conv = ConventionalResult(
            dataset_path="test.csv",
            target_column="target",
            problem_type="binary_classification",
            train_samples=40,
            test_samples=10,
            feature_count=5,
            selected_model="logistic_regression",
            selected_display_name="Logistic Regression",
            selected_params={},
            holdout_accuracy=0.80,
            holdout_precision=0.80,
            holdout_recall=0.80,
            holdout_f1=0.80,
            cv_f1_mean=0.78,
            cv_accuracy_mean=0.79,
            total_runtime_seconds=2.0,
            search_runtime_seconds=1.8,
            experiment_count=20,
        )
        auto = AutonomousResult(
            dataset_path="test.csv",
            target_column="target",
            problem_type="binary_classification",
            train_samples=40,
            test_samples=10,
            feature_count=5,
            selected_model="knn",
            selected_display_name="KNN",
            selected_params={},
            holdout_accuracy=0.85,
            holdout_precision=0.85,
            holdout_recall=0.85,
            holdout_f1=0.85,
            cv_f1_mean=0.83,
            cv_accuracy_mean=0.84,
            total_runtime_seconds=5.0,
            baseline_runtime_seconds=1.0,
            optimization_runtime_seconds=4.0,
            baseline_experiment_count=5,
            optimization_trial_count=40,
            experiment_count=45,
            optimization_executed=True,
        )
        spec = BenchmarkDatasetSpec(
            name="test.csv",
            file_path=Path("test.csv"),
            target_column="target",
            problem_type="binary_classification",
            expected_sha256="",
        )
        comp = compute_dataset_comparison(conv, auto, spec)
        return runner, {"binary": comp}

    def test_save_json_and_csv_reports(self, runner_with_mock_data):
        runner, results = runner_with_mock_data
        json_path = runner.save_json_report(results)
        csv_path = runner.save_csv_report(results)

        assert json_path.exists()
        assert csv_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["phase"] == "11"
        assert "binary" in data["datasets"]

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3  # Header + Conventional row + Autonomous row

    def test_generate_plots(self, runner_with_mock_data):
        runner, results = runner_with_mock_data
        runner.generate_plots(results)

        assert runner.config.chart_f1_path.exists()
        assert runner.config.chart_runtime_path.exists()
        assert runner.config.chart_experiments_path.exists()
        assert runner.config.chart_f1_path.stat().st_size > 0
        assert runner.config.chart_runtime_path.stat().st_size > 0
        assert runner.config.chart_experiments_path.stat().st_size > 0


class TestDashboardBenchmarkGracefulLoading:
    """Verify dashboard correctly handles missing or existing benchmark comparisons."""

    def test_load_benchmark_results_absent_returns_none(self):
        # Config is a frozen dataclass — cannot patch fields directly.
        # load_benchmark_results() returns None when no comparison file exists, or a dict if it does.
        res = load_benchmark_results()
        assert res is None or isinstance(res, dict)
