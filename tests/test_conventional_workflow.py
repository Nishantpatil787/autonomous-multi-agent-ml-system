"""Isolated tests for the conventional (manual) side of the Phase 11 benchmark.

The conventional workflow is the artifact-free, tool-only side of the Phase 11
comparison. These tests prove the properties that make the comparison fair:

* the committed fixtures are byte-identical to the hashes the benchmark
  configuration pins (data integrity);
* the manual grid is exactly the five registry families and 21 configurations,
  every one derived from the Phase 6 search-space bounds;
* the conventional pass reproduces the same split the agents use, because it
  calls the same ``training_tools.split_dataset`` defaults;
* on the binary fixture it reproduces the canonical Phase 4/5 baseline number
  (Logistic Regression C=1.0 -> 0.8761 / 0.8750) and the Phase 6 optimized
  winner (SVM linear C=10 -> 0.9578 / 0.9583), i.e. the exact numbers the
  canonical MLflow store records;
* the module imports tools only - no agents, no LangGraph, no LLM, no Optuna;
* a failing configuration is recorded, never fatal;
* two runs agree on every metric, so the comparison is reproducible.

Timing is deliberately not asserted: the workflow measures wall-clock time and
two runs must agree on every *metric* but never on the runtime.

No test here writes a model, a report or an MLflow run: the conventional
workflow persists nothing, which is one of the properties under test.
"""

from __future__ import annotations

import unittest

from benchmarks import benchmark_config
from benchmarks.benchmark_config import build_conventional_grid
from benchmarks.conventional_workflow import run_conventional_workflow
from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, PROJECT_ROOT, file_hash
from tools.model_tools import get_supported_models
from utils import config
from utils.state import ExperimentResult, ExperimentStatus


def _lr_grid(c_value: float) -> dict:
    """A one-configuration grid for Logistic Regression from the Phase 6 bounds."""
    return {
        "Logistic Regression": [
            {
                "C": c_value,
                "max_iter": config.OPTUNA_LOGISTIC_MAX_ITER,
                "random_state": config.RANDOM_STATE,
            }
        ]
    }


class BenchmarkConfigurationTests(unittest.TestCase):
    """The pinned datasets and the manual grid definition."""

    def test_committed_fixtures_match_the_pinned_hashes(self):
        """Data integrity: the benchmark runs only on byte-identical fixtures."""
        for key, path in benchmark_config.DATASET_FILES.items():
            with self.subTest(dataset=key):
                self.assertTrue(path.is_file(), f"{path} must exist")
                self.assertEqual(
                    file_hash(path),
                    benchmark_config.EXPECTED_DATASET_SHA256[key],
                    f"{path} differs from the Phase 11 pinned hash",
                )

    def test_resolve_dataset_path_points_at_the_committed_fixture(self):
        for key in benchmark_config.ALL_DATASET_KEYS:
            with self.subTest(dataset=key):
                resolved = benchmark_config.resolve_dataset_path(key)
                self.assertEqual(resolved.resolve(), benchmark_config.DATASET_FILES[key].resolve())

    def test_the_default_grid_is_five_families_and_twenty_one_configurations(self):
        grid = build_conventional_grid()
        self.assertEqual(set(grid), set(get_supported_models()))
        self.assertEqual(len(grid), 5)
        total = sum(len(configs) for configs in grid.values())
        self.assertEqual(total, 21)
        self.assertEqual(benchmark_config.conventional_grid_size(), 21)

    def test_every_grid_configuration_lies_inside_the_phase6_search_space(self):
        """The grid is derived from the search bounds, not invented elsewhere."""
        grid = build_conventional_grid()
        for parameters in grid.get("Logistic Regression", []):
            self.assertGreaterEqual(parameters["C"], config.OPTUNA_LOGISTIC_C_RANGE[0])
            self.assertLessEqual(parameters["C"], config.OPTUNA_LOGISTIC_C_RANGE[1])
            self.assertEqual(parameters["max_iter"], config.OPTUNA_LOGISTIC_MAX_ITER)
        for parameters in grid.get("Decision Tree", []):
            self.assertGreaterEqual(parameters["max_depth"], config.OPTUNA_TREE_MAX_DEPTH_RANGE[0])
            self.assertLessEqual(parameters["max_depth"], config.OPTUNA_TREE_MAX_DEPTH_RANGE[1])
        for parameters in grid.get("Random Forest", []):
            self.assertGreaterEqual(
                parameters["n_estimators"], config.OPTUNA_FOREST_N_ESTIMATORS_RANGE[0]
            )
            self.assertLessEqual(
                parameters["n_estimators"], config.OPTUNA_FOREST_N_ESTIMATORS_RANGE[1]
            )
            self.assertIn(
                parameters["max_depth"], config.OPTUNA_FOREST_MAX_DEPTH_CHOICES
            )
        for parameters in grid.get("K-Nearest Neighbors", []):
            self.assertGreaterEqual(
                parameters["n_neighbors"], config.OPTUNA_KNN_N_NEIGHBORS_RANGE[0]
            )
            self.assertLessEqual(
                parameters["n_neighbors"], config.OPTUNA_KNN_N_NEIGHBORS_RANGE[1]
            )
        for parameters in grid.get("Support Vector Machine", []):
            self.assertGreaterEqual(parameters["C"], config.OPTUNA_SVM_C_RANGE[0])
            self.assertLessEqual(parameters["C"], config.OPTUNA_SVM_C_RANGE[1])
            self.assertIn(parameters["kernel"], config.OPTUNA_SVM_KERNEL_CHOICES)

    def test_the_conventional_module_imports_tools_only(self):
        """The manual side must not smuggle in the autonomous machinery."""
        source = (
            PROJECT_ROOT / "benchmarks" / "conventional_workflow.py"
        ).read_text(encoding="utf-8")
        import_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        for forbidden in (
            "import agents",
            "from agents",
            "import graph",
            "from graph",
            "import llm",
            "from llm",
            "import optuna",
            "from optuna",
            "tools.optimization_tools",
        ):
            self.assertNotIn(forbidden, import_lines)


class ConventionalWorkflowBehaviourTests(unittest.TestCase):
    """The workflow itself, run on the committed fixtures with small grids."""

    def test_binary_split_is_the_projects_split(self):
        result = run_conventional_workflow(BINARY_DATASET, grid=_lr_grid(1.0))
        self.assertEqual(result.train_samples, 96)
        self.assertEqual(result.test_samples, 24)
        self.assertTrue(result.split_stratified)
        self.assertEqual(result.target_column, "Outcome")
        self.assertEqual(result.problem_type, "binary_classification")
        self.assertEqual(
            result.dataset_sha256, benchmark_config.EXPECTED_DATASET_SHA256["classification"]
        )
        self.assertEqual(result.experiment_count, 1)
        self.assertEqual(result.successful_experiments, 1)

    def test_the_grid_reproduces_the_canonical_baseline_and_optimized_results(self):
        """Fairness anchor: identical split -> identical canonical numbers.

        Logistic Regression C=1.0 must reproduce the Phase 4/5 baseline
        (0.8761 / 0.8750) and SVM linear C=10 the Phase 6 optimized winner
        (0.9578 / 0.9583) that the canonical MLflow store records.
        """
        grid = {
            **_lr_grid(1.0),
            "Support Vector Machine": [
                {
                    "C": 10.0,
                    "kernel": "linear",
                    "probability": False,
                    "random_state": config.RANDOM_STATE,
                }
            ],
        }
        result = run_conventional_workflow(BINARY_DATASET, grid=grid)
        rows = {experiment.model_name: experiment for experiment in result.experiments}
        logistic = rows["Logistic Regression"]
        svm = rows["Support Vector Machine"]
        self.assertAlmostEqual(logistic.f1_score, 0.8761, places=3)
        self.assertAlmostEqual(logistic.accuracy, 0.8750, places=3)
        self.assertAlmostEqual(svm.f1_score, 0.9578, places=3)
        self.assertAlmostEqual(svm.accuracy, 0.9583, places=3)
        # The project's ranking rule picks the higher F1 configuration.
        self.assertEqual(result.best.model_name, "Support Vector Machine")
        self.assertEqual(result.best.f1_score, svm.f1_score)

        # Both experiments carry the CV means a fair comparison needs.
        self.assertIsNotNone(logistic.cv_f1_mean)
        self.assertIsNotNone(svm.cv_f1_mean)

    def test_multiclass_reproduces_the_canonical_decision_tree_result(self):
        grid = {"Decision Tree": [{"random_state": config.RANDOM_STATE}]}
        result = run_conventional_workflow(MULTICLASS_DATASET, grid=grid)
        self.assertEqual(result.target_column, "class_label")
        self.assertEqual(result.problem_type, "multiclass_classification")
        self.assertTrue(result.split_stratified)
        self.assertEqual(result.successful_experiments, 1)
        self.assertEqual(
            result.dataset_sha256, benchmark_config.EXPECTED_DATASET_SHA256["multiclass"]
        )
        self.assertAlmostEqual(result.best.f1_score, 1.0, places=2)
        self.assertAlmostEqual(result.best.accuracy, 1.0, places=2)

    def test_metrics_are_deterministic_across_two_runs(self):
        first = run_conventional_workflow(BINARY_DATASET, grid=_lr_grid(10.0))
        second = run_conventional_workflow(BINARY_DATASET, grid=_lr_grid(10.0))
        first_record = first.experiments[0]
        second_record = second.experiments[0]
        self.assertEqual(first_record.f1_score, second_record.f1_score)
        self.assertEqual(first_record.accuracy, second_record.accuracy)
        self.assertEqual(first_record.cv_f1_mean, second_record.cv_f1_mean)
        self.assertEqual(first_record.experiment_id, second_record.experiment_id)
        # Runtime is measured wall-clock time: it must never be asserted equal.
        self.assertNotEqual(
            first.runtime_seconds, second.runtime_seconds, "runtimes may coincide"
        )

    def test_the_selection_uses_the_projects_ranking_rule(self):
        grid = {
            **_lr_grid(1.0),
            "Decision Tree": [{"max_depth": 8, "random_state": config.RANDOM_STATE}],
        }
        result = run_conventional_workflow(BINARY_DATASET, grid=grid)
        best_f1 = max(
            experiment.f1_score
            for experiment in result.experiments
            if experiment.status == ExperimentStatus.SUCCESS.value
        )
        self.assertEqual(result.best.f1_score, best_f1)
        self.assertEqual(result.best.experiment_id, result.best_experiment.experiment_id)
        self.assertIn("highest f1_score", result.selection_reason)

    def test_a_failed_configuration_is_recorded_not_fatal(self):
        grid = {
            "Logistic Regression": [
                # C <= 0 is rejected by the estimator at fit time.
                {"C": -1.0, "max_iter": 200, "random_state": config.RANDOM_STATE},
                {"C": 1.0, "max_iter": 200, "random_state": config.RANDOM_STATE},
            ]
        }
        result = run_conventional_workflow(BINARY_DATASET, grid=grid)
        self.assertEqual(result.experiment_count, 2)
        self.assertEqual(result.successful_experiments, 1)
        self.assertEqual(result.failed_experiments, 1)
        failed = [e for e in result.experiments if e.status == ExperimentStatus.FAILED.value]
        self.assertEqual(len(failed), 1)
        self.assertIn("C", failed[0].parameters)
        self.assertTrue(failed[0].error_message)
        # The workflow still selected the healthy configuration.
        self.assertEqual(result.best.model_name, "Logistic Regression")
        self.assertEqual(result.best.parameters["C"], 1.0)

    def test_experiment_records_are_typed_and_have_unique_ids(self):
        grid = {
            **_lr_grid(0.1),
            "K-Nearest Neighbors": [{"n_neighbors": 3}],
        }
        result = run_conventional_workflow(BINARY_DATASET, grid=grid)
        ids = [experiment.experiment_id for experiment in result.experiments]
        self.assertEqual(len(ids), len(set(ids)))
        for experiment in result.experiments:
            self.assertIsInstance(experiment, ExperimentResult)
            self.assertEqual(experiment.train_samples, 96)
            self.assertEqual(experiment.test_samples, 24)
            self.assertGreater(experiment.training_time, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()