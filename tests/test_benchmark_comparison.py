"""Hermetic tests for the Phase 11 benchmark runner and its comparison report.

The runner compares the conventional workflow (tools only, manual grid) against
the autonomous workflow (existing ``run_workflow`` graph) on the same committed
datasets. These tests prove the contract of the comparison:

* the pinned fixture hashes and the separate benchmark MLflow store location
  (never ``experiments/tracking/``) are intact;
* the autonomous-side extraction reads ``FinalModelSelection`` and the recorded
  experiment/trial counts - it does not re-count or re-train anything;
* deltas, relative deltas and verdicts are computed with one consistent rule;
* MLflow recording works against a scratch store and *never* raises when the
  store is unavailable (a failed recording cannot fail the benchmark);
* the JSON/CSV/PNG report artifacts are written deterministically;
* the dashboard section reads the report and never triggers a workflow run;
* ``app.py --benchmark`` dispatches without building any agent.

Most tests stub the two expensive workflows with small in-memory records and
run everything else for real; one integration test runs the true conventional
workflow against a true (single-trial) autonomous workflow on the binary
fixture, isolated in a temporary directory and a scratch MLflow store.

No test writes into ``models/`` or ``experiments/tracking/`` (guarded by the
session fixtures in ``tests/conftest.py``): autonomous persistence goes to a
scratch directory and benchmark MLflow goes to a scratch sqlite store.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from benchmarks import benchmark_config
from benchmarks.benchmark_runner import (
    record_benchmark_mLflow,
    run_dataset_benchmark,
    run_benchmark,
)
from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash
from utils import config

BINARY_SHA = benchmark_config.EXPECTED_DATASET_SHA256["classification"]
MULTICLASS_SHA = benchmark_config.EXPECTED_DATASET_SHA256["multiclass"]


def _scratch_dir() -> Path:
    """A temporary directory that survives SQLite engine locks.

    MLflow keeps the SQLAlchemy engine (and therefore a file handle) alive after
    a client is done with the store, so ``TemporaryDirectory`` cleanup would
    fail on Windows. A best-effort ``rmtree(ignore_errors=True)`` at the end of
    each test leaves the locked file for the OS; it is removed when the process
    exits.
    """
    path = Path(tempfile.mkdtemp(prefix="phase11_test_"))
    return path


def _drop_scratch_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def make_stub_workflow_result():
    """A realistic completed workflow result with controlled numbers.

    Mirrors what a real set of agents would record for the binary fixture: five
    baseline experiments, five completed Optuna trials, and an optimized SVM
    winner whose holdout scores match the conventional grid's SVM C=10 config.
    """
    from graph.state import NodeRecord, NodeStatus, WorkflowResult, WorkflowStatus
    from utils.state import (
        AgentState,
        DataSplitInfo,
        EvaluationReport,
        ExperimentCount,
        FinalModelSelection,
        FinalModelSource,
        ModelEvaluation,
        OptimizationCounts,
        OptimizationReport,
        PersistenceReport,
    )

    agent_state = AgentState(
        target_column="Outcome",
        problem_type="binary_classification",
        data_split=DataSplitInfo(
            test_size=0.2,
            random_state=42,
            stratified=True,
            stratification_reason="",
            train_samples=96,
            test_samples=24,
            total_samples=120,
            duplicate_rows_detected=3,
            duplicate_rows_removed=3,
            duplicate_handling_strategy="remove_exact_duplicates",
            input_feature_count=7,
        ),
        evaluation_report=EvaluationReport(
            overall_status="completed",
            problem_type="binary_classification",
            test_samples=24,
            experiment_count=ExperimentCount(
                baseline_models_attempted=5,
                successful_experiments=5,
                partial_experiments=0,
                failed_experiments=0,
                total_experiments=5,
                hyperparameter_experiments=0,
            ),
            model_evaluations=[
                ModelEvaluation(
                    model_name="Logistic Regression",
                    precision=0.8761,
                    recall=0.8750,
                )
            ],
        ),
        optimization_report=OptimizationReport(
            overall_status="completed",
            n_trials_per_model=5,
            counts=OptimizationCounts(
                baseline_models_attempted=5,
                baseline_successful_experiments=5,
                baseline_failed_experiments=0,
                optimized_models_attempted=2,
                successful_optimizations=2,
                failed_optimizations=0,
                completed_trials=5,
                failed_trials=0,
                pruned_trials=0,
                total_trials=5,
            ),
        ),
        persistence_report=PersistenceReport(
            overall_status="completed",
            dataset_name="sample_classification.csv",
            target_column="Outcome",
            task_type="binary_classification",
            feature_count=7,
            selected_f1_score=0.9578,
            selected_accuracy=0.9583,
            selection=FinalModelSelection(
                model_name="Support Vector Machine",
                selection_source=FinalModelSource.OPTIMIZED.value,
                baseline_model_name="Logistic Regression",
                baseline_cv_f1_mean=0.91,
                optimized_cv_f1_mean=0.93,
                best_params={"C": 10.0, "kernel": "linear"},
            ),
        ),
    )
    return WorkflowResult(
        workflow_status=WorkflowStatus.COMPLETED,
        agent_state=agent_state,
        node_records=[
            NodeRecord(
                node="training",
                display_name="Training Agent",
                status=NodeStatus.COMPLETED,
                attempts=1,
                retries=0,
                duration=1.25,
            ),
            NodeRecord(
                node="optimization",
                display_name="Optimization Agent",
                status=NodeStatus.COMPLETED,
                attempts=1,
                retries=0,
                duration=3.5,
            ),
            NodeRecord(
                node="persistence",
                display_name="Model Persistence Agent",
                status=NodeStatus.COMPLETED,
                attempts=1,
                retries=0,
                duration=0.2,
            ),
        ],
    )


def fabricated_conventional():
    """A conventional result shaped like the real workflow's, without training."""
    from utils.state import BestModelInfo, ExperimentResult, ExperimentStatus

    experiment = ExperimentResult(
        experiment_id="benchmark-conv-svm",
        model_name="Support Vector Machine",
        model_type="kernel_based",
        preprocessing_scaling=True,
        parameters={"C": 10.0, "kernel": "linear", "probability": False},
        input_feature_count=7,
        train_samples=96,
        test_samples=24,
        accuracy=0.9583,
        precision=0.96,
        recall=0.9583,
        f1_score=0.9578,
        cv_accuracy_mean=0.95,
        cv_f1_mean=0.93,
        status=ExperimentStatus.SUCCESS.value,
    )
    best = BestModelInfo(
        model_name="Support Vector Machine",
        experiment_id="benchmark-conv-svm",
        f1_score=0.9578,
        accuracy=0.9583,
        cv_f1_mean=0.93,
        preprocessing_scaling=True,
        parameters={"C": 10.0, "kernel": "linear", "probability": False},
        selection_reason="highest f1_score (0.9578)",
    )
    from benchmarks.conventional_workflow import ConventionalWorkflowResult

    return ConventionalWorkflowResult(
        dataset_name="sample_classification.csv",
        dataset_path=str(BINARY_DATASET),
        dataset_sha256=BINARY_SHA,
        target_column="Outcome",
        problem_type="binary_classification",
        problem_type_reason="binary label column",
        feature_count=7,
        train_samples=96,
        test_samples=24,
        split_stratified=True,
        split_reason="",
        candidate_families=[
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "K-Nearest Neighbors",
            "Support Vector Machine",
        ],
        grid_configurations=1,
        experiments=[experiment],
        successful_experiments=1,
        failed_experiments=0,
        best=best,
        best_experiment=experiment,
        selection_reason="highest f1_weighted (0.9578)",
        stage_times={"experiments": 10.0},
        runtime_seconds=11.0,
    )


class DataIntegrityTests(unittest.TestCase):
    """The benchmark only ever runs against the pinned bytes."""

    def test_pinned_hashes_match_the_committed_fixtures(self):
        self.assertEqual(file_hash(BINARY_DATASET), BINARY_SHA)
        self.assertEqual(file_hash(MULTICLASS_DATASET), MULTICLASS_SHA)

    def test_the_benchmark_mLflow_store_is_never_the_canonical_store(self):
        self.assertNotEqual(
            benchmark_config.BENCHMARK_MLFLOW_TRACKING_URI, config.MLFLOW_TRACKING_URI
        )
        self.assertIn("tracking_benchmark", benchmark_config.BENCHMARK_MLFLOW_TRACKING_URI)
        self.assertIn(
            "experiments/tracking/mlflow.db", config.MLFLOW_TRACKING_URI
        )
        self.assertNotIn(
            "experiments/tracking/mlflow.db", benchmark_config.BENCHMARK_MLFLOW_TRACKING_URI
        )

    def test_expected_dataset_keys_cover_both_fixtures(self):
        self.assertEqual(
            list(benchmark_config.ALL_DATASET_KEYS), ["classification", "multiclass"]
        )


class AutonomousExtractionTests(unittest.TestCase):
    """The autonomous side is read, not re-counted."""

    def test_extraction_reads_the_final_selection_and_the_recorded_counts(self):
        from benchmarks import benchmark_runner as module

        with mock.patch.object(
            module, "run_workflow", return_value=make_stub_workflow_result()
        ) as fake_run:
            record = module._run_autonomous_side(
                "classification", str(BINARY_DATASET), n_trials=5
            )
        fake_run.assert_called_once()
        self.assertEqual(record.selected_model, "Support Vector Machine")
        self.assertEqual(record.selection_source, "optimized")
        self.assertEqual(record.f1_score, 0.9578)
        self.assertEqual(record.accuracy, 0.9583)
        self.assertEqual(record.cv_f1_mean, 0.93)
        self.assertEqual(record.parameters, {"C": 10.0, "kernel": "linear"})
        self.assertEqual(record.baseline_experiments, 5)
        self.assertEqual(record.trials, 5)
        self.assertEqual(record.experiments, 10)
        self.assertEqual(record.problem_type, "binary_classification")
        self.assertEqual(record.train_samples, 96)
        self.assertEqual(record.test_samples, 24)
        self.assertEqual(record.node_durations["training"], 1.25)

    def test_the_autonomous_run_recovers_split_facts_from_data_split(self):
        from benchmarks import benchmark_runner as module

        with mock.patch.object(
            module, "run_workflow", return_value=make_stub_workflow_result()
        ):
            record = module._run_autonomous_side(
                "classification", str(BINARY_DATASET), n_trials=5
            )
        self.assertTrue(record.split_stratified)
        self.assertEqual(record.feature_count, 7)


class ComparisonArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.conventional = fabricated_conventional()
        self.autonomous_record = _autonomous_summary_record()
        self.comparison = _build_comparison_from(self.conventional, self.autonomous_record)

    def test_deltas_are_autonomous_minus_conventional(self):
        self.assertEqual(self.comparison.delta_f1, 0.0)
        self.assertEqual(self.comparison.delta_accuracy, 0.0)
        self.assertAlmostEqual(self.comparison.delta_runtime, 12.0)
        self.assertEqual(self.comparison.delta_experiments, 9)  # 10 - 1
        self.assertEqual(self.comparison.f1_relative_percent, 0.0)

    def test_verdict_reflects_the_measured_delta(self):
        self.assertEqual(
            self.comparison.verdict,
            "autonomous matched or exceeded conventional F1 (delta_f1=+0.0000)",
        )

    def test_positive_delta_yields_the_same_verdict_wording(self):
        autonomous = _autonomous_summary_record()
        autonomous.f1_score = 0.96
        autonomous.accuracy = 0.96
        comparison = _build_comparison_from(self.conventional, autonomous)
        self.assertIn("autonomous matched or exceeded", comparison.verdict)

    def test_negative_delta_names_the_conventional_side(self):
        autonomous = _autonomous_summary_record()
        autonomous.f1_score = 0.90
        autonomous.accuracy = 0.91
        comparison = _build_comparison_from(self.conventional, autonomous)
        self.assertIn("conventional scored higher F1", comparison.verdict)
        self.assertLess(comparison.delta_f1, 0.0)


def _autonomous_summary_record():
    from benchmarks.benchmark_runner import WorkflowBenchmarkResult

    return WorkflowBenchmarkResult(
        workflow="autonomous",
        dataset_key="classification",
        dataset_sha=BINARY_SHA,
        selected_model="Support Vector Machine",
        selection_source="optimized",
        f1_score=0.9578,
        accuracy=0.9583,
        cv_f1_mean=0.93,
        parameters={"C": 10.0, "kernel": "linear"},
        runtime_seconds=23.0,
        experiments=10,
        baseline_experiments=5,
        trials=5,
        target_column="Outcome",
        problem_type="binary_classification",
        feature_count=7,
        train_samples=96,
        test_samples=24,
        node_durations={"optimization": 3.5},
        status="success",
    )


def _build_comparison_from(conventional, autonomous):
    from benchmarks.benchmark_runner import _build_comparison

    return _build_comparison("classification", BINARY_SHA, conventional, autonomous)


class MLflowRecordingTests(unittest.TestCase):
    def test_happy_path_records_both_workflows_into_the_given_store(self):
        scratch = _scratch_dir()
        try:
            store = scratch / "benchmark.mlflow.db"
            comparison = _build_comparison_from(
                fabricated_conventional(), _autonomous_summary_record()
            )
            outcome = record_benchmark_mLflow(
                comparison,
                tracking_uri=f"sqlite:///{store.as_posix()}",
                experiment_name="test-benchmark",
            )
            self.assertTrue(outcome["recorded"])
            self.assertEqual(len(outcome["run_ids"]), 2)

            from mlflow.client import MlflowClient

            client = MlflowClient(tracking_uri=f"sqlite:///{store.as_posix()}")
            experiment = client.get_experiment_by_name("test-benchmark")
            self.assertIsNotNone(experiment)
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            self.assertEqual(len(runs), 2)
            workflows = {run.data.tags["workflow"] for run in runs}
            self.assertEqual(workflows, {"conventional", "autonomous"})
            autonomous_run = next(
                run for run in runs if run.data.tags["workflow"] == "autonomous"
            )
            self.assertAlmostEqual(autonomous_run.data.metrics["f1_score"], 0.9578)
        finally:
            _drop_scratch_dir(scratch)

    def test_a_blocked_store_never_raises_and_reports_the_failure(self):
        comparison = _build_comparison_from(
            fabricated_conventional(), _autonomous_summary_record()
        )
        with mock.patch(
            "mlflow.client.MlflowClient",
            side_effect=RuntimeError("tracking store blocked"),
        ):
            outcome = record_benchmark_mLflow(
                comparison, tracking_uri="sqlite:///blocked.db", experiment_name="x"
            )
        self.assertFalse(outcome["recorded"])
        self.assertIn("error", outcome)
        self.assertIn("blocked", outcome["error"])

    def test_a_malformed_tracking_uri_is_also_reported_not_raised(self):
        comparison = _build_comparison_from(
            fabricated_conventional(), _autonomous_summary_record()
        )
        outcome = record_benchmark_mLflow(
            comparison, tracking_uri="sqlite:///Z:/definitely/missing/dir/nope.db"
        )
        self.assertFalse(outcome["recorded"])
        self.assertIn("error", outcome)


class ReportArtifactTests(unittest.TestCase):
    def test_run_benchmark_writes_json_csv_and_three_charts(self):
        """Full pipeline with both workflows stubbed; artifacts written for real."""
        from benchmarks import benchmark_runner as module

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            with mock.patch.object(
                module, "run_workflow", return_value=make_stub_workflow_result()
            ), mock.patch.object(
                module, "run_conventional_workflow", return_value=fabricated_conventional()
            ):
                suite = run_benchmark(
                    ["classification"],
                    n_trials=5,
                    benchmark_mLflow=False,
                    scratch_dir=str(tmp / "autonomous_scratch"),
                    output_dir=tmp / "out",
                )

            comparison = suite.comparison_for("classification")
            self.assertIsNotNone(comparison)
            self.assertEqual(comparison.conventional.experiments, 1)
            self.assertEqual(comparison.autonomous.experiments, 10)

            json_path = tmp / "out" / "phase11_comparison.json"
            csv_path = tmp / "out" / "phase11_comparison.csv"
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            for chart in (
                "phase11_f1_comparison.png",
                "phase11_runtime_comparison.png",
                "phase11_experiment_comparison.png",
            ):
                self.assertTrue((tmp / "out" / chart).is_file(), chart)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            datasets = payload["datasets"]
            self.assertIn("classification", datasets)
            self.assertEqual(
                datasets["classification"]["conventional"]["f1_score"], 0.9578
            )
            self.assertEqual(
                datasets["classification"]["autonomous"]["experiments"], 10
            )

            rows = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0].split(",")[0], "dataset")
            self.assertEqual(len(rows), 3)  # header + two workflows

    def test_the_reports_are_written_into_an_explicit_output_dir(self):
        from benchmarks import benchmark_runner as module

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            with mock.patch.object(
                module, "run_workflow", return_value=make_stub_workflow_result()
            ), mock.patch.object(
                module, "run_conventional_workflow", return_value=fabricated_conventional()
            ):
                suite = run_benchmark(
                    ["classification"],
                    n_trials=5,
                    benchmark_mLflow=False,
                    output_dir=tmp / "custom_reports",
                )
            self.assertTrue((tmp / "custom_reports" / "phase11_comparison.json").is_file())
            for written in suite.files_written:
                self.assertTrue(Path(written).is_file())


class IntegrationTests(unittest.TestCase):
    """One honest end-to-end run, small on purpose (single Optuna trial)."""

    def test_the_real_workflows_complete_on_the_binary_fixture_in_isolation(self):
        scratch = _scratch_dir()
        try:
            comparison = run_dataset_benchmark(
                "classification",
                n_trials=1,
                benchmark_mLflow=True,
                tracking_uri=f"sqlite:///{(scratch / 'benchmark.mlflow.db').as_posix()}",
                experiment_name="test-integration",
                scratch_dir=str(scratch / "autonomous_scratch"),
            )
            self.assertEqual(comparison.dataset_sha256, BINARY_SHA)
            self.assertEqual(comparison.autonomous.status, "success")
            self.assertIsNotNone(comparison.autonomous.f1_score)
            self.assertGreater(comparison.autonomous.runtime_seconds, 0.0)
            # One experiment per model: five baselines plus one trial per winner.
            self.assertEqual(comparison.autonomous.baseline_experiments, 5)
            self.assertEqual(comparison.autonomous.trials, 2)
            self.assertEqual(comparison.autonomous.experiments, 7)
            self.assertNotEqual(comparison.conventional.experiments, 0)
            self.assertEqual(comparison.conventional.experiments, 21)
            self.assertTrue(comparison.conventional.selected_model)
            self.assertIsNotNone(comparison.conventional.f1_score)
        finally:
            _drop_scratch_dir(scratch)


class DashboardAndCliTests(unittest.TestCase):
    def test_the_dashboard_reports_no_benchmark_when_the_file_is_absent(self):
        from tools.dashboard_tools import build_benchmark_section

        section = build_benchmark_section(str(Path(tempfile.mkdtemp()) / "missing.json"))
        self.assertFalse(section["available"])
        self.assertEqual(section["note"], config.DASHBOARD_NO_BENCHMARK)

    def test_the_dashboard_reads_a_present_report_without_training(self):
        from tools.dashboard_tools import build_benchmark_section

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            with mock.patch(
                "benchmarks.benchmark_runner.run_workflow",
                return_value=make_stub_workflow_result(),
            ), mock.patch(
                "benchmarks.benchmark_runner.run_conventional_workflow",
                return_value=fabricated_conventional(),
            ):
                run_benchmark(
                    ["classification"],
                    n_trials=5,
                    benchmark_mLflow=False,
                    output_dir=tmp,
                )
            section = build_benchmark_section(str(tmp / "phase11_comparison.json"))
        self.assertTrue(section["available"])
        self.assertEqual(len(section["datasets"]), 1)
        dataset = section["datasets"][0]
        self.assertEqual(len(dataset["rows"]), 2)
        self.assertEqual(dataset["rows"][0]["workflow"], "Conventional")
        self.assertEqual(dataset["rows"][1]["workflow"], "Autonomous")

    def test_the_dashboard_view_never_triggers_a_workflow(self):
        """The benchmark section is read-only by construction; prove it on source."""
        view = (
            Path(__file__).resolve().parent.parent / "dashboard.py"
        ).read_text(encoding="utf-8")
        body = view[view.index("def render_benchmark(") : view.index("def render_security_note(")]
        self.assertIn("helpers.build_benchmark_section()", body)
        for forbidden in ("run_benchmark(", "run_workflow(", "run_dataset_benchmark("):
            self.assertNotIn(forbidden, body)

    def test_cli_benchmark_dispatches_without_running_agents(self):
        import app

        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            with mock.patch("benchmarks.benchmark_runner.run_benchmark") as fake_runner:
                fake_runner.return_value = _fake_suite(tmp)
                output = StringIO()
                with redirect_stdout(output):
                    code = app.main(
                        ["--benchmark", "--dataset", "classification", "--no-mlflow"]
                    )
                self.assertEqual(code, 0)
            self.assertIn("Project Status: Phase 11", output.getvalue())
            self.assertIn("Benchmark complete.", output.getvalue())
            args, kwargs = fake_runner.call_args
            self.assertEqual(list(args[0]), ["classification"])
            self.assertFalse(kwargs["benchmark_mLflow"])

    def test_cli_benchmark_all_runs_both_keys(self):
        import app

        with mock.patch("benchmarks.benchmark_runner.run_benchmark") as fake_runner:
            fake_runner.return_value = _fake_suite(Path(tempfile.mkdtemp()))
            app.main(["--benchmark", "--all", "--no-mlflow"])
        args, kwargs = fake_runner.call_args
        self.assertEqual(list(args[0]), ["classification", "multiclass"])
        self.assertFalse(kwargs["benchmark_mLflow"])


def _fake_suite(directory: Path):
    """A minimal suite that satisfies the CLI's success path."""
    from benchmarks.benchmark_runner import BenchmarkSuiteResult

    return BenchmarkSuiteResult(files_written=[])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()