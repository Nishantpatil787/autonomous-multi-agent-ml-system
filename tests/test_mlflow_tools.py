"""Tests for the Phase 10 MLflow tracking layer.

The suite is organised around the claims Phase 10 makes rather than around the
functions, so that a failure names the property that broke:

* **Non-intrusive** - the module decides nothing and imports no agent or graph.
* **Total** - every public entry point returns rather than raises, including on
  a state that stopped before most stages ran.
* **Isolated** - tracking failure never changes a machine learning outcome.
* **Honest** - nothing fabricated: absent facts stay absent, never zero.

Where a test needs a real MLflow store it is given a temporary SQLite file.
The project's own store is never touched, which is what keeps the suite
independent of whether tracking has ever been run.
"""

from __future__ import annotations

import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash
from tools import mlflow_tools
from tools.mlflow_tools import (
    TrackingStatus,
    TrackedRun,
    comparison_rows,
    confusion_matrix_record,
    dataset_record,
    evaluation_record,
    experiment_summary,
    file_sha256,
    final_model_record,
    metric_value,
    mlflow_available,
    optimization_record,
    param_value,
    preprocessing_record,
    recent_runs,
    reproducibility_record,
    run_tags,
    slug,
    timing_record,
    track_workflow_run,
)
from utils import config
from utils.state import AgentState

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Hashes of the committed fixtures. Nothing in Phase 10 may change them.
EXPECTED_BINARY_HASH = "49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c"
EXPECTED_MULTICLASS_HASH = (
    "0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def trained_state(dataset: Path, *, optimize: bool = False) -> AgentState:
    """Run the real pipeline up to (and optionally including) optimization.

    Args:
        dataset: The CSV to run on.
        optimize: Whether to also run the hyperparameter search.

    Returns:
        A finished :class:`~utils.state.AgentState`.
    """
    state = DataAnalysisAgent().run(AgentState(dataset_path=str(dataset)))
    state = PreprocessingAgent().run(state)
    state = ModelSelectionAgent().run(state)
    state = TrainingAgent().run(state)
    state = EvaluationAgent().run(state)
    if optimize:
        state = OptimizationAgent().run(state)
    return state


def persisted_state(dataset: Path, output_dir: Path, *, optimize: bool = False):
    """Produce a state that Phase 9 has already persisted.

    Args:
        dataset: The CSV to run on.
        output_dir: A private directory for the Phase 9 artifacts.
        optimize: Whether to also run the hyperparameter search.

    Returns:
        The state, with ``persistence_report`` populated.
    """
    state = trained_state(dataset, optimize=optimize)
    # The directory must be passed explicitly, exactly like the graph node does
    # (``graph/nodes.py``). ``AgentState`` has no ``persistence_output_dir``
    # field, so a dynamic attribute is invisible to the agent: without the
    # argument below the write silently resolves to ``config.MODELS_DIR`` and a
    # test run replaces the canonical deployment artifact.
    state = PersistenceAgent().run(state, output_dir=output_dir)
    return state


#: A single fully completed state, built once and reused by every store-backed
#: test class. Rebuilding it per class would multiply an already slow Optuna
#: search by the number of classes for no extra coverage: the tracking layer
#: only ever reads it, so sharing one immutable-by-convention object is safe.
_SHARED_STATE: Dict[tuple, AgentState] = {}
_SHARED_ARTIFACTS: List[Path] = []


def complete_state(dataset: Path, *, optimize: bool = False) -> AgentState:
    """Return a state representing a finished ``--graph`` run, including Phase 9.

    A complete run is what the tracking layer is designed to describe, so this
    is the default for the tests that assert on what was recorded. Tests that
    care about a run that stopped earlier pass a state of their own.

    Args:
        dataset: The CSV to run on.
        optimize: Whether to also run the hyperparameter search.

    Returns:
        The state, with ``persistence_report`` populated.
    """
    key = (str(dataset), optimize)
    if key not in _SHARED_STATE:
        directory = Path(tempfile.mkdtemp(prefix="phase10-state-"))
        _SHARED_ARTIFACTS.append(directory)
        _SHARED_STATE[key] = persisted_state(dataset, directory, optimize=optimize)
    return _SHARED_STATE[key]


def tearDownModule() -> None:
    """Remove the shared Phase 9 artifact directories."""
    for directory in _SHARED_ARTIFACTS:
        shutil.rmtree(directory, ignore_errors=True)
    _SHARED_ARTIFACTS.clear()
    _SHARED_STATE.clear()


def _release_mlflow() -> None:
    """Close any run a test left open on MLflow's global active-run stack.

    MLflow's fluent API keeps one process-wide stack of active runs. A run left
    open by one test would be the implicit target of the next test's
    ``log_param``, so every test ends by clearing it. This is test hygiene, not
    a product behaviour: nothing in Phase 10 relies on a run staying open.
    """
    try:
        import mlflow

        while mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:  # noqa: BLE001 - cleanup must never fail a test
        pass


class TrackingStoreTestCase(unittest.TestCase):
    """Base class: a temporary MLflow store, removed after every test."""

    #: Whether the expensive optimization search is included in the state.
    optimize = False

    @classmethod
    def setUpClass(cls) -> None:
        """Build one real trained state for the whole class."""
        cls.state = complete_state(BINARY_DATASET, optimize=cls.optimize)

    def setUp(self) -> None:
        """Give each test its own store and its own artifact directory."""
        self.store = Path(tempfile.mkdtemp())
        self.out = self.store / "artifacts"
        self.addCleanup(shutil.rmtree, self.store, ignore_errors=True)
        self.uri = "sqlite:///" + str(self.store / "mlflow.db").replace("\\", "/")
        self.experiment = "phase10-test"
        self.addCleanup(_release_mlflow)

    def open_run(self) -> TrackedRun:
        """Open a run directly, for tests that exercise the guard layer itself.

        Returns:
            A started :class:`~tools.mlflow_tools.TrackedRun`.
        """
        run = TrackedRun(tracking_uri=self.uri, experiment_name=self.experiment)
        self.assertTrue(run.start({}), "the run could not be opened")
        self.addCleanup(run.end, "completed")
        return run

    def track(self, state: Any = None, **overrides: Any):
        """Record a run into this test's store.

        Args:
            state: The state to record, or None for ``self.state``.
            **overrides: Extra keyword arguments for ``track_workflow_run``.

        Returns:
            The :class:`~tools.mlflow_tools.TrackingSummary`.
        """
        options: Dict[str, Any] = {
            "enabled": True,
            "tracking_uri": self.uri,
            "experiment_name": self.experiment,
            "workflow_status": "completed",
        }
        options.update(overrides)
        return track_workflow_run(self.state if state is None else state, None, **options)

    def client(self):
        """Return an MLflow client bound to this test's store.

        Returns:
            An ``MlflowClient``.
        """
        from mlflow.tracking import MlflowClient

        return MlflowClient(tracking_uri=self.uri)

    def runs(self) -> List[Any]:
        """Return every run in this test's experiment.

        Returns:
            The runs, as ``Run`` objects.
        """
        experiment = self.client().get_experiment_by_name(self.experiment)
        if experiment is None:
            return []
        return list(self.client().search_runs([experiment.experiment_id]))


# --------------------------------------------------------------------------- #
# 1. Non-intrusiveness: the layer decides nothing
# --------------------------------------------------------------------------- #
class TestLayering(unittest.TestCase):
    """The structural claims that keep MLflow out of the decision path."""

    def test_it_imports_no_agent_and_no_graph(self):
        """No ML component may be reachable through this module."""
        source = (PROJECT_ROOT / "tools" / "mlflow_tools.py").read_text(
            encoding="utf-8-sig"
        )
        for forbidden in ("from agents", "import agents", "from graph", "import graph"):
            self.assertNotIn(
                forbidden,
                source,
                f"tools/mlflow_tools.py must not import {forbidden!r}: it is an "
                "observer, and importing the ML layer would let it reach back into "
                "a decision.",
            )

    def test_it_does_not_import_a_sklearn_estimator(self):
        """No model may be constructed here; the workflow owns that."""
        source = (PROJECT_ROOT / "tools" / "mlflow_tools.py").read_text(
            encoding="utf-8-sig"
        )
        for forbidden in ("from sklearn", "import sklearn"):
            self.assertNotIn(forbidden, source)

    def test_the_graph_calls_it_only_after_the_run_is_finished(self):
        """The tracking call must sit inside ``run_workflow``, after the result."""
        source = (PROJECT_ROOT / "graph" / "workflow.py").read_text(encoding="utf-8-sig")
        body = source[source.index("def run_workflow(") : source.index("def describe_structure(")]
        result_at = body.index("result = build_workflow_result(final)")
        tracker_at = body.index("_resolve_tracker()")
        self.assertLess(
            result_at,
            tracker_at,
            "run_workflow must finish the workflow and derive its status before "
            "recording it; a run recorded earlier could describe a result that "
            "never happened.",
        )

    def test_the_status_is_derived_before_it_is_recorded(self):
        """Step 17: MLflow mirrors the status the workflow actually derived."""
        source = (PROJECT_ROOT / "graph" / "workflow.py").read_text(encoding="utf-8-sig")
        body = source[source.index("def run_workflow(") : source.index("def describe_structure(")]
        derived_at = body.index('final["workflow_status"] = derive_workflow_status(final).value')
        tracker_at = body.index("_resolve_tracker()")
        self.assertLess(derived_at, tracker_at)

    def test_the_tracking_call_is_guarded(self):
        """A tracker that raises must not propagate into the ML path."""
        source = (PROJECT_ROOT / "graph" / "workflow.py").read_text(encoding="utf-8-sig")
        body = source[source.index("def run_workflow(") : source.index("def describe_structure(")]
        tracker_at = body.index("_resolve_tracker()")
        window = body[tracker_at : tracker_at + 1400]
        self.assertIn(
            "except Exception",
            window,
            "run_workflow must guard the tracking call: 'tracking failure must not "
            "break successful ML execution' is only true if it is enforced here.",
        )

    def test_no_graph_node_was_added_for_tracking(self):
        """A node could not span the run; the graph shape must be unchanged."""
        import graph.workflow as workflow

        self.assertEqual(
            len(workflow.WORKFLOW_ORDER),
            7,
            "Phase 7 defined 7 nodes and Phase 9's persistence node is the last; "
            "Phase 10 must not add an eighth.",
        )
        self.assertNotIn(
            "TRACKING",
            {node.name for node in workflow.GraphNode},
            "Tracking is not a graph node: a node runs after the others, so it "
            "could not open a run that spans the whole workflow.",
        )


# --------------------------------------------------------------------------- #
# 2. Configuration
# --------------------------------------------------------------------------- #
class TestConfiguration(unittest.TestCase):
    """Step 3 and Step 19: the tracking store is local and configurable."""

    def test_the_default_store_is_inside_the_experiments_directory(self):
        self.assertEqual(
            config.MLFLOW_TRACKING_DIR.parent,
            config.EXPERIMENTS_DIR,
            "The tracking store belongs under experiments/, never in models/ or "
            "datasets/, so generated state cannot be mistaken for an artifact.",
        )

    def test_the_default_store_is_not_inside_the_model_directory(self):
        self.assertNotIn(str(config.MLFLOW_TRACKING_DIR), str(config.MODELS_DIR))

    def test_the_default_uri_is_local(self):
        self.assertTrue(
            config.MLFLOW_TRACKING_URI.startswith(("sqlite:", "file:")),
            f"Local-first means a local store, got {config.MLFLOW_TRACKING_URI!r}",
        )

    def test_the_default_uri_carries_no_remote_host(self):
        self.assertNotIn("http://", config.MLFLOW_TRACKING_URI)
        self.assertNotIn("https://", config.MLFLOW_TRACKING_URI)

    def test_the_uri_is_built_with_the_sqlite_scheme(self):
        """MLflow routes a bare file URI to its deprecated filesystem store."""
        self.assertTrue(
            config.MLFLOW_TRACKING_URI.startswith("sqlite:///"),
            "MLflow 3.x refuses to write to the filesystem store and only accepts a "
            "sqlite:/// URI for a local database.",
        )

    def test_the_uri_helper_keeps_the_drive_letter(self):
        from utils.config import _sqlite_uri

        uri = _sqlite_uri(Path("C:/projects/app/experiments/tracking/mlflow.db"))
        self.assertEqual(uri, "sqlite:///C:/projects/app/experiments/tracking/mlflow.db")

    def test_the_experiment_name_is_not_empty(self):
        self.assertTrue(config.MLFLOW_EXPERIMENT_NAME.strip())

    def test_the_history_limit_is_bounded(self):
        self.assertGreater(config.MLFLOW_HISTORY_LIMIT, 0)
        self.assertLessEqual(config.MLFLOW_HISTORY_LIMIT, 100)

    def test_the_model_artifact_logging_is_off_by_default(self):
        """Phase 9's joblib file stays the single deployment artifact."""
        self.assertFalse(
            config.MLFLOW_LOG_MODEL_ARTIFACT,
            "models/best_model.joblib remains the primary deployment artifact; a "
            "second serialized copy would be a second thing to keep in sync.",
        )

    def test_the_version_whitelist_is_closed(self):
        """A full pip freeze would be noise and a disclosure risk."""
        self.assertIn("scikit-learn", config.MLFLOW_VERSION_WHITELIST)
        self.assertIn("optuna", config.MLFLOW_VERSION_WHITELIST)
        self.assertIn("python", config.MLFLOW_VERSION_WHITELIST)
        self.assertNotIn("*", config.MLFLOW_VERSION_WHITELIST)

    def test_the_cli_keeps_a_disable_switch(self):
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8-sig")
        self.assertIn("--no-mlflow", source)


# --------------------------------------------------------------------------- #
# 3. Availability
# --------------------------------------------------------------------------- #
class TestAvailability(unittest.TestCase):
    """Step 18: an absent mlflow is an ordinary state, not an error."""

    def test_the_probe_never_raises(self):
        with mock.patch.object(
            mlflow_tools, "import_mlflow", side_effect=mlflow_tools.TrackingError("no")
        ):
            self.assertFalse(mlflow_available())

    def test_the_probe_reports_true_when_the_import_works(self):
        with mock.patch.object(mlflow_tools, "import_mlflow", return_value=object()):
            self.assertTrue(mlflow_available())

    def test_the_module_imports_without_mlflow(self):
        """The import is deferred, so this file loads on a machine without it."""
        source = (PROJECT_ROOT / "tools" / "mlflow_tools.py").read_text(
            encoding="utf-8-sig"
        )
        top_level = [
            line
            for line in source.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        for line in top_level:
            self.assertNotIn(
                "mlflow",
                line.replace("from tools import mlflow_tools", ""),
                "mlflow must not be imported at module scope; the module has to "
                "load on a machine where the package is absent.",
            )

    def test_a_missing_package_is_reported_as_unavailable(self):
        with mock.patch.object(
            mlflow_tools, "import_mlflow", side_effect=mlflow_tools.TrackingError("x")
        ):
            summary = track_workflow_run(None, None, enabled=True)
        self.assertIs(summary.status, TrackingStatus.UNAVAILABLE)
        self.assertFalse(summary.usable)
        self.assertIn("not installed", summary.reason)


# --------------------------------------------------------------------------- #
# 4. Value coercion
# --------------------------------------------------------------------------- #
class TestCoercion(unittest.TestCase):
    """MLflow rejects values it cannot store, and silently losing a parameter
    would be worse than refusing it."""

    def test_none_is_absent_not_empty(self):
        self.assertIsNone(param_value(None))

    def test_booleans_stay_booleans(self):
        self.assertIs(param_value(True), True)
        self.assertIs(param_value(False), False)

    def test_numbers_stay_numbers(self):
        self.assertEqual(param_value(42), 42)
        self.assertEqual(param_value(0.5), 0.5)

    def test_strings_pass_through(self):
        self.assertEqual(param_value("linear"), "linear")

    def test_a_long_string_is_truncated_to_the_documented_limit(self):
        value = param_value("x" * 10_000)
        self.assertEqual(len(value), config.MLFLOW_MAX_PARAM_VALUE)

    def test_a_sequence_becomes_json(self):
        self.assertEqual(param_value(["a", "b"]), '["a", "b"]')

    def test_a_mapping_becomes_sorted_json(self):
        self.assertEqual(param_value({"b": 1, "a": 2}), '{"a": 2, "b": 1}')

    def test_an_arbitrary_object_becomes_its_text(self):
        class Thing:
            def __str__(self) -> str:
                return "thing"

        self.assertEqual(param_value(Thing()), "thing")

    def test_nan_and_infinity_are_not_parameters(self):
        self.assertIsNone(param_value(float("nan")))
        self.assertIsNone(param_value(float("inf")))
        self.assertIsNone(param_value(float("-inf")))

    def test_a_metric_must_be_a_finite_number(self):
        self.assertEqual(metric_value("0.5"), 0.5)
        self.assertEqual(metric_value(3), 3.0)

    def test_a_boolean_is_not_a_metric(self):
        """True would silently become 1.0 and read as a real measurement."""
        self.assertIsNone(metric_value(True))

    def test_an_unparsable_metric_is_absent(self):
        self.assertIsNone(metric_value("not a number"))
        self.assertIsNone(metric_value(None))
        self.assertIsNone(metric_value(float("nan")))

    def test_slug_flattens_whitespace(self):
        self.assertEqual(slug("a  b\nc"), "a b c")

    def test_slug_removes_path_separators(self):
        """MLflow rejects a key containing a newline or a slash."""
        self.assertNotIn("/", slug("a/b"))
        self.assertNotIn("\\", slug("a\\b"))

    def test_slug_of_empty_text_is_empty(self):
        self.assertEqual(slug(""), "")
        self.assertEqual(slug(None), "")

    def test_slug_respects_the_limit(self):
        self.assertEqual(len(slug("y" * 500)), 200)
        self.assertEqual(len(slug("z" * 500, limit=50)), 50)


# --------------------------------------------------------------------------- #
# 5. Extraction: the honest reader
# --------------------------------------------------------------------------- #
class TestDatasetRecord(unittest.TestCase):
    """Step 7: dataset metadata, never raw data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET)

    def test_it_reports_the_dataset_identity(self):
        record = dataset_record(self.state, BINARY_DATASET)
        self.assertEqual(record["dataset.name"], BINARY_DATASET.name)
        self.assertEqual(record["dataset.target_column"], "Outcome")
        self.assertEqual(record["dataset.task_type"], "binary_classification")

    def test_it_reports_the_shape(self):
        record = dataset_record(self.state, BINARY_DATASET)
        self.assertEqual(record["dataset.rows"], 123)
        self.assertEqual(record["dataset.columns"], 9)

    def test_it_records_the_sha256_of_the_input(self):
        record = dataset_record(self.state, BINARY_DATASET)
        self.assertEqual(record["dataset.sha256"], EXPECTED_BINARY_HASH)

    def test_the_hash_is_the_real_file_hash(self):
        self.assertEqual(file_sha256(BINARY_DATASET), EXPECTED_BINARY_HASH)
        self.assertEqual(file_sha256(MULTICLASS_DATASET), EXPECTED_MULTICLASS_HASH)

    def test_an_unreadable_file_hashes_to_empty_rather_than_raising(self):
        self.assertEqual(file_sha256(self.missing()), "")
        self.assertEqual(file_sha256(None), "")

    def missing(self) -> Path:
        """Return a path that does not exist."""
        return Path(tempfile.gettempdir()) / "phase10-does-not-exist.csv"

    def test_it_reports_the_data_quality_counts(self):
        record = dataset_record(self.state, BINARY_DATASET)
        self.assertEqual(record["dataset.missing_values"], 17)
        self.assertEqual(record["dataset.duplicate_rows"], 3)

    def test_it_contains_no_dataset_row(self):
        record = dataset_record(self.state, BINARY_DATASET)
        text = json.dumps(record, default=str)
        self.assertNotIn("R&D Spend", text)
        self.assertNotIn("Marketing Spend", text)

    def test_it_contains_no_absolute_path(self):
        """An absolute path would identify the developer who ran it."""
        record = dataset_record(self.state, BINARY_DATASET)
        self.assertNotIn("C:\\", json.dumps(record, default=str))

    def test_an_absent_state_is_empty_rather_than_an_error(self):
        self.assertEqual(dataset_record(None), {})


class TestPreprocessingRecord(unittest.TestCase):
    """Step 9: the decisions, not the fitted transformers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET)

    def test_it_reports_the_imputation_and_encoding(self):
        record = preprocessing_record(self.state)
        self.assertEqual(record["preprocessing.numerical_imputation"], "median")
        self.assertEqual(record["preprocessing.categorical_imputation"], "most_frequent")
        self.assertEqual(record["preprocessing.encoding"], "one_hot")

    def test_it_reports_scaling(self):
        record = preprocessing_record(self.state)
        self.assertIs(record["preprocessing.scale_numeric"], True)
        self.assertEqual(record["preprocessing.scaling_strategy"], "standard")

    def test_it_records_why_a_feature_was_excluded(self):
        record = preprocessing_record(self.state)
        self.assertEqual(
            record["preprocessing.excluded.record_id"],
            "identifier_like",
            "A reader has to be able to tell which column went where and why.",
        )

    def test_it_reports_the_feature_counts(self):
        record = preprocessing_record(self.state)
        self.assertEqual(record["preprocessing.numerical_features"], 4)
        self.assertEqual(record["preprocessing.categorical_features"], 3)
        self.assertEqual(record["preprocessing.used_feature_count"], 7)

    def test_it_does_not_serialise_a_fitted_transformer(self):
        """Parameters must be readable in a table, not a pickled object."""
        record = preprocessing_record(self.state)
        for value in record.values():
            self.assertIsInstance(value, (str, int, float, bool))

    def test_it_reports_the_fit_scope(self):
        record = preprocessing_record(self.state)
        self.assertEqual(record["preprocessing.fit_scope"], "training_data_only")

    def test_an_absent_state_is_empty(self):
        self.assertEqual(preprocessing_record(None), {})


class TestComparisonRows(unittest.TestCase):
    """Step 10: one row per model, stored as data rather than as runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET)

    def test_there_is_one_row_per_evaluated_model(self):
        self.assertEqual(len(comparison_rows(self.state)), 5)

    def test_every_row_names_a_model_and_carries_its_metrics(self):
        for row in comparison_rows(self.state):
            self.assertIn("model_name", row)
            for key in (
                "holdout_accuracy",
                "holdout_precision",
                "holdout_recall",
                "holdout_f1_score",
                "cv_accuracy_mean",
                "cv_f1_mean",
            ):
                self.assertIn(key, row, f"{row.get('model_name')} is missing {key}")

    def test_rows_are_ordered_by_the_recorded_rank(self):
        ranks = [row["rank"] for row in comparison_rows(self.state)]
        self.assertEqual(ranks, sorted(ranks))

    def test_the_first_row_is_the_phase_five_winner(self):
        rows = comparison_rows(self.state)
        self.assertEqual(
            rows[0]["model_name"],
            self.state.evaluation_report.best_model.best_baseline_model,
        )

    def test_the_rows_match_the_evaluation_report(self):
        for row in comparison_rows(self.state):
            source = next(
                entry
                for entry in self.state.evaluation_report.model_evaluations
                if entry.model_name == row["model_name"]
            )
            self.assertAlmostEqual(row["holdout_f1_score"], source.f1_score, places=12)

    def test_it_creates_no_experiments_and_no_runs(self):
        """One parent run is the whole point; five runs per run is not."""
        source = (PROJECT_ROOT / "tools" / "mlflow_tools.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertEqual(
            source.count("create_experiment("),
            1,
            "The experiment is created once per recorded run; the models inside a "
            "run are rows in an artifact, not separate runs.",
        )

    def test_an_absent_state_is_empty(self):
        self.assertEqual(comparison_rows(None), [])


class TestEvaluationRecord(unittest.TestCase):
    """Step 11: the Phase 5 outcome, never recomputed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET)

    def test_params_and_metrics_are_separated(self):
        params, metrics = evaluation_record(self.state)
        self.assertIn("evaluation.ranking_rule", params)
        self.assertIn("evaluation.holdout_f1_score", metrics)

    def test_it_reports_the_selected_model_and_its_holdout_metrics(self):
        params, metrics = evaluation_record(self.state)
        self.assertEqual(params["evaluation.selected_model"], "Logistic Regression")
        self.assertAlmostEqual(metrics["evaluation.holdout_f1_score"], 0.8761343012704175)
        self.assertAlmostEqual(metrics["evaluation.holdout_accuracy"], 0.875)

    def test_it_reports_the_cv_metrics(self):
        _, metrics = evaluation_record(self.state)
        self.assertAlmostEqual(metrics["evaluation.cv_f1_mean"], 0.8941460412129413)
        self.assertAlmostEqual(
            metrics["evaluation.cv_accuracy_mean"], 0.8952631578947369
        )

    def test_it_reports_the_generalization_gap_and_its_category(self):
        params, metrics = evaluation_record(self.state)
        self.assertAlmostEqual(metrics["evaluation.generalization_gap"], -0.018012)
        self.assertEqual(params["evaluation.gap_classification"], "close")

    def test_it_reports_the_split_actually_used(self):
        params, _ = evaluation_record(self.state)
        self.assertEqual(params["evaluation.test_samples"], 24)
        self.assertEqual(params["evaluation.train_samples"], 96)
        self.assertIs(params["evaluation.stratified"], True)

    def test_the_confusion_matrix_is_available(self):
        matrix = confusion_matrix_record(self.state)
        self.assertTrue(matrix)
        self.assertEqual(matrix["labels"], ["0", "1"])
        self.assertEqual(matrix["total_samples"], 24)
        self.assertEqual(matrix["correct_predictions"], 21)

    def test_the_matrix_carries_per_class_metrics(self):
        matrix = confusion_matrix_record(self.state)
        self.assertEqual(len(matrix["class_metrics"]), 2)
        for entry in matrix["class_metrics"]:
            self.assertIn("precision", entry)
            self.assertIn("recall", entry)
            self.assertIn("f1_score", entry)

    def test_it_does_not_recompute_the_evaluation(self):
        """The numbers are copied, so a bug here cannot change a metric."""
        _, metrics = evaluation_record(self.state)
        best = self.state.evaluation_report.best_model
        self.assertIs(metrics["evaluation.holdout_f1_score"], best.f1_score)
        self.assertIs(metrics["evaluation.holdout_accuracy"], best.accuracy)

    def test_absent_states_are_empty(self):
        self.assertEqual(evaluation_record(None), ({}, {}))
        self.assertEqual(confusion_matrix_record(None), {})


class TestOptimizationRecord(unittest.TestCase):
    """Step 12: Optuna results are recorded, never run by MLflow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(MULTICLASS_DATASET, optimize=True)

    def test_it_reports_the_search_configuration(self):
        params, _, _ = optimization_record(self.state)
        self.assertEqual(params["optimization.sampler"], "tpe")
        self.assertEqual(params["optimization.sampler_seed"], 42)
        self.assertEqual(params["optimization.n_trials_per_model"], 20)
        self.assertEqual(params["optimization.objective_metric"], "f1_weighted")
        self.assertEqual(params["optimization.cv_n_splits"], 5)

    def test_it_reports_the_best_parameters(self):
        params, _, _ = optimization_record(self.state)
        self.assertTrue(params["optimization.best_params"])
        self.assertEqual(params["optimization.best_model"], "Decision Tree")

    def test_it_reports_the_baseline_and_the_improvement(self):
        _, metrics, _ = optimization_record(self.state)
        self.assertEqual(metrics["optimization.baseline_cv_f1_mean"], 1.0)
        self.assertEqual(metrics["optimization.optimized_cv_f1_mean"], 1.0)
        self.assertEqual(
            metrics["optimization.improvement"],
            0.0,
            "This split is trivially separable, so a zero improvement is the real "
            "result and must not be inflated.",
        )

    def test_it_reports_the_holdout_comparison(self):
        _, metrics, _ = optimization_record(self.state)
        self.assertEqual(metrics["optimization.holdout_f1_score"], 1.0)
        self.assertEqual(metrics["optimization.holdout_f1_delta"], 0.0)

    def test_the_trials_are_recorded_one_per_trial(self):
        _, _, trials = optimization_record(self.state)
        self.assertEqual(len(trials["trials"]), 40, "2 models x 20 trials")

    def test_each_trial_carries_its_number_state_value_and_parameters(self):
        _, _, trials = optimization_record(self.state)
        entry = trials["trials"][0]
        for key in ("trial_number", "state", "value", "params", "model_name"):
            self.assertIn(key, entry)

    def test_it_does_not_run_optuna(self):
        source = (PROJECT_ROOT / "tools" / "mlflow_tools.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertNotIn("import optuna", source)
        self.assertNotIn("optuna.create_study", source)
        self.assertNotIn("optuna.Trial", source)

    def test_an_unoptimized_run_records_no_search_result(self):
        state = trained_state(BINARY_DATASET)
        params, metrics, trials = optimization_record(state)
        self.assertEqual(params, {})
        self.assertEqual(metrics, {})
        self.assertEqual(trials, {})

    def test_an_absent_state_is_empty(self):
        self.assertEqual(optimization_record(None), ({}, {}, {}))


class TestFinalModelRecord(unittest.TestCase):
    """Step 13: reuse Phase 9's selection, never a second policy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp())
        cls.state = persisted_state(BINARY_DATASET, cls.directory)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_it_reports_the_selected_model_and_its_source(self):
        params, _ = final_model_record(self.state)
        self.assertEqual(params["final_model.name"], "Logistic Regression")
        self.assertEqual(params["final_model.selection_source"], "baseline")
        self.assertIs(params["final_model.optimization_used"], False)

    def test_it_reports_the_persistence_status_and_filenames(self):
        params, _ = final_model_record(self.state)
        self.assertEqual(params["persistence.status"], "completed")
        self.assertEqual(params["persistence.model_artifact"], "best_model.joblib")

    def test_it_reports_the_final_fit_strategy(self):
        params, _ = final_model_record(self.state)
        self.assertEqual(
            params["persistence.final_fit_strategy"], "reused_fitted_pipeline"
        )

    def test_it_reports_the_persisted_metrics(self):
        _, metrics = final_model_record(self.state)
        self.assertAlmostEqual(metrics["final_model.holdout_f1_score"], 0.8761343012704175)
        self.assertAlmostEqual(metrics["final_model.holdout_accuracy"], 0.875)

    def test_it_reads_the_existing_selection_rather_than_re_deriving_it(self):
        """The name must be the object Phase 9 chose, by identity."""
        params, _ = final_model_record(self.state)
        self.assertIs(
            params["final_model.name"], self.state.final_model_selection.model_name
        )

    def test_an_absent_state_is_empty(self):
        self.assertEqual(final_model_record(None), ({}, {}))


class TestTimingRecord(unittest.TestCase):
    """Step 15: read the timings Phase 7 already measured."""

    @staticmethod
    def _record(node: str, duration: Optional[float], attempts: int = 1):
        """Build a real ``NodeRecord``.

        The real type is used rather than a stub so this test cannot drift from
        the record shape the workflow actually produces.

        Args:
            node: The node name.
            duration: Seconds the node took, or None when it never ran.
            attempts: How many attempts the node needed.

        Returns:
            The record.
        """
        from graph.state import NodeRecord, NodeStatus

        return NodeRecord(
            node=node,
            display_name=node.replace("_", " ").title(),
            status=NodeStatus.COMPLETED,
            attempts=attempts,
            retries=max(0, attempts - 1),
            duration=duration,
        )

    def test_it_reads_per_node_durations_from_the_workflow_result(self):
        record = self._record("training", 1.25)
        node_metrics, _ = timing_record(_Result(node_records=[record]))
        self.assertAlmostEqual(node_metrics["timing.node.training.seconds"], 1.25)

    def test_it_also_reports_the_total(self):
        record = self._record("evaluation", 0.5)
        state = AgentState(execution_time=4.5)
        _, summary = timing_record(_Result(node_records=[record], agent_state=state))
        self.assertAlmostEqual(summary["timing.workflow.total_seconds"], 4.5)

    def test_the_node_sum_is_not_mislabelled_as_the_total(self):
        """Retries sit between the nodes, so the sum is its own number."""
        records = [
            self._record("training", 1.0),
            self._record("evaluation", 2.0),
        ]
        _, summary = timing_record(_Result(node_records=records))
        self.assertAlmostEqual(summary["timing.node_sum_seconds"], 3.0)
        self.assertNotIn("timing.workflow.total_seconds", summary)

    def test_it_ignores_records_without_a_duration(self):
        record = self._record("training", None, attempts=0)
        node_metrics, _ = timing_record(_Result(node_records=[record]))
        self.assertEqual(node_metrics, {})

    def test_an_empty_result_is_empty(self):
        self.assertEqual(timing_record(None), ({}, {}))
        self.assertEqual(timing_record(_Result(node_records=[])), ({}, {}))

    def test_it_starts_no_clock_of_its_own(self):
        """Step 15: the timings are read, never re-measured.

        The agents do own timers - Phase 4 has always recorded per-experiment
        training time - but the tracking layer must not add a second one. If it
        did, two slightly different durations for the same node would both be
        true and the discrepancy would be unexplainable.

        Checked against the parsed code, not the text: ``timing_record``'s own
        docstring has to name ``time.perf_counter`` to explain where the number
        came from.
        """
        tree = ast.parse(
            (PROJECT_ROOT / "tools" / "mlflow_tools.py").read_text(encoding="utf-8-sig")
        )
        clocks = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"perf_counter", "monotonic", "time", "now", "utcnow"}
        }
        self.assertEqual(
            clocks,
            set(),
            f"mlflow_tools must not measure time itself; it calls {sorted(clocks)}. "
            "NodeRecord.duration is the single source of truth.",
        )


class _Result:
    """A minimal stand-in for ``WorkflowResult`` in the timing tests."""

    def __init__(self, node_records: List[Any], agent_state: Any = None) -> None:
        self.node_records = node_records
        self._agent_state = agent_state

    @property
    def agent_state(self) -> Any:
        return self._agent_state


class TestReproducibilityRecord(unittest.TestCase):
    """Step 16: a controlled whitelist, never a dump of the environment."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET)

    def test_it_records_the_seeds(self):
        record = reproducibility_record(self.state, BINARY_DATASET)
        self.assertEqual(record["seeds"]["split.random_state"], 42)
        self.assertEqual(record["seeds"]["cv.random_state"], 42)

    def test_it_records_the_platform(self):
        record = reproducibility_record(self.state, BINARY_DATASET)
        for key in ("system", "machine", "python_implementation"):
            self.assertIn(key, record["platform"])

    def test_it_records_the_versions_from_the_whitelist(self):
        record = reproducibility_record(self.state, BINARY_DATASET)
        versions = record["versions"]
        for name in config.MLFLOW_VERSION_WHITELIST:
            self.assertIn(name, versions)
        self.assertNotEqual(versions["scikit-learn"], "not installed")

    def test_it_records_the_dataset_hash(self):
        record = reproducibility_record(self.state, BINARY_DATASET)
        self.assertEqual(record["dataset_sha256"], EXPECTED_BINARY_HASH)

    def test_it_does_not_dump_the_environment(self):
        """A full environment listing would record whatever else is installed."""
        record = reproducibility_record(self.state, BINARY_DATASET)
        versions = record["versions"]
        self.assertLessEqual(
            len(versions), len(config.MLFLOW_VERSION_WHITELIST) + 1,
            "Only the whitelist may be recorded.",
        )

    def test_it_records_no_environment_variable(self):
        import os

        record = reproducibility_record(self.state, BINARY_DATASET)
        text = json.dumps(record, default=str)
        for name in list(os.environ)[:20]:
            value = os.environ[name]
            if len(value) > 8:
                self.assertNotIn(value, text, f"leaked environment variable {name}")

    def test_it_works_without_a_state(self):
        record = reproducibility_record(None, BINARY_DATASET)
        self.assertIn("versions", record)


# --------------------------------------------------------------------------- #
# 6. Tags
# --------------------------------------------------------------------------- #
class TestRunTags(unittest.TestCase):
    """Step 6: a run is identifiable at a glance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET)

    def test_it_carries_every_suggested_tag(self):
        tags = run_tags(
            self.state, workflow_status="completed", enable_optimization=True,
            enable_llm=False, enable_persistence=True,
        )
        for key in (
            "project", "workflow", "task_type", "dataset", "phase",
            "execution_mode", "optimization_enabled", "llm_enabled",
            "persistence_enabled", "workflow_status",
        ):
            self.assertIn(key, tags)

    def test_the_flags_are_lowercase_booleans(self):
        tags = run_tags(self.state, enable_optimization=True, enable_persistence=True)
        self.assertEqual(tags["optimization_enabled"], "true")
        self.assertEqual(tags["persistence_enabled"], "true")
        self.assertEqual(tags["llm_enabled"], "false")

    def test_every_value_is_a_string(self):
        for value in run_tags(self.state).values():
            self.assertIsInstance(value, str)

    def test_it_records_no_credential(self):
        tags = run_tags(self.state)
        text = json.dumps(tags)
        for secret in ("api_key", "token", "password", "secret"):
            self.assertNotIn(secret, text.lower())

    def test_it_records_no_absolute_path(self):
        self.assertNotIn("C:\\", json.dumps(run_tags(self.state)))

    def test_an_unknown_workflow_still_produces_usable_tags(self):
        tags = run_tags(None)
        self.assertEqual(tags["task_type"], "unknown")
        self.assertEqual(tags["dataset"], "unknown")


# --------------------------------------------------------------------------- #
# 7. Run lifecycle against a real local store
# --------------------------------------------------------------------------- #
class TestLifecycle(TrackingStoreTestCase):
    """Steps 4, 5, 6 and 17: start, log, end, with the right status."""

    def test_a_recorded_run_reports_itself_as_recorded(self):
        summary = self.track()
        self.assertIs(summary.status, TrackingStatus.RECORDED)
        self.assertTrue(summary.recorded)
        self.assertTrue(summary.usable)
        self.assertEqual(summary.failures, [])

    def test_the_experiment_is_created_once_and_reused(self):
        self.track()
        self.track()
        names = [experiment.name for experiment in self.client().search_experiments()]
        self.assertEqual(
            names.count(self.experiment), 1, "Repeated runs must reuse the experiment."
        )

    def test_each_run_creates_a_distinct_run(self):
        first = self.track()
        second = self.track()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(len(self.runs()), 2)

    def test_parameters_are_recorded(self):
        self.track()
        params = dict(self.runs()[0].data.params)
        self.assertIn("dataset.name", params)
        self.assertIn("final_model.name", params)
        self.assertGreater(len(params), 40)

    def test_metrics_are_recorded(self):
        self.track()
        metrics = dict(self.runs()[0].data.metrics)
        self.assertIn("evaluation.holdout_f1_score", metrics)
        self.assertGreater(len(metrics), 10)

    def test_tags_are_recorded(self):
        self.track()
        tags = dict(self.runs()[0].data.tags)
        self.assertEqual(tags["project"], "autonomous-multi-agent-ml")
        self.assertEqual(tags["workflow_status"], "completed")

    def test_artifacts_are_recorded(self):
        self.track()
        paths = {item.path for item in self.client().list_artifacts(self.runs()[0].info.run_id)}
        for expected in (
            config.MLFLOW_ARTIFACT_DATASET,
            config.MLFLOW_ARTIFACT_PREPROCESSING,
            config.MLFLOW_ARTIFACT_MODEL_COMPARISON,
            config.MLFLOW_ARTIFACT_CONFUSION_MATRIX,
            config.MLFLOW_ARTIFACT_REPRODUCIBILITY,
        ):
            self.assertIn(expected, paths)

    def test_the_comparison_artifact_lists_every_model(self):
        self.track()
        body = self._artifact(config.MLFLOW_ARTIFACT_MODEL_COMPARISON)
        self.assertEqual(len(body["models"]), 5)
        for model in body["models"]:
            self.assertIn("holdout_f1_score", model)

    def test_the_confusion_matrix_artifact_is_machine_readable(self):
        self.track()
        body = self._artifact(config.MLFLOW_ARTIFACT_CONFUSION_MATRIX)
        self.assertEqual(body["labels"], ["0", "1"])
        self.assertEqual(body["correct_predictions"], 21)

    def test_the_reproducibility_artifact_is_json_primitive(self):
        self.track()
        body = self._artifact(config.MLFLOW_ARTIFACT_REPRODUCIBILITY)
        json.dumps(body)  # must not raise
        self.assertIn("versions", body)
        self.assertIn("seeds", body)

    def test_no_raw_dataset_is_uploaded(self):
        """The purpose is reproducibility, not dataset duplication."""
        self.track()
        run_id = self.runs()[0].info.run_id
        paths = [item.path for item in self.client().list_artifacts(run_id)]
        self.assertEqual(
            [path for path in paths if path.endswith(".csv")], [],
            "No CSV may be uploaded; the input already lives in datasets/.",
        )

    def test_no_model_artifact_is_logged_by_default(self):
        """Phase 9's joblib file stays the single deployment artifact."""
        self.track()
        run_id = self.runs()[0].info.run_id
        paths = [item.path for item in self.client().list_artifacts(run_id)]
        self.assertEqual(
            [path for path in paths if path.endswith(".joblib")], [],
            "MLflow must not replace or duplicate Phase 9 persistence.",
        )

    def _artifact(self, name: str) -> Dict[str, Any]:
        """Download and parse one artifact of the newest run.

        Args:
            name: The artifact file name.

        Returns:
            The parsed document.
        """
        import mlflow

        run_id = self.runs()[0].info.run_id
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=name, tracking_uri=self.uri
        )
        return json.loads(Path(local).read_text(encoding="utf-8-sig"))


class TestRunStatus(TrackingStoreTestCase):
    """Step 17: the MLflow status mirrors the deterministic workflow status."""

    def test_a_completed_workflow_is_finished(self):
        self.track(workflow_status="completed")
        self.assertEqual(self.runs()[0].info.status, "FINISHED")

    def test_a_partial_workflow_is_finished(self):
        """MLflow has no PARTIAL, so the exact value travels as a tag."""
        self.track(workflow_status="partial")
        run = self.runs()[0]
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(
            dict(run.data.tags)["workflow_status"],
            "partial",
            "A partial run must not be indistinguishable from a clean one.",
        )

    def test_a_failed_workflow_is_failed(self):
        self.track(workflow_status="failed")
        self.assertEqual(self.runs()[0].info.status, "FAILED")

    def test_the_tag_always_carries_the_exact_status(self):
        for status in ("completed", "partial", "failed"):
            summary = self.track(workflow_status=status)
            run = next(
                item
                for item in self.runs()
                if item.info.run_id == summary.run_id
            )
            self.assertEqual(dict(run.data.tags)["workflow_status"], status)


class TestDisabledTracking(unittest.TestCase):
    """Step 19: tracking can be switched off cleanly."""

    def test_disabled_tracking_records_nothing(self):
        summary = track_workflow_run(None, None, enabled=False)
        self.assertIs(summary.status, TrackingStatus.DISABLED)
        self.assertFalse(summary.usable)
        self.assertFalse(summary.recorded)

    def test_disabled_tracking_explains_itself(self):
        summary = track_workflow_run(None, None, enabled=False)
        self.assertIn("disabled", summary.reason.lower())

    def test_disabled_tracking_never_reaches_mlflow(self):
        with mock.patch.object(mlflow_tools, "import_mlflow") as importer:
            track_workflow_run(None, None, enabled=False)
        importer.assert_not_called()


# --------------------------------------------------------------------------- #
# 8. Failure isolation
# --------------------------------------------------------------------------- #
class TestFailureIsolation(TrackingStoreTestCase):
    """Step 18, and the single most important property of Phase 10."""

    def test_an_initialisation_failure_is_recorded_not_raised(self):
        with mock.patch.object(
            mlflow_tools, "import_mlflow", side_effect=mlflow_tools.TrackingError("no")
        ):
            summary = self.track()
        self.assertIs(summary.status, TrackingStatus.UNAVAILABLE)
        self.assertFalse(summary.usable)

    def test_a_broken_store_is_recorded_not_raised(self):
        summary = self.track(tracking_uri="sqlite:///Z:/no/such/place/db.db")
        self.assertIs(summary.status, TrackingStatus.FAILED)
        self.assertTrue(
            summary.failures, "A broken store must be reported, not hidden."
        )

    def test_a_logging_failure_does_not_stop_the_other_logging(self):
        """One bad parameter must not cost the metrics.

        The MLflow module object is shared process-wide, so the patch is applied
        as a context manager: leaving ``log_param`` permanently replaced would
        silently break every later test in the session.
        """
        run = self.open_run()
        with mock.patch.object(
            run._mlflow, "log_param", side_effect=RuntimeError("boom")  # noqa: SLF001
        ):
            logged_params = run.log_params({"a": 1})
            logged_metrics = run.log_metrics({"b": 1.0, "c": 2.0})
        self.assertEqual(logged_params, 0)
        self.assertEqual(
            logged_metrics, 2, "A param failure must not affect the metrics."
        )
        self.assertTrue(run._failures)  # noqa: SLF001

    def test_the_failures_are_listed_individually(self):
        run = self.open_run()
        with mock.patch.object(
            run._mlflow, "log_metric", side_effect=RuntimeError("boom")  # noqa: SLF001
        ):
            run.log_metrics({"a": 1.0, "b": 2.0})
        self.assertEqual(
            len(run._failures), 2, "each failed value must be reported on its own"  # noqa: SLF001
        )

    def test_an_artifact_failure_is_reported(self):
        run = self.open_run()
        with mock.patch.object(
            run._mlflow, "log_dict", side_effect=RuntimeError("boom")  # noqa: SLF001
        ):
            self.assertFalse(run.log_json("x.json", {"a": 1}))
        self.assertTrue(
            any("artifact" in item for item in run._failures)  # noqa: SLF001
        )

    def test_the_run_is_still_closed_after_a_logging_failure(self):
        """A run left open would stay RUNNING forever in the history."""
        run = self.open_run()
        with mock.patch.object(
            run._mlflow, "log_param", side_effect=RuntimeError("boom")  # noqa: SLF001
        ):
            run.log_params({"a": 1})
        self.assertTrue(run.end("completed"))
        stored = next(
            item for item in self.runs() if item.info.run_id == run.run_id
        )
        self.assertEqual(stored.info.status, "FINISHED")

    def test_a_run_that_logged_nothing_is_still_reported_as_recorded(self):
        """Tracking that recorded nothing is a partial record, not a failure."""
        with mock.patch.object(
            TrackedRun, "log_metrics", return_value=0
        ):
            summary = track_workflow_run(
                self.state, None, enabled=True, tracking_uri=self.uri,
                experiment_name=self.experiment, workflow_status="completed",
            )
        self.assertIs(summary.status, TrackingStatus.RECORDED)
        self.assertTrue(summary.recorded)

    def test_the_failure_reason_names_the_operation(self):
        summary = self.track(tracking_uri="sqlite:///Z:/no/such/place/db.db")
        self.assertIn("start:", " ".join(summary.failures))


class TestWorkflowStillSucceeds(unittest.TestCase):
    """Step 18, end to end: a broken tracker must not change the ML outcome."""

    def test_a_tracker_that_raises_leaves_the_workflow_completed(self):
        import graph.workflow as workflow

        with mock.patch.object(
            workflow,
            "_resolve_tracker",
            return_value=mock.Mock(side_effect=RuntimeError("tracker exploded")),
        ):
            directory = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            result = workflow.run_workflow(
                dataset_path=str(BINARY_DATASET),
                enable_optimization=False,
                enable_tracking=True,
                persistence_output_dir=str(directory),
            )

        self.assertIs(result.workflow_status.value, "completed")
        report = workflow.format_workflow_report(result)
        self.assertIn("Logistic Regression", report)
        self.assertIn("0.8761", report)

    def test_no_tracker_at_all_leaves_the_workflow_completed(self):
        import graph.workflow as workflow

        with mock.patch.object(workflow, "_resolve_tracker", return_value=None):
            directory = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
            result = workflow.run_workflow(
                dataset_path=str(BINARY_DATASET),
                enable_optimization=False,
                enable_tracking=True,
                persistence_output_dir=str(directory),
            )
        self.assertIs(result.workflow_status.value, "completed")

    def test_a_failing_payload_extractor_does_not_escape(self):
        """A bug in an extractor is reported as a tracking failure."""
        with mock.patch.object(
            mlflow_tools,
            "dataset_record",
            side_effect=RuntimeError("extractor bug"),
        ):
            summary = track_workflow_run(
                None, None, enabled=True,
                tracking_uri="sqlite:///" + str(self.store_path()),
                experiment_name="isolation",
                workflow_status="completed",
            )
        self.assertIs(summary.status, TrackingStatus.FAILED)
        self.assertTrue(any("payload" in item for item in summary.failures))

    def store_path(self) -> Path:
        """Return a temporary database path."""
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return directory / "mlflow.db"


# --------------------------------------------------------------------------- #
# 9. Duplicate handling
# --------------------------------------------------------------------------- #
class TestDuplicateParameters(TrackingStoreTestCase):
    """MLflow parameters are immutable, so a repeat would raise."""

    def test_a_repeated_parameter_is_reported_not_raised(self):
        """A second write of the same key is what MLflow actually rejects.

        A Python dict literal cannot express a duplicate key, so the second
        attempt is what has to be guarded.
        """
        run = self.open_run()
        self.assertEqual(run.log_params({"a": 1}), 1)
        self.assertEqual(run.log_params({"a": 2}), 0)
        self.assertTrue(
            run._failures, "the rejected write must be reported, not hidden"  # noqa: SLF001
        )
        self.assertTrue(run.end("completed"))
        params = dict(self.runs()[0].data.params)
        self.assertEqual(params["a"], "1", "The first value is the decided one.")

    def test_the_first_of_two_identical_keys_in_one_call_wins(self):
        """A mapping that arrives already collapsed keeps a single write."""
        run = self.open_run()
        self.assertEqual(run.log_params({"a": 1}), 1)
        self.assertEqual(run.log_params({"b": 2, "c": 3}), 2)
        self.assertTrue(run.end("completed"))
        params = dict(self.runs()[0].data.params)
        self.assertEqual(sorted(params), ["a", "b", "c"])

    def test_an_unrepresentable_value_is_skipped(self):
        run = self.open_run()
        logged = run.log_params({"a": None, "b": float("nan"), "c": 3})
        self.assertEqual(
            logged, 1, "an absent value is omitted, never written as empty text"
        )
        self.assertTrue(run.end("completed"))
        params = dict(self.runs()[0].data.params)
        self.assertNotIn("a", params)
        self.assertNotIn("b", params)
        self.assertIn("c", params)


# --------------------------------------------------------------------------- #
# 10. Read side, for the dashboard
# --------------------------------------------------------------------------- #
class TestRecentRuns(TrackingStoreTestCase):
    """Step 22: a bounded history, and never a query on every render."""

    def test_it_returns_nothing_when_nothing_was_recorded(self):
        self.assertEqual(
            recent_runs(tracking_uri=self.uri, experiment_name=self.experiment), []
        )

    def test_it_returns_the_recorded_runs(self):
        self.track()
        rows = recent_runs(tracking_uri=self.uri, experiment_name=self.experiment)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "FINISHED")

    def test_the_rows_carry_the_dashboard_fields(self):
        self.track()
        row = recent_runs(tracking_uri=self.uri, experiment_name=self.experiment)[0]
        for key in (
            "run_id", "start_time", "task_type", "model", "f1_score",
            "accuracy", "optimization_status", "status", "workflow_status",
        ):
            self.assertIn(key, row)

    def test_it_respects_the_limit(self):
        for _ in range(3):
            self.track()
        self.assertEqual(
            len(recent_runs(2, tracking_uri=self.uri, experiment_name=self.experiment)), 2
        )

    def test_a_limit_of_zero_returns_nothing(self):
        self.track()
        self.assertEqual(
            recent_runs(0, tracking_uri=self.uri, experiment_name=self.experiment), []
        )

    def test_it_returns_nothing_when_tracking_is_disabled(self):
        self.track()
        with mock.patch.object(config, "MLFLOW_ENABLED", False):
            self.assertEqual(
                recent_runs(tracking_uri=self.uri, experiment_name=self.experiment), []
            )

    def test_it_returns_nothing_when_mlflow_is_absent(self):
        with mock.patch.object(
            mlflow_tools, "import_mlflow", side_effect=mlflow_tools.TrackingError("no")
        ):
            self.assertEqual(
                recent_runs(tracking_uri=self.uri, experiment_name=self.experiment), []
            )

    def test_an_unreadable_store_returns_nothing_rather_than_raising(self):
        self.assertEqual(
            recent_runs(tracking_uri="sqlite:///Z:/no/such/place/db.db", experiment_name="x"),
            [],
        )

    def test_the_rows_report_the_real_metrics(self):
        self.track()
        row = recent_runs(tracking_uri=self.uri, experiment_name=self.experiment)[0]
        self.assertAlmostEqual(row["f1_score"], 0.8761343012704175, places=6)
        self.assertEqual(row["model"], "Logistic Regression")

    def test_the_timestamp_is_iso_8601_utc(self):
        self.track()
        row = recent_runs(tracking_uri=self.uri, experiment_name=self.experiment)[0]
        self.assertTrue(row["start_time"].endswith("Z"), row["start_time"])
        self.assertIn("T", row["start_time"])


class TestExperimentSummary(TrackingStoreTestCase):
    """Step 20: the dashboard's read-side contract."""

    def test_it_always_returns_the_same_keys(self):
        for summary in (
            experiment_summary(tracking_uri=self.uri, experiment_name=self.experiment),
            experiment_summary(tracking_uri="sqlite:///Z:/nope/db.db", experiment_name="x"),
        ):
            for key in (
                "available", "enabled", "installed", "tracking_uri",
                "experiment_name", "latest", "runs", "note",
            ):
                self.assertIn(key, summary)

    def test_it_reports_unavailable_when_mlflow_is_missing(self):
        with mock.patch.object(
            mlflow_tools, "import_mlflow", side_effect=mlflow_tools.TrackingError("no")
        ):
            summary = experiment_summary(
                tracking_uri=self.uri, experiment_name=self.experiment
            )
        self.assertFalse(summary["available"])
        self.assertFalse(summary["installed"])
        self.assertIn("unavailable", summary["note"].lower())

    def test_it_reports_disabled_when_tracking_is_off(self):
        with mock.patch.object(config, "MLFLOW_ENABLED", False):
            summary = experiment_summary(
                tracking_uri=self.uri, experiment_name=self.experiment
            )
        self.assertFalse(summary["available"])
        self.assertFalse(summary["enabled"])

    def test_it_reports_an_empty_store_distinctly_from_no_tracking(self):
        summary = experiment_summary(
            tracking_uri=self.uri, experiment_name=self.experiment
        )
        self.assertFalse(summary["available"])
        self.assertIn("no run has been recorded", summary["note"].lower())

    def test_it_reports_the_latest_run_when_there_is_one(self):
        self.track()
        summary = experiment_summary(
            tracking_uri=self.uri, experiment_name=self.experiment
        )
        self.assertTrue(summary["available"])
        self.assertEqual(summary["latest"]["status"], "FINISHED")
        self.assertEqual(summary["latest"]["model"], "Logistic Regression")


# --------------------------------------------------------------------------- #
# 11. Dashboard integration
# --------------------------------------------------------------------------- #
class TestDashboardIntegration(unittest.TestCase):
    """Steps 20 to 23: the section renders, and the dashboard stays pure."""

    def setUp(self) -> None:
        self.source = (PROJECT_ROOT / "tools" / "dashboard_tools.py").read_text(
            encoding="utf-8-sig"
        )
        self.view = (PROJECT_ROOT / "dashboard.py").read_text(encoding="utf-8-sig")

    def test_the_logic_module_imports_no_ml_library(self):
        """Otherwise the dashboard could not start without mlflow.

        Checked against the parsed import statements rather than the raw text,
        because the module's docstrings legitimately *mention* ``mlflow_tools``
        to explain where the injected data comes from.
        """
        for node in ast.walk(ast.parse(self.source)):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                self.assertNotIn(
                    "mlflow",
                    name,
                    "tools/dashboard_tools.py must import no ML library: the "
                    "tracking data is injected so this module stays "
                    "dependency-free and the dashboard can start without mlflow.",
                )

    def test_the_logic_module_imports_no_agent_graph_or_llm(self):
        for node in ast.walk(ast.parse(self.source)):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                for forbidden in ("agents", "graph", "llm"):
                    self.assertFalse(
                        name.startswith(forbidden),
                        f"tools/dashboard_tools.py must not import {name!r}.",
                    )

    def test_the_section_builder_exists(self):
        from tools.dashboard_tools import build_experiment_section

        self.assertTrue(callable(build_experiment_section))

    def test_the_view_defers_the_import_into_a_guard(self):
        self.assertIn("from tools import mlflow_tools", self.view)
        self.assertIn("except Exception", self.view)

    def test_the_summary_is_cached(self):
        """A page rerun must not query the tracking store every time."""
        self.assertIn("@st.cache_data", self.view)
        self.assertIn("cached_tracking_summary", self.view)

    def test_the_section_numbers_are_eleven_and_twelve(self):
        self.assertIn('"11. Experiment tracking"', self.view)
        self.assertIn('"12. Conventional vs Autonomous"', self.view)

    def test_every_section_is_wired_into_main(self):
        """Sections 2 through 12 must all be rendered."""
        body = self.view[self.view.index("def main(") :]
        for renderer in (
            "render_system_status",
            "render_dataset",
            "render_comparison",
            "render_evaluation",
            "render_confusion_matrix",
            "render_optimization",
            "render_llm",
            "render_prediction",
            "render_model_information",
            "render_experiment_tracking",
            "render_benchmark",
        ):
            self.assertIn(f"{renderer}(", body, f"{renderer} must be called from main()")

    def test_the_experiment_section_is_not_the_thing_that_fails_first(self):
        """It comes last, so a tracking outage cannot hide the other sections."""
        body = self.view[self.view.index("def main(") :]
        self.assertLess(
            body.index("render_model_information("),
            body.index("render_experiment_tracking("),
            "The optional section is rendered after the artifact sections.",
        )

    def test_the_view_starts_no_run(self):
        """Rendering the page must not create tracking state."""
        for forbidden in ("start_run", "end_run", "log_param", "log_metric", "log_artifact"):
            self.assertNotIn(forbidden, self.view, f"dashboard.py must never call {forbidden}()")

    def test_the_view_does_not_train(self):
        """Checked against parsed code, not prose: the docstrings discuss this."""
        tree = ast.parse(self.view)
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for forbidden in ("run_workflow", "create_study", "fit", "fit_predict"):
            self.assertNotIn(forbidden, calls | names)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        for module in ("agents", "graph", "optuna", "sklearn"):
            self.assertNotIn(module, imported)

    def test_the_tracking_section_takes_no_input_widget(self):
        """Step 23: no arbitrary URI or path may come from a reader."""
        body = self.view[
            self.view.index("def render_experiment_tracking(") : self.view.index(
                "def render_security_note("
            )
        ]
        for widget in ("text_input", "number_input", "selectbox", "file_uploader", "button"):
            self.assertNotIn(
                widget,
                body,
                f"the tracking section must be read-only; it exposes no {widget}.",
            )

    def test_the_view_uploads_nothing(self):
        """Phase 9's security property must survive Phase 10."""
        self.assertNotIn("file_uploader", self.view)

    def test_the_tracking_store_comes_from_configuration_only(self):
        """Step 23: a reader cannot redirect tracking to their own server."""
        self.assertNotIn("text_input", self.view[self.view.index("def render_experiment_tracking(") :])
        self.assertNotIn("set_tracking_uri", self.view)

    def test_an_unavailable_summary_renders_the_documented_message(self):
        from tools.dashboard_tools import build_experiment_section

        section = build_experiment_section(None)
        self.assertFalse(section["available"])
        self.assertEqual(section["note"], config.DASHBOARD_NO_EXPERIMENT_TRACKING)
        self.assertIn("unavailable or disabled", section["note"])

    def test_a_summary_with_runs_populates_the_history(self):
        from tools.dashboard_tools import build_experiment_section

        section = build_experiment_section(
            {
                "available": True,
                "enabled": True,
                "installed": True,
                "tracking_uri": "sqlite:///tmp/x.db",
                "experiment_name": "exp",
                "note": "",
                "latest": {
                    "run_id": "abcdef0123456789",
                    "status": "FINISHED",
                    "workflow_status": "completed",
                    "start_time": "2026-10-05T00:00:00Z",
                    "model": "Logistic Regression",
                    "f1_score": 0.8761,
                    "accuracy": 0.875,
                    "optimization_status": "completed",
                },
                "runs": [
                    {
                        "run_id": "abcdef0123456789",
                        "start_time": "2026-10-05T00:00:00Z",
                        "task_type": "binary_classification",
                        "dataset": "sample_classification.csv",
                        "model": "Logistic Regression",
                        "f1_score": 0.8761,
                        "accuracy": 0.875,
                        "optimization_status": "completed",
                        "status": "FINISHED",
                        "workflow_status": "completed",
                        "run_name": "run",
                    }
                ],
            }
        )
        self.assertTrue(section["available"])
        self.assertEqual(len(section["runs"]), 1)
        self.assertEqual(section["latest"]["short_run_id"], "abcdef01")
        self.assertEqual(section["latest"]["f1_score"], "0.8761")

    def test_a_missing_metric_renders_as_not_recorded_not_zero(self):
        """0.0000 would read as a real measurement."""
        from tools.dashboard_tools import build_experiment_section

        section = build_experiment_section(
            {
                "available": True,
                "runs": [{"run_id": "a", "f1_score": None, "accuracy": None}],
                "latest": {"run_id": "a", "f1_score": None},
            }
        )
        self.assertEqual(section["latest"]["f1_score"], "-")
        self.assertEqual(section["runs"][0]["accuracy"], "-")


# --------------------------------------------------------------------------- #
# 12. Data integrity
# --------------------------------------------------------------------------- #
class TestDataIntegrity(unittest.TestCase):
    """Step 7 and Step 24: nothing under test may move."""

    def test_the_binary_dataset_is_unchanged(self):
        self.assertEqual(file_hash(BINARY_DATASET), EXPECTED_BINARY_HASH)

    def test_the_multiclass_dataset_is_unchanged(self):
        self.assertEqual(file_hash(MULTICLASS_DATASET), EXPECTED_MULTICLASS_HASH)

    def test_tracking_writes_nothing_into_the_dataset_directory(self):
        before = sorted(item.name for item in config.DATASETS_DIR.iterdir())
        track_workflow_run(None, None, enabled=False)
        self.assertEqual(sorted(item.name for item in config.DATASETS_DIR.iterdir()), before)

    def test_tracking_writes_nothing_into_the_model_directory(self):
        before = sorted(
            (item.name, item.stat().st_mtime_ns)
            for item in config.MODELS_DIR.iterdir()
            if item.is_file()
        )
        track_workflow_run(None, None, enabled=False)
        self.assertEqual(
            sorted(
                (item.name, item.stat().st_mtime_ns)
                for item in config.MODELS_DIR.iterdir()
                if item.is_file()
            ),
            before,
            "Tracking must not touch models/; only the Phase 9 agent writes there.",
        )

    def test_the_tracking_store_is_not_inside_the_model_directory(self):
        self.assertNotIn(str(config.MODELS_DIR), str(config.MLFLOW_TRACKING_DIR))


class TestGeneratedDataIsIgnored(unittest.TestCase):
    """Step 27: generated tracking state must not be committed."""

    def test_the_gitignore_covers_the_tracking_store(self):
        text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8-sig")
        self.assertIn("experiments/", text)

    def test_the_gitignore_does_not_ignore_source_or_tests(self):
        lines = [
            line.strip()
            for line in (PROJECT_ROOT / ".gitignore")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        ]
        for pattern in ("tools/", "agents/", "graph/", "tests/", "dashboard.py"):
            self.assertNotIn(pattern, lines, f"{pattern} must not be ignored.")


if __name__ == "__main__":
    unittest.main()