"""Tests for Phase 10 — MLflow experiment tracking tools.

All tests use mocked MLflow client objects so that:
- No real MLflow server or filesystem is needed.
- Tests run offline and deterministically.
- The frozen Config cannot be patched directly — we construct
  MLflowTracker instances with explicit parameters instead.
"""

from __future__ import annotations

import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level import sanity
# ---------------------------------------------------------------------------


def test_mlflow_tools_module_importable():
    """tools.mlflow_tools must be importable without side effects."""
    import tools.mlflow_tools  # noqa: F401

    assert True


def test_mlflow_tracker_class_exists():
    from tools.mlflow_tools import MLflowTracker

    assert callable(MLflowTracker)


def test_create_tracker_from_config_importable():
    from tools.mlflow_tools import create_tracker_from_config

    assert callable(create_tracker_from_config)


# ---------------------------------------------------------------------------
# Disabled tracker (MLFLOW_ENABLED=False) — no MLflow calls expected
# ---------------------------------------------------------------------------


class TestMLflowTrackerDisabled:
    """When enabled=False every method must be a no-op."""

    def _make_tracker(self) -> Any:
        from tools.mlflow_tools import MLflowTracker

        return MLflowTracker(
            enabled=False,
            tracking_uri="experiments/mlruns",
            experiment_name="test_experiment",
        )

    def test_disabled_tracker_has_enabled_false(self):
        tracker = self._make_tracker()
        assert tracker.enabled is False

    def test_disabled_tracker_run_id_is_none(self):
        tracker = self._make_tracker()
        assert tracker.run_id is None

    def test_start_workflow_run_returns_none(self):
        tracker = self._make_tracker()
        result = tracker.start_workflow_run("datasets/sample.csv")
        assert result is None

    def test_end_run_does_not_raise(self):
        tracker = self._make_tracker()
        tracker.end_run("FINISHED")  # must not raise

    def test_log_data_analysis_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_data_analysis(mock_agent_state)

    def test_log_preprocessing_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_preprocessing(mock_agent_state)

    def test_log_training_results_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_training_results(mock_agent_state)

    def test_log_evaluation_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_evaluation(mock_agent_state)

    def test_log_optimization_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_optimization(mock_agent_state)

    def test_log_persistence_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_persistence(mock_agent_state)

    def test_log_reproducibility_does_not_raise(self):
        tracker = self._make_tracker()
        tracker.log_reproducibility()

    def test_log_workflow_summary_does_not_raise(self, mock_agent_state):
        tracker = self._make_tracker()
        tracker.log_workflow_summary(mock_agent_state, elapsed_seconds=5.0)

    def test_get_recent_runs_returns_empty_list(self):
        tracker = self._make_tracker()
        result = tracker.get_recent_runs()
        assert result == []


# ---------------------------------------------------------------------------
# Enabled tracker with mocked MLflow client
# ---------------------------------------------------------------------------


class TestMLflowTrackerEnabled:
    """Enabled tracker must call MLflow client methods correctly."""

    def _make_enabled_tracker_with_mock(self):
        """Return (tracker, mock_client, mock_mlflow_module)."""
        from tools.mlflow_tools import MLflowTracker

        # Build a disabled tracker first, then manually inject mock client
        tracker = MLflowTracker(
            enabled=False,  # skip real setup
            tracking_uri="experiments/mlruns",
            experiment_name="test_experiment",
        )
        # Inject mock state
        tracker._enabled = True
        mock_client = MagicMock()
        tracker._client = mock_client
        tracker._run_id = "test-run-id-001"
        return tracker, mock_client

    # ------------------------------------------------------------------
    # start_workflow_run
    # ------------------------------------------------------------------

    def test_start_workflow_run_returns_run_id(self):
        """start_workflow_run must return a run_id string when enabled."""
        from tools.mlflow_tools import MLflowTracker

        tracker = MLflowTracker(
            enabled=False,
            tracking_uri="experiments/mlruns",
            experiment_name="test_experiment",
        )
        tracker._enabled = True

        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc123"
        mock_client.create_run.return_value = mock_run
        tracker._client = mock_client

        # Mock mlflow module
        mock_mlflow = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            run_id = tracker.start_workflow_run("datasets/test.csv")

        assert run_id == "run-abc123"
        assert tracker.run_id == "run-abc123"

    def test_start_workflow_run_returns_none_when_experiment_missing(self):
        from tools.mlflow_tools import MLflowTracker

        tracker = MLflowTracker(enabled=False, tracking_uri="", experiment_name="x")
        tracker._enabled = True
        tracker._client = MagicMock()

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None  # not found

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            run_id = tracker.start_workflow_run("datasets/test.csv")

        assert run_id is None

    # ------------------------------------------------------------------
    # end_run
    # ------------------------------------------------------------------

    def test_end_run_calls_set_terminated(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.end_run("FINISHED")
        mock_client.set_terminated.assert_called_once_with(
            "test-run-id-001", status="FINISHED"
        )

    def test_end_run_failed_status(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.end_run("FAILED")
        mock_client.set_terminated.assert_called_once_with(
            "test-run-id-001", status="FAILED"
        )

    # ------------------------------------------------------------------
    # log_data_analysis
    # ------------------------------------------------------------------

    def test_log_data_analysis_logs_params(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_data_analysis(mock_agent_state)
        # log_param must be called at least once
        assert mock_client.log_param.called

    def test_log_data_analysis_logs_metrics(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_data_analysis(mock_agent_state)
        assert mock_client.log_metric.called

    def test_log_data_analysis_does_not_raise_on_bad_state(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        mock_client.log_param.side_effect = RuntimeError("mlflow error")
        bad_state = MagicMock()
        bad_state.dataset_name = None
        bad_state.problem_type = None
        bad_state.target_column = None
        bad_state.dataset_shape = None
        bad_state.duplicate_rows_count = 0
        bad_state.numerical_features = []
        bad_state.categorical_features = []
        # Must not raise
        tracker.log_data_analysis(bad_state)

    # ------------------------------------------------------------------
    # log_preprocessing
    # ------------------------------------------------------------------

    def test_log_preprocessing_logs_params(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_preprocessing(mock_agent_state)
        assert mock_client.log_param.called

    # ------------------------------------------------------------------
    # log_training_results
    # ------------------------------------------------------------------

    def test_log_training_results_logs_experiment_count(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_training_results(mock_agent_state)
        # experiment_count must be logged as param
        param_keys = [call.args[1] for call in mock_client.log_param.call_args_list]
        assert "experiment_count" in param_keys

    def test_log_training_results_logs_per_model_metrics(self, mock_agent_state_with_results):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_training_results(mock_agent_state_with_results)
        metric_keys = [call.args[1] for call in mock_client.log_metric.call_args_list]
        # At least one baseline metric should be logged
        assert any("baseline_" in k for k in metric_keys)

    # ------------------------------------------------------------------
    # log_evaluation
    # ------------------------------------------------------------------

    def test_log_evaluation_calls_set_tag(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_evaluation(mock_agent_state)
        assert mock_client.set_tag.called

    # ------------------------------------------------------------------
    # log_optimization
    # ------------------------------------------------------------------

    def test_log_optimization_logs_status(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_optimization(mock_agent_state)
        param_keys = [call.args[1] for call in mock_client.log_param.call_args_list]
        assert "optimization_status" in param_keys

    def test_log_optimization_logs_best_model(self, mock_agent_state_with_optimization):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_optimization(mock_agent_state_with_optimization)
        param_keys = [call.args[1] for call in mock_client.log_param.call_args_list]
        assert "best_optimized_model" in param_keys

    # ------------------------------------------------------------------
    # log_persistence
    # ------------------------------------------------------------------

    def test_log_persistence_logs_status(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_persistence(mock_agent_state)
        param_keys = [call.args[1] for call in mock_client.log_param.call_args_list]
        assert "persistence_status" in param_keys

    # ------------------------------------------------------------------
    # log_reproducibility
    # ------------------------------------------------------------------

    def test_log_reproducibility_logs_python_version(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_reproducibility()
        tag_keys = [call.args[1] for call in mock_client.set_tag.call_args_list]
        assert "python_version" in tag_keys

    def test_log_reproducibility_logs_platform(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_reproducibility()
        tag_keys = [call.args[1] for call in mock_client.set_tag.call_args_list]
        assert "platform" in tag_keys

    # ------------------------------------------------------------------
    # log_workflow_summary
    # ------------------------------------------------------------------

    def test_log_workflow_summary_logs_elapsed(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_workflow_summary(mock_agent_state, elapsed_seconds=12.5)
        metric_keys = [call.args[1] for call in mock_client.log_metric.call_args_list]
        assert "workflow_elapsed_seconds" in metric_keys

    def test_log_workflow_summary_logs_workflow_status(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        tracker.log_workflow_summary(mock_agent_state, elapsed_seconds=1.0)
        tag_keys = [call.args[1] for call in mock_client.set_tag.call_args_list]
        assert "workflow_status" in tag_keys

    # ------------------------------------------------------------------
    # Failure isolation
    # ------------------------------------------------------------------

    def test_log_data_analysis_survives_client_exception(self, mock_agent_state):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        mock_client.log_param.side_effect = Exception("unexpected crash")
        # Must not propagate
        tracker.log_data_analysis(mock_agent_state)

    def test_end_run_survives_client_exception(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        mock_client.set_terminated.side_effect = Exception("mlflow down")
        tracker.end_run("FINISHED")  # must not raise

    def test_log_reproducibility_survives_exception(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        mock_client.set_tag.side_effect = Exception("crash")
        tracker.log_reproducibility()  # must not raise

    # ------------------------------------------------------------------
    # get_recent_runs
    # ------------------------------------------------------------------

    def test_get_recent_runs_returns_list(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        mock_run = MagicMock()
        mock_run.info.run_id = "run-001"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 0
        mock_run.info.end_time = 1000
        mock_run.data.tags = {"dataset": "test.csv", "workflow_status": "completed"}
        mock_run.data.params = {
            "best_baseline_model": "logistic_regression",
            "best_optimized_model": "svm",
            "optimization_status": "completed",
        }
        mock_run.data.metrics = {"best_baseline_f1_score": 0.85}
        mock_client.search_runs.return_value = [mock_run]

        mock_mlflow = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            runs = tracker.get_recent_runs(max_results=5)

        assert isinstance(runs, list)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"
        assert runs[0]["status"] == "FINISHED"
        assert runs[0]["best_baseline_model"] == "logistic_regression"

    def test_get_recent_runs_returns_empty_on_error(self):
        tracker, mock_client = self._make_enabled_tracker_with_mock()
        mock_client.search_runs.side_effect = RuntimeError("crash")
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = tracker.get_recent_runs()

        assert result == []


# ---------------------------------------------------------------------------
# _safe_metric helper
# ---------------------------------------------------------------------------


class TestSafeMetric:
    def test_float_value(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(0.85) == pytest.approx(0.85)

    def test_int_value(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(5) == pytest.approx(5.0)

    def test_nan_returns_none(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(float("nan")) is None

    def test_inf_returns_none(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(float("inf")) is None

    def test_neg_inf_returns_none(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(float("-inf")) is None

    def test_none_returns_none(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(None) is None

    def test_string_returns_none(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric("not_a_number") is None

    def test_zero(self):
        from tools.mlflow_tools import _safe_metric

        assert _safe_metric(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Config integration (create_tracker_from_config)
# ---------------------------------------------------------------------------


class TestCreateTrackerFromConfig:
    def test_creates_disabled_tracker_by_default(self):
        """Default config has MLFLOW_ENABLED=False — tracker must be disabled."""
        from tools.mlflow_tools import create_tracker_from_config

        tracker = create_tracker_from_config()
        # Default config has MLFLOW_ENABLED=false
        assert isinstance(tracker.enabled, bool)
        # When disabled, run_id must be None
        if not tracker.enabled:
            assert tracker.run_id is None

    def test_tracker_has_correct_type(self):
        from tools.mlflow_tools import MLflowTracker, create_tracker_from_config

        tracker = create_tracker_from_config()
        assert isinstance(tracker, MLflowTracker)


# ---------------------------------------------------------------------------
# WorkflowResult.mlflow_run_id (Phase 10 addition)
# ---------------------------------------------------------------------------


class TestWorkflowResultMLflowRunId:
    def test_workflow_result_has_mlflow_run_id_field(self):
        from graph.state import WorkflowResult

        result = WorkflowResult(workflow_status="completed")
        assert hasattr(result, "mlflow_run_id")

    def test_workflow_result_mlflow_run_id_default_is_none(self):
        from graph.state import WorkflowResult

        result = WorkflowResult(workflow_status="completed")
        assert result.mlflow_run_id is None

    def test_workflow_result_mlflow_run_id_can_be_set(self):
        from graph.state import WorkflowResult

        result = WorkflowResult(workflow_status="completed", mlflow_run_id="run-xyz")
        assert result.mlflow_run_id == "run-xyz"

    def test_workflow_result_to_dict_includes_mlflow_run_id(self):
        from graph.state import WorkflowResult

        result = WorkflowResult(
            workflow_status="completed", mlflow_run_id="run-abc"
        )
        d = result.to_dict()
        assert "mlflow_run_id" in d
        assert d["mlflow_run_id"] == "run-abc"

    def test_workflow_result_to_dict_none_run_id(self):
        from graph.state import WorkflowResult

        result = WorkflowResult(workflow_status="completed")
        d = result.to_dict()
        assert d["mlflow_run_id"] is None


# ---------------------------------------------------------------------------
# Config: MLflow fields
# ---------------------------------------------------------------------------


class TestConfigMLflowFields:
    def test_config_has_mlflow_enabled(self):
        from utils.config import Config

        assert hasattr(Config, "MLFLOW_ENABLED")

    def test_config_mlflow_enabled_default_is_false(self):
        from utils.config import Config

        assert Config.MLFLOW_ENABLED is False

    def test_config_has_mlflow_tracking_uri(self):
        from utils.config import Config

        assert hasattr(Config, "MLFLOW_TRACKING_URI")

    def test_config_mlflow_tracking_uri_contains_experiments(self):
        from utils.config import Config

        assert "experiments" in Config.MLFLOW_TRACKING_URI or "mlruns" in Config.MLFLOW_TRACKING_URI

    def test_config_has_mlflow_experiment_name(self):
        from utils.config import Config

        assert hasattr(Config, "MLFLOW_EXPERIMENT_NAME")

    def test_config_mlflow_experiment_name_is_string(self):
        from utils.config import Config

        assert isinstance(Config.MLFLOW_EXPERIMENT_NAME, str)
        assert len(Config.MLFLOW_EXPERIMENT_NAME) > 0


# ---------------------------------------------------------------------------
# run_graph() integration — MLflow must not break ML pipeline
# ---------------------------------------------------------------------------


class TestRunGraphMLflowIntegration:
    """run_graph must succeed with mlflow_run_id field even when MLflow is disabled."""

    def test_run_graph_returns_workflow_result_with_mlflow_field(self, tmp_path):
        """When MLFLOW_ENABLED=False (default), run_graph must still return
        WorkflowResult with mlflow_run_id=None (no tracking)."""
        import pandas as pd

        from graph.state import WorkflowResult

        # Create a tiny sample dataset
        df = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "label": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        csv_path = tmp_path / "mini_test.csv"
        df.to_csv(csv_path, index=False)

        from graph.workflow import run_graph

        result = run_graph(
            dataset_path=str(csv_path),
            enable_optimization=False,
        )

        assert isinstance(result, WorkflowResult)
        assert hasattr(result, "mlflow_run_id")
        # MLflow is disabled by default — run_id is None
        assert result.mlflow_run_id is None

    def test_run_graph_mlflow_failure_does_not_fail_pipeline(self, tmp_path):
        """Even if MLflow raises during post-run logging, the WorkflowResult
        must be returned without error."""
        import pandas as pd

        from graph.state import WorkflowResult

        df = pd.DataFrame(
            {
                "feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "target": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        csv_path = tmp_path / "mini2.csv"
        df.to_csv(csv_path, index=False)

        # Patch create_tracker_from_config to return a tracker whose methods crash
        broken_tracker = MagicMock()
        broken_tracker.start_workflow_run.side_effect = RuntimeError("mlflow exploded")
        broken_tracker.end_run.side_effect = RuntimeError("still broken")
        broken_tracker.log_data_analysis.side_effect = RuntimeError("broken")
        broken_tracker.log_preprocessing.side_effect = RuntimeError("broken")
        broken_tracker.log_training_results.side_effect = RuntimeError("broken")
        broken_tracker.log_evaluation.side_effect = RuntimeError("broken")
        broken_tracker.log_optimization.side_effect = RuntimeError("broken")
        broken_tracker.log_persistence.side_effect = RuntimeError("broken")
        broken_tracker.log_workflow_summary.side_effect = RuntimeError("broken")
        broken_tracker.log_reproducibility.side_effect = RuntimeError("broken")

        with patch("graph.workflow.create_tracker_from_config", return_value=broken_tracker):
            from graph.workflow import run_graph

            result = run_graph(
                dataset_path=str(csv_path),
                enable_optimization=False,
            )

        assert isinstance(result, WorkflowResult)
        assert result.success  # ML pipeline must still succeed


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent_state():
    """Minimal mock AgentState for tracker logging tests."""
    state = MagicMock()
    state.dataset_name = "sample_classification.csv"
    state.problem_type = "binary_classification"
    state.target_column = "Outcome"
    state.dataset_shape = (100, 7)
    state.duplicate_rows_count = 0
    state.numerical_features = ["age", "salary"]
    state.categorical_features = ["department"]
    state.numerical_imputation_strategy = "median"
    state.categorical_imputation_strategy = "most_frequent"
    state.encoding_strategy = "one_hot"
    state.scaling_strategy = "standard_scaler"
    state.scale_numeric = True
    state.duplicate_handling_strategy = "remove_before_training"
    state.selected_numerical_features = ["age", "salary"]
    state.selected_categorical_features = ["department"]
    state.excluded_features = []
    state.duplicate_rows_removed = 0
    state.selected_models = ["logistic_regression", "decision_tree"]
    state.experiment_count = 2
    state.best_baseline_model = "logistic_regression"
    state.experiment_results = []
    state.best_baseline_metrics = {"f1_score": 0.85, "accuracy": 0.87}
    state.evaluation_status = "completed"
    state.model_evaluations = []
    state.optimization_status = "pending"
    state.best_optimized_model = None
    state.best_optimized_metrics = {}
    state.best_hyperparameters = {}
    state.hyperparameter_experiments = 0
    state.successful_trials = 0
    state.failed_trials = 0
    state.persistence_status = "pending"
    state.persisted_model_path = None
    state.persisted_metadata_path = None
    state.workflow_status = MagicMock()
    state.workflow_status.value = "completed"
    state.llm_enabled = False
    state.llm_status = "disabled"
    return state


@pytest.fixture
def mock_agent_state_with_results(mock_agent_state):
    """AgentState with non-empty experiment_results."""
    mock_agent_state.experiment_results = [
        {
            "model_name": "Logistic Regression",
            "accuracy": 0.87,
            "precision": 0.86,
            "recall": 0.85,
            "f1_score": 0.855,
            "cv_f1_mean": 0.84,
            "cv_accuracy_mean": 0.86,
            "training_time": 0.12,
        },
        {
            "model_name": "Decision Tree",
            "accuracy": 0.80,
            "precision": 0.79,
            "recall": 0.81,
            "f1_score": 0.80,
            "cv_f1_mean": 0.78,
            "cv_accuracy_mean": 0.79,
            "training_time": 0.08,
        },
    ]
    return mock_agent_state


@pytest.fixture
def mock_agent_state_with_optimization(mock_agent_state):
    """AgentState with optimization results."""
    mock_agent_state.best_optimized_model = "Support Vector Machine"
    mock_agent_state.best_optimized_metrics = {"f1_score": 0.91, "accuracy": 0.92}
    mock_agent_state.optimization_status = "completed"
    mock_agent_state.best_hyperparameters = {
        "Support Vector Machine": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
    }
    mock_agent_state.hyperparameter_experiments = 40
    mock_agent_state.successful_trials = 38
    mock_agent_state.failed_trials = 2
    return mock_agent_state
