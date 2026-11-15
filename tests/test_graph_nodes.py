"""Tests for graph/nodes.py — individual node behaviour."""

import pytest

from graph.nodes import (
    data_analysis_node,
    evaluation_node,
    model_selection_node,
    optimization_node,
    preprocessing_node,
    training_node,
)
from graph.state import NodeStatus, extract_agent_state, make_graph_state
from utils.config import Config
from utils.state import AgentState, WorkflowStatus

BINARY_CSV = str(Config.DATASETS_DIR / "sample_classification.csv")
MULTICLASS_CSV = str(Config.DATASETS_DIR / "sample_multiclass.csv")


# ---------------------------------------------------------------------------
# Helper: run an agent chain through to a given point
# ---------------------------------------------------------------------------

def _gs(csv_path: str = BINARY_CSV, enable_opt: bool = True) -> dict:
    """Make a fresh graph state dict for a given CSV."""
    return make_graph_state(AgentState(dataset_path=csv_path), enable_optimization=enable_opt)


def _run_through_analysis(csv_path: str = BINARY_CSV) -> dict:
    gs = _gs(csv_path)
    return data_analysis_node(gs)


def _run_through_preprocessing(csv_path: str = BINARY_CSV) -> dict:
    gs = _run_through_analysis(csv_path)
    return preprocessing_node(gs)


def _run_through_model_selection(csv_path: str = BINARY_CSV) -> dict:
    gs = _run_through_preprocessing(csv_path)
    return model_selection_node(gs)


def _run_through_training(csv_path: str = BINARY_CSV) -> dict:
    gs = _run_through_model_selection(csv_path)
    return training_node(gs)


def _run_through_evaluation(csv_path: str = BINARY_CSV) -> dict:
    gs = _run_through_training(csv_path)
    return evaluation_node(gs)


# ---------------------------------------------------------------------------
# 6. data_analysis_node
# ---------------------------------------------------------------------------

class TestDataAnalysisNode:
    def test_returns_dict(self):
        gs = data_analysis_node(_gs())
        assert isinstance(gs, dict)

    def test_adds_to_completed_nodes(self):
        gs = data_analysis_node(_gs())
        assert "data_analysis" in gs["completed_nodes"]

    def test_node_status_completed(self):
        gs = data_analysis_node(_gs())
        assert gs["node_statuses"]["data_analysis"] == NodeStatus.COMPLETED.value

    def test_agent_state_has_target_column(self):
        gs = data_analysis_node(_gs())
        state = extract_agent_state(gs)
        assert state.target_column is not None

    def test_no_failed_node_on_success(self):
        gs = data_analysis_node(_gs())
        assert gs["failed_node"] is None

    def test_failed_node_set_on_bad_csv(self):
        initial = make_graph_state(AgentState(dataset_path="nonexistent.csv"), max_retries=0)
        gs = data_analysis_node(initial)
        assert gs["failed_node"] == "data_analysis"


# ---------------------------------------------------------------------------
# 7. preprocessing_node
# ---------------------------------------------------------------------------

class TestPreprocessingNode:
    def test_preprocessing_completes(self):
        gs = _run_through_preprocessing()
        assert "preprocessing" in gs["completed_nodes"]

    def test_preprocessing_status_completed(self):
        gs = _run_through_preprocessing()
        assert gs["node_statuses"]["preprocessing"] == NodeStatus.COMPLETED.value

    def test_pipeline_config_populated(self):
        gs = _run_through_preprocessing()
        state = extract_agent_state(gs)
        assert len(state.pipeline_config) > 0


# ---------------------------------------------------------------------------
# 8. model_selection_node
# ---------------------------------------------------------------------------

class TestModelSelectionNode:
    def test_model_selection_completes(self):
        gs = _run_through_model_selection()
        assert "model_selection" in gs["completed_nodes"]

    def test_selected_models_populated(self):
        gs = _run_through_model_selection()
        state = extract_agent_state(gs)
        assert len(state.selected_models) > 0


# ---------------------------------------------------------------------------
# 9. training_node
# ---------------------------------------------------------------------------

class TestTrainingNode:
    def test_training_completes(self):
        gs = _run_through_training()
        assert "training" in gs["completed_nodes"]

    def test_experiment_results_populated(self):
        gs = _run_through_training()
        state = extract_agent_state(gs)
        assert len(state.experiment_results) > 0

    def test_best_baseline_model_set(self):
        gs = _run_through_training()
        state = extract_agent_state(gs)
        assert state.best_baseline_model is not None


# ---------------------------------------------------------------------------
# 10. evaluation_node
# ---------------------------------------------------------------------------

class TestEvaluationNode:
    def test_evaluation_completes(self):
        gs = _run_through_evaluation()
        assert "evaluation" in gs["completed_nodes"]

    def test_model_evaluations_populated(self):
        gs = _run_through_evaluation()
        state = extract_agent_state(gs)
        assert len(state.model_evaluations) > 0

    def test_evaluation_status_completed(self):
        gs = _run_through_evaluation()
        assert gs["node_statuses"]["evaluation"] == NodeStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# 11. optimization_node
# ---------------------------------------------------------------------------

class TestOptimizationNode:
    def test_optimization_node_completes(self):
        gs = _run_through_evaluation()
        # Allow optimization to run
        gs_opt = optimization_node(gs)
        assert "optimization" in gs_opt["node_statuses"]

    def test_optimization_done_set_to_true(self):
        gs = _run_through_evaluation()
        gs_opt = optimization_node(gs)
        assert gs_opt["optimization_done"] is True

    def test_baseline_model_preserved_after_optimization(self):
        gs = _run_through_evaluation()
        state_before = extract_agent_state(gs)
        baseline_before = state_before.best_baseline_model

        gs_opt = optimization_node(gs)
        state_after = extract_agent_state(gs_opt)
        assert state_after.best_baseline_model == baseline_before


# ---------------------------------------------------------------------------
# 12. Node status updates
# ---------------------------------------------------------------------------

class TestNodeStatusUpdates:
    def test_all_required_nodes_have_status(self):
        gs = _run_through_evaluation()
        expected = {"data_analysis", "preprocessing", "model_selection", "training", "evaluation"}
        assert expected.issubset(gs["node_statuses"].keys())

    def test_all_required_nodes_completed(self):
        gs = _run_through_evaluation()
        for node in ["data_analysis", "preprocessing", "model_selection", "training", "evaluation"]:
            assert gs["node_statuses"][node] == NodeStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# 13. Node error propagation
# ---------------------------------------------------------------------------

class TestNodeErrorPropagation:
    def test_data_analysis_failure_sets_failed_node(self):
        gs = make_graph_state(AgentState(dataset_path="bad_path.csv"), max_retries=0)
        gs = data_analysis_node(gs)
        assert gs["failed_node"] == "data_analysis"
        assert gs["error_message"] is not None
