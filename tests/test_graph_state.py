"""Tests for graph/state.py — GraphState creation, AgentState preservation, serialization."""

import json
import pytest

from graph.state import (
    NodeStatus,
    WorkflowResult,
    extract_agent_state,
    make_graph_state,
)
from utils.config import Config
from utils.state import AgentState, WorkflowStatus

BINARY_CSV = str(Config.DATASETS_DIR / "sample_classification.csv")


# ---------------------------------------------------------------------------
# 1. GraphState creation
# ---------------------------------------------------------------------------

class TestMakeGraphState:
    def test_returns_dict(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert isinstance(gs, dict)

    def test_contains_agent_state_dict(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert "agent_state_dict" in gs
        assert isinstance(gs["agent_state_dict"], dict)

    def test_default_enable_optimization_true(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert gs["enable_optimization"] is True

    def test_enable_optimization_false(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state, enable_optimization=False)
        assert gs["enable_optimization"] is False

    def test_default_max_retries(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state, max_retries=1)
        assert gs["max_retries"] == 1

    def test_optimization_done_false_initially(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert gs["optimization_done"] is False

    def test_completed_nodes_empty_initially(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert gs["completed_nodes"] == []

    def test_failed_node_none_initially(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert gs["failed_node"] is None

    def test_retry_counts_empty_initially(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert gs["retry_counts"] == {}

    def test_workflow_complete_false_initially(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        assert gs["workflow_complete"] is False


# ---------------------------------------------------------------------------
# 2. AgentState preservation through graph state
# ---------------------------------------------------------------------------

class TestAgentStatePreservation:
    def test_dataset_path_preserved(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        restored = extract_agent_state(gs)
        assert restored.dataset_path == BINARY_CSV

    def test_roundtrip_preserves_workflow_status(self):
        state = AgentState(dataset_path=BINARY_CSV)
        state.workflow_status = WorkflowStatus.TRAINING_COMPLETED
        gs = make_graph_state(state)
        restored = extract_agent_state(gs)
        assert restored.workflow_status == WorkflowStatus.TRAINING_COMPLETED

    def test_roundtrip_preserves_target_column(self):
        state = AgentState(dataset_path=BINARY_CSV)
        state.target_column = "Outcome"
        gs = make_graph_state(state)
        restored = extract_agent_state(gs)
        assert restored.target_column == "Outcome"


# ---------------------------------------------------------------------------
# 3. JSON-safe serialization
# ---------------------------------------------------------------------------

class TestGraphStateSerialisation:
    def test_graph_state_is_json_serializable(self):
        state = AgentState(dataset_path=BINARY_CSV)
        gs = make_graph_state(state)
        # Should not raise
        serialized = json.dumps(gs, default=str)
        assert isinstance(serialized, str)

    def test_agent_state_dict_is_json_serializable(self):
        state = AgentState(dataset_path=BINARY_CSV)
        state.target_column = "Outcome"
        gs = make_graph_state(state)
        serialized = json.dumps(gs["agent_state_dict"], default=str)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# 4. NodeStatus enum
# ---------------------------------------------------------------------------

class TestNodeStatus:
    def test_all_statuses_defined(self):
        statuses = [s.value for s in NodeStatus]
        assert "pending" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "skipped" in statuses
        assert "retrying" in statuses

    def test_is_string_enum(self):
        assert isinstance(NodeStatus.COMPLETED, str)


# ---------------------------------------------------------------------------
# 5. WorkflowResult
# ---------------------------------------------------------------------------

class TestWorkflowResult:
    def test_success_true_when_no_failure(self):
        result = WorkflowResult(workflow_status="optimization_completed")
        assert result.success is True

    def test_success_false_when_failed_node(self):
        result = WorkflowResult(workflow_status="failed", failed_node="training")
        assert result.success is False

    def test_to_dict_keys(self):
        result = WorkflowResult(
            workflow_status="optimization_completed",
            completed_nodes=["data_analysis", "preprocessing"],
        )
        d = result.to_dict()
        expected_keys = {
            "workflow_status", "completed_nodes", "failed_node",
            "error_message", "retry_counts", "optimization_executed",
            "optimization_enabled", "best_baseline_model", "best_optimized_model",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_is_json_serializable(self):
        result = WorkflowResult(
            workflow_status="optimization_completed",
            completed_nodes=["data_analysis"],
            retry_counts={"data_analysis": 0},
        )
        serialized = json.dumps(result.to_dict(), default=str)
        assert isinstance(serialized, str)

    def test_best_models_none_without_agent_state(self):
        result = WorkflowResult(workflow_status="completed")
        assert result.best_baseline_model is None
        assert result.best_optimized_model is None
