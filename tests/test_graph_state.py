"""Tests for the graph state layer (Phase 7, checks 1-5).

Covers graph state creation, AgentState preservation, JSON-safe serialisation,
the default workflow status and the retry counter.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import fields, is_dataclass

from graph.state import (
    GRAPH_NODE_DISPLAY_NAMES,
    GRAPH_STATE_KEYS,
    REQUIRED_NODES,
    GraphError,
    GraphNode,
    GraphState,
    NodeRecord,
    NodeStatus,
    OptimizationDecision,
    OptimizationOutcome,
    WorkflowResult,
    WorkflowStatus,
    create_graph_state,
)
from utils import config
from utils.state import AgentState


class TestGraphNodeEnumeration(unittest.TestCase):
    """Checks 1 and 2: the node vocabulary is complete and self-describing."""

    def test_seven_nodes_exist_one_per_agent(self):
        """Phase 9 appended Model Persistence as the seventh node.

        It is an agent like the others and is always in the default graph, because
        the documented run for the dashboard is ``python app.py --graph`` followed
        by ``streamlit run dashboard.py``.
        """
        self.assertEqual(len(GraphNode), 7)
        self.assertEqual(
            [node.value for node in GraphNode.ordered()],
            [
                "data_analysis",
                "preprocessing",
                "model_selection",
                "training",
                "evaluation",
                "optimization",
                "model_persistence",
            ],
        )

    def test_required_nodes_exclude_optimization_and_persistence(self):
        """A failure in either of the last two nodes degrades rather than fails.

        Neither produces a metric, and a run whose model simply was not written is
        still a valid machine learning run worth reporting as ``PARTIAL``.
        """
        self.assertNotIn(GraphNode.OPTIMIZATION, REQUIRED_NODES)
        self.assertNotIn(GraphNode.MODEL_PERSISTENCE, REQUIRED_NODES)
        self.assertEqual(len(REQUIRED_NODES), 5)

    def test_every_node_has_a_display_name(self):
        for node in GraphNode:
            self.assertIsInstance(node.display_name, str)
            self.assertTrue(node.display_name)
            self.assertIn(node, GRAPH_NODE_DISPLAY_NAMES)

    def test_node_values_are_unique(self):
        values = [node.value for node in GraphNode]
        self.assertEqual(len(values), len(set(values)))


class TestGraphStateCreation(unittest.TestCase):
    """Check 1: graph state creation."""

    def test_create_graph_state_populates_every_key(self):
        state = create_graph_state()
        for key in GRAPH_STATE_KEYS:
            self.assertIn(key, state, f"{key} missing from the initial state")

    def test_create_graph_state_defaults(self):
        state = create_graph_state()
        self.assertEqual(state["max_retries"], config.LANGGRAPH_MAX_RETRIES)
        self.assertTrue(state["enable_optimization"])
        self.assertTrue(state["cross_validation"])
        self.assertTrue(state["stop_on_failure"])
        self.assertFalse(state["stop_requested"])
        self.assertIsNone(state["failed_node"])
        self.assertEqual(state["node_records"], [])
        self.assertEqual(state["node_errors"], [])
        self.assertEqual(state["retries"], {})

    def test_create_graph_state_honours_overrides(self):
        state = create_graph_state(
            dataset_path="x.csv",
            target_column="y",
            enable_optimization=False,
            cross_validation=False,
            n_trials=7,
            max_retries=3,
            stop_on_failure=False,
        )
        self.assertEqual(state["dataset_path"], "x.csv")
        self.assertEqual(state["target_column"], "y")
        self.assertFalse(state["enable_optimization"])
        self.assertFalse(state["cross_validation"])
        self.assertEqual(state["n_trials"], 7)
        self.assertEqual(state["max_retries"], 3)
        self.assertFalse(state["stop_on_failure"])

    def test_every_node_starts_pending(self):
        state = create_graph_state()
        self.assertEqual(len(state["node_status"]), len(GraphNode))
        for value in state["node_status"].values():
            self.assertEqual(value, NodeStatus.PENDING.value)

    def test_graph_state_is_a_typed_dict(self):
        """LangGraph requires a dict-like schema, not a dataclass."""
        self.assertTrue(issubclass(GraphState, dict))
        self.assertEqual(set(GRAPH_STATE_KEYS), set(GraphState.__annotations__))


class TestAgentStatePreservation(unittest.TestCase):
    """Check 2: AgentState preservation."""

    def test_new_agent_state_created_when_none_given(self):
        state = create_graph_state()
        self.assertIsInstance(state["agent_state"], AgentState)

    def test_existing_agent_state_is_carried_by_reference(self):
        """The graph holds one reference; it must not copy or rebuild the state."""
        original = AgentState()
        original.dataset_name = "carried.csv"
        original.experiment_count = 3
        state = create_graph_state(agent_state=original)
        self.assertIs(state["agent_state"], original)
        self.assertEqual(state["agent_state"].dataset_name, "carried.csv")
        self.assertEqual(state["agent_state"].experiment_count, 3)

    def test_graph_does_not_duplicate_agent_state_results(self):
        """The graph may mirror inputs, never machine learning outputs.

        ``dataset_path`` and ``target_column`` are workflow *inputs*: the graph
        has to know them before the Data Analysis Agent has run, so mirroring
        them is deliberate. Everything the agents compute must be read back from
        the single carried AgentState instead of being copied into graph keys.
        """
        allowed_inputs = {"dataset_path", "target_column"}
        ml_fields = {f.name for f in fields(AgentState)}
        mirrored = (set(GRAPH_STATE_KEYS) & ml_fields) - allowed_inputs
        self.assertEqual(mirrored, set())

    def test_graph_state_carries_no_machine_learning_result_fields(self):
        result_keys = {
            "experiment_results",
            "cross_validation_results",
            "model_comparison",
            "evaluation_report",
            "optimization_report",
            "best_baseline_model",
            "data_split",
            "preprocessing_summary",
            "model_selection_summary",
            "analysis_summary",
        }
        self.assertEqual(set(GRAPH_STATE_KEYS) & result_keys, set())

    def test_agent_state_survives_a_node_update(self):
        state = create_graph_state()
        marker = AgentState()
        marker.dataset_name = "marker.csv"
        updated = {**state, "agent_state": marker}
        self.assertIs(updated["agent_state"], marker)
        self.assertEqual(updated["agent_state"].dataset_name, "marker.csv")


class TestJsonSafeSerialisation(unittest.TestCase):
    """Check 3: JSON-safe serialisation."""

    def test_records_are_dataclasses_with_to_dict(self):
        for cls in (GraphError, NodeRecord, WorkflowResult):
            self.assertTrue(is_dataclass(cls))
            self.assertTrue(hasattr(cls, "to_dict"))

    def test_graph_error_to_dict_is_json_safe(self):
        error = GraphError(
            node="training",
            error_type="RuntimeError",
            message="boom",
            attempt=1,
            retryable=True,
        )
        payload = error.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["retryable"], True)

    def test_node_record_to_dict_is_json_safe(self):
        record = NodeRecord(
            node="training",
            display_name="Training",
            status=NodeStatus.COMPLETED,
            attempts=2,
            retries=1,
            duration=1.25,
        )
        payload = record.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["status"], "completed")

    def test_workflow_result_to_dict_is_json_safe(self):
        result = WorkflowResult(
            workflow_status=WorkflowStatus.COMPLETED,
            agent_state=AgentState(),
            node_records=[
                NodeRecord(
                    node="training",
                    display_name="Training",
                    status=NodeStatus.COMPLETED,
                    attempts=1,
                    retries=0,
                    duration=0.5,
                )
            ],
            optimization_decision=OptimizationDecision.RUN,
            optimization_outcome=OptimizationOutcome.COMPLETED,
        )
        payload = result.to_dict()
        text = json.dumps(payload)
        self.assertEqual(json.loads(text), payload)
        self.assertEqual(payload["workflow_status"], "completed")
        self.assertEqual(payload["optimization_decision"], "run")
        self.assertEqual(payload["optimization_outcome"], "completed")
        self.assertEqual(payload["executed_nodes"], ["training"])
        self.assertEqual(payload["completed_nodes"], ["training"])
        self.assertEqual(payload["failed_nodes"], [])

    def test_enums_serialise_as_plain_strings(self):
        result = WorkflowResult(
            workflow_status=WorkflowStatus.PARTIAL,
            agent_state=AgentState(),
            optimization_decision=OptimizationDecision.SKIP_ALREADY_DONE,
            optimization_outcome=OptimizationOutcome.FAILED,
        )
        payload = result.to_dict()
        self.assertIsInstance(payload["workflow_status"], str)
        self.assertIsInstance(payload["optimization_decision"], str)
        self.assertIsInstance(payload["optimization_outcome"], str)
        json.dumps(payload)

    def test_workflow_result_embeds_agent_state_dict(self):
        state = AgentState()
        state.dataset_name = "embedded.csv"
        payload = WorkflowResult(
            workflow_status=WorkflowStatus.COMPLETED, agent_state=state
        ).to_dict()
        self.assertEqual(payload["agent_state"]["dataset_name"], "embedded.csv")


class TestWorkflowStatusDefaults(unittest.TestCase):
    """Check 4: default workflow status."""

    def test_initial_status_is_pending(self):
        self.assertEqual(create_graph_state()["workflow_status"], "pending")
        self.assertEqual(WorkflowStatus.PENDING.value, "pending")

    def test_initial_decision_is_not_a_run(self):
        state = create_graph_state()
        self.assertEqual(
            state["optimization_decision"], OptimizationDecision.SKIP_DISABLED.value
        )
        self.assertEqual(state["optimization_outcome"], "not_run")

    def test_status_success_property(self):
        self.assertTrue(WorkflowStatus.COMPLETED.is_success)
        self.assertTrue(WorkflowStatus.PARTIAL.is_success)
        self.assertFalse(WorkflowStatus.FAILED.is_success)
        self.assertFalse(WorkflowStatus.PENDING.is_success)
        self.assertFalse(WorkflowStatus.RUNNING.is_success)

    def test_node_status_success_property(self):
        self.assertTrue(NodeStatus.COMPLETED.is_success)
        for value in (NodeStatus.PENDING, NodeStatus.FAILED, NodeStatus.SKIPPED):
            self.assertFalse(value.is_success)

    def test_workflow_result_defaults(self):
        result = WorkflowResult(
            workflow_status=WorkflowStatus.COMPLETED, agent_state=AgentState()
        )
        self.assertEqual(result.node_records, [])
        self.assertEqual(result.errors, [])
        self.assertEqual(result.total_retries, 0)
        self.assertEqual(result.total_attempts, 0)
        self.assertIsNone(result.failed_node)
        self.assertIsNone(result.last_error)
        self.assertFalse(result.stopped_early)
        self.assertFalse(result.step_limit_reached)


class TestRetryCounter(unittest.TestCase):
    """Check 5: the retry counter."""

    def test_retry_counter_starts_empty(self):
        self.assertEqual(create_graph_state()["retries"], {})

    def test_graph_errors_do_not_shadow_agent_state_errors(self):
        """Both exist during a run, so the names must stay distinct."""
        state = create_graph_state()
        self.assertIn("node_errors", state)
        self.assertNotIn("errors", state)
        self.assertEqual(state["node_errors"], [])

    def test_retry_counter_is_per_node(self):
        state = create_graph_state()
        state["retries"] = {"training": 2, "evaluation": 0}
        self.assertEqual(state["retries"]["training"], 2)
        self.assertEqual(state["retries"]["evaluation"], 0)

    def test_total_retries_sums_the_counter(self):
        result = WorkflowResult(
            workflow_status=WorkflowStatus.FAILED,
            agent_state=AgentState(),
            total_retries=3,
        )
        self.assertEqual(result.total_retries, 3)

    def test_attempts_are_recorded_per_node(self):
        """Each record keeps its own attempt count; the sum is built by the
        workflow layer, which is covered in the routing tests."""
        result = WorkflowResult(
            workflow_status=WorkflowStatus.FAILED,
            agent_state=AgentState(),
            node_records=[
                NodeRecord("training", "Training", NodeStatus.FAILED, 2, 1, 0.1),
                NodeRecord("evaluation", "Evaluation", NodeStatus.COMPLETED, 1, 0, 0.2),
            ],
            total_attempts=3,
        )
        self.assertEqual(result.record_for(GraphNode.TRAINING).attempts, 2)
        self.assertEqual(result.record_for(GraphNode.TRAINING).retries, 1)
        self.assertEqual(result.record_for(GraphNode.EVALUATION).attempts, 1)
        self.assertEqual(result.total_attempts, 3)
        self.assertEqual(result.failed_nodes, ["training"])

    def test_default_max_retries_is_one(self):
        self.assertEqual(config.LANGGRAPH_MAX_RETRIES, 1)
        self.assertEqual(create_graph_state()["max_retries"], 1)


class TestDerivedViews(unittest.TestCase):
    """The typed views a caller relies on to inspect a run."""

    def _result(self):
        return WorkflowResult(
            workflow_status=WorkflowStatus.FAILED,
            agent_state=AgentState(),
            node_records=[
                NodeRecord("data_analysis", "Data Analysis", NodeStatus.COMPLETED, 1, 0, 0.1),
                NodeRecord(
                    "preprocessing",
                    "Preprocessing",
                    NodeStatus.FAILED,
                    2,
                    1,
                    0.2,
                    error_type="RuntimeError",
                    error_message="boom",
                ),
            ],
            errors=[
                GraphError("preprocessing", "RuntimeError", "boom", 2, True),
            ],
            failed_node="preprocessing",
            total_retries=1,
        )

    def test_record_for_accepts_enum_or_name(self):
        result = self._result()
        by_enum = result.record_for(GraphNode.PREPROCESSING)
        by_name = result.record_for("preprocessing")
        self.assertEqual(by_enum, by_name)
        self.assertEqual(by_enum.status, NodeStatus.FAILED)

    def test_record_for_missing_node_returns_none(self):
        self.assertIsNone(self._result().record_for(GraphNode.TRAINING))

    def test_last_error_is_the_final_one(self):
        self.assertEqual(self._result().last_error.error_type, "RuntimeError")

    def test_final_agent_state_is_the_carried_state(self):
        state = AgentState()
        result = WorkflowResult(
            workflow_status=WorkflowStatus.COMPLETED, agent_state=state
        )
        self.assertIs(result.final_agent_state, state)

    def test_executed_nodes_follow_execution_order(self):
        self.assertEqual(
            self._result().executed_nodes, ["data_analysis", "preprocessing"]
        )


if __name__ == "__main__":
    unittest.main()
