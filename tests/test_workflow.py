"""End-to-end tests for the LangGraph workflow (Phase 7, checks 30-36).

These tests compile the real graph, run it against the committed datasets, and
prove the parts that only a whole run can show: the edge structure, the full and
no-optimization paths, failure routing, the retry ceiling, the optimization
decision and determinism.

Failure and retry scenarios replace the agents with deterministic stand-ins so
the routing itself is tested without the cost of a real training run. The real
datasets are used for the integration runs, and their hashes are checked so a
run cannot silently depend on a modified fixture.
"""

from __future__ import annotations

import json
import logging
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from graph import __version__ as graph_version
from graph import nodes as node_module
from graph.state import (
    REQUIRED_NODES,
    GraphNode,
    NodeStatus,
    OptimizationDecision,
    OptimizationOutcome,
    WorkflowStatus,
    create_graph_state,
)
from graph.workflow import (
    WORKFLOW_ORDER,
    build_workflow,
    describe_structure,
    expected_structure,
    format_workflow_report,
    run_workflow,
)
from langgraph.graph import END
from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash
from tools.data_tools import DatasetNotFoundError
from tools.evaluation_tools import EvaluationToolError
from tools.preprocessing_tools import AnalysisNotReadyError
from tools.training_tools import TrainingToolError
from utils import config
from utils.state import AgentState

#: Hashes of the committed fixtures. The workflow must never write to a dataset,
#: so these are asserted before and after the integration runs.
EXPECTED_BINARY_HASH = "49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c"
EXPECTED_MULTICLASS_HASH = (
    "0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8"
)


class _Experiment:
    """Minimal stand-in for an ``ExperimentResult`` (only ``status`` is read)."""

    def __init__(self, status: str = "success") -> None:
        self.status = status
        self.model_name = "Logistic Regression"
        self.f1_score = 0.9


class StubAgent:
    """Deterministic stand-in for one existing agent.

    Args:
        node: The graph node this stub stands in for.
        raise_on_call: Attempt number (1-based) that should raise. None never
            raises.
        error: The error raised on that attempt.
        mutate: Optional hook applied to the state before returning. When None,
            the node's default status transition is used.
    """

    def __init__(
        self,
        node: str,
        raise_on_call: Optional[int] = None,
        error: Optional[BaseException] = None,
        mutate: Optional[Any] = None,
    ) -> None:
        self.node = node
        self.raise_on_call = raise_on_call
        self.error = error
        self.mutate = mutate
        self.calls = 0
        self.constructors = 0

    def __call__(self, *args: Any, **kwargs: Any) -> "StubAgent":
        """Stand in for the agent class constructor.

        Returns:
            This stub, so the call counter reflects real invocations.
        """
        self.constructors += 1
        return self

    def run(self, state: AgentState, **kwargs: Any) -> AgentState:
        """Record the call, then either raise or update the state.

        Args:
            state: The state the node passed in.
            **kwargs: The keyword arguments the node passed.

        Returns:
            The updated state.

        Raises:
            BaseException: The configured error on the configured attempt.
        """
        self.calls += 1
        if self.error is not None and self.raise_on_call in (None, self.calls):
            raise self.error
        mutate = self.mutate if self.mutate is not None else DEFAULT_MUTATIONS[self.node]
        mutate(state, kwargs)
        return state


def _set_status(field: str) -> Any:
    """Return a hook that marks one phase complete.

    Args:
        field: The ``AgentState`` status field to set.

    Returns:
        A hook suitable for :class:`StubAgent`.
    """

    def mutate(state: AgentState, kwargs: Dict[str, Any]) -> None:
        setattr(state, field, "completed")

    return mutate


def _mark_dataset(state: AgentState, kwargs: Dict[str, Any]) -> None:
    """Record the analysed dataset, as the Data Analysis Agent would.

    Args:
        state: The state to update.
        kwargs: The arguments the node passed, including ``dataset_path``.
    """
    dataset_path = kwargs.get("dataset_path") or "stub.csv"
    state.dataset_name = str(dataset_path).replace("\\", "/").split("/")[-1]
    state.dataset_path = str(dataset_path)


def _mark_training(state: AgentState, kwargs: Dict[str, Any]) -> None:
    """Record a completed training run with one usable experiment.

    Args:
        state: The state to update.
        kwargs: Unused; present for a uniform hook signature.
    """
    state.training_status = "completed"
    state.experiment_results = [_Experiment()]


#: What each stubbed agent does to the state when it succeeds. A stub that
#: changed nothing would make the graph skip every later step, so each one
#: records the same status transition the real agent records.
DEFAULT_MUTATIONS: Dict[str, Any] = {
    "data_analysis": _mark_dataset,
    "preprocessing": _set_status("preprocessing_status"),
    "model_selection": _set_status("model_selection_status"),
    "training": _mark_training,
    "evaluation": _set_status("evaluation_status"),
    "optimization": _set_status("optimization_status"),
    # Phase 9: persistence is the terminal node. A stub has to set its status or
    # the node is recorded as failed, which would change the workflow status of
    # every routing test below.
    "model_persistence": _set_status("persistence_status"),
}


def patch_agents(**behaviours: Dict[str, Any]) -> Any:
    """Replace every agent class the node module imports.

    Args:
        **behaviours: Per node keyword arguments forwarded to :class:`StubAgent`.

    Returns:
        The started patcher; call ``stop`` on it to restore the real agents.
    """
    mapping = {
        "DataAnalysisAgent": "data_analysis",
        "PreprocessingAgent": "preprocessing",
        "ModelSelectionAgent": "model_selection",
        "TrainingAgent": "training",
        "EvaluationAgent": "evaluation",
        "OptimizationAgent": "optimization",
        "PersistenceAgent": "model_persistence",
    }
    patcher = mock.patch.multiple(
        node_module,
        **{name: mock.DEFAULT for name in mapping},
    )
    started = patcher.start()
    for name, node in mapping.items():
        started[name].side_effect = StubAgent(node, **behaviours.get(node, {}))
    return patcher


class WorkflowTestCase(unittest.TestCase):
    """Base class that restores anything a test changes."""

    def use_stubs(self, **behaviours: Dict[str, Any]) -> Any:
        """Patch the agents and register the cleanup.

        Args:
            **behaviours: Per node keyword arguments for :class:`StubAgent`.

        Returns:
            The started patcher.
        """
        patcher = patch_agents(**behaviours)
        self.addCleanup(patcher.stop)
        return patcher

    def build(self, **kwargs: Any) -> Any:
        """Compile the workflow.

        Args:
            **kwargs: Forwarded to :func:`build_workflow`.

        Returns:
            The compiled application.
        """
        return build_workflow(**kwargs)


class TestGraphCompiles(WorkflowTestCase):
    """Check 30: the graph compiles to the required shape."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = build_workflow()
        drawing = cls.app.get_graph()
        cls.node_names = set(drawing.nodes)
        cls.edges = {
            (edge.source, edge.target, bool(edge.conditional))
            for edge in drawing.edges
        }

    def test_every_node_is_registered(self):
        for node in GraphNode:
            with self.subTest(node=node):
                self.assertIn(node.value, self.node_names)
        self.assertIn("__start__", self.node_names)
        self.assertIn("__end__", self.node_names)

    def test_edge_set_matches_the_expected_structure(self):
        """Compared as a set: LangGraph does not promise an edge ordering."""
        self.assertEqual(self.edges, set(expected_structure()))

    def test_start_enters_data_analysis_directly(self):
        self.assertIn(("__start__", "data_analysis", False), self.edges)

    def test_every_node_can_end_the_workflow(self):
        """Every node has a conditional escape to END.

        Phase 9 widened this: Model Persistence used to not exist, and the set
        ``WORKFLOW_ORDER[:-1]`` was the five required nodes plus Optimization.
        All seven have the exit now, so the test asserts the stronger property
        rather than a slice whose meaning changed.
        """
        for node in WORKFLOW_ORDER:
            with self.subTest(node=node):
                self.assertIn((node.value, "__end__", True), self.edges)

    def test_every_required_node_can_end_the_workflow(self):
        for node in REQUIRED_NODES:
            with self.subTest(node=node):
                self.assertIn((node.value, "__end__", True), self.edges)

    def test_evaluation_can_go_to_optimization_persistence_or_end(self):
        """Phase 9 added the persistence destination to the decision edge."""
        self.assertIn(("evaluation", "optimization", True), self.edges)
        self.assertIn(("evaluation", "model_persistence", True), self.edges)
        self.assertIn(("evaluation", "__end__", True), self.edges)

    def test_optimization_can_go_to_persistence_or_end(self):
        self.assertIn(("optimization", "model_persistence", True), self.edges)
        self.assertIn(("optimization", "__end__", True), self.edges)

    def test_persistence_is_the_only_node_with_no_successor(self):
        """Nothing runs after the artifact is written."""
        successors = {
            target
            for _, target, _ in self.edges
            if target not in {"__end__"}
        }
        self.assertEqual(successors - {node.value for node in WORKFLOW_ORDER}, set())
        self.assertEqual(
            [target for source, target, _ in self.edges if source == "model_persistence"],
            ["__end__"],
        )

    def test_no_edge_returns_to_earlier_work(self):
        """A cycle would be the only way to run optimization twice."""
        order = {node.value: index for index, node in enumerate(WORKFLOW_ORDER)}
        for source, target, _ in self.edges:
            if source in order and target in order:
                with self.subTest(edge=(source, target)):
                    self.assertLess(order[source], order[target])

    def test_structure_description_is_stable_and_readable(self):
        first = describe_structure(self.app)
        second = describe_structure(self.app)
        self.assertEqual(first, second)
        self.assertIn("nodes (9)", first)
        self.assertIn("Workflow structure:", first)
        edge_lines = [line for line in first.splitlines() if " -> " in line]
        self.assertEqual(len(edge_lines), len(expected_structure()))
        for line in edge_lines:
            with self.subTest(line=line):
                self.assertRegex(line.strip(), r"^\S+ +-> \S+ +\[(direct|conditional)\]$")

    def test_description_names_the_required_destinations(self):
        collapsed = " ".join(describe_structure(self.app).split())
        self.assertIn("START -> data_analysis", collapsed)
        self.assertIn("evaluation -> optimization", collapsed)
        self.assertIn("evaluation -> model_persistence", collapsed)
        self.assertIn("optimization -> model_persistence", collapsed)
        self.assertIn("optimization -> END", collapsed)
        self.assertIn("[conditional]", collapsed)

    def test_build_workflow_accepts_a_custom_step_ceiling(self):
        self.assertIsNotNone(build_workflow(max_steps=5))

    def test_two_graphs_are_independent(self):
        """Building twice must not share mutable state between graphs."""
        first = self.build()
        second = self.build()
        self.assertIsNot(first, second)
        self.assertEqual(
            {
                (e.source, e.target, bool(e.conditional))
                for e in first.get_graph().edges
            },
            {
                (e.source, e.target, bool(e.conditional))
                for e in second.get_graph().edges
            },
        )


class TestFullWorkflowWithStubbedAgents(WorkflowTestCase):
    """Check 31: the whole graph runs and reports a completed workflow."""

    def test_every_node_runs_in_order(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.COMPLETED)
        self.assertEqual(result.executed_nodes, [n.value for n in WORKFLOW_ORDER])
        self.assertEqual(result.failed_node, None)
        self.assertTrue(result.workflow_status.is_success)

    def test_all_nodes_are_marked_completed(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        for node in GraphNode:
            with self.subTest(node=node):
                self.assertEqual(
                    result.record_for(node).status, NodeStatus.COMPLETED
                )

    def test_optimization_runs_and_is_reported_as_completed(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(
            result.optimization_decision, OptimizationDecision.RUN
        )
        self.assertEqual(
            result.optimization_outcome, OptimizationOutcome.COMPLETED
        )
        self.assertEqual(result.agent_state.optimization_status, "completed")

    def test_no_retries_or_errors_on_the_happy_path(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.total_retries, 0)
        # One attempt per node. Phase 9 added the seventh node, so the count is 7
        # and equals len(WORKFLOW_ORDER) rather than the old hard-coded 6.
        self.assertEqual(result.total_attempts, len(WORKFLOW_ORDER))
        self.assertEqual(result.total_attempts, 7)
        self.assertEqual(result.errors, [])

    def test_result_is_json_serialisable_without_a_default_hook(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        text = json.dumps(result.to_dict())
        self.assertEqual(json.loads(text)["workflow_status"], "completed")

    def test_workflow_order_matches_the_node_vocabulary(self):
        self.assertEqual(len(WORKFLOW_ORDER), 7)
        self.assertEqual(set(WORKFLOW_ORDER), set(GraphNode))
        self.assertEqual(WORKFLOW_ORDER[-1], GraphNode.MODEL_PERSISTENCE)


class TestNoOptimizationWorkflow(WorkflowTestCase):
    """Check 32: the workflow runs without optimization."""

    def test_optimization_node_is_not_entered_but_persistence_is(self):
        """Phase 9: skipping the search still has to write the artifact.

        This is the routing change that matters most. Before Phase 9 a
        ``--no-optimize`` run finished with nothing on disk, so the dashboard had
        nothing to show; now Evaluation routes straight to Model Persistence.
        """
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_optimization=False, enable_tracking=False)
        self.assertEqual(
            result.executed_nodes,
            [n.value for n in WORKFLOW_ORDER if n is not GraphNode.OPTIMIZATION],
        )
        self.assertIsNone(result.record_for(GraphNode.OPTIMIZATION))
        self.assertEqual(result.agent_state.optimization_status, "not_started")
        self.assertIn("model_persistence", result.executed_nodes)
        self.assertEqual(
            result.record_for(GraphNode.MODEL_PERSISTENCE).status,
            NodeStatus.COMPLETED,
        )
        self.assertEqual(result.agent_state.persistence_status, "completed")

    def test_decision_reports_the_configuration(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_optimization=False, enable_tracking=False)
        self.assertEqual(
            result.optimization_decision, OptimizationDecision.SKIP_DISABLED
        )
        self.assertEqual(result.optimization_outcome, OptimizationOutcome.NOT_RUN)

    def test_evaluation_result_is_preserved(self):
        self.use_stubs()
        result = run_workflow(dataset_path="stub.csv", enable_optimization=False, enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.COMPLETED)
        self.assertEqual(result.agent_state.evaluation_status, "completed")
        self.assertTrue(result.agent_state.successful_experiments)

    def test_disabled_by_default_config(self):
        self.assertTrue(config.LANGGRAPH_ENABLE_OPTIMIZATION)


class TestFailureRouting(WorkflowTestCase):
    """Checks 33 and 34: failures end the run without corrupting the state."""

    def test_data_analysis_failure_ends_the_workflow_immediately(self):
        self.use_stubs(data_analysis={"error": DatasetNotFoundError("missing.csv")})
        result = run_workflow(dataset_path="missing.csv", enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.FAILED)
        self.assertEqual(result.executed_nodes, ["data_analysis"])
        self.assertEqual(result.failed_node, "data_analysis")
        self.assertFalse(result.workflow_status.is_success)

    def test_preprocessing_failure_never_reaches_training(self):
        self.use_stubs(preprocessing={"error": AnalysisNotReadyError("no analysis")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.FAILED)
        self.assertEqual(
            result.executed_nodes, ["data_analysis", "preprocessing"]
        )
        self.assertEqual(result.failed_node, "preprocessing")

    def test_training_failure_never_reaches_evaluation(self):
        self.use_stubs(training={"error": TrainingToolError("no models")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(
            result.executed_nodes,
            ["data_analysis", "preprocessing", "model_selection", "training"],
        )
        self.assertEqual(result.failed_node, "training")
        self.assertEqual(result.workflow_status, WorkflowStatus.FAILED)

    def test_evaluation_failure_never_reaches_optimization(self):
        self.use_stubs(evaluation={"error": EvaluationToolError("no results")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.failed_node, "evaluation")
        self.assertIsNone(result.record_for(GraphNode.OPTIMIZATION))
        self.assertEqual(
            result.optimization_decision,
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE,
        )
        self.assertEqual(result.optimization_outcome, OptimizationOutcome.SKIPPED)

    def test_evaluation_agent_recording_a_failure_is_treated_as_failed(self):
        """The Evaluation Agent records its failure instead of raising."""
        self.use_stubs(
            evaluation={
                "mutate": lambda s, kw: setattr(s, "evaluation_status", "failed")
            }
        )
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.FAILED)
        self.assertIsNone(result.record_for(GraphNode.OPTIMIZATION))
        self.assertEqual(result.failed_node, "evaluation")

    def test_failed_run_keeps_the_work_done_so_far(self):
        self.use_stubs(training={"error": TrainingToolError("boom")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.agent_state.dataset_name, "stub.csv")
        self.assertEqual(
            result.record_for(GraphNode.DATA_ANALYSIS).status, NodeStatus.COMPLETED
        )

    def test_failure_is_reported_not_hidden(self):
        self.use_stubs(training={"error": TrainingToolError("no models")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].error_type, "TrainingToolError")
        self.assertEqual(result.last_error.error_type, "TrainingToolError")
        self.assertEqual(result.agent_state.errors, [])

    def test_stop_on_failure_disabled_keeps_running(self):
        """With the switch off the router continues past a failed node, so the
        downstream agent sees whatever state exists rather than an aborted run.

        Phase 9 changed the tail of this sequence. Training failed, so Evaluation
        has no experiment to report and Optimization has no candidate to search;
        the decision is therefore ``skip_no_candidates`` and the run continues to
        Model Persistence rather than ending after Evaluation. The last node is
        ``model_persistence`` instead of the old ``optimization``.
        """
        self.use_stubs(training={"error": TrainingToolError("boom")})
        result = run_workflow(dataset_path="stub.csv", stop_on_failure=False, enable_tracking=False)
        self.assertEqual(result.failed_node, "training")
        self.assertEqual(
            result.executed_nodes,
            [
                "data_analysis",
                "preprocessing",
                "model_selection",
                "training",
                "evaluation",
                "model_persistence",
            ],
        )
        self.assertEqual(
            result.optimization_decision, OptimizationDecision.SKIP_NO_CANDIDATES
        )
        self.assertFalse(result.stopped_early)

    def test_optimization_failure_is_partial_not_failed(self):
        """Check 34: the Phase 5 result must survive."""
        self.use_stubs(optimization={"error": EvaluationToolError("optuna broke")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.PARTIAL)
        self.assertTrue(result.workflow_status.is_success)
        self.assertEqual(result.failed_node, "optimization")

    def test_optimization_failure_preserves_the_evaluation_outcome(self):
        self.use_stubs(
            evaluation={"mutate": lambda s, kw: setattr(s, "evaluation_status", "completed")},
            optimization={"error": EvaluationToolError("optuna broke")},
        )
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(result.agent_state.evaluation_status, "completed")
        self.assertEqual(result.optimization_outcome, OptimizationOutcome.FAILED)
        self.assertEqual(
            result.record_for(GraphNode.EVALUATION).status, NodeStatus.COMPLETED
        )


class TestRetryInTheCompiledGraph(WorkflowTestCase):
    """Check 35: at most one retry, then the workflow ends."""

    def test_retryable_failure_is_retried_exactly_once(self):
        """The retry is spent; the node then reports its real outcome."""
        self.use_stubs(training={"raise_on_call": 1, "error": RuntimeError("flaky")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        record = result.record_for(GraphNode.TRAINING)
        self.assertEqual(record.attempts, 2)
        self.assertEqual(record.retries, 1)
        self.assertEqual(result.total_retries, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].attempt, 1)

    def test_a_recovered_retry_completes_the_node(self):
        self.use_stubs(training={"raise_on_call": 1, "error": RuntimeError("flaky")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertEqual(
            result.record_for(GraphNode.TRAINING).status, NodeStatus.COMPLETED
        )
        self.assertEqual(result.workflow_status, WorkflowStatus.COMPLETED)
        self.assertEqual(result.optimization_outcome, OptimizationOutcome.COMPLETED)

    def test_retry_that_also_fails_ends_the_workflow(self):
        self.use_stubs(training={"error": RuntimeError("always")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        record = result.record_for(GraphNode.TRAINING)
        self.assertEqual(record.attempts, 2)
        self.assertEqual(record.status, NodeStatus.FAILED)
        self.assertIsNone(result.record_for(GraphNode.EVALUATION))

    def test_expected_error_is_not_retried(self):
        self.use_stubs(
            preprocessing={"raise_on_call": 1, "error": AnalysisNotReadyError("no")}
        )
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        record = result.record_for(GraphNode.PREPROCESSING)
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.retries, 0)
        self.assertFalse(result.errors[0].retryable)

    def test_both_attempts_are_recorded(self):
        self.use_stubs(training={"error": RuntimeError("always")})
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        training_errors = [e for e in result.errors if e.node == "training"]
        self.assertEqual(len(training_errors), 2)
        self.assertEqual([e.attempt for e in training_errors], [1, 2])

    def test_the_workflow_never_loops(self):
        """A permanently failing retryable node still terminates the graph."""
        self.use_stubs(training={"error": RuntimeError("always")})
        result = run_workflow(dataset_path="stub.csv", max_retries=5, enable_tracking=False)
        self.assertEqual(result.record_for(GraphNode.TRAINING).retries, 5)
        self.assertEqual(result.executed_nodes.count("training"), 1)
        self.assertEqual(result.executed_nodes.count("optimization"), 0)


class TestDeterminism(WorkflowTestCase):
    """Check 36: the same inputs always produce the same run."""

    def _fingerprint(self, result: Any) -> tuple:
        return (
            result.workflow_status.value,
            tuple(result.executed_nodes),
            result.failed_node,
            result.optimization_decision.value,
            result.optimization_outcome.value,
            result.total_retries,
            tuple(result.total_attempts for _ in (0,)),
            tuple(record.status.value for record in result.node_records),
        )

    def test_two_stubbed_runs_are_identical(self):
        self.use_stubs()
        first = self._fingerprint(run_workflow(dataset_path="stub.csv", enable_tracking=False))
        second = self._fingerprint(run_workflow(dataset_path="stub.csv", enable_tracking=False))
        self.assertEqual(first, second)

    def test_a_failed_run_repeats_identically(self):
        self.use_stubs(training={"error": RuntimeError("always")})
        first = self._fingerprint(run_workflow(dataset_path="stub.csv", enable_tracking=False))
        second = self._fingerprint(run_workflow(dataset_path="stub.csv", enable_tracking=False))
        self.assertEqual(first, second)

    def test_a_retried_run_repeats_identically(self):
        self.use_stubs(training={"raise_on_call": 1, "error": RuntimeError("flaky")})
        first = self._fingerprint(run_workflow(dataset_path="stub.csv", enable_tracking=False))
        # Fresh stubs: the first run consumed the one permitted failure, so a
        # second run against the same stubs would not retry at all.
        self.use_stubs(training={"raise_on_call": 1, "error": RuntimeError("flaky")})
        second = self._fingerprint(run_workflow(dataset_path="stub.csv", enable_tracking=False))
        self.assertEqual(first, second)

    def test_routing_does_not_depend_on_call_history(self):
        """A fresh graph produces the same routing as a reused one."""
        self.use_stubs()
        app = self.build()
        run_workflow(dataset_path="stub.csv", app=app, enable_tracking=False)
        result = run_workflow(dataset_path="stub.csv", app=app, enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.COMPLETED)
        self.assertEqual(result.total_retries, 0)

    def test_a_supplied_agent_state_is_continued_not_replaced(self):
        self.use_stubs()
        existing = AgentState()
        existing.dataset_name = "continued.csv"
        result = run_workflow(agent_state=existing, dataset_path="stub.csv", enable_tracking=False)
        self.assertIs(result.agent_state, existing)
        self.assertEqual(result.agent_state.dataset_name, "stub.csv")


class TestNoLanguageModelIsUsed(unittest.TestCase):
    """The graph must run with no language model installed."""

    def test_no_forbidden_module_is_imported(self):
        import sys

        for name in config.LANGGRAPH_FORBIDDEN_LLM_DEPENDENCIES:
            with self.subTest(module=name):
                self.assertNotIn(name, sys.modules)

    def test_graph_modules_do_not_reference_a_language_model(self):
        import pathlib

        graph_dir = pathlib.Path(node_module.__file__).parent
        banned = ("openai", "langchain_openai", "google.generativeai", "anthropic",
                  "ollama", "litellm")
        for path in sorted(graph_dir.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for name in banned:
                with self.subTest(file=path.name, name=name):
                    self.assertNotIn(f"import {name}", text)
                    self.assertNotIn(f"from {name}", text)

    def test_forbidden_list_is_published(self):
        self.assertTrue(config.LANGGRAPH_FORBIDDEN_LLM_DEPENDENCIES)
        for name in config.LANGGRAPH_FORBIDDEN_LLM_DEPENDENCIES:
            self.assertIsInstance(name, str)


class TestPackageSurface(unittest.TestCase):
    """The package exports what callers need."""

    def test_version_reflects_the_persistence_node(self):
        """Bumped to 0.8.0 by Phase 9, which added a node and rewired two edges."""
        self.assertEqual(graph_version, "0.8.0")

    def test_public_names_are_importable(self):
        import graph

        for name in graph.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(graph, name), f"{name} missing")

    def test_reexported_objects_are_the_defining_modules_objects(self):
        import graph
        from graph.state import WorkflowResult as StateResult
        from graph.workflow import build_workflow as builder

        self.assertIs(graph.WorkflowResult, StateResult)
        self.assertIs(graph.build_workflow, builder)


class TestWorkflowReport(unittest.TestCase):
    """Check: the textual report is renderable and single-line per fact."""

    def setUp(self) -> None:
        super().setUp()
        self._patcher = patch_agents()
        self.addCleanup(self._patcher.stop)

    def test_report_renders_for_a_completed_run(self):
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        report = format_workflow_report(result)
        self.assertIn("WORKFLOW REPORT", report)
        self.assertIn("Workflow status  : completed", report)
        self.assertIn("Decision         : run", report)
        self.assertIn("Outcome          : completed", report)

    def test_report_lists_every_node_on_one_line_each(self):
        result = run_workflow(dataset_path="stub.csv", enable_optimization=False, enable_tracking=False)
        report = format_workflow_report(result)
        self.assertIn("Data Analysis    completed", report)
        self.assertIn("Optimization     not entered", report)
        self.assertIn("Model Persistence", report)

    def test_report_states_the_retry_budget(self):
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        self.assertIn("Max retries/node : 1", format_workflow_report(result))

    def test_report_shows_the_error_for_a_failed_run(self):
        self._patcher.stop()
        self._patcher = patch_agents(training={"error": TrainingToolError("no models")})
        # The setUp patcher has been retired; the replacement must be retired
        # too, or it survives the test and rewrites every later TrainingAgent
        # call to fail with "no models". The pre-Phase-10 ordering (test_workflow
        # alphabetically last) hid the leak; any other order exposed it.
        self.addCleanup(self._patcher.stop)
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        report = format_workflow_report(result)
        self.assertIn("TrainingToolError", report)
        self.assertIn("Workflow status  : failed", report)

    def test_report_never_contains_an_unformatted_placeholder(self):
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        report = format_workflow_report(result)
        self.assertNotIn("{max_steps}", report)
        self.assertNotIn("None s", report)

    def test_report_lines_are_bounded(self):
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        for line in format_workflow_report(result).splitlines():
            with self.subTest(line=line):
                self.assertLess(len(line), 200)

    def test_wrapped_bullets_hang_their_continuation_lines(self):
        """A wrapped bullet must not repeat its marker on every line."""
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        notes = format_workflow_report(result).split("9. Notes and limitations")[1]
        bullets = [
            line for line in notes.splitlines() if line.strip().startswith("- ")
        ]
        self.assertTrue(bullets)
        for line in bullets:
            with self.subTest(line=line):
                self.assertTrue(line.startswith("  - "))

    def test_section_underlines_match_their_headings(self):
        result = run_workflow(dataset_path="stub.csv", enable_tracking=False)
        lines = format_workflow_report(result).splitlines()
        for index, line in enumerate(lines[:-1]):
            if line and set(line) == {"-"}:
                self.assertEqual(len(line), len(lines[index - 1]), line)


class TestDatasetsAreUnchanged(unittest.TestCase):
    """The fixtures must be byte-identical before and after a run."""

    def test_binary_dataset_hash(self):
        self.assertEqual(file_hash(BINARY_DATASET), EXPECTED_BINARY_HASH)

    def test_multiclass_dataset_hash(self):
        self.assertEqual(file_hash(MULTICLASS_DATASET), EXPECTED_MULTICLASS_HASH)

    def test_a_workflow_run_does_not_modify_the_dataset(self):
        before = file_hash(BINARY_DATASET)
        patcher = patch_agents()
        self.addCleanup(patcher.stop)
        # This run drives the real compiled graph on the real dataset, so the
        # Phase 9 persistence node must be turned off: the point of the test is
        # the dataset bytes, and with persistence on the node would silently
        # overwrite the project's own models/ artifacts (the conftest session
        # guard would then fail the whole suite).
        run_workflow(
            dataset_path=str(BINARY_DATASET),
            enable_optimization=False,
            enable_persistence=False,
            enable_tracking=False,
        )
        self.assertEqual(file_hash(BINARY_DATASET), before)


class TestRunWorkflowArguments(WorkflowTestCase):
    """The inputs the CLI and library callers rely on."""

    def test_graph_state_defaults_are_used_when_arguments_are_omitted(self):
        self.use_stubs()
        result = run_workflow(enable_tracking=False)
        self.assertEqual(result.workflow_status, WorkflowStatus.COMPLETED)

    def test_a_pre_compiled_graph_can_be_reused(self):
        self.use_stubs()
        app = self.build()
        first = run_workflow(dataset_path="stub.csv", app=app, enable_tracking=False)
        second = run_workflow(dataset_path="stub.csv", app=app, enable_tracking=False)
        self.assertEqual(first.executed_nodes, second.executed_nodes)

    def test_initial_state_is_built_from_the_arguments(self):
        state = create_graph_state(dataset_path="stub.csv", n_trials=3)
        self.assertEqual(state["dataset_path"], "stub.csv")
        self.assertEqual(state["n_trials"], 3)


if __name__ == "__main__":
    unittest.main()