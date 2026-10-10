"""Tests for the graph nodes (Phase 7, checks 6-13).

Covers the node-to-agent mapping, the recorded node status, the recorded
errors, failure propagation, the bounded retry policy and the single-line
logging contract.

Failure routing and retry are expressed through ``_execute_node`` here, and are
proven end to end through a real compiled graph in ``tests/test_workflow.py``.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

from graph import nodes
from graph.nodes import (
    NODE_FUNCTIONS,
    _execute_node,
    build_agent_invoker,
    data_analysis_node,
    evaluation_node,
    log_event,
    model_selection_node,
    optimization_node,
    preprocessing_node,
    resolve_max_retries,
    single_line,
    training_node,
)
from graph.state import (
    GraphError,
    GraphNode,
    GraphState,
    NodeStatus,
    OptimizationDecision,
    REQUIRED_NODES,
    create_graph_state,
)
from tools.data_tools import DataToolError, DatasetNotFoundError
from tools.optimization_tools import OptimizationToolError
from tools.preprocessing_tools import PreprocessingError
from tools.training_tools import TrainingToolError
from utils import config
from utils.state import AgentState

#: Every agent class the node module imports, so a test can assert that a node
#: touched exactly one of them.
AGENT_ATTRIBUTES = {
    GraphNode.DATA_ANALYSIS: "DataAnalysisAgent",
    GraphNode.PREPROCESSING: "PreprocessingAgent",
    GraphNode.MODEL_SELECTION: "ModelSelectionAgent",
    GraphNode.TRAINING: "TrainingAgent",
    GraphNode.EVALUATION: "EvaluationAgent",
    GraphNode.OPTIMIZATION: "OptimizationAgent",
    # Phase 9: the seventh agent, invoked by the terminal node.
    GraphNode.MODEL_PERSISTENCE: "PersistenceAgent",
}


class AgentRecorder:
    """Records how a node invoked its agent.

    The graph instantiates an agent and then calls ``run`` on it, so both the
    construction and the call arguments have to be captured to prove that a node
    touched exactly one agent and passed the expected inputs.
    """

    def __init__(
        self,
        name: str,
        error: Optional[BaseException] = None,
        mutate: Optional[Callable[[AgentState], None]] = None,
    ) -> None:
        """Store the configured behaviour.

        Args:
            name: The agent class name being replaced.
            error: When given, ``run`` raises this instead of returning.
            mutate: Optional hook run before ``run`` returns, used to simulate
                an agent that records a phase-level failure on the state.
        """
        self.name = name
        self.error = error
        self.mutate = mutate
        self.constructions = 0
        self.calls: List[tuple] = []

    def __call__(self, *args: Any, **kwargs: Any) -> "AgentRecorder":
        """Stand in for the agent class constructor.

        Returns:
            An object exposing ``run`` with the configured behaviour.
        """
        self.constructions += 1
        return self

    def run(self, state: AgentState, **kwargs: Any) -> AgentState:
        """Record the call, then behave as configured.

        Args:
            state: The state the node passed in.
            **kwargs: The keyword arguments the node passed.

        Returns:
            The unchanged state.

        Raises:
            BaseException: The configured error, when one was given.
        """
        self.calls.append((state, kwargs))
        if self.error is not None:
            raise self.error
        if self.mutate is not None:
            self.mutate(state)
        return state

    @property
    def last_kwargs(self) -> Dict[str, Any]:
        """Return the keyword arguments of the most recent ``run`` call."""
        return self.calls[-1][1]

    @property
    def last_state(self) -> AgentState:
        """Return the state argument of the most recent ``run`` call."""
        return self.calls[-1][0]

    def run_args(self) -> tuple:
        """Return the positional arguments of the most recent ``run`` call."""
        return self.calls[-1][0]


class _Experiment:
    """Minimal stand-in for an ``ExperimentResult``.

    ``AgentState.successful_experiments`` filters ``experiment_results`` on
    ``status``, so that attribute is all the routing decision needs.
    """

    def __init__(self, status: str = "success") -> None:
        self.status = status
        self.model_name = "Logistic Regression"


def record_successful_evaluation(state: AgentState) -> None:
    """Make a state look as if the Evaluation Agent had succeeded.

    Args:
        state: The state to mutate in place, as a real agent would.
    """
    state.evaluation_status = "completed"
    state.experiment_results = [_Experiment()]


class NodeTestCase(unittest.TestCase):
    """Base class that patches all six agent classes and captures log output."""

    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.multiple(
            nodes,
            **{
                name: mock.DEFAULT
                for name in AGENT_ATTRIBUTES.values()
            },
        )
        self.mocks = patcher.start()
        self.addCleanup(patcher.stop)
        for attribute, mock_class in self.mocks.items():
            mock_class.side_effect = AgentRecorder(attribute)

        self.log_records: List[logging.LogRecord] = []
        handler = self._capture_handler()
        nodes.logger.addHandler(handler)
        self.addCleanup(nodes.logger.removeHandler, handler)
        # ``tests/__init__.py`` pins the log level to WARNING so the suite stays
        # readable; the graph log contract is written at INFO, so raise it here.
        previous_level = nodes.logger.level
        nodes.logger.setLevel(logging.INFO)
        self.addCleanup(nodes.logger.setLevel, previous_level)

    def _capture_handler(self) -> logging.Handler:
        """Return a handler that stores emitted records on the test case."""
        test_case = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                test_case.log_records.append(record)

        return _Handler()

    def use_agent(self, node: GraphNode, **kwargs: Any) -> AgentRecorder:
        """Replace the agent used by one node with a configured stand-in.

        Args:
            node: The node whose agent should be replaced.
            **kwargs: Forwarded to :class:`AgentRecorder`.

        Returns:
            The recorder standing in for the agent class.
        """
        name = AGENT_ATTRIBUTES[node]
        recorder = AgentRecorder(name, **kwargs)
        setattr(nodes, name, recorder)
        return recorder

    def agent_calls(self) -> List[str]:
        """Return the names of every agent class that was instantiated."""
        return sorted(name for name, m in self.mocks.items() if m.called)

    def log_lines(self) -> List[str]:
        """Return the emitted graph log messages."""
        return [r.getMessage() for r in self.log_records]

    def graph_log_lines(self) -> List[str]:
        """Return only the ``[GRAPH]`` lines."""
        return [line for line in self.log_lines() if config.LANGGRAPH_LOG_PREFIX in line]


class TestNodeFunctionsExist(NodeTestCase):
    """Check 6: one node function per agent."""

    def test_there_are_seven_node_functions(self):
        """One per agent. Phase 9 added the seventh, Model Persistence."""
        self.assertEqual(len(NODE_FUNCTIONS), 7)
        self.assertEqual(set(NODE_FUNCTIONS), set(GraphNode))

    def test_node_function_matches_its_node_name(self):
        for node, function in NODE_FUNCTIONS.items():
            with self.subTest(node=node):
                self.assertEqual(function.__name__, f"{node.value}_node")

    def test_required_nodes_have_functions(self):
        for node in REQUIRED_NODES:
            with self.subTest(node=node):
                self.assertIn(node, NODE_FUNCTIONS)
                self.assertTrue(callable(NODE_FUNCTIONS[node]))

    def test_module_level_names_are_importable(self):
        for function in (
            data_analysis_node,
            preprocessing_node,
            model_selection_node,
            training_node,
            evaluation_node,
            optimization_node,
        ):
            self.assertTrue(callable(function))


class TestNodeInvokesExactlyOneAgent(NodeTestCase):
    """Check 7: a node delegates to exactly one existing agent."""

    def _run_node(self, node: GraphNode, state: Optional[GraphState] = None) -> GraphState:
        graph_state = state if state is not None else create_graph_state()
        return NODE_FUNCTIONS[node](graph_state)

    def test_each_node_touches_only_its_own_agent(self):
        for node in GraphNode:
            with self.subTest(node=node):
                self.setUp()
                self._run_node(node)
                self.assertEqual(self.agent_calls(), [AGENT_ATTRIBUTES[node]])

    def test_agent_receives_the_carried_state(self):
        state = create_graph_state()
        self.use_agent(GraphNode.DATA_ANALYSIS)
        update = data_analysis_node(state)
        self.assertIs(update["agent_state"], state["agent_state"])

    def test_data_analysis_passes_the_workflow_inputs(self):
        state = create_graph_state(dataset_path="data/x.csv", target_column="y")
        agent = self.use_agent(GraphNode.DATA_ANALYSIS)
        data_analysis_node(state)
        self.assertEqual(len(agent.calls), 1)
        self.assertIs(agent.last_state, state["agent_state"])
        self.assertEqual(
            agent.last_kwargs,
            {"dataset_path": "data/x.csv", "target_column": "y"},
        )

    def test_training_receives_the_cross_validation_flag(self):
        for flag in (True, False):
            with self.subTest(cross_validation=flag):
                self.setUp()
                state = create_graph_state(cross_validation=flag)
                agent = self.use_agent(GraphNode.TRAINING)
                training_node(state)
                self.assertEqual(agent.last_kwargs["cross_validation"], flag)

    def test_optimization_receives_the_trial_count(self):
        state = create_graph_state(n_trials=7, cross_validation=False)
        agent = self.use_agent(GraphNode.OPTIMIZATION)
        optimization_node(state)
        self.assertEqual(agent.last_kwargs["n_trials"], 7)
        self.assertEqual(agent.last_kwargs["cross_validation"], False)

    def test_agents_without_extra_inputs_are_called_positionally(self):
        for node in (
            GraphNode.PREPROCESSING,
            GraphNode.MODEL_SELECTION,
            GraphNode.EVALUATION,
        ):
            with self.subTest(node=node):
                self.setUp()
                state = create_graph_state()
                agent = self.use_agent(node)
                NODE_FUNCTIONS[node](state)
                self.assertEqual(len(agent.calls), 1)
                self.assertIs(agent.last_state, state["agent_state"])
                self.assertEqual(agent.last_kwargs, {})

    def test_agents_are_instantiated_fresh_per_node(self):
        state = create_graph_state()
        agent = self.use_agent(GraphNode.EVALUATION)
        evaluation_node(state)
        evaluation_node({**state, "agent_state": AgentState()})
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(agent.constructions, 2)
        self.assertIsNot(agent.calls[0][0], agent.calls[1][0])

    def test_invoker_rejects_an_unknown_node(self):
        with self.assertRaises(ValueError):
            build_agent_invoker(create_graph_state(), "not_a_node")


class TestNodeStatusIsRecorded(NodeTestCase):
    """Check 8: the node status is recorded."""

    def test_success_records_completed(self):
        state = create_graph_state()
        update = training_node(state)
        self.assertEqual(update["node_status"][GraphNode.TRAINING.value], "completed")
        self.assertEqual(update["node_status"][GraphNode.EVALUATION.value], "pending")

    def test_failure_records_failed(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("no"))
        state = create_graph_state()
        update = training_node(state)
        self.assertEqual(update["node_status"][GraphNode.TRAINING.value], "failed")

    def test_node_status_is_returned_in_full(self):
        """LangGraph replaces a key unless it has a reducer, so a node must return
        the whole merged mapping rather than only its own entry."""
        state = create_graph_state()
        first = training_node(state)
        merged = {**state, **first}
        second = training_node(merged)
        statuses: Dict[str, str] = second["node_status"]
        self.assertEqual(len(statuses), len(GraphNode))
        self.assertEqual(statuses[GraphNode.DATA_ANALYSIS.value], "pending")

    def test_a_record_is_appended_per_node(self):
        state = create_graph_state()
        update = training_node(state)
        self.assertEqual(len(update["node_records"]), 1)
        record = update["node_records"][0]
        self.assertEqual(record.node, "training")
        self.assertEqual(record.display_name, "Training")
        self.assertEqual(record.status, NodeStatus.COMPLETED)
        self.assertIsNone(record.error_type)

    def test_current_node_is_published(self):
        update = evaluation_node(create_graph_state())
        self.assertEqual(update["current_node"], "evaluation")

    def test_workflow_is_marked_running_while_executing(self):
        update = evaluation_node(create_graph_state())
        self.assertEqual(update["workflow_status"], "running")

    def test_agents_that_record_a_phase_failure_keep_node_completed(self):
        """The Evaluation Agent records its own failure on the state instead of
        raising, so the node itself completed successfully."""
        self.use_agent(
            GraphNode.EVALUATION, mutate=lambda s: setattr(s, "evaluation_status", "failed")
        )
        update = evaluation_node(create_graph_state())
        self.assertEqual(update["node_status"]["evaluation"], "completed")
        self.assertEqual(update["agent_state"].evaluation_status, "failed")

    def test_failed_node_is_recorded_only_while_failing(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("no"))
        state = create_graph_state()
        update = training_node(state)
        self.assertEqual(update["failed_node"], "training")
        self.use_agent(GraphNode.TRAINING)
        later = evaluation_node({**state, **update})
        self.assertEqual(later["failed_node"], "training")


class TestNodeErrorsAreRecorded(NodeTestCase):
    """Check 9: failures become error records, never silent."""

    def test_error_is_recorded_with_type_and_message(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("no models"))
        update = training_node(create_graph_state())
        self.assertEqual(len(update["node_errors"]), 1)
        error = update["node_errors"][0]
        self.assertEqual(error.node, "training")
        self.assertEqual(error.error_type, "TrainingToolError")
        self.assertIn("no models", error.message)

    def test_error_message_is_collapsed_to_one_line(self):
        error = TrainingToolError("line one\nline two\n\nline three")
        self.use_agent(GraphNode.TRAINING, error=error)
        update = training_node(create_graph_state())
        message = update["node_errors"][0].message
        self.assertNotIn("\n", message)
        self.assertEqual(message, "line one line two line three")

    def test_error_retryability_is_recorded(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("x"))
        self.assertFalse(training_node(create_graph_state())["node_errors"][0].retryable)

        self.setUp()
        self.use_agent(GraphNode.TRAINING, error=RuntimeError("x"))
        self.assertTrue(training_node(create_graph_state())["node_errors"][0].retryable)

    def test_last_error_is_published(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("boom"))
        update = training_node(create_graph_state())
        self.assertEqual(update["last_error"].error_type, "TrainingToolError")

    def test_successful_node_records_no_error(self):
        update = training_node(create_graph_state())
        self.assertEqual(update["node_errors"], [])
        self.assertIsNone(update["last_error"])

    def test_node_record_carries_the_error_summary(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("boom"))
        record = training_node(create_graph_state())["node_records"][0]
        self.assertEqual(record.status, NodeStatus.FAILED)
        self.assertEqual(record.error_type, "TrainingToolError")
        self.assertIn("boom", record.error_message)

    def test_a_node_never_raises(self):
        for error in (
            DataToolError("x"),
            DatasetNotFoundError("x"),
            PreprocessingError("x"),
            OptimizationToolError("x"),
            RuntimeError("x"),
            ValueError("x"),
        ):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                self.use_agent(GraphNode.OPTIMIZATION, error=error)
                update = optimization_node(create_graph_state())
                self.assertEqual(update["node_status"]["optimization"], "failed")


class TestFailureStopsTheRun(NodeTestCase):
    """Check 10: a required node failure requests a stop."""

    def test_required_failure_requests_stop(self):
        for node in REQUIRED_NODES:
            with self.subTest(node=node):
                self.setUp()
                self.use_agent(node, error=RuntimeError("boom"))
                update = NODE_FUNCTIONS[node](create_graph_state())
                self.assertTrue(update["stop_requested"])
                self.assertEqual(update["failed_node"], node.value)

    def test_stop_on_failure_can_be_disabled(self):
        self.use_agent(GraphNode.TRAINING, error=RuntimeError("boom"))
        state = create_graph_state(stop_on_failure=False)
        update = training_node(state)
        self.assertFalse(update["stop_requested"])
        self.assertEqual(update["node_status"]["training"], "failed")

    def test_existing_stop_request_is_preserved(self):
        state = create_graph_state()
        state["stop_requested"] = True
        update = preprocessing_node(state)
        self.assertTrue(update["stop_requested"])

    def test_optimization_failure_does_not_stop_the_run(self):
        """Check 11: the Phase 5 result must survive an optimization failure."""
        self.use_agent(GraphNode.OPTIMIZATION, error=RuntimeError("boom"))
        update = optimization_node(create_graph_state())
        self.assertFalse(update["stop_requested"])
        self.assertEqual(update["node_status"]["optimization"], "failed")

    def test_optimization_failure_is_reported_as_failed_outcome(self):
        self.use_agent(GraphNode.OPTIMIZATION, error=RuntimeError("boom"))
        update = optimization_node(create_graph_state())
        self.assertEqual(update["optimization_outcome"], "failed")


class TestRetryPolicy(NodeTestCase):
    """Checks 12 and 13: at most one retry, only for unexpected errors."""

    def test_retryable_failure_is_retried_once(self):
        agent = self.use_agent(GraphNode.TRAINING, error=RuntimeError("transient"))
        update = training_node(create_graph_state())
        self.assertEqual(len(agent.calls), 2)
        record = update["node_records"][0]
        self.assertEqual(record.attempts, 2)
        self.assertEqual(record.retries, 1)
        self.assertEqual(record.status, NodeStatus.FAILED)
        self.assertEqual(update["retries"]["training"], 1)

    def test_retryable_failure_that_succeeds_completes(self):
        class _FlakyRecorder(AgentRecorder):
            """Fails the first call and succeeds afterwards."""

            def run(self, state: AgentState, **kwargs: Any) -> AgentState:
                self.calls.append((state, kwargs))
                if len(self.calls) == 1:
                    raise RuntimeError("transient")
                return state

        setattr(nodes, "TrainingAgent", _FlakyRecorder("TrainingAgent"))
        update = training_node(create_graph_state())
        record = update["node_records"][0]
        self.assertEqual(record.status, NodeStatus.COMPLETED)
        self.assertEqual(record.attempts, 2)
        self.assertEqual(record.retries, 1)
        self.assertEqual(update["node_errors"][0].error_type, "RuntimeError")

    def test_expected_error_is_not_retried(self):
        for error in (
            DataToolError("x"),
            PreprocessingError("x"),
            TrainingToolError("x"),
            OptimizationToolError("x"),
        ):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                agent = self.use_agent(GraphNode.TRAINING, error=error)
                update = training_node(create_graph_state())
                self.assertEqual(len(agent.calls), 1)
                self.assertEqual(update["node_records"][0].attempts, 1)
                self.assertEqual(update["node_records"][0].retries, 0)

    def test_retry_ceiling_is_one_by_default(self):
        self.assertEqual(config.LANGGRAPH_MAX_RETRIES, 1)

    def test_more_than_one_retry_is_not_attempted(self):
        """Even if the budget is raised, a permanently failing node stops."""
        agent = self.use_agent(GraphNode.TRAINING, error=RuntimeError("always"))
        state = create_graph_state(max_retries=3)
        update = training_node(state)
        self.assertEqual(len(agent.calls), 4)
        self.assertEqual(update["node_records"][0].retries, 3)

    def test_a_recovered_node_does_not_publish_a_stale_last_error(self):
        class _FlakyRecorder(AgentRecorder):
            def run(self, state: AgentState, **kwargs: Any) -> AgentState:
                self.calls.append((state, kwargs))
                if len(self.calls) == 1:
                    raise RuntimeError("transient")
                return state

        previous = GraphError("training", "OldError", "old", 1, True)
        state = create_graph_state()
        state["last_error"] = previous
        setattr(nodes, "TrainingAgent", _FlakyRecorder("TrainingAgent"))
        update = training_node(state)
        self.assertIs(update["last_error"], previous)
        self.assertIsNone(update["node_records"][0].error_type)
        self.assertEqual(update["node_status"]["training"], "completed")

    def test_zero_retries_means_one_attempt(self):
        agent = self.use_agent(GraphNode.TRAINING, error=RuntimeError("always"))
        state = create_graph_state(max_retries=0)
        update = training_node(state)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(update["node_records"][0].retries, 0)

    def test_resolve_max_retries_clamps_and_defaults(self):
        self.assertEqual(resolve_max_retries({"max_retries": None}), 1)
        self.assertEqual(resolve_max_retries({"max_retries": 5}), 5)
        self.assertEqual(resolve_max_retries({"max_retries": -3}), 0)
        self.assertEqual(resolve_max_retries({}), 1)
        self.assertEqual(resolve_max_retries({"max_retries": "abc"}), 1)
        self.assertEqual(resolve_max_retries({"max_retries": "2"}), 2)

    def test_malformed_budget_logs_a_warning(self):
        before = len(self.log_records)
        resolve_max_retries({"max_retries": object()})
        warnings = [r for r in self.log_records[before:] if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("max_retries", warnings[0].getMessage())

    def test_every_attempt_is_recorded_as_an_error(self):
        self.use_agent(GraphNode.TRAINING, error=RuntimeError("always"))
        update = training_node(create_graph_state())
        self.assertEqual(len(update["node_errors"]), 2)
        self.assertEqual(
            [e.attempt for e in update["node_errors"]], [1, 2]
        )
        self.assertEqual(
            [e.message for e in update["node_errors"]],
            ["always", "always"],
        )


class TestSingleLineLogging(NodeTestCase):
    """Check 13: concise single-line graph logging."""

    def test_single_line_collapses_whitespace(self):
        self.assertEqual(single_line("a\nb\t c"), "a b c")
        self.assertEqual(single_line("  padded  "), "padded")
        self.assertEqual(single_line(""), "")

    def test_single_line_truncates(self):
        line = single_line("x" * 500, limit=50)
        self.assertEqual(len(line), 50)
        self.assertTrue(line.endswith("..."))

    def test_single_line_handles_non_strings(self):
        self.assertEqual(single_line(ValueError("bad")), "bad")
        self.assertEqual(single_line(42), "42")
        self.assertEqual(single_line(None), "None")

    def test_every_graph_log_line_is_one_line(self):
        self.use_agent(
            GraphNode.OPTIMIZATION,
            error=OptimizationToolError("line one\nline two\nline three"),
        )
        optimization_node(create_graph_state())
        lines = self.graph_log_lines()
        self.assertTrue(lines)
        for line in lines:
            with self.subTest(line=line):
                self.assertNotIn("\n", line)

    def test_log_events_use_the_configured_prefix(self):
        log_event(GraphNode.TRAINING, "STARTED")
        self.assertEqual(
            self.graph_log_lines(),
            [f"{config.LANGGRAPH_LOG_PREFIX} Training -> STARTED"],
        )

    def test_log_event_appends_the_detail_on_the_same_line(self):
        log_event(GraphNode.TRAINING, "COMPLETED", "1.5 s")
        self.assertEqual(
            self.graph_log_lines(),
            [f"{config.LANGGRAPH_LOG_PREFIX} Training -> COMPLETED (1.5 s)"],
        )

    def test_start_and_completion_are_logged(self):
        evaluation_node(create_graph_state())
        lines = self.graph_log_lines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(
            f"{config.LANGGRAPH_LOG_PREFIX} Evaluation -> STARTED"
        ))
        self.assertIn(
            f"{config.LANGGRAPH_LOG_PREFIX} Evaluation -> COMPLETED", lines[1]
        )

    def test_retry_is_logged_with_the_budget(self):
        self.use_agent(GraphNode.TRAINING, error=RuntimeError("always"))
        training_node(create_graph_state())
        retries = [
            line for line in self.graph_log_lines() if "RETRY" in line
        ]
        self.assertEqual(len(retries), 1)
        self.assertIn("RETRY 1/1", retries[0])

    def test_failure_is_logged(self):
        self.use_agent(GraphNode.TRAINING, error=TrainingToolError("no"))
        training_node(create_graph_state())
        failures = [line for line in self.graph_log_lines() if "FAILED" in line]
        self.assertTrue(failures)
        for line in failures:
            self.assertIn("TrainingToolError", line)


class TestOptimizationDecisionIsStoredByTheNode(NodeTestCase):
    """The decision is recorded before the routing edge is taken."""

    def test_successful_evaluation_records_run(self):
        state = create_graph_state()
        self.use_agent(GraphNode.EVALUATION, mutate=record_successful_evaluation)
        update = evaluation_node(state)
        self.assertEqual(
            update["optimization_decision"], OptimizationDecision.RUN.value
        )

    def test_disabled_records_skip_disabled(self):
        update = evaluation_node(create_graph_state(enable_optimization=False))
        self.assertEqual(
            update["optimization_decision"], OptimizationDecision.SKIP_DISABLED.value
        )

    def test_failed_evaluation_records_a_skip(self):
        self.use_agent(
            GraphNode.EVALUATION, mutate=lambda s: setattr(s, "evaluation_status", "failed")
        )
        update = evaluation_node(create_graph_state())
        self.assertEqual(
            update["optimization_decision"],
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE.value,
        )

    def test_other_nodes_do_not_write_a_decision(self):
        for node in (
            GraphNode.DATA_ANALYSIS,
            GraphNode.PREPROCESSING,
            GraphNode.MODEL_SELECTION,
            GraphNode.TRAINING,
            GraphNode.OPTIMIZATION,
        ):
            with self.subTest(node=node):
                self.setUp()
                update = NODE_FUNCTIONS[node](create_graph_state())
                self.assertNotIn("optimization_decision", update)


class TestExecuteNodeIsUsableDirectly(NodeTestCase):
    """``_execute_node`` is the shared runner every node delegates to."""

    def test_invoke_is_called_with_the_carried_state(self):
        seen: List[AgentState] = []

        def invoke(state: AgentState) -> AgentState:
            seen.append(state)
            return state

        state = create_graph_state()
        update = _execute_node(state, GraphNode.TRAINING, invoke)
        self.assertEqual(seen, [state["agent_state"]])
        self.assertEqual(update["current_node"], "training")

    def test_records_and_errors_are_returned_as_new_lists(self):
        """They carry an append reducer, so only the new entries are returned."""
        state = create_graph_state()
        first = _execute_node(state, GraphNode.TRAINING, lambda s: s)
        second = _execute_node(
            {**state, **first}, GraphNode.EVALUATION, lambda s: s
        )
        self.assertEqual(len(first["node_records"]), 1)
        self.assertEqual(len(second["node_records"]), 1)
        self.assertEqual(second["node_records"][0].node, "evaluation")

    def test_retries_mapping_is_returned_in_full(self):
        state = create_graph_state()
        first = _execute_node(state, GraphNode.TRAINING, lambda s: s)
        merged = {**state, **first}
        self.assertEqual(merged["retries"], {"training": 0})
        second = _execute_node(merged, GraphNode.EVALUATION, lambda s: s)
        self.assertEqual(second["retries"], {"training": 0, "evaluation": 0})

    def test_the_runner_does_not_inspect_the_agent_return_value(self):
        """The runner trusts the agent and only records whether it raised, so a
        falsy return is passed through unchanged rather than second-guessed."""

        def invoke(state: AgentState) -> AgentState:
            return None

        update = _execute_node(create_graph_state(), GraphNode.TRAINING, invoke)
        self.assertEqual(update["node_status"]["training"], "completed")
        self.assertIsNone(update["agent_state"])


if __name__ == "__main__":
    unittest.main()
