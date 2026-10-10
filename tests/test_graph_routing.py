"""Tests for the graph routing layer (Phase 7, checks 14-19).

Covers the conditional optimization decision, the "already optimized" guard that
prevents a second Optuna run, and the determinism of every router.

The failure-routing and retry checks (20-29) also live here because both are
expressed through these routers; ``tests/test_workflow.py`` proves them end to end
through a real graph run.
"""

from __future__ import annotations

import unittest
from graph.routing import (
    REQUIRED_NODE_VALUES,
    build_workflow_result,
    decide_optimization,
    derive_workflow_status,
    describe_retry_policy,
    describe_routing_rule,
    evaluation_succeeded,
    has_optimization_candidates,
    is_retryable,
    make_step_router,
    non_retryable_error_types,
    optimization_was_performed,
    recorded_optimization_decision,
    route_after_evaluation,
    route_to_end,
)
from graph.state import (
    GraphNode,
    GraphState,
    NodeStatus,
    OptimizationDecision,
    OptimizationOutcome,
    WorkflowStatus,
    create_graph_state,
)
from langgraph.graph import END
from tools.data_tools import DataToolError, DatasetNotFoundError
from tools.evaluation_tools import EvaluationToolError
from tools.model_tools import ModelToolError
from tools.optimization_tools import OptimizationToolError
from tools.preprocessing_tools import AnalysisNotReadyError, PreprocessingError
from tools.training_tools import TrainingToolError
from utils.state import AgentState


class _Experiment:
    """Minimal stand-in for an ``ExperimentResult``.

    ``AgentState.successful_experiments`` is a read-only property that filters
    ``experiment_results`` on ``status``, so only that attribute is needed to
    exercise the routing rules.
    """

    def __init__(self, status: str = "success", model_name: str = "Logistic Regression"):
        self.status = status
        self.model_name = model_name
        self.f1_score = 0.9
        self.accuracy = 0.9


def ready_state(**overrides) -> AgentState:
    """Return an AgentState that looks like a completed evaluation.

    Args:
        **overrides: Field values applied after the defaults.

    Returns:
        An AgentState with one successful experiment and a completed evaluation.
    """
    state = AgentState()
    state.evaluation_status = "completed"
    state.experiment_results = [_Experiment()]
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _state_with_eval(value: str) -> AgentState:
    """Return an AgentState whose evaluation status is ``value``."""
    state = AgentState()
    state.evaluation_status = value
    return state


class TestEvaluationToOptimization(unittest.TestCase):
    """Checks 14 and 19: evaluation success routes to optimization."""

    def test_successful_evaluation_with_candidates_runs_optimization(self):
        state = create_graph_state(agent_state=ready_state())
        self.assertEqual(decide_optimization(state), OptimizationDecision.RUN)
        self.assertTrue(decide_optimization(state).should_run)

    def test_route_after_evaluation_goes_to_optimization(self):
        state = create_graph_state(agent_state=ready_state())
        self.assertEqual(
            route_after_evaluation(state), GraphNode.OPTIMIZATION.value
        )

    def test_route_after_evaluation_prefers_end_when_stopping(self):
        state = create_graph_state(agent_state=ready_state())
        state["stop_requested"] = True
        self.assertEqual(route_after_evaluation(state), END)

    def test_evaluation_succeeded_reads_the_status_string(self):
        for value, expected in (
            ("completed", True),
            ("partial", False),
            ("failed", False),
            ("skipped", False),
            ("not_started", False),
            ("", False),
        ):
            with self.subTest(status=value):
                state = AgentState()
                state.evaluation_status = value
                self.assertEqual(evaluation_succeeded(state), expected)

    def test_successful_experiments_is_derived_not_assigned(self):
        """The candidate check must read the derived property."""
        state = ready_state()
        self.assertEqual(len(state.successful_experiments), 1)
        self.assertTrue(has_optimization_candidates(state))
        state.experiment_results = [_Experiment(status="failed")]
        self.assertEqual(state.successful_experiments, [])
        self.assertFalse(has_optimization_candidates(state))

    def test_partial_experiment_counts_as_a_candidate(self):
        """A partial experiment still carries usable metrics."""
        state = ready_state(experiment_results=[_Experiment(status="partial")])
        self.assertTrue(has_optimization_candidates(state))


class TestEvaluationFailureRoutesToEnd(unittest.TestCase):
    """Check 15: a failed evaluation never optimizes."""

    def test_failed_evaluation_skips(self):
        state = create_graph_state(agent_state=_state_with_eval("failed"))
        self.assertEqual(
            decide_optimization(state),
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE,
        )

    def test_failed_evaluation_routes_to_persistence(self):
        state = create_graph_state(agent_state=_state_with_eval("failed"))
        self.assertEqual(
            route_after_evaluation(state), GraphNode.MODEL_PERSISTENCE.value
        )

    def test_failed_evaluation_still_ends_when_stop_was_requested(self):
        """A stop request is the one case that still ends after Evaluation.

        Phase 9 rerouted the skip branches from END to Model Persistence, because a
        run that writes no artifact is not a useful run. That must not override an
        explicit stop, so ``stop_requested`` keeps its END destination.
        """
        state = create_graph_state(agent_state=_state_with_eval("failed"))
        state["stop_requested"] = True
        self.assertEqual(route_after_evaluation(state), END)

    def test_partial_evaluation_skips(self):
        state = create_graph_state(agent_state=_state_with_eval("partial"))
        self.assertEqual(
            decide_optimization(state),
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE,
        )

    def test_evaluation_not_started_skips(self):
        state = create_graph_state(agent_state=AgentState())
        self.assertEqual(
            decide_optimization(state),
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE,
        )


class TestOptimizationDisabledRoutesToEnd(unittest.TestCase):
    """Check 16: the configuration switch."""

    def test_disabled_skips_even_with_a_perfect_state(self):
        state = create_graph_state(agent_state=ready_state(), enable_optimization=False)
        self.assertEqual(
            decide_optimization(state), OptimizationDecision.SKIP_DISABLED
        )

    def test_disabled_routes_to_persistence(self):
        """A ``--no-optimize`` run still has to write the best baseline."""
        state = create_graph_state(agent_state=ready_state(), enable_optimization=False)
        self.assertEqual(
            route_after_evaluation(state), GraphNode.MODEL_PERSISTENCE.value
        )

    def test_disabled_is_checked_before_evaluation_state(self):
        """Ordering matters: the recorded reason must be the first that applied."""
        state = create_graph_state(
            agent_state=_state_with_eval("failed"), enable_optimization=False
        )
        self.assertEqual(
            decide_optimization(state), OptimizationDecision.SKIP_DISABLED
        )


class TestNoCandidatesRoutesToEnd(unittest.TestCase):
    """Check 17: nothing to tune means nothing to do."""

    def test_no_successful_experiments_skips(self):
        state = AgentState()
        state.evaluation_status = "completed"
        state.experiment_results = [_Experiment(status="failed")]
        graph_state = create_graph_state(agent_state=state)
        self.assertEqual(
            decide_optimization(graph_state), OptimizationDecision.SKIP_NO_CANDIDATES
        )

    def test_no_candidates_routes_to_persistence(self):
        """Nothing to search is not a reason to skip persistence.

        With no successful experiment the agent records the failure rather than
        writing a meaningless artifact; the node still runs and reports why.
        """
        state = AgentState()
        state.evaluation_status = "completed"
        state.experiment_results = []
        graph_state = create_graph_state(agent_state=state)
        self.assertEqual(
            route_after_evaluation(graph_state), GraphNode.MODEL_PERSISTENCE.value
        )

    def test_candidate_check_reads_successful_experiments_not_training_status(self):
        """``training_status`` is ``completed`` even when every experiment failed,
        so the check must use the experiments themselves."""
        state = AgentState()
        state.evaluation_status = "completed"
        state.training_status = "completed"
        state.experiment_results = [_Experiment(status="failed")]
        self.assertFalse(has_optimization_candidates(state))
        self.assertEqual(
            decide_optimization(create_graph_state(agent_state=state)),
            OptimizationDecision.SKIP_NO_CANDIDATES,
        )


class TestAlreadyOptimizedRoutesToEnd(unittest.TestCase):
    """Check 18: the graph must never start a second Optuna run."""

    def test_completed_optimization_blocks_a_second_run(self):
        state = create_graph_state(agent_state=ready_state(optimization_status="completed"))
        self.assertEqual(
            decide_optimization(state), OptimizationDecision.SKIP_ALREADY_DONE
        )

    def test_partial_optimization_blocks_a_second_run(self):
        state = create_graph_state(agent_state=ready_state(optimization_status="partial"))
        self.assertEqual(
            decide_optimization(state), OptimizationDecision.SKIP_ALREADY_DONE
        )

    def test_failed_optimization_blocks_a_second_run(self):
        state = create_graph_state(agent_state=ready_state(optimization_status="failed"))
        self.assertEqual(
            decide_optimization(state), OptimizationDecision.SKIP_ALREADY_DONE
        )

    def test_recorded_skip_counts_as_already_handled(self):
        state = create_graph_state(agent_state=ready_state(optimization_status="skipped"))
        self.assertEqual(
            decide_optimization(state), OptimizationDecision.SKIP_ALREADY_DONE
        )

    def test_not_started_is_the_only_value_that_allows_a_run(self):
        state = ready_state(optimization_status="not_started")
        self.assertFalse(optimization_was_performed(state))
        self.assertEqual(
            decide_optimization(create_graph_state(agent_state=state)),
            OptimizationDecision.RUN,
        )

    def test_already_optimized_routes_to_persistence(self):
        state = create_graph_state(agent_state=ready_state(optimization_status="completed"))
        self.assertEqual(
            route_after_evaluation(state), GraphNode.MODEL_PERSISTENCE.value
        )


class TestDeterminism(unittest.TestCase):
    """Check 19: the same state always routes the same way."""

    def test_decision_is_stable_across_repeated_calls(self):
        graph_state = create_graph_state(agent_state=ready_state())
        results = {decide_optimization(graph_state) for _ in range(50)}
        self.assertEqual(results, {OptimizationDecision.RUN})

    def test_decision_is_stable_across_identical_but_distinct_states(self):
        first = create_graph_state(agent_state=ready_state())
        second = create_graph_state(agent_state=ready_state())
        self.assertEqual(decide_optimization(first), decide_optimization(second))

    def test_decision_does_not_mutate_the_state(self):
        graph_state = create_graph_state(agent_state=ready_state())
        before = dict(graph_state)
        decide_optimization(graph_state)
        self.assertEqual(dict(graph_state), before)

    def test_every_decision_has_a_non_empty_reason(self):
        for decision in OptimizationDecision:
            with self.subTest(decision=decision):
                self.assertTrue(decision.reason.strip())
                self.assertNotIn("{", decision.reason)

    def test_step_routers_are_deterministic(self):
        router = make_step_router(GraphNode.TRAINING)
        ok_state: GraphState = {"stop_requested": False}
        stop_state: GraphState = {"stop_requested": True}
        self.assertEqual(
            [router(ok_state) for _ in range(20)],
            [GraphNode.TRAINING.value] * 20,
        )
        self.assertEqual([router(stop_state) for _ in range(20)], [END] * 20)

    def test_route_to_end_always_ends(self):
        self.assertEqual(route_to_end({}), END)
        self.assertEqual(route_to_end({"stop_requested": False}), END)


class TestRetryability(unittest.TestCase):
    """Checks 26-29 at the classification level: what may be retried."""

    def test_project_errors_are_not_retryable(self):
        for error in (
            DataToolError("x"),
            DatasetNotFoundError("x"),
            PreprocessingError("x"),
            AnalysisNotReadyError("x"),
            ModelToolError("x"),
            TrainingToolError("x"),
            EvaluationToolError("x"),
            OptimizationToolError("x"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_retryable(error))

    def test_unexpected_errors_are_retryable(self):
        for error in (RuntimeError("x"), MemoryError("x"), OSError("x"), ValueError("x")):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(is_retryable(error))

    def test_subclasses_of_project_errors_are_not_retryable(self):
        class CustomDataError(DataToolError):
            pass

        self.assertFalse(is_retryable(CustomDataError("x")))

    def test_non_retryable_tuple_is_stable(self):
        types = non_retryable_error_types()
        self.assertEqual(types, non_retryable_error_types())
        for expected in (
            DataToolError,
            PreprocessingError,
            ModelToolError,
            TrainingToolError,
            EvaluationToolError,
            OptimizationToolError,
        ):
            self.assertIn(expected, types)

    def test_retry_rule_text_names_the_limit(self):
        self.assertIn("1", describe_retry_policy())


class TestWorkflowStatusDerivation(unittest.TestCase):
    """How the overall status is derived from node outcomes."""

    def _state_with(self, **node_status) -> GraphState:
        state = create_graph_state()
        merged = dict(state["node_status"])
        merged.update(node_status)
        state["node_status"] = merged
        return state

    def test_all_required_nodes_fine_is_completed(self):
        state = self._state_with(
            **{
                node: NodeStatus.COMPLETED.value
                for node in REQUIRED_NODE_VALUES
            }
        )
        self.assertEqual(derive_workflow_status(state), WorkflowStatus.COMPLETED)

    def test_failed_required_node_is_failed(self):
        state = self._state_with(training=NodeStatus.FAILED.value)
        self.assertEqual(derive_workflow_status(state), WorkflowStatus.FAILED)

    def test_failed_evaluation_is_failed(self):
        state = self._state_with(evaluation=NodeStatus.FAILED.value)
        self.assertEqual(derive_workflow_status(state), WorkflowStatus.FAILED)

    def test_failed_optimization_is_partial(self):
        """The Phase 5 result survives an optimization failure."""
        state = self._state_with(
            **{node: NodeStatus.COMPLETED.value for node in REQUIRED_NODE_VALUES},
            optimization=NodeStatus.FAILED.value,
        )
        self.assertEqual(derive_workflow_status(state), WorkflowStatus.PARTIAL)
        self.assertTrue(WorkflowStatus.PARTIAL.is_success)

    def test_stop_requested_without_a_failed_node_is_failed(self):
        state = self._state_with(
            **{node: NodeStatus.COMPLETED.value for node in REQUIRED_NODE_VALUES}
        )
        state["stop_requested"] = True
        self.assertEqual(derive_workflow_status(state), WorkflowStatus.FAILED)

    def test_step_ceiling_is_failed(self):
        state = self._state_with()
        state["step_limit_reached"] = True
        self.assertEqual(derive_workflow_status(state), WorkflowStatus.FAILED)


class TestRecordedDecision(unittest.TestCase):
    """The decision reported must be the one that was acted on."""

    def test_stored_decision_wins_over_recomputation(self):
        """After a successful run the state satisfies "already done", so a naive
        recomputation would wrongly report skip_already_done."""
        state = create_graph_state(
            agent_state=ready_state(optimization_status="completed")
        )
        state["optimization_decision"] = OptimizationDecision.RUN.value
        result = build_workflow_result(state)
        self.assertEqual(result.optimization_decision, OptimizationDecision.RUN)

    def test_missing_stored_decision_is_recomputed(self):
        state = create_graph_state(agent_state=ready_state())
        state["optimization_decision"] = ""
        self.assertEqual(
            recorded_optimization_decision(state), OptimizationDecision.RUN
        )

    def test_unrecognised_stored_decision_falls_back(self):
        state = create_graph_state(agent_state=ready_state())
        state["optimization_decision"] = "not_a_real_decision"
        self.assertEqual(
            recorded_optimization_decision(state), OptimizationDecision.RUN
        )


class TestResultDerivation(unittest.TestCase):
    """Check: the counters in the result are summed from the recorded state."""

    def test_totals_are_summed_from_the_state(self):
        state = create_graph_state(agent_state=ready_state())
        state["retries"] = {"training": 1, "evaluation": 0}
        state["node_status"] = dict(state["node_status"])
        state["node_status"]["training"] = NodeStatus.FAILED.value
        state["failed_node"] = "training"
        result = build_workflow_result(state)
        self.assertEqual(result.total_retries, 1)
        self.assertEqual(result.failed_node, "training")
        self.assertEqual(result.workflow_status, WorkflowStatus.FAILED)

    def test_optimization_outcome_reads_the_agent_status(self):
        state = create_graph_state(
            agent_state=ready_state(optimization_status="completed")
        )
        state["node_status"] = dict(state["node_status"])
        state["node_status"]["optimization"] = NodeStatus.COMPLETED.value
        self.assertEqual(
            build_workflow_result(state).optimization_outcome,
            OptimizationOutcome.COMPLETED,
        )

    def test_not_run_outcome_when_disabled(self):
        state = create_graph_state(agent_state=ready_state(), enable_optimization=False)
        result = build_workflow_result(state)
        self.assertEqual(result.optimization_outcome, OptimizationOutcome.NOT_RUN)

    def test_skipped_outcome_when_decided_against(self):
        """A decision other than skip_disabled that was never acted on is 'skipped'.

        The evaluation node stores the decision before the routing edge is taken,
        so a faithful test has to store it too.
        """
        state = create_graph_state(agent_state=_state_with_eval("failed"))
        state["optimization_decision"] = (
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE.value
        )
        result = build_workflow_result(state)
        self.assertEqual(
            result.optimization_decision,
            OptimizationDecision.SKIP_EVALUATION_INCOMPLETE,
        )
        self.assertEqual(result.optimization_outcome, OptimizationOutcome.SKIPPED)

    def test_outcome_distinguishes_never_entered_from_disabled(self):
        """Both leave the node pending, but they mean different things."""
        for decision, expected in (
            (OptimizationDecision.SKIP_DISABLED, OptimizationOutcome.NOT_RUN),
            (OptimizationDecision.SKIP_NO_CANDIDATES, OptimizationOutcome.SKIPPED),
            (OptimizationDecision.SKIP_ALREADY_DONE, OptimizationOutcome.SKIPPED),
        ):
            with self.subTest(decision=decision):
                state = create_graph_state(agent_state=ready_state())
                state["optimization_decision"] = decision.value
                self.assertEqual(
                    build_workflow_result(state).optimization_outcome, expected
                )


class TestRuleDescriptions(unittest.TestCase):
    """The rule text reported to the user."""

    def test_routing_rule_mentions_determinism_and_no_llm(self):
        text = describe_routing_rule().lower()
        self.assertIn("deterministic", text)
        self.assertIn("language model", text)

    def test_routing_rule_has_no_unformatted_placeholder(self):
        self.assertNotIn("{", describe_routing_rule())


if __name__ == "__main__":
    unittest.main()
