"""Tests for graph/routing.py — conditional routing logic."""

import pytest

from graph.routing import is_terminal_failure, route_after_evaluation
from graph.state import make_graph_state
from utils.config import Config
from utils.state import AgentState

BINARY_CSV = str(Config.DATASETS_DIR / "sample_classification.csv")


def _gs_with_evaluations(
    ranked: bool = True,
    failed_node: str = None,
    enable_opt: bool = True,
    opt_done: bool = False,
) -> dict:
    """Build a minimal graph state for routing tests."""
    state = AgentState(dataset_path=BINARY_CSV)
    if ranked:
        state.model_evaluations = [
            {"model_name": "random_forest", "rank": 1, "f1_score": 0.85},
            {"model_name": "logistic_regression", "rank": 2, "f1_score": 0.80},
        ]
    else:
        state.model_evaluations = []
    gs = make_graph_state(state, enable_optimization=enable_opt)
    gs["failed_node"] = failed_node
    gs["optimization_done"] = opt_done
    return gs


# ---------------------------------------------------------------------------
# 14. Evaluation success → optimization
# ---------------------------------------------------------------------------

class TestRouteAfterEvaluation:
    def test_success_with_ranked_models_routes_to_optimize(self):
        gs = _gs_with_evaluations()
        result = route_after_evaluation(gs)
        assert result == "optimize"

    def test_disabled_optimization_routes_to_end(self):
        gs = _gs_with_evaluations(enable_opt=False)
        result = route_after_evaluation(gs)
        assert result == "__end__"

    def test_failed_node_routes_to_end(self):
        gs = _gs_with_evaluations(failed_node="evaluation")
        result = route_after_evaluation(gs)
        assert result == "__end__"

    def test_no_ranked_candidates_routes_to_end(self):
        gs = _gs_with_evaluations(ranked=False)
        result = route_after_evaluation(gs)
        assert result == "__end__"

    def test_optimization_already_done_routes_to_end(self):
        gs = _gs_with_evaluations(opt_done=True)
        result = route_after_evaluation(gs)
        assert result == "__end__"

    def test_routing_is_deterministic(self):
        gs = _gs_with_evaluations()
        r1 = route_after_evaluation(gs)
        r2 = route_after_evaluation(gs)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 15. Terminal failure detection
# ---------------------------------------------------------------------------

class TestIsTerminalFailure:
    def test_no_failure_returns_false(self):
        gs = _gs_with_evaluations()
        assert is_terminal_failure(gs) is False

    def test_failed_node_returns_true(self):
        gs = _gs_with_evaluations(failed_node="training")
        assert is_terminal_failure(gs) is True

    def test_workflow_complete_returns_true(self):
        gs = _gs_with_evaluations()
        gs["workflow_complete"] = True
        assert is_terminal_failure(gs) is True


# ---------------------------------------------------------------------------
# 16. Optimization disabled → END
# ---------------------------------------------------------------------------

class TestOptimizationDisabledRouting:
    def test_enable_false_always_routes_to_end(self):
        gs = _gs_with_evaluations(enable_opt=False)
        assert route_after_evaluation(gs) == "__end__"

    def test_enable_false_even_with_ranked_models(self):
        state = AgentState(dataset_path=BINARY_CSV)
        state.model_evaluations = [{"model_name": "rf", "rank": 1}]
        gs = make_graph_state(state, enable_optimization=False)
        assert route_after_evaluation(gs) == "__end__"


# ---------------------------------------------------------------------------
# 17. No candidates → END
# ---------------------------------------------------------------------------

class TestNoCandidatesRouting:
    def test_empty_evaluations_routes_to_end(self):
        state = AgentState(dataset_path=BINARY_CSV)
        state.model_evaluations = []
        gs = make_graph_state(state)
        assert route_after_evaluation(gs) == "__end__"

    def test_only_failed_experiments_routes_to_end(self):
        state = AgentState(dataset_path=BINARY_CSV)
        state.model_evaluations = [
            {"model_name": "rf", "rank": None, "f1_score": 0.0}
        ]
        gs = make_graph_state(state)
        assert route_after_evaluation(gs) == "__end__"


# ---------------------------------------------------------------------------
# 18. Optimization already completed → END
# ---------------------------------------------------------------------------

class TestOptimizationAlreadyDone:
    def test_optimization_done_true_routes_to_end(self):
        gs = _gs_with_evaluations(opt_done=True)
        assert route_after_evaluation(gs) == "__end__"

    def test_second_call_with_done_flag_routes_to_end(self):
        gs = _gs_with_evaluations()
        # Simulate post-optimization state
        gs["optimization_done"] = True
        result = route_after_evaluation(gs)
        assert result == "__end__"


# ---------------------------------------------------------------------------
# 19. Deterministic routing
# ---------------------------------------------------------------------------

class TestDeterministicRouting:
    def test_same_state_same_route_optimize(self):
        gs = _gs_with_evaluations()
        assert route_after_evaluation(gs) == route_after_evaluation(gs)

    def test_same_state_same_route_end(self):
        gs = _gs_with_evaluations(enable_opt=False)
        assert route_after_evaluation(gs) == route_after_evaluation(gs)
