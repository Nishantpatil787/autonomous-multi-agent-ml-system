"""Conditional routing functions for Phase-7 LangGraph orchestration.

All routing decisions are deterministic — no LLM is used.
Decisions are based solely on measurable state fields.
"""

from typing import Any, Dict, Literal

from utils.logger import get_logger
from utils.state import WorkflowStatus

logger = get_logger("GraphRouting")

# Routing outcome type aliases
_OPTIMIZE = "optimize"
_END = "__end__"
_FAILED = "failed_end"

# ---------------------------------------------------------------------------
# Post-evaluation router
# ---------------------------------------------------------------------------

def route_after_evaluation(
    graph_state: Dict[str, Any],
) -> Literal["optimize", "__end__"]:
    """Decide whether to execute the OptimizationAgent after Evaluation.

    Optimization is SKIPPED when ANY of the following conditions hold:

    1. Evaluation failed (failed_node == 'evaluation').
    2. enable_optimization is False.
    3. optimization_done is already True (prevents double-run).
    4. model_evaluations is empty (no ranked candidates available).

    Otherwise optimization runs.

    Args:
        graph_state: Current LangGraph state dict.

    Returns:
        ``"optimize"`` to proceed to OptimizationAgent,
        ``"__end__"`` to finish the workflow immediately.
    """
    # 1. Fatal failure earlier in the graph
    if graph_state.get("failed_node") is not None:
        logger.info("[GRAPH] Routing: FAILED node detected -> END")
        return _END

    # 2. Optimization explicitly disabled
    if not graph_state.get("enable_optimization", True):
        logger.info("[GRAPH] Routing: optimization disabled -> END")
        return _END

    # 3. Optimization already completed (prevents double-run / infinite loop)
    if graph_state.get("optimization_done", False):
        logger.info("[GRAPH] Routing: optimization already completed -> END")
        return _END

    # 4. No ranked evaluation candidates
    agent_state_dict = graph_state.get("agent_state_dict", {})
    model_evaluations = agent_state_dict.get("model_evaluations", [])
    ranked = [m for m in model_evaluations if m.get("rank") is not None]
    if not ranked:
        logger.info("[GRAPH] Routing: no ranked candidates -> END (skip optimization)")
        return _END

    logger.info("[GRAPH] Routing: evaluation succeeded -> OPTIMIZE")
    return _OPTIMIZE


def route_after_node(
    graph_state: Dict[str, Any],
    next_node: str,
) -> str:
    """Route to next_node if no failure has occurred, else route to END."""
    if graph_state.get("failed_node") is not None or graph_state.get("workflow_complete", False):
        logger.info(f"[GRAPH] Routing: failure detected at '{graph_state.get('failed_node')}' -> END")
        return _END
    return next_node


def is_terminal_failure(graph_state: Dict[str, Any]) -> bool:
    """Return True if the graph is in a terminal failure state."""
    return (
        graph_state.get("failed_node") is not None
        or graph_state.get("workflow_complete", False) is True
    )
