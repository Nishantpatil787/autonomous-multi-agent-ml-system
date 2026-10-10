"""Optional LangGraph nodes for the local LLM layer (Phase 8).

Three nodes, each of which calls exactly one :class:`~llm.service.LLMService`
method and appends one :class:`~llm.state.LLMResult`. They are placed *after* the
deterministic step whose results they explain, never before it:

    data_analysis -> llm_data_analysis -> preprocessing
    evaluation    -> llm_evaluation    -> optimization decision
    optimization  -> llm_optimization  -> END

The rules these nodes obey are the whole point of the phase, so they are worth
stating plainly:

* **They never raise.** A node cannot fail the graph, and none of them can set
  ``stop_requested``. A missing Ollama, an uninstalled model, a timeout or
  malformed JSON all end as a recorded status.
* **They never mutate the ``AgentState``.** No field of the authoritative machine
  learning state is read-modified-written here, so generated text cannot change a
  metric, a model or a report. The only state key they touch is the append-only
  ``llm_results`` list.
* **They never route.** Every one of them is followed by an ordinary conditional
  edge whose router reads only recorded state, exactly as in Phase 7. There is no
  code path from a model response to a routing decision.
* **They run last.** Because each node follows its deterministic step, the LLM
  cannot influence which node runs next, only narrate what already happened.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from graph.state import (
    LLMGraphNode,
    GraphState,
    NodeRecord,
    NodeStatus,
)
from llm.service import LLMService
from llm.state import LLMResult, LLMRole, LLMStatus
from utils import config
from utils.logger import setup_logger
from utils.state import AgentState

logger = setup_logger("graph.llm_nodes")

#: Signature of one interpretation call: take the machine learning state, return the
#: validated result. Mirrors the ``AgentInvoker`` signature in
#: :mod:`graph.nodes`, which keeps the two node families symmetrical.
LLMInvoker = Callable[[AgentState], LLMResult]


def build_llm_invoker(
    state: GraphState, node: LLMGraphNode
) -> LLMInvoker:
    """Return the callable that runs one interpretation step.

    The service is read from the graph state rather than imported at module level,
    so a test can inject a stub and no node ever constructs a client itself. That
    indirection is what guarantees the nodes cannot reach the network by accident:
    they only call whatever object was handed to them.

    Args:
        state: The graph state holding the workflow inputs.
        node: The interpretation node.

    Returns:
        A callable that runs the step and returns the result.

    Raises:
        ValueError: If ``node`` is not a known LLM node.
    """
    service: Optional[LLMService] = state.get("llm_service")

    if node is LLMGraphNode.DATA_ANALYSIS:

        def run_data_analysis(agent_state: AgentState) -> LLMResult:
            return service.interpret_data_analysis(agent_state)

        return run_data_analysis

    if node is LLMGraphNode.EVALUATION:

        def run_evaluation(agent_state: AgentState) -> LLMResult:
            return service.interpret_evaluation(agent_state)

        return run_evaluation

    if node is LLMGraphNode.OPTIMIZATION:

        def run_optimization(agent_state: AgentState) -> LLMResult:
            return service.plan_optimization(agent_state)

        return run_optimization

    raise ValueError(f"Unknown LLM graph node: {node!r}")


def _node_status(result: LLMResult) -> NodeStatus:
    """Map an LLM result onto the node vocabulary.

    An interpretation step that could not produce text is ``COMPLETED`` as a *node*:
    the node ran, handled the situation and returned cleanly, which is exactly the
    Phase 8 contract. Reporting it as ``failed`` would let an optional layer break a
    workflow that actually succeeded, so the real outcome lives in the appended
    :class:`~llm.state.LLMResult` instead.

    Args:
        result: The interpretation result.

    Returns:
        Always :attr:`~graph.state.NodeStatus.COMPLETED`.
    """
    return NodeStatus.COMPLETED


def _detail(result: LLMResult) -> str:
    """Return a bounded one-line summary of an interpretation result."""
    if result.status is LLMStatus.COMPLETED:
        text = f"{result.status.value}, model={result.model or 'unknown'}"
        claims = result.forbidden_claims
        if claims:
            text += f", {len(claims)} phrasing warning(s)"
        return text
    if result.status is LLMStatus.DISABLED:
        return f"{result.status.value} (no request was made)"
    return f"{result.status.value} ({result.error_category.value})"


def log_llm_event(node: LLMGraphNode, result: LLMResult) -> None:
    """Emit one single-line log entry for an interpretation node.

    Args:
        node: The node reporting the event.
        result: The interpretation result.
    """
    logger.info(
        "%s %s -> %s (%s)",
        config.LANGGRAPH_LOG_PREFIX,
        node.display_name,
        result.status.value.upper(),
        _detail(result),
    )


def execute_llm_node(
    state: GraphState,
    node: LLMGraphNode,
    invoke: LLMInvoker,
) -> GraphState:
    """Run one interpretation step and record it. Never raises.

    Args:
        state: The incoming graph state.
        node: The interpretation node being executed.
        invoke: Callable that runs the step and returns the result.

    Returns:
        The partial state update: the appended result and the node record. Nothing
        else is touched, so an interpretation step can never alter the machine
        learning state or a routing field.
    """
    started = time.perf_counter()
    agent_state: AgentState = state["agent_state"]

    try:
        result = invoke(agent_state)
    except Exception as error:  # noqa: BLE001 - an optional layer cannot fail a run
        # The service is written not to raise, so reaching this means a bug in the
        # service or in the injected object. It is still converted into a recorded
        # status rather than propagated, because the workflow must continue.
        logger.error(
            "%s %s -> FAILED (%s: %s)",
            config.LANGGRAPH_LOG_PREFIX,
            node.display_name,
            type(error).__name__,
            " ".join(str(error).split()),
        )
        result = _unexpected_failure(node, error)

    log_llm_event(node, result)

    record = NodeRecord(
        node=node.value,
        display_name=node.display_name,
        status=_node_status(result),
        attempts=1,
        retries=0,
        duration=time.perf_counter() - started,
        error_type=None,
        error_message=None,
    )

    return {
        "llm_results": [result],
        "node_records": [record],
        "current_node": node.value,
    }


def _unexpected_failure(node: LLMGraphNode, error: BaseException) -> LLMResult:
    """Build a result for an exception that escaped the service.

    Args:
        node: The node that raised.
        error: The escaped exception.

    Returns:
        A failed :class:`~llm.state.LLMResult` carrying the reason.
    """
    from llm.state import LLMErrorCategory

    return LLMResult(
        role=node.role,
        status=LLMStatus.REQUEST_FAILED,
        detail=(
            f"Unexpected failure inside the interpretation layer: "
            f"{type(error).__name__}: {' '.join(str(error).split())}"
        ),
        error_category=LLMErrorCategory.CONNECTION,
    )


# --------------------------------------------------------------------------- #
# The three nodes
# --------------------------------------------------------------------------- #


def llm_data_analysis_node(state: GraphState) -> GraphState:
    """Interpret the Data Analysis Agent result.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return execute_llm_node(
        state,
        LLMGraphNode.DATA_ANALYSIS,
        build_llm_invoker(state, LLMGraphNode.DATA_ANALYSIS),
    )


def llm_evaluation_node(state: GraphState) -> GraphState:
    """Interpret the evaluation result and the recorded ranking.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return execute_llm_node(
        state,
        LLMGraphNode.EVALUATION,
        build_llm_invoker(state, LLMGraphNode.EVALUATION),
    )


def llm_optimization_node(state: GraphState) -> GraphState:
    """Propose a high-level optimization plan from the recorded search result.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return execute_llm_node(
        state,
        LLMGraphNode.OPTIMIZATION,
        build_llm_invoker(state, LLMGraphNode.OPTIMIZATION),
    )


#: Node to callable mapping, used when building the LLM-enabled graph.
LLM_NODE_FUNCTIONS: Dict[LLMGraphNode, Callable[[GraphState], GraphState]] = {
    LLMGraphNode.DATA_ANALYSIS: llm_data_analysis_node,
    LLMGraphNode.EVALUATION: llm_evaluation_node,
    LLMGraphNode.OPTIMIZATION: llm_optimization_node,
}


def describe_llm_nodes() -> str:
    """Return a one-line description of the optional nodes.

    Returns:
        A single line naming each node and the service method it calls.
    """
    return "; ".join(
        f"{node.value} -> {method}"
        for node, method in (
            (LLMGraphNode.DATA_ANALYSIS, "interpret_data_analysis"),
            (LLMGraphNode.EVALUATION, "interpret_evaluation"),
            (LLMGraphNode.OPTIMIZATION, "plan_optimization"),
        )
    )


__all__ = [
    "LLMInvoker",
    "LLM_NODE_FUNCTIONS",
    "LLMGraphNode",
    "LLMRole",
    "build_llm_invoker",
    "describe_llm_nodes",
    "execute_llm_node",
    "llm_data_analysis_node",
    "llm_evaluation_node",
    "llm_optimization_node",
    "log_llm_event",
]