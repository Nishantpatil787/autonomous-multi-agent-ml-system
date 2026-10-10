"""Deterministic routing decisions for the LangGraph workflow (Phase 7).

Every function here is a pure comparison of values that are already recorded in
the state. There is no randomness, no clock-dependent branch and no language
model anywhere in this module, so the same state and configuration always
produce the same route.

The central decision is :func:`decide_optimization`, which answers "should the
optimization node run?". Its conditions are ordered from cheapest and most
absolute to most data dependent, so the recorded reason is always the *first*
reason that applied rather than an arbitrary one.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from graph.state import (
    GraphNode,
    GraphState,
    OptimizationDecision,
    WorkflowResult,
    WorkflowStatus,
)
from utils import config

# The LangGraph end-of-graph sentinel. Imported lazily-safe: langgraph.graph
# exposes END as a string constant and importing it here keeps this module free
# of any other langgraph symbol.
from langgraph.graph import END

#: Nodes that the workflow requires. A failure in one of these ends the run.
#: Phase 7 excluded the Optimization node and Phase 9 added the Model Persistence
#: node to the same exclusion, because neither node produces the machine learning
#: result: the search is optional extra information, and persistence only writes
#: files. Both failures downgrade the workflow to ``partial``.
REQUIRED_NODE_VALUES: Tuple[str, ...] = tuple(
    node.value
    for node in GraphNode
    if node not in (GraphNode.OPTIMIZATION, GraphNode.MODEL_PERSISTENCE)
)

#: Nodes whose failure makes the workflow ``partial`` instead of ``failed``.
#: Keeping them in one tuple means the status rule and the report cannot drift
#: apart: the same list decides both.
DEGRADABLE_NODE_VALUES: Tuple[str, ...] = (
    GraphNode.OPTIMIZATION.value,
    GraphNode.MODEL_PERSISTENCE.value,
)


# --------------------------------------------------------------------------- #
# Error classification (used by the retry mechanism)
# --------------------------------------------------------------------------- #


def non_retryable_error_types() -> Tuple[Type[BaseException], ...]:
    """Return the exception types that must never be retried.

    The project's own error taxonomy signals a deterministic problem: a missing
    file, an unmet prerequisite or an invalid configuration. Repeating the same
    call with the same inputs reproduces the same error, so retrying would only
    waste time. The class objects are imported here rather than stored in
    ``utils.config`` so that the config module keeps no dependency on the tools
    package.

    Returns:
        The tuple of exception classes treated as non-retryable.
    """
    from tools.data_tools import DataToolError
    from tools.evaluation_tools import EvaluationToolError
    from tools.model_tools import ModelToolError
    from tools.optimization_tools import OptimizationToolError
    from tools.persistence_tools import PersistenceError
    from tools.preprocessing_tools import PreprocessingError
    from tools.training_tools import TrainingToolError

    return (
        DataToolError,
        PreprocessingError,
        ModelToolError,
        TrainingToolError,
        EvaluationToolError,
        OptimizationToolError,
        PersistenceError,
    )


def is_retryable(error: BaseException) -> bool:
    """Return True when ``error`` is worth one more attempt.

    Args:
        error: The exception that a node raised.

    Returns:
        False for the project's own deterministic errors, True for anything
        unexpected, which may be transient.
    """
    return not isinstance(error, non_retryable_error_types())


def describe_retry_policy() -> str:
    """Return the one-line description of the retry rule for reports."""
    return config.LANGGRAPH_RETRY_RULE.format(max_retries=config.LANGGRAPH_MAX_RETRIES)


# --------------------------------------------------------------------------- #
# The optimization decision
# --------------------------------------------------------------------------- #


def optimization_was_performed(agent_state: Any) -> bool:
    """Return True when the optimization agent already ran on this state.

    Used to guarantee the graph never starts a second Optuna run. Any state left
    by the optimization agent counts, including a recorded skip and a recorded
    failure, because in both cases a search was already attempted or explicitly
    declined.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        True when the optimization node has already had its turn.
    """
    status = getattr(agent_state, "optimization_status", None)
    return bool(status) and str(status) != "not_started"


def decide_optimization(graph_state: GraphState) -> OptimizationDecision:
    """Decide whether the optimization node should run.

    The conditions are evaluated in this fixed order and the first match wins:

    1. optimization is disabled for this run;
    2. the evaluation phase did not complete successfully;
    3. the optimization agent already ran, so a second run is refused;
    4. no baseline experiment produced usable metrics, so there is nothing to tune.

    Args:
        graph_state: The current graph state.

    Returns:
        The :class:`OptimizationDecision` to record and act on.
    """
    agent_state = graph_state.get("agent_state")

    if not graph_state.get("enable_optimization", False):
        return OptimizationDecision.SKIP_DISABLED

    if not evaluation_succeeded(agent_state):
        return OptimizationDecision.SKIP_EVALUATION_INCOMPLETE

    if optimization_was_performed(agent_state):
        return OptimizationDecision.SKIP_ALREADY_DONE

    if not has_optimization_candidates(agent_state):
        return OptimizationDecision.SKIP_NO_CANDIDATES

    return OptimizationDecision.RUN


def evaluation_succeeded(agent_state: Any) -> bool:
    """Return True when the evaluation phase produced a usable result.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        True only when the evaluation status is ``completed``.
    """
    status = str(getattr(agent_state, "evaluation_status", "") or "")
    return status == "completed"


def has_optimization_candidates(agent_state: Any) -> bool:
    """Return True when at least one baseline model can be tuned.

    The check uses the successful experiments, because ``training_status`` is
    ``completed`` even when every single experiment failed. Reading the status
    alone would therefore let the graph start a search with nothing to search
    over.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        True when at least one experiment carries usable metrics.
    """
    successful = getattr(agent_state, "successful_experiments", None)
    if successful is not None:
        return len(successful) > 0
    return False


# --------------------------------------------------------------------------- #
# Edge routers
# --------------------------------------------------------------------------- #


def route_after_evaluation(graph_state: GraphState) -> str:
    """Route after the evaluation node using the optimization decision.

    Phase 9 changed one thing here: "no optimization" no longer means "end of
    workflow". It means "go straight to Model Persistence", so a run with
    ``--no-optimize`` still leaves a usable artifact for the dashboard.

    The decision itself is unchanged from Phase 7 and is still made by
    :func:`decide_optimization`, which reads only recorded state.

    Args:
        graph_state: The current graph state.

    Returns:
        The optimization node name when the search should run, otherwise the Model
        Persistence node name.
    """
    if graph_state.get("stop_requested"):
        return END
    if decide_optimization(graph_state).should_run:
        return GraphNode.OPTIMIZATION.value
    return GraphNode.MODEL_PERSISTENCE.value


def make_step_router(next_node: GraphNode) -> Callable[[GraphState], str]:
    """Build the router that runs after a required node.

    Every required node is followed by a conditional edge so that a failure can
    end the run explicitly and visibly in the graph structure, instead of the
    next node being entered with incomplete state and raising there.

    Args:
        next_node: The node to enter when the current one succeeded.

    Returns:
        A router callable returning ``next_node`` or the end sentinel.
    """

    def router(graph_state: GraphState) -> str:
        if graph_state.get("stop_requested"):
            return END
        return next_node.value

    router.__name__ = f"route_after_{next_node.value}"
    router.__doc__ = (
        f"Route to {next_node.display_name} on success, otherwise to the end of "
        "the workflow."
    )
    return router


def route_after_optimization(graph_state: GraphState) -> str:
    """Route after the optimization node to Model Persistence.

    Phase 9 made Model Persistence the terminal node, so optimization no longer
    ends the workflow. The router stays as small as every other router here: it
    reads ``stop_requested``, a field no agent writes on a success path, and
    returns the fixed next node. It deliberately reads nothing about the search
    result, because a search that found nothing must still produce an artifact.

    Args:
        graph_state: The current graph state.

    Returns:
        The Model Persistence node name, or the end sentinel when the workflow was
        asked to stop.
    """
    if graph_state.get("stop_requested"):
        return END
    return GraphNode.MODEL_PERSISTENCE.value


def route_after_llm_evaluation(graph_state: GraphState) -> str:
    """Apply the recorded optimization decision *after* the interpretation node.

    Phase 8 places the decision router behind ``llm_evaluation`` so the generated
    text is available when the workflow report is written. This is the load-bearing
    part of the architecture: the router reads the decision the evaluation node
    already stored and the recorded metrics, and never reads ``llm_results``,
    ``llm_status`` or anything else the language model produced.

    Args:
        graph_state: The current graph state.

    Returns:
        The optimization node name when the recorded decision was ``run``,
        otherwise the Model Persistence node name.
    """
    if graph_state.get("stop_requested"):
        return END
    if recorded_optimization_decision(graph_state).should_run:
        return GraphNode.OPTIMIZATION.value
    return GraphNode.MODEL_PERSISTENCE.value


def route_to_end(graph_state: GraphState) -> str:
    """Router for the final node: the workflow always ends here.

    Args:
        graph_state: The current graph state. Unused; present for a uniform
            router signature.

    Returns:
        The LangGraph end sentinel.
    """
    return END


def route_after_llm_node(next_node: Any) -> Callable[[GraphState], str]:
    """Build the router that runs after an optional interpretation node.

    Phase 8: an LLM node must be completely invisible to routing, so this router
    takes no parameter that a model could influence. It reads ``stop_requested``
    - a field no interpretation node may write - and returns the fixed next node.

    The signature deliberately accepts the destination rather than being built per
    node in :mod:`graph.workflow`, which keeps the "LLM never routes" property
    checkable: the router body below contains no reference to any LLM state.

    Args:
        next_node: The node to enter when this one succeeded.

    Returns:
        A router callable returning ``next_node`` or the end sentinel.
    """

    def router(graph_state: GraphState) -> str:
        if graph_state.get("stop_requested"):
            return END
        return next_node.value

    router.__name__ = f"route_after_{next_node.value}"
    router.__doc__ = (
        f"Route to {next_node.display_name} on success, otherwise to the end of "
        "the workflow. This router does not read any LLM state."
    )
    return router


def llm_does_not_influence_routing(graph_state: GraphState) -> bool:
    """Return whether the recorded LLM state could affect any routing decision.

    This exists as a checkable property rather than as a convention. It compares the
    decision with the interpretation results removed from the state, so a test can
    assert that the LLM layer is genuinely orthogonal to the graph's control flow:
    whichever way the model answered, the same route is taken.

    Args:
        graph_state: The graph state to test, normally mid-run.

    Returns:
        True when the recorded LLM keys do not participate in any decision.
    """
    from graph.state import LLMGraphNode

    decision = recorded_optimization_decision(graph_state)

    stripped = dict(graph_state)
    stripped["llm_results"] = []
    stripped["llm_status"] = {}
    stripped["llm_service"] = None
    for node in LLMGraphNode:
        stripped["node_status"] = {
            key: value
            for key, value in (graph_state.get("node_status") or {}).items()
            if key != node.value
        }
        stripped["node_records"] = [
            record
            for record in (graph_state.get("node_records") or [])
            if getattr(record, "node", None) != node.value
        ]

    return decide_optimization(stripped) is decision


def optimization_destinations() -> Dict[str, str]:
    """Return the mapping of route outcomes to node names for the decision edge.

    Both node outcomes lead to a node rather than one of them leading to the end
    of the workflow: the search runs first, or Model Persistence runs directly.
    Either way the run ends by writing an artifact. ``END`` stays in the mapping
    because both routers still return it when the workflow was asked to stop, and
    a conditional edge must be able to name every destination it can return.

    Returns:
        A mapping of ``"optimization"``, ``"model_persistence"`` and ``END`` to
        themselves.
    """
    return {
        GraphNode.OPTIMIZATION.value: GraphNode.OPTIMIZATION.value,
        GraphNode.MODEL_PERSISTENCE.value: GraphNode.MODEL_PERSISTENCE.value,
        END: END,
    }


def describe_routing_rule() -> str:
    """Return the one-line description of the optimization decision rule."""
    return config.LANGGRAPH_ROUTING_RULE


# --------------------------------------------------------------------------- #
# Outcome derivation
# --------------------------------------------------------------------------- #


#: Required node to the ``AgentState`` field that records whether the agent
#: behind it actually succeeded. Most agents report a failure by setting their
#: own status to ``failed`` and *returning*, so the node itself completes and
#: the node status alone would misreport the run as successful.
AGENT_STATUS_FIELDS: Dict[str, str] = {
    GraphNode.PREPROCESSING.value: "preprocessing_status",
    GraphNode.MODEL_SELECTION.value: "model_selection_status",
    GraphNode.TRAINING.value: "training_status",
    GraphNode.EVALUATION.value: "evaluation_status",
}


def agent_reported_failure(graph_state: GraphState) -> bool:
    """Return whether a required agent recorded a phase-level failure.

    Only the literal value ``failed`` counts. A node that was never entered
    leaves its status at ``not_started``, and a partially successful evaluation
    is not a failure, so neither may be treated as one.

    Args:
        graph_state: The current graph state.

    Returns:
        True when a required agent reported ``failed`` on the shared state.
    """
    agent_state = graph_state.get("agent_state")
    if agent_state is None:
        return False
    return any(
        str(getattr(agent_state, field, "") or "") == "failed"
        for field in AGENT_STATUS_FIELDS.values()
    )


def agent_failure_node(graph_state: GraphState) -> Optional[str]:
    """Return the required node whose agent reported a phase-level failure.

    Args:
        graph_state: The current graph state.

    Returns:
        The node name, or None when every agent that ran reported success.
    """
    agent_state = graph_state.get("agent_state")
    if agent_state is None:
        return None
    for node, field in AGENT_STATUS_FIELDS.items():
        if str(getattr(agent_state, field, "") or "") == "failed":
            return node
    return None


def derive_workflow_status(graph_state: GraphState) -> WorkflowStatus:
    """Derive the overall workflow status from the recorded node outcomes.

    The rules are deliberately narrow so that "completed" always means the
    required work ran:

    * a failed required node, a required agent that reported a phase-level
      failure, or the step ceiling, means ``failed``;
    * required nodes fine but the optimization node or the model persistence node
      failed means ``partial``;
    * anything else that did not request a stop means ``completed``.

    Args:
        graph_state: The current graph state.

    Returns:
        The derived :class:`WorkflowStatus`.
    """
    if graph_state.get("step_limit_reached"):
        return WorkflowStatus.FAILED

    node_status: Dict[str, str] = graph_state.get("node_status") or {}
    if any(node_status.get(value) == "failed" for value in REQUIRED_NODE_VALUES):
        return WorkflowStatus.FAILED

    if agent_reported_failure(graph_state):
        return WorkflowStatus.FAILED

    if any(
        node_status.get(value) == "failed" for value in DEGRADABLE_NODE_VALUES
    ):
        return WorkflowStatus.PARTIAL

    if graph_state.get("stop_requested"):
        return WorkflowStatus.FAILED

    return WorkflowStatus.COMPLETED


def derive_optimization_outcome(graph_state: GraphState) -> str:
    """Derive what happened to the optimization node.

    Args:
        graph_state: The current graph state.

    Returns:
        The value of the matching :class:`~graph.state.OptimizationOutcome`.
    """
    from graph.state import NodeStatus, OptimizationOutcome

    status = (graph_state.get("node_status") or {}).get(GraphNode.OPTIMIZATION.value)
    if status == NodeStatus.COMPLETED.value:
        return OptimizationOutcome.COMPLETED.value
    if status == NodeStatus.FAILED.value:
        return OptimizationOutcome.FAILED.value

    # The node was never entered. That is only legitimate when a decision said so.
    decision = graph_state.get("optimization_decision") or ""
    if decision == OptimizationDecision.SKIP_DISABLED.value:
        return OptimizationOutcome.NOT_RUN.value
    return OptimizationOutcome.SKIPPED.value


def recorded_optimization_decision(graph_state: GraphState) -> OptimizationDecision:
    """Return the optimization decision that was actually acted on.

    The evaluation node stores its decision *before* the routing edge is taken,
    and that stored value is the authoritative one. Recomputing it from the final
    state would be wrong: once the optimization node has run,
    ``optimization_was_performed`` is true and a recomputation would report
    ``skip_already_done`` for a run that did in fact optimize.

    Args:
        graph_state: The graph state, final or mid-run.

    Returns:
        The stored :class:`OptimizationDecision`, or a freshly computed one when
        the state carries no decision yet.
    """
    stored = graph_state.get("optimization_decision")
    if stored:
        try:
            return OptimizationDecision(stored)
        except ValueError:
            # An unknown value means the state was built elsewhere; recompute
            # rather than crash, and let the caller see a real decision.
            return decide_optimization(graph_state)
    return decide_optimization(graph_state)


def build_workflow_result(graph_state: GraphState) -> WorkflowResult:
    """Build the typed :class:`WorkflowResult` from a finished graph state.

    Args:
        graph_state: The final graph state returned by ``invoke``.

    Returns:
        The :class:`WorkflowResult` for the caller.
    """
    from graph.state import OptimizationOutcome

    node_records: List[Any] = list(graph_state.get("node_records") or [])
    errors: List[Any] = list(graph_state.get("node_errors") or [])
    retries: Dict[str, int] = graph_state.get("retries") or {}

    optimization_status = str(
        getattr(graph_state.get("agent_state"), "optimization_status", "") or ""
    )
    if optimization_status == "completed":
        outcome = OptimizationOutcome.COMPLETED
    elif optimization_status == "partial":
        outcome = OptimizationOutcome.PARTIAL
    elif optimization_status == "failed":
        outcome = OptimizationOutcome.FAILED
    else:
        outcome = OptimizationOutcome(derive_optimization_outcome(graph_state))

    # A node that raised sets ``failed_node`` itself. When an agent instead
    # reported a phase-level failure, no node failed, so the node is identified
    # here; otherwise the report would say "failed" with no failed node.
    failed_node = graph_state.get("failed_node")
    if failed_node is None:
        failed_node = agent_failure_node(graph_state)

    llm_results = list(graph_state.get("llm_results") or [])

    return WorkflowResult(
        workflow_status=derive_workflow_status(graph_state),
        agent_state=graph_state.get("agent_state"),
        node_records=node_records,
        optimization_decision=recorded_optimization_decision(graph_state),
        optimization_outcome=outcome,
        failed_node=failed_node,
        total_retries=sum(retries.values()),
        total_attempts=sum(record.attempts for record in node_records),
        errors=errors,
        stopped_early=bool(graph_state.get("stop_requested")),
        step_limit_reached=bool(graph_state.get("step_limit_reached")),
        # Phase 8: taken from the state, never recomputed, and never able to change
        # the three fields above.
        llm_enabled=bool(graph_state.get("llm_enabled")),
        llm_results=llm_results,
    )
