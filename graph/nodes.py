"""Graph nodes, one per existing agent (Phase 7).

Every node here is a thin coordinator. It invokes exactly one existing agent,
carries the authoritative ``AgentState`` through the graph, and records what
happened. No preprocessing, training, evaluation, Optuna search or model
ranking is reimplemented in this module: if a behaviour is not in ``agents/`` and
``tools/``, it does not belong here.

Two behaviours are shared by all six nodes and live in :func:`_execute_node`:

* a node never raises. Exceptions are captured as :class:`GraphError` records so
  the graph can stop cleanly and the caller still receives the details;
* a failed node may be retried, but only for an *unexpected* error. Errors from
  the project's own taxonomy are deterministic and are never retried.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from graph.routing import decide_optimization, is_retryable
from graph.state import (
    GraphError,
    GraphNode,
    GraphState,
    NodeRecord,
    NodeStatus,
    OptimizationOutcome,
    REQUIRED_NODES,
)
from llm.state import LLMStatus
from utils import config
from utils.logger import setup_logger
from utils.state import AgentState

logger = setup_logger("graph.nodes")

#: Signature of an agent invocation: take the machine learning state, run one
#: existing agent, return the updated state.
AgentInvoker = Callable[[AgentState], AgentState]

#: Optimization status strings that the graph is willing to publish as an
#: ``OptimizationOutcome``. ``not_started`` is deliberately absent: it means the
#: node did not get as far as doing anything.
_OPTIMIZATION_OUTCOME_VALUES = frozenset(
    member.value for member in OptimizationOutcome
)


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def single_line(text: Any, limit: int = 200) -> str:
    """Flatten text to one bounded line.

    Several scikit-learn errors span several lines. The project logs one line per
    event, so a multi-line message is collapsed here instead of dumping a wall of
    text into the console. The untruncated message is still kept on the record.

    Args:
        text: Any value; it is converted with ``str``.
        limit: Maximum length of the returned line.

    Returns:
        A single-line, length-bounded string.
    """
    flattened = " ".join(str(text).split())
    if len(flattened) > limit:
        return flattened[: limit - 3] + "..."
    return flattened


def log_event(node: GraphNode, event: str, detail: str = "") -> None:
    """Emit one single-line graph log entry.

    Args:
        node: The node reporting the event.
        event: The event name, for example ``STARTED`` or ``RETRY 1/1``.
        detail: Optional extra context appended to the same line.
    """
    message = f"{config.LANGGRAPH_LOG_PREFIX} {node.display_name} -> {event}"
    if detail:
        message = f"{message} ({detail})"
    logger.info(message)


# --------------------------------------------------------------------------- #
# Agent invocations
# --------------------------------------------------------------------------- #
# Each factory returns a closure over the graph-level inputs, so a run never
# depends on module state and two graphs can never interfere with each other.


def build_agent_invoker(
    state: GraphState, node: GraphNode
) -> AgentInvoker:
    """Return the callable that runs one existing agent for one node.

    Every agent is called with the project's real signature. The graph-level
    inputs that do not live on ``AgentState`` (dataset path, target column,
    cross-validation flag, trial count) are read from the graph state once and
    captured, so the invocation itself stays a single readable line.

    Args:
        state: The graph state holding the workflow inputs.
        node: The node whose agent should be invoked.

    Returns:
        A callable that runs the agent and returns the updated ``AgentState``.

    Raises:
        ValueError: If ``node`` is not a known graph node.
    """
    dataset_path = state.get("dataset_path")
    target_column = state.get("target_column")
    cross_validation = bool(state.get("cross_validation", True))
    n_trials = int(state.get("n_trials") or config.OPTUNA_N_TRIALS)
    output_dir = state.get("persistence_output_dir")
    enable_persistence = bool(state.get("enable_persistence", config.PERSISTENCE_ENABLED))

    if node is GraphNode.DATA_ANALYSIS:

        def run_data_analysis(agent_state: AgentState) -> AgentState:
            return DataAnalysisAgent().run(
                agent_state, dataset_path=dataset_path, target_column=target_column
            )

        return run_data_analysis

    if node is GraphNode.PREPROCESSING:

        def run_preprocessing(agent_state: AgentState) -> AgentState:
            return PreprocessingAgent().run(agent_state)

        return run_preprocessing

    if node is GraphNode.MODEL_SELECTION:

        def run_model_selection(agent_state: AgentState) -> AgentState:
            return ModelSelectionAgent().run(agent_state)

        return run_model_selection

    if node is GraphNode.TRAINING:

        def run_training(agent_state: AgentState) -> AgentState:
            return TrainingAgent().run(agent_state, cross_validation=cross_validation)

        return run_training

    if node is GraphNode.EVALUATION:

        def run_evaluation(agent_state: AgentState) -> AgentState:
            return EvaluationAgent().run(agent_state)

        return run_evaluation

    if node is GraphNode.OPTIMIZATION:

        def run_optimization(agent_state: AgentState) -> AgentState:
            return OptimizationAgent().run(
                agent_state, n_trials=n_trials, cross_validation=cross_validation
            )

        return run_optimization

    if node is GraphNode.MODEL_PERSISTENCE:
        # The Phase 8 interpretation records travel in the graph state rather than
        # on the AgentState, so they are read here and copied into the dashboard
        # read model. Only their already-recorded text is passed on: the node
        # cannot change whether a model is persisted.
        llm_results = list(state.get("llm_results") or [])

        def run_persistence(agent_state: AgentState) -> AgentState:
            return PersistenceAgent().run(
                agent_state,
                output_dir=output_dir,
                llm_results=llm_results,
                enabled=enable_persistence,
            )

        return run_persistence

    raise ValueError(f"Unknown graph node: {node!r}")


# --------------------------------------------------------------------------- #
# The shared node runner
# --------------------------------------------------------------------------- #


def resolve_max_retries(state: GraphState) -> int:
    """Return the retry budget for this run, clamped to a non-negative integer.

    A malformed value falls back to the configured default instead of becoming
    an unlimited retry loop.

    Args:
        state: The graph state.

    Returns:
        The number of retries allowed per node.
    """
    raw = state.get("max_retries")
    if raw is None:
        return max(0, int(config.LANGGRAPH_MAX_RETRIES))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning(
            "%s invalid max_retries %r; using the configured default %d",
            config.LANGGRAPH_LOG_PREFIX,
            raw,
            config.LANGGRAPH_MAX_RETRIES,
        )
        return max(0, int(config.LANGGRAPH_MAX_RETRIES))


def _execute_node(
    state: GraphState,
    node: GraphNode,
    invoke: AgentInvoker,
) -> GraphState:
    """Run one agent with the bounded retry policy and record the outcome.

    The returned mapping is a partial state update. ``node_records`` and
    ``errors`` are lists with an append reducer in the graph schema, so only the
    entries produced by this call are returned; every other key is replaced with
    its full new value.

    Args:
        state: The incoming graph state.
        node: The node being executed.
        invoke: Callable that runs the agent and returns the updated state.

    Returns:
        The partial state update for this node.
    """
    max_retries = resolve_max_retries(state)
    agent_state: AgentState = state["agent_state"]

    log_event(node, "STARTED")
    started_at = time.perf_counter()

    attempts = 0
    new_errors = []
    # `last_error` is the most recent error seen; `final_error` is the error that
    # was still standing when the node gave up. They differ when a retry
    # succeeds, and a node that recovers must not be reported as failed.
    last_error: Optional[GraphError] = None
    final_error: Optional[GraphError] = None

    while True:
        attempts += 1
        try:
            agent_state = invoke(agent_state)
            final_error = None
            break
        except Exception as error:  # noqa: BLE001 - captured on purpose
            retryable = is_retryable(error)
            last_error = GraphError(
                node=node.value,
                error_type=type(error).__name__,
                message=single_line(error, limit=1000),
                attempt=attempts,
                retryable=retryable,
            )
            new_errors.append(last_error)
            log_event(
                node,
                "FAILED",
                f"{type(error).__name__} on attempt {attempts}"
                + ("" if retryable else ", not retryable"),
            )
            if retryable and (attempts - 1) < max_retries:
                log_event(node, f"RETRY {attempts}/{max_retries}")
                continue
            final_error = last_error
            break

    duration = time.perf_counter() - started_at
    failed = final_error is not None
    recovered = last_error is not None and not failed
    status = NodeStatus.FAILED if failed else NodeStatus.COMPLETED
    retries_used = max(0, attempts - 1)

    if failed:
        log_event(
            node,
            "FAILED",
            f"giving up after {attempts} attempt(s): "
            f"{final_error.error_type if final_error else 'unknown'}",
        )
    else:
        detail = f"{duration:.4f} s"
        if recovered:
            detail = f"{detail}, recovered on attempt {attempts}"
        log_event(node, "COMPLETED", detail)

    is_required = node in REQUIRED_NODES
    stop_requested = bool(state.get("stop_requested"))
    # Only a required node stops the run. An optimization failure must not
    # destroy a valid Phase 5 evaluation result, and a persistence failure must
    # not destroy it either - it only means no file was written. Both are
    # reported as `partial`.
    if failed and is_required and state.get("stop_on_failure", True):
        stop_requested = True

    node_status: Dict[str, str] = dict(state.get("node_status") or {})
    node_status[node.value] = status.value

    retries: Dict[str, int] = dict(state.get("retries") or {})
    retries[node.value] = retries_used

    record = NodeRecord(
        node=node.value,
        display_name=node.display_name,
        status=status,
        attempts=attempts,
        retries=retries_used,
        duration=duration,
        error_type=final_error.error_type if final_error else None,
        error_message=final_error.message if final_error else None,
    )

    update: GraphState = {
        "agent_state": agent_state,
        "current_node": node.value,
        "workflow_status": "running",
        "node_status": node_status,
        "node_records": [record],
        "node_errors": new_errors,
        "retries": retries,
        "stop_requested": stop_requested,
        "failed_node": node.value if failed else state.get("failed_node"),
        # Only an error that was never recovered from stays as `last_error`; a
        # node that succeeded on a retry leaves the previous value in place.
        "last_error": final_error if final_error else state.get("last_error"),
    }
    if node is GraphNode.OPTIMIZATION:
        # The optimization node owns the outcome. The agent's own status is
        # preferred because it distinguishes completed from partial, but it is
        # only trusted when it is a known value; a node that raised reports
        # `failed` and a node that never started reports `not_run`.
        agent_value = str(getattr(agent_state, "optimization_status", "") or "")
        if failed:
            update["optimization_outcome"] = OptimizationOutcome.FAILED.value
        elif agent_value in _OPTIMIZATION_OUTCOME_VALUES:
            update["optimization_outcome"] = agent_value
        else:
            update["optimization_outcome"] = OptimizationOutcome.NOT_RUN.value
    return update


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def data_analysis_node(state: GraphState) -> GraphState:
    """Run the Data Analysis Agent.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return _execute_node(
        state,
        GraphNode.DATA_ANALYSIS,
        build_agent_invoker(state, GraphNode.DATA_ANALYSIS),
    )


def preprocessing_node(state: GraphState) -> GraphState:
    """Run the Preprocessing Agent.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return _execute_node(
        state,
        GraphNode.PREPROCESSING,
        build_agent_invoker(state, GraphNode.PREPROCESSING),
    )


def model_selection_node(state: GraphState) -> GraphState:
    """Run the Model Selection Agent.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return _execute_node(
        state,
        GraphNode.MODEL_SELECTION,
        build_agent_invoker(state, GraphNode.MODEL_SELECTION),
    )


def training_node(state: GraphState) -> GraphState:
    """Run the Training Agent.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return _execute_node(
        state,
        GraphNode.TRAINING,
        build_agent_invoker(state, GraphNode.TRAINING),
    )


def evaluation_node(state: GraphState) -> GraphState:
    """Run the Evaluation Agent and record the optimization decision.

    The decision is stored here, after evaluation has finished and before the
    routing edge is taken, so the recorded reason is exactly the one that was
    acted on. Recomputing it from the final state would be wrong, because once
    the optimization node has run the state no longer satisfies the "not already
    done" condition.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    update = _execute_node(
        state,
        GraphNode.EVALUATION,
        build_agent_invoker(state, GraphNode.EVALUATION),
    )
    merged: GraphState = {**state, **update}
    update["optimization_decision"] = decide_optimization(merged).value
    if state.get("llm_enabled"):
        # Phase 8: the recorded decision is mirrored into the LLM status map so a
        # report can show what the model was told, without the model being able to
        # change it. The value written here is the deterministic decision only.
        llm_status: Dict[str, str] = dict(state.get("llm_status") or {})
        llm_status["optimization_decision"] = update["optimization_decision"]
        update["llm_status"] = llm_status
    return update


def optimization_node(state: GraphState) -> GraphState:
    """Run the Optimization Agent.

    A failure here does not stop the workflow: the Phase 5 evaluation result is
    preserved and the run is reported as ``partial``.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return _execute_node(
        state,
        GraphNode.OPTIMIZATION,
        build_agent_invoker(state, GraphNode.OPTIMIZATION),
    )


def model_persistence_node(state: GraphState) -> GraphState:
    """Run the Model Persistence Agent.

    A failure here does not stop the workflow and does not make it ``failed``: the
    evaluation result the dashboard would have shown is still correct, so the run is
    reported as ``partial`` and the node records the reason the artifact is
    missing.

    The node decides nothing. Whether a model is persisted, and which one, is
    settled by the recorded Phase 4-6 results; no language model output and no
    earlier node's status is consulted here beyond the prerequisites the agent
    itself verifies.

    Args:
        state: The current graph state.

    Returns:
        The partial state update.
    """
    return _execute_node(
        state,
        GraphNode.MODEL_PERSISTENCE,
        build_agent_invoker(state, GraphNode.MODEL_PERSISTENCE),
    )


#: Node to callable mapping, used when building the graph.
NODE_FUNCTIONS: Dict[GraphNode, Callable[[GraphState], GraphState]] = {
    GraphNode.DATA_ANALYSIS: data_analysis_node,
    GraphNode.PREPROCESSING: preprocessing_node,
    GraphNode.MODEL_SELECTION: model_selection_node,
    GraphNode.TRAINING: training_node,
    GraphNode.EVALUATION: evaluation_node,
    GraphNode.OPTIMIZATION: optimization_node,
    GraphNode.MODEL_PERSISTENCE: model_persistence_node,
}
