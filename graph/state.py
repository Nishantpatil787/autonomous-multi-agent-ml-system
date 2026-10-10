"""Typed state for the LangGraph orchestration layer (Phase 7).

The existing :class:`utils.state.AgentState` stays the authoritative machine
learning state. Nothing in this module duplicates its fields: the graph carries
a single reference to it and adds only the information an orchestrator needs,
namely which node is running, how many times it was retried, what error
occurred, whether optimization was requested and whether the workflow should
stop.

Two representations live here and they are deliberately different types:

``GraphState``
    A :class:`typing.TypedDict`, because that is the shape LangGraph requires for
    a graph state schema. It is the object that travels between nodes.

    Note the deliberate name difference: the graph records failures in
    ``node_errors`` (a list of :class:`GraphError` objects), while
    ``AgentState.errors`` holds plain strings. Both exist during a run, so they
    must not share a name.

``WorkflowResult``
    A frozen dataclass, because that is what the caller receives. It is the
    typed, inspectable summary of a finished run.

Phase 8 adds three optional :class:`LLMGraphNode` values and the
``llm_enabled`` / ``llm_results`` bookkeeping. They default to off and empty, and
they are only ever *appended* to, so a run that does not use the layer produces
exactly the Phase 7 result it produced before.

No live scikit-learn estimator and no live Optuna ``Study`` object is stored
here. The only live object is the ``AgentState`` reference, which already
excludes its own runtime-only pipeline fields from ``to_dict()``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from llm.state import LLMResult, LLMRole, LLMStatus
from utils.state import AgentState

# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


class GraphNode(Enum):
    """One graph node per existing agent.

    The values are the LangGraph node names, so they double as keys in the state
    and as the identifiers used in logs and reports.

    Phase 9 appends :attr:`MODEL_PERSISTENCE`. It is the only node whose output is
    a set of files rather than state, and it is always present in the default
    graph - not gated behind a flag - because the documented dashboard flow is
    ``python app.py --graph`` followed by ``streamlit run dashboard.py``, and the
    dashboard can only read an artifact that the default run wrote.
    """

    DATA_ANALYSIS = "data_analysis"
    PREPROCESSING = "preprocessing"
    MODEL_SELECTION = "model_selection"
    TRAINING = "training"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    MODEL_PERSISTENCE = "model_persistence"

    @property
    def display_name(self) -> str:
        """Return the human readable name used in logs and reports."""
        return GRAPH_NODE_DISPLAY_NAMES[self]

    @classmethod
    def ordered(cls) -> List["GraphNode"]:
        """Return every node in canonical workflow order.

        The order is defined by the enum declaration order, which is the order
        the workflow runs in.
        """
        return list(cls)


GRAPH_NODE_DISPLAY_NAMES: Dict[GraphNode, str] = {
    GraphNode.DATA_ANALYSIS: "Data Analysis",
    GraphNode.PREPROCESSING: "Preprocessing",
    GraphNode.MODEL_SELECTION: "Model Selection",
    GraphNode.TRAINING: "Training",
    GraphNode.EVALUATION: "Evaluation",
    GraphNode.OPTIMIZATION: "Optimization",
    GraphNode.MODEL_PERSISTENCE: "Model Persistence",
}

#: Nodes that must succeed for the workflow to be considered successful. A
#: failure in one of these ends the run. Two nodes are deliberately excluded:
#:
#: :attr:`GraphNode.OPTIMIZATION`
#:     A search can legitimately find nothing better than the baseline, and a
#:     failed trial is a normal search outcome. Losing it costs information, not
#:     a result.
#:
#: :attr:`GraphNode.MODEL_PERSISTENCE`
#:     Writing files is not a machine learning result. If the write fails, the
#:     evaluation that produced the numbers is still correct and still usable; the
#:     workflow is ``partial``, not ``failed``.
#:
#: Both map to ``WorkflowStatus.PARTIAL`` in
#: :func:`graph.routing.derive_workflow_status` rather than to ``FAILED``.
REQUIRED_NODES: tuple = (
    GraphNode.DATA_ANALYSIS,
    GraphNode.PREPROCESSING,
    GraphNode.MODEL_SELECTION,
    GraphNode.TRAINING,
    GraphNode.EVALUATION,
)


class LLMGraphNode(Enum):
    """Optional Phase 8 interpretation nodes.

    These are kept separate from :class:`GraphNode` on purpose, for two reasons.
    First, one node per existing agent is the Phase 7 contract and mixing optional
    commentary into that enum would blur it. Second, and more importantly, an LLM
    node can never be required: no LLM failure may stop the workflow, so these
    nodes must not appear in :data:`REQUIRED_NODES` and must never set
    ``stop_requested``.

    Each value corresponds to exactly one
    :meth:`~llm.service.LLMService` method, and each runs *after* the deterministic
    step whose results it explains.
    """

    DATA_ANALYSIS = "llm_data_analysis"
    EVALUATION = "llm_evaluation"
    OPTIMIZATION = "llm_optimization"

    @property
    def display_name(self) -> str:
        """Return the human readable name used in logs and reports."""
        return LLM_GRAPH_NODE_DISPLAY_NAMES[self]

    @property
    def role(self) -> Any:
        """Return the :class:`~llm.state.LLMRole` this node answers for."""
        return LLM_NODE_ROLES[self]

    @classmethod
    def ordered(cls) -> List["LLMGraphNode"]:
        """Return every LLM node in canonical workflow order."""
        return list(cls)


LLM_GRAPH_NODE_DISPLAY_NAMES: Dict["LLMGraphNode", str] = {
    LLMGraphNode.DATA_ANALYSIS: "LLM Data Analysis",
    LLMGraphNode.EVALUATION: "LLM Evaluation",
    LLMGraphNode.OPTIMIZATION: "LLM Optimization",
}

#: Each LLM node is bound to exactly one role. This mapping is what makes the
#: binding checkable: an LLM node cannot be wired to the wrong role by accident.
LLM_NODE_ROLES: Dict["LLMGraphNode", Any] = {
    LLMGraphNode.DATA_ANALYSIS: LLMRole.DATA_ANALYSIS,
    LLMGraphNode.EVALUATION: LLMRole.EVALUATION,
    LLMGraphNode.OPTIMIZATION: LLMRole.OPTIMIZATION,
}


# --------------------------------------------------------------------------- #
# Status vocabularies
# --------------------------------------------------------------------------- #


class NodeStatus(Enum):
    """Outcome of a single node execution."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def is_success(self) -> bool:
        """Return True when the node finished without an error."""
        return self is NodeStatus.COMPLETED


class WorkflowStatus(Enum):
    """Outcome of the whole graph run.

    ``COMPLETED`` means every required node succeeded and neither the optional
    optimization node nor the Phase 9 persistence node failed.
    ``PARTIAL`` means the required nodes succeeded but one of those two nodes
    failed: the Phase 5 evaluation result is still valid and is deliberately
    preserved, so the run is reported as usable-with-a-caveat rather than broken.
    ``FAILED`` means a required node failed.

    Phase 9 added the persistence node to the ``PARTIAL`` set, for the same reason
    the optimization node was there: neither one produces the machine learning
    result, so neither is allowed to destroy it.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"

    @property
    def is_success(self) -> bool:
        """Return True for outcomes where the required work completed."""
        return self in (WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL)


class OptimizationDecision(Enum):
    """Why the graph did or did not run the optimization node.

    Exactly one member is recorded per run, so a skipped search always carries a
    reason and is never indistinguishable from a search that was never
    considered.
    """

    RUN = "run"
    SKIP_DISABLED = "skip_disabled"
    SKIP_EVALUATION_INCOMPLETE = "skip_evaluation_incomplete"
    SKIP_NO_CANDIDATES = "skip_no_candidates"
    SKIP_ALREADY_DONE = "skip_already_done"

    @property
    def should_run(self) -> bool:
        """Return True only for :attr:`RUN`."""
        return self is OptimizationDecision.RUN

    @property
    def is_skip(self) -> bool:
        """Return True for every skip reason."""
        return not self.should_run

    @property
    def reason(self) -> str:
        """Return a one-line explanation of the decision."""
        return GRAPH_DECISION_REASONS[self]


GRAPH_DECISION_REASONS: Dict[OptimizationDecision, str] = {
    OptimizationDecision.RUN: (
        "Optimization will run: it is enabled, the evaluation phase succeeded, "
        "baseline candidates exist and it has not already run in this workflow."
    ),
    OptimizationDecision.SKIP_DISABLED: (
        "Optimization was skipped because it is disabled for this run "
        "(--no-optimize or LANGGRAPH_ENABLE_OPTIMIZATION = False)."
    ),
    OptimizationDecision.SKIP_EVALUATION_INCOMPLETE: (
        "Optimization was skipped because the evaluation phase did not complete "
        "successfully, so there is no validated result to improve on."
    ),
    OptimizationDecision.SKIP_NO_CANDIDATES: (
        "Optimization was skipped because no baseline experiment produced usable "
        "metrics, so there is no model to tune."
    ),
    OptimizationDecision.SKIP_ALREADY_DONE: (
        "Optimization was skipped because the optimization agent has already run in "
        "this workflow. Starting it again would repeat the same search, so the graph "
        "routes straight to model persistence instead."
    ),
}


class OptimizationOutcome(Enum):
    """What actually happened to the optimization node."""

    NOT_RUN = "not_run"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def ran(self) -> bool:
        """Return True when the optimization node was actually executed."""
        return self is not OptimizationOutcome.NOT_RUN


# --------------------------------------------------------------------------- #
# Typed records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphError:
    """A single raised exception, captured instead of propagated.

    Keeping the exception as data is what allows the workflow to stop cleanly:
    the graph never crashes, and the caller still receives the type, message and
    attempt number of every failure.
    """

    node: str
    error_type: str
    message: str
    attempt: int
    retryable: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "node": self.node,
            "error_type": self.error_type,
            "message": self.message,
            "attempt": self.attempt,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class NodeRecord:
    """The recorded outcome of one node execution."""

    node: str
    display_name: str
    status: NodeStatus
    attempts: int
    retries: int
    duration: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "node": self.node,
            "display_name": self.display_name,
            "status": self.status.value,
            "attempts": self.attempts,
            "retries": self.retries,
            "duration": self.duration,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class WorkflowResult:
    """The typed summary returned by a finished workflow run."""

    workflow_status: WorkflowStatus
    agent_state: AgentState
    node_records: List[NodeRecord] = field(default_factory=list)
    optimization_decision: OptimizationDecision = OptimizationDecision.SKIP_DISABLED
    optimization_outcome: OptimizationOutcome = OptimizationOutcome.NOT_RUN
    failed_node: Optional[str] = None
    total_retries: int = 0
    total_attempts: int = 0
    errors: List[GraphError] = field(default_factory=list)
    stopped_early: bool = False
    step_limit_reached: bool = False
    # Phase 8: the optional interpretation layer. ``llm_enabled`` records whether
    # it was requested for this run; ``llm_results`` holds one validated record per
    # interpretation step. Both default to "off" so a Phase 7 caller that does not
    # know about the layer keeps behaving exactly as before.
    llm_enabled: bool = False
    llm_results: List[LLMResult] = field(default_factory=list)

    # -- derived views ----------------------------------------------------- #

    @property
    def executed_nodes(self) -> List[str]:
        """Return the node names that were executed, in execution order."""
        return [r.node for r in self.node_records]

    @property
    def completed_nodes(self) -> List[str]:
        """Return the node names that completed successfully."""
        return [r.node for r in self.node_records if r.status.is_success]

    @property
    def failed_nodes(self) -> List[str]:
        """Return the node names that failed."""
        return [r.node for r in self.node_records if r.status is NodeStatus.FAILED]

    @property
    def last_error(self) -> Optional[GraphError]:
        """Return the final recorded error, or None when nothing failed."""
        return self.errors[-1] if self.errors else None

    @property
    def final_agent_state(self) -> AgentState:
        """Return the authoritative machine learning state of this run."""
        return self.agent_state

    def record_for(self, node: Any) -> Optional[NodeRecord]:
        """Return the record of one node.

        Args:
            node: A :class:`GraphNode` or a node name.

        Returns:
            The matching :class:`NodeRecord`, or None when the node never ran.
        """
        name = node.value if isinstance(node, GraphNode) else str(node)
        for record in self.node_records:
            if record.node == name:
                return record
        return None

    # -- Phase 8 views ----------------------------------------------------- #

    @property
    def llm_statuses(self) -> Dict[str, str]:
        """Return the recorded LLM status per role, in workflow order.

        Returns:
            A mapping of role value to :class:`~llm.state.LLMStatus` value.
        """
        return {result.role.value: result.status.value for result in self.llm_results}

    @property
    def llm_completed_roles(self) -> List[str]:
        """Return the roles that produced a validated response."""
        return [r.role.value for r in self.llm_results if r.succeeded]

    @property
    def llm_unavailable_roles(self) -> List[str]:
        """Return the roles whose step failed for a non-model reason.

        This is the fallback signal: when it is non-empty, the pipeline completed
        without the generated text, which is the expected outcome when Ollama is
        not installed.
        """
        return [r.role.value for r in self.llm_results if r.failed]

    def llm_result_for(self, role: Any) -> Optional[LLMResult]:
        """Return the record of one interpretation step.

        Args:
            role: An :class:`~llm.state.LLMRole` or a role value.

        Returns:
            The matching :class:`~llm.state.LLMResult`, or None when the step did
            not run.
        """
        name = role.value if isinstance(role, LLMRole) else str(role)
        for result in self.llm_results:
            if result.role.value == name:
                return result
        return None

    @property
    def llm_forbidden_claims(self) -> List[str]:
        """Return every over-certain phrase found in generated text.

        Surfaced so a claim the project forbids is never silently printed as if it
        were an acceptable interpretation.
        """
        found: List[str] = []
        for result in self.llm_results:
            found.extend(result.forbidden_claims)
        return found

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation of the whole run.

        The nested ``AgentState`` is serialised with its own ``to_dict()``, which
        already excludes the live pipeline objects, so the result contains no
        scikit-learn or Optuna object.
        """
        return {
            "workflow_status": self.workflow_status.value,
            "stopped_early": self.stopped_early,
            "step_limit_reached": self.step_limit_reached,
            "failed_node": self.failed_node,
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "executed_nodes": self.executed_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "optimization_decision": self.optimization_decision.value,
            "optimization_decision_reason": self.optimization_decision.reason,
            "optimization_outcome": self.optimization_outcome.value,
            "node_records": [r.to_dict() for r in self.node_records],
            "errors": [e.to_dict() for e in self.errors],
            # Phase 8: present on every result, and empty when the layer was not
            # requested, so the key set does not change between runs.
            "llm_enabled": self.llm_enabled,
            "llm_results": [r.to_dict() for r in self.llm_results],
            "agent_state": self.agent_state.to_dict(),
        }


# --------------------------------------------------------------------------- #
# LangGraph state schema
# --------------------------------------------------------------------------- #


class GraphState(TypedDict, total=False):
    """The state object that travels between LangGraph nodes.

    ``agent_state`` holds the one and only reference to the authoritative machine
    learning state; the remaining keys are orchestration bookkeeping. Two list
    fields use an append reducer so that a node returning a single record
    accumulates instead of replacing the history.
    """

    # -- the authoritative machine learning state -------------------------- #
    agent_state: AgentState

    # -- inputs the workflow was started with ----------------------------- #
    dataset_path: Optional[str]
    target_column: Optional[str]
    enable_optimization: bool
    cross_validation: bool
    n_trials: int
    max_retries: int
    stop_on_failure: bool
    # Phase 9: where the artifacts are written. ``None`` means the configured
    # directory, and it is an input rather than an orchestration result, so a test
    # can redirect the workflow into a temporary folder without touching global
    # configuration.
    persistence_output_dir: Optional[str]
    enable_persistence: bool

    # -- orchestration bookkeeping ---------------------------------------- #
    current_node: str
    workflow_status: str
    node_status: Dict[str, str]
    node_records: Annotated[List[NodeRecord], operator.add]
    node_errors: Annotated[List[GraphError], operator.add]
    retries: Dict[str, int]
    failed_node: Optional[str]
    stop_requested: bool
    optimization_decision: str
    optimization_outcome: str
    last_error: Optional[GraphError]
    # Phase 8: the service the interpretation nodes call. It lives in the state
    # rather than in a module global so a run is self-contained, two graphs cannot
    # interfere, and a test can inject a stub without monkeypatching anything.
    llm_service: Optional[Any]
    llm_status: Dict[str, str]
    # Phase 8: set once at construction and never changed by a node, so no
    # interpretation step can alter it. ``llm_results`` uses an append reducer, so
    # a node returns only the record it produced.
    llm_enabled: bool
    llm_results: Annotated[List[LLMResult], operator.add]


#: Keys that :class:`GraphState` defines. Exposed for tests and for the
#: structural description of the graph.
GRAPH_STATE_KEYS: tuple = tuple(GraphState.__annotations__.keys())


def create_graph_state(
    *,
    agent_state: Optional[AgentState] = None,
    dataset_path: Optional[str] = None,
    target_column: Optional[str] = None,
    enable_optimization: bool = True,
    cross_validation: bool = True,
    n_trials: int = 20,
    max_retries: int = 1,
    stop_on_failure: bool = True,
    enable_llm: bool = False,
    enable_persistence: bool = True,
    persistence_output_dir: Optional[str] = None,
) -> GraphState:
    """Build the initial graph state for a workflow run.

    Every field is given an explicit starting value rather than being left to the
    TypedDict defaults, so the behaviour is identical no matter how a node
    happens to read it. That matters for determinism.

    Args:
        agent_state: An existing machine learning state to continue from. A new
            empty :class:`AgentState` is created when None.
        dataset_path: CSV file the Data Analysis node should read.
        target_column: Explicit target column, or None to let the agent detect it.
        enable_optimization: Whether the optimization node may run.
        cross_validation: Whether the training node may use cross-validation.
        n_trials: Trials per model for the optimization node.
        max_retries: Retry budget for each node.
        stop_on_failure: Whether a failed required node ends the workflow.
        enable_llm: Whether the optional Phase 8 interpretation layer may run.
            False by default, which is what keeps the Phase 7 graph unchanged.
        enable_persistence: Whether the Phase 9 persistence node may write files.
        persistence_output_dir: Directory the artifacts are written to, or None
            for the configured one.

    Returns:
        A fully populated :class:`GraphState`.
    """
    from utils import config

    return GraphState(
        agent_state=agent_state if agent_state is not None else AgentState(),
        dataset_path=dataset_path,
        target_column=target_column,
        enable_optimization=enable_optimization,
        cross_validation=cross_validation,
        n_trials=n_trials,
        max_retries=max_retries,
        stop_on_failure=stop_on_failure,
        enable_persistence=enable_persistence,
        persistence_output_dir=persistence_output_dir,
        current_node="",
        workflow_status=WorkflowStatus.PENDING.value,
        node_status={node.value: NodeStatus.PENDING.value for node in GraphNode},
        node_records=[],
        node_errors=[],
        retries={},
        failed_node=None,
        stop_requested=False,
        optimization_decision=OptimizationDecision.SKIP_DISABLED.value,
        optimization_outcome=OptimizationOutcome.NOT_RUN.value,
        last_error=None,
        llm_enabled=bool(enable_llm),
        llm_results=[],
        llm_service=None,
        llm_status={},
    )
