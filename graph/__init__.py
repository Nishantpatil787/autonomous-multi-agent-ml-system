"""LangGraph orchestration layer for the Autonomous Multi-Agent ML System.

Phase 7 adds coordination only. The six agents written in Phases 2-6 are
unchanged and remain individually usable; this package decides *when* each of
them runs, what to do when one fails, and whether the optimization search should
happen at all.

Phase 8 adds an optional interpretation layer. Phase 9 appends the Model
Persistence node as the terminal node, so the graph now always ends by writing
``models/best_model.joblib`` and its metadata. Both additions are additive: the
persistence node decides nothing, and a language model - if one is enabled - is
still never able to influence a routing decision.

Typical use::

    from graph.workflow import run_workflow

    result = run_workflow(dataset_path="datasets/sample_classification.csv")
    print(result.workflow_status.value, result.optimization_decision.value)
"""

__version__ = "0.8.0"

from graph.llm_nodes import (
    LLM_NODE_FUNCTIONS,
    build_llm_invoker,
    describe_llm_nodes,
    llm_data_analysis_node,
    llm_evaluation_node,
    llm_optimization_node,
)
from graph.nodes import (
    NODE_FUNCTIONS,
    build_agent_invoker,
    data_analysis_node,
    evaluation_node,
    log_event,
    model_persistence_node,
    model_selection_node,
    optimization_node,
    preprocessing_node,
    training_node,
)
from graph.routing import (
    DEGRADABLE_NODE_VALUES,
    decide_optimization,
    derive_workflow_status,
    describe_retry_policy,
    describe_routing_rule,
    has_optimization_candidates,
    is_retryable,
    llm_does_not_influence_routing,
    make_step_router,
    non_retryable_error_types,
    optimization_destinations,
    optimization_was_performed,
    route_after_evaluation,
    route_after_llm_evaluation,
    route_after_llm_node,
    route_after_optimization,
    route_to_end,
)
from graph.state import (
    GRAPH_STATE_KEYS,
    LLM_NODE_ROLES,
    REQUIRED_NODES,
    GraphError,
    GraphNode,
    GraphState,
    LLMGraphNode,
    NodeRecord,
    NodeStatus,
    OptimizationDecision,
    OptimizationOutcome,
    WorkflowResult,
    WorkflowStatus,
    create_graph_state,
)
from graph.workflow import (
    LLM_INSERT_AFTER,
    PERSISTENCE_NODE,
    REWIRED_BY_LLM,
    WORKFLOW_ORDER,
    build_workflow,
    describe_structure,
    expected_structure,
    format_workflow_report,
    run_workflow,
)

__all__ = [
    "__version__",
    # state
    "GraphNode",
    "NodeStatus",
    "WorkflowStatus",
    "OptimizationDecision",
    "OptimizationOutcome",
    "GraphError",
    "NodeRecord",
    "WorkflowResult",
    "GraphState",
    "GRAPH_STATE_KEYS",
    "REQUIRED_NODES",
    "DEGRADABLE_NODE_VALUES",
    "create_graph_state",
    # Phase 8: optional interpretation nodes
    "LLMGraphNode",
    "LLM_NODE_ROLES",
    "LLM_INSERT_AFTER",
    "LLM_NODE_FUNCTIONS",
    "llm_data_analysis_node",
    "llm_evaluation_node",
    "llm_optimization_node",
    "build_llm_invoker",
    "describe_llm_nodes",
    # nodes
    "data_analysis_node",
    "preprocessing_node",
    "model_selection_node",
    "training_node",
    "evaluation_node",
    "optimization_node",
    "model_persistence_node",
    "NODE_FUNCTIONS",
    "build_agent_invoker",
    "log_event",
    # routing
    "decide_optimization",
    "derive_workflow_status",
    "has_optimization_candidates",
    "optimization_was_performed",
    "route_after_evaluation",
    "route_after_optimization",
    "route_after_llm_evaluation",
    "route_after_llm_node",
    "route_to_end",
    "optimization_destinations",
    "llm_does_not_influence_routing",
    "make_step_router",
    "is_retryable",
    "non_retryable_error_types",
    "describe_routing_rule",
    "describe_retry_policy",
    # workflow
    "build_workflow",
    "run_workflow",
    "describe_structure",
    "expected_structure",
    "format_workflow_report",
    "WORKFLOW_ORDER",
    "PERSISTENCE_NODE",
    "REWIRED_BY_LLM",
]
