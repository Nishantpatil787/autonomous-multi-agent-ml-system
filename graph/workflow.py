"""LangGraph workflow builder and runner for Phase-7 orchestration.

The multi-agent ML pipeline is compiled into a LangGraph StateGraph.
No ML logic is implemented here — the graph delegates entirely to
the existing agents via the node functions in graph/nodes.py.

Normal workflow:
    START → data_analysis → preprocessing → model_selection →
    training → evaluation → [should_optimize?] → optimization → END
                                              └─────────────────────→ END

Failure at any required node routes directly to END.
Optimization failure preserves Phase-5 results and routes to END.
"""

import warnings
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    data_analysis_node,
    evaluation_node,
    llm_data_analysis_node,
    llm_evaluation_node,
    llm_optimization_node,
    model_selection_node,
    optimization_node,
    persistence_node,
    preprocessing_node,
    training_node,
)
from graph.routing import route_after_evaluation
from graph.state import (
    WorkflowResult,
    extract_agent_state,
    make_graph_state,
)
from tools.mlflow_tools import MLflowTracker, create_tracker_from_config
from utils.config import Config
from utils.logger import get_logger
from utils.state import AgentState

logger = get_logger("GraphWorkflow")

# ---------------------------------------------------------------------------
# Node name constants (avoids string typos across modules)
# ---------------------------------------------------------------------------
_N_DATA_ANALYSIS = "data_analysis"
_N_LLM_DATA_ANALYSIS = "llm_data_analysis"
_N_PREPROCESSING = "preprocessing"
_N_MODEL_SELECTION = "model_selection"
_N_TRAINING = "training"
_N_EVALUATION = "evaluation"
_N_LLM_EVALUATION = "llm_evaluation"
_N_OPTIMIZATION = "optimization"
_N_LLM_OPTIMIZATION = "llm_optimization"
_N_PERSISTENCE = "model_persistence"

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_ml_workflow(
    enable_optimization: bool = True,
    max_retries: int = 1,
    enable_llm: bool = False,
):
    """Compile and return the LangGraph StateGraph for the ML pipeline.

    Args:
        enable_optimization: If False the optimization node is excluded
                             and routing ends at evaluation.
        max_retries: Maximum retries per node. Stored in GraphState so
                     each node can check it at runtime.
        enable_llm: Whether optional Phase-8 LLM nodes are active.

    Returns:
        Compiled LangGraph application (``CompiledStateGraph``).
    """
    graph = StateGraph(dict)

    # 1. Core deterministic nodes
    graph.add_node(_N_DATA_ANALYSIS, data_analysis_node)
    graph.add_node(_N_PREPROCESSING, preprocessing_node)
    graph.add_node(_N_MODEL_SELECTION, model_selection_node)
    graph.add_node(_N_TRAINING, training_node)
    graph.add_node(_N_EVALUATION, evaluation_node)
    graph.add_node(_N_OPTIMIZATION, optimization_node)
    graph.add_node(_N_PERSISTENCE, persistence_node)

    # 2. Optional LLM nodes
    if enable_llm:
        graph.add_node(_N_LLM_DATA_ANALYSIS, llm_data_analysis_node)
        graph.add_node(_N_LLM_EVALUATION, llm_evaluation_node)
        graph.add_node(_N_LLM_OPTIMIZATION, llm_optimization_node)

        # Edges with LLM
        graph.add_edge(START, _N_DATA_ANALYSIS)
        graph.add_conditional_edges(
            _N_DATA_ANALYSIS,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_LLM_DATA_ANALYSIS,
            {_N_LLM_DATA_ANALYSIS: _N_LLM_DATA_ANALYSIS, END: END},
        )
        graph.add_edge(_N_LLM_DATA_ANALYSIS, _N_PREPROCESSING)

        graph.add_conditional_edges(
            _N_PREPROCESSING,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_MODEL_SELECTION,
            {_N_MODEL_SELECTION: _N_MODEL_SELECTION, END: END},
        )
        graph.add_conditional_edges(
            _N_MODEL_SELECTION,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_TRAINING,
            {_N_TRAINING: _N_TRAINING, END: END},
        )
        graph.add_conditional_edges(
            _N_TRAINING,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_EVALUATION,
            {_N_EVALUATION: _N_EVALUATION, END: END},
        )
        graph.add_conditional_edges(
            _N_EVALUATION,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_LLM_EVALUATION,
            {_N_LLM_EVALUATION: _N_LLM_EVALUATION, END: END},
        )

        graph.add_conditional_edges(
            _N_LLM_EVALUATION,
            route_after_evaluation,
            {
                "optimize": _N_OPTIMIZATION,
                END: _N_PERSISTENCE,
            },
        )
        graph.add_edge(_N_OPTIMIZATION, _N_LLM_OPTIMIZATION)
        graph.add_edge(_N_LLM_OPTIMIZATION, _N_PERSISTENCE)
        graph.add_edge(_N_PERSISTENCE, END)

    else:
        # Standard deterministic edges without LLM
        graph.add_edge(START, _N_DATA_ANALYSIS)
        graph.add_conditional_edges(
            _N_DATA_ANALYSIS,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_PREPROCESSING,
            {_N_PREPROCESSING: _N_PREPROCESSING, END: END},
        )
        graph.add_conditional_edges(
            _N_PREPROCESSING,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_MODEL_SELECTION,
            {_N_MODEL_SELECTION: _N_MODEL_SELECTION, END: END},
        )
        graph.add_conditional_edges(
            _N_MODEL_SELECTION,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_TRAINING,
            {_N_TRAINING: _N_TRAINING, END: END},
        )
        graph.add_conditional_edges(
            _N_TRAINING,
            lambda state: END if (state.get("failed_node") or state.get("workflow_complete")) else _N_EVALUATION,
            {_N_EVALUATION: _N_EVALUATION, END: END},
        )

        graph.add_conditional_edges(
            _N_EVALUATION,
            route_after_evaluation,
            {
                "optimize": _N_OPTIMIZATION,
                END: _N_PERSISTENCE,
            },
        )
        graph.add_edge(_N_OPTIMIZATION, _N_PERSISTENCE)
        graph.add_edge(_N_PERSISTENCE, END)

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_graph(
    dataset_path: str,
    target_column: Optional[str] = None,
    enable_optimization: bool = True,
    max_retries: int = 1,
    enable_llm: bool = False,
    n_trials: Optional[int] = None,
    top_n: Optional[int] = None,
) -> WorkflowResult:
    """Run the full multi-agent ML workflow as a LangGraph graph.

    Args:
        dataset_path: Absolute or relative path to the input CSV.
        target_column: Explicit target column name (optional; auto-detected if None).
        enable_optimization: Whether the Optimization node should execute.
        max_retries: Maximum retries per node on transient failure.
        enable_llm: Whether optional Phase-8 LLM nodes are active.
        n_trials: Override for Optuna trials (uses Config default if None).
        top_n: Override for top-N candidates (uses Config default if None).

    Returns:
        WorkflowResult containing the final state and execution summary.
    """
    import time  # noqa: PLC0415

    logger.info("[GRAPH] Workflow starting — dataset: %s", dataset_path)

    # Phase 10: initialise MLflow tracker (no-op when MLFLOW_ENABLED=false)
    tracker = create_tracker_from_config()
    mlflow_run_id: Optional[str] = None
    workflow_elapsed: float = 0.0
    _start_time = time.monotonic()

    try:
        mlflow_run_id = tracker.start_workflow_run(
            dataset_path=dataset_path,
            enable_optimization=enable_optimization,
            enable_llm=enable_llm,
        )
        tracker.log_reproducibility()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MLflow] Pre-run logging failed (ignored): %s", exc)

    # 1. Bootstrap an initial AgentState
    initial_agent_state = AgentState(dataset_path=dataset_path)
    if target_column:
        initial_agent_state.target_column = target_column
    if enable_llm:
        initial_agent_state.llm_enabled = True

    # 2. Build the graph-level state dict
    graph_state = make_graph_state(
        agent_state=initial_agent_state,
        enable_optimization=enable_optimization,
        max_retries=max_retries,
        enable_llm=enable_llm,
    )

    # 3. Compile and run the graph
    app = build_ml_workflow(
        enable_optimization=enable_optimization,
        max_retries=max_retries,
        enable_llm=enable_llm,
    )

    ml_failed = False
    try:
        final_state: Dict[str, Any] = app.invoke(graph_state)
    except Exception:
        ml_failed = True
        try:
            tracker.end_run("FAILED")
        except Exception:  # noqa: BLE001
            pass
        raise

    # 5. Reconstruct the final AgentState
    final_agent_state = extract_agent_state(final_state)

    # Phase 10: post-run MLflow logging (all wrapped — never interrupts ML)
    try:
        workflow_elapsed = time.monotonic() - _start_time
        tracker.log_data_analysis(final_agent_state)
        tracker.log_preprocessing(final_agent_state)
        tracker.log_training_results(final_agent_state)
        tracker.log_evaluation(final_agent_state)
        tracker.log_optimization(final_agent_state)
        tracker.log_persistence(final_agent_state)
        tracker.log_workflow_summary(final_agent_state, workflow_elapsed)
        tracker.end_run("FINISHED" if not final_state.get("failed_node") else "FAILED")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[MLflow] Post-run logging failed (ignored): %s", exc)

    # 6. Determine optimization_executed flag
    opt_statuses = final_state.get("node_statuses", {})
    optimization_executed = _N_OPTIMIZATION in opt_statuses

    # 7. Build WorkflowResult
    result = WorkflowResult(
        workflow_status=str(final_agent_state.workflow_status.value)
        if hasattr(final_agent_state.workflow_status, "value")
        else str(final_agent_state.workflow_status),
        completed_nodes=list(final_state.get("completed_nodes", [])),
        failed_node=final_state.get("failed_node"),
        error_message=final_state.get("error_message"),
        retry_counts=dict(final_state.get("retry_counts", {})),
        optimization_executed=optimization_executed,
        optimization_enabled=enable_optimization,
        final_agent_state=final_agent_state,
        mlflow_run_id=mlflow_run_id,
    )

    if result.failed_node:
        logger.error(
            "[GRAPH] Workflow FAILED at node '%s': %s",
            result.failed_node,
            result.error_message,
        )
    else:
        logger.info(
            "[GRAPH] Workflow COMPLETED — nodes: %s | optimization: %s",
            result.completed_nodes,
            "executed" if optimization_executed else "skipped",
        )

    return result
