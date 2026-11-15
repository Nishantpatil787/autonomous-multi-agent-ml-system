"""Graph node implementations for Phase-7 LangGraph orchestration.

Each node:
1. Extracts AgentState from graph state.
2. Delegates to exactly one existing agent.
3. Updates graph state fields (status, logs, errors).
4. Returns the updated graph state dict.

No ML logic is implemented here — that belongs to the existing agents.
"""

from typing import Any, Dict

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from graph.state import NodeStatus, extract_agent_state
from utils.config import Config
from utils.logger import get_logger
from utils.state import WorkflowStatus

logger = get_logger("GraphNodes")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NODE_LOG_PREFIX = "[GRAPH]"


def _node_started(graph_state: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    """Record that a node has started; returns updated graph state fields."""
    logger.info(f"{_NODE_LOG_PREFIX} {node_name} -> STARTED")
    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[node_name] = NodeStatus.RUNNING.value
    return {
        **graph_state,
        "current_node": node_name,
        "node_statuses": statuses,
    }


def _node_completed(
    graph_state: Dict[str, Any],
    node_name: str,
    updated_agent_state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Record successful node completion; returns updated graph state."""
    logger.info(f"{_NODE_LOG_PREFIX} {node_name} -> COMPLETED")
    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[node_name] = NodeStatus.COMPLETED.value
    completed = list(graph_state.get("completed_nodes", [])) + [node_name]
    return {
        **graph_state,
        "agent_state_dict": updated_agent_state_dict,
        "node_statuses": statuses,
        "completed_nodes": completed,
        "current_node": node_name,
        "failed_node": None,
        "error_message": None,
    }


def _node_failed(
    graph_state: Dict[str, Any],
    node_name: str,
    error_message: str,
    updated_agent_state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Record node failure; returns updated graph state."""
    logger.error(f"{_NODE_LOG_PREFIX} {node_name} -> FAILED: {error_message}")
    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[node_name] = NodeStatus.FAILED.value
    return {
        **graph_state,
        "agent_state_dict": updated_agent_state_dict,
        "node_statuses": statuses,
        "current_node": node_name,
        "failed_node": node_name,
        "error_message": error_message,
        "workflow_complete": True,
    }


def _increment_retry(graph_state: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    """Increment retry counter for node; returns updated graph state."""
    retry_counts = dict(graph_state.get("retry_counts", {}))
    current = retry_counts.get(node_name, 0)
    retry_counts[node_name] = current + 1
    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[node_name] = NodeStatus.RETRYING.value
    logger.warning(
        f"{_NODE_LOG_PREFIX} {node_name} -> RETRY {current + 1}/{graph_state.get('max_retries', 1)}"
    )
    return {
        **graph_state,
        "retry_counts": retry_counts,
        "node_statuses": statuses,
    }


def _should_retry(graph_state: Dict[str, Any], node_name: str) -> bool:
    """Return True if this node has remaining retries."""
    max_r = graph_state.get("max_retries", 1)
    used = graph_state.get("retry_counts", {}).get(node_name, 0)
    return used < max_r


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def data_analysis_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke DataAnalysisAgent."""
    NODE = "data_analysis"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = DataAnalysisAgent().run(agent_state)
    except Exception as exc:
        error = f"DataAnalysisAgent raised an exception: {exc}"
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            try:
                agent_state = extract_agent_state(graph_state)
                agent_state = DataAnalysisAgent().run(agent_state)
            except Exception as exc2:
                return _node_failed(graph_state, NODE, str(exc2), agent_state.to_dict())
        else:
            return _node_failed(graph_state, NODE, error, agent_state.to_dict())

    if agent_state.workflow_status == WorkflowStatus.FAILED:
        err = agent_state.error_message or "DataAnalysisAgent failed."
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            fresh_state = extract_agent_state(graph_state)
            fresh_state = DataAnalysisAgent().run(fresh_state)
            if fresh_state.workflow_status != WorkflowStatus.ANALYSIS_COMPLETED:
                return _node_failed(graph_state, NODE, fresh_state.error_message or err, fresh_state.to_dict())
            agent_state = fresh_state
        else:
            return _node_failed(graph_state, NODE, err, agent_state.to_dict())

    return _node_completed(graph_state, NODE, agent_state.to_dict())


def preprocessing_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke PreprocessingAgent."""
    NODE = "preprocessing"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = PreprocessingAgent().run(agent_state)
    except Exception as exc:
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            try:
                agent_state = extract_agent_state(graph_state)
                agent_state = PreprocessingAgent().run(agent_state)
            except Exception as exc2:
                return _node_failed(graph_state, NODE, str(exc2), agent_state.to_dict())
        else:
            return _node_failed(graph_state, NODE, str(exc), agent_state.to_dict())

    if agent_state.workflow_status == WorkflowStatus.FAILED:
        err = agent_state.error_message or "PreprocessingAgent failed."
        return _node_failed(graph_state, NODE, err, agent_state.to_dict())

    return _node_completed(graph_state, NODE, agent_state.to_dict())


def model_selection_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke ModelSelectionAgent."""
    NODE = "model_selection"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = ModelSelectionAgent().run(agent_state)
    except Exception as exc:
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            try:
                agent_state = extract_agent_state(graph_state)
                agent_state = ModelSelectionAgent().run(agent_state)
            except Exception as exc2:
                return _node_failed(graph_state, NODE, str(exc2), agent_state.to_dict())
        else:
            return _node_failed(graph_state, NODE, str(exc), agent_state.to_dict())

    if agent_state.workflow_status == WorkflowStatus.FAILED:
        err = agent_state.error_message or "ModelSelectionAgent failed."
        return _node_failed(graph_state, NODE, err, agent_state.to_dict())

    return _node_completed(graph_state, NODE, agent_state.to_dict())


def training_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke TrainingAgent."""
    NODE = "training"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = TrainingAgent().run(agent_state)
    except Exception as exc:
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            try:
                agent_state = extract_agent_state(graph_state)
                agent_state = TrainingAgent().run(agent_state)
            except Exception as exc2:
                return _node_failed(graph_state, NODE, str(exc2), agent_state.to_dict())
        else:
            return _node_failed(graph_state, NODE, str(exc), agent_state.to_dict())

    if agent_state.workflow_status == WorkflowStatus.FAILED:
        err = agent_state.error_message or "TrainingAgent failed."
        return _node_failed(graph_state, NODE, err, agent_state.to_dict())

    return _node_completed(graph_state, NODE, agent_state.to_dict())


def evaluation_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke EvaluationAgent."""
    NODE = "evaluation"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = EvaluationAgent().run(agent_state)
    except Exception as exc:
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            try:
                agent_state = extract_agent_state(graph_state)
                agent_state = EvaluationAgent().run(agent_state)
            except Exception as exc2:
                return _node_failed(graph_state, NODE, str(exc2), agent_state.to_dict())
        else:
            return _node_failed(graph_state, NODE, str(exc), agent_state.to_dict())

    if agent_state.workflow_status == WorkflowStatus.FAILED:
        err = agent_state.error_message or "EvaluationAgent failed."
        return _node_failed(graph_state, NODE, err, agent_state.to_dict())

    return _node_completed(graph_state, NODE, agent_state.to_dict())


def optimization_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke OptimizationAgent.

    If optimization fails, the Phase-5 evaluation results are preserved.
    The workflow does NOT fail terminally — it completes with an
    optimization_status of 'failed' but all baseline results intact.
    """
    NODE = "optimization"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = OptimizationAgent().run(
            agent_state,
            n_trials=Config.OPTUNA_N_TRIALS,
            top_n=Config.OPTUNA_TOP_MODELS,
        )
    except Exception as exc:
        if _should_retry(graph_state, NODE):
            graph_state = _increment_retry(graph_state, NODE)
            try:
                fresh = extract_agent_state(graph_state)
                fresh = OptimizationAgent().run(
                    fresh,
                    n_trials=Config.OPTUNA_N_TRIALS,
                    top_n=Config.OPTUNA_TOP_MODELS,
                )
                agent_state = fresh
            except Exception as exc2:
                logger.warning(
                    f"{_NODE_LOG_PREFIX} {NODE} failed after retry ({exc2}). "
                    "Preserving Phase-5 evaluation result."
                )
                # Mark optimization done to prevent looping; preserve evaluation
                return {
                    **graph_state,
                    "agent_state_dict": agent_state.to_dict(),
                    "node_statuses": {
                        **graph_state.get("node_statuses", {}),
                        NODE: NodeStatus.FAILED.value,
                    },
                    "completed_nodes": list(graph_state.get("completed_nodes", [])) + [NODE],
                    "optimization_done": True,
                    "workflow_complete": True,
                }
        else:
            logger.warning(
                f"{_NODE_LOG_PREFIX} {NODE} failed ({exc}). "
                "Preserving Phase-5 evaluation result."
            )
            return {
                **graph_state,
                "agent_state_dict": agent_state.to_dict(),
                "node_statuses": {
                    **graph_state.get("node_statuses", {}),
                    NODE: NodeStatus.FAILED.value,
                },
                "completed_nodes": list(graph_state.get("completed_nodes", [])) + [NODE],
                "optimization_done": True,
                "workflow_complete": True,
            }

    if agent_state.workflow_status == WorkflowStatus.FAILED:
        # Optimization agent itself reported failure — preserve eval results
        logger.warning(
            f"{_NODE_LOG_PREFIX} {NODE} -> optimization failed, Phase-5 results preserved."
        )
        statuses = dict(graph_state.get("node_statuses", {}))
        statuses[NODE] = NodeStatus.FAILED.value
        completed = list(graph_state.get("completed_nodes", [])) + [NODE]
        return {
            **graph_state,
            "agent_state_dict": agent_state.to_dict(),
            "node_statuses": statuses,
            "completed_nodes": completed,
            "optimization_done": True,
            "workflow_complete": True,
        }

    logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> COMPLETED")
    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[NODE] = NodeStatus.COMPLETED.value
    completed = list(graph_state.get("completed_nodes", [])) + [NODE]
    return {
        **graph_state,
        "agent_state_dict": agent_state.to_dict(),
        "node_statuses": statuses,
        "completed_nodes": completed,
        "current_node": NODE,
        "optimization_done": True,
        "workflow_complete": True,
    }


def persistence_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: invoke PersistenceAgent to select and persist final pipeline."""
    NODE = "model_persistence"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    try:
        agent_state = PersistenceAgent().run(agent_state)
    except Exception as exc:
        logger.error(f"{_NODE_LOG_PREFIX} {NODE} -> failed with exception: {exc}")
        agent_state.persistence_status = "failed"
        agent_state.error_message = str(exc)

    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[NODE] = (
        NodeStatus.COMPLETED.value
        if agent_state.persistence_status == "completed"
        else NodeStatus.FAILED.value
    )
    completed = list(graph_state.get("completed_nodes", [])) + [NODE]
    logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> COMPLETED (status: {agent_state.persistence_status})")
    return {
        **graph_state,
        "agent_state_dict": agent_state.to_dict(),
        "node_statuses": statuses,
        "completed_nodes": completed,
        "current_node": NODE,
        "workflow_complete": True,
    }


# ---------------------------------------------------------------------------
# Phase 8: Optional Local LLM Interpretation Nodes
# ---------------------------------------------------------------------------

def llm_data_analysis_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """Optional node: invoke Local LLM for Phase 2 data analysis explanation."""
    NODE = "llm_data_analysis"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    if not graph_state.get("enable_llm", False) and not Config.OLLAMA_ENABLED:
        logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> SKIPPED (LLM disabled)")
        statuses = dict(graph_state.get("node_statuses", {}))
        statuses[NODE] = NodeStatus.SKIPPED.value
        return {**graph_state, "node_statuses": statuses}

    try:
        from llm.service import LLMService
        service = LLMService()
        agent_state.llm_enabled = True
        explanation = service.generate_data_analysis_explanation(agent_state)
        agent_state.data_analysis_explanation = explanation
        agent_state.llm_status = explanation.get("status", "unknown")
        logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> COMPLETED (status: {agent_state.llm_status})")
    except Exception as exc:
        logger.warning(f"{_NODE_LOG_PREFIX} {NODE} -> non-fatal error: {exc}. Continuing ML pipeline.")
        agent_state.llm_status = "unavailable"
        agent_state.data_analysis_explanation = {"status": "unavailable", "error": str(exc)}

    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[NODE] = NodeStatus.COMPLETED.value
    completed = list(graph_state.get("completed_nodes", [])) + [NODE]
    return {
        **graph_state,
        "agent_state_dict": agent_state.to_dict(),
        "node_statuses": statuses,
        "completed_nodes": completed,
        "current_node": NODE,
    }


def llm_evaluation_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """Optional node: invoke Local LLM for Phase 5 evaluation explanation."""
    NODE = "llm_evaluation"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    if not graph_state.get("enable_llm", False) and not Config.OLLAMA_ENABLED:
        logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> SKIPPED (LLM disabled)")
        statuses = dict(graph_state.get("node_statuses", {}))
        statuses[NODE] = NodeStatus.SKIPPED.value
        return {**graph_state, "node_statuses": statuses}

    try:
        from llm.service import LLMService
        service = LLMService()
        agent_state.llm_enabled = True
        explanation = service.generate_evaluation_explanation(agent_state)
        agent_state.evaluation_explanation = explanation
        agent_state.llm_status = explanation.get("status", "unknown")
        logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> COMPLETED (status: {agent_state.llm_status})")
    except Exception as exc:
        logger.warning(f"{_NODE_LOG_PREFIX} {NODE} -> non-fatal error: {exc}. Continuing ML pipeline.")
        agent_state.evaluation_explanation = {"status": "unavailable", "error": str(exc)}

    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[NODE] = NodeStatus.COMPLETED.value
    completed = list(graph_state.get("completed_nodes", [])) + [NODE]
    return {
        **graph_state,
        "agent_state_dict": agent_state.to_dict(),
        "node_statuses": statuses,
        "completed_nodes": completed,
        "current_node": NODE,
    }


def llm_optimization_node(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """Optional node: invoke Local LLM for Phase 6 hyperparameter tuning explanation."""
    NODE = "llm_optimization"
    graph_state = _node_started(graph_state, NODE)
    agent_state = extract_agent_state(graph_state)

    if not graph_state.get("enable_llm", False) and not Config.OLLAMA_ENABLED:
        logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> SKIPPED (LLM disabled)")
        statuses = dict(graph_state.get("node_statuses", {}))
        statuses[NODE] = NodeStatus.SKIPPED.value
        return {**graph_state, "node_statuses": statuses}

    try:
        from llm.service import LLMService
        service = LLMService()
        agent_state.llm_enabled = True
        explanation = service.generate_optimization_explanation(agent_state)
        agent_state.optimization_explanation = explanation
        logger.info(f"{_NODE_LOG_PREFIX} {NODE} -> COMPLETED (status: {explanation.get('status', 'unknown')})")
    except Exception as exc:
        logger.warning(f"{_NODE_LOG_PREFIX} {NODE} -> non-fatal error: {exc}. Continuing ML pipeline.")
        agent_state.optimization_explanation = {"status": "unavailable", "error": str(exc)}

    statuses = dict(graph_state.get("node_statuses", {}))
    statuses[NODE] = NodeStatus.COMPLETED.value
    completed = list(graph_state.get("completed_nodes", [])) + [NODE]
    return {
        **graph_state,
        "agent_state_dict": agent_state.to_dict(),
        "node_statuses": statuses,
        "completed_nodes": completed,
        "current_node": NODE,
    }
