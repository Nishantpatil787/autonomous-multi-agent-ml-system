"""Graph-level state definitions for Phase-7 LangGraph orchestration.

Design principles:
- The existing AgentState (utils/state.py) is the authoritative ML state.
  It is carried through the graph inside GraphState.agent_state as a plain
  dict (via AgentState.to_dict()) so that all graph state remains JSON-safe.
- GraphState is a TypedDict — required by LangGraph's StateGraph.
- WorkflowResult is a plain dataclass used to expose results to callers.
- No live sklearn or Optuna objects are stored here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.state import AgentState


class NodeStatus(str, Enum):
    """Execution status of a single graph node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


# ---------------------------------------------------------------------------
# LangGraph state — TypedDict so LangGraph can merge / pass it between nodes
# ---------------------------------------------------------------------------

# We use a plain dict (not TypedDict) to stay compatible across Python 3.10+
# and to allow dynamic key merging in LangGraph 0.2.x without annotation issues.
# All fields documented below.

def make_graph_state(
    agent_state: AgentState,
    enable_optimization: bool = True,
    max_retries: int = 1,
    enable_llm: bool = False,
) -> Dict[str, Any]:
    """Create the initial GraphState dict from a bootstrapped AgentState.

    Args:
        agent_state: The pre-populated AgentState (must have dataset_path set).
        enable_optimization: Whether Phase-6 optimization should run.
        max_retries: Maximum retries per node on transient failure.
        enable_llm: Whether optional Phase-8 LLM nodes should execute.

    Returns:
        A plain dict suitable for passing to ``compiled_graph.invoke()``.
    """
    return {
        "agent_state_dict": agent_state.to_dict(),
        "current_node": "",
        "node_statuses": {},
        "completed_nodes": [],
        "failed_node": None,
        "error_message": None,
        "retry_counts": {},
        "max_retries": max_retries,
        "enable_optimization": enable_optimization,
        "enable_llm": enable_llm,
        "optimization_done": False,
        "workflow_complete": False,
    }


def extract_agent_state(graph_state: Dict[str, Any]) -> AgentState:
    """Reconstruct AgentState from the graph state dict.

    Args:
        graph_state: The current LangGraph state dict.

    Returns:
        Reconstructed AgentState instance.
    """
    return AgentState.from_dict(dict(graph_state["agent_state_dict"]))


# ---------------------------------------------------------------------------
# WorkflowResult — returned to callers of run_graph()
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Summary of a completed graph execution."""

    workflow_status: str
    completed_nodes: List[str] = field(default_factory=list)
    failed_node: Optional[str] = None
    error_message: Optional[str] = None
    retry_counts: Dict[str, int] = field(default_factory=dict)
    optimization_executed: bool = False
    optimization_enabled: bool = True
    final_agent_state: Optional[AgentState] = None
    mlflow_run_id: Optional[str] = None  # Phase 10: MLflow tracking

    # Convenience accessors ------------------------------------------------

    @property
    def success(self) -> bool:
        """True if the workflow completed without a fatal failure."""
        return self.failed_node is None and self.workflow_status not in ("failed",)

    @property
    def best_baseline_model(self) -> Optional[str]:
        if self.final_agent_state:
            return self.final_agent_state.best_baseline_model
        return None

    @property
    def best_optimized_model(self) -> Optional[str]:
        if self.final_agent_state:
            return getattr(self.final_agent_state, "best_optimized_model", None)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary dict."""
        return {
            "workflow_status": self.workflow_status,
            "completed_nodes": self.completed_nodes,
            "failed_node": self.failed_node,
            "error_message": self.error_message,
            "retry_counts": self.retry_counts,
            "optimization_executed": self.optimization_executed,
            "optimization_enabled": self.optimization_enabled,
            "best_baseline_model": self.best_baseline_model,
            "best_optimized_model": self.best_optimized_model,
            "mlflow_run_id": self.mlflow_run_id,
        }
