"""Graph orchestration package for Autonomous Multi-Agent Machine Learning System.

Phase 7 — LangGraph orchestration layer.

The existing agents (DataAnalysisAgent, PreprocessingAgent, ModelSelectionAgent,
TrainingAgent, EvaluationAgent, OptimizationAgent) are orchestrated by a
LangGraph StateGraph.  Their internal logic is NOT duplicated here.

Exports:
    - make_graph_state   : Factory that creates a graph state dict
    - extract_agent_state: Reconstruct AgentState from graph state
    - NodeStatus         : Enum of node execution statuses
    - WorkflowResult     : Dataclass summarising a completed graph run
    - build_ml_workflow  : Factory that compiles the LangGraph StateGraph
    - run_graph          : High-level runner that returns a WorkflowResult
"""

from graph.state import NodeStatus, WorkflowResult, make_graph_state, extract_agent_state
from graph.workflow import build_ml_workflow, run_graph

__all__ = [
    "NodeStatus",
    "WorkflowResult",
    "make_graph_state",
    "extract_agent_state",
    "build_ml_workflow",
    "run_graph",
]
