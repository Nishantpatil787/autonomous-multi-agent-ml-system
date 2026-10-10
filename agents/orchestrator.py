"""Orchestrator Agent - Controls workflow and maintains shared state."""

from agents.base_agent import BaseAgent
from utils.state import AgentState


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent responsible for workflow coordination."""

    def __init__(self) -> None:
        """Initialize orchestrator agent."""
        super().__init__(name="OrchestratorAgent")

    def run(self, state: AgentState) -> AgentState:
        """Execute orchestration logic.

        Args:
            state: Current shared state

        Returns:
            Updated shared state
        """
        self.log_message(state, "Orchestrator initialized. Workflow coordination ready.")
        return state
