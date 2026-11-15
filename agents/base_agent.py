"""Base agent interface for the Autonomous Multi-Agent Machine Learning System."""

from abc import ABC, abstractmethod
from typing import Any
from utils.logger import get_logger
from utils.state import AgentState, WorkflowStatus


class BaseAgent(ABC):
    """Abstract base class for all specialized workflow agents."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(name)

    @abstractmethod
    def run(self, state: AgentState, **kwargs: Any) -> AgentState:
        """Execute the agent's core capability on the shared state.

        Args:
            state: The shared workflow state.
            **kwargs: Optional runtime overrides (e.g. target_column).

        Returns:
            Updated AgentState instance.
        """
        pass
