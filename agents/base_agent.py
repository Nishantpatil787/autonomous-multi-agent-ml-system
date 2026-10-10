"""Base agent class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Any

from utils.logger import setup_logger
from utils.state import AgentState


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str) -> None:
        """Initialize base agent.

        Args:
            name: Name of the agent
        """
        self.name = name
        self.logger = setup_logger(f"agent.{name}")

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """Execute agent logic on shared state.

        Args:
            state: Current shared state

        Returns:
            Updated shared state
        """
        pass

    def log_message(self, state: AgentState, message: str, level: str = "info") -> None:
        """Log message to both logger and state."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)

        state.add_message(self.name, message, level)
