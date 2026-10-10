"""Specialized agents of the Autonomous Multi-Agent ML System.

Phase 9 adds the seventh agent. It runs after Evaluation (or after Optimization,
when that runs) and is the only agent whose product is a set of files rather than
shared state. It cannot change any earlier result: it selects a model using the
ranking rule the earlier phases already use, copies their metrics unchanged, and
writes them out.
"""

__version__ = "0.8.0"

from agents.base_agent import BaseAgent
from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent

__all__ = [
    "BaseAgent",
    "DataAnalysisAgent",
    "PreprocessingAgent",
    "ModelSelectionAgent",
    "TrainingAgent",
    "EvaluationAgent",
    "OptimizationAgent",
    "PersistenceAgent",
]