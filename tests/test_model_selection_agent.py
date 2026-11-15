"""Tests for Model Selection Agent."""

import unittest
from pathlib import Path
from agents.data_analysis_agent import DataAnalysisAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from utils.state import AgentState, WorkflowStatus

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
SAMPLE_CSV = DATASETS_DIR / "sample_classification.csv"


class TestModelSelectionAgent(unittest.TestCase):
    """Unit tests for agents/model_selection_agent.py."""

    def setUp(self) -> None:
        """Run Phase 2 and Phase 3 on sample dataset to establish valid state."""
        state = AgentState(dataset_path=str(SAMPLE_CSV))
        state = DataAnalysisAgent().run(state)
        self.state = PreprocessingAgent().run(state)
        self.agent = ModelSelectionAgent()

    def test_01_successful_model_selection(self) -> None:
        """Verify standard model selection selects all 5 supported models."""
        res_state = self.agent.run(self.state)
        self.assertEqual(res_state.workflow_status, WorkflowStatus.MODELS_SELECTED)
        self.assertEqual(len(res_state.selected_models), 5)
        self.assertIn("logistic_regression", res_state.selected_models)
        self.assertIn("random_forest", res_state.selected_models)
        self.assertEqual(len(res_state.model_selection_reasoning), 5)
        self.assertEqual(len(res_state.selected_model_configs), 5)
        self.assertIn("total_candidates", res_state.model_selection_summary)

    def test_02_explicit_candidate_override(self) -> None:
        """Verify that passing explicit candidate models filters correctly."""
        res_state = self.agent.run(
            self.state,
            candidate_models=["random_forest", "logistic_regression"],
        )
        self.assertEqual(res_state.selected_models, ["random_forest", "logistic_regression"])
        self.assertEqual(len(res_state.selected_model_configs), 2)

    def test_03_missing_target_fails(self) -> None:
        """Verify that missing target column fails gracefully."""
        invalid_state = AgentState(dataset_path=str(SAMPLE_CSV), problem_type="binary_classification")
        res_state = self.agent.run(invalid_state)
        self.assertEqual(res_state.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("Target column not identified", res_state.error_message)

    def test_04_unsupported_problem_type_fails(self) -> None:
        """Verify that unsupported problem types (e.g. regression) fail gracefully."""
        self.state.problem_type = "regression"
        res_state = self.agent.run(self.state)
        self.assertEqual(res_state.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("unsupported", res_state.error_message.lower())

    def test_05_missing_preprocessing_fails(self) -> None:
        """Verify that missing preprocessing configuration fails gracefully."""
        raw_state = AgentState(
            dataset_path=str(SAMPLE_CSV),
            target_column="Outcome",
            problem_type="binary_classification",
        )
        res_state = self.agent.run(raw_state)
        self.assertEqual(res_state.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("Preprocessing pipeline configuration missing", res_state.error_message)


if __name__ == "__main__":
    unittest.main()
