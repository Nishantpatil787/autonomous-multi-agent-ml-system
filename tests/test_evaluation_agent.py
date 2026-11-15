"""Tests for Evaluation Agent and end-to-end Phase 5 multi-agent flow."""

import json
import unittest
from pathlib import Path
from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from utils.state import AgentState, WorkflowStatus

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
SAMPLE_BIN_CSV = DATASETS_DIR / "sample_classification.csv"
SAMPLE_MULTI_CSV = DATASETS_DIR / "sample_multiclass.csv"


class TestEvaluationAgent(unittest.TestCase):
    """Unit and integration tests for agents/evaluation_agent.py."""

    def test_01_evaluation_prerequisites_missing_results(self) -> None:
        """Verify that EvaluationAgent fails gracefully if no training experiment results exist."""
        empty_state = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        agent = EvaluationAgent()
        res_state = agent.run(empty_state)
        self.assertEqual(res_state.workflow_status, WorkflowStatus.FAILED)
        self.assertEqual(res_state.evaluation_status, "failed")
        self.assertIn("No experiment results found", res_state.error_message)

    def test_02_evaluation_agent_end_to_end_binary(self) -> None:
        """Verify complete Phase 1 -> Phase 5 workflow on binary classification dataset."""
        state = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)
        state = EvaluationAgent().run(state)

        self.assertEqual(state.workflow_status, WorkflowStatus.EVALUATION_COMPLETED)
        self.assertEqual(state.evaluation_status, "completed")
        self.assertIsNotNone(state.best_baseline_model)

        report = state.evaluation_report
        self.assertIn("best_model", report)
        self.assertIn("selection_reason", report)
        self.assertIn("gap_analysis", report)
        self.assertIn("confusion_matrix_analyses", report)
        self.assertIn("execution_time_analysis", report)
        self.assertIn("observations", report)
        self.assertIn("limitations", report)
        self.assertIn("recommendations", report)

        # Gap analysis checks
        self.assertEqual(len(state.gap_analyses), 5)
        for gap in state.gap_analyses:
            self.assertIn("f1_classification", gap)
            self.assertIn("interpretation", gap)
            self.assertIn(gap["f1_classification"], ["close agreement", "moderate difference", "substantial difference", "not applicable"])

        # Confusion matrix analysis checks
        self.assertEqual(len(state.confusion_matrix_analyses), 5)
        for model_name, cm_diag in state.confusion_matrix_analyses.items():
            self.assertTrue(cm_diag["is_binary"])
            self.assertIn("dominant_error_pattern", cm_diag)
            self.assertEqual(cm_diag["total_samples"], 12)

    def test_03_evaluation_agent_end_to_end_multiclass(self) -> None:
        """Verify complete Phase 1 -> Phase 5 workflow on multiclass dataset."""
        state = AgentState(dataset_path=str(SAMPLE_MULTI_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)
        state = EvaluationAgent().run(state)

        self.assertEqual(state.workflow_status, WorkflowStatus.EVALUATION_COMPLETED)
        self.assertEqual(state.evaluation_status, "completed")
        self.assertIsNotNone(state.best_baseline_model)

        # Multiclass confusion matrix checks
        first_model = state.selected_models[0]
        cm_diag = state.confusion_matrix_analyses[first_model]
        self.assertFalse(cm_diag["is_binary"])
        self.assertIn("class_error_counts", cm_diag)
        self.assertIn("dominant_confusion_pairs", cm_diag)

    def test_04_json_safety_of_evaluation_report(self) -> None:
        """Verify that AgentState containing the complete EvaluationReport is 100% JSON-serializable."""
        state = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)
        state = EvaluationAgent().run(state)

        state_dict = state.to_dict()
        json_output = json.dumps(state_dict)
        self.assertIsInstance(json_output, str)
        self.assertTrue(len(json_output) > 0)

        # Round-trip deserialization
        restored = AgentState.from_dict(state_dict)
        self.assertEqual(restored.workflow_status, WorkflowStatus.EVALUATION_COMPLETED)
        self.assertEqual(restored.best_baseline_model, state.best_baseline_model)
        self.assertEqual(restored.evaluation_report["best_model"], state.evaluation_report["best_model"])


if __name__ == "__main__":
    unittest.main()
