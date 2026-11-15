"""Tests for Training Agent and end-to-end Phase 4 multi-agent flow."""

import json
import unittest
from pathlib import Path
from agents.data_analysis_agent import DataAnalysisAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from utils.state import AgentState, WorkflowStatus

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
SAMPLE_BIN_CSV = DATASETS_DIR / "sample_classification.csv"
SAMPLE_MULTI_CSV = DATASETS_DIR / "sample_multiclass.csv"
SAMPLE_EDGE_CSV = DATASETS_DIR / "sample_edge_cases.csv"


class TestTrainingAgent(unittest.TestCase):
    """Unit and integration tests for agents/training_agent.py."""

    def test_01_training_agent_end_to_end_binary(self) -> None:
        """Verify complete Phase 1 -> Phase 4 workflow on binary dataset."""
        state = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)

        self.assertEqual(state.workflow_status, WorkflowStatus.TRAINING_COMPLETED)
        self.assertEqual(state.experiment_count, 5)
        self.assertEqual(len(state.experiment_results), 5)
        self.assertEqual(len(state.model_comparison), 5)
        self.assertIsNotNone(state.best_baseline_model)
        self.assertIn("f1_score", state.best_baseline_metrics)
        self.assertIn("accuracy", state.best_baseline_metrics)

        # Check each experiment record
        for exp in state.experiment_results:
            self.assertEqual(exp["status"], "success")
            self.assertIn("accuracy", exp)
            self.assertIn("f1_score", exp)
            self.assertIn("confusion_matrix", exp)
            self.assertTrue(len(exp["confusion_matrix"]) > 0)
            self.assertIn("cv_f1_mean", exp)

    def test_02_training_agent_end_to_end_multiclass(self) -> None:
        """Verify complete Phase 1 -> Phase 4 workflow on multiclass dataset."""
        state = AgentState(dataset_path=str(SAMPLE_MULTI_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)

        self.assertEqual(state.workflow_status, WorkflowStatus.TRAINING_COMPLETED)
        self.assertEqual(state.experiment_count, 5)
        self.assertIsNotNone(state.best_baseline_model)

        # Multiclass confusion matrix should have 3 classes
        first_exp = state.experiment_results[0]
        self.assertEqual(len(first_exp["confusion_matrix"]), 3)
        self.assertEqual(len(first_exp["confusion_matrix_labels"]), 3)

    def test_03_missing_prerequisites_fails_gracefully(self) -> None:
        """Verify that training fails gracefully with clear error when prerequisites are absent."""
        agent = TrainingAgent()

        # Missing dataset path
        state_no_data = AgentState()
        res1 = agent.run(state_no_data)
        self.assertEqual(res1.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("dataset_path is missing", res1.error_message)

        # Missing target column
        state_no_target = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        res2 = agent.run(state_no_target)
        self.assertEqual(res2.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("target_column is missing", res2.error_message)

        # Missing selected models
        state_no_models = AgentState(dataset_path=str(SAMPLE_BIN_CSV), target_column="Outcome")
        res3 = agent.run(state_no_models)
        self.assertEqual(res3.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("No models selected", res3.error_message)

    def test_04_json_safety_of_experiment_records(self) -> None:
        """Verify that AgentState with all training experiment records is 100% JSON-serializable."""
        state = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)

        # Convert to dictionary and verify json serialization
        state_dict = state.to_dict()
        json_str = json.dumps(state_dict)
        self.assertIsInstance(json_str, str)
        self.assertTrue(len(json_str) > 0)

        # Verify round-trip deserialization
        restored_state = AgentState.from_dict(state_dict)
        self.assertEqual(restored_state.workflow_status, WorkflowStatus.TRAINING_COMPLETED)
        self.assertEqual(restored_state.best_baseline_model, state.best_baseline_model)

    def test_05_reproducibility(self) -> None:
        """Verify that running the training agent twice produces identical metrics within tolerance."""
        state1 = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        state1 = DataAnalysisAgent().run(state1)
        state1 = PreprocessingAgent().run(state1)
        state1 = ModelSelectionAgent().run(state1)
        state1 = TrainingAgent().run(state1)

        state2 = AgentState(dataset_path=str(SAMPLE_BIN_CSV))
        state2 = DataAnalysisAgent().run(state2)
        state2 = PreprocessingAgent().run(state2)
        state2 = ModelSelectionAgent().run(state2)
        state2 = TrainingAgent().run(state2)

        self.assertEqual(state1.best_baseline_model, state2.best_baseline_model)
        for exp1, exp2 in zip(state1.experiment_results, state2.experiment_results):
            self.assertEqual(exp1["model_name"], exp2["model_name"])
            self.assertAlmostEqual(exp1["accuracy"], exp2["accuracy"], places=4)
            self.assertAlmostEqual(exp1["f1_score"], exp2["f1_score"], places=4)

    def test_06_edge_cases_excluded_from_model(self) -> None:
        """Verify that excluded columns (record_id, etc.) are never part of the fitted model."""
        state = AgentState(dataset_path=str(SAMPLE_EDGE_CSV))
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        state = ModelSelectionAgent().run(state)
        state = TrainingAgent().run(state)

        self.assertEqual(state.workflow_status, WorkflowStatus.TRAINING_COMPLETED)
        self.assertIn("record_id", state.excluded_features)
        self.assertIn("constant_feature", state.excluded_features)
        self.assertEqual(state.experiment_count, 5)


if __name__ == "__main__":
    unittest.main()
