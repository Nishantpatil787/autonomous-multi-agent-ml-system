"""Tests for OptimizationAgent (Phase 6).

Covers: prerequisite handling, state updates, source CSV integrity,
binary integration, multiclass integration, and repeat-run reproducibility.
"""

import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from utils.config import Config
from utils.state import AgentState, WorkflowStatus

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BINARY_CSV = Config.DATASETS_DIR / "sample_classification.csv"
MULTICLASS_CSV = Config.DATASETS_DIR / "sample_multiclass.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _run_pipeline_through_evaluation(csv_path: Path, n_trials: int = 3) -> AgentState:
    """Run all agents through Phase 5 evaluation and return state."""
    state = AgentState(dataset_path=str(csv_path))
    state = DataAnalysisAgent().run(state)
    state = PreprocessingAgent().run(state)
    state = ModelSelectionAgent().run(state)
    state = TrainingAgent().run(state)
    state = EvaluationAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# 1. Prerequisite checks
# ---------------------------------------------------------------------------

class TestOptimizationAgentPrerequisites:
    def test_missing_dataset_path_causes_failure(self):
        state = AgentState()
        state.workflow_status = WorkflowStatus.EVALUATION_COMPLETED
        result = OptimizationAgent().run(state)
        assert result.workflow_status == WorkflowStatus.FAILED
        assert result.error_message is not None

    def test_missing_target_column_causes_failure(self):
        state = AgentState(dataset_path=str(BINARY_CSV))
        state.workflow_status = WorkflowStatus.EVALUATION_COMPLETED
        # No target_column set
        result = OptimizationAgent().run(state)
        assert result.workflow_status == WorkflowStatus.FAILED

    def test_wrong_workflow_status_causes_failure(self):
        state = AgentState(dataset_path=str(BINARY_CSV))
        state.target_column = "Outcome"
        state.pipeline_config = {"numerical_features": ["Age"], "categorical_features": []}
        state.model_evaluations = [{"model_name": "rf", "rank": 1}]
        state.workflow_status = WorkflowStatus.TRAINING_COMPLETED  # Wrong status
        result = OptimizationAgent().run(state)
        assert result.workflow_status == WorkflowStatus.FAILED

    def test_empty_model_evaluations_causes_failure(self):
        state = AgentState(dataset_path=str(BINARY_CSV))
        state.target_column = "Outcome"
        state.pipeline_config = {"numerical_features": ["Age"], "categorical_features": []}
        state.workflow_status = WorkflowStatus.EVALUATION_COMPLETED
        state.model_evaluations = []
        result = OptimizationAgent().run(state)
        assert result.workflow_status == WorkflowStatus.FAILED


# ---------------------------------------------------------------------------
# 2. Source CSV integrity
# ---------------------------------------------------------------------------

class TestSourceCSVIntegrity:
    def test_binary_csv_unchanged_after_optimization(self):
        sha_before = _sha256(BINARY_CSV)
        state = _run_pipeline_through_evaluation(BINARY_CSV)
        OptimizationAgent().run(state, n_trials=2, top_n=1)
        sha_after = _sha256(BINARY_CSV)
        assert sha_before == sha_after, "Binary CSV was modified during optimization!"

    def test_multiclass_csv_unchanged_after_optimization(self):
        if not MULTICLASS_CSV.exists():
            pytest.skip("Multiclass CSV not found.")
        sha_before = _sha256(MULTICLASS_CSV)
        state = _run_pipeline_through_evaluation(MULTICLASS_CSV)
        OptimizationAgent().run(state, n_trials=2, top_n=1)
        sha_after = _sha256(MULTICLASS_CSV)
        assert sha_before == sha_after, "Multiclass CSV was modified during optimization!"


# ---------------------------------------------------------------------------
# 3. State updates after successful run
# ---------------------------------------------------------------------------

class TestStateUpdatesAfterOptimization:
    @pytest.fixture(scope="class")
    def optimized_state(self):
        state = _run_pipeline_through_evaluation(BINARY_CSV)
        return OptimizationAgent().run(state, n_trials=3, top_n=2)

    def test_workflow_status_is_optimization_completed(self, optimized_state):
        assert optimized_state.workflow_status == WorkflowStatus.OPTIMIZATION_COMPLETED

    def test_optimization_results_populated(self, optimized_state):
        assert isinstance(optimized_state.optimization_results, dict)
        assert len(optimized_state.optimization_results) > 0

    def test_best_optimized_model_set(self, optimized_state):
        assert optimized_state.best_optimized_model is not None

    def test_best_baseline_model_preserved(self, optimized_state):
        # Phase 4 best baseline must NOT be overwritten
        assert optimized_state.best_baseline_model is not None

    def test_optimization_comparison_populated(self, optimized_state):
        assert isinstance(optimized_state.optimization_comparison, list)
        assert len(optimized_state.optimization_comparison) > 0

    def test_optimization_comparison_has_required_keys(self, optimized_state):
        required = {
            "model_name", "baseline_cv_f1", "optimized_cv_f1",
            "improvement", "n_trials", "status",
        }
        for row in optimized_state.optimization_comparison:
            assert required.issubset(row.keys()), f"Missing keys in: {row.keys()}"

    def test_hyperparameter_experiments_count_positive(self, optimized_state):
        assert optimized_state.hyperparameter_experiments > 0

    def test_best_hyperparameters_dict_populated(self, optimized_state):
        assert isinstance(optimized_state.best_hyperparameters, dict)

    def test_optimization_status_field_set(self, optimized_state):
        assert optimized_state.optimization_status in ("completed", "partial", "failed")

    def test_agent_logs_contain_optimization_entries(self, optimized_state):
        log_messages = [log.message for log in optimized_state.agent_logs]
        assert any("optim" in msg.lower() for msg in log_messages)

    def test_to_dict_is_json_serializable(self, optimized_state):
        import json
        d = optimized_state.to_dict()
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# 4. Binary classification integration
# ---------------------------------------------------------------------------

class TestBinaryClassificationIntegration:
    def test_full_pipeline_binary_completes(self):
        state = _run_pipeline_through_evaluation(BINARY_CSV)
        result = OptimizationAgent().run(state, n_trials=3, top_n=1)
        assert result.workflow_status == WorkflowStatus.OPTIMIZATION_COMPLETED

    def test_binary_optimized_cv_f1_in_valid_range(self):
        state = _run_pipeline_through_evaluation(BINARY_CSV)
        result = OptimizationAgent().run(state, n_trials=3, top_n=1)
        for model_result in result.optimization_results.values():
            cv_f1 = model_result.get("optimized_cv_f1", 0.0)
            assert 0.0 <= cv_f1 <= 1.0


# ---------------------------------------------------------------------------
# 5. Multiclass classification integration
# ---------------------------------------------------------------------------

class TestMulticlassClassificationIntegration:
    def test_full_pipeline_multiclass_completes(self):
        if not MULTICLASS_CSV.exists():
            pytest.skip("Multiclass CSV not found.")
        state = _run_pipeline_through_evaluation(MULTICLASS_CSV)
        result = OptimizationAgent().run(state, n_trials=3, top_n=1)
        assert result.workflow_status == WorkflowStatus.OPTIMIZATION_COMPLETED

    def test_multiclass_best_optimized_model_is_string(self):
        if not MULTICLASS_CSV.exists():
            pytest.skip("Multiclass CSV not found.")
        state = _run_pipeline_through_evaluation(MULTICLASS_CSV)
        result = OptimizationAgent().run(state, n_trials=3, top_n=1)
        if result.best_optimized_model is not None:
            assert isinstance(result.best_optimized_model, str)


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_same_best_model(self):
        state1 = _run_pipeline_through_evaluation(BINARY_CSV)
        state2 = _run_pipeline_through_evaluation(BINARY_CSV)

        result1 = OptimizationAgent().run(state1, n_trials=3, top_n=1)
        result2 = OptimizationAgent().run(state2, n_trials=3, top_n=1)

        assert result1.best_optimized_model == result2.best_optimized_model

    def test_same_seed_produces_same_cv_f1(self):
        state1 = _run_pipeline_through_evaluation(BINARY_CSV)
        state2 = _run_pipeline_through_evaluation(BINARY_CSV)

        result1 = OptimizationAgent().run(state1, n_trials=3, top_n=1)
        result2 = OptimizationAgent().run(state2, n_trials=3, top_n=1)

        for model_key in result1.optimization_results:
            cv1 = result1.optimization_results[model_key].get("optimized_cv_f1", -1)
            cv2 = result2.optimization_results.get(model_key, {}).get("optimized_cv_f1", -2)
            assert cv1 == pytest.approx(cv2, abs=1e-6)
