"""Tests for tools/persistence_tools.py and agents/persistence_agent.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from agents.persistence_agent import PersistenceAgent
from tools import persistence_tools
from utils.config import Config
from utils.state import AgentState, WorkflowStatus

BINARY_CSV = str(Config.DATASETS_DIR / "sample_classification.csv")
MULTICLASS_CSV = str(Config.DATASETS_DIR / "sample_multiclass.csv")


def _make_dummy_agent_state(optimized: bool = True) -> AgentState:
    state = AgentState(dataset_path=BINARY_CSV)
    state.dataset_name = "sample_classification.csv"
    state.target_column = "Outcome"
    state.problem_type = "binary_classification"
    state.selected_numerical_features = ["age", "salary", "credit_score"]
    state.selected_categorical_features = ["department", "education"]
    state.pipeline_config = {
        "selected_numerical_features": ["age", "salary", "credit_score"],
        "selected_categorical_features": ["department", "education"],
        "numerical_imputation_strategy": "median",
        "categorical_imputation_strategy": "most_frequent",
        "scaling_strategy": "standard_scaler",
        "scale_numeric": True,
        "duplicate_handling_strategy": "remove_before_training",
    }
    state.best_baseline_model = "logistic_regression"
    state.best_baseline_metrics = {"f1_score": 0.6667, "accuracy": 0.6667}
    state.experiment_results = [
        {"model_name": "logistic_regression", "f1_score": 0.6667, "accuracy": 0.6667, "cv_f1_mean": 0.6129}
    ]
    if optimized:
        state.optimization_status = "completed"
        state.best_optimized_model = "k_nearest_neighbors"
        state.optimization_results = {
            "k_nearest_neighbors": {
                "status": "completed",
                "optimized_cv_f1": 0.6984,
                "baseline_cv_f1": 0.6357,
                "improvement": 0.0627,
                "best_params": {"n_neighbors": 7, "weights": "uniform"},
                "holdout_metrics": {"f1_score": 0.70, "accuracy": 0.72},
            }
        }
    else:
        state.optimization_status = "failed"
        state.best_optimized_model = None
        state.optimization_results = {}
    return state


class TestModelSelectionPolicy:
    def test_selects_optimized_model_when_available(self):
        state = _make_dummy_agent_state(optimized=True)
        sel = persistence_tools.select_final_model_for_persistence(state)
        assert sel["selection_source"] == "optimized"
        assert sel["model_name"] == "k_nearest_neighbors"
        assert sel["best_params"] == {"n_neighbors": 7, "weights": "uniform"}

    def test_falls_back_to_baseline_when_optimization_unavailable(self):
        state = _make_dummy_agent_state(optimized=False)
        sel = persistence_tools.select_final_model_for_persistence(state)
        assert sel["selection_source"] == "baseline"
        assert sel["model_name"] == "logistic_regression"


class TestFinalPipelineFitting:
    def test_fits_and_returns_valid_pipeline(self):
        state = _make_dummy_agent_state(optimized=False)
        pipe, feature_cols = persistence_tools.fit_final_pipeline_for_persistence(
            state=state,
            model_name="logistic_regression",
            best_params={"max_iter": 500},
        )
        assert isinstance(pipe, Pipeline)
        assert "age" in feature_cols
        assert "salary" in feature_cols


class TestModelPersistenceAndReload:
    def test_save_and_load_roundtrip(self, tmp_path):
        state = _make_dummy_agent_state(optimized=False)
        pipe, feature_cols = persistence_tools.fit_final_pipeline_for_persistence(
            state=state,
            model_name="logistic_regression",
        )
        metadata = {
            "model_name": "logistic_regression",
            "feature_columns": feature_cols,
            "task_type": "binary_classification",
        }

        m_file = tmp_path / "test_model.joblib"
        meta_file = tmp_path / "test_model_meta.json"

        persistence_tools.save_persisted_model(pipe, metadata, m_file, meta_file)

        assert m_file.exists()
        assert meta_file.exists()

        loaded_pipe, loaded_meta = persistence_tools.load_persisted_model(m_file, meta_file)
        assert isinstance(loaded_pipe, Pipeline)
        assert loaded_meta["model_name"] == "logistic_regression"

    def test_prediction_with_reloaded_pipeline(self, tmp_path):
        state = _make_dummy_agent_state(optimized=False)
        pipe, feature_cols = persistence_tools.fit_final_pipeline_for_persistence(
            state=state,
            model_name="logistic_regression",
        )
        metadata = {
            "model_name": "logistic_regression",
            "feature_columns": feature_cols,
            "task_type": "binary_classification",
            "numerical_features": ["Age", "BMI", "BloodPressure"],
        }

        m_file = tmp_path / "test_model.joblib"
        meta_file = tmp_path / "test_model_meta.json"
        persistence_tools.save_persisted_model(pipe, metadata, m_file, meta_file)

        loaded_pipe, loaded_meta = persistence_tools.load_persisted_model(m_file, meta_file)

        # Single row raw feature dictionary
        input_data = {
            "age": 33.0,
            "salary": 50000.0,
            "credit_score": 650.0,
            "department": "Engineering",
            "education": "Master",
        }
        pred_res = persistence_tools.predict_with_persisted_pipeline(
            loaded_pipe, loaded_meta, input_data
        )

        assert "predictions" in pred_res
        assert len(pred_res["predictions"]) == 1
        assert "probabilities" in pred_res

    def test_missing_artifact_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            persistence_tools.load_persisted_model(
                tmp_path / "missing.joblib", tmp_path / "missing.json"
            )


class TestPersistenceAgentExecution:
    def test_persistence_agent_updates_state(self, tmp_path):
        state = _make_dummy_agent_state(optimized=True)
        agent = PersistenceAgent()

        # Pass custom file names inside tmp_path
        m_file = tmp_path / "custom_model.joblib"
        meta_file = tmp_path / "custom_meta.json"

        with patch.object(persistence_tools, "DEFAULT_MODEL_FILENAME", str(m_file)), \
             patch.object(persistence_tools, "DEFAULT_METADATA_FILENAME", str(meta_file)):
            updated_state = agent.run(state, model_filename=str(m_file), metadata_filename=str(meta_file))
            assert updated_state.persistence_status == "completed"
            assert updated_state.persisted_model_path is not None
            assert Path(updated_state.persisted_model_path).exists()
            assert Path(updated_state.persisted_metadata_path).exists()
            assert updated_state.persisted_model_metadata["model_name"] == "k_nearest_neighbors"

    def test_persistence_agent_prerequisites_validation(self):
        state = AgentState()  # Missing dataset_path, target_column, etc.
        agent = PersistenceAgent()
        res = agent.run(state)
        assert res.persistence_status == "failed"
