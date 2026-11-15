"""Tests for llm/service.py — LLMService coordinator, fallbacks, and schema population."""

from unittest.mock import MagicMock

import pytest

from llm.client import LLMResponse, LLMStatus, LocalLLMClient
from llm.service import LLMService
from utils.config import Config
from utils.state import AgentState


def _make_dummy_state() -> AgentState:
    state = AgentState(dataset_path="test.csv")
    state.dataset_name = "test.csv"
    state.dataset_shape = (50, 4)
    state.target_column = "target"
    state.problem_type = "binary_classification"
    state.numerical_features = ["age", "fare"]
    state.categorical_features = ["sex"]
    state.best_baseline_model = "logistic_regression"
    state.best_baseline_metrics = {"f1_score": 0.88, "accuracy": 0.86}
    state.model_evaluations = [
        {"rank": 1, "model_name": "logistic_regression", "f1_score": 0.88, "accuracy": 0.86}
    ]
    state.experiment_results = [{"model_name": "logistic_regression", "status": "success"}]
    state.optimization_summary = {
        "best_optimized_model": "svm",
        "best_baseline_model": "logistic_regression",
        "total_hyperparameter_experiments": 20,
    }
    state.optimization_comparison = [
        {"model_name": "svm", "baseline_cv_f1": 0.80, "optimized_cv_f1": 0.85, "improvement": 0.05}
    ]
    return state


class TestLLMServiceDisabled:
    def test_data_analysis_disabled_by_default(self):
        state = _make_dummy_state()
        state.llm_enabled = False
        service = LLMService()
        res = service.generate_data_analysis_explanation(state)
        assert res["status"] == LLMStatus.DISABLED.value

    def test_evaluation_disabled_by_default(self):
        state = _make_dummy_state()
        state.llm_enabled = False
        service = LLMService()
        res = service.generate_evaluation_explanation(state)
        assert res["status"] == LLMStatus.DISABLED.value


class TestLLMServiceMockedSuccess:
    def test_generate_data_analysis_explanation_success(self):
        state = _make_dummy_state()
        state.llm_enabled = True

        mock_client = MagicMock(spec=LocalLLMClient)
        mock_client.generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "A clean dataset", "observations": ["50 rows"]}',
            model="qwen2.5:3b",
            duration_seconds=1.2,
        )

        service = LLMService(client=mock_client)
        res = service.generate_data_analysis_explanation(state)
        assert res["status"] == LLMStatus.COMPLETED.value
        assert res["summary"] == "A clean dataset"
        assert res["model"] == "qwen2.5:3b"

    def test_generate_evaluation_explanation_success(self):
        state = _make_dummy_state()
        state.llm_enabled = True

        mock_client = MagicMock(spec=LocalLLMClient)
        mock_client.generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "Logistic regression led rankings", "observations": ["F1 was 0.88"]}',
            model="qwen2.5:3b",
        )

        service = LLMService(client=mock_client)
        res = service.generate_evaluation_explanation(state)
        assert res["status"] == LLMStatus.COMPLETED.value
        assert "Logistic regression" in res["summary"]

    def test_generate_optimization_plan_suggestion_success(self):
        state = _make_dummy_state()
        state.llm_enabled = True

        mock_client = MagicMock(spec=LocalLLMClient)
        mock_client.generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "Tune regularization", "focus_hyperparameters": ["C", "penalty"]}',
            model="qwen2.5:3b",
        )

        service = LLMService(client=mock_client)
        res = service.generate_optimization_plan_suggestion(state, candidate_models=["logistic_regression"])
        assert res["status"] == LLMStatus.COMPLETED.value
        assert "C" in res["focus_hyperparameters"]

    def test_generate_optimization_explanation_success(self):
        state = _make_dummy_state()
        state.llm_enabled = True

        mock_client = MagicMock(spec=LocalLLMClient)
        mock_client.generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "SVM improved baseline", "observations": ["0.05 gain"]}',
            model="qwen2.5:3b",
        )

        service = LLMService(client=mock_client)
        res = service.generate_optimization_explanation(state)
        assert res["status"] == LLMStatus.COMPLETED.value
        assert "0.05 gain" in res["observations"]


class TestLLMServiceFallbackHandling:
    def test_ollama_unavailable_graceful_fallback(self):
        state = _make_dummy_state()
        state.llm_enabled = True

        mock_client = MagicMock(spec=LocalLLMClient)
        mock_client.generate.return_value = LLMResponse(
            status=LLMStatus.UNAVAILABLE,
            error_message="Ollama service is down",
        )

        service = LLMService(client=mock_client)
        res = service.generate_data_analysis_explanation(state)
        assert res["status"] == LLMStatus.UNAVAILABLE.value
        assert "Ollama service is down" in res["error"]

    def test_malformed_llm_output_graceful_fallback(self):
        state = _make_dummy_state()
        state.llm_enabled = True

        mock_client = MagicMock(spec=LocalLLMClient)
        mock_client.generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content="This is plain text with no JSON at all",
        )

        service = LLMService(client=mock_client)
        res = service.generate_evaluation_explanation(state)
        assert res["status"] == LLMStatus.INVALID_RESPONSE.value
        assert "error" in res
