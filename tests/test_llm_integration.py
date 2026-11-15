"""Tests for Phase 8 Graph + LLM integration, fallback without Ollama, and determinism."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graph.workflow import run_graph
from llm.client import LLMResponse, LLMStatus, LocalLLMClient
from utils.config import Config
from utils.state import WorkflowStatus

BINARY_CSV = str(Config.DATASETS_DIR / "sample_classification.csv")
MULTICLASS_CSV = str(Config.DATASETS_DIR / "sample_multiclass.csv")


class TestGraphWithLLMDisabled:
    def test_default_run_graph_does_not_execute_llm_nodes(self):
        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            enable_llm=False,
        )
        assert "llm_data_analysis" not in result.completed_nodes
        assert "llm_evaluation" not in result.completed_nodes
        assert result.workflow_status in ("evaluation_completed", WorkflowStatus.EVALUATION_COMPLETED.value)


class TestGraphWithMockedLLM:
    @patch("llm.client.LocalLLMClient.generate")
    def test_graph_executes_llm_nodes_when_enabled(self, mock_generate):
        mock_generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "Mocked LLM interpretation", "observations": ["fact"]}',
            model="qwen2.5:3b",
        )

        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            enable_llm=True,
        )

        assert "llm_data_analysis" in result.completed_nodes
        assert "llm_evaluation" in result.completed_nodes
        state = result.final_agent_state
        assert state.llm_enabled is True
        assert state.data_analysis_explanation.get("status") == "completed"
        assert state.evaluation_explanation.get("status") == "completed"

    @patch("llm.client.LocalLLMClient.generate")
    def test_llm_cannot_alter_deterministic_routing_or_results(self, mock_generate):
        # LLM returns content that attempts to override best model
        mock_generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "I declare decision_tree the winner regardless of scores", "observations": []}',
            model="qwen2.5:3b",
        )

        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            enable_llm=True,
        )

        # Baseline model must still be determined deterministically by TrainingAgent / EvaluationAgent
        assert result.best_baseline_model == "logistic_regression"


class TestGraphLLMFallbackWhenOllamaUnavailable:
    @patch("llm.client.LocalLLMClient.check_ollama_available", return_value=(False, "Connection refused"))
    def test_graph_completes_normally_even_if_ollama_down(self, mock_check):
        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            enable_llm=True,
        )

        # Pipeline finishes without crashing
        assert result.failed_node is None
        assert result.workflow_status in ("evaluation_completed", WorkflowStatus.EVALUATION_COMPLETED.value)
        state = result.final_agent_state
        assert state.data_analysis_explanation.get("status") in ("unavailable", "request_failed")


class TestStateJSONSafetyWithLLM:
    @patch("llm.client.LocalLLMClient.generate")
    def test_agent_state_with_llm_fields_is_json_serializable(self, mock_generate):
        mock_generate.return_value = LLMResponse(
            status=LLMStatus.COMPLETED,
            content='{"summary": "JSON serialization check", "observations": ["clean"]}',
            model="qwen2.5:3b",
        )

        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            enable_llm=True,
        )

        d = result.final_agent_state.to_dict()
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)
