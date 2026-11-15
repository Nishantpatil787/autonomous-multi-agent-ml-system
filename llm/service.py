"""High-level LLM Service coordinating prompts, client generation, and structured parsing.

Gracefully handles unavailability, malformed outputs, and timeouts without crashing
the deterministic machine learning workflow.
"""

from typing import Any, Dict, List, Optional

from llm.client import LLMResponse, LLMStatus, LocalLLMClient
from llm.parser import parse_and_validate_llm_json
from llm.prompts import (
    SYSTEM_PROMPT,
    build_data_analysis_prompt,
    build_evaluation_prompt,
    build_optimization_explanation_prompt,
    build_optimization_plan_prompt,
)
from utils.config import Config
from utils.logger import get_logger
from utils.state import AgentState

logger = get_logger("LLMService")


class LLMService:
    """Service providing safe, structured LLM interpretations for workflow stages."""

    def __init__(self, client: Optional[LocalLLMClient] = None) -> None:
        self.client = client or LocalLLMClient()

    def generate_data_analysis_explanation(
        self, state: AgentState
    ) -> Dict[str, Any]:
        """Generate interpretation for Phase 2 data analysis findings."""
        if not Config.OLLAMA_ENABLED and not state.llm_enabled:
            return {"status": LLMStatus.DISABLED.value, "reason": "LLM disabled in config/state."}

        summary = {
            "dataset_name": state.dataset_name or "dataset",
            "rows": state.dataset_shape[0] if state.dataset_shape else 0,
            "columns": state.dataset_shape[1] if state.dataset_shape else 0,
            "target_column": state.target_column,
            "problem_type": state.problem_type,
            "numerical_features": state.numerical_features,
            "categorical_features": state.categorical_features,
            "missing_total": state.missing_values.get("total_missing", 0) if isinstance(state.missing_values, dict) else 0,
            "duplicate_count": state.duplicate_count or state.duplicate_rows_count,
            "class_distribution": state.class_distribution,
        }

        prompt = build_data_analysis_prompt(summary)
        resp = self.client.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        if resp.status != LLMStatus.COMPLETED or not resp.content:
            return {
                "status": resp.status.value,
                "error": resp.error_message or "Failed to complete generation.",
            }

        valid, data, err = parse_and_validate_llm_json(
            resp.content, required_keys=["summary", "observations"]
        )
        if not valid:
            return {
                "status": LLMStatus.INVALID_RESPONSE.value,
                "error": err,
                "raw": resp.content,
            }

        data["status"] = LLMStatus.COMPLETED.value
        data["model"] = resp.model
        data["duration"] = resp.duration_seconds
        return data

    def generate_evaluation_explanation(
        self, state: AgentState
    ) -> Dict[str, Any]:
        """Generate interpretation for Phase 5 evaluation results."""
        if not Config.OLLAMA_ENABLED and not state.llm_enabled:
            return {"status": LLMStatus.DISABLED.value, "reason": "LLM disabled in config/state."}

        summary = {
            "best_model": state.best_baseline_model,
            "best_f1": state.best_baseline_metrics.get("f1_score", 0.0),
            "total_experiments": len(state.experiment_results),
        }

        prompt = build_evaluation_prompt(summary, state.model_evaluations)
        resp = self.client.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        if resp.status != LLMStatus.COMPLETED or not resp.content:
            return {
                "status": resp.status.value,
                "error": resp.error_message or "Failed to complete generation.",
            }

        valid, data, err = parse_and_validate_llm_json(
            resp.content, required_keys=["summary", "observations"]
        )
        if not valid:
            return {
                "status": LLMStatus.INVALID_RESPONSE.value,
                "error": err,
                "raw": resp.content,
            }

        data["status"] = LLMStatus.COMPLETED.value
        data["model"] = resp.model
        data["duration"] = resp.duration_seconds
        return data

    def generate_optimization_plan_suggestion(
        self, state: AgentState, candidate_models: List[str]
    ) -> Dict[str, Any]:
        """Propose non-binding hyperparameter tuning suggestions before Phase 6."""
        if not Config.OLLAMA_ENABLED and not state.llm_enabled:
            return {"status": LLMStatus.DISABLED.value, "reason": "LLM disabled in config/state."}

        summary = {
            "best_model": state.best_baseline_model,
            "best_f1": state.best_baseline_metrics.get("f1_score", 0.0),
        }

        prompt = build_optimization_plan_prompt(candidate_models, summary)
        resp = self.client.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        if resp.status != LLMStatus.COMPLETED or not resp.content:
            return {
                "status": resp.status.value,
                "error": resp.error_message or "Failed to complete generation.",
            }

        valid, data, err = parse_and_validate_llm_json(
            resp.content, required_keys=["summary", "focus_hyperparameters"]
        )
        if not valid:
            return {
                "status": LLMStatus.INVALID_RESPONSE.value,
                "error": err,
                "raw": resp.content,
            }

        data["status"] = LLMStatus.COMPLETED.value
        data["model"] = resp.model
        data["duration"] = resp.duration_seconds
        return data

    def generate_optimization_explanation(
        self, state: AgentState
    ) -> Dict[str, Any]:
        """Generate interpretation for Phase 6 hyperparameter tuning results."""
        if not Config.OLLAMA_ENABLED and not state.llm_enabled:
            return {"status": LLMStatus.DISABLED.value, "reason": "LLM disabled in config/state."}

        summary = state.optimization_summary or {
            "best_optimized_model": state.best_optimized_model,
            "best_baseline_model": state.best_baseline_model,
            "total_hyperparameter_experiments": state.hyperparameter_experiments,
        }

        prompt = build_optimization_explanation_prompt(
            summary, state.optimization_comparison
        )
        resp = self.client.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        if resp.status != LLMStatus.COMPLETED or not resp.content:
            return {
                "status": resp.status.value,
                "error": resp.error_message or "Failed to complete generation.",
            }

        valid, data, err = parse_and_validate_llm_json(
            resp.content, required_keys=["summary", "observations"]
        )
        if not valid:
            return {
                "status": LLMStatus.INVALID_RESPONSE.value,
                "error": err,
                "raw": resp.content,
            }

        data["status"] = LLMStatus.COMPLETED.value
        data["model"] = resp.model
        data["duration"] = resp.duration_seconds
        return data
