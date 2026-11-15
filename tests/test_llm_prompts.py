"""Tests for llm/prompts.py — structured prompt construction, boundaries, and privacy."""

import pytest

from llm.prompts import (
    SYSTEM_PROMPT,
    build_data_analysis_prompt,
    build_evaluation_prompt,
    build_optimization_explanation_prompt,
    build_optimization_plan_prompt,
)


class TestPromptStructureAndDiscipline:
    def test_system_prompt_instructs_no_metric_invention(self):
        assert "not invent" in SYSTEM_PROMPT.lower()
        assert "authoritative" in SYSTEM_PROMPT.lower()

    def test_system_prompt_distinguishes_observed_and_interpretation(self):
        assert "OBSERVED" in SYSTEM_PROMPT
        assert "INTERPRETATION" in SYSTEM_PROMPT
        assert "HYPOTHESIS" in SYSTEM_PROMPT

    def test_data_analysis_prompt_contains_summary_keys(self):
        summary = {
            "dataset_name": "test.csv",
            "rows": 100,
            "columns": 5,
            "target_column": "target",
            "problem_type": "binary_classification",
            "numerical_features": ["f1", "f2"],
            "categorical_features": ["f3"],
            "missing_total": 0,
            "duplicate_count": 0,
            "class_distribution": {"0": 50, "1": 50},
        }
        prompt = build_data_analysis_prompt(summary)
        assert "test.csv" in prompt
        assert "100" in prompt
        assert "binary_classification" in prompt
        assert "target" in prompt

    def test_prompt_demands_json_schema(self):
        prompt = build_data_analysis_prompt({})
        assert '"summary":' in prompt
        assert '"observations":' in prompt
        assert '"interpretation":' in prompt

    def test_evaluation_prompt_contains_model_rankings(self):
        summary = {"best_model": "logistic_regression", "best_f1": 0.85, "total_experiments": 5}
        rankings = [
            {"rank": 1, "model_name": "logistic_regression", "f1_score": 0.85, "accuracy": 0.84, "cv_f1_mean": 0.82, "f1_gap": 0.03}
        ]
        prompt = build_evaluation_prompt(summary, rankings)
        assert "logistic_regression" in prompt
        assert "0.85" in prompt
        assert "cv_f1_mean" in prompt

    def test_optimization_plan_prompt_includes_candidates(self):
        candidates = ["random_forest", "svm"]
        summary = {"best_model": "random_forest", "best_f1": 0.90}
        prompt = build_optimization_plan_prompt(candidates, summary)
        assert "random_forest" in prompt
        assert "svm" in prompt
        assert "focus_hyperparameters" in prompt

    def test_optimization_explanation_prompt_structure(self):
        summary = {"best_optimized_model": "svm", "best_baseline_model": "rf", "total_hyperparameter_experiments": 20}
        comparison = [{"model_name": "svm", "baseline_cv_f1": 0.80, "optimized_cv_f1": 0.85, "improvement": 0.05, "best_params": {"C": 1.0}}]
        prompt = build_optimization_explanation_prompt(summary, comparison)
        assert "svm" in prompt
        assert "0.05" in prompt
        assert "observations" in prompt


class TestPromptContextPrivacy:
    def test_prompts_do_not_contain_env_secrets(self):
        summary = {"dataset_name": "clean.csv"}
        prompt = build_data_analysis_prompt(summary)
        assert "API_KEY" not in prompt
        assert "password" not in prompt.lower()
        assert "token" not in prompt.lower()
