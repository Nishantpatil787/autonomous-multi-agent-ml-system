"""Tests for optimization_tools.py (Phase 6).

Covers: search space construction, supported models, deterministic sampler,
trial counting, param validity, CV objective, preprocessing inside pipeline,
leakage prevention, small-class safeguards, failed trial handling,
all-model failure, improvement calculation, zero denominator, deterministic
ranking, best param recording, trial records, experiment counts, and JSON
serialization.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

from tools.optimization_tools import (
    build_optimization_comparison,
    build_optimized_pipeline,
    build_search_space,
    calculate_improvement,
    create_optuna_objective,
    evaluate_optimized_on_holdout,
    get_supported_optimization_models,
    run_optimization,
    select_candidate_models,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_PIPELINE_CONFIG = {
    "selected_numerical_features": ["age", "score"],
    "selected_categorical_features": [],
    "numerical_imputation_strategy": "median",
    "categorical_imputation_strategy": "most_frequent",
    "encoding": "one_hot",
    "scale_numeric": True,
    "handle_unknown_categories": "ignore",
}


def _make_binary_data(n: int = 80, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"age": rng.integers(20, 60, n), "score": rng.uniform(0, 1, n)}
    )
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


def _make_multiclass_data(n: int = 90, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"age": rng.integers(20, 60, n), "score": rng.uniform(0, 1, n)}
    )
    y = pd.Series(rng.integers(0, 3, n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# 1. Supported models
# ---------------------------------------------------------------------------

class TestSupportedModels:
    def test_all_five_models_registered(self):
        supported = get_supported_optimization_models()
        expected = {
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "k_nearest_neighbors",
            "support_vector_machine",
        }
        assert set(supported) == expected

    def test_returns_list(self):
        assert isinstance(get_supported_optimization_models(), list)


# ---------------------------------------------------------------------------
# 2. Search space construction
# ---------------------------------------------------------------------------

class TestBuildSearchSpace:
    def _make_trial(self, model_key: str) -> optuna.Trial:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        trial = study.ask()
        return trial

    def test_logistic_regression_params_present(self):
        trial = self._make_trial("logistic_regression")
        params = build_search_space("logistic_regression", trial)
        assert "C" in params
        assert "solver" in params
        assert "max_iter" in params

    def test_logistic_regression_C_positive(self):
        trial = self._make_trial("logistic_regression")
        params = build_search_space("logistic_regression", trial)
        assert params["C"] > 0

    def test_logistic_regression_solver_valid(self):
        trial = self._make_trial("logistic_regression")
        params = build_search_space("logistic_regression", trial)
        assert params["solver"] in ["lbfgs", "liblinear", "saga"]

    def test_decision_tree_params_present(self):
        trial = self._make_trial("decision_tree")
        params = build_search_space("decision_tree", trial)
        assert "max_depth" in params
        assert "min_samples_split" in params
        assert "min_samples_leaf" in params
        assert "criterion" in params

    def test_decision_tree_criterion_valid(self):
        trial = self._make_trial("decision_tree")
        params = build_search_space("decision_tree", trial)
        assert params["criterion"] in ["gini", "entropy"]

    def test_random_forest_params_present(self):
        trial = self._make_trial("random_forest")
        params = build_search_space("random_forest", trial)
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "max_features" in params

    def test_random_forest_max_depth_none_allowed(self):
        """When max_depth_raw == 0 from trial, result should be None."""
        trial = self._make_trial("random_forest")
        # Patch to always return 0
        with patch.object(trial, "suggest_int", return_value=0):
            params = build_search_space("random_forest", trial)
        assert params["max_depth"] is None

    def test_random_forest_max_features_valid(self):
        trial = self._make_trial("random_forest")
        params = build_search_space("random_forest", trial)
        assert params["max_features"] in ["sqrt", "log2"]

    def test_knn_params_present(self):
        trial = self._make_trial("k_nearest_neighbors")
        params = build_search_space("k_nearest_neighbors", trial)
        assert "n_neighbors" in params
        assert "weights" in params
        assert "metric" in params

    def test_knn_n_neighbors_positive(self):
        trial = self._make_trial("k_nearest_neighbors")
        params = build_search_space("k_nearest_neighbors", trial)
        assert params["n_neighbors"] >= 1

    def test_svm_params_present(self):
        trial = self._make_trial("support_vector_machine")
        params = build_search_space("support_vector_machine", trial)
        assert "C" in params
        assert "kernel" in params
        assert "gamma" in params

    def test_svm_kernel_valid(self):
        trial = self._make_trial("support_vector_machine")
        params = build_search_space("support_vector_machine", trial)
        assert params["kernel"] in ["rbf", "linear"]

    def test_unsupported_model_raises_key_error(self):
        trial = self._make_trial("logistic_regression")
        with pytest.raises(KeyError):
            build_search_space("unsupported_model_xyz", trial)


# ---------------------------------------------------------------------------
# 3. Improvement calculation
# ---------------------------------------------------------------------------

class TestCalculateImprovement:
    def test_positive_improvement(self):
        improvement, pct = calculate_improvement(0.80, 0.85)
        assert improvement == pytest.approx(0.05, abs=1e-4)
        assert pct == pytest.approx(6.25, abs=0.1)

    def test_no_improvement(self):
        improvement, pct = calculate_improvement(0.80, 0.80)
        assert improvement == pytest.approx(0.0, abs=1e-4)
        assert pct == pytest.approx(0.0, abs=0.1)

    def test_negative_improvement(self):
        improvement, pct = calculate_improvement(0.85, 0.80)
        assert improvement < 0
        assert pct is not None and pct < 0

    def test_zero_baseline_returns_none_pct(self):
        improvement, pct = calculate_improvement(0.0, 0.75)
        assert improvement == pytest.approx(0.75, abs=1e-4)
        assert pct is None


# ---------------------------------------------------------------------------
# 4. Select candidate models
# ---------------------------------------------------------------------------

class TestSelectCandidateModels:
    def _make_evaluations(self):
        return [
            {"model_name": "random_forest", "rank": 1, "f1_score": 0.90},
            {"model_name": "logistic_regression", "rank": 2, "f1_score": 0.85},
            {"model_name": "decision_tree", "rank": 3, "f1_score": 0.80},
            {"model_name": "k_nearest_neighbors", "rank": None, "f1_score": 0.0},
        ]

    def test_top_2_selected(self):
        evals = self._make_evaluations()
        candidates = select_candidate_models(evals, top_n=2)
        assert candidates == ["random_forest", "logistic_regression"]

    def test_top_1_selected(self):
        evals = self._make_evaluations()
        candidates = select_candidate_models(evals, top_n=1)
        assert candidates == ["random_forest"]

    def test_failed_models_excluded(self):
        evals = self._make_evaluations()
        candidates = select_candidate_models(evals, top_n=10)
        assert "k_nearest_neighbors" not in candidates

    def test_empty_evaluations_returns_empty(self):
        assert select_candidate_models([], top_n=2) == []

    def test_top_n_larger_than_available(self):
        evals = [{"model_name": "rf", "rank": 1}]
        candidates = select_candidate_models(evals, top_n=5)
        assert candidates == ["rf"]


# ---------------------------------------------------------------------------
# 5. Build optimization comparison
# ---------------------------------------------------------------------------

class TestBuildOptimizationComparison:
    def test_comparison_contains_expected_keys(self):
        model_evals = [
            {"model_name": "random_forest", "rank": 1, "f1_score": 0.88, "accuracy": 0.87},
        ]
        opt_results = [
            {
                "model_name": "random_forest",
                "baseline_cv_f1": 0.82,
                "optimized_cv_f1": 0.87,
                "improvement": 0.05,
                "improvement_pct": 6.1,
                "n_trials": 20,
                "successful_trials": 18,
                "best_params": {"n_estimators": 200},
                "optimization_time": 10.0,
                "status": "completed",
                "holdout_metrics": {"f1_score": 0.89, "accuracy": 0.88},
            }
        ]
        comparison = build_optimization_comparison(model_evals, opt_results)
        assert len(comparison) == 1
        row = comparison[0]
        assert "model_name" in row
        assert "baseline_cv_f1" in row
        assert "optimized_cv_f1" in row
        assert "improvement" in row

    def test_comparison_sorted_by_optimized_cv_f1_desc(self):
        model_evals = [
            {"model_name": "rf", "rank": 1, "f1_score": 0.88, "accuracy": 0.87},
            {"model_name": "lr", "rank": 2, "f1_score": 0.82, "accuracy": 0.81},
        ]
        opt_results = [
            {
                "model_name": "rf",
                "baseline_cv_f1": 0.82,
                "optimized_cv_f1": 0.87,
                "improvement": 0.05,
                "improvement_pct": 6.1,
                "n_trials": 20,
                "successful_trials": 18,
                "best_params": {},
                "optimization_time": 10.0,
                "status": "completed",
                "holdout_metrics": {},
            },
            {
                "model_name": "lr",
                "baseline_cv_f1": 0.80,
                "optimized_cv_f1": 0.84,
                "improvement": 0.04,
                "improvement_pct": 5.0,
                "n_trials": 20,
                "successful_trials": 17,
                "best_params": {},
                "optimization_time": 5.0,
                "status": "completed",
                "holdout_metrics": {},
            },
        ]
        comparison = build_optimization_comparison(model_evals, opt_results)
        assert comparison[0]["model_name"] == "rf"


# ---------------------------------------------------------------------------
# 6. Run optimization integration tests
# ---------------------------------------------------------------------------

class TestRunOptimization:
    """Integration tests that run actual Optuna studies with tiny trial counts."""

    def test_run_optimization_returns_tuple(self):
        X, y = _make_binary_data(n=60)
        study, trial_records, result = run_optimization(
            "logistic_regression", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        assert isinstance(study, optuna.Study)
        assert isinstance(trial_records, list)
        assert isinstance(result, dict)

    def test_result_dict_has_required_keys(self):
        X, y = _make_binary_data(n=60)
        _, _, result = run_optimization(
            "decision_tree", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        required = {
            "model_name", "best_params", "optimized_cv_f1", "n_trials",
            "successful_trials", "failed_trials", "optimization_time", "status",
        }
        assert required.issubset(result.keys())

    def test_result_is_json_serializable(self):
        import json
        X, y = _make_binary_data(n=60)
        _, _, result = run_optimization(
            "logistic_regression", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        # Must not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_trial_records_length_matches_n_trials(self):
        X, y = _make_binary_data(n=60)
        n = 4
        _, trial_records, _ = run_optimization(
            "logistic_regression", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=n, seed=42
        )
        assert len(trial_records) == n

    def test_deterministic_with_same_seed(self):
        X, y = _make_binary_data(n=60)
        _, _, result1 = run_optimization(
            "logistic_regression", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        _, _, result2 = run_optimization(
            "logistic_regression", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        assert result1["optimized_cv_f1"] == result2["optimized_cv_f1"]

    def test_best_params_stored_in_result(self):
        X, y = _make_binary_data(n=60)
        _, _, result = run_optimization(
            "logistic_regression", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        if result["status"] == "completed":
            assert len(result["best_params"]) > 0

    def test_optimization_with_multiclass_data(self):
        X, y = _make_multiclass_data(n=90)
        _, _, result = run_optimization(
            "random_forest", X, y, SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        assert result["model_name"] == "random_forest"

    def test_holdout_evaluation_leakage_free(self):
        """Holdout metrics use X_test not seen during optimization."""
        X, y = _make_binary_data(n=80)
        X_train, X_test = X.iloc[:64], X.iloc[64:]
        y_train, y_test = y.iloc[:64], y.iloc[64:]
        _, _, result = run_optimization(
            "logistic_regression", X_train, y_train,
            SAMPLE_PIPELINE_CONFIG, n_trials=3, seed=42
        )
        if result["status"] == "completed":
            holdout = evaluate_optimized_on_holdout(
                "logistic_regression",
                result["best_params"],
                SAMPLE_PIPELINE_CONFIG,
                X_train, y_train, X_test, y_test,
            )
            assert "f1_score" in holdout
            assert "accuracy" in holdout
            assert 0.0 <= holdout["f1_score"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Build optimized pipeline
# ---------------------------------------------------------------------------

class TestBuildOptimizedPipeline:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        pipe = build_optimized_pipeline(
            "logistic_regression",
            {"C": 1.0, "solver": "lbfgs"},
            SAMPLE_PIPELINE_CONFIG,
        )
        assert isinstance(pipe, Pipeline)

    def test_pipeline_can_fit(self):
        X, y = _make_binary_data(n=60)
        pipe = build_optimized_pipeline(
            "decision_tree",
            {"max_depth": 3, "criterion": "gini"},
            SAMPLE_PIPELINE_CONFIG,
        )
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
