"""Hyperparameter optimization utilities using Optuna.

Provides leakage-free Optuna study creation, search space definitions,
objective functions, and result collection for each registered model.
All optimization runs exclusively on training data; the held-out test
set is NEVER used inside the objective function.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline

from tools.model_tools import MODEL_REGISTRY, build_estimator, get_model_definition
from tools.preprocessing_tools import recreate_column_transformer_from_config
from tools.training_tools import evaluate_classifier, run_cross_validation
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("OptimizationTools")


# ---------------------------------------------------------------------------
# Search space definitions (one per registered model key)
# ---------------------------------------------------------------------------

def _search_space_logistic_regression(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for Logistic Regression."""
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"]),
        "max_iter": 2000,
    }


def _search_space_decision_tree(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for Decision Tree."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def _search_space_random_forest(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for Random Forest.

    max_depth: 0 is mapped to None (unlimited depth).
    """
    max_depth_raw = trial.suggest_int("max_depth", 0, 20)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": None if max_depth_raw == 0 else max_depth_raw,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _search_space_knn(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for K-Nearest Neighbors."""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
    }


def _search_space_svm(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for Support Vector Machine."""
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


# Registry mapping model_key -> search space function
_SEARCH_SPACE_REGISTRY: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {
    "logistic_regression": _search_space_logistic_regression,
    "decision_tree": _search_space_decision_tree,
    "random_forest": _search_space_random_forest,
    "k_nearest_neighbors": _search_space_knn,
    "support_vector_machine": _search_space_svm,
}


def get_supported_optimization_models() -> List[str]:
    """Return model keys that have defined search spaces."""
    return list(_SEARCH_SPACE_REGISTRY.keys())


def build_search_space(model_key: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for *model_key* using *trial*.

    Args:
        model_key: Registered model identifier.
        trial: Active Optuna trial.

    Returns:
        Dictionary of suggested hyperparameter values.

    Raises:
        KeyError: If *model_key* has no registered search space.
    """
    key = model_key.lower().strip()
    if key not in _SEARCH_SPACE_REGISTRY:
        raise KeyError(
            f"No search space registered for model '{model_key}'. "
            f"Supported: {list(_SEARCH_SPACE_REGISTRY.keys())}"
        )
    return _SEARCH_SPACE_REGISTRY[key](trial)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def create_optuna_objective(
    model_key: str,
    X_train: Any,
    y_train: Any,
    pipeline_config: Dict[str, Any],
    cv_folds: int = 5,
    random_state: int = 42,
) -> Callable[[optuna.Trial], float]:
    """Build a closure that Optuna can call for each trial.

    The closure:
    1. Suggests hyperparameters from the model-specific search space.
    2. Builds a fresh preprocessor + estimator pipeline.
    3. Runs k-fold CV exclusively on *X_train* / *y_train*.
    4. Returns the mean weighted-F1 cross-validation score.

    Leakage guarantee: *X_test* / *y_test* are never referenced inside the
    objective.  The held-out evaluation is performed separately after
    optimization completes.

    Args:
        model_key: Model registry key.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        pipeline_config: Serializable preprocessing configuration from Phase 3.
        cv_folds: Number of cross-validation folds.
        random_state: Seed for reproducibility.

    Returns:
        Callable suitable for ``study.optimize(objective, ...)``.
    """
    model_def = get_model_definition(model_key)
    scaling_required = bool(model_def["scaling_required"])

    def objective(trial: optuna.Trial) -> float:
        # 1. Sample hyperparameters
        params = build_search_space(model_key, trial)

        # 2. Build preprocessing config with model-appropriate scaling
        prep_config = dict(pipeline_config)
        prep_config["scale_numeric"] = scaling_required

        # 3. Reconstruct preprocessor (fresh per trial, no leakage across folds)
        preprocessor = recreate_column_transformer_from_config(prep_config)

        # 4. Instantiate estimator with suggested params + fixed random_state
        estimator = build_estimator(model_key, param_overrides=params)

        # 5. Assemble pipeline
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        # 6. Cross-validate on training set only
        cv_result = run_cross_validation(
            pipe,
            X_train,
            y_train,
            n_splits=cv_folds,
            random_state=random_state,
        )

        if cv_result.get("status") in ("skipped", "failed"):
            raise optuna.exceptions.TrialPruned()

        cv_f1 = cv_result.get("cv_f1_mean")
        return float(cv_f1) if cv_f1 is not None else 0.0

    return objective


# ---------------------------------------------------------------------------
# Core optimization runner
# ---------------------------------------------------------------------------

def run_optimization(
    model_key: str,
    X_train: Any,
    y_train: Any,
    pipeline_config: Dict[str, Any],
    n_trials: int = 20,
    seed: int = 42,
    cv_folds: int = 5,
    timeout: Optional[int] = None,
) -> Tuple[optuna.Study, List[Dict[str, Any]], Dict[str, Any]]:
    """Run an Optuna study for *model_key* and return structured results.

    Args:
        model_key: Model registry identifier.
        X_train: Training features (pandas DataFrame).
        y_train: Training labels (pandas Series).
        pipeline_config: Preprocessing configuration dict from Phase 3.
        n_trials: Number of Optuna trials.
        seed: Random seed for TPE sampler.
        cv_folds: Cross-validation folds used inside each trial.
        timeout: Optional wall-clock timeout (seconds) per study.

    Returns:
        Tuple of:
        - ``study``: The completed ``optuna.Study`` object (NOT stored in AgentState).
        - ``trial_records``: List of per-trial result dicts (JSON-serializable).
        - ``result_dict``: Summary dict describing optimization outcome.
    """
    t_start = time.perf_counter()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = create_optuna_objective(
        model_key=model_key,
        X_train=X_train,
        y_train=y_train,
        pipeline_config=pipeline_config,
        cv_folds=cv_folds,
        random_state=seed,
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
            catch=(Exception,),
        )
    except Exception as exc:
        logger.error(f"Optuna study for '{model_key}' encountered an error: {exc}")

    optimization_time = round(time.perf_counter() - t_start, 4)

    # Collect per-trial records (JSON-serializable dicts only)
    trial_records: List[Dict[str, Any]] = []
    for t in study.trials:
        record: Dict[str, Any] = {
            "trial_number": t.number,
            "params": dict(t.params),
            "cv_f1_score": t.value if t.value is not None else 0.0,
            "status": t.state.name.lower(),  # "complete", "pruned", "fail"
            "error_message": None,
        }
        trial_records.append(record)

    # Summarise trial counts
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed_trials_list = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.FAIL
    ]

    n_successful = len(completed_trials)
    n_failed = len(pruned_trials) + len(failed_trials_list)

    if completed_trials and study.best_trial is not None:
        best_params = dict(study.best_params)
        optimized_cv_f1 = float(study.best_value)
        status = "completed"
        error_msg = None
    else:
        best_params = {}
        optimized_cv_f1 = 0.0
        status = "failed"
        error_msg = "No successful trials completed."

    result_dict: Dict[str, Any] = {
        "model_name": model_key,
        "best_params": best_params,
        "optimized_cv_f1": round(optimized_cv_f1, 4),
        "n_trials": n_trials,
        "successful_trials": n_successful,
        "failed_trials": n_failed,
        "optimization_time": optimization_time,
        "status": status,
        "error_message": error_msg,
        "holdout_metrics": {},      # filled externally after optimization
        "baseline_cv_f1": 0.0,      # filled externally by caller
        "improvement": 0.0,         # filled externally by caller
        "improvement_pct": None,    # filled externally by caller
    }

    logger.info(
        f"Optimization for '{model_key}': {n_successful}/{n_trials} trials completed. "
        f"Best CV F1={optimized_cv_f1:.4f} in {optimization_time:.2f}s."
    )

    return study, trial_records, result_dict


# ---------------------------------------------------------------------------
# Post-optimization helpers
# ---------------------------------------------------------------------------

def build_optimized_pipeline(
    model_key: str,
    best_params: Dict[str, Any],
    pipeline_config: Dict[str, Any],
) -> Pipeline:
    """Assemble a fresh, un-fitted pipeline using the best Optuna parameters.

    Args:
        model_key: Model registry identifier.
        best_params: Best hyperparameter dict from ``study.best_params``.
        pipeline_config: Preprocessing configuration dict from Phase 3.

    Returns:
        Un-fitted ``sklearn.pipeline.Pipeline`` ready for ``.fit()``.
    """
    model_def = get_model_definition(model_key)
    scaling_required = bool(model_def["scaling_required"])

    prep_config = dict(pipeline_config)
    prep_config["scale_numeric"] = scaling_required

    preprocessor = recreate_column_transformer_from_config(prep_config)
    estimator = build_estimator(model_key, param_overrides=best_params)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def calculate_improvement(
    baseline_cv_f1: float,
    optimized_cv_f1: float,
) -> Tuple[float, Optional[float]]:
    """Compute absolute and percentage improvement over the baseline.

    Args:
        baseline_cv_f1: Baseline cross-validation F1 score.
        optimized_cv_f1: Post-optimization cross-validation F1 score.

    Returns:
        Tuple of (absolute_improvement, percentage_improvement_or_None).
        percentage is None when baseline is 0.0 to avoid division by zero.
    """
    improvement = round(optimized_cv_f1 - baseline_cv_f1, 4)
    if baseline_cv_f1 > 0.0:
        improvement_pct = round((improvement / baseline_cv_f1) * 100.0, 2)
    else:
        improvement_pct = None
    return improvement, improvement_pct


def select_candidate_models(
    model_evaluations: List[Dict[str, Any]],
    top_n: int = 2,
) -> List[str]:
    """Select top-N model names from Phase 5 evaluation ranking.

    Models are taken in ascending rank order (rank 1 = best).  Only models
    with a numeric rank (i.e. successful experiments) are included.

    Args:
        model_evaluations: List of evaluation dicts as stored in AgentState.
        top_n: Maximum number of candidates to return.

    Returns:
        Ordered list of model_name strings.
    """
    ranked = [
        m for m in model_evaluations
        if m.get("rank") is not None
    ]
    ranked.sort(key=lambda m: m["rank"])
    return [m["model_name"] for m in ranked[:top_n]]


def evaluate_optimized_on_holdout(
    model_key: str,
    best_params: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, float]:
    """Fit the optimized pipeline on training data and evaluate on the held-out test set.

    This function is called ONCE per model AFTER optimization completes.
    The test set is NEVER exposed inside the Optuna objective.

    Args:
        model_key: Model registry identifier.
        best_params: Best hyperparameters selected by Optuna.
        pipeline_config: Preprocessing configuration dict from Phase 3.
        X_train: Training features.
        y_train: Training labels.
        X_test: Held-out test features.
        y_test: Held-out test labels.

    Returns:
        Dict with keys: accuracy, precision, recall, f1_score.
    """
    pipe = build_optimized_pipeline(model_key, best_params, pipeline_config)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return evaluate_classifier(y_test, y_pred)


def build_optimization_comparison(
    model_evaluations: List[Dict[str, Any]],
    optimization_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a side-by-side comparison between baseline and optimized results.

    Args:
        model_evaluations: Phase 5 evaluation dicts (with baseline metrics).
        optimization_results: Phase 6 optimization result dicts.

    Returns:
        List of comparison dicts, one per optimized model.
    """
    # Index baseline evaluations by model_name
    baseline_map: Dict[str, Dict[str, Any]] = {
        m["model_name"]: m for m in model_evaluations
    }

    comparison: List[Dict[str, Any]] = []
    for result in optimization_results:
        model_name = result["model_name"]
        baseline = baseline_map.get(model_name, {})
        row: Dict[str, Any] = {
            "model_name": model_name,
            # Baseline metrics
            "baseline_f1": baseline.get("f1_score", 0.0),
            "baseline_accuracy": baseline.get("accuracy", 0.0),
            "baseline_cv_f1": result.get("baseline_cv_f1", 0.0),
            # Optimized metrics
            "optimized_cv_f1": result.get("optimized_cv_f1", 0.0),
            "optimized_f1": result.get("holdout_metrics", {}).get("f1_score"),
            "optimized_accuracy": result.get("holdout_metrics", {}).get("accuracy"),
            # Improvement
            "improvement": result.get("improvement", 0.0),
            "improvement_pct": result.get("improvement_pct"),
            # Meta
            "n_trials": result.get("n_trials", 0),
            "successful_trials": result.get("successful_trials", 0),
            "best_params": result.get("best_params", {}),
            "optimization_time": result.get("optimization_time", 0.0),
            "status": result.get("status", "unknown"),
        }
        comparison.append(row)

    # Sort by optimized_cv_f1 descending
    comparison.sort(key=lambda r: r.get("optimized_cv_f1", 0.0), reverse=True)
    return comparison
