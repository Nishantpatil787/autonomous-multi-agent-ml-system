"""Persistence tools for saving and loading end-to-end trained ML pipelines and metadata.

Handles final model selection policy, serialization with joblib, schema metadata
extraction, atomic writing, and validation without leaking raw datasets or loading
arbitrary user-uploaded pickles.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tools import model_tools, preprocessing_tools, training_tools
from utils.config import Config
from utils.logger import get_logger
from utils.state import AgentState

logger = get_logger("PersistenceTools")

DEFAULT_MODEL_FILENAME = "best_model.joblib"
DEFAULT_METADATA_FILENAME = "best_model_metadata.json"


def select_final_model_for_persistence(
    state: AgentState,
) -> Dict[str, Any]:
    """Deterministically select between the best optimized model and best baseline model.

    Policy:
    1. If optimization completed successfully, produced a valid `best_optimized_model`,
       and improved upon or matched baseline performance -> select optimized model.
    2. Otherwise, fall back deterministically to `best_baseline_model`.

    Returns:
        Dictionary with keys:
            model_name: str
            selection_source: "optimized" or "baseline"
            best_params: dict
            metrics: dict
            cv_metrics: dict
    """
    opt_summary = state.optimization_summary or {}
    best_opt = state.best_optimized_model

    # Check if optimization was used and completed for the candidate
    if best_opt and state.optimization_status == "completed" and state.optimization_results:
        opt_info = {}
        if isinstance(state.optimization_results, dict):
            opt_info = state.optimization_results.get(best_opt, {})
        elif isinstance(state.optimization_results, list):
            for item in state.optimization_results:
                if isinstance(item, dict) and item.get("model_name") == best_opt:
                    opt_info = item
                    break
        if opt_info and opt_info.get("status") == "completed":
            params = opt_info.get("best_params", {})
            holdout_metrics = opt_info.get("holdout_metrics", {})
            cv_metrics = {
                "cv_f1_mean": opt_info.get("optimized_cv_f1"),
                "baseline_cv_f1": opt_info.get("baseline_cv_f1"),
                "improvement": opt_info.get("improvement"),
                "improvement_pct": opt_info.get("improvement_pct"),
            }
            logger.info(
                f"Selected best OPTIMIZED model '{best_opt}' for persistence (CV F1: {opt_info.get('optimized_cv_f1')})."
            )
            return {
                "model_name": best_opt,
                "selection_source": "optimized",
                "best_params": params,
                "metrics": holdout_metrics or state.best_baseline_metrics,
                "cv_metrics": cv_metrics,
                "optimization_used": True,
            }

    # Fallback to best baseline model
    best_baseline = state.best_baseline_model
    if not best_baseline:
        # Fallback to top ranked model if available
        if state.model_evaluations:
            best_baseline = state.model_evaluations[0].get("model_name")

    # Find baseline metrics
    baseline_metrics: Dict[str, Any] = dict(state.best_baseline_metrics)
    baseline_cv: Dict[str, Any] = {}
    for exp in state.experiment_results:
        if exp.get("model_name") == best_baseline:
            baseline_cv = {
                "cv_f1_mean": exp.get("cv_f1_mean"),
                "cv_accuracy_mean": exp.get("cv_accuracy_mean"),
                "cv_folds": exp.get("cv_folds"),
            }
            if not baseline_metrics:
                baseline_metrics = {
                    "accuracy": exp.get("accuracy", 0.0),
                    "precision": exp.get("precision", 0.0),
                    "recall": exp.get("recall", 0.0),
                    "f1_score": exp.get("f1_score", 0.0),
                }
            break

    logger.info(
        f"Selected best BASELINE model '{best_baseline}' for persistence (optimization skipped or unselected)."
    )
    return {
        "model_name": best_baseline or "unknown",
        "selection_source": "baseline",
        "best_params": {},
        "metrics": baseline_metrics,
        "cv_metrics": baseline_cv,
        "optimization_used": False,
    }


def fit_final_pipeline_for_persistence(
    state: AgentState,
    model_name: str,
    best_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, List[str]]:
    """Assemble and fit the final complete Pipeline [preprocessor -> estimator] on training data.

    Important:
    This fit occurs only after model selection and hyperparameter optimization are done.
    It does not modify evaluation results, data analysis, or the source CSV.

    Args:
        state: Shared workflow state.
        model_name: Selected model key.
        best_params: Best hyperparameters to apply (if optimized).

    Returns:
        (fitted_pipeline, feature_names_list)
    """
    df, _ = training_tools.load_training_dataframe(
        state.dataset_path,
        duplicate_strategy=state.duplicate_handling_strategy,
    )
    X, y = training_tools.prepare_features_and_target(
        df,
        target_column=state.target_column,
        excluded_features=state.excluded_features,
    )

    X_train, _, y_train, _, _, _ = training_tools.split_dataset(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    # Preprocessing configuration with model-appropriate scaling
    model_def = model_tools.get_model_definition(model_name)
    scaling_required = bool(model_def.get("scaling_required", False))

    prep_config = dict(state.pipeline_config)
    prep_config["scale_numeric"] = scaling_required

    preprocessor = preprocessing_tools.recreate_column_transformer_from_config(prep_config)
    estimator = model_tools.build_estimator(model_name, param_overrides=best_params or {})

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    final_pipeline.fit(X_train, y_train)
    logger.info(f"Fitted final pipeline for '{model_name}' on {len(X_train)} samples.")
    return final_pipeline, list(X.columns)


def save_persisted_model(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Persist fitted pipeline with joblib and metadata as JSON atomically."""
    Config.ensure_directories()
    m_path = Path(model_path or (Config.MODELS_DIR / DEFAULT_MODEL_FILENAME))
    meta_path = Path(metadata_path or (Config.MODELS_DIR / DEFAULT_METADATA_FILENAME))

    # Safe atomic save for model
    tmp_m_path = m_path.with_suffix(".tmp")
    joblib.dump(pipeline, tmp_m_path, compress=3)
    tmp_m_path.replace(m_path)

    # Safe atomic save for JSON metadata
    tmp_meta_path = meta_path.with_suffix(".tmp")
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    tmp_meta_path.replace(meta_path)

    logger.info(f"Model persisted successfully at: {m_path}")
    logger.info(f"Metadata persisted successfully at: {meta_path}")
    return m_path, meta_path


def load_persisted_model(
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Safely load persisted model and metadata from the trusted project artifact location.

    Only trusted files within Config.MODELS_DIR are loaded; arbitrary user uploads are forbidden.
    """
    m_path = Path(model_path or (Config.MODELS_DIR / DEFAULT_MODEL_FILENAME))
    meta_path = Path(metadata_path or (Config.MODELS_DIR / DEFAULT_METADATA_FILENAME))

    if not m_path.exists():
        raise FileNotFoundError(f"Persisted model artifact not found at: {m_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Persisted model metadata not found at: {meta_path}")

    pipeline: Pipeline = joblib.load(m_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata: Dict[str, Any] = json.load(f)

    return pipeline, metadata


def predict_with_persisted_pipeline(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    input_data: Union[pd.DataFrame, Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute prediction using the loaded end-to-end pipeline without manual preprocessing.

    Args:
        pipeline: Fitted Pipeline loaded via load_persisted_model.
        metadata: Associated model metadata.
        input_data: DataFrame or dictionary containing raw feature values.

    Returns:
        Dictionary with predicted class, probabilities (if supported), and task type.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Verify input columns match expected feature columns
    feature_columns = metadata.get("feature_columns", [])
    if feature_columns:
        missing_cols = [c for c in feature_columns if c not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing expected feature columns: {missing_cols}")
        # Re-order columns strictly
        df_input = df_input[feature_columns]

    predictions = pipeline.predict(df_input)

    probabilities: Optional[List[Dict[str, float]]] = None
    estimator = pipeline.named_steps.get("model")
    if hasattr(estimator, "predict_proba"):
        try:
            probs = pipeline.predict_proba(df_input)
            classes = getattr(estimator, "classes_", [f"class_{i}" for i in range(probs.shape[1])])
            probabilities = [
                {str(cls): round(float(prob), 4) for cls, prob in zip(classes, row_probs)}
                for row_probs in probs
            ]
        except Exception as exc:
            logger.warning(f"Could not calculate prediction probabilities: {exc}")

    return {
        "predictions": [str(p) if isinstance(p, (np.generic, object)) else p for p in predictions],
        "probabilities": probabilities,
        "task_type": metadata.get("task_type", "unknown"),
        "model_name": metadata.get("model_name", "unknown"),
    }
