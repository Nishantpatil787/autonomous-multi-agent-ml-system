"""Reusable training, splitting, cross-validation, and evaluation utilities.

Provides leakage-free pipeline training, stratified splitting with fallback,
metric calculation, confusion matrix generation, and experiment ranking.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from tools import data_tools, model_tools, preprocessing_tools
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("TrainingTools")


def load_training_dataframe(
    csv_path: Union[str, Path],
    duplicate_strategy: str = "remove_before_training",
) -> Tuple[pd.DataFrame, int]:
    """Load dataset and apply in-memory duplicate strategy without altering the source CSV."""
    df = data_tools.load_csv(csv_path)
    initial_count = len(df)
    duplicates_removed = 0

    if duplicate_strategy == "remove_before_training":
        df = df.drop_duplicates(inplace=False).reset_index(drop=True)
        duplicates_removed = initial_count - len(df)
        logger.info(f"Duplicate handling: removed {duplicates_removed} duplicate rows in-memory.")

    return df, duplicates_removed


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    excluded_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, ensuring target is excluded and exclusions are dropped.

    Args:
        df: Input DataFrame.
        target_column: Name of target column.
        excluded_features: Optional list of column names to exclude.

    Returns:
        Tuple of (X DataFrame, y Series).

    Raises:
        ValueError: If target_column is not found in DataFrame.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    # Extract target series
    y = df[target_column].copy()

    # Safely handle missing values in target to prevent training failure
    if y.isna().any():
        non_null_vals = set(y.dropna().unique())
        if non_null_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1"}):
            # Binary indicator column (e.g. actively_hiring where NaN = 0)
            y = y.fillna(0).astype(int)
        else:
            # Drop rows with missing target labels
            valid_idx = y.dropna().index
            y = y.loc[valid_idx]
            df = df.loc[valid_idx]

    # Drop target column to prevent target leakage
    X = df.drop(columns=[target_column], inplace=False)
    assert target_column not in X.columns, f"Critical error: target '{target_column}' leaked into feature matrix."

    # Drop common CSV export artifacts (e.g. unnamed index columns)
    unnamed_cols = [c for c in X.columns if str(c).lower().startswith("unnamed:")]
    if unnamed_cols:
        X = X.drop(columns=unnamed_cols, inplace=False)

    # Drop any explicitly excluded features
    if excluded_features:
        cols_to_drop = [c for c in excluded_features if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop, inplace=False)

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool, str]:
    """Perform train/test split with stratified sampling and safe fallback for tiny classes."""
    class_counts = y.value_counts()
    min_class_samples = class_counts.min() if not class_counts.empty else 0

    # Stratification requires at least 2 samples per class
    if min_class_samples >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                True,
                f"Stratified split applied (smallest class count: {min_class_samples}).",
            )
        except Exception as exc:
            logger.warning(f"Stratification failed despite checks ({str(exc)}); falling back to random split.")

    # Fallback to unstratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    reason = (
        f"Non-stratified split used: smallest class has only {min_class_samples} sample(s) "
        "(stratification requires >= 2 samples per class)."
    )
    return X_train, X_test, y_train, y_test, False, reason


def build_model_pipeline(
    base_pipeline_config: Dict[str, Any],
    model_key: str,
) -> Tuple[Pipeline, bool]:
    """Assemble an end-to-end scikit-learn Pipeline [preprocessor -> estimator] with model-appropriate scaling.

    Args:
        base_pipeline_config: Base preprocessing configuration from Phase 3.
        model_key: Identifier of model in registry.

    Returns:
        Tuple of (un-fitted Pipeline, scale_numeric_flag).
    """
    model_def = model_tools.get_model_definition(model_key)
    scaling_required = bool(model_def["scaling_required"])

    # Clone configuration and adapt scaling strictly for this model
    model_prep_config = dict(base_pipeline_config)
    model_prep_config["scale_numeric"] = scaling_required

    preprocessor = preprocessing_tools.recreate_column_transformer_from_config(model_prep_config)
    estimator = model_tools.build_estimator(model_key)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    return pipeline, scaling_required


def evaluate_classifier(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Calculate standard classification evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }


def calculate_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """Compute confusion matrix as a JSON-serializable list of lists with class labels."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    labels.sort()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "matrix": cm.tolist(),
        "labels": [str(lbl) for lbl in labels],
    }


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute stratified k-fold cross validation using the full preprocessor+model pipeline."""
    class_counts = y.value_counts()
    min_class_samples = class_counts.min() if not class_counts.empty else 0

    if min_class_samples < 2:
        return {
            "cv_accuracy_mean": None,
            "cv_accuracy_std": None,
            "cv_f1_mean": None,
            "cv_folds": 0,
            "status": "skipped",
            "reason": f"Cross-validation skipped: minority class has only {min_class_samples} sample(s).",
        }

    # Reduce folds if smallest class has fewer samples than n_splits
    safe_splits = min(n_splits, min_class_samples)
    if safe_splits < 2:
        return {
            "cv_accuracy_mean": None,
            "cv_accuracy_std": None,
            "cv_f1_mean": None,
            "cv_folds": 0,
            "status": "skipped",
            "reason": "Cross-validation skipped: insufficient samples for 2 folds.",
        }

    cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=random_state)
    try:
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
            n_jobs=1,
            error_score="raise",
        )
        acc_mean = float(np.mean(scores["test_accuracy"]))
        acc_std = float(np.std(scores["test_accuracy"]))
        f1_mean = float(np.mean(scores["test_f1"]))
        return {
            "cv_accuracy_mean": round(acc_mean, 4),
            "cv_accuracy_std": round(acc_std, 4),
            "cv_f1_mean": round(f1_mean, 4),
            "cv_folds": safe_splits,
            "status": "completed",
            "reason": f"Stratified {safe_splits}-fold CV completed.",
        }
    except Exception as exc:
        logger.warning(f"Cross-validation failed ({str(exc)}); recording skipped status.")
        return {
            "cv_accuracy_mean": None,
            "cv_accuracy_std": None,
            "cv_f1_mean": None,
            "cv_folds": 0,
            "status": "failed",
            "reason": f"Cross-validation error: {str(exc)}",
        }


def rank_experiments(
    experiments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Rank successful experiments: Primary = F1-score descending, Secondary = Accuracy descending."""
    successful = [exp for exp in experiments if exp.get("status") == "success"]
    failed = [exp for exp in experiments if exp.get("status") != "success"]

    # Sort key: f1_score descending, accuracy descending, cv_f1_mean descending
    def sort_key(exp: Dict[str, Any]) -> Tuple[float, float, float]:
        f1 = exp.get("f1_score", 0.0) or 0.0
        acc = exp.get("accuracy", 0.0) or 0.0
        cv_f1 = exp.get("cv_f1_mean", 0.0) or 0.0
        return (f1, acc, cv_f1)

    successful.sort(key=sort_key, reverse=True)

    ranked_comparison: List[Dict[str, Any]] = []
    for rank, exp in enumerate(successful, start=1):
        ranked_comparison.append(
            {
                "rank": rank,
                "model_name": exp["model_name"],
                "accuracy": exp["accuracy"],
                "f1_score": exp["f1_score"],
                "precision": exp["precision"],
                "recall": exp["recall"],
                "cv_f1_mean": exp.get("cv_f1_mean"),
                "cv_accuracy_mean": exp.get("cv_accuracy_mean"),
                "training_time": exp["training_time"],
                "scaling": exp["preprocessing_scaling"],
            }
        )

    # Append failed models at bottom
    for exp in failed:
        ranked_comparison.append(
            {
                "rank": None,
                "model_name": exp["model_name"],
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "cv_f1_mean": None,
                "cv_accuracy_mean": None,
                "training_time": exp.get("training_time", 0.0),
                "scaling": exp.get("preprocessing_scaling", False),
                "error": exp.get("error_message"),
            }
        )

    best_model = successful[0] if successful else None
    return ranked_comparison, best_model
