"""Reusable training and evaluation operations (Phase 4).

Everything that touches scikit-learn for training lives here so the agents stay
thin orchestration layers. The most important invariant of this module is the
**leakage rule**:

    preprocessing and estimator are combined into a single sklearn `Pipeline`
    that is fitted on the training split only

    pipeline.fit(X_train, y_train)      -> imputers, encoders and scalers learn
                                            from the training data only
    pipeline.predict(X_test)            -> the test data is only transformed

Neither `evaluate_classifier` nor `run_cross_validation` ever receives raw,
pre-transformed data: they always receive the fitted-or-unfitted pipeline
together with the raw feature frame, which makes leakage structurally
impossible rather than merely unlikely.

Nothing in this module writes to the source CSV file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from tools import model_tools
from tools.model_tools import (
    ModelConfigurationError,
    ModelSelectionError,
    config_for_scaling,
)
from tools.preprocessing_tools import PreprocessingError, build_column_transformer
from utils import config
from utils.logger import setup_logger
from utils.state import (
    BestModelInfo,
    ConfusionMatrixData,
    CrossValidationConfig,
    CrossValidationResult,
    ExperimentStatus,
    ModelComparison,
    ModelComparisonEntry,
    PreprocessingConfig,
)

logger = setup_logger("tools.training_tools")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class TrainingToolError(Exception):
    """Base class for training tool errors."""


class TrainingDataError(TrainingToolError):
    """Raised when the training data cannot be prepared (missing target, ...)."""


class DatasetTooSmallError(TrainingToolError):
    """Raised when the dataset is too small to train or validate at all."""


# --------------------------------------------------------------------------- #
# Local records
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    """Outcome of the train/test split, including whether it was stratified."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_samples: int
    test_samples: int
    total_samples: int
    stratified: bool
    reason: str
    class_distribution_before: Dict[str, int] = field(default_factory=dict)
    class_distribution_train: Dict[str, int] = field(default_factory=dict)
    class_distribution_test: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def load_training_dataframe(dataset_path: str) -> pd.DataFrame:
    """Load the source CSV again for training.

    The Data Analysis Agent and the Preprocessing Agent are read-only, so the
    training step reads the file itself. This guarantees the training data is
    exactly what is on disk and that no agent has to carry a DataFrame inside
    the shared state.

    Args:
        dataset_path: Path to the CSV file.

    Returns:
        A freshly loaded DataFrame.

    Raises:
        TrainingDataError: The file cannot be read or is empty.
    """
    try:
        dataframe = pd.read_csv(dataset_path)
    except FileNotFoundError as error:
        raise TrainingDataError(f"Dataset file does not exist: {dataset_path}") from error
    except Exception as error:
        raise TrainingDataError(f"Could not read dataset '{dataset_path}': {error}") from error

    if dataframe.shape[0] == 0 or dataframe.shape[1] == 0:
        raise TrainingDataError(f"Dataset '{dataset_path}' is empty.")
    return dataframe


def prepare_features_and_target(
    dataframe: pd.DataFrame,
    target_column: Optional[str],
    excluded_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target.

    The target column is removed from the feature matrix **explicitly**, so it
    can never enter the model. Columns excluded by the Preprocessing Agent are
    removed as well, which keeps the feature frame aligned with the columns the
    `ColumnTransformer` knows about.

    Args:
        dataframe: The in-memory dataset (after duplicate handling).
        target_column: Name of the target column.
        excluded_columns: Columns the preprocessing phase excluded.

    Returns:
        A `(X, y)` tuple.

    Raises:
        TrainingDataError: The target is missing, empty or constant, or the
            remaining frame has no features.
    """
    if not target_column:
        raise TrainingDataError(
            "No target column is available, so the training data cannot be prepared."
        )
    if target_column not in dataframe.columns:
        raise TrainingDataError(
            f"Target column '{target_column}' is not present in the dataset. "
            f"Available columns: {list(dataframe.columns)}"
        )

    target = dataframe[target_column]
    if target.isna().any():
        raise TrainingDataError(
            f"Target column '{target_column}' contains {int(target.isna().sum())} missing "
            "value(s). Rows without a target cannot be used for supervised training."
        )
    if int(target.nunique()) < 2:
        raise TrainingDataError(
            f"Target column '{target_column}' has a single distinct value "
            f"({target.iloc[0]!r}); there is nothing to learn."
        )

    drop_columns = [target_column] + [
        column for column in (excluded_columns or []) if column != target_column
    ]
    present = [column for column in drop_columns if column in dataframe.columns]
    features = dataframe.drop(columns=present)
    if features.shape[1] == 0:
        raise TrainingDataError(
            "No feature column is left after removing the target and the excluded "
            "columns. Check the preprocessing configuration."
        )
    return features, target


def class_distribution(target: pd.Series) -> Dict[str, int]:
    """Return the sample count per class label, as JSON-safe strings."""
    counts = target.value_counts().to_dict()
    return {str(label): int(count) for label, count in sorted(counts.items(), key=str)}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def determine_safe_n_splits(
    target: pd.Series,
    requested: int = config.CV_FOLDS,
    minimum: int = config.CV_MIN_FOLDS,
) -> Tuple[int, str]:
    """Reduce the number of cross-validation folds to what the data supports.

    A stratified fold needs at least ``n_splits`` samples in every class. When
    that is not possible the fold count is lowered to the largest safe value and
    the reason is returned so it can be recorded instead of failing silently.

    Args:
        target: Target column.
        requested: Preferred number of folds.
        minimum: Lowest accepted number of folds.

    Returns:
        A `(n_splits, reason)` tuple. `reason` is empty when the requested value
        was already safe.

    Raises:
        DatasetTooSmallError: Even `minimum` folds are impossible.
    """
    counts = target.value_counts()
    smallest_class = int(counts.min()) if not counts.empty else 0
    total = int(target.shape[0])

    if smallest_class >= requested and total >= requested:
        return requested, ""

    if smallest_class < minimum or total < minimum:
        raise DatasetTooSmallError(
            f"Cross-validation is impossible: the smallest class has {smallest_class} "
            f"sample(s) out of {total} row(s), but at least {minimum} are required."
        )

    safe = smallest_class
    return safe, (
        f"reduced from {requested} to {safe} folds because the smallest class has only "
        f"{smallest_class} sample(s)"
    )


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify_if_possible: bool = True,
) -> SplitResult:
    """Split features and target into a training and a test set.

    Stratification is used for classification whenever it is possible. When the
    smallest class has fewer than `MIN_SAMPLES_PER_CLASS_FOR_STRATIFICATION`
    samples, or the dataset has fewer than `MIN_SAMPLES_FOR_STRATIFIED_SPLIT`
    rows, the split falls back to a plain random split and the reason is
    recorded in the returned `SplitResult`.

    Args:
        features: Feature frame (target already removed).
        target: Target column.
        test_size: Share of rows held out for testing.
        random_state: Seed that makes the split reproducible.
        stratify_if_possible: Set to False to always use a plain split.

    Returns:
        A `SplitResult` describing the split.

    Raises:
        DatasetTooSmallError: Not enough rows for the requested split.
    """
    total = int(features.shape[0])
    smallest_class = int(target.value_counts().min())

    reason = ""
    use_stratification = stratify_if_possible
    if use_stratification and (
        smallest_class < config.MIN_SAMPLES_PER_CLASS_FOR_STRATIFICATION
        or total < config.MIN_SAMPLES_FOR_STRATIFIED_SPLIT
    ):
        use_stratification = False
        reason = (
            f"stratified split skipped: smallest class has {smallest_class} sample(s) "
            f"and the dataset has {total} row(s); a plain random split was used instead"
        )
        logger.warning(reason)

    if total < 2:
        raise DatasetTooSmallError(
            f"At least 2 rows are needed for a train/test split, got {total}."
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target if use_stratification else None,
        )
    except ValueError as error:
        raise DatasetTooSmallError(
            f"The requested split (test_size={test_size}) is not possible for "
            f"{total} row(s): {error}"
        ) from error

    result = SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_samples=int(X_train.shape[0]),
        test_samples=int(X_test.shape[0]),
        total_samples=total,
        stratified=use_stratification,
        reason=reason,
        class_distribution_before=class_distribution(target),
        class_distribution_train=class_distribution(y_train),
        class_distribution_test=class_distribution(y_test),
    )
    logger.info(
        "Split %d rows into %d train / %d test (stratified=%s)",
        total,
        result.train_samples,
        result.test_samples,
        result.stratified,
    )
    return result


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_model_pipeline(
    preprocessing_config: PreprocessingConfig,
    model_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    *,
    step_names: Tuple[str, str] = ("preprocessor", "model"),
) -> Pipeline:
    """Build the leakage-safe `preprocessor -> estimator` Pipeline for one model.

    The preprocessing configuration is passed by value: the function derives the
    scaling variant required by the model (`config_for_scaling`) instead of
    mutating the Phase 3 configuration stored in the shared state.

    Args:
        preprocessing_config: Configuration from the Preprocessing Agent.
        model_name: Registry name, e.g. "Support Vector Machine".
        parameters: Optional baseline parameter overrides.
        step_names: Names of the two pipeline steps.

    Returns:
        An **unfitted** scikit-learn `Pipeline`.

    Raises:
        ModelConfigurationError: The model is unknown or the configuration is broken.
    """
    definition = model_tools.get_model_definition(model_name)
    scaling_config = config_for_scaling(preprocessing_config, definition.scaling_required)

    preprocessor = build_column_transformer(scaling_config)
    estimator = model_tools.build_estimator(model_name, parameters)
    pipeline = Pipeline(steps=[(step_names[0], preprocessor), (step_names[1], estimator)])

    logger.debug(
        "Built pipeline for %s (scale_numeric=%s)",
        model_name,
        scaling_config.scale_numeric,
    )
    return pipeline


def build_preprocessing_variants(
    preprocessing_config: PreprocessingConfig,
    scale_numeric: bool,
) -> Tuple[PreprocessingConfig, Any]:
    """Return the scaling variant configuration and its unfitted transformer.

    Two variants are enough for Phase 4: scaled for Logistic Regression / KNN /
    SVM and unscaled for the tree-based models.

    Args:
        preprocessing_config: Configuration from the Preprocessing Agent.
        scale_numeric: Whether the variant scales numerical features.

    Returns:
        A `(config, column_transformer)` tuple; the transformer is unfitted.

    Raises:
        ModelConfigurationError: The variant cannot be derived.
    """
    variant = config_for_scaling(preprocessing_config, scale_numeric)
    return variant, build_column_transformer(variant)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_classifier(
    y_true: Any,
    y_pred: Any,
    *,
    average: str = config.METRIC_AVERAGE,
) -> Dict[str, float]:
    """Compute accuracy, precision, recall and F1 for a classifier.

    Weighted averages are used so the metrics stay meaningful for multiclass
    targets and for imbalanced binary targets. `zero_division=0` keeps precision
    and recall defined when a model never predicts a rare class instead of
    raising or returning NaN.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Averaging strategy passed to precision/recall/F1.

    Returns:
        A dictionary with `accuracy`, `precision`, `recall` and `f1_score`.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, average=average, zero_division=config.ZERO_DIVISION
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average=average, zero_division=config.ZERO_DIVISION
            )
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, average=average, zero_division=config.ZERO_DIVISION)
        ),
    }
    return metrics


def calculate_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Sequence[Any]] = None,
) -> ConfusionMatrixData:
    """Build a JSON-safe confusion matrix.

    No plotting happens here; the Streamlit dashboard (Phase 9) will render the
    stored numbers.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional explicit label order.

    Returns:
        A `ConfusionMatrixData` record with string labels and integer counts.
    """
    if labels is None:
        # Union of the true and predicted labels, de-duplicated while keeping the
        # original Python types so scikit-learn can match them. Passing the
        # concatenation instead would repeat labels and produce a matrix that is
        # larger than the number of classes.
        combined = list(pd.Series(np.asarray(y_true)).unique())
        for value in pd.Series(np.asarray(y_pred)).unique():
            if value not in combined:
                combined.append(value)
        labels = sorted(combined, key=str)

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return ConfusionMatrixData(
        labels=[str(label) for label in labels],
        matrix=[[int(value) for value in row] for row in matrix],
        total=int(matrix.sum()),
    )


def count_transformed_features(pipeline: Pipeline) -> Optional[int]:
    """Return the number of output columns of a fitted pipeline.

    Returns:
        The feature count after preprocessing, or None when the pipeline has
        not been fitted yet or does not expose feature names.
    """
    try:
        check_is_fitted(pipeline)
    except Exception:
        return None
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return None
    try:
        return int(len(preprocessor.get_feature_names_out()))
    except Exception:  # pragma: no cover - defensive, never hidden
        logger.debug("Could not read transformed feature names from the pipeline")
        return None


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def build_cross_validation_config(
    target: pd.Series,
    *,
    enabled: bool = True,
    n_splits: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> CrossValidationConfig:
    """Build the cross-validation configuration for a target column.

    The requested fold count is reduced automatically when the smallest class
    cannot support it, and the reason is stored in `adjusted_reason`.

    Args:
        target: Target column.
        enabled: Set to False to skip cross-validation entirely.
        n_splits: Preferred number of folds.
        random_state: Seed for the fold shuffling.

    Returns:
        A `CrossValidationConfig`. `enabled` is False when not even two
        stratified folds are possible, with the reason in `adjusted_reason`.

    Note:
        A dataset too small for cross-validation is **not** an error: it is
        reported as a disabled configuration so the caller can record the reason
        and keep the single train/test evaluation.
    """
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
    }
    if not enabled:
        return CrossValidationConfig(
            enabled=False,
            n_splits=0,
            shuffle=config.CV_SHUFFLE,
            random_state=random_state,
            scoring=scoring,
            adjusted_reason="cross-validation was disabled by the caller",
        )

    try:
        safe_folds, reason = determine_safe_n_splits(target, requested=n_splits)
    except DatasetTooSmallError as error:
        logger.warning("Cross-validation skipped: %s", error)
        return CrossValidationConfig(
            enabled=False,
            n_splits=0,
            shuffle=config.CV_SHUFFLE,
            random_state=random_state,
            scoring=scoring,
            adjusted_reason=str(error),
        )

    if reason:
        logger.warning("Cross-validation %s", reason)
    return CrossValidationConfig(
        enabled=True,
        n_splits=safe_folds,
        shuffle=config.CV_SHUFFLE,
        random_state=random_state,
        scoring=scoring,
        adjusted_reason=reason,
    )


def run_cross_validation(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv_config: CrossValidationConfig,
    model_name: str = "",
) -> CrossValidationResult:
    """Run baseline stratified cross-validation on the complete Pipeline.

    The *unfitted* pipeline is passed on purpose: `cross_validate` clones and
    fits it inside every fold, so the imputers, encoders and scalers only ever
    see that fold's training data. This is what prevents leakage inside the
    cross-validation itself.

    Args:
        pipeline: Unfitted `preprocessor -> estimator` Pipeline.
        features: Raw feature frame (target already removed).
        target: Target column.
        cv_config: Fold configuration from `build_cross_validation_config`.
        model_name: Model name, recorded in the result.

    Returns:
        A `CrossValidationResult` with the mean and standard deviation of the
        accuracy and weighted F1 across folds.

    Raises:
        PreprocessingError, ModelSelectionError, TrainingToolError: Any failure
            raised by scikit-learn or by the pipeline, so the caller can record
            the error instead of hiding it.
    """
    if not cv_config.enabled:
        return CrossValidationResult(
            model_name=model_name,
            n_splits=0,
            status="skipped",
            reason=cv_config.adjusted_reason or "cross-validation was disabled",
        )

    splitter = StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state,
    )

    logger.info(
        "Cross-validating %s with %d stratified folds (leakage safe: the pipeline is "
        "refitted inside every fold)",
        model_name,
        cv_config.n_splits,
    )
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=splitter,
        scoring=cv_config.scoring,
        n_jobs=1,          # sequential keeps the run reproducible
        error_score="raise",  # fail loudly instead of writing NaN into the report
    )

    accuracy_scores = [float(value) for value in scores["test_accuracy"]]
    f1_scores = [float(value) for value in scores["test_f1"]]
    return CrossValidationResult(
        model_name=model_name,
        n_splits=cv_config.n_splits,
        status="success",
        reason=cv_config.adjusted_reason,
        accuracy_mean=float(np.mean(accuracy_scores)),
        accuracy_std=float(np.std(accuracy_scores)),
        f1_mean=float(np.mean(f1_scores)),
        f1_std=float(np.std(f1_scores)),
        fold_accuracies=accuracy_scores,
    )


# --------------------------------------------------------------------------- #
# Comparison and ranking
# --------------------------------------------------------------------------- #
def rank_experiments(
    experiments: Sequence[Any],
    *,
    primary_metric: str = config.BEST_MODEL_PRIMARY_METRIC,
    secondary_metric: str = config.BEST_MODEL_SECONDARY_METRIC,
) -> ModelComparison:
    """Rank the experiments and build the structured comparison.

    Ranking rule (documented and deterministic):

    1. only experiments that produced metrics are ranked;
    2. primary metric descending (``f1_score`` by default);
    3. secondary metric descending (``accuracy``) as tie-breaker;
    4. remaining stable registry order, so equal scores never depend on timing.

    Because Phase 4 uses baseline parameters only, this identifies the best
    **baseline** model, not a tuned one.

    Args:
        experiments: `ExperimentResult` records.
        primary_metric: Field used for the main ranking.
        secondary_metric: Field used to break ties.

    Returns:
        A `ModelComparison` with the ranked entries and a printable table.
    """
    ranking_rule = (
        f"primary: {primary_metric} (descending); "
        f"secondary: {secondary_metric} (descending); "
        "experiments without metrics are excluded; Phase 4 uses baseline "
        "parameters only, so this is the best BASELINE model"
    )

    usable = [
        item
        for item in experiments
        if getattr(item, "status", None) in (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
        and getattr(item, primary_metric, None) is not None
    ]
    failed = len(experiments) - len(usable)

    def sort_key(item: Any) -> Tuple[float, float, str]:
        primary = getattr(item, primary_metric, None)
        secondary = getattr(item, secondary_metric, None)
        return (
            -float(primary) if primary is not None else 0.0,
            -float(secondary) if secondary is not None else 0.0,
            str(getattr(item, "model_name", "")),
        )

    ordered = sorted(usable, key=sort_key)

    entries: List[ModelComparisonEntry] = []
    for position, item in enumerate(ordered, start=1):
        entries.append(
            ModelComparisonEntry(
                model_name=item.model_name,
                model_type=item.model_type,
                preprocessing_scaling=item.preprocessing_scaling,
                accuracy=item.accuracy,
                precision=item.precision,
                recall=item.recall,
                f1_score=item.f1_score,
                cv_accuracy_mean=item.cv_accuracy_mean,
                cv_f1_mean=item.cv_f1_mean,
                total_time=item.total_time,
                rank=position,
            )
        )

    best_name = entries[0].model_name if entries else None
    notes: List[str] = []
    if failed:
        notes.append(f"{failed} experiment(s) failed and are not part of the ranking.")
    partial = [
        item.model_name
        for item in usable
        if getattr(item, "status", None) == ExperimentStatus.PARTIAL.value
    ]
    if partial:
        notes.append(
            "Cross-validation was skipped for: " + ", ".join(partial) + "."
        )

    comparison = ModelComparison(
        ranking_rule=ranking_rule,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        entries=entries,
        successful_count=len(usable),
        failed_count=failed,
        best_model_name=best_name,
        notes=notes,
    )
    comparison.table_text = build_comparison_table(comparison)
    return comparison


def build_comparison_table(comparison: ModelComparison) -> str:
    """Render the comparison as a fixed-width text table."""
    header = (
        f"{'rank':<5}{'model':<26}{'accuracy':>9}{'f1':>8}{'precision':>11}"
        f"{'recall':>9}{'cv acc':>9}{'cv f1':>8}{'scaling':>9}"
    )
    separator = "-" * len(header)
    rows = [header, separator]
    for entry in comparison.entries:
        rows.append(
            f"{str(entry.rank or '-'):<5}{entry.model_name:<26}"
            f"{_format_metric(entry.accuracy):>9}{_format_metric(entry.f1_score):>8}"
            f"{_format_metric(entry.precision):>11}{_format_metric(entry.recall):>9}"
            f"{_format_metric(entry.cv_accuracy_mean):>9}{_format_metric(entry.cv_f1_mean):>8}"
            f"{('on' if entry.preprocessing_scaling else 'off'):>9}"
        )
    if comparison.best_model_name:
        rows.append(separator)
        rows.append(f"Best baseline model: {comparison.best_model_name}")
    return "\n".join(rows)


def _format_metric(value: Optional[float]) -> str:
    """Format a metric for the table, showing '-' when it is missing."""
    return "-" if value is None else f"{value:.4f}"


def select_best_experiment(
    experiments: Sequence[Any],
    *,
    primary_metric: str = config.BEST_MODEL_PRIMARY_METRIC,
    secondary_metric: str = config.BEST_MODEL_SECONDARY_METRIC,
) -> Optional[BestModelInfo]:
    """Return the best baseline model record, or None when nothing succeeded.

    Args:
        experiments: `ExperimentResult` records.
        primary_metric: Field used for the main ranking.
        secondary_metric: Field used to break ties.

    Returns:
        A `BestModelInfo`, or None when no experiment produced metrics.
    """
    comparison = rank_experiments(
        experiments, primary_metric=primary_metric, secondary_metric=secondary_metric
    )
    if not comparison.entries:
        logger.warning("No experiment produced metrics, so there is no best model")
        return None

    best_name = comparison.best_model_name
    winner = next(item for item in experiments if item.model_name == best_name)
    return BestModelInfo(
        model_name=winner.model_name,
        experiment_id=winner.experiment_id,
        f1_score=getattr(winner, primary_metric, None),
        accuracy=getattr(winner, secondary_metric, None),
        cv_f1_mean=winner.cv_f1_mean,
        preprocessing_scaling=winner.preprocessing_scaling,
        parameters=dict(winner.parameters),
        selection_reason=(
            f"highest {primary_metric} ({_format_metric(getattr(winner, primary_metric, None))}) "
            f"among {comparison.successful_count} successful baseline experiment(s); "
            f"tie-break on {secondary_metric}"
        ),
    )


# --------------------------------------------------------------------------- #
# Experiment bookkeeping
# --------------------------------------------------------------------------- #
def build_experiment_id(model_name: str, parameters: Optional[Dict[str, Any]]) -> str:
    """Build a stable, readable identifier for one experiment.

    The id is **deterministic**: it is derived from the model name and a short
    fingerprint of its parameters instead of a random UUID. Two runs of the same
    configuration therefore produce the same ids, which is what makes the
    recorded experiment list reproducible. Different parameter sets still get
    different ids, so the ids stay unique within a run.

    Args:
        model_name: Display name of the model, e.g. ``"Random Forest"``.
        parameters: The effective parameters the model was built with.

    Returns:
        An identifier such as ``"exp_random_forest_3f9a1c"``.
    """
    slug = "_".join(model_name.lower().split())
    if not parameters:
        return f"exp_{slug}_default"
    fingerprint = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:6]
    return f"exp_{slug}_{fingerprint}"


# --------------------------------------------------------------------------- #
# Error helpers
# --------------------------------------------------------------------------- #
def describe_training_error(error: Exception) -> str:
    """Turn an exception into a single-line, loggable message."""
    return f"{type(error).__name__}: {error}"