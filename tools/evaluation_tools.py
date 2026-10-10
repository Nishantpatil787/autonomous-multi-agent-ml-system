"""Reusable evaluation and interpretation operations (Phase 5).

This module contains **no training code**. Every function is a pure, side-effect
free operation on the records the Training Agent already produced
(`ExperimentResult`, `CrossValidationResult`, `ModelComparison`), so the
Evaluation Agent can be re-run on a stored state without touching scikit-learn
and without re-fitting a single model.

Three principles shape the code:

1. **Observation, interpretation and hypothesis are kept apart.**
   A measured number (`f1_gap`) is stored next to its reporting category
   (`f1_gap_classification`) and next to the sentence that explains it
   (`gap_interpretation`). Nothing is ever collapsed into a verdict.

2. **A difference is not a diagnosis.**
   The gap between the hold-out score and the cross-validation mean is
   classified with the reporting thresholds in `utils.config`. Those
   thresholds are heuristics for wording a number; they are not statistical
   tests, and this module never concludes that a model "is overfitting" from
   them. Where a larger difference deserves attention, the cautious
   formulation in `config.OVERFITTING_CAUTION_NOTE` is used.

3. **Missing data is reported, never invented.**
   Without a cross-validation result there is no gap (and no gap claim); without
   a confusion matrix there is no error pattern. Per-class precision/recall/F1
   are *derived* from the stored confusion matrix where possible - the project
   stores no predictions and no training-set metrics, so nothing is recomputed
   by retraining a model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils import config
from utils.logger import setup_logger
from utils.state import (
    ClassMetrics,
    ConfusionMatrixAnalysis,
    ConfusionMatrixData,
    ConfusionPair,
    EvaluationStatus,
    EvaluationReport,
    ExecutionTimeAnalysis,
    ExperimentCount,
    ExperimentStatus,
    GapAnalysis,
    GapClassification,
    ModelComparison,
    ModelEvaluation,
    ModelRecommendation,
)

logger = setup_logger("tools.evaluation_tools")

# Decimal places used for every derived number. Rounding keeps the report
# readable *and* byte-identical between two identical runs.
ROUNDING = 6


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class EvaluationToolError(Exception):
    """Base class for evaluation tool errors."""


class EvaluationDataError(EvaluationToolError):
    """Raised when the evaluation records are malformed or inconsistent."""


# --------------------------------------------------------------------------- #
# Hold-out vs cross-validation gap
# --------------------------------------------------------------------------- #
def calculate_cv_holdout_gap(
    holdout_value: Optional[float],
    cv_value: Optional[float],
) -> Optional[float]:
    """Return `holdout - cv` for one metric.

    The sign carries the direction: a negative gap means the hold-out score is
    *lower* than the cross-validation mean. No absolute value is taken here, so
    the caller keeps the information needed to describe the direction.

    Args:
        holdout_value: Metric measured on the hold-out test split.
        cv_value: Mean of the same metric across cross-validation folds.

    Returns:
        The signed difference rounded to `ROUNDING` decimals, or None when
        either value is missing - a gap is never estimated.
    """
    if holdout_value is None or cv_value is None:
        return None
    return round(float(holdout_value) - float(cv_value), ROUNDING)


def classify_performance_gap(
    gap: Optional[float],
    *,
    close_threshold: float = config.GAP_CLOSE_THRESHOLD,
    substantial_threshold: float = config.GAP_SUBSTANTIAL_THRESHOLD,
) -> GapClassification:
    """Classify a signed gap into a reporting category.

    Args:
        gap: Signed difference from `calculate_cv_holdout_gap`.
        close_threshold: Upper bound (exclusive) of the 'close' category.
        substantial_threshold: Lower bound (inclusive) of 'substantial'.

    Returns:
        A `GapClassification`. `NOT_AVAILABLE` when no gap could be computed.
        `NOT_AVAILABLE` is also returned when the thresholds contradict each
        other, because silently picking one of them would misreport the data.

    Note:
        The categories describe the **size of a difference**, nothing else. They
        are never used as evidence of overfitting.
    """
    if gap is None:
        return GapClassification.NOT_AVAILABLE
    if close_threshold < 0 or substantial_threshold < close_threshold:
        raise EvaluationDataError(
            f"Invalid gap thresholds: close={close_threshold}, "
            f"substantial={substantial_threshold}; expected "
            f"0 <= close < substantial."
        )

    magnitude = abs(float(gap))
    if magnitude < close_threshold:
        return GapClassification.CLOSE
    if magnitude < substantial_threshold:
        return GapClassification.MODERATE
    return GapClassification.SUBSTANTIAL


def _describe_gap(
    metric_name: str,
    holdout: Optional[float],
    cv: Optional[float],
    gap: Optional[float],
    classification: GapClassification,
) -> str:
    """Build one cautious sentence about a hold-out vs CV difference.

    The wording follows the pattern demanded by the project: state the numbers,
    state the direction, state the category under the configured threshold, and
    stop there.
    """
    if gap is None or classification is GapClassification.NOT_AVAILABLE:
        return (
            f"{metric_name}: the hold-out score is {_fmt(holdout)} and no matching "
            "cross-validation score is available, so no difference is reported."
        )

    direction = "lower than" if gap < 0 else "higher than"
    magnitude = abs(gap)
    if gap == 0:
        direction = "equal to"
        magnitude = 0.0

    if classification is GapClassification.CLOSE:
        tail = (
            "the difference is small under the configured reporting threshold, so the "
            "hold-out score and the cross-validation mean are in close agreement."
        )
    elif classification is GapClassification.MODERATE:
        tail = (
            "the difference is moderate under the configured reporting threshold; it is "
            "reported as an observation and not interpreted as a sign of overfitting."
        )
    else:
        tail = config.OVERFITTING_CAUTION_NOTE

    return (
        f"{metric_name}: the hold-out score is {_fmt(holdout)} and the cross-validation "
        f"mean is {_fmt(cv)}, which is {magnitude:.4f} {direction} the hold-out score; "
        f"the gap is classified as '{classification.value}'. {tail}"
    )


# --------------------------------------------------------------------------- #
# Confusion matrix analysis
# --------------------------------------------------------------------------- #
def _validate_matrix(
    labels: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Validate a confusion matrix and return it as a list of int rows.

    Raises:
        EvaluationDataError: The matrix is not square, does not match the labels
            or contains a negative or non-integer count. A malformed matrix is
            reported, never silently patched.
    """
    size = len(labels)
    if size == 0:
        raise EvaluationDataError("The confusion matrix has no labels.")
    if len(matrix) != size:
        raise EvaluationDataError(
            f"The confusion matrix has {len(matrix)} row(s) but {size} label(s)."
        )

    rows: List[List[int]] = []
    for label, row in zip(labels, matrix):
        if len(row) != size:
            raise EvaluationDataError(
                f"Row '{label}' of the confusion matrix has {len(row)} value(s) but "
                f"{size} label(s) are declared."
            )
        converted: List[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvaluationDataError(
                    f"The confusion matrix contains a non-numeric value: {value!r}."
                )
            if value < 0:
                raise EvaluationDataError(
                    f"The confusion matrix contains a negative count: {value!r}."
                )
            converted.append(int(value))
        rows.append(converted)
    return rows


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Divide, returning None instead of raising or producing NaN/inf."""
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), ROUNDING)


def _derive_class_metrics(
    labels: Sequence[str],
    rows: Sequence[Sequence[int]],
) -> List[ClassMetrics]:
    """Derive per-class precision, recall and F1 from a confusion matrix.

    Each label is treated once as the positive class, which is symmetric and
    needs no domain assumption about which label is "interesting".

    Args:
        labels: Label order used by the matrix.
        rows: Validated matrix rows.

    Returns:
        One `ClassMetrics` record per label, in the order of `labels`.
    """
    total = sum(sum(row) for row in rows)
    records: List[ClassMetrics] = []

    for index, label in enumerate(labels):
        true_positives = rows[index][index]
        support = sum(rows[index])
        predicted_total = sum(row[index] for row in rows)
        false_positives = predicted_total - true_positives
        false_negatives = support - true_positives
        true_negatives = total - support - false_positives

        precision = _safe_ratio(true_positives, true_positives + false_positives)
        recall = _safe_ratio(true_positives, true_positives + false_negatives)
        if precision is None or recall is None or (precision + recall) == 0:
            f1_score = None
        else:
            # Harmonic mean, computed from the two ratios above.
            f1_score = round(2 * precision * recall / (precision + recall), ROUNDING)

        records.append(
            ClassMetrics(
                label=str(label),
                support=int(support),
                true_positives=int(true_positives),
                false_positives=int(false_positives),
                false_negatives=int(false_negatives),
                true_negatives=int(true_negatives),
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                derived=True,
            )
        )
    return records


def _binary_error_pattern(class_metrics: Sequence[ClassMetrics]) -> str:
    """Describe the binary error pattern without assuming a positive class."""
    hardest = max(class_metrics, key=lambda item: item.false_negatives + item.false_positives)
    errors = hardest.false_negatives + hardest.false_positives
    if errors == 0:
        return (
            "All hold-out samples were classified correctly, so there is no error "
            "pattern to describe."
        )

    if hardest.false_negatives > hardest.false_positives:
        comparison = (
            f"false negatives ({hardest.false_negatives}) exceed false positives "
            f"({hardest.false_positives})"
        )
    elif hardest.false_positives > hardest.false_negatives:
        comparison = (
            f"false positives ({hardest.false_positives}) exceed false negatives "
            f"({hardest.false_negatives})"
        )
    else:
        comparison = (
            f"false positives and false negatives are equal "
            f"({hardest.false_positives} each)"
        )

    return (
        f"Reading the label '{hardest.label}' as the positive class, {comparison}, so "
        f"the model accounts for most of its {errors} mistake(s) on that label. This is "
        "a count of errors only: the project has no domain information, so no "
        "real-world consequence is attached to this direction."
    )


def _multiclass_error_pattern(
    labels: Sequence[str],
    class_metrics: Sequence[ClassMetrics],
    pairs: Sequence[ConfusionPair],
    total_errors: int,
) -> str:
    """Describe the dominant multiclass error pattern.

    `total_errors` is the number of misclassified samples (the off-diagonal
    sum). Per class, false negatives and false positives are reported separately
    instead of added together, because a single mistake is counted once as a
    false negative of the true class and once as a false positive of the
    predicted class.
    """
    if total_errors == 0:
        return (
            "All hold-out samples were classified correctly, so there is no error "
            "pattern to describe."
        )

    # Classes without a single mistake are dropped before the list is truncated,
    # so "the classes with the most errors" never names a perfect class.
    hardest = [
        item
        for item in sorted(
            class_metrics,
            key=lambda entry: (
                -(entry.false_negatives + entry.false_positives),
                entry.label,
            ),
        )
        if item.false_negatives + item.false_positives > 0
    ][: config.TOP_ERROR_CLASSES]

    classes_text = ", ".join(
        f"'{item.label}' ({item.false_negatives} missed and "
        f"{item.false_positives} wrongly assigned)"
        for item in hardest
    ) or "no single class dominates the errors"
    # The pairs are listed in the compact "N 'true' predicted as 'predicted'"
    # form so the sentence and the printed report line share one wording.
    pair_text = (
        "; ".join(
            f"{pair.count} '{pair.true_label}' predicted as "
            f"'{pair.predicted_label}'"
            for pair in pairs
        )
        or "no single confusion pair dominates the errors"
    )
    return (
        f"Across {len(labels)} classes the model misclassified {total_errors} hold-out "
        f"sample(s). The classes involved in the most errors are {classes_text}. "
        f"Confusion pairs: {pair_text}. No domain interpretation is added: "
        "the project does not know what these labels represent."
    )


def analyze_confusion_matrix(
    confusion: Optional[ConfusionMatrixData],
    *,
    model_name: str = "",
    top_pairs: int = config.TOP_CONFUSION_PAIRS,
) -> ConfusionMatrixAnalysis:
    """Interpret one hold-out confusion matrix.

    Args:
        confusion: The stored matrix, or None when the experiment has none.
        model_name: Model the matrix belongs to.
        top_pairs: How many dominant confusion pairs to keep.

    Returns:
        A `ConfusionMatrixAnalysis`. When `confusion` is None the record has
        `available=False` and explains what is missing instead of returning
        invented numbers.

    Raises:
        EvaluationDataError: The matrix is malformed (see `_validate_matrix`).
    """
    if confusion is None:
        return ConfusionMatrixAnalysis(
            model_name=model_name,
            available=False,
            per_class_metrics_note=config.PER_CLASS_UNAVAILABLE_NOTE,
            notes=[
                "No confusion matrix was stored for this experiment, so no error "
                "pattern and no per-class metrics can be reported.",
            ],
        )

    labels = [str(label) for label in confusion.labels]
    rows = _validate_matrix(labels, confusion.matrix)

    total = sum(sum(row) for row in rows)
    correct = sum(rows[index][index] for index in range(len(labels)))
    incorrect = total - correct

    class_metrics = _derive_class_metrics(labels, rows)

    pairs: List[ConfusionPair] = []
    for row_index, label in enumerate(labels):
        for column_index, predicted in enumerate(labels):
            if row_index == column_index:
                continue
            count = rows[row_index][column_index]
            if count:
                pairs.append(
                    ConfusionPair(
                        true_label=label, predicted_label=predicted, count=int(count)
                    )
                )
    # Deterministic order: most errors first, then label names.
    pairs.sort(key=lambda pair: (-pair.count, pair.true_label, pair.predicted_label))
    dominant_pairs = pairs[: max(int(top_pairs), 0)]

    ranked_errors = sorted(
        class_metrics,
        key=lambda item: (-(item.false_negatives + item.false_positives), item.label),
    )
    classes_with_most_errors = [
        item.label
        for item in ranked_errors[: config.TOP_ERROR_CLASSES]
        if item.false_negatives + item.false_positives > 0
    ]

    if len(labels) == 2:
        error_pattern = _binary_error_pattern(class_metrics)
    elif len(labels) > 2:
        error_pattern = _multiclass_error_pattern(
            labels, class_metrics, dominant_pairs, incorrect
        )
    else:
        error_pattern = (
            f"The confusion matrix contains a single label ('{labels[0] if labels else ''}'), "
            "so no false positive / false negative breakdown is possible."
        )

    notes: List[str] = []
    if total == 0:
        notes.append(
            "The confusion matrix is empty (total = 0), so accuracy cannot be derived "
            "from it."
        )
    elif incorrect and correct:
        notes.append(
            f"{correct} of {total} hold-out sample(s) were classified correctly and "
            f"{incorrect} were not; one sample is worth about "
            f"{round(1 / total, ROUNDING):.4f} accuracy."
        )

    return ConfusionMatrixAnalysis(
        model_name=model_name,
        available=True,
        labels=labels,
        total_samples=int(total),
        correct_predictions=int(correct),
        incorrect_predictions=int(incorrect),
        accuracy_from_matrix=_safe_ratio(correct, total),
        class_metrics=class_metrics,
        dominant_confusion_pairs=dominant_pairs,
        classes_with_most_errors=classes_with_most_errors,
        error_pattern=error_pattern,
        per_class_metrics_note=config.PER_CLASS_DERIVATION_NOTE,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Model comparison and ranking
# --------------------------------------------------------------------------- #
def _index_cross_validation(
    cross_validation_results: Sequence[Any],
) -> Dict[str, Any]:
    """Index the usable cross-validation records by model name."""
    usable = {
        ExperimentStatus.SUCCESS.value,
        ExperimentStatus.PARTIAL.value,
    }
    indexed: Dict[str, Any] = {}
    for result in cross_validation_results or []:
        name = getattr(result, "model_name", "")
        if not name or getattr(result, "status", "") not in usable:
            continue
        if getattr(result, "f1_mean", None) is None and getattr(
            result, "accuracy_mean", None
        ) is None:
            continue
        indexed.setdefault(str(name), result)
    return indexed


def compare_models(
    experiments: Sequence[Any],
    cross_validation_results: Sequence[Any] = (),
    *,
    primary_metric: str = config.BEST_MODEL_PRIMARY_METRIC,
    secondary_metric: str = config.BEST_MODEL_SECONDARY_METRIC,
) -> List[ModelEvaluation]:
    """Turn the Phase 4 experiments into per-model evaluation records.

    Every experiment becomes exactly one `ModelEvaluation`, including the ones
    that failed: a failed experiment is part of the evaluation story and is
    marked as such instead of disappearing.

    The cross-validation numbers are read from `CrossValidationResult` when it is
    available (it also carries the standard deviations) and fall back to the
    copies stored on the experiment otherwise.

    Args:
        experiments: `ExperimentResult` records from the Training Agent.
        cross_validation_results: `CrossValidationResult` records.
        primary_metric: Ranking metric, recorded in the evaluation status.
        secondary_metric: Tie-break metric.

    Returns:
        The evaluations in the order the experiments were run.
    """
    cv_by_model = _index_cross_validation(cross_validation_results)
    evaluations: List[ModelEvaluation] = []

    for experiment in experiments or []:
        experiment_status = getattr(experiment, "status", "") or ""
        has_metrics = getattr(experiment, primary_metric, None) is not None
        cv_result = cv_by_model.get(str(getattr(experiment, "model_name", "")))

        cv_f1 = _first_not_none(
            getattr(cv_result, "f1_mean", None), getattr(experiment, "cv_f1_mean", None)
        )
        cv_accuracy = _first_not_none(
            getattr(cv_result, "accuracy_mean", None),
            getattr(experiment, "cv_accuracy_mean", None),
        )

        f1_gap = calculate_cv_holdout_gap(getattr(experiment, "f1_score", None), cv_f1)
        accuracy_gap = calculate_cv_holdout_gap(
            getattr(experiment, "accuracy", None), cv_accuracy
        )
        f1_classification = classify_performance_gap(f1_gap)
        accuracy_classification = classify_performance_gap(accuracy_gap)

        notes: List[str] = []
        if experiment_status == ExperimentStatus.FAILED.value:
            evaluation_status = EvaluationStatus.FAILED.value
            notes.append(
                "This experiment failed during training and produced no metrics; it is "
                f"excluded from the ranking. Reported error: "
                f"{getattr(experiment, 'error_message', '') or 'not recorded'}."
            )
        elif not has_metrics:
            evaluation_status = EvaluationStatus.SKIPPED.value
            notes.append(
                f"This experiment reported the status '{experiment_status}' but no "
                f"{primary_metric}, so it cannot be compared."
            )
        elif f1_gap is None or accuracy_gap is None:
            evaluation_status = EvaluationStatus.PARTIAL.value
            notes.append(
                "Hold-out metrics are available but no matching cross-validation score "
                "was recorded, so the hold-out vs cross-validation gap cannot be "
                "reported for this model."
            )
        elif experiment_status == ExperimentStatus.PARTIAL.value:
            evaluation_status = EvaluationStatus.PARTIAL.value
        else:
            evaluation_status = EvaluationStatus.COMPLETED.value

        if has_metrics and getattr(experiment, "training_time", None) is None:
            notes.append("No training time was recorded for this experiment.")

        evaluations.append(
            ModelEvaluation(
                rank=None,
                model_name=str(getattr(experiment, "model_name", "") or ""),
                model_type=str(getattr(experiment, "model_type", "") or ""),
                experiment_id=str(getattr(experiment, "experiment_id", "") or ""),
                experiment_status=experiment_status,
                evaluation_status=evaluation_status,
                preprocessing_scaling=bool(
                    getattr(experiment, "preprocessing_scaling", False)
                ),
                accuracy=getattr(experiment, "accuracy", None),
                precision=getattr(experiment, "precision", None),
                recall=getattr(experiment, "recall", None),
                f1_score=getattr(experiment, "f1_score", None),
                cv_accuracy_mean=cv_accuracy,
                cv_accuracy_std=getattr(cv_result, "accuracy_std", None),
                cv_f1_mean=cv_f1,
                cv_f1_std=getattr(cv_result, "f1_std", None),
                cv_n_splits=int(getattr(cv_result, "n_splits", 0) or 0),
                f1_gap=f1_gap,
                accuracy_gap=accuracy_gap,
                f1_gap_classification=f1_classification.value,
                accuracy_gap_classification=accuracy_classification.value,
                gap_interpretation=" ".join(
                    part
                    for part in (
                        _describe_gap(
                            "F1",
                            getattr(experiment, "f1_score", None),
                            cv_f1,
                            f1_gap,
                            f1_classification,
                        ),
                        _describe_gap(
                            "Accuracy",
                            getattr(experiment, "accuracy", None),
                            cv_accuracy,
                            accuracy_gap,
                            accuracy_classification,
                        ),
                    )
                    if part
                ),
                training_time=float(getattr(experiment, "training_time", 0.0) or 0.0),
                total_time=float(getattr(experiment, "total_time", 0.0) or 0.0),
                notes=notes,
            )
        )

    logger.debug("Compared %d experiment(s) into evaluations", len(evaluations))
    return evaluations


def rank_models(
    model_evaluations: Sequence[ModelEvaluation],
    *,
    primary_metric: str = config.BEST_MODEL_PRIMARY_METRIC,
    secondary_metric: str = config.BEST_MODEL_SECONDARY_METRIC,
) -> List[ModelEvaluation]:
    """Rank the evaluations and assign `rank`, using the Phase 4 rule.

    The rule is intentionally identical to `training_tools.rank_experiments`
    (primary metric descending, secondary metric descending, model name as the
    final deterministic tie-break) so that the evaluation report can never
    announce a different winner than the training report.

    Args:
        model_evaluations: Records from `compare_models`.
        primary_metric: Main ranking metric.
        secondary_metric: Tie-break metric.

    Returns:
        A new list ordered by rank. Evaluations without metrics are moved to the
        end and keep `rank=None`.
    """

    def sort_key(item: ModelEvaluation) -> Tuple[float, float, str]:
        primary = getattr(item, primary_metric, None)
        secondary = getattr(item, secondary_metric, None)
        return (
            -float(primary) if primary is not None else 1.0,
            -float(secondary) if secondary is not None else 1.0,
            str(item.model_name),
        )

    ordered = sorted(model_evaluations, key=sort_key)
    ranked: List[ModelEvaluation] = []
    position = 0
    for item in ordered:
        if getattr(item, primary_metric, None) is None:
            item.rank = None
        else:
            position += 1
            item.rank = position
        ranked.append(item)
    return ranked


def build_evaluation_table(model_evaluations: Sequence[ModelEvaluation]) -> str:
    """Render the Phase 5 comparison table as fixed-width text.

    The table is what the CLI report and the tests read, so every column comes
    from a stored field - nothing is recomputed here.
    """
    header = (
        f"{'rank':<5}{'model':<24}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}"
        f"{'cv acc':>8}{'cv f1':>8}{'f1 gap':>8}{'acc gap':>8}"
        f"{'train s':>9}{'total s':>9}  status"
    )
    rows = [header, "-" * len(header)]
    for item in model_evaluations:
        rows.append(
            f"{str(item.rank if item.rank is not None else '-'):<5}"
            f"{item.model_name:<24}"
            f"{_fmt(item.accuracy):>7}{_fmt(item.precision):>7}{_fmt(item.recall):>7}"
            f"{_fmt(item.f1_score):>7}{_fmt(item.cv_accuracy_mean):>8}"
            f"{_fmt(item.cv_f1_mean):>8}{_fmt(item.f1_gap):>8}"
            f"{_fmt(item.accuracy_gap):>8}"
            f"{item.training_time:>9.4f}{item.total_time:>9.4f}"
            f"  {item.evaluation_status}"
        )
    if not model_evaluations:
        rows.append("  (no experiment was evaluated)")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Gap analysis
# --------------------------------------------------------------------------- #
def analyze_performance_gaps(
    model_evaluations: Sequence[ModelEvaluation],
    *,
    metric: str = "f1",
    close_threshold: float = config.GAP_CLOSE_THRESHOLD,
    substantial_threshold: float = config.GAP_SUBSTANTIAL_THRESHOLD,
) -> GapAnalysis:
    """Aggregate the hold-out vs cross-validation differences of all models.

    Args:
        model_evaluations: Ranked evaluations from `rank_models`.
        metric: Which gap to aggregate, `f1` or `accuracy`.
        close_threshold: Upper bound (exclusive) of the 'close' category.
        substantial_threshold: Lower bound (inclusive) of 'substantial'.

    Returns:
        A `GapAnalysis`. `models_without_gap` lists the models for which no gap
        exists, so a missing number is visible instead of being skipped.
    """
    gap_field = f"{metric}_gap"
    classification_field = f"{metric}_gap_classification"
    label = "F1-score" if metric == "f1" else "Accuracy"
    lower_label = "F1-score" if metric == "f1" else "accuracy"

    close: List[str] = []
    moderate: List[str] = []
    substantial: List[str] = []
    without: List[str] = []
    notes: List[str] = []

    largest_gap: Optional[float] = None
    largest_model = ""
    with_gap = 0

    for item in model_evaluations:
        gap = getattr(item, gap_field, None)
        classification = getattr(item, classification_field, "")
        if gap is None or classification == GapClassification.NOT_AVAILABLE.value:
            without.append(item.model_name)
            continue

        with_gap += 1
        if classification == GapClassification.CLOSE.value:
            close.append(item.model_name)
        elif classification == GapClassification.MODERATE.value:
            moderate.append(item.model_name)
        else:
            substantial.append(item.model_name)

        if largest_gap is None or abs(gap) > abs(largest_gap):
            largest_gap = gap
            largest_model = item.model_name

    if with_gap == 0:
        summary = (
            f"No {lower_label} gap could be computed for any of the "
            f"{len(model_evaluations)} evaluated model(s): no matching cross-validation "
            "score was recorded."
        )
        interpretation = (
            "Without cross-validation results this run cannot say whether the hold-out "
            "score is representative; it is a single measurement without a reference."
        )
    else:
        parts = [
            f"{len(close)} model(s) in close agreement",
            f"{len(moderate)} model(s) with a moderate difference",
            f"{len(substantial)} model(s) with a substantial difference",
        ]
        summary = (
            f"{label} gaps (hold-out minus cross-validation mean) for {with_gap} "
            f"evaluated model(s): " + ", ".join(parts) + "."
        )
        if substantial:
            interpretation = config.OVERFITTING_CAUTION_NOTE
        elif moderate:
            interpretation = (
                f"The largest observed {lower_label} gap is "
                f"{abs(largest_gap):.4f} for '{largest_model}'. Differences of this size "
                "are reported as an observation; with a single split they are within the "
                "range that sampling variation can produce."
            )
        else:
            interpretation = (
                f"The largest observed {lower_label} gap is "
                f"{abs(largest_gap):.4f} for '{largest_model}', below the configured "
                "close-agreement threshold."
            )

    if without:
        notes.append(
            "No gap reported for: " + ", ".join(without) + " (missing cross-validation "
            "result or missing metric)."
        )

    return GapAnalysis(
        close_threshold=close_threshold,
        substantial_threshold=substantial_threshold,
        rule=config.GAP_CLASSIFICATION_RULE,
        disclaimer=config.GAP_HEURISTIC_DISCLAIMER,
        models_with_gap=with_gap,
        models_without_gap=without,
        close_agreement_models=close,
        moderate_difference_models=moderate,
        substantial_difference_models=substantial,
        metric=metric,
        largest_absolute_gap=largest_gap,
        largest_absolute_gap_model=largest_model,
        summary=summary,
        interpretation=interpretation,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Execution time and experiment counts
# --------------------------------------------------------------------------- #
def analyze_execution_time(experiments: Sequence[Any]) -> ExecutionTimeAnalysis:
    """Summarise how long the baseline experiments took.

    Only experiments that produced metrics are timed: a model that crashed
    after a fraction of a second would otherwise distort the average. The
    failed ones are still counted in `ExperimentCount`.

    Args:
        experiments: `ExperimentResult` records.

    Returns:
        An `ExecutionTimeAnalysis` with the total, the average and the fastest
        and slowest successful model. Ties are broken by model name so the
        result is deterministic.
    """
    timed = [
        item
        for item in experiments or []
        if getattr(item, "status", "") in (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
        and getattr(item, "f1_score", None) is not None
    ]

    total_time = round(sum(float(getattr(item, "total_time", 0.0) or 0.0) for item in timed), ROUNDING)
    average = _safe_ratio(total_time, len(timed)) or 0.0

    fastest = ""
    slowest = ""
    fastest_time = 0.0
    slowest_time = 0.0
    if timed:
        ordered = sorted(
            timed, key=lambda item: (float(item.total_time or 0.0), str(item.model_name))
        )
        fastest, slowest = ordered[0].model_name, ordered[-1].model_name
        fastest_time = round(float(ordered[0].total_time or 0.0), ROUNDING)
        slowest_time = round(float(ordered[-1].total_time or 0.0), ROUNDING)

    return ExecutionTimeAnalysis(
        total_experiment_time=total_time,
        average_experiment_time=average,
        total_training_time=round(
            sum(float(getattr(item, "training_time", 0.0) or 0.0) for item in timed), ROUNDING
        ),
        total_prediction_time=round(
            sum(float(getattr(item, "prediction_time", 0.0) or 0.0) for item in timed), ROUNDING
        ),
        total_cross_validation_time=round(
            sum(float(getattr(item, "cv_time", 0.0) or 0.0) for item in timed), ROUNDING
        ),
        fastest_model=fastest,
        fastest_time=fastest_time,
        slowest_model=slowest,
        slowest_time=slowest_time,
        timed_experiments=len(timed),
        notes=[
            config.LIMITATION_SINGLE_RUN_TIMING,
            "The training, prediction and cross-validation times are copied from the "
            "Phase 4 experiment records; this agent re-measures nothing, and the "
            "total is their sum.",
        ],
    )


def count_experiments(experiments: Sequence[Any]) -> ExperimentCount:
    """Count the baseline experiments by outcome.

    `total_experiments` counts models evaluated once each. It is deliberately
    kept apart from `hyperparameter_experiments`, which stays 0 because Phase 5
    performs no search; the Phase 11 workflow comparison needs both numbers and
    they are not interchangeable.

    Args:
        experiments: `ExperimentResult` records.

    Returns:
        An `ExperimentCount` record.
    """
    items = list(experiments or [])
    successful_statuses = (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
    known_statuses = successful_statuses + (ExperimentStatus.FAILED.value,)

    successful = [
        item for item in items if getattr(item, "status", "") == ExperimentStatus.SUCCESS.value
    ]
    partial = [
        item for item in items if getattr(item, "status", "") == ExperimentStatus.PARTIAL.value
    ]
    # Everything that is not a success or a partial result - including a record
    # with an unknown status - is counted as a failure, so the counts always add
    # up to the number of attempts and nothing disappears silently.
    failed = [
        item
        for item in items
        if getattr(item, "status", "") not in successful_statuses
    ]
    unclassified = [
        item
        for item in items
        if getattr(item, "status", "") not in known_statuses
    ]

    note = (
        f"{len(items)} model(s) were trained once each with baseline parameters; "
        "0 hyperparameter configuration(s) were evaluated. 'Total experiments' counts "
        "models evaluated once, not hyperparameter trials."
    )
    if unclassified:
        note += (
            f" {len(unclassified)} record(s) carried an unknown status and are counted "
            "as failures."
        )

    return ExperimentCount(
        baseline_models_attempted=len(items),
        successful_experiments=len(successful),
        partial_experiments=len(partial),
        failed_experiments=len(failed),
        total_experiments=len(items),
        hyperparameter_experiments=0,
        failed_model_names=[str(item.model_name) for item in failed],
        note=note,
    )


# --------------------------------------------------------------------------- #
# Summary, recommendation, limitations
# --------------------------------------------------------------------------- #
def generate_evaluation_summary(model_evaluations: Sequence[ModelEvaluation]) -> str:
    """Build the one-paragraph performance summary of the best model.

    The paragraph only contains numbers that are present in the records; a
    missing cross-validation score is left out instead of being guessed.
    """
    ranked = [item for item in model_evaluations if item.rank == 1]
    if not ranked:
        return (
            "No model produced usable metrics, so no performance summary can be given."
        )
    best = ranked[0]

    parts = [
        f"{best.model_name} achieved a hold-out accuracy of {_fmt(best.accuracy)}, "
        f"precision {_fmt(best.precision)}, recall {_fmt(best.recall)} and an F1-score "
        f"of {_fmt(best.f1_score)}"
    ]
    if best.cv_f1_mean is not None:
        fold_text = (
            f" over {best.cv_n_splits} stratified fold(s)" if best.cv_n_splits else ""
        )
        parts.append(
            f"its cross-validation mean F1-score{fold_text} was {_fmt(best.cv_f1_mean)}"
        )
    if best.f1_gap is not None:
        direction = "lower" if best.f1_gap < 0 else "higher"
        parts.append(
            f"the hold-out F1-score is {abs(best.f1_gap):.4f} {direction} than the "
            f"cross-validation mean (classified as '{best.f1_gap_classification}')"
        )
    if best.total_time:
        parts.append(f"the experiment took {best.total_time:.4f} s")

    runner_up = next((item for item in model_evaluations if item.rank == 2), None)
    if runner_up is not None and runner_up.f1_score is not None:
        margin = float(best.f1_score) - float(runner_up.f1_score)
        parts.append(
            f"the next best model was {runner_up.model_name} with an F1-score of "
            f"{_fmt(runner_up.f1_score)}, a margin of {margin:.4f}"
        )

    return "; ".join(parts) + "."


def generate_model_recommendation(
    model_evaluations: Sequence[ModelEvaluation],
    *,
    primary_metric: str = config.BEST_MODEL_PRIMARY_METRIC,
    secondary_metric: str = config.BEST_MODEL_SECONDARY_METRIC,
) -> Optional[ModelRecommendation]:
    """Build the baseline recommendation from the ranked evaluations.

    Args:
        model_evaluations: Ranked evaluations from `rank_models`.
        primary_metric: Metric the recommendation is based on.
        secondary_metric: Metric used to break ties.

    Returns:
        A `ModelRecommendation` for the top-ranked model, or None when no model
        produced metrics.
    """
    best = next((item for item in model_evaluations if item.rank == 1), None)
    if best is None:
        logger.warning("No evaluated model produced metrics, so nothing can be recommended")
        return None

    evaluated = [
        item for item in model_evaluations if getattr(item, primary_metric, None) is not None
    ]
    tied = [
        item.model_name
        for item in evaluated
        if float(getattr(item, primary_metric)) == float(getattr(best, primary_metric))
    ]

    reason = (
        f"highest hold-out {primary_metric} ({_fmt(getattr(best, primary_metric))}) among "
        f"{len(evaluated)} successful baseline experiment(s)"
    )
    if len(tied) > 1:
        reason += (
            f"; the tie on {primary_metric} was broken on {secondary_metric} and then on "
            "the model name, following the Phase 4 ranking rule"
        )
    else:
        reason += f"; ties are broken on {secondary_metric}"

    return ModelRecommendation(
        best_baseline_model=best.model_name,
        experiment_id=best.experiment_id,
        model_type=best.model_type,
        selection_reason=reason,
        performance_summary=generate_evaluation_summary(model_evaluations),
        f1_score=best.f1_score,
        accuracy=best.accuracy,
        precision=best.precision,
        recall=best.recall,
        cv_f1_mean=best.cv_f1_mean,
        cv_accuracy_mean=best.cv_accuracy_mean,
        f1_gap=best.f1_gap,
        total_time=best.total_time,
        preprocessing_scaling=best.preprocessing_scaling,
        parameters={},
        confusion_matrix_error_pattern="",
        limitations=[],
    )


def generate_observations(
    model_evaluations: Sequence[ModelEvaluation],
    gap_analysis: Optional[GapAnalysis],
    confusion_analyses: Sequence[ConfusionMatrixAnalysis],
    execution_time: Optional[ExecutionTimeAnalysis],
    experiment_count: Optional[ExperimentCount],
) -> List[str]:
    """Collect the observed facts of the run, without interpreting them."""
    observations: List[str] = []

    ranked = [item for item in model_evaluations if item.rank is not None]
    if ranked:
        spread = max(float(item.f1_score) for item in ranked if item.f1_score is not None) - min(
            float(item.f1_score) for item in ranked if item.f1_score is not None
        )
        observations.append(
            f"{len(ranked)} model(s) were evaluated on the same hold-out split; the F1 "
            f"spread between the best and the weakest of them is {spread:.4f}."
        )

    for item in ranked[:3]:
        observations.append(
            f"{item.model_name}: hold-out accuracy {_fmt(item.accuracy)}, F1 "
            f"{_fmt(item.f1_score)}, cross-validation mean F1 {_fmt(item.cv_f1_mean)}, "
            f"F1 gap {_fmt(item.f1_gap)} ('{item.f1_gap_classification}'), "
            f"{item.total_time:.4f} s."
        )

    failed = [item for item in model_evaluations if item.evaluation_status == EvaluationStatus.FAILED.value]
    if failed:
        observations.append(
            "Experiment(s) without metrics: "
            + ", ".join(f"{item.model_name} ({item.experiment_status})" for item in failed)
            + "."
        )

    if gap_analysis is not None and gap_analysis.models_with_gap:
        observations.append(gap_analysis.summary)

    for analysis in confusion_analyses:
        if not analysis.available:
            observations.append(
                f"{analysis.model_name or 'A model'}: {config.PER_CLASS_UNAVAILABLE_NOTE}"
            )
            continue
        observations.append(
            f"{analysis.model_name}: {analysis.correct_predictions} of "
            f"{analysis.total_samples} hold-out sample(s) were classified correctly and "
            f"{analysis.incorrect_predictions} were not "
            f"(accuracy from the confusion matrix: "
            f"{_fmt(analysis.accuracy_from_matrix)})."
        )

    if execution_time is not None and execution_time.timed_experiments:
        observations.append(
            f"The evaluated experiments needed {execution_time.total_experiment_time:.4f} s "
            f"in total and {execution_time.average_experiment_time:.4f} s on average; "
            f"the fastest was {execution_time.fastest_model} "
            f"({execution_time.fastest_time:.4f} s) and the slowest was "
            f"{execution_time.slowest_model} ({execution_time.slowest_time:.4f} s)."
        )

    if experiment_count is not None:
        observations.append(
            f"{experiment_count.baseline_models_attempted} baseline model(s) were "
            f"attempted, {experiment_count.successful_experiments} succeeded, "
            f"{experiment_count.partial_experiments} were partial and "
            f"{experiment_count.failed_experiments} failed; "
            f"{experiment_count.hyperparameter_experiments} hyperparameter "
            "configuration(s) were evaluated."
        )

    return observations


def generate_limitations(
    *,
    model_evaluations: Sequence[ModelEvaluation],
    gap_analysis: Optional[GapAnalysis],
    confusion_analyses: Sequence[ConfusionMatrixAnalysis],
    execution_time: Optional[ExecutionTimeAnalysis],
    test_samples: int = 0,
) -> List[str]:
    """List what this baseline evaluation cannot claim.

    The list is data-driven: a limitation is only reported when it applies to
    the run that was actually executed.
    """
    limitations: List[str] = [config.LIMITATION_NO_TRAINING_METRICS, config.LIMITATION_NO_TUNING]

    if any(item.rank is not None for item in model_evaluations):
        limitations.append(config.LIMITATION_SINGLE_SPLIT)

    if test_samples and test_samples < 20:
        limitations.append(
            config.LIMITATION_SMALL_TEST_SPLIT.format(
                test_samples=test_samples, step=1 / test_samples
            )
        )

    if any(item.f1_gap is None for item in model_evaluations if item.rank is not None):
        limitations.append(
            "At least one evaluated model has no cross-validation result, so its "
            "hold-out score cannot be compared with a cross-validation mean."
        )

    if not any(analysis.available for analysis in confusion_analyses):
        limitations.append(config.PER_CLASS_UNAVAILABLE_NOTE)
    else:
        limitations.append(
            "Per-class precision, recall and F1 are derived from the confusion matrix "
            "of the hold-out split only; they are not validated against a second "
            "measurement."
        )

    if execution_time is not None and execution_time.timed_experiments:
        limitations.append(config.LIMITATION_SINGLE_RUN_TIMING)

    if gap_analysis is not None and not gap_analysis.models_with_gap:
        limitations.append(
            "No hold-out vs cross-validation comparison is possible for this run, so the "
            "stability of the reported scores is unknown."
        )

    return limitations


def generate_recommendations(
    recommendation: Optional[ModelRecommendation],
    gap_analysis: Optional[GapAnalysis],
    execution_time: Optional[ExecutionTimeAnalysis],
) -> List[str]:
    """Suggest the next concrete steps, bounded by what Phase 5 may do."""
    items: List[str] = [config.LIMITATION_NO_TUNING_RECOMMENDATION]

    if recommendation is None:
        items.append(
            "No baseline model can be recommended yet: at least one model must produce "
            "usable metrics first."
        )
        return items

    items.append(
        f"Use {recommendation.best_baseline_model} as the reference model for the next "
        "phase, and re-run this evaluation whenever the model changes so the comparison "
        "stays reproducible."
    )

    if gap_analysis is not None and gap_analysis.substantial_difference_models:
        items.append(
            "Investigate "
            + ", ".join(gap_analysis.substantial_difference_models)
            + ": repeat the evaluation with a different random_state to see whether the "
            "hold-out score or the cross-validation mean moves."
        )
    if gap_analysis is not None and gap_analysis.models_without_gap:
        items.append(
            "Enable cross-validation for "
            + ", ".join(gap_analysis.models_without_gap)
            + " (it was disabled or could not run for those models)."
        )
    if execution_time is not None and execution_time.slowest_model:
        items.append(
            f"{execution_time.slowest_model} was the slowest experiment "
            f"({execution_time.slowest_time:.4f} s); it is the first candidate to look at "
            "when the experiment budget grows."
        )

    return items


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _overall_status(experiments: Sequence[Any]) -> str:
    """Derive the report status from the recorded experiments."""
    items = list(experiments or [])
    if not items:
        return EvaluationStatus.SKIPPED.value
    usable = [
        item
        for item in items
        if getattr(item, "status", "")
        in (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
        and getattr(item, "f1_score", None) is not None
    ]
    if not usable:
        return EvaluationStatus.FAILED.value
    if len(usable) == len(items):
        return EvaluationStatus.COMPLETED.value
    return EvaluationStatus.PARTIAL.value


def build_evaluation_report(
    *,
    experiments: Sequence[Any],
    cross_validation_results: Sequence[Any] = (),
    model_comparison: Optional[ModelComparison] = None,
    dataset_name: str = "",
    problem_type: str = "",
    target_column: str = "",
    test_samples: int = 0,
    parameters_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
) -> EvaluationReport:
    """Assemble the complete Phase 5 evaluation report.

    This is the single entry point the Evaluation Agent calls: it reads the
    Phase 4 records, ranks them, builds every sub-analysis and returns one
    JSON-safe `EvaluationReport`. No model is fitted or evaluated here.

    Args:
        experiments: `ExperimentResult` records from the Training Agent.
        cross_validation_results: `CrossValidationResult` records.
        model_comparison: The Phase 4 comparison, used for its ranking rule and
            notes. It is rebuilt from the experiments when it is missing.
        dataset_name: Name of the evaluated dataset, for the report header.
        problem_type: Detected problem type.
        target_column: Target column name.
        test_samples: Size of the hold-out split, used by the limitations.
        parameters_by_model: Effective baseline parameters per model, copied into
            the recommendation so it stays self-contained.

    Returns:
        An `EvaluationReport`. A run without usable experiments still returns a
        report (status `skipped` or `failed`) with the reason recorded, so the
        caller never has to special-case an empty result.
    """
    evaluations = compare_models(experiments, cross_validation_results)
    ranked = rank_models(evaluations)

    confusion_analyses = [
        analyze_confusion_matrix(getattr(experiment, "confusion_matrix", None),
                                 model_name=str(getattr(experiment, "model_name", "")))
        for experiment in experiments or []
        if getattr(experiment, "status", "")
        in (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
    ]

    gap_analysis = analyze_performance_gaps(ranked)
    execution_time = analyze_execution_time(experiments)
    counts = count_experiments(experiments)
    recommendation = generate_model_recommendation(ranked)
    summary = generate_evaluation_summary(ranked)

    notes: List[str] = []
    if recommendation is not None:
        recommendation.performance_summary = summary
        if parameters_by_model:
            recommendation.parameters = dict(
                parameters_by_model.get(recommendation.best_baseline_model, {})
            )
        best_analysis = next(
            (
                analysis
                for analysis in confusion_analyses
                if analysis.model_name == recommendation.best_baseline_model
            ),
            None,
        )
        if best_analysis is not None:
            recommendation.confusion_matrix_error_pattern = best_analysis.error_pattern
        recommendation.limitations = [
            config.LIMITATION_NO_TUNING,
            config.LIMITATION_SINGLE_SPLIT,
        ]

    ranking_rule = ""
    if model_comparison is not None:
        ranking_rule = model_comparison.ranking_rule
        notes.extend(model_comparison.notes)
    else:
        ranking_rule = (
            f"rebuilt by the Evaluation Agent: primary: "
            f"{config.BEST_MODEL_PRIMARY_METRIC} (descending); secondary: "
            f"{config.BEST_MODEL_SECONDARY_METRIC} (descending); model name as the final "
            "tie-break"
        )
        notes.append(
            "No Phase 4 model comparison was available, so the ranking rule was rebuilt "
            "with the same criteria."
        )

    if not experiments:
        notes.append(
            "The Training Agent recorded no experiment, so there is nothing to evaluate."
        )

    report = EvaluationReport(
        overall_status=_overall_status(experiments),
        dataset_name=str(dataset_name or ""),
        problem_type=str(problem_type or ""),
        target_column=str(target_column or ""),
        test_samples=int(test_samples or 0),
        ranking_rule=ranking_rule,
        best_model=recommendation,
        selection_reason=recommendation.selection_reason if recommendation else (
            "No model produced usable metrics, so no baseline model is recommended."
        ),
        performance_summary=summary,
        model_evaluations=ranked,
        gap_analysis=gap_analysis,
        confusion_matrix_analysis=confusion_analyses,
        execution_time_analysis=execution_time,
        experiment_count=counts,
        observations=generate_observations(
            ranked, gap_analysis, confusion_analyses, execution_time, counts
        ),
        limitations=generate_limitations(
            model_evaluations=ranked,
            gap_analysis=gap_analysis,
            confusion_analyses=confusion_analyses,
            execution_time=execution_time,
            test_samples=test_samples,
        ),
        recommendations=generate_recommendations(recommendation, gap_analysis, execution_time),
        notes=notes,
    )
    report.table_text = build_evaluation_table(ranked)

    logger.debug(
        "Built evaluation report: status=%s, %d evaluation(s)",
        report.overall_status,
        len(ranked),
    )
    return report


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float]) -> str:
    """Format a metric for a report sentence, showing '-' when it is missing."""
    return "-" if value is None else f"{float(value):.4f}"


def _first_not_none(*values: Optional[float]) -> Optional[float]:
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "EvaluationDataError",
    "EvaluationToolError",
    "analyze_confusion_matrix",
    "analyze_execution_time",
    "analyze_performance_gaps",
    "build_evaluation_report",
    "build_evaluation_table",
    "calculate_cv_holdout_gap",
    "classify_performance_gap",
    "compare_models",
    "count_experiments",
    "generate_evaluation_summary",
    "generate_limitations",
    "generate_model_recommendation",
    "generate_observations",
    "generate_recommendations",
    "rank_models",
]
