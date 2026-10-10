"""Reusable dataset-analysis tools (Phase 2).

This module contains small, testable functions for loading and inspecting a
tabular CSV dataset. It contains no orchestration logic: agents decide *when*
to call these functions, the tools only do the data work.

Every function is read-only with respect to the source file, and no function
modifies, drops or imputes data. Problems are only reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from utils import config
from utils.logger import setup_logger
from utils.state import (
    CategoricalStats,
    ClassDistributionEntry,
    Confidence,
    DataQualityIssue,
    DatasetShape,
    MissingValueReport,
    NumericalStats,
    ProblemType,
    Severity,
    TargetCandidate,
    TargetDetection,
    TargetDetectionMethod,
)

logger = setup_logger("tools.data_tools")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class DataToolError(Exception):
    """Base class for all data tool errors."""


class DatasetNotFoundError(DataToolError):
    """Raised when the dataset path does not exist or is not a file."""


class UnsupportedFileFormatError(DataToolError):
    """Raised when the dataset file extension is not supported."""


class DatasetLoadError(DataToolError):
    """Raised when a CSV file cannot be parsed."""


class EmptyDatasetError(DataToolError):
    """Raised when a dataset contains no rows or no columns."""


class InvalidTargetColumnError(DataToolError):
    """Raised when the requested target column is not part of the dataset."""


# --------------------------------------------------------------------------- #
# Tool level result records
# --------------------------------------------------------------------------- #
@dataclass
class FeatureTypes:
    """Columns grouped by the kind of processing they will require."""

    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    unsupported_columns: List[str] = field(default_factory=list)


@dataclass
class ProblemTypeResult:
    """Detected problem type together with the reason behind the decision."""

    problem_type: ProblemType
    reason: str
    class_count: int
    is_supported: bool

    @property
    def is_classification(self) -> bool:
        """True when the target can be handled as a classification task."""
        return self.problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        )


@dataclass
class TargetAnalysis:
    """Everything known about the target column."""

    column: str
    dtype: str
    non_null_count: int
    missing_count: int
    class_count: int
    problem_type: ProblemType
    problem_type_reason: str
    is_supported: bool
    class_distribution: Dict[str, ClassDistributionEntry] = field(default_factory=dict)
    rare_classes: List[str] = field(default_factory=list)


@dataclass
class DatasetAnalysis:
    """Aggregate result produced by `analyze_dataset`."""

    shape: DatasetShape
    column_dtypes: Dict[str, str]
    feature_types: FeatureTypes
    missing_values: MissingValueReport
    duplicate_count: int
    numerical_statistics: Dict[str, NumericalStats]
    categorical_statistics: Dict[str, CategoricalStats]
    target_detection: TargetDetection
    target_analysis: Optional[TargetAnalysis]
    data_quality_issues: List[DataQualityIssue]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _to_float_or_none(value: Any) -> Optional[float]:
    """Convert a value to float, mapping NaN/None to None (JSON friendly)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


def _percentage(part: float, whole: float) -> float:
    """Return part/whole as a percentage rounded to two decimals."""
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100, 2)


def _is_integer_like(series: pd.Series) -> bool:
    """True when every non-null value of a numeric series is a whole number."""
    values = pd.to_numeric(series.dropna(), errors="coerce")
    if values.empty or values.isna().any():
        return False
    return bool(np.allclose(np.asarray(values, dtype=float) % 1, 0))


def _validate_dataset_path(dataset_path: Union[str, Path]) -> Path:
    """Validate that the path points to a readable, supported file."""
    if dataset_path is None or str(dataset_path).strip() == "":
        raise DatasetNotFoundError("No dataset path was provided.")

    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise DatasetNotFoundError(f"Dataset file does not exist: {path}")
    if not path.is_file():
        raise DatasetNotFoundError(f"Dataset path is not a file: {path}")
    if path.suffix.lower() not in config.SUPPORTED_DATA_FORMATS:
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{path.suffix}'. "
            f"Supported formats: {', '.join(config.SUPPORTED_DATA_FORMATS)}"
        )
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_csv(
    dataset_path: Union[str, Path],
    *,
    encoding: Optional[str] = None,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    The source file is opened read-only and is never modified.

    Args:
        dataset_path: Path to the CSV file.
        encoding: Optional text encoding (defaults to pandas' own default).
        sep: Optional column separator (auto-detected by pandas when omitted).

    Returns:
        The loaded DataFrame.

    Raises:
        DatasetNotFoundError: Path is missing, empty or not a file.
        UnsupportedFileFormatError: File extension is not supported.
        DatasetLoadError: The file exists but cannot be parsed as CSV.
        EmptyDatasetError: The file has no rows or no columns.
    """
    path = _validate_dataset_path(dataset_path)
    logger.info("Loading dataset: %s", path)

    # Only forward optional arguments when they were actually provided: some
    # pandas versions do not accept explicit None values for these keywords.
    read_options: Dict[str, Any] = {}
    if encoding is not None:
        read_options["encoding"] = encoding
    if sep is not None:
        read_options["sep"] = sep

    try:
        dataframe = pd.read_csv(path, **read_options)
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError(f"Dataset file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DatasetLoadError(f"Could not parse CSV file '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(
            f"Could not decode file '{path}'. Try an explicit encoding, e.g. encoding='latin-1'."
        ) from exc

    if dataframe.shape[0] == 0 or dataframe.shape[1] == 0:
        raise EmptyDatasetError(
            f"Dataset '{path.name}' has no usable rows or columns (shape {dataframe.shape})."
        )

    logger.info("Loaded '%s' with shape %s", path.name, dataframe.shape)
    return dataframe


# --------------------------------------------------------------------------- #
# Structure inspection
# --------------------------------------------------------------------------- #
def get_dataset_shape(dataframe: pd.DataFrame) -> DatasetShape:
    """Return the number of rows and columns."""
    return DatasetShape(rows=int(dataframe.shape[0]), columns=int(dataframe.shape[1]))


def get_column_info(dataframe: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping of column name to pandas dtype string."""
    return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}


def detect_feature_types(
    dataframe: pd.DataFrame,
    *,
    exclude_columns: Optional[Sequence[str]] = None,
) -> FeatureTypes:
    """Split columns into numerical, categorical and unsupported groups.

    Booleans are treated as categorical because they are labels rather than
    quantities. Datetime columns are reported as unsupported so the pipeline
    can decide explicitly what to do with them later.

    Args:
        dataframe: Dataset to inspect.
        exclude_columns: Column names to skip (for example the target).

    Returns:
        A `FeatureTypes` record with three disjoint lists of column names.
    """
    excluded = {str(column) for column in (exclude_columns or [])}
    result = FeatureTypes()

    for column in dataframe.columns:
        name = str(column)
        if name in excluded:
            continue

        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series):
            result.categorical_features.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            result.numerical_features.append(name)
        elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_complex_dtype(series):
            result.unsupported_columns.append(name)
        else:
            # object, string, category and anything else label-like
            result.categorical_features.append(name)

    logger.debug(
        "Feature types -> numerical=%d, categorical=%d, unsupported=%d",
        len(result.numerical_features),
        len(result.categorical_features),
        len(result.unsupported_columns),
    )
    return result


# --------------------------------------------------------------------------- #
# Missing values and duplicates
# --------------------------------------------------------------------------- #
def analyze_missing_values(dataframe: pd.DataFrame) -> MissingValueReport:
    """Count missing values overall and per column."""
    rows, columns = dataframe.shape
    total_cells = rows * columns
    missing_per_column = dataframe.isna().sum()

    per_column: Dict[str, int] = {}
    per_column_percentage: Dict[str, float] = {}
    for column, count in missing_per_column.items():
        count = int(count)
        if count > 0:
            name = str(column)
            per_column[name] = count
            per_column_percentage[name] = _percentage(count, rows)

    total = int(missing_per_column.sum())
    return MissingValueReport(
        total=total,
        total_percentage=_percentage(total, total_cells),
        affected_columns=len(per_column),
        per_column=per_column,
        per_column_percentage=per_column_percentage,
    )


def count_duplicates(dataframe: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows."""
    duplicates = int(dataframe.duplicated().sum())
    logger.debug("Duplicate rows: %d", duplicates)
    return duplicates


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def compute_numerical_statistics(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> Dict[str, NumericalStats]:
    """Compute count, mean, standard deviation, minimum and maximum per column."""
    statistics: Dict[str, NumericalStats] = {}
    for column in columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        statistics[column] = NumericalStats(
            column=column,
            count=int(series.count()),
            mean=_to_float_or_none(series.mean()) if series.count() else None,
            std=_to_float_or_none(series.std()) if series.count() > 1 else None,
            minimum=_to_float_or_none(series.min()) if series.count() else None,
            maximum=_to_float_or_none(series.max()) if series.count() else None,
        )
    return statistics


def compute_categorical_statistics(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    *,
    top_n: int = config.TOP_CATEGORICAL_VALUES,
) -> Dict[str, CategoricalStats]:
    """Compute unique counts and the most frequent values per categorical column."""
    statistics: Dict[str, CategoricalStats] = {}
    rows = int(dataframe.shape[0])

    for column in columns:
        series = dataframe[column].dropna()
        counts = series.value_counts().head(top_n)
        statistics[column] = CategoricalStats(
            column=column,
            unique_count=int(series.nunique()),
            top_values={str(label): int(count) for label, count in counts.items()},
            top_values_percentage={
                str(label): _percentage(int(count), rows) for label, count in counts.items()
            },
        )
    return statistics


# --------------------------------------------------------------------------- #
# Target column and problem type
# --------------------------------------------------------------------------- #
def calculate_class_distribution(series: pd.Series) -> Dict[str, ClassDistributionEntry]:
    """Return class counts and percentages for a target series."""
    non_null = series.dropna()
    total = int(non_null.shape[0])
    if total == 0:
        return {}

    distribution: Dict[str, ClassDistributionEntry] = {}
    for label, count in non_null.value_counts().items():
        count = int(count)
        distribution[str(label)] = ClassDistributionEntry(
            count=count, percentage=_percentage(count, total)
        )
    return distribution


def detect_problem_type(
    series: pd.Series,
    *,
    column_name: str = "target",
) -> ProblemTypeResult:
    """Detect whether a target column describes a classification task.

    The rules are deliberately conservative: whenever the target does not look
    like a class label the result is `UNSUPPORTED_OR_UNCERTAIN` (or
    `REGRESSION`) together with an explicit reason, so classification is never
    silently assumed.

    Args:
        series: The target column.
        column_name: Name used in the explanation messages.

    Returns:
        A `ProblemTypeResult` with the decision and its justification.
    """
    non_null = series.dropna()
    total = int(non_null.shape[0])

    if total == 0:
        return ProblemTypeResult(
            problem_type=ProblemType.UNSUPPORTED_OR_UNCERTAIN,
            reason=f"Target column '{column_name}' contains no non-missing values.",
            class_count=0,
            is_supported=False,
        )

    class_count = int(non_null.nunique())

    if class_count <= 1:
        return ProblemTypeResult(
            problem_type=ProblemType.UNSUPPORTED_OR_UNCERTAIN,
            reason=(
                f"Target column '{column_name}' has only {class_count} distinct value(s); "
                "a classification target needs at least two classes."
            ),
            class_count=class_count,
            is_supported=False,
        )

    unique_ratio = class_count / total
    ratio_check_applies = total >= config.MIN_ROWS_FOR_UNIQUE_RATIO_CHECK
    too_many_classes = class_count > config.MAX_CLASSES_FOR_CLASSIFICATION
    looks_continuous = ratio_check_applies and unique_ratio > config.CLASSIFICATION_MAX_UNIQUE_RATIO

    is_bool = pd.api.types.is_bool_dtype(series)
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric and not is_bool:
        if not _is_integer_like(series):
            return ProblemTypeResult(
                problem_type=ProblemType.REGRESSION,
                reason=(
                    f"Target column '{column_name}' is numeric with non-integer values, "
                    "which indicates a continuous target (regression). Regression is not "
                    "supported in the current phase."
                ),
                class_count=class_count,
                is_supported=False,
            )
        if looks_continuous or too_many_classes:
            return ProblemTypeResult(
                problem_type=ProblemType.REGRESSION,
                reason=(
                    f"Target column '{column_name}' is numeric with {class_count} distinct values "
                    f"({_percentage(class_count, total)}% of {total} rows), which indicates a "
                    "continuous target (regression). Regression is not supported in the "
                    "current phase."
                ),
                class_count=class_count,
                is_supported=False,
            )

    if looks_continuous or too_many_classes:
        return ProblemTypeResult(
            problem_type=ProblemType.UNSUPPORTED_OR_UNCERTAIN,
            reason=(
                f"Target column '{column_name}' has {class_count} distinct values across {total} "
                f"rows. This is too high-cardinality to be treated as a class label; it may be a "
                "regression target, an identifier, or free text. Please confirm the target column."
            ),
            class_count=class_count,
            is_supported=False,
        )

    problem_type = (
        ProblemType.BINARY_CLASSIFICATION
        if class_count == 2
        else ProblemType.MULTICLASS_CLASSIFICATION
    )
    reason = (
        f"Target column '{column_name}' has {class_count} distinct class label(s) across "
        f"{total} rows, which matches a {problem_type.value} task."
    )
    return ProblemTypeResult(
        problem_type=problem_type, reason=reason, class_count=class_count, is_supported=True
    )


def analyze_target(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    rare_class_min_samples: int = config.RARE_CLASS_MIN_SAMPLES,
) -> TargetAnalysis:
    """Describe the target column and detect the problem type.

    Args:
        dataframe: Dataset containing the target column.
        target_column: Name of the target column.
        rare_class_min_samples: Minimum samples a class needs to avoid a warning.

    Returns:
        A `TargetAnalysis` record.

    Raises:
        InvalidTargetColumnError: The column is not present in the dataset.
    """
    if target_column not in dataframe.columns and str(target_column) not in [
        str(column) for column in dataframe.columns
    ]:
        raise InvalidTargetColumnError(
            f"Target column '{target_column}' is not present in the dataset. "
            f"Available columns: {[str(column) for column in dataframe.columns]}"
        )

    series = dataframe[target_column]
    problem = detect_problem_type(series, column_name=str(target_column))
    distribution = (
        calculate_class_distribution(series) if problem.is_classification else {}
    )
    rare_classes = [
        label
        for label, entry in distribution.items()
        if entry.count < rare_class_min_samples
    ]

    logger.debug(
        "Target '%s' -> %s (classes=%d)",
        target_column, problem.problem_type.value, problem.class_count,
    )
    return TargetAnalysis(
        column=str(target_column),
        dtype=str(series.dtype),
        non_null_count=int(series.notna().sum()),
        missing_count=int(series.isna().sum()),
        class_count=problem.class_count,
        problem_type=problem.problem_type,
        problem_type_reason=problem.reason,
        is_supported=problem.is_supported,
        class_distribution=distribution,
        rare_classes=sorted(rare_classes),
    )


def _score_target_candidate(series: pd.Series, rows: int) -> tuple:
    """Score a column as a target candidate and explain the score.

    The score combines four transparent signals: column name conventions,
    data type, cardinality relative to the number of rows, and completeness.
    It is a heuristic, not a guarantee.
    """
    name = str(series.name).strip().lower()
    score = 0.0
    reasons: List[str] = []

    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / rows if rows else 1.0
    missing_ratio = float(series.isna().mean())

    # 1. Column name conventions
    name_parts = set(name.replace("-", "_").replace(" ", "_").split("_"))
    if any(hint in name_parts for hint in config.TARGET_NAME_HINTS):
        score += 3.0
        reasons.append("column name matches a common target/label convention (+3.0)")
    if any(part in name_parts for part in config.TARGET_NAME_PENALTIES):
        score -= 4.0
        reasons.append("column name looks like an identifier or metadata (-4.0)")

    # 2. Data type
    if pd.api.types.is_bool_dtype(series):
        score += 2.0
        reasons.append("boolean column, typical of a yes/no target (+2.0)")
    elif pd.api.types.is_numeric_dtype(series):
        if _is_integer_like(series):
            score += 2.0
            reasons.append("integer-like numeric column, common for encoded labels (+2.0)")
        else:
            reasons.append("continuous numeric column, unlikely to be a class label (+0.0)")
    else:
        score += 1.5
        reasons.append("categorical column, typical of a class label (+1.5)")

    # 3. Cardinality
    if unique_count <= config.MAX_CLASSES_FOR_CLASSIFICATION and (
        unique_ratio <= config.CLASSIFICATION_MAX_UNIQUE_RATIO or rows < config.MIN_ROWS_FOR_UNIQUE_RATIO_CHECK
    ):
        score += 2.5
        reasons.append(
            f"low cardinality ({unique_count} distinct values), typical of a class label (+2.5)"
        )
    elif unique_count == 2:
        score += 2.0
        reasons.append("exactly two distinct values, typical of a binary target (+2.0)")
    elif unique_ratio > config.HIGH_CARDINALITY_RATIO:
        score -= 3.0
        reasons.append(
            f"very high cardinality ({unique_count} distinct values), identifier-like (-3.0)"
        )
    if rows > 2 and unique_count >= rows:
        score -= 2.0
        reasons.append("every row has a unique value, identifier-like (-2.0)")

    # 4. Completeness
    if missing_ratio > config.HIGH_MISSING_RATIO_WARNING:
        score -= 1.5
        reasons.append(f"high ratio of missing values ({missing_ratio:.0%}) (-1.5)")

    # 5. Constant columns cannot be a target
    if unique_count <= 1:
        score -= 5.0
        reasons.append("constant column, cannot be used as a target (-5.0)")

    return score, reasons


def detect_target_column(
    dataframe: pd.DataFrame,
    explicit_target: Optional[str] = None,
    *,
    top_candidates: int = config.TARGET_DETECTION_TOP_CANDIDATES,
) -> TargetDetection:
    """Resolve the target column, either explicitly or heuristically.

    When no target is supplied the function scores every column and returns the
    best candidate together with a confidence level and the full ranking, so the
    user can always see why a column was chosen and override it.

    Args:
        dataframe: Dataset to inspect.
        explicit_target: Target column provided by the user (highest priority).
        top_candidates: How many scored candidates to keep.

    Returns:
        A `TargetDetection` record.

    Raises:
        InvalidTargetColumnError: `explicit_target` is not a column of the dataset.
    """
    available_columns = [str(column) for column in dataframe.columns]

    if explicit_target is not None and str(explicit_target).strip() != "":
        target = str(explicit_target)
        if target not in available_columns:
            raise InvalidTargetColumnError(
                f"Target column '{target}' is not present in the dataset. "
                f"Available columns: {available_columns}"
            )
        return TargetDetection(
            column=target,
            method=TargetDetectionMethod.EXPLICIT.value,
            confidence=Confidence.HIGH.value,
            reason=f"Target column '{target}' was explicitly provided by the user.",
            candidates=[],
        )

    rows = int(dataframe.shape[0])
    scored: List[TargetCandidate] = []
    for column in dataframe.columns:
        score, reasons = _score_target_candidate(dataframe[column], rows)
        scored.append(TargetCandidate(column=str(column), score=round(score, 2), reasons=reasons))

    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    candidates = scored[:top_candidates]
    best = candidates[0] if candidates else None

    if best is None or best.score < config.TARGET_DETECTION_MIN_SCORE:
        return TargetDetection(
            column=None,
            method=TargetDetectionMethod.NOT_DETECTED.value,
            confidence=Confidence.NONE.value,
            reason=(
                "Automatic target detection is uncertain: no column scored high enough to be "
                "used as a target. Please specify the target column explicitly."
            ),
            candidates=candidates,
        )

    if best.score >= config.TARGET_DETECTION_HIGH_SCORE:
        confidence = Confidence.HIGH
    elif best.score >= config.TARGET_DETECTION_MEDIUM_SCORE:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    reason = (
        f"Column '{best.column}' was selected automatically with {confidence.value} confidence "
        f"(heuristic score {best.score}). This is a suggestion, not a guarantee: confirm it or "
        "provide the target column explicitly."
    )
    return TargetDetection(
        column=best.column,
        method=TargetDetectionMethod.AUTOMATIC.value,
        confidence=confidence.value,
        reason=reason,
        candidates=candidates,
    )


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def _make_issue(
    kind: str,
    severity: Severity,
    message: str,
    *,
    column: Optional[str] = None,
    **details: Any,
) -> DataQualityIssue:
    """Build a `DataQualityIssue` record."""
    return DataQualityIssue(
        kind=kind,
        severity=severity.value,
        message=message,
        column=column,
        details=details,
    )


def detect_data_quality_issues(
    dataframe: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    class_distribution: Optional[Dict[str, ClassDistributionEntry]] = None,
    feature_types: Optional[FeatureTypes] = None,
    target_problem_type: Optional[ProblemType] = None,
    rare_class_min_samples: int = config.RARE_CLASS_MIN_SAMPLES,
) -> List[DataQualityIssue]:
    """Report potential data quality problems.

    Nothing is modified or removed here: the function only describes what a
    later stage (preprocessing) may want to deal with.

    Args:
        dataframe: Dataset to inspect.
        target_column: Target column name, if already known.
        class_distribution: Class distribution, if already computed.
        feature_types: Feature grouping, if already computed.
        target_problem_type: Detected problem type, if already computed.
        rare_class_min_samples: Minimum samples a class needs to be considered normal.

    Returns:
        A list of `DataQualityIssue` records, ordered by severity.

    Note:
        Columns that are constant, empty, unsupported or extremely high
        cardinality are only reported. Nothing is deleted or modified here.
    """
    rows, columns_count = dataframe.shape
    issues: List[DataQualityIssue] = []

    # ---- Missing values -------------------------------------------------- #
    missing = analyze_missing_values(dataframe)
    for column, count in missing.per_column.items():
        ratio = count / rows if rows else 0.0
        if ratio > config.HIGH_MISSING_RATIO_ERROR:
            severity = Severity.ERROR
        elif ratio > config.HIGH_MISSING_RATIO_WARNING:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        issues.append(
            _make_issue(
                "missing_values",
                severity,
                f"Column '{column}' has {count} missing value(s) "
                f"({missing.per_column_percentage[column]}% of the column).",
                column=column,
                missing_count=count,
                missing_percentage=missing.per_column_percentage[column],
            )
        )

    # ---- Duplicate rows -------------------------------------------------- #
    duplicates = count_duplicates(dataframe)
    if duplicates > 0:
        issues.append(
            _make_issue(
                "duplicate_rows",
                Severity.WARNING,
                f"Dataset contains {duplicates} fully duplicated row(s) "
                f"({_percentage(duplicates, rows)}% of all rows).",
                duplicate_count=duplicates,
                duplicate_percentage=_percentage(duplicates, rows),
            )
        )

    # ---- Column level checks --------------------------------------------- #
    for column in dataframe.columns:
        name = str(column)
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())

        if rows > 0 and missing_count == rows:
            issues.append(
                _make_issue(
                    "empty_column",
                    Severity.ERROR,
                    f"Column '{name}' is completely empty.",
                    column=name,
                )
            )
            continue

        if unique_count <= 1:
            issues.append(
                _make_issue(
                    "constant_column",
                    Severity.WARNING,
                    f"Column '{name}' has a single distinct value "
                    f"({_percent_or_unknown(series)}); it carries no information.",
                    column=name,
                    unique_count=unique_count,
                )
            )
            continue

        # High cardinality is only a problem when the column cannot legitimately
        # contain many values: identifiers (unique per row) and categorical or
        # text columns. Continuous measurements such as `bmi` or `glucose`
        # normally have many distinct values and are not reported.
        unique_ratio = unique_count / rows if rows else 1.0
        looks_like_id = rows > 2 and unique_ratio >= config.IDENTIFIER_UNIQUE_RATIO
        is_numeric_measure = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        cardinality_threshold = max(
            config.HIGH_CARDINALITY_MIN_UNIQUE, rows * config.HIGH_CARDINALITY_RATIO
        )
        if looks_like_id or (not is_numeric_measure and unique_count >= cardinality_threshold):
            issues.append(
                _make_issue(
                    "high_cardinality_column",
                    Severity.WARNING if looks_like_id else Severity.INFO,
                    f"Column '{name}' has {unique_count} distinct values across {rows} rows"
                    + (
                        f"; {unique_ratio:.0%} of rows have a unique value, so it is probably "
                        "an identifier."
                        if looks_like_id
                        else "."
                    ),
                    column=name,
                    unique_count=unique_count,
                    unique_ratio=round(unique_ratio, 4),
                    looks_like_identifier=looks_like_id,
                )
            )

    # ---- Unsupported column types ---------------------------------------- #
    types = feature_types or detect_feature_types(
        dataframe, exclude_columns=[target_column] if target_column else None
    )
    for column in types.unsupported_columns:
        issues.append(
            _make_issue(
                "unsupported_column_type",
                Severity.WARNING,
                f"Column '{column}' has dtype '{dataframe[column].dtype}', which the current "
                "pipeline cannot process automatically.",
                column=column,
                dtype=str(dataframe[column].dtype),
            )
        )

    # ---- Target specific checks ------------------------------------------ #
    if target_column is not None and target_column in [str(c) for c in dataframe.columns]:
        target_series = dataframe[target_column]
        target_missing = int(target_series.isna().sum())
        if target_missing > 0:
            issues.append(
                _make_issue(
                    "target_missing_values",
                    Severity.WARNING,
                    f"Target column '{target_column}' has {target_missing} missing value(s).",
                    column=str(target_column),
                    missing_count=target_missing,
                )
            )

        if target_problem_type == ProblemType.REGRESSION:
            issues.append(
                _make_issue(
                    "unsupported_problem_type",
                    Severity.ERROR,
                    f"Target column '{target_column}' looks like a regression target, which is "
                    "not supported in the current phase.",
                    column=str(target_column),
                )
            )
        elif target_problem_type == ProblemType.UNSUPPORTED_OR_UNCERTAIN:
            issues.append(
                _make_issue(
                    "uncertain_problem_type",
                    Severity.WARNING,
                    f"The problem type for target column '{target_column}' could not be "
                    "determined reliably.",
                    column=str(target_column),
                )
            )

    if class_distribution:
        for label, entry in class_distribution.items():
            if entry.count < rare_class_min_samples:
                issues.append(
                    _make_issue(
                        "rare_class",
                        Severity.WARNING,
                        f"Class '{label}' of the target has only {entry.count} sample(s), "
                        "which is too few for reliable evaluation.",
                        column=target_column,
                        class_label=label,
                        class_count=entry.count,
                    )
                )
        percentages = [entry.percentage for entry in class_distribution.values()]
        if percentages:
            major = max(class_distribution.items(), key=lambda item: item[1].count)
            if major[1].percentage > config.IMBALANCE_MAJORITY_PERCENTAGE:
                issues.append(
                    _make_issue(
                        "imbalanced_classes",
                        Severity.WARNING,
                        f"Class '{major[0]}' covers {major[1].percentage}% of the target values; "
                        "the dataset may be strongly imbalanced.",
                        column=target_column,
                        majority_class=major[0],
                        majority_percentage=major[1].percentage,
                    )
                )

    severity_order = {Severity.ERROR.value: 0, Severity.WARNING.value: 1, Severity.INFO.value: 2}
    issues.sort(key=lambda issue: severity_order.get(issue.severity, 3))
    logger.debug("Data quality issues detected: %d", len(issues))
    return issues


def _percent_or_unknown(series: pd.Series) -> str:
    """Small helper returning a readable share of the most frequent value."""
    non_null = series.dropna()
    if non_null.empty:
        return "no non-missing values"
    top_count = int(non_null.value_counts().iloc[0])
    return f"{_percentage(top_count, int(non_null.shape[0]))}% of rows share it"


# --------------------------------------------------------------------------- #
# Aggregate entry point
# --------------------------------------------------------------------------- #
def analyze_dataset(
    dataframe: pd.DataFrame,
    *,
    explicit_target: Optional[str] = None,
    top_categorical_values: int = config.TOP_CATEGORICAL_VALUES,
) -> DatasetAnalysis:
    """Run the full analysis pipeline on an already-loaded DataFrame.

    Args:
        dataframe: Dataset to analyse.
        explicit_target: Target column supplied by the user, if any.
        top_categorical_values: How many frequent values to keep per column.

    Returns:
        A `DatasetAnalysis` record with every Phase 2 result.
    """
    shape = get_dataset_shape(dataframe)
    column_dtypes = get_column_info(dataframe)

    target_detection = detect_target_column(dataframe, explicit_target=explicit_target)
    target_column = target_detection.column

    feature_types = detect_feature_types(
        dataframe, exclude_columns=[target_column] if target_column else None
    )

    target_analysis: Optional[TargetAnalysis] = None
    if target_column is not None:
        target_analysis = analyze_target(dataframe, target_column)

    missing_values = analyze_missing_values(dataframe)
    duplicate_count = count_duplicates(dataframe)

    numerical_statistics = compute_numerical_statistics(
        dataframe, feature_types.numerical_features
    )
    categorical_statistics = compute_categorical_statistics(
        dataframe, feature_types.categorical_features, top_n=top_categorical_values
    )

    data_quality_issues = detect_data_quality_issues(
        dataframe,
        target_column=target_column,
        class_distribution=target_analysis.class_distribution if target_analysis else None,
        feature_types=feature_types,
        target_problem_type=target_analysis.problem_type if target_analysis else None,
    )

    return DatasetAnalysis(
        shape=shape,
        column_dtypes=column_dtypes,
        feature_types=feature_types,
        missing_values=missing_values,
        duplicate_count=duplicate_count,
        numerical_statistics=numerical_statistics,
        categorical_statistics=categorical_statistics,
        target_detection=target_detection,
        target_analysis=target_analysis,
        data_quality_issues=data_quality_issues,
    )