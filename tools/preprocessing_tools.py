"""Reusable preprocessing tools (Phase 3).

This module contains the actual preprocessing logic. The agent in
`agents/preprocessing_agent.py` only orchestrates: it reads the analysis from
the shared state, calls these functions and records the decisions.

Nothing in this module modifies the source dataset on disk. Duplicate removal
returns a *new* DataFrame; every other function only inspects data.

Data leakage rule
-----------------
`build_column_transformer` returns an **unfitted** pipeline. It must be fitted
on training data only:

    X_train, X_test, y_train, y_test = train_test_split(...)
    pipeline = build_column_transformer(config)   # unfitted
    X_train_t = pipeline.fit_transform(X_train)   # statistics from train only
    X_test_t = pipeline.transform(X_test)         # never fitted on test data

Imputers, encoders and scalers therefore learn medians, modes, categories and
means from the training split only, which keeps the evaluation honest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import config
from utils.logger import setup_logger
from utils.state import (
    DataQualityIssue,
    DuplicateHandling,
    DuplicateStrategy,
    ExclusionReason,
    ExcludedFeature,
    PreprocessingConfig,
    PreprocessingStep,
    ScalingStrategy,
)

logger = setup_logger("tools.preprocessing_tools")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class PreprocessingError(Exception):
    """Base class for preprocessing errors."""


class AnalysisNotReadyError(PreprocessingError):
    """Raised when the Data Analysis Agent has not produced a usable state."""


class PreprocessingConfigError(PreprocessingError):
    """Raised when a preprocessing configuration is invalid or inconsistent."""


# Valid strategy values reused by the validation function.
VALID_NUMERICAL_IMPUTERS = ("median", "mean", "most_frequent", "constant")
VALID_CATEGORICAL_IMPUTERS = ("most_frequent", "constant", "mean", "median")
VALID_ENCODINGS = ("one_hot",)
VALID_SCALING_STRATEGIES = (ScalingStrategy.STANDARD.value, ScalingStrategy.NONE.value)
VALID_DUPLICATE_STRATEGIES = tuple(item.value for item in DuplicateStrategy)
# Only "drop" is allowed: "passthrough" would silently push excluded columns
# (identifier-like, constant, target, ...) into the model features.
VALID_REMAINDERS = ("drop",)


# --------------------------------------------------------------------------- #
# Column identification helpers
# --------------------------------------------------------------------------- #
def _columns_from_issues(
    data_quality_issues: Optional[Sequence[DataQualityIssue]],
    kind: str,
    *,
    details_filter: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return the columns of Phase 2 issues of one kind (and optional detail)."""
    columns: List[str] = []
    for issue in data_quality_issues or []:
        if issue.kind != kind or issue.column is None:
            continue
        if details_filter and not all(issue.details.get(key) == value
                                      for key, value in details_filter.items()):
            continue
        if issue.column not in columns:
            columns.append(issue.column)
    return columns


def _is_integer_like(series: pd.Series) -> bool:
    """True when every non-null value of a numeric series is a whole number."""
    values = pd.to_numeric(series.dropna(), errors="coerce")
    if values.empty or values.isna().any():
        return False
    remainder = [float(value) % 1 for value in values]
    return all(abs(item) < 1e-9 for item in remainder)


def identify_empty_columns(dataframe: pd.DataFrame) -> List[str]:
    """Return columns where every value is missing."""
    rows = int(dataframe.shape[0])
    if rows == 0:
        return []
    return [
        str(column)
        for column in dataframe.columns
        if int(dataframe[column].isna().sum()) == rows
    ]


def identify_constant_columns(
    dataframe: pd.DataFrame,
    data_quality_issues: Optional[Sequence[DataQualityIssue]] = None,
) -> List[str]:
    """Return columns with a single distinct value (and at least one value)."""
    from_issues = _columns_from_issues(data_quality_issues, "constant_column")

    detected: List[str] = []
    for column in dataframe.columns:
        series = dataframe[column]
        if series.notna().sum() == 0:
            continue  # empty columns are reported separately
        if int(series.nunique(dropna=True)) <= 1 and str(column) not in detected:
            detected.append(str(column))

    for column in from_issues:
        if column not in detected:
            detected.append(column)
    return detected


def identify_identifier_columns(
    dataframe: pd.DataFrame,
    data_quality_issues: Optional[Sequence[DataQualityIssue]] = None,
    *,
    target_column: Optional[str] = None,
    min_unique_ratio: float = config.IDENTIFIER_UNIQUE_RATIO,
) -> List[str]:
    """Return identifier-like columns that should not be model features.

    A column is treated as an identifier when (almost) every row has a unique
    value. The Phase 2 data quality issues are used when available and a local
    heuristic is applied as well, so the function also works standalone.

    Only near-unique columns are excluded: ordinary integer measurements are
    kept, because dropping every integer column would lose real signal. The
    ratio rule (not the data type) decides, so a text column of unique ids is
    detected as well.

    Args:
        dataframe: Dataset to inspect.
        data_quality_issues: Phase 2 issues (optional but preferred).
        target_column: Never returned, the target is not a feature.
        min_unique_ratio: Share of rows that must hold a unique value.

    Returns:
        Names of identifier-like columns.
    """
    rows = int(dataframe.shape[0])
    detected: List[str] = _columns_from_issues(
        data_quality_issues, "high_cardinality_column", details_filter={"looks_like_identifier": True}
    )

    if rows > 2:
        for column in dataframe.columns:
            name = str(column)
            if name == target_column or name in detected:
                continue
            series = dataframe[column]
            if int(series.isna().sum()) >= rows:
                continue
            unique_count = int(series.nunique(dropna=True))
            if unique_count / rows >= min_unique_ratio:
                detected.append(name)

    logger.debug("Identifier-like columns: %s", detected)
    return detected


def estimate_onehot_feature_count(dataframe: pd.DataFrame, columns: Sequence[str]) -> int:
    """Estimate how many columns one-hot encoding `columns` would produce.

    Each categorical column contributes one encoded column per distinct
    non-missing value. Two details matter:

    * missing values are imputed before encoding, so they do not add a column;
    * `handle_unknown="ignore"` encodes unseen categories as an all-zero row
      instead of creating a new column, so test data can never widen the output.
    """
    total = 0
    for column in columns:
        total += int(dataframe[column].dropna().nunique())
    return total


def identify_high_cardinality_columns(
    dataframe: pd.DataFrame,
    data_quality_issues: Optional[Sequence[DataQualityIssue]] = None,
    *,
    target_column: Optional[str] = None,
    max_encoded_features: int = config.HIGH_CARDINALITY_MAX_ENCODED_FEATURES,
    max_unique_ratio: float = config.HIGH_CARDINALITY_MAX_UNIQUE_RATIO,
    categorical_features: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return categorical columns that must not be one-hot encoded.

    A categorical column is excluded when its encoded width would be
    unreasonable, either because of its share of distinct values or because the
    number of generated columns exceeds `max_encoded_features`.

    Args:
        dataframe: Dataset to inspect.
        data_quality_issues: Phase 2 issues (optional).
        target_column: Never returned.
        max_encoded_features: Maximum acceptable encoded width per column.
        max_unique_ratio: Maximum acceptable share of distinct values.
        categorical_features: Restrict the check to these columns when given.

    Returns:
        Names of the high-cardinality columns.
    """
    rows = int(dataframe.shape[0])
    candidates = (
        list(categorical_features)
        if categorical_features is not None
        else [
            str(column)
            for column in dataframe.columns
            if not pd.api.types.is_numeric_dtype(dataframe[column])
            and not pd.api.types.is_bool_dtype(dataframe[column])
        ]
    )

    flagged = set(
        _columns_from_issues(data_quality_issues, "high_cardinality_column")
    )

    detected: List[str] = []
    for column in candidates:
        if column == target_column:
            continue
        series = dataframe[column].dropna()
        if series.empty:
            continue
        unique_count = int(series.nunique())
        unique_ratio = unique_count / rows if rows else 0.0
        encoded_width = unique_count + 1

        too_many_features = encoded_width > max_encoded_features
        too_many_values = (
            rows >= config.MIN_ROWS_FOR_UNIQUE_RATIO_CHECK
            and unique_count >= config.HIGH_CARDINALITY_MIN_UNIQUE
            and unique_ratio > max_unique_ratio
        )
        if too_many_features or too_many_values or column in flagged:
            if column not in detected:
                detected.append(column)

    logger.debug("High-cardinality columns: %s", detected)
    return detected


def identify_unsupported_columns(
    dataframe: pd.DataFrame,
    unsupported_columns: Optional[Sequence[str]] = None,
    *,
    target_column: Optional[str] = None,
) -> List[str]:
    """Return columns whose dtype the current pipeline cannot process.

    Datetime and complex columns are excluded for now. A future phase may
    extract year/month/day features from them instead of dropping them.
    """
    from_state = [column for column in (unsupported_columns or []) if column != target_column]

    detected: List[str] = list(from_state)
    for column in dataframe.columns:
        name = str(column)
        if name == target_column or name in detected:
            continue
        series = dataframe[column]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_complex_dtype(series):
            detected.append(name)
    return detected


def select_model_features(
    dataframe: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    excluded_columns: Optional[Sequence[str]] = None,
    unsupported_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split the remaining columns into numerical and categorical model features.

    Args:
        dataframe: Dataset to inspect.
        target_column: Column to treat as the target (never a feature).
        excluded_columns: Columns removed by the preprocessing decisions.
        unsupported_columns: Extra columns to leave out (e.g. from Phase 2).

    Returns:
        A `(numerical_features, categorical_features)` tuple of column names.
    """
    excluded = {str(column) for column in (excluded_columns or [])}
    excluded.update(str(column) for column in (unsupported_columns or []))

    numerical: List[str] = []
    categorical: List[str] = []
    for column in dataframe.columns:
        name = str(column)
        if name == target_column or name in excluded:
            continue
        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numerical.append(name)
        else:
            categorical.append(name)

    logger.debug("Selected features -> numerical=%s categorical=%s", numerical, categorical)
    return numerical, categorical


# --------------------------------------------------------------------------- #
# Duplicate rows
# --------------------------------------------------------------------------- #
def decide_duplicate_handling(
    duplicate_count: int,
    *,
    strategy: str = config.DUPLICATE_STRATEGY,
) -> DuplicateHandling:
    """Record the duplicate handling decision (nothing is removed here).

    Args:
        duplicate_count: Number of fully duplicated rows found in Phase 2.
        strategy: One of `DuplicateStrategy`.

    Returns:
        A `DuplicateHandling` record. `duplicate_rows_removed` stays 0 in this
        phase because the rows are removed later, on the in-memory training
        frame, by `remove_duplicate_rows`.
    """
    if strategy not in VALID_DUPLICATE_STRATEGIES:
        raise PreprocessingConfigError(
            f"Invalid duplicate strategy '{strategy}'. Valid options: {VALID_DUPLICATE_STRATEGIES}"
        )

    if strategy == DuplicateStrategy.KEEP_ALL.value:
        notes = (
            f"{duplicate_count} duplicate row(s) detected and deliberately kept. "
            "They may bias the model towards repeated records."
        )
        to_remove = 0
    else:
        notes = (
            f"{duplicate_count} duplicate row(s) detected. They will be removed from the "
            "in-memory training frame before the train/test split. The source CSV file is "
            "never modified."
        )
        to_remove = duplicate_count

    return DuplicateHandling(
        strategy=strategy,
        duplicate_rows_detected=duplicate_count,
        duplicate_rows_removed=0,
        duplicate_rows_to_remove=to_remove,
        notes=notes,
    )


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
    *,
    strategy: str = config.DUPLICATE_STRATEGY,
) -> Tuple[pd.DataFrame, int]:
    """Return a new DataFrame without exact duplicate rows.

    The input DataFrame is left untouched and the source file is never changed.
    Used by the training code (Phase 4) before splitting the data.

    Args:
        dataframe: Training frame.
        strategy: `remove_exact_duplicates` or `keep_all`.

    Returns:
        A `(dataframe, removed_count)` tuple.

    Raises:
        PreprocessingConfigError: Unknown strategy.
    """
    if strategy == DuplicateStrategy.KEEP_ALL.value:
        return dataframe.copy(), 0
    if strategy != DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value:
        raise PreprocessingConfigError(
            f"Invalid duplicate strategy '{strategy}'. Valid options: {VALID_DUPLICATE_STRATEGIES}"
        )

    before = int(dataframe.shape[0])
    deduplicated = dataframe.drop_duplicates().reset_index(drop=True)
    removed = before - int(deduplicated.shape[0])
    logger.info("Removed %d duplicate row(s) from the in-memory training frame", removed)
    return deduplicated, removed


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_numerical_pipeline(
    *,
    imputation: str = config.NUMERICAL_IMPUTATION_STRATEGY,
    scale_numeric: bool = config.SCALE_NUMERIC,
    scaling_strategy: str = config.SCALING_STRATEGY,
) -> Pipeline:
    """Build the numerical branch: imputer (and optional scaler).

    Median imputation is the default because it is far less influenced by
    outliers than the mean. The returned pipeline is unfitted.

    Args:
        imputation: Strategy for `SimpleImputer`.
        scale_numeric: Whether to append a scaler.
        scaling_strategy: `standard` (StandardScaler) or `none`.

    Returns:
        An unfitted sklearn `Pipeline`.

    Raises:
        PreprocessingConfigError: Invalid strategy values.
    """
    if imputation not in VALID_NUMERICAL_IMPUTERS:
        raise PreprocessingConfigError(
            f"Invalid numerical imputation '{imputation}'. Valid options: {VALID_NUMERICAL_IMPUTERS}"
        )
    if scaling_strategy not in VALID_SCALING_STRATEGIES:
        raise PreprocessingConfigError(
            f"Invalid scaling strategy '{scaling_strategy}'. "
            f"Valid options: {VALID_SCALING_STRATEGIES}"
        )

    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=imputation))]
    if scale_numeric and scaling_strategy == ScalingStrategy.STANDARD.value:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def build_categorical_pipeline(
    *,
    imputation: str = config.CATEGORICAL_IMPUTATION_STRATEGY,
    encoding: str = config.ENCODING_STRATEGY,
    handle_unknown: str = config.ENCODING_HANDLE_UNKNOWN,
    sparse_output: bool = config.ENCODING_SPARSE_OUTPUT,
    drop: Optional[str] = None,
) -> Pipeline:
    """Build the categorical branch: imputer then one-hot encoder.

    `handle_unknown="ignore"` is mandatory in practice: categories that were
    not present in the training split become all-zero rows instead of raising
    an error.

    Returns:
        An unfitted sklearn `Pipeline`.

    Raises:
        PreprocessingConfigError: Invalid strategy values.
    """
    if imputation not in VALID_CATEGORICAL_IMPUTERS:
        raise PreprocessingConfigError(
            f"Invalid categorical imputation '{imputation}'. "
            f"Valid options: {VALID_CATEGORICAL_IMPUTERS}"
        )
    if encoding not in VALID_ENCODINGS:
        raise PreprocessingConfigError(
            f"Invalid encoding '{encoding}'. Valid options: {VALID_ENCODINGS}"
        )
    if handle_unknown != "ignore":
        raise PreprocessingConfigError(
            "handle_unknown must be 'ignore' so unseen test categories cannot break "
            f"the pipeline (received '{handle_unknown}')."
        )

    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=imputation))]
    steps.append(
        (
            "encoder",
            OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=sparse_output,
                drop=drop,
            ),
        )
    )
    return Pipeline(steps=steps)


def describe_pipeline_steps(config_record: PreprocessingConfig) -> List[PreprocessingStep]:
    """Describe the pipeline branches in a human readable, serialisable way."""
    steps: List[PreprocessingStep] = []

    if config_record.numerical_features:
        transformers = [f"SimpleImputer(strategy='{config_record.numerical_imputation}')"]
        if config_record.scale_numeric and config_record.scaling_strategy == ScalingStrategy.STANDARD.value:
            transformers.append("StandardScaler()")
        steps.append(
            PreprocessingStep(
                name="numerical",
                applies_to="numerical_features",
                transformers=transformers,
            )
        )

    if config_record.categorical_features:
        steps.append(
            PreprocessingStep(
                name="categorical",
                applies_to="categorical_features",
                transformers=[
                    f"SimpleImputer(strategy='{config_record.categorical_imputation}')",
                    (
                        f"OneHotEncoder(handle_unknown='{config_record.encoding_handle_unknown}', "
                        f"sparse_output={str(config_record.encoding_sparse_output).lower()})"
                    ),
                ],
            )
        )

    return steps


def validate_preprocessing_config(
    config_record: PreprocessingConfig,
    dataframe: Optional[pd.DataFrame] = None,
) -> None:
    """Validate a preprocessing configuration before it is used.

    Args:
        config_record: Configuration to validate.
        dataframe: Optional dataset used to check the referenced columns.

    Raises:
        PreprocessingConfigError: The configuration is inconsistent or refers
            to columns that do not exist.
    """
    if not config_record.numerical_features and not config_record.categorical_features:
        raise PreprocessingConfigError(
            "No usable features left after preprocessing decisions: both the numerical "
            "and the categorical feature lists are empty."
        )

    overlap = set(config_record.numerical_features) & set(config_record.categorical_features)
    if overlap:
        raise PreprocessingConfigError(
            f"Columns cannot be numerical and categorical at the same time: {sorted(overlap)}"
        )

    excluded = {item.column for item in config_record.excluded_features}
    selected = set(config_record.numerical_features) | set(config_record.categorical_features)
    conflict = excluded & selected
    if conflict:
        raise PreprocessingConfigError(
            f"Excluded columns must not appear in the selected features: {sorted(conflict)}"
        )

    if config_record.numerical_features and config_record.numerical_imputation not in VALID_NUMERICAL_IMPUTERS:
        raise PreprocessingConfigError(
            f"Invalid numerical imputation '{config_record.numerical_imputation}'."
        )
    if config_record.categorical_features and config_record.categorical_imputation not in VALID_CATEGORICAL_IMPUTERS:
        raise PreprocessingConfigError(
            f"Invalid categorical imputation '{config_record.categorical_imputation}'."
        )
    if config_record.encoding not in VALID_ENCODINGS:
        raise PreprocessingConfigError(f"Invalid encoding '{config_record.encoding}'.")
    if config_record.encoding_handle_unknown != "ignore":
        raise PreprocessingConfigError(
            "encoding_handle_unknown must be 'ignore' to handle unseen categories."
        )
    if config_record.scaling_strategy not in VALID_SCALING_STRATEGIES:
        raise PreprocessingConfigError(f"Invalid scaling strategy '{config_record.scaling_strategy}'.")
    if config_record.scale_numeric and config_record.scaling_strategy == ScalingStrategy.NONE.value:
        raise PreprocessingConfigError(
            "scale_numeric is True but scaling_strategy is 'none'. Set scale_numeric=False "
            "when scaling is disabled."
        )
    if config_record.remainder not in VALID_REMAINDERS:
        raise PreprocessingConfigError(
            f"Invalid remainder '{config_record.remainder}'. Use 'drop' so excluded columns "
            "cannot leak into the features."
        )
    if config_record.duplicate_handling.strategy not in VALID_DUPLICATE_STRATEGIES:
        raise PreprocessingConfigError(
            f"Invalid duplicate strategy '{config_record.duplicate_handling.strategy}'."
        )

    if dataframe is not None:
        available = {str(column) for column in dataframe.columns}
        missing = sorted(selected - available)
        if missing:
            raise PreprocessingConfigError(
                f"Selected features are not present in the dataset: {missing}"
            )


def build_column_transformer(config_record: PreprocessingConfig) -> ColumnTransformer:
    """Build the scikit-learn `ColumnTransformer` described by a configuration.

    The result is **unfitted**: it must be fitted on training data only. See the
    module docstring for the leakage-safe usage pattern.

    `remainder="drop"` is used so that excluded columns (identifier-like,
    constant, high-cardinality, unsupported, and the target itself) can never
    reach the model, even if the caller passes a DataFrame that still contains
    them.

    Args:
        config_record: Validated preprocessing configuration.

    Returns:
        An unfitted `ColumnTransformer`.

    Raises:
        PreprocessingConfigError: The configuration is invalid.
    """
    validate_preprocessing_config(config_record)

    transformers: List[Tuple[str, Any, List[str]]] = []

    if config_record.numerical_features:
        transformers.append(
            (
                "numerical",
                build_numerical_pipeline(
                    imputation=config_record.numerical_imputation,
                    scale_numeric=config_record.scale_numeric,
                    scaling_strategy=config_record.scaling_strategy,
                ),
                list(config_record.numerical_features),
            )
        )

    if config_record.categorical_features:
        transformers.append(
            (
                "categorical",
                build_categorical_pipeline(
                    imputation=config_record.categorical_imputation,
                    encoding=config_record.encoding,
                    handle_unknown=config_record.encoding_handle_unknown,
                    sparse_output=config_record.encoding_sparse_output,
                    drop=config_record.encoding_drop,
                ),
                list(config_record.categorical_features),
            )
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder=config_record.remainder,
        verbose_feature_names_out=True,
    )
    logger.debug(
        "Built unfitted ColumnTransformer with branches: %s",
        [name for name, _, _ in transformers],
    )
    return transformer


def estimate_transformed_feature_count(
    dataframe: pd.DataFrame,
    config_record: PreprocessingConfig,
) -> int:
    """Estimate the number of output columns after transformation.

    Numerical columns map 1:1, categorical columns map to one column per
    distinct non-missing value (see `estimate_onehot_feature_count`). The value
    is an estimate: it becomes exact once the pipeline has been fitted, and it
    is verified to match the real output width for the sample datasets.
    """
    count = len(config_record.numerical_features)
    if config_record.categorical_features:
        count += estimate_onehot_feature_count(dataframe, config_record.categorical_features)
    return count


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_preprocessing(
    dataframe: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    duplicate_count: int = 0,
    data_quality_issues: Optional[Sequence[DataQualityIssue]] = None,
    unsupported_columns: Optional[Sequence[str]] = None,
    duplicate_strategy: str = config.DUPLICATE_STRATEGY,
    scale_numeric: bool = config.SCALE_NUMERIC,
    scaling_strategy: str = config.SCALING_STRATEGY,
    numerical_imputation: str = config.NUMERICAL_IMPUTATION_STRATEGY,
    categorical_imputation: str = config.CATEGORICAL_IMPUTATION_STRATEGY,
) -> PreprocessingConfig:
    """Turn the Phase 2 analysis into a preprocessing configuration.

    Exclusions are resolved in a fixed order so that every excluded column gets
    one clear and accurate reason (first match wins):

        empty column -> constant column -> unsupported dtype -> identifier-like
        -> high cardinality

    Unsupported types are resolved before the identifier check on purpose: a
    datetime column with one value per row should be reported as "not supported
    yet", not as an identifier.

    Args:
        dataframe: Dataset the configuration will be used on.
        target_column: Target column, never used as a feature.
        duplicate_count: Duplicate rows reported by Phase 2.
        data_quality_issues: Phase 2 issues, reused when available.
        unsupported_columns: Columns Phase 2 could not type.
        duplicate_strategy: See `DuplicateStrategy`.
        scale_numeric: Whether numerical features are scaled.
        scaling_strategy: `standard` or `none`.
        numerical_imputation: Imputation strategy for numerical features.
        categorical_imputation: Imputation strategy for categorical features.

    Returns:
        A validated `PreprocessingConfig`.

    Raises:
        PreprocessingConfigError: The resulting configuration is unusable.
    """
    issues = list(data_quality_issues or [])
    resolved: List[str] = []

    empty_columns = identify_empty_columns(dataframe)
    resolved.extend(empty_columns)

    constant_columns = [
        column
        for column in identify_constant_columns(dataframe, issues)
        if column not in resolved
    ]
    resolved.extend(constant_columns)

    unsupported = [
        column
        for column in identify_unsupported_columns(
            dataframe, unsupported_columns, target_column=target_column
        )
        if column not in resolved
    ]
    resolved.extend(unsupported)

    identifier_columns = [
        column
        for column in identify_identifier_columns(
            dataframe, issues, target_column=target_column
        )
        if column not in resolved
    ]
    resolved.extend(identifier_columns)

    _, categorical_candidates = select_model_features(
        dataframe, target_column=target_column, excluded_columns=resolved
    )
    high_cardinality_columns = [
        column
        for column in identify_high_cardinality_columns(
            dataframe, issues, target_column=target_column,
            categorical_features=categorical_candidates,
        )
        if column not in resolved
    ]
    resolved.extend(high_cardinality_columns)

    numerical_features, categorical_features = select_model_features(
        dataframe, target_column=target_column, excluded_columns=resolved
    )

    exclusions: List[ExcludedFeature] = []
    rows = int(dataframe.shape[0])

    for column in empty_columns:
        exclusions.append(
            ExcludedFeature(
                column=column,
                reason=ExclusionReason.EMPTY_COLUMN.value,
                detail="every value is missing, the column carries no information",
            )
        )
    for column in constant_columns:
        exclusions.append(
            ExcludedFeature(
                column=column,
                reason=ExclusionReason.CONSTANT_COLUMN.value,
                detail="single distinct value, it carries no predictive information",
            )
        )
    for column in identifier_columns:
        unique_count = int(dataframe[column].nunique(dropna=True))
        ratio = unique_count / rows if rows else 0.0
        exclusions.append(
            ExcludedFeature(
                column=column,
                reason=ExclusionReason.IDENTIFIER_LIKE.value,
                detail=(
                    f"identifier-like column: {unique_count} distinct values for {rows} rows "
                    f"({ratio:.0%} unique); it would only add noise"
                ),
            )
        )
    for column in unsupported:
        exclusions.append(
            ExcludedFeature(
                column=column,
                reason=ExclusionReason.UNSUPPORTED_TYPE.value,
                detail=(
                    f"dtype '{dataframe[column].dtype}' is not supported yet; date features "
                    "can be engineered in a future phase"
                ),
            )
        )
    for column in high_cardinality_columns:
        encoded_width = estimate_onehot_feature_count(dataframe, [column])
        exclusions.append(
            ExcludedFeature(
                column=column,
                reason=ExclusionReason.HIGH_CARDINALITY.value,
                detail=(
                    f"high-cardinality categorical column: one-hot encoding would create "
                    f"about {encoded_width} features, which is unreasonable"
                ),
            )
        )

    config_record = PreprocessingConfig(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        excluded_features=exclusions,
        target_column=target_column,
        numerical_imputation=numerical_imputation,
        categorical_imputation=categorical_imputation,
        encoding=config.ENCODING_STRATEGY,
        encoding_handle_unknown=config.ENCODING_HANDLE_UNKNOWN,
        encoding_sparse_output=config.ENCODING_SPARSE_OUTPUT,
        scale_numeric=scale_numeric,
        scaling_strategy=scaling_strategy if scale_numeric else ScalingStrategy.NONE.value,
        duplicate_handling=decide_duplicate_handling(duplicate_count, strategy=duplicate_strategy),
        remainder=config.PREPROCESSING_REMAINDER,
        fit_scope=config.PREPROCESSING_FIT_SCOPE,
    )
    config_record.steps = describe_pipeline_steps(config_record)

    validate_preprocessing_config(config_record, dataframe)
    logger.info(
        "Preprocessing plan: %d numerical, %d categorical, %d excluded",
        len(numerical_features), len(categorical_features), len(exclusions),
    )
    return config_record