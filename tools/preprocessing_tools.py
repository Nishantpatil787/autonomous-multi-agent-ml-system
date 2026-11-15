"""Reusable preprocessing pipeline tools and column transformer builders.

This module constructs reproducible scikit-learn preprocessing pipelines,
identifies unusable/identifier columns, and manages pipeline configurations
without modifying source datasets or leaking data across train/test splits.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import Config
from utils.logger import get_logger

logger = get_logger("PreprocessingTools")

# Recognized common identifier names (case-insensitive)
IDENTIFIER_CANDIDATE_NAMES = [
    "id",
    "record_id",
    "user_id",
    "row_id",
    "customer_id",
    "index",
    "id_number",
    "uuid",
    "guid",
    "patient_id",
    "client_id",
    "transaction_id",
]

SUPPORTED_NUMERICAL_IMPUTATION = ["median", "mean", "most_frequent", "constant"]
SUPPORTED_CATEGORICAL_IMPUTATION = ["most_frequent", "constant"]


def is_identifier_column(col_name: str, total_rows: int = 0, unique_count: int = 0) -> bool:
    """Determine whether a column matches known identifier patterns or metrics."""
    cleaned = col_name.strip().lower()
    if cleaned in IDENTIFIER_CANDIDATE_NAMES:
        return True

    # Check for suffix/prefix like '_id' or 'id_'
    if cleaned.endswith("_id") or cleaned.startswith("id_"):
        return True

    # High cardinality check: 100% unique in non-trivial datasets
    if total_rows > 10 and unique_count == total_rows:
        return True

    return False


def filter_usable_features(
    feature_names: List[str],
    numerical_features: List[str],
    categorical_features: List[str],
    categorical_statistics: Optional[Dict[str, Dict[str, Any]]] = None,
    data_quality_issues: Optional[List[Dict[str, Any]]] = None,
    max_cardinality: Optional[int] = None,
    total_rows: int = 0,
) -> Dict[str, Any]:
    """Filter features into selected and excluded subsets with recorded reasoning.

    Args:
        feature_names: All candidate feature names (excluding target).
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.
        categorical_statistics: Optional dictionary of categorical statistics from Phase 2.
        data_quality_issues: Optional list of quality diagnostics from Phase 2.
        max_cardinality: Maximum permitted distinct categories for OneHotEncoder.
        total_rows: Total rows in the dataset.

    Returns:
        Dictionary containing selected_numerical, selected_categorical,
        excluded_features, and exclusion_reasons.
    """
    cardinality_limit = max_cardinality or Config.MAX_CATEGORICAL_CARDINALITY
    quality_issues = data_quality_issues or []
    cat_stats = categorical_statistics or {}

    excluded_features: List[str] = []
    exclusion_reasons: Dict[str, str] = {}
    constant_columns: List[str] = []
    high_cardinality_columns: List[str] = []
    datetime_columns: List[str] = []

    # Map quality issues by column name
    column_issues: Dict[str, List[Dict[str, Any]]] = {}
    for issue in quality_issues:
        col = issue.get("column")
        if col:
            column_issues.setdefault(col, []).append(issue)

    # 1. Inspect all features for exclusion rules
    for col in feature_names:
        # Check constant columns from Phase 2 quality checks
        col_issue_types = [iss.get("type") for iss in column_issues.get(col, [])]
        if "constant_column" in col_issue_types:
            excluded_features.append(col)
            constant_columns.append(col)
            exclusion_reasons[col] = "Constant / zero-variance column"
            continue

        # Check completely empty columns
        if "empty_column" in col_issue_types:
            excluded_features.append(col)
            exclusion_reasons[col] = "Empty column (100% missing)"
            continue

        # Check identifier columns
        unique_cnt = cat_stats.get(col, {}).get("unique_count", 0)
        if "high_cardinality_identifier" in col_issue_types or is_identifier_column(col, total_rows, unique_cnt):
            excluded_features.append(col)
            exclusion_reasons[col] = "Identifier-like column excluded from model features"
            continue

        # Check datetime columns (name pattern check)
        lower_col = col.lower()
        is_dt = (
            any(keyword in lower_col for keyword in ["timestamp", "datetime", "date_time"])
            or lower_col == "date"
            or lower_col.startswith("date_")
            or lower_col.endswith("_date")
        )
        if is_dt:
            excluded_features.append(col)
            datetime_columns.append(col)
            exclusion_reasons[col] = "Datetime column unsupported in this phase"
            continue

        # Check high cardinality categorical columns
        if col in categorical_features:
            cat_unique = cat_stats.get(col, {}).get("unique_count", 0)
            if cat_unique > cardinality_limit:
                excluded_features.append(col)
                high_cardinality_columns.append(col)
                exclusion_reasons[col] = (
                    f"Categorical column exceeds cardinality limit ({cat_unique} > {cardinality_limit} unique categories)"
                )
                continue

    # 2. Derive selected subsets
    selected_num = [col for col in numerical_features if col not in excluded_features]
    selected_cat = [col for col in categorical_features if col not in excluded_features]

    return {
        "selected_numerical_features": selected_num,
        "selected_categorical_features": selected_cat,
        "excluded_features": excluded_features,
        "exclusion_reasons": exclusion_reasons,
        "constant_columns_excluded": constant_columns,
        "high_cardinality_columns_excluded": high_cardinality_columns,
        "datetime_columns_excluded": datetime_columns,
    }


def build_numerical_pipeline(
    impute_strategy: str = "median",
    scale: bool = True,
) -> Pipeline:
    """Build a scikit-learn Pipeline for numerical feature transformation.

    Args:
        impute_strategy: Imputation strategy ('median', 'mean', etc.).
        scale: Whether to apply StandardScaler.

    Returns:
        Configured un-fitted scikit-learn Pipeline.
    """
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=impute_strategy)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def build_categorical_pipeline(
    impute_strategy: str = "most_frequent",
    handle_unknown: str = "ignore",
) -> Pipeline:
    """Build a scikit-learn Pipeline for categorical feature transformation.

    Args:
        impute_strategy: Imputation strategy ('most_frequent', etc.).
        handle_unknown: Strategy for handling unseen categories ('ignore').

    Returns:
        Configured un-fitted scikit-learn Pipeline.
    """
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("encoder", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)),
    ]
    return Pipeline(steps=steps)


def build_column_transformer(
    numerical_features: List[str],
    categorical_features: List[str],
    numerical_imputation: str = "median",
    categorical_imputation: str = "most_frequent",
    scale_numeric: bool = True,
    handle_unknown: str = "ignore",
) -> ColumnTransformer:
    """Construct an un-fitted ColumnTransformer targeting specified feature subsets.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        numerical_imputation: Imputation strategy for numerical columns.
        categorical_imputation: Imputation strategy for categorical columns.
        scale_numeric: Whether to scale numerical columns with StandardScaler.
        handle_unknown: OneHotEncoder unknown category handling strategy.

    Returns:
        Un-fitted ColumnTransformer.
    """
    transformers: List[Tuple[str, Pipeline, List[str]]] = []

    if numerical_features:
        num_pipe = build_numerical_pipeline(
            impute_strategy=numerical_imputation,
            scale=scale_numeric,
        )
        transformers.append(("numerical", num_pipe, numerical_features))

    if categorical_features:
        cat_pipe = build_categorical_pipeline(
            impute_strategy=categorical_imputation,
            handle_unknown=handle_unknown,
        )
        transformers.append(("categorical", cat_pipe, categorical_features))

    # Any column not explicitly in transformers is dropped
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline_config(
    selected_numerical_features: List[str],
    selected_categorical_features: List[str],
    excluded_features: List[str],
    exclusion_reasons: Dict[str, str],
    numerical_imputation_strategy: str = "median",
    categorical_imputation_strategy: str = "most_frequent",
    encoding_strategy: str = "one_hot",
    scaling_strategy: str = "standard_scaler",
    scale_numeric: bool = True,
    duplicate_handling_strategy: str = "remove_before_training",
) -> Dict[str, Any]:
    """Generate a clean, JSON-serializable pipeline specification dictionary."""
    return {
        "selected_numerical_features": list(selected_numerical_features),
        "selected_categorical_features": list(selected_categorical_features),
        "excluded_features": list(excluded_features),
        "exclusion_reasons": dict(exclusion_reasons),
        "numerical_imputation_strategy": numerical_imputation_strategy,
        "categorical_imputation_strategy": categorical_imputation_strategy,
        "encoding_strategy": encoding_strategy,
        "scaling_strategy": scaling_strategy if scale_numeric else "none",
        "scale_numeric": bool(scale_numeric),
        "duplicate_handling_strategy": duplicate_handling_strategy,
        "handle_unknown_categories": "ignore",
    }


def recreate_column_transformer_from_config(config: Dict[str, Any]) -> ColumnTransformer:
    """Instantiate an un-fitted ColumnTransformer directly from a serializable config dictionary.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Ready-to-fit ColumnTransformer instance.
    """
    return build_column_transformer(
        numerical_features=config.get("selected_numerical_features", []),
        categorical_features=config.get("selected_categorical_features", []),
        numerical_imputation=config.get("numerical_imputation_strategy", "median"),
        categorical_imputation=config.get("categorical_imputation_strategy", "most_frequent"),
        scale_numeric=config.get("scale_numeric", True),
        handle_unknown=config.get("handle_unknown_categories", "ignore"),
    )


def validate_preprocessing_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that preprocessing configuration adheres to supported parameters."""
    num_strat = config.get("numerical_imputation_strategy", "median")
    if num_strat not in SUPPORTED_NUMERICAL_IMPUTATION:
        return False, f"Unsupported numerical imputation strategy: '{num_strat}'"

    cat_strat = config.get("categorical_imputation_strategy", "most_frequent")
    if cat_strat not in SUPPORTED_CATEGORICAL_IMPUTATION:
        return False, f"Unsupported categorical imputation strategy: '{cat_strat}'"

    num_features = config.get("selected_numerical_features", [])
    cat_features = config.get("selected_categorical_features", [])

    if not isinstance(num_features, list) or not isinstance(cat_features, list):
        return False, "Feature lists must be valid lists."

    if len(num_features) == 0 and len(cat_features) == 0:
        return False, "No usable features selected for preprocessing pipeline."

    return True, "Configuration is valid."
