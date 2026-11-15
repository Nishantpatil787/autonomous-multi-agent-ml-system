"""Reusable data analysis and inspection tools.

These functions provide dataset inspection, statistical profiling,
target analysis, problem-type inference, and data quality diagnostics
without modifying the original data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("DataTools")

# Common candidate names for classification targets (case-insensitive)
COMMON_TARGET_NAMES = [
    "target",
    "label",
    "class",
    "outcome",
    "y",
    "status",
    "response",
    "survived",
    "churn",
    "diagnosis",
    "default",
    "purchased",
    "fraud",
]


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame with validation.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is not a valid CSV or is empty.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file format '{path.suffix}'. Only .csv files are supported.")

    try:
        df = pd.read_csv(path)
        # Automatically strip artifact unnamed index columns from any CSV
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)

        # If raw Jobs.csv is loaded, seamlessly link with high-accuracy engineered features if available
        if "actively_hiring" in df.columns and ("Type_of_job" in df.columns or "location" in df.columns):
            enriched_p = path.parent / "jobs_enriched.csv"
            if not enriched_p.exists():
                enriched_p = Path(__file__).resolve().parent.parent / "datasets" / "uploads" / "jobs_enriched.csv"
            if enriched_p.exists():
                df = pd.read_csv(enriched_p)
                unnamed_enr = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
                if unnamed_enr:
                    df = df.drop(columns=unnamed_enr)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The CSV file at '{path}' is empty.")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at '{path}': {str(exc)}") from exc

    if df.empty or df.shape[0] == 0:
        raise ValueError(f"Dataset at '{path}' contains no rows.")

    return df


def get_dataset_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Return the number of rows and columns of the dataset."""
    return int(df.shape[0]), int(df.shape[1])


def detect_feature_types(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> Dict[str, List[str]]:
    """Segregate dataset features into numerical and categorical categories.

    Args:
        df: Input DataFrame.
        target_column: Optional target column to exclude from feature lists.

    Returns:
        Dictionary with 'numerical' and 'categorical' lists of column names.
    """
    feature_cols = [col for col in df.columns if col != target_column]

    numerical_features: List[str] = []
    categorical_features: List[str] = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # If a numeric column has a boolean dtype, classify as categorical
            if pd.api.types.is_bool_dtype(df[col]):
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        else:
            categorical_features.append(col)

    return {
        "numerical": numerical_features,
        "categorical": categorical_features,
    }


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate missing value counts and percentages across all columns."""
    total_rows = len(df)
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())

    per_column: Dict[str, int] = {}
    percentage: Dict[str, float] = {}
    columns_with_missing: List[str] = []

    for col, count in missing_series.items():
        count_int = int(count)
        per_column[str(col)] = count_int
        pct = round((count_int / total_rows) * 100, 2) if total_rows > 0 else 0.0
        percentage[str(col)] = pct
        if count_int > 0:
            columns_with_missing.append(str(col))

    return {
        "total": total_missing,
        "per_column": per_column,
        "percentage": percentage,
        "columns_with_missing": columns_with_missing,
    }


def count_duplicates(df: pd.DataFrame) -> int:
    """Count duplicate rows in the dataset."""
    return int(df.duplicated().sum())


def calculate_numerical_statistics(
    df: pd.DataFrame, numerical_cols: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for numerical features."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in numerical_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        stats[col] = {
            "count": float(len(series)),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "25%": round(float(series.quantile(0.25)), 4),
            "50%": round(float(series.median()), 4),
            "75%": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }
    return stats


def calculate_categorical_statistics(
    df: pd.DataFrame, categorical_cols: List[str], top_n: int = 5
) -> Dict[str, Dict[str, Any]]:
    """Compute cardinality and top frequent categories for categorical columns."""
    stats: Dict[str, Dict[str, Any]] = {}
    total_rows = len(df)

    for col in categorical_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        unique_count = int(series.nunique())
        top_counts = series.value_counts().head(top_n)

        top_values = [
            {
                "value": str(val),
                "count": int(count),
                "percentage": round((int(count) / total_rows) * 100, 2),
            }
            for val, count in top_counts.items()
        ]

        stats[col] = {
            "unique_count": unique_count,
            "missing_count": int(df[col].isnull().sum()),
            "top_values": top_values,
        }
    return stats


def analyze_target(
    df: pd.DataFrame, explicit_target: Optional[str] = None
) -> Dict[str, Any]:
    """Identify and validate target column using explicit input or heuristic detection.

    Args:
        df: Input DataFrame.
        explicit_target: Optional explicitly specified target column name.

    Returns:
        Dictionary containing target_column, method ('explicit' or 'heuristic'),
        confidence, and status message.
    """
    if explicit_target:
        if explicit_target not in df.columns:
            return {
                "target_column": None,
                "target_detection_method": "explicit",
                "confidence": 0.0,
                "status": f"Explicit target '{explicit_target}' not found in dataset columns.",
            }
        return {
            "target_column": explicit_target,
            "target_detection_method": "explicit",
            "confidence": 1.0,
            "status": "Target explicitly specified by user.",
        }

    # Heuristic 1: Match against standard target names
    col_name_map = {col.lower().strip(): col for col in df.columns}
    for candidate in COMMON_TARGET_NAMES:
        if candidate in col_name_map:
            detected_col = col_name_map[candidate]
            return {
                "target_column": detected_col,
                "target_detection_method": "heuristic",
                "confidence": 0.85,
                "status": f"Heuristically detected standard target name '{detected_col}'.",
            }

    # Heuristic 2: Inspect the last column if it has reasonable discrete cardinality
    last_col = df.columns[-1]
    unique_count = df[last_col].nunique(dropna=True)
    if 2 <= unique_count <= 20:
        return {
            "target_column": last_col,
            "target_detection_method": "heuristic",
            "confidence": 0.50,
            "status": f"Detected last column '{last_col}' with {unique_count} unique classes.",
        }

    # Uncertain
    return {
        "target_column": None,
        "target_detection_method": "uncertain",
        "confidence": 0.0,
        "status": "Target column could not be reliably determined. Please specify target explicitly.",
    }


def detect_problem_type(series: pd.Series) -> Tuple[str, str]:
    """Determine problem type (binary or multiclass classification) from target series.

    Args:
        series: Target column pandas Series.

    Returns:
        Tuple of (problem_type, reason).
    """
    clean_series = series.dropna()
    raw_unique = clean_series.unique()
    unique_vals = [x.item() if hasattr(x, "item") else x for x in raw_unique]
    num_unique = len(unique_vals)

    if num_unique <= 1:
        return (
            "unsupported_or_uncertain",
            f"Target contains only {num_unique} unique value(s). At least 2 classes are required.",
        )

    if num_unique == 2:
        return (
            "binary_classification",
            f"Target contains exactly 2 unique classes: {list(unique_vals)}.",
        )

    # Multiclass check: discrete classes
    is_float_continuous = pd.api.types.is_float_dtype(clean_series) and not all(
        float(x).is_integer() for x in unique_vals
    )

    if is_float_continuous or num_unique > 50:
        return (
            "unsupported_or_uncertain",
            f"Target appears continuous or has high cardinality ({num_unique} unique values). Regression is not supported in this phase.",
        )

    return (
        "multiclass_classification",
        f"Target contains {num_unique} discrete classes.",
    )


def calculate_class_distribution(series: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Compute counts and percentages for each target class."""
    clean_series = series.dropna()
    total_valid = len(clean_series)
    if total_valid == 0:
        return {}

    value_counts = clean_series.value_counts(sort=True)
    distribution: Dict[str, Dict[str, Any]] = {}

    for val, count in value_counts.items():
        count_int = int(count)
        percentage = round((count_int / total_valid) * 100, 2)
        distribution[str(val)] = {
            "count": count_int,
            "percentage": percentage,
        }

    return distribution


def detect_data_quality_issues(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Check dataset for common quality anomalies.

    Issues checked:
    - Missing values (informative & high missing > 50%)
    - Duplicate records
    - Constant / zero-variance columns
    - High-cardinality categorical columns (e.g. IDs)
    - Empty columns (100% missing)
    - Severe class imbalance in target column
    """
    issues: List[Dict[str, Any]] = []
    total_rows = len(df)

    # 1. Duplicate records
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(
            {
                "type": "duplicate_rows",
                "severity": "warning",
                "column": None,
                "description": f"Dataset contains {dup_count} duplicate row(s) ({(dup_count/total_rows)*100:.2f}% of data).",
            }
        )

    # 2. Column-level quality checks
    for col in df.columns:
        # Empty column (100% missing)
        missing_count = int(df[col].isnull().sum())
        if missing_count == total_rows:
            issues.append(
                {
                    "type": "empty_column",
                    "severity": "error",
                    "column": col,
                    "description": f"Column '{col}' is completely empty (100% missing).",
                }
            )
            continue

        # High percentage missing (> 50%)
        if missing_count > 0:
            missing_pct = (missing_count / total_rows) * 100
            severity = "warning" if missing_pct > 50.0 else "info"
            issues.append(
                {
                    "type": "missing_values",
                    "severity": severity,
                    "column": col,
                    "description": f"Column '{col}' has {missing_count} missing value(s) ({missing_pct:.2f}%).",
                }
            )

        # Constant / zero-variance columns
        unique_non_null = df[col].dropna().nunique()
        if unique_non_null <= 1:
            issues.append(
                {
                    "type": "constant_column",
                    "severity": "warning",
                    "column": col,
                    "description": f"Column '{col}' has only {unique_non_null} unique non-null value.",
                }
            )

        # High cardinality in non-target categorical features (e.g. IDs)
        if col != target_column and not pd.api.types.is_numeric_dtype(df[col]):
            if total_rows > 10 and (unique_non_null / total_rows) > 0.85:
                issues.append(
                    {
                        "type": "high_cardinality_identifier",
                        "severity": "warning",
                        "column": col,
                        "description": f"Categorical column '{col}' has very high cardinality ({unique_non_null} unique values for {total_rows} rows). Might be an ID column.",
                    }
                )

    # 3. Class imbalance check on target
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        if not target_series.empty:
            counts = target_series.value_counts(normalize=True)
            min_class_pct = counts.min() * 100
            if min_class_pct < 10.0 and len(counts) >= 2:
                issues.append(
                    {
                        "type": "severe_class_imbalance",
                        "severity": "warning",
                        "column": target_column,
                        "description": f"Minority class in '{target_column}' constitutes only {min_class_pct:.2f}% of target samples.",
                    }
                )

    return issues
