"""Shared helpers and fixtures for the Phase 2 test suite."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

BINARY_DATASET = DATASETS_DIR / "sample_classification.csv"
MULTICLASS_DATASET = DATASETS_DIR / "sample_multiclass.csv"

# Expected properties of the committed synthetic fixtures.
BINARY_EXPECTED = {
    "rows": 123,
    "columns": 9,
    "target": "Outcome",
    "missing_total": 17,
    "missing_by_column": {"age": 6, "bmi": 5, "smoking": 4, "region": 2},
    "duplicates": 3,
    "numerical_features": ["record_id", "age", "bmi", "glucose", "blood_pressure"],
    "categorical_features": ["gender", "region", "smoking"],
}

MULTICLASS_EXPECTED = {
    "rows": 92,
    "columns": 7,
    "target": "class_label",
    "missing_total": 6,
    "duplicates": 2,
    "class_count": 3,
}


def file_hash(path: Path) -> str:
    """Return the SHA-256 hash of a file, to prove it was not modified."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(directory: Path, name: str, frame: pd.DataFrame) -> Path:
    """Write a temporary CSV fixture and return its path."""
    path = directory / name
    frame.to_csv(path, index=False)
    return path


class ProjectArtifactIsolationGuard:
    """Fail the suite if a test writes into the project's own ``models/``.

    Phase 9 made persistence part of the default workflow, and the Phase 8
    integration tests drive the *real* agents on the *real* datasets. A test that
    forgets to redirect the artifact directory therefore does not merely risk a
    flaky assertion: it silently replaces the committed artifact with a
    different, usually worse one, and nothing reports it.

    A missing assertion cannot be added after the fact, so this guard is
    installed around the whole session instead: it hashes every file in
    ``models/`` before the first test and again after the last, and reports
    anything that appeared, vanished or changed.

    The guard is a safety net, not a substitute for passing an explicit output
    directory. It catches the mistake; it does not explain which test made it.
    """

    def __init__(self, directory: Path = MODELS_DIR) -> None:
        """Record which directory is being protected.

        Args:
            directory: The artifact directory that must stay untouched.
        """
        self.directory = Path(directory)
        self.before: Dict[str, str] = {}
        self.checked = False

    def snapshot(self) -> Dict[str, str]:
        """Hash every file in the protected directory.

        Returns:
            A mapping of file name to SHA-256, empty when the directory is
            absent (a fresh checkout with no run yet is a valid state).
        """
        if not self.directory.is_dir():
            return {}
        return {
            path.name: file_hash(path)
            for path in sorted(self.directory.iterdir())
            if path.is_file()
        }

    def start(self) -> None:
        """Take the baseline snapshot."""
        self.before = self.snapshot()

    def verify(self) -> List[str]:
        """Compare the current directory against the baseline.

        Returns:
            A human-readable description of every difference. An empty list
            means the protected directory is byte-identical to how it started.
        """
        after = self.snapshot()
        problems: List[str] = []
        for name in sorted(set(self.before) - set(after)):
            problems.append(f"{name}: deleted by a test")
        for name in sorted(set(after) - set(self.before)):
            problems.append(f"{name}: created by a test")
        for name in sorted(set(after) & set(self.before)):
            if after[name] != self.before[name]:
                problems.append(f"{name}: overwritten by a test")
        self.checked = True
        return problems


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to CSV bytes."""
    return frame.to_csv(index=False).encode("utf-8")


def issue_kinds(issues) -> Dict[str, int]:
    """Count data quality issues per kind."""
    counts: Dict[str, int] = {}
    for issue in issues:
        counts[issue.kind] = counts.get(issue.kind, 0) + 1
    return counts


def issues_for_column(issues, column: str):
    """Return all issues attached to one column."""
    return [issue for issue in issues if issue.column == column]


# --------------------------------------------------------------------------- #
# Synthetic frames for the Phase 3 (preprocessing) tests
# --------------------------------------------------------------------------- #
def frame_mixed(rows: int = 40) -> pd.DataFrame:
    """A frame containing every case the Preprocessing Agent must handle.

    Includes: an identifier column, numerical features with missing values,
    categorical features with missing values, a constant column, a datetime
    column and a binary target.
    """
    records = []
    for index in range(rows):
        records.append(
            {
                "record_id": index + 1,
                "age": 20 + (index % 30),
                "bmi": round(19.0 + (index % 15) * 0.7, 2),
                "city": ["north", "south", "east", "west"][index % 4],
                "signup_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "constant_col": "same",
                "churn": index % 2,
            }
        )
    frame = pd.DataFrame(records)

    # Missing values in both feature types.
    frame.loc[[1, 5, 9], "age"] = None
    frame.loc[[2, 6], "bmi"] = None
    frame.loc[[3, 7, 8], "city"] = None
    return frame


def frame_clean_numerical(rows: int = 30) -> pd.DataFrame:
    """Numerical features only, no missing values, no duplicates.

    Values repeat across rows on purpose: a column that is unique per row would
    legitimately be treated as an identifier and excluded.
    """
    return pd.DataFrame(
        {
            "feature_a": [float(20 + (index % 6)) for index in range(rows)],
            "feature_b": [float(1.5 * (index % 8)) for index in range(rows)],
        }
    )


def frame_clean_categorical(rows: int = 30) -> pd.DataFrame:
    """Categorical features only, no missing values."""
    return pd.DataFrame(
        {
            "colour": ["red", "green", "blue"] * (rows // 3) + ["red"] * (rows % 3),
            "size": ["s", "m", "l"] * (rows // 3) + ["s"] * (rows % 3),
            "label": [index % 2 for index in range(rows)],
        }
    )


def frame_with_duplicates(rows: int = 20, duplicates: int = 3) -> pd.DataFrame:
    """Frame where the last rows are exact copies of earlier rows.

    `x` is unique per row and is therefore treated as an identifier, exactly as
    a real dataset would be. `label` is the classification target.
    """
    frame = pd.DataFrame(
        {
            "x": [float(index) for index in range(rows)],
            "y": [index % 3 for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )
    duplicated = frame.iloc[:duplicates].copy()
    return pd.concat([frame, duplicated], ignore_index=True)


def frame_with_high_cardinality(rows: int = 200) -> pd.DataFrame:
    """Frame with a text column that must not be one-hot encoded.

    The column has 80 distinct values over 200 rows: the unique ratio (0.4)
    stays below the identifier threshold, but one-hot encoding would create
    about 81 columns, which is unreasonable.
    """
    return pd.DataFrame(
        {
            "amount": [float(10 + (index % 25)) for index in range(rows)],
            "free_text": [f"note number {index % 80}" for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def frame_with_constant_and_empty(rows: int = 20) -> pd.DataFrame:
    """Frame with a constant column and a completely empty column."""
    frame = pd.DataFrame(
        {
            "good": [float(10 + (index % 5)) for index in range(rows)],
            "constant_col": ["fixed"] * rows,
            "empty_col": [None] * rows,
            "label": [index % 2 for index in range(rows)],
        }
    )
    return frame


def frame_identifier_only(rows: int = 15) -> pd.DataFrame:
    """Frame whose only feature is an identifier-like column."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{index:04d}" for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )