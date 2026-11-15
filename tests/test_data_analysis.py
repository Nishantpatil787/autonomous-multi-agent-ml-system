"""Comprehensive unit and integration tests for the Data Analysis Agent and tools."""

import pytest
import pandas as pd
from pathlib import Path

from tools import data_tools
from agents.data_analysis_agent import DataAnalysisAgent
from utils.state import AgentState, ProblemType, WorkflowStatus

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
SAMPLE_BINARY_CSV = DATASETS_DIR / "sample_classification.csv"
SAMPLE_MULTI_CSV = DATASETS_DIR / "sample_multiclass.csv"


def test_load_csv_valid() -> None:
    """Verify loading of a valid CSV file."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 60


def test_load_csv_invalid_path() -> None:
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data_tools.load_csv("datasets/non_existent_file.csv")


def test_dataset_shape_detection() -> None:
    """Verify accurate extraction of rows and columns."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    rows, cols = data_tools.get_dataset_shape(df)
    assert rows == 60
    assert cols == 6


def test_feature_type_detection() -> None:
    """Verify accurate segregation of numerical and categorical columns."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    feature_types = data_tools.detect_feature_types(df, target_column="Outcome")
    
    # age, salary, credit_score should be numerical
    assert "age" in feature_types["numerical"]
    assert "salary" in feature_types["numerical"]
    assert "credit_score" in feature_types["numerical"]

    # department, education should be categorical
    assert "department" in feature_types["categorical"]
    assert "education" in feature_types["categorical"]
    
    # Target column must be excluded from feature lists
    assert "Outcome" not in feature_types["numerical"]
    assert "Outcome" not in feature_types["categorical"]


def test_missing_values_analysis() -> None:
    """Verify missing value counts and percentage calculations."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    analysis = data_tools.analyze_missing_values(df)

    assert analysis["total"] == 3
    assert analysis["per_column"]["salary"] == 2
    assert analysis["per_column"]["education"] == 1
    assert "salary" in analysis["columns_with_missing"]
    assert "education" in analysis["columns_with_missing"]
    assert analysis["percentage"]["salary"] > 0


def test_duplicate_count() -> None:
    """Verify exact count of duplicate records."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    dups = data_tools.count_duplicates(df)
    assert dups == 2


def test_target_detection() -> None:
    """Verify explicit target selection and heuristic detection."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)

    # 1. Heuristic detection of common name 'Outcome'
    heuristic = data_tools.analyze_target(df)
    assert heuristic["target_column"] == "Outcome"
    assert heuristic["target_detection_method"] == "heuristic"

    # 2. Explicit target specification
    explicit = data_tools.analyze_target(df, explicit_target="Outcome")
    assert explicit["target_column"] == "Outcome"
    assert explicit["target_detection_method"] == "explicit"

    # 3. Non-existent explicit target
    invalid = data_tools.analyze_target(df, explicit_target="non_existent")
    assert invalid["target_column"] is None


def test_binary_classification_detection() -> None:
    """Verify that a target with 2 unique values is classified as binary classification."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    prob_type, reason = data_tools.detect_problem_type(df["Outcome"])
    assert prob_type == ProblemType.BINARY_CLASSIFICATION.value
    assert "2 unique classes" in reason


def test_multiclass_classification_detection() -> None:
    """Verify that a target with 3 classes is classified as multiclass classification."""
    df = data_tools.load_csv(SAMPLE_MULTI_CSV)
    prob_type, reason = data_tools.detect_problem_type(df["class_label"])
    assert prob_type == ProblemType.MULTICLASS_CLASSIFICATION.value
    assert "3 discrete classes" in reason


def test_class_distribution_calculation() -> None:
    """Verify class distribution calculation with counts and percentages."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    dist = data_tools.calculate_class_distribution(df["Outcome"])
    
    assert "0" in dist
    assert "1" in dist
    total_count = dist["0"]["count"] + dist["1"]["count"]
    assert total_count == 60
    assert round(dist["0"]["percentage"] + dist["1"]["percentage"], 1) == 100.0


def test_data_quality_issues() -> None:
    """Verify data quality issues detection for missing values and duplicates."""
    df = data_tools.load_csv(SAMPLE_BINARY_CSV)
    issues = data_tools.detect_data_quality_issues(df, target_column="Outcome")
    
    issue_types = [issue["type"] for issue in issues]
    assert "duplicate_rows" in issue_types
    assert "missing_values" in issue_types


def test_data_analysis_agent_execution() -> None:
    """Verify that DataAnalysisAgent executes end-to-end and updates AgentState correctly."""
    agent = DataAnalysisAgent()
    state = AgentState(dataset_path=str(SAMPLE_BINARY_CSV))

    updated_state = agent.run(state)

    assert updated_state.workflow_status == WorkflowStatus.ANALYSIS_COMPLETED
    assert updated_state.dataset_shape == (60, 6)
    assert updated_state.target_column == "Outcome"
    assert updated_state.problem_type == ProblemType.BINARY_CLASSIFICATION.value
    assert len(updated_state.numerical_features) == 3
    assert len(updated_state.categorical_features) == 2
    assert updated_state.duplicate_count == 2
    assert updated_state.missing_values["total"] == 3
    assert len(updated_state.agent_logs) >= 2
    assert updated_state.error_message is None
    assert updated_state.analysis_summary["dataset_shape"]["rows"] == 60


def test_data_analysis_agent_invalid_file() -> None:
    """Verify graceful handling and state failure recording on missing file."""
    agent = DataAnalysisAgent()
    state = AgentState(dataset_path="datasets/missing_non_existent.csv")

    updated_state = agent.run(state)

    assert updated_state.workflow_status == WorkflowStatus.FAILED
    assert updated_state.error_message is not None
    assert "not found" in updated_state.error_message.lower()
