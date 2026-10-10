"""Data Analysis Agent (Phase 2).

The agent inspects a CSV dataset and fills the `AgentState` with structured,
reusable information for the agents that run later:

    Dataset Analysis Agent
        -> Preprocessing Agent (Phase 3)
        -> Model Selection Agent (Phase 4)

Responsibilities:
    1. Validate the dataset path.
    2. Load the CSV (read-only, the source file is never modified).
    3. Analyse shape, columns, feature types, missing values, duplicates,
       statistics, class distribution and data quality.
    4. Resolve the target column (explicit input or heuristic detection).
    5. Detect the problem type (binary / multiclass classification).
    6. Write everything into the shared state and log the main operations.

The agent never preprocesses data, never removes rows or columns and never
trains a model. Problems are reported only.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from agents.base_agent import BaseAgent
from tools.data_tools import (
    DataToolError,
    DatasetAnalysis,
    analyze_dataset,
    load_csv,
)
from utils.state import (
    AgentStatus,
    AnalysisSummary,
    AgentState,
    ProblemType,
    Severity,
)


class DataAnalysisAgent(BaseAgent):
    """Agent that analyses a tabular dataset and populates the shared state."""

    def __init__(self) -> None:
        """Initialize the Data Analysis Agent."""
        super().__init__(name="DataAnalysisAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        state: AgentState,
        dataset_path: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
    ) -> AgentState:
        """Analyse a dataset and update the shared state.

        Args:
            state: Shared state to update. When `state.dataset_path` is already
                set it is used as default dataset path.
            dataset_path: Path to the CSV dataset. Optional if the state
                already contains one.
            target_column: Target column supplied by the user. When omitted the
                agent attempts heuristic detection and reports its confidence.

        Returns:
            The updated `AgentState`.

        Raises:
            DataToolError: The dataset could not be loaded or analysed. The
                error is logged and recorded in the state before being
                re-raised, so it is never silently swallowed.
        """
        state.status = AgentStatus.RUNNING.value
        resolved_path = dataset_path or state.dataset_path
        if not resolved_path:
            message = "No dataset path was provided to the Data Analysis Agent."
            self.log_message(state, message, level="error")
            state.add_error(message)
            raise DataToolError(message)

        state.dataset_path = str(resolved_path)
        state.dataset_name = Path(resolved_path).name
        # A target already present in the state (e.g. chosen by the user in an
        # earlier step) is honoured, so re-running the agent stays stable.
        explicit_target = target_column or state.target_column
        if explicit_target:
            state.target_column = str(explicit_target)

        self.log_message(state, f"Analysing dataset '{state.dataset_name}'.")

        try:
            dataframe = load_csv(resolved_path)
            analysis = analyze_dataset(dataframe, explicit_target=explicit_target)
            self._apply_analysis(state, dataframe, analysis)
        except DataToolError as error:
            self.log_message(state, f"Data analysis failed: {error}", level="error")
            state.add_error(str(error))
            raise
        except Exception as error:  # pragma: no cover - defensive, never hidden
            self.log_message(
                state, f"Unexpected error during data analysis: {error!r}", level="error"
            )
            state.add_error(f"Unexpected error during data analysis: {error!r}")
            raise

        state.status = AgentStatus.ANALYSIS_COMPLETED.value
        return state

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _apply_analysis(
        self,
        state: AgentState,
        dataframe: pd.DataFrame,
        analysis: DatasetAnalysis,
    ) -> None:
        """Copy the analysis result into the shared state."""
        # Dataset structure
        state.dataset_shape = analysis.shape
        state.column_names = [str(column) for column in dataframe.columns]
        state.column_dtypes = analysis.column_dtypes

        # Features (the target, when known, is not a feature)
        target_column = analysis.target_detection.column
        state.numerical_features = list(analysis.feature_types.numerical_features)
        state.categorical_features = list(analysis.feature_types.categorical_features)
        state.unsupported_columns = list(analysis.feature_types.unsupported_columns)
        state.feature_names = state.numerical_features + state.categorical_features

        # Target column
        state.target_column = target_column
        state.target_detection_method = analysis.target_detection.method
        state.target_detection_confidence = analysis.target_detection.confidence
        state.target_detection_reason = analysis.target_detection.reason
        state.target_candidates = list(analysis.target_detection.candidates)

        # Problem type
        target = analysis.target_analysis
        if target is not None:
            state.problem_type = target.problem_type.value
            state.problem_type_reason = target.problem_type_reason
            state.class_distribution = dict(target.class_distribution)
        else:
            state.problem_type = ProblemType.UNSUPPORTED_OR_UNCERTAIN.value
            state.problem_type_reason = (
                "No target column could be resolved, so the problem type is unknown."
            )
            state.class_distribution = {}

        # Data quality statistics
        state.total_missing_values = analysis.missing_values.total
        state.missing_values = analysis.missing_values
        state.duplicate_count = analysis.duplicate_count
        state.numerical_statistics = analysis.numerical_statistics
        state.categorical_statistics = analysis.categorical_statistics
        state.data_quality_issues = list(analysis.data_quality_issues)
        state.analysis_summary = self._build_summary(state)

        self._log_findings(state, analysis)

    def _build_summary(self, state: AgentState) -> AnalysisSummary:
        """Create a compact, display-ready summary of the analysis."""
        severity_counts = {severity.value: 0 for severity in Severity}
        for issue in state.data_quality_issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        shape = state.dataset_shape
        rows = shape.rows if shape else 0
        columns = shape.columns if shape else 0
        missing_total = state.total_missing_values

        if state.target_column is None:
            target_text = "no target column resolved"
        elif state.target_detection_method == "explicit":
            target_text = f"target '{state.target_column}' (user provided)"
        else:
            target_text = (
                f"target '{state.target_column}' (auto-detected, "
                f"{state.target_detection_confidence} confidence)"
            )

        summary_text = (
            f"Dataset '{state.dataset_name}' has {rows} rows and {columns} columns "
            f"({len(state.numerical_features)} numerical, "
            f"{len(state.categorical_features)} categorical features). "
            f"Missing values: {missing_total}; duplicated rows: {state.duplicate_count}. "
            f"Identified {target_text} with problem type "
            f"'{state.problem_type}'. "
            f"Data quality issues: {len(state.data_quality_issues)}."
        )

        return AnalysisSummary(
            dataset_name=state.dataset_name,
            rows=rows,
            columns=columns,
            numerical_feature_count=len(state.numerical_features),
            categorical_feature_count=len(state.categorical_features),
            total_missing_values=missing_total,
            duplicate_count=state.duplicate_count,
            target_column=state.target_column,
            target_detection_method=state.target_detection_method or "not_detected",
            problem_type=state.problem_type,
            problem_type_reason=state.problem_type_reason,
            issue_count=len(state.data_quality_issues),
            issues_by_severity=severity_counts,
            summary_text=summary_text,
        )

    def _log_findings(self, state: AgentState, analysis: DatasetAnalysis) -> None:
        """Log the important findings of the analysis into the state."""
        shape = analysis.shape
        self.log_message(
            state,
            f"Dataset shape: {shape.rows} rows x {shape.columns} columns.",
        )
        self.log_message(
            state,
            f"Feature types: {len(analysis.feature_types.numerical_features)} numerical, "
            f"{len(analysis.feature_types.categorical_features)} categorical, "
            f"{len(analysis.feature_types.unsupported_columns)} unsupported.",
        )
        self.log_message(
            state,
            f"Missing values: {analysis.missing_values.total} in "
            f"{analysis.missing_values.affected_columns} column(s).",
        )
        self.log_message(state, f"Duplicate rows: {analysis.duplicate_count}.")

        detection = analysis.target_detection
        if detection.column is None:
            self.log_message(
                state,
                f"Target column was not detected. {detection.reason}",
                level="warning",
            )
        else:
            self.log_message(
                state,
                f"Target column '{detection.column}' resolved via {detection.method} "
                f"(confidence: {detection.confidence}).",
            )

        target = analysis.target_analysis
        if target is not None:
            if target.is_supported:
                self.log_message(state, f"Problem type: {target.problem_type.value}.")
                if target.problem_type == ProblemType.BINARY_CLASSIFICATION:
                    self.log_message(state, _format_class_distribution(target.class_distribution))
                elif target.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
                    classes = ", ".join(
                        f"{label} ({entry.count} rows, {entry.percentage}%)"
                        for label, entry in target.class_distribution.items()
                    )
                    self.log_message(
                        state,
                        f"Class distribution over {target.class_count} classes: {classes}.",
                    )
            else:
                self.log_message(
                    state, f"Problem type is not supported: {target.problem_type_reason}",
                    level="warning",
                )

        if analysis.data_quality_issues:
            errors = sum(
                1 for issue in analysis.data_quality_issues if issue.severity == Severity.ERROR.value
            )
            warnings = sum(
                1
                for issue in analysis.data_quality_issues
                if issue.severity == Severity.WARNING.value
            )
            self.log_message(
                state,
                f"Data quality issues reported (not modified): {len(analysis.data_quality_issues)} "
                f"total, {errors} error(s), {warnings} warning(s).",
                level="warning" if errors or warnings else "info",
            )
        else:
            self.log_message(state, "No data quality issues detected.")


def _format_class_distribution(distribution: dict) -> str:
    """Render a binary class distribution as one readable line."""
    parts: List[str] = [
        f"{label}: {entry.count} samples ({entry.percentage}%)"
        for label, entry in distribution.items()
    ]
    return "Class distribution: " + ", ".join(parts) + "."