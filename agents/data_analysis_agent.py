"""Data Analysis Agent for autonomous tabular dataset inspection and profiling."""

from pathlib import Path
from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from tools import data_tools
from utils.state import AgentState, ProblemType, WorkflowStatus


class DataAnalysisAgent(BaseAgent):
    """Specialized agent responsible for inspecting and profiling tabular CSV datasets."""

    def __init__(self, name: str = "DataAnalysisAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Execute comprehensive dataset inspection and update shared state.

        Args:
            state: The shared workflow state.
            dataset_path: Optional override for dataset file path.
            target_column: Optional explicit target column name.

        Returns:
            Updated AgentState containing analysis results and diagnostics.
        """
        # Determine dataset path
        target_path = dataset_path or state.dataset_path
        if not target_path:
            err_msg = "Dataset path not provided in state or arguments."
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        self.logger.info(f"Beginning dataset analysis on: {target_path}")
        state.workflow_status = WorkflowStatus.ANALYZING
        state.dataset_path = str(target_path)
        state.dataset_name = Path(target_path).name
        state.add_log(self.name, f"Starting analysis on '{state.dataset_name}'")

        # 1. Load CSV
        try:
            df = data_tools.load_csv(target_path)
        except Exception as exc:
            err_msg = f"Failed to load dataset: {str(exc)}"
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        # 2. Extract Shape & Columns
        rows, cols = data_tools.get_dataset_shape(df)
        state.dataset_shape = (rows, cols)
        state.column_names = list(df.columns)
        self.logger.info(f"Loaded dataset with {rows} rows and {cols} columns.")

        # 3. Analyze Target Column & Detection Method
        explicit_target = target_column or state.target_column
        target_info = data_tools.analyze_target(df, explicit_target=explicit_target)
        resolved_target = target_info["target_column"]
        detection_method = target_info["target_detection_method"]

        state.target_column = resolved_target
        state.target_detection_method = detection_method
        self.logger.info(
            f"Target column: '{resolved_target}' (Method: {detection_method}, Confidence: {target_info['confidence']})"
        )

        # 4. Feature Type Classification (excluding target)
        feature_types = data_tools.detect_feature_types(df, target_column=resolved_target)
        state.numerical_features = feature_types["numerical"]
        state.categorical_features = feature_types["categorical"]
        state.feature_names = state.numerical_features + state.categorical_features

        # 5. Missing Values & Duplicate Records
        missing_analysis = data_tools.analyze_missing_values(df)
        state.missing_values = missing_analysis
        state.missing_value_summary = {
            col: count
            for col, count in missing_analysis["per_column"].items()
            if count > 0
        }

        dup_count = data_tools.count_duplicates(df)
        state.duplicate_count = dup_count
        state.duplicate_rows_count = dup_count

        # 6. Summary Statistics
        state.numerical_statistics = data_tools.calculate_numerical_statistics(
            df, state.numerical_features
        )
        state.categorical_statistics = data_tools.calculate_categorical_statistics(
            df, state.categorical_features
        )

        # 7. Problem Type & Class Distribution (if target detected)
        if resolved_target and resolved_target in df.columns:
            target_series = df[resolved_target]
            detected_type, problem_reason = data_tools.detect_problem_type(target_series)
            state.problem_type = detected_type
            self.logger.info(f"Detected problem type: {detected_type} - {problem_reason}")

            if detected_type in [
                ProblemType.BINARY_CLASSIFICATION.value,
                ProblemType.MULTICLASS_CLASSIFICATION.value,
            ]:
                state.class_distribution = data_tools.calculate_class_distribution(target_series)
            else:
                state.class_distribution = {}
        else:
            state.problem_type = ProblemType.UNSUPPORTED_OR_UNCERTAIN.value
            state.class_distribution = {}
            self.logger.warning("No target column identified. Problem type set to unsupported_or_uncertain.")

        # 8. Data Quality Checks
        quality_issues = data_tools.detect_data_quality_issues(
            df, target_column=resolved_target
        )
        state.data_quality_issues = quality_issues

        # 9. Compile Structured Analysis Summary
        summary_payload: Dict[str, Any] = {
            "dataset_shape": {"rows": rows, "columns": cols},
            "numerical_features": state.numerical_features,
            "categorical_features": state.categorical_features,
            "missing_values": {
                "total": missing_analysis["total"],
                "columns_with_missing": missing_analysis["columns_with_missing"],
            },
            "duplicate_count": dup_count,
            "target_column": resolved_target,
            "target_detection_method": detection_method,
            "problem_type": state.problem_type,
            "class_distribution": state.class_distribution,
            "data_quality_issues": quality_issues,
        }

        state.analysis_report = summary_payload
        state.analysis_summary = summary_payload
        state.workflow_status = WorkflowStatus.ANALYSIS_COMPLETED
        state.add_log(
            self.name,
            f"Dataset analysis completed successfully for '{state.dataset_name}'.",
            details={
                "rows": rows,
                "columns": cols,
                "problem_type": state.problem_type,
                "target": resolved_target,
            },
        )
        self.logger.info("Dataset analysis successfully finished and state updated.")

        return state
