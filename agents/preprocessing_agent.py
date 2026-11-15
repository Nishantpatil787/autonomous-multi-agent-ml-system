"""Preprocessing Agent for autonomous pipeline design and feature transformation planning."""

from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from tools import preprocessing_tools
from utils.config import Config
from utils.state import AgentState, WorkflowStatus


class PreprocessingAgent(BaseAgent):
    """Specialized agent responsible for determining feature preprocessing strategies and building reproducible pipelines."""

    def __init__(self, name: str = "PreprocessingAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        scale_numeric: Optional[bool] = None,
        numerical_imputation: Optional[str] = None,
        categorical_imputation: Optional[str] = None,
        duplicate_strategy: Optional[str] = None,
        max_cardinality: Optional[int] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Execute preprocessing planning based on Phase 2 analysis state.

        Args:
            state: The shared workflow state containing Phase 2 analysis outputs.
            scale_numeric: Optional boolean override for numeric scaling.
            numerical_imputation: Optional override for numerical imputation strategy.
            categorical_imputation: Optional override for categorical imputation strategy.
            duplicate_strategy: Optional override for duplicate row handling.
            max_cardinality: Optional override for maximum categorical cardinality.

        Returns:
            Updated AgentState containing preprocessing decisions and pipeline specification.
        """
        # 1. Verify prior analysis has taken place
        if state.workflow_status not in [
            WorkflowStatus.ANALYSIS_COMPLETED,
            WorkflowStatus.PREPROCESSING,
        ] and not state.analysis_summary:
            err_msg = (
                "Cannot execute PreprocessingAgent: Data Analysis has not been completed. "
                f"Current status is '{state.workflow_status.value}'."
            )
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        self.logger.info("Initiating autonomous preprocessing pipeline configuration.")
        state.workflow_status = WorkflowStatus.PREPROCESSING
        state.add_log(self.name, "Starting preprocessing strategy formulation.")

        # 2. Extract analysis inputs
        total_rows = state.dataset_shape[0] if state.dataset_shape else 0
        feature_names = list(state.feature_names)
        num_features = list(state.numerical_features)
        cat_features = list(state.categorical_features)
        quality_issues = state.data_quality_issues
        cat_stats = state.categorical_statistics

        # 3. Identify usable vs excluded features
        feature_filter = preprocessing_tools.filter_usable_features(
            feature_names=feature_names,
            numerical_features=num_features,
            categorical_features=cat_features,
            categorical_statistics=cat_stats,
            data_quality_issues=quality_issues,
            max_cardinality=max_cardinality or Config.MAX_CATEGORICAL_CARDINALITY,
            total_rows=total_rows,
        )

        selected_num = feature_filter["selected_numerical_features"]
        selected_cat = feature_filter["selected_categorical_features"]
        excluded = feature_filter["excluded_features"]
        exclusion_reasons = feature_filter["exclusion_reasons"]

        self.logger.info(
            f"Feature filtering complete: {len(selected_num)} numerical and {len(selected_cat)} "
            f"categorical features selected. {len(excluded)} feature(s) excluded."
        )

        # 4. Resolve strategies
        scaling_enabled = (
            scale_numeric if scale_numeric is not None else Config.DEFAULT_SCALE_NUMERIC
        )
        num_impute = numerical_imputation or Config.DEFAULT_NUMERICAL_IMPUTATION
        cat_impute = categorical_imputation or Config.DEFAULT_CATEGORICAL_IMPUTATION
        dup_handling = duplicate_strategy or Config.DEFAULT_DUPLICATE_STRATEGY

        dup_detected = state.duplicate_count or state.duplicate_rows_count
        dup_removed = dup_detected if dup_handling == "remove_before_training" else 0

        # 5. Build serializable pipeline configuration
        pipe_config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=selected_num,
            selected_categorical_features=selected_cat,
            excluded_features=excluded,
            exclusion_reasons=exclusion_reasons,
            numerical_imputation_strategy=num_impute,
            categorical_imputation_strategy=cat_impute,
            encoding_strategy="one_hot",
            scaling_strategy="standard_scaler" if scaling_enabled else "none",
            scale_numeric=scaling_enabled,
            duplicate_handling_strategy=dup_handling,
        )

        # 6. Validate configuration and test ColumnTransformer instantiation
        is_valid, validation_msg = preprocessing_tools.validate_preprocessing_config(pipe_config)
        if not is_valid:
            err_msg = f"Preprocessing configuration validation failed: {validation_msg}"
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        try:
            # Instantiate to verify structure without fitting
            transformer = preprocessing_tools.recreate_column_transformer_from_config(pipe_config)
            self.logger.info(f"Verified ColumnTransformer structure with {len(transformer.transformers)} transformers.")
        except Exception as exc:
            err_msg = f"Failed to instantiate ColumnTransformer from config: {str(exc)}"
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        # 7. Update AgentState
        state.original_feature_count = len(feature_names)
        state.selected_numerical_features = selected_num
        state.selected_categorical_features = selected_cat
        state.excluded_features = excluded
        state.exclusion_reasons = exclusion_reasons
        state.numerical_imputation_strategy = num_impute
        state.categorical_imputation_strategy = cat_impute
        state.encoding_strategy = "one_hot"
        state.scaling_strategy = "standard_scaler" if scaling_enabled else "none"
        state.scale_numeric = scaling_enabled
        state.duplicate_handling_strategy = dup_handling
        state.duplicate_rows_detected = dup_detected
        state.duplicate_rows_removed = dup_removed
        state.constant_columns_excluded = feature_filter["constant_columns_excluded"]
        state.high_cardinality_columns_excluded = feature_filter["high_cardinality_columns_excluded"]
        state.datetime_columns_excluded = feature_filter["datetime_columns_excluded"]
        state.pipeline_config = pipe_config
        state.preprocessing_decisions = pipe_config

        summary: Dict[str, Any] = {
            "original_features": len(feature_names),
            "selected_features_count": len(selected_num) + len(selected_cat),
            "selected_numerical_features": selected_num,
            "selected_categorical_features": selected_cat,
            "excluded_features": excluded,
            "exclusion_reasons": exclusion_reasons,
            "numerical_imputation": num_impute,
            "categorical_imputation": cat_impute,
            "encoding": "one_hot (handle_unknown='ignore')",
            "scale_numeric": scaling_enabled,
            "duplicate_handling": dup_handling,
            "duplicates_detected": dup_detected,
            "duplicates_to_remove": dup_removed,
        }
        state.preprocessing_summary = summary
        state.workflow_status = WorkflowStatus.PREPROCESSING_COMPLETED

        state.add_log(
            self.name,
            f"Preprocessing pipeline successfully built. {len(selected_num)} numerical and {len(selected_cat)} categorical features retained.",
            details=summary,
        )
        self.logger.info("Preprocessing Agent completed successfully.")
        return state
