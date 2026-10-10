"""Preprocessing Agent (Phase 3).

The agent consumes the structured `AgentState` produced by the Data Analysis
Agent and turns the analysis into a reproducible scikit-learn preprocessing
pipeline.

Responsibilities:
    1. Verify that the Data Analysis Agent has completed.
    2. Read the analysis (feature types, quality issues, duplicates) from the
       shared state.
    3. Decide which columns become model features and why the others do not.
    4. Record every decision as typed records in the shared state.
    5. Build an **unfitted** `ColumnTransformer` from that configuration.

The agent does not train a model, does not fit the pipeline and never modifies
the source dataset. Fitting happens later, on training data only, which is what
prevents data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from agents.base_agent import BaseAgent
from tools.data_tools import DataToolError, load_csv
from tools.preprocessing_tools import (
    AnalysisNotReadyError,
    PreprocessingConfigError,
    PreprocessingError,
    build_column_transformer,
    estimate_transformed_feature_count,
    plan_preprocessing,
)
from utils import config as project_config
from utils.state import (
    AgentState,
    AgentStatus,
    ExclusionReason,
    PreprocessingConfig,
    PreprocessingStatus,
    PreprocessingSummary,
    ProblemType,
)


class PreprocessingAgent(BaseAgent):
    """Agent that builds the preprocessing pipeline from the analysis."""

    def __init__(self) -> None:
        """Initialize the Preprocessing Agent."""
        super().__init__(name="PreprocessingAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        state: AgentState,
        *,
        dataset_path: Optional[Union[str, Path]] = None,
        duplicate_strategy: Optional[str] = None,
        scale_numeric: Optional[bool] = None,
        numerical_imputation: Optional[str] = None,
        categorical_imputation: Optional[str] = None,
    ) -> AgentState:
        """Plan and build the preprocessing pipeline.

        Args:
            state: Shared state produced by the Data Analysis Agent.
            dataset_path: Optional override; defaults to `state.dataset_path`.
            duplicate_strategy: `remove_exact_duplicates` or `keep_all`.
            scale_numeric: Set to False to skip StandardScaler.
            numerical_imputation: Override for the numerical imputation.
            categorical_imputation: Override for the categorical imputation.

        Returns:
            The updated `AgentState`.

        Raises:
            AnalysisNotReadyError: Data analysis has not been completed.
            PreprocessingError: The pipeline could not be built. The error is
                logged and recorded in the state before being re-raised.
        """
        try:
            self._verify_analysis_complete(state)
            resolved_path = dataset_path or state.dataset_path
            dataframe = load_csv(resolved_path)

            skip_reason = self._skip_reason(state)
            if skip_reason is not None:
                return self._record_skip(state, skip_reason)

            preprocessing_config = self._plan(
                state,
                dataframe,
                duplicate_strategy=duplicate_strategy,
                scale_numeric=scale_numeric,
                numerical_imputation=numerical_imputation,
                categorical_imputation=categorical_imputation,
            )

            pipeline = build_column_transformer(preprocessing_config)
            self._store(state, preprocessing_config, dataframe, pipeline)
        except (AnalysisNotReadyError, PreprocessingConfigError, PreprocessingError, DataToolError) as error:
            state.preprocessing_status = PreprocessingStatus.FAILED.value
            self.log_message(state, f"Preprocessing failed: {error}", level="error")
            state.add_error(str(error))
            raise
        except Exception as error:  # pragma: no cover - defensive, never hidden
            state.preprocessing_status = PreprocessingStatus.FAILED.value
            self.log_message(
                state, f"Unexpected error during preprocessing: {error!r}", level="error"
            )
            state.add_error(f"Unexpected error during preprocessing: {error!r}")
            raise

        state.status = AgentStatus.PREPROCESSING_COMPLETED.value
        return state

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _verify_analysis_complete(self, state: AgentState) -> None:
        """Ensure the Data Analysis Agent produced a usable state."""
        ready_statuses = {
            AgentStatus.ANALYSIS_COMPLETED.value,
            AgentStatus.PREPROCESSING_COMPLETED.value,
            AgentStatus.COMPLETED.value,
        }
        if state.status not in ready_statuses or state.dataset_shape is None or not state.column_names:
            raise AnalysisNotReadyError(
                "The Data Analysis Agent has not been completed, so the Preprocessing Agent "
                "cannot start. Run DataAnalysisAgent.run(state, dataset_path=...) first."
            )
        if not state.dataset_path:
            raise AnalysisNotReadyError(
                "The shared state does not contain a dataset path, so the dataset cannot be read."
            )

    def _skip_reason(self, state: AgentState) -> Optional[str]:
        """Return why preprocessing cannot run, or None when it can."""
        if state.target_column is None:
            return (
                "No target column was resolved during data analysis. Preprocessing needs a "
                "target to exclude it from the features, so it was skipped."
            )
        if state.problem_type not in (
            ProblemType.BINARY_CLASSIFICATION.value,
            ProblemType.MULTICLASS_CLASSIFICATION.value,
        ):
            return (
                f"Problem type '{state.problem_type}' is not supported for machine learning "
                "preprocessing in this phase (only binary and multiclass classification). "
                "Preprocessing was skipped."
            )
        return None

    def _record_skip(self, state: AgentState, reason: str) -> AgentState:
        """Record a skipped run without building any pipeline."""
        state.preprocessing_status = PreprocessingStatus.SKIPPED.value
        state.preprocessing_config = None
        state.preprocessing_pipeline = None
        state.preprocessing_steps = []
        state.preprocessing_summary = PreprocessingSummary(
            status=PreprocessingStatus.SKIPPED.value,
            status_reason=reason,
            target_column=state.target_column,
            problem_type=state.problem_type,
            original_feature_count=max(len(state.column_names) - 1, 0),
            duplicate_rows_detected=state.duplicate_count,
            duplicate_strategy=project_config.DUPLICATE_STRATEGY,
            pipeline_ready=False,
            summary_text=f"Preprocessing skipped: {reason}",
        )
        self.log_message(state, f"Preprocessing skipped. {reason}", level="warning")
        return state

    def _plan(
        self,
        state: AgentState,
        dataframe: pd.DataFrame,
        *,
        duplicate_strategy: Optional[str],
        scale_numeric: Optional[bool],
        numerical_imputation: Optional[str],
        categorical_imputation: Optional[str],
    ) -> PreprocessingConfig:
        """Build the preprocessing configuration from the analysis."""
        return plan_preprocessing(
            dataframe,
            target_column=state.target_column,
            duplicate_count=state.duplicate_count,
            data_quality_issues=state.data_quality_issues,
            unsupported_columns=state.unsupported_columns,
            duplicate_strategy=duplicate_strategy or project_config.DUPLICATE_STRATEGY,
            scale_numeric=(
                project_config.SCALE_NUMERIC if scale_numeric is None else bool(scale_numeric)
            ),
            numerical_imputation=numerical_imputation or project_config.NUMERICAL_IMPUTATION_STRATEGY,
            categorical_imputation=(
                categorical_imputation or project_config.CATEGORICAL_IMPUTATION_STRATEGY
            ),
        )

    def _store(
        self,
        state: AgentState,
        preprocessing_config: PreprocessingConfig,
        dataframe: pd.DataFrame,
        pipeline: object,
    ) -> None:
        """Write the configuration, pipeline and summary into the state."""
        estimated_output = estimate_transformed_feature_count(dataframe, preprocessing_config)
        summary = self._build_summary(state, preprocessing_config, estimated_output)

        state.preprocessing_config = preprocessing_config
        state.preprocessing_steps = list(preprocessing_config.steps)
        # Unfitted runtime object: never serialised, always rebuildable from
        # `preprocessing_config` with build_column_transformer(config).
        state.preprocessing_pipeline = pipeline
        state.preprocessing_status = PreprocessingStatus.COMPLETED.value
        state.preprocessing_summary = summary

        self._log_decisions(state, preprocessing_config, estimated_output)

    def _build_summary(
        self,
        state: AgentState,
        preprocessing_config: PreprocessingConfig,
        estimated_output: int,
    ) -> PreprocessingSummary:
        """Create the display-ready preprocessing summary."""
        excluded_by_reason: Dict[str, int] = {}
        for item in preprocessing_config.excluded_features:
            excluded_by_reason[item.reason] = excluded_by_reason.get(item.reason, 0) + 1

        original_feature_count = max(len(state.column_names) - 1, 0)
        used = len(preprocessing_config.numerical_features) + len(
            preprocessing_config.categorical_features
        )

        if preprocessing_config.excluded_features:
            excluded_text = (
                f"Excluded {len(preprocessing_config.excluded_features)} column(s): "
                + ", ".join(
                    f"{item.column} ({item.reason})"
                    for item in preprocessing_config.excluded_features
                )
                + "."
            )
        else:
            excluded_text = "No column was excluded."

        summary_text = (
            f"Selected {used} of {original_feature_count} candidate features "
            f"({len(preprocessing_config.numerical_features)} numerical, "
            f"{len(preprocessing_config.categorical_features)} categorical). "
            f"{excluded_text} "
            f"Numerical imputation: {preprocessing_config.numerical_imputation}; "
            f"categorical imputation: {preprocessing_config.categorical_imputation}; "
            f"encoding: {preprocessing_config.encoding} "
            f"(handle_unknown='{preprocessing_config.encoding_handle_unknown}'); "
            f"scaling: "
            f"{preprocessing_config.scaling_strategy if preprocessing_config.scale_numeric else 'disabled'}. "
            f"Duplicate strategy: {preprocessing_config.duplicate_handling.strategy} "
            f"({preprocessing_config.duplicate_handling.duplicate_rows_detected} detected). "
            f"Estimated output features after transformation: {estimated_output}."
        )

        return PreprocessingSummary(
            status=PreprocessingStatus.COMPLETED.value,
            target_column=state.target_column,
            problem_type=state.problem_type,
            original_feature_count=original_feature_count,
            numerical_feature_count=len(preprocessing_config.numerical_features),
            categorical_feature_count=len(preprocessing_config.categorical_features),
            used_feature_count=used,
            excluded_feature_count=len(preprocessing_config.excluded_features),
            excluded_by_reason=excluded_by_reason,
            estimated_output_feature_count=estimated_output,
            duplicate_rows_detected=preprocessing_config.duplicate_handling.duplicate_rows_detected,
            duplicate_strategy=preprocessing_config.duplicate_handling.strategy,
            pipeline_ready=True,
            summary_text=summary_text,
        )

    def _log_decisions(
        self,
        state: AgentState,
        preprocessing_config: PreprocessingConfig,
        estimated_output: int,
    ) -> None:
        """Log every decision so the workflow stays explainable."""
        self.log_message(
            state,
            f"Preprocessing status: {PreprocessingStatus.COMPLETED.value}.",
        )
        self.log_message(
            state,
            "Numerical features ({}): {}".format(
                len(preprocessing_config.numerical_features),
                ", ".join(preprocessing_config.numerical_features) or "none",
            ),
        )
        self.log_message(
            state,
            "Categorical features ({}): {}".format(
                len(preprocessing_config.categorical_features),
                ", ".join(preprocessing_config.categorical_features) or "none",
            ),
        )

        for reason in ExclusionReason:
            excluded = [
                item for item in preprocessing_config.excluded_features if item.reason == reason.value
            ]
            if not excluded:
                continue
            columns = ", ".join(item.column for item in excluded)
            level = "warning" if reason == ExclusionReason.HIGH_CARDINALITY else "info"
            self.log_message(
                state,
                f"Excluded {len(excluded)} column(s) as {reason.value}: {columns}. "
                f"Reason: {excluded[0].detail}.",
                level=level,
            )

        self.log_message(
            state,
            f"Numerical imputation: SimpleImputer(strategy='{preprocessing_config.numerical_imputation}') "
            "(median is robust against outliers).",
        )
        self.log_message(
            state,
            f"Categorical imputation: SimpleImputer(strategy='{preprocessing_config.categorical_imputation}'), "
            f"encoding: OneHotEncoder(handle_unknown='{preprocessing_config.encoding_handle_unknown}') "
            "so unseen categories in test data cannot break the pipeline.",
        )
        self.log_message(
            state,
            "Scaling: {}.".format(
                "StandardScaler applied to numerical features"
                if preprocessing_config.scale_numeric
                and preprocessing_config.scaling_strategy == "standard"
                else "disabled (scale_numeric=False)"
            ),
        )
        self.log_message(
            state,
            f"Duplicate handling: {preprocessing_config.duplicate_handling.notes}",
        )
        self.log_message(
            state,
            "Pipeline built but intentionally NOT fitted. It must be fitted on the training "
            "split only (fit_scope='training_data_only') to prevent data leakage.",
        )
        self.log_message(
            state,
            f"Estimated number of output features after transformation: {estimated_output}.",
        )