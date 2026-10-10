"""Model Selection Agent (Phase 4).

The agent decides **which** classification models will be trained. It reads the
shared state, checks that the earlier phases produced what is needed, and records
the candidate models together with the reason each one was selected.

Decision rules (deterministic, no LLM involved):

* every model of the controlled registry in `tools/model_tools.py` whose
  `supported_problem_types` contains the detected problem type is selected;
* each model is paired with its scaling recommendation, because tree-based
  models are scale invariant while Logistic Regression, KNN and SVM are not;
* when the problem type is not supported in Phase 4 (regression, or an
  unresolved target) the agent records `skipped` with a reason instead of
  guessing.

No estimator is instantiated here and nothing is trained: the agent only
produces the candidate list consumed by the Training Agent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from tools.model_tools import (
    ModelSelectionError,
    ModelToolError,
    get_excluded_model_reasons,
    get_supported_models,
    select_candidate_models,
)
from tools.preprocessing_tools import PreprocessingError
from utils import config as project_config
from utils.state import (
    AgentState,
    AgentStatus,
    ModelSelectionSummary,
    PreprocessingStatus,
    ProblemType,
    TrainingStatus,
)

# Short note appended to the summary so the ranking rule is always visible.
RANKING_NOTE = (
    f"the best model will be ranked by {project_config.BEST_MODEL_PRIMARY_METRIC} first, "
    f"then {project_config.BEST_MODEL_SECONDARY_METRIC}."
)


class ModelSelectionAgent(BaseAgent):
    """Agent that selects the candidate models for the detected problem type."""

    def __init__(self) -> None:
        """Initialize the Model Selection Agent."""
        super().__init__(name="ModelSelectionAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        state: AgentState,
        *,
        model_names: Optional[Sequence[str]] = None,
        exclude_models: Optional[Sequence[str]] = None,
    ) -> AgentState:
        """Select the candidate models and record them in the state.

        Args:
            state: Shared state produced by the Preprocessing Agent.
            model_names: Optional subset of the registry to consider.
            exclude_models: Optional model names to leave out.

        Returns:
            The updated `AgentState`.

        Raises:
            PreprocessingError: A prerequisite phase did not complete. The
                reason is logged, recorded in the state and re-raised.
            ModelSelectionError: No candidate model could be selected.
        """
        try:
            self._verify_prerequisites(state)

            skip_reason = self._skip_reason(state)
            if skip_reason is not None:
                return self._record_skip(state, skip_reason)

            selected = select_candidate_models(
                state.problem_type,
                model_names,
                exclude=exclude_models,
            )
            excluded = get_excluded_model_reasons(state.problem_type, model_names)
            for name in exclude_models or []:
                excluded[name] = "excluded by the caller"
            summary = self._build_summary(state, selected, excluded)

            state.selected_models = selected
            state.model_selection_summary = summary
            state.model_selection_status = TrainingStatus.COMPLETED.value
            self._log_decisions(state, selected, excluded, summary)
        except (PreprocessingError, ModelToolError) as error:
            state.model_selection_status = TrainingStatus.FAILED.value
            self.log_message(state, f"Model selection failed: {error}", level="error")
            state.add_error(str(error))
            raise
        except Exception as error:  # pragma: no cover - defensive, never hidden
            state.model_selection_status = TrainingStatus.FAILED.value
            self.log_message(
                state, f"Unexpected error during model selection: {error!r}", level="error"
            )
            state.add_error(f"Unexpected error during model selection: {error!r}")
            raise

        state.status = AgentStatus.MODELS_SELECTED.value
        return state

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _verify_prerequisites(self, state: AgentState) -> None:
        """Ensure the analysis and preprocessing phases produced what is needed.

        Raises:
            PreprocessingError: Any prerequisite is missing.
        """
        problems: List[str] = []

        if state.dataset_shape is None or not state.column_names:
            problems.append("the Data Analysis Agent has not been completed")
        if state.preprocessing_status != PreprocessingStatus.COMPLETED.value:
            problems.append(
                f"the Preprocessing Agent has not been completed "
                f"(status: {state.preprocessing_status})"
            )
        if state.preprocessing_config is None:
            problems.append("the shared state does not contain a preprocessing configuration")
        if not state.target_column:
            problems.append("no target column was resolved")
        if not state.problem_type:
            problems.append("no problem type was detected")

        if problems:
            raise PreprocessingError(
                "The Model Selection Agent cannot start because "
                + "; ".join(problems)
                + ". Run DataAnalysisAgent and PreprocessingAgent first."
            )

    def _skip_reason(self, state: AgentState) -> Optional[str]:
        """Return why model selection cannot run, or None when it can."""
        if state.problem_type not in (
            ProblemType.BINARY_CLASSIFICATION.value,
            ProblemType.MULTICLASS_CLASSIFICATION.value,
        ):
            return (
                f"problem type '{state.problem_type}' is not supported for model selection "
                "in Phase 4 (only binary and multiclass classification)"
            )
        if not state.selected_models and not state.preprocessing_config.numerical_features \
                and not state.preprocessing_config.categorical_features:
            return "the preprocessing configuration does not contain any usable feature"
        return None

    def _record_skip(self, state: AgentState, reason: str) -> AgentState:
        """Record a skipped run without selecting any model."""
        state.model_selection_status = TrainingStatus.SKIPPED.value
        state.selected_models = []
        state.model_selection_summary = ModelSelectionSummary(
            status=TrainingStatus.SKIPPED.value,
            status_reason=reason,
            problem_type=state.problem_type,
            target_column=state.target_column,
            summary_text=f"Model selection skipped: {reason}",
        )
        self.log_message(state, f"Model selection skipped. {reason}", level="warning")
        return state

    def _build_summary(
        self,
        state: AgentState,
        selected: List,
        excluded: Dict[str, str],
    ) -> ModelSelectionSummary:
        """Create the display-ready selection summary."""
        scaled = [item.name for item in selected if item.scaling_required]
        unscaled = [item.name for item in selected if not item.scaling_required]

        summary_text = (
            f"Selected {len(selected)} baseline candidate model(s) for "
            f"{state.problem_type}: {', '.join(item.name for item in selected)}. "
            f"Ranking rule: {RANKING_NOTE} "
            f"Parameters are baseline defaults (max_iter=1000 for Logistic Regression, "
            f"n_estimators=100 for Random Forest, n_neighbors={project_config.KNN_PARAMS['n_neighbors']} for KNN, "
            f"probability=False for SVM); hyperparameter optimization is Phase 6. "
            f"Models that benefit from scaling: {', '.join(scaled) or 'none'}. "
            f"Models that do not need scaling: {', '.join(unscaled) or 'none'}."
        )

        return ModelSelectionSummary(
            status=TrainingStatus.COMPLETED.value,
            problem_type=state.problem_type,
            target_column=state.target_column,
            selected_count=len(selected),
            selected_models=[item.name for item in selected],
            scaling_required_models=scaled,
            scaling_optional_models=unscaled,
            excluded_models=dict(excluded),
            summary_text=summary_text,
        )

    def _log_decisions(
        self,
        state: AgentState,
        selected: List,
        excluded: Dict[str, str],
        summary: ModelSelectionSummary,
    ) -> None:
        """Log the selection decisions so the workflow stays explainable."""
        self.log_message(
            state,
            f"Model selection completed: {summary.selected_count} candidate(s) for "
            f"{summary.problem_type}.",
        )
        for item in selected:
            self.log_message(
                state,
                f"Selected {item.name} ({item.model_type}): {item.selection_reason} "
                f"Baseline parameters: {item.default_parameters}",
            )
        if excluded:
            for name, reason in excluded.items():
                self.log_message(
                    state,
                    f"Not selected {name}: {reason}",
                    level="warning",
                )
        self.log_message(
            state,
            "Scaling plan: {} runs with scaling enabled, {} without.".format(
                ", ".join(summary.scaling_required_models) or "no model",
                ", ".join(summary.scaling_optional_models) or "no model",
            ),
        )
        self.log_message(
            state,
            "No model was instantiated or trained at this stage: the Training Agent "
            "builds one Pipeline per candidate.",
        )
        self.log_message(state, f"Ranking rule for Phase 4: {RANKING_NOTE}")


__all__ = ["ModelSelectionAgent"]