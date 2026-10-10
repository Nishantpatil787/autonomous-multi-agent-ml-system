"""Model Persistence Agent (Phase 9).

The last stage of the workflow. It writes three files and then stops:

* ``models/best_model.joblib`` - the fitted ``preprocessor -> estimator`` pipeline;
* ``models/best_model_metadata.json`` - what the pipeline is and how it scored;
* ``models/run_results.json`` - the dashboard read model for the report sections.

What this agent deliberately does **not** do:

* it does not search. No Optuna study is created or resumed;
* it does not rescore. Every metric in the metadata was produced by Phase 4, 5 or
  6 and is copied unchanged;
* it does not reselect. The choice between the baseline winner and the tuned
  winner uses the ranking rule those phases already use - hold-out ``f1_score``
  then hold-out ``accuracy`` then the model name. A new threshold here would make
  the dashboard's "best model" disagree with the CLI report's "best model";
* it does not re-read the hold-out rows for anything except the numbers Phase 5
  already measured. The final fit happens **after** the selection and only to
  produce the artifact.

The agent is also the only stage whose failure does not invalidate the machine
learning result. A run that could not write a file still produced a valid model,
so the agent records the failure, sets ``persistence_status`` and re-raises for the
caller. The graph maps that re-raise to a ``partial`` workflow, not a ``failed``
one, because the evaluation result it would replace is still correct.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from tools.persistence_tools import (
    ArtifactPaths,
    NoModelToPersistError,
    PersistenceError,
    artifact_paths,
    build_final_pipeline,
    select_final_model,
    write_artifacts,
)
from tools.preprocessing_tools import PreprocessingError
from tools.training_tools import TrainingToolError
from utils import config as project_config
from utils.state import (
    AgentState,
    EvaluationStatus,
    FinalModelSelection,
    FinalModelSource,
    PersistenceReport,
    PersistenceStatus,
    TrainingStatus,
)

#: The agent name recorded in ``state.agent_messages``.
AGENT_NAME = "PersistenceAgent"


class PersistenceAgent(BaseAgent):
    """Agent that persists the final model and its metadata."""

    def __init__(self) -> None:
        """Initialize the Model Persistence Agent."""
        super().__init__(name=AGENT_NAME)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        state: AgentState,
        *,
        output_dir: Optional[Path] = None,
        llm_results: Optional[Sequence[Any]] = None,
        enabled: bool = project_config.PERSISTENCE_ENABLED,
        overwrite: bool = True,
    ) -> AgentState:
        """Select, fit and write the final model.

        The three steps are separate and ordered: selection first, final fit
        second, write third. A failure in step three must never change what step
        one decided, which is why the selection is recorded on the state before
        anything is written.

        Args:
            state: Shared state produced by the Evaluation (or Optimization) Agent.
            output_dir: Directory to write into. None uses
                ``config.PERSISTENCE_OUTPUT_DIR`` or ``config.MODELS_DIR``.
            llm_results: Optional Phase 8 interpretation records. They live on the
                graph state rather than on ``AgentState``, so they are passed in to
                be copied into the dashboard read model.
            enabled: Set to False to skip the stage entirely (``--no-save-model``).
            overwrite: Set to False to refuse replacing an existing artifact instead
                of overwriting it.

        Returns:
            The updated `AgentState`, with `persistence_status` set and
            `persistence_report` filled in.

        Raises:
            NoModelToPersistError: The state carries no model that could honestly
                be written. The reason is logged, recorded and re-raised.
            PersistenceError: The fit or the write failed. The reason is logged,
                recorded and re-raised.
        """
        started = time.perf_counter()

        if not enabled:
            return self._record_skip(
                state,
                "model persistence was disabled by the caller (--no-save-model)",
            )

        try:
            self._verify_prerequisites(state)

            selection = select_final_model(state)
            self.log_message(state, f"Final model selected: {selection.selection_reason}")

            final = build_final_pipeline(state, selection)
            self.log_message(
                state,
                f"Final fit for '{selection.model_name}' used the "
                f"'{final.strategy}' strategy.",
            )

            self._verify_paths(output_dir, overwrite=overwrite)
            paths = write_artifacts(
                state,
                selection,
                final,
                output_dir=output_dir,
                llm_results=llm_results,
            )
        except PersistenceError as error:
            # Covers NoModelToPersistError, ArtifactWriteError and ArtifactReadError.
            self._record_failure(state, str(error))
            raise
        except (TrainingToolError, PreprocessingError) as error:
            # The split could not be rebuilt, so nothing was written. This is a
            # reported failure, not an "unexpected" one.
            message = f"The final fit could not be prepared: {error}"
            self._record_failure(state, message)
            raise PersistenceError(message) from error
        except Exception as error:  # pragma: no cover - defensive, never hidden
            message = f"Unexpected error during model persistence: {error!r}"
            self._record_failure(state, message)
            raise PersistenceError(message) from error

        report = PersistenceReport(
            overall_status=PersistenceStatus.COMPLETED.value,
            dataset_name=str(state.dataset_name or ""),
            target_column=str(state.target_column or ""),
            task_type=str(state.problem_type or ""),
            selection=selection,
            model_artifact=paths.model.name,
            metadata_artifact=paths.metadata.name,
            results_artifact=paths.results.name,
            final_fit_strategy=final.strategy,
            final_fit_note=final.note,
            selected_f1_score=(
                selection.optimized_f1_score
                if selection.selection_source == FinalModelSource.OPTIMIZED.value
                else selection.baseline_f1_score
            ),
            selected_accuracy=(
                selection.optimized_accuracy
                if selection.selection_source == FinalModelSource.OPTIMIZED.value
                else selection.baseline_accuracy
            ),
            feature_count=len(list(state.preprocessing_numerical_features or []))
            + len(list(state.preprocessing_categorical_features or [])),
            class_count=len(state.class_distribution or {}),
            schema_version=int(project_config.PERSISTENCE_SCHEMA_VERSION),
            observations=self._observations(selection),
            limitations=[
                project_config.PERSISTENCE_LIMITATION_SINGLE_SPLIT,
                project_config.PERSISTENCE_LIMITATION_NO_UPLOAD,
                project_config.PERSISTENCE_LIMITATION_NO_RETRAIN,
            ],
            notes=[
                project_config.PERSISTENCE_NOTE_NO_METRICS,
                selection.selection_rule,
            ],
        )
        state.persistence_report = report
        state.persistence_status = PersistenceStatus.COMPLETED.value

        state.execution_time = round(
            float(state.execution_time or 0.0) + (time.perf_counter() - started), 6
        )
        self._log_summary(state, report, paths)
        return state

    # ------------------------------------------------------------------ #
    # Prerequisites and output location
    # ------------------------------------------------------------------ #
    def _verify_prerequisites(self, state: AgentState) -> None:
        """Ensure the machine learning result is complete enough to write.

        Only two things are required: that the Evaluation Agent finished, and that
        a winner was recorded. Whether optimization ran, and which model won, are
        not preconditions - they belong to the selection policy.

        Raises:
            NoModelToPersistError: A prerequisite is missing. The reason is listed
                in full so the caller does not have to guess which stage to fix.
        """
        problems: List[str] = []

        if state.training_status != TrainingStatus.COMPLETED.value:
            problems.append(
                f"the Training Agent has not been completed "
                f"(status: {state.training_status})"
            )
        if state.evaluation_status != EvaluationStatus.COMPLETED.value:
            problems.append(
                f"the Evaluation Agent has not been completed "
                f"(status: {state.evaluation_status})"
            )
        if state.best_baseline_model is None:
            problems.append("no best baseline model was recorded")
        if not state.preprocessing_config:
            problems.append("no preprocessing configuration is available")
        if not state.dataset_path:
            problems.append("no dataset path is available to rebuild the training split")

        if problems:
            raise NoModelToPersistError(
                "The Model Persistence Agent cannot start because "
                + "; ".join(problems)
                + ". Run DataAnalysisAgent, PreprocessingAgent, ModelSelectionAgent, "
                "TrainingAgent and EvaluationAgent first."
            )

    def _verify_paths(self, output_dir: Optional[Path], *, overwrite: bool) -> ArtifactPaths:
        """Resolve the artifact paths and honour the no-overwrite switch.

        Args:
            output_dir: Directory requested by the caller, or None for the
                configured one.
            overwrite: When False, an existing artifact raises instead of being
                replaced.

        Returns:
            The resolved :class:`~tools.persistence_tools.ArtifactPaths`.

        Raises:
            PersistenceError: An artifact already exists and overwrite is False.
                Overwriting happens only when a new workflow run asks for it.
        """
        paths = artifact_paths(output_dir)
        if overwrite:
            return paths

        existing = [
            item for item in (paths.model, paths.metadata, paths.results) if item.exists()
        ]
        if existing:
            raise PersistenceError(
                "Refusing to overwrite the existing artifact(s) "
                + ", ".join(item.name for item in existing)
                + ". Pass overwrite=True (the default) to replace them."
            )
        return paths

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def _record_skip(self, state: AgentState, reason: str) -> AgentState:
        """Record a skipped stage without touching any file.

        Args:
            state: Shared state to update.
            reason: Why the stage was skipped.

        Returns:
            The updated `AgentState`.
        """
        state.persistence_status = PersistenceStatus.SKIPPED.value
        self.log_message(state, f"Model persistence skipped. {reason}", level="warning")
        return state

    def _record_failure(self, state: AgentState, reason: str) -> None:
        """Record a failure on the state before re-raising it.

        The status and the message are written to the state first, so the graph can
        report *why* the artifact is missing even though the exception propagated
        to the caller.

        Args:
            state: Shared state to update.
            reason: The full error message.
        """
        state.persistence_status = PersistenceStatus.FAILED.value
        self.log_message(state, f"Model persistence failed: {reason}", level="error")
        state.add_error(reason)

    def _observations(self, selection: FinalModelSelection) -> List[str]:
        """Build the observation lines stored in the report.

        Args:
            selection: The recorded final model choice.

        Returns:
            A list of one-line observations, including both candidates so the
            decision can be checked without rerunning anything.
        """
        lines = [
            f"Final model: {selection.model_name} "
            f"(source: {selection.selection_source}).",
            f"Baseline candidate: {selection.baseline_model_name or '-'} "
            f"(hold-out f1_score {_fmt(selection.baseline_f1_score)}).",
            f"Optimized candidate: {selection.optimized_model_name or '-'} "
            f"(hold-out f1_score {_fmt(selection.optimized_f1_score)}).",
        ]
        if selection.best_params:
            lines.append(f"Persisted parameters: {_format_params(selection.best_params)}.")
        lines.append(
            "No metric was recomputed by persistence; every score above was measured "
            "before this stage ran."
        )
        return lines

    def _log_summary(
        self,
        state: AgentState,
        report: PersistenceReport,
        paths: ArtifactPaths,
    ) -> None:
        """Log the stage as single-line messages.

        Args:
            state: Shared state, used for message recording.
            report: The report that was just built.
            paths: The artifact paths that were written.
        """
        selection = report.selection
        self.log_message(
            state,
            f"Model persistence status: {report.overall_status} "
            f"({selection.model_name}, source: {selection.selection_source}, "
            f"{report.feature_count} feature(s), {report.class_count} class(es)).",
        )
        self.log_message(
            state,
            f"{project_config.PERSISTENCE_LOG_PREFIX} artifacts written to "
            f"{paths.directory}: {paths.model.name}, {paths.metadata.name}, "
            f"{paths.results.name}.",
        )


# --------------------------------------------------------------------------- #
# Small local helpers
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float]) -> str:
    """Format a metric for a sentence, showing '-' when it is missing."""
    return "-" if value is None else f"{float(value):.4f}"


def _format_params(parameters: Dict[str, Any]) -> str:
    """Render a parameter dict compactly, with keys sorted for determinism.

    Args:
        parameters: The effective estimator parameters.

    Returns:
        A ``"key=value, key=value"`` string.
    """
    return ", ".join(f"{key}={parameters[key]!r}" for key in sorted(parameters))


__all__ = ["AGENT_NAME", "PersistenceAgent"]