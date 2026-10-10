"""Evaluation Agent (Phase 5).

The agent interprets the baseline models that the Training Agent already
evaluated. It **never trains, predicts or refits anything**: its whole input is
the typed records stored in the shared state.

    state.experiment_results        -> hold-out metrics + confusion matrices
    state.cross_validation_results  -> cross-validation means and deviations
    state.model_comparison          -> the Phase 4 ranking rule and its notes
    state.best_baseline_model       -> the Phase 4 selection

What it produces is a single `EvaluationReport` that answers, with the evidence
attached:

1. which model performed best and why it was selected;
2. how the hold-out performance compares with cross-validation;
3. whether the recorded evidence shows instability, and in which direction;
4. how the models compare on accuracy, precision, recall and F1;
5. what the confusion matrix indicates about the errors;
6. how much time the experiments required;
7. how many experiments were performed;
8. which limitations apply to this run.

Research language is enforced on purpose. A hold-out vs cross-validation gap is
reported as a *measured difference* with a reporting category, never as proof of
overfitting or of robustness, and the report always separates what was observed
from what is interpreted and from what is merely hypothesised.
"""

from __future__ import annotations

import time
from typing import Dict, List

from agents.base_agent import BaseAgent
from tools.evaluation_tools import (
    EvaluationToolError,
    build_evaluation_report,
)
from tools.preprocessing_tools import PreprocessingError
from utils.logger import setup_logger
from utils.state import (
    AgentState,
    AgentStatus,
    EvaluationReport,
    EvaluationStatus,
    TrainingStatus,
)

logger = setup_logger("agent.EvaluationAgent")


class EvaluationAgent(BaseAgent):
    """Agent that evaluates and interprets the baseline models."""

    def __init__(self) -> None:
        """Initialize the Evaluation Agent."""
        super().__init__(name="EvaluationAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, state: AgentState) -> AgentState:
        """Evaluate the recorded baseline experiments and build the report.

        Args:
            state: Shared state produced by the Training Agent.

        Returns:
            The updated `AgentState`, with `evaluation_status` set and
            `evaluation_report` filled in.

        Raises:
            PreprocessingError: The Training Agent has not produced a usable
                result. The reason is logged, recorded in the state and
                re-raised.
            EvaluationToolError: The recorded results are malformed. The reason
                is logged, recorded in the state and re-raised.
        """
        started = time.perf_counter()

        try:
            self._verify_prerequisites(state)

            report = build_evaluation_report(
                experiments=state.experiment_results,
                cross_validation_results=state.cross_validation_results,
                model_comparison=state.model_comparison,
                dataset_name=state.dataset_name or state.dataset_path or "",
                problem_type=state.problem_type or "",
                target_column=state.target_column or "",
                test_samples=state.data_split.test_samples if state.data_split else 0,
                parameters_by_model=self._parameters_by_model(state),
            )
        except (PreprocessingError, EvaluationToolError) as error:
            state.evaluation_status = EvaluationStatus.FAILED.value
            self.log_message(state, f"Evaluation aborted: {error}", level="error")
            state.add_error(str(error))
            raise
        except Exception as error:  # pragma: no cover - defensive, never hidden
            state.evaluation_status = EvaluationStatus.FAILED.value
            self.log_message(
                state, f"Unexpected error during evaluation: {error!r}", level="error"
            )
            state.add_error(f"Unexpected error during evaluation: {error!r}")
            raise

        state.evaluation_report = report
        state.evaluation_status = report.overall_status

        # The workflow time stays cumulative: Phase 4 recorded the training wall
        # clock, and this agent only adds the time it needed to interpret it.
        state.execution_time = round(
            float(state.execution_time or 0.0) + (time.perf_counter() - started), 6
        )
        if report.best_model is None:
            self.log_message(
                state,
                "No model produced usable metrics, so the evaluation could not recommend "
                "a baseline model.",
                level="warning",
            )

        # The stage itself ran to the end; `evaluation_status` carries the detail.
        state.status = AgentStatus.EVALUATION_COMPLETED.value
        self._log_summary(state, report)
        return state

    # ------------------------------------------------------------------ #
    # Prerequisites
    # ------------------------------------------------------------------ #
    def _verify_prerequisites(self, state: AgentState) -> None:
        """Ensure the Training Agent produced something to evaluate.

        The agent does not repeat the training: if training did not complete
        there are no metrics to interpret, and silently reporting an empty
        evaluation would hide that. The reason is raised instead.
        """
        problems: List[str] = []

        if state.training_status != TrainingStatus.COMPLETED.value:
            problems.append(
                f"the Training Agent has not been completed (status: {state.training_status})"
            )
        if not state.experiment_results:
            problems.append("no experiment was recorded by the Training Agent")
        if not state.problem_type:
            problems.append("no problem type was resolved")

        if problems:
            raise PreprocessingError(
                "The Evaluation Agent cannot start because "
                + "; ".join(problems)
                + ". Run DataAnalysisAgent, PreprocessingAgent, ModelSelectionAgent and "
                "TrainingAgent first."
            )

    def _parameters_by_model(self, state: AgentState) -> Dict[str, Dict[str, object]]:
        """Collect the effective baseline parameters of every experiment."""
        return {
            str(experiment.model_name): dict(experiment.parameters or {})
            for experiment in state.experiment_results
        }

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log_summary(self, state: AgentState, report: EvaluationReport) -> None:
        """Log the evaluation process as single-line messages.

        The multi-line tables live in `EvaluationReport.table_text` and in the
        CLI report; repeating them in the log would interleave confusingly with
        the console output.
        """
        self.log_message(
            state,
            f"Evaluation status: {report.overall_status} "
            f"({len(report.model_evaluations)} experiment(s) evaluated; no model was "
            "retrained).",
        )

        if report.best_model is not None:
            best = report.best_model
            self.log_message(
                state,
                "Best baseline model: {} (hold-out F1 {}, accuracy {}, CV mean F1 {}).".format(
                    best.best_baseline_model,
                    _fmt(best.f1_score),
                    _fmt(best.accuracy),
                    _fmt(best.cv_f1_mean),
                ),
            )
            self.log_message(state, f"Selection reason: {best.selection_reason}")
            self.log_message(state, f"Performance summary: {best.performance_summary}")

        gap = report.gap_analysis
        if gap is not None:
            self.log_message(state, f"Hold-out vs cross-validation: {gap.summary}")
            self.log_message(state, f"Interpretation: {gap.interpretation}")
            self.log_message(
                state,
                f"Gap rule: {gap.rule}. {gap.disclaimer}",
            )
            for note in gap.notes:
                self.log_message(state, note, level="warning")

        for analysis in report.confusion_matrix_analysis:
            if analysis.available:
                self.log_message(
                    state,
                    f"Confusion matrix ({analysis.model_name}): "
                    f"{analysis.correct_predictions}/{analysis.total_samples} correct, "
                    f"{analysis.incorrect_predictions} incorrect.",
                )
                self.log_message(state, f"Error pattern: {analysis.error_pattern}")
            else:
                self.log_message(
                    state,
                    f"Confusion matrix ({analysis.model_name or 'unknown model'}) is not "
                    f"available. {analysis.per_class_metrics_note}",
                    level="warning",
                )

        timing = report.execution_time_analysis
        if timing is not None:
            self.log_message(
                state,
                f"Execution time: {timing.total_experiment_time:.4f} s total, "
                f"{timing.average_experiment_time:.4f} s average over "
                f"{timing.timed_experiments} experiment(s); fastest "
                f"{timing.fastest_model or '-'} ({timing.fastest_time:.4f} s), slowest "
                f"{timing.slowest_model or '-'} ({timing.slowest_time:.4f} s).",
            )

        counts = report.experiment_count
        if counts is not None:
            self.log_message(
                state,
                f"Experiments: {counts.baseline_models_attempted} baseline model(s) "
                f"attempted, {counts.successful_experiments} successful, "
                f"{counts.partial_experiments} partial, {counts.failed_experiments} "
                f"failed, {counts.hyperparameter_experiments} hyperparameter "
                "configuration(s) evaluated.",
            )

        for observation in report.observations:
            self.log_message(state, f"Observation: {observation}")
        for note in report.notes:
            self.log_message(state, note, level="warning")
        for limitation in report.limitations:
            self.log_message(state, f"Limitation: {limitation}", level="warning")
        for recommendation in report.recommendations:
            self.log_message(state, f"Recommendation: {recommendation}")

        logger.debug(
            "Evaluation finished with %d limitation(s) and %d recommendation(s)",
            len(report.limitations),
            len(report.recommendations),
        )


def _fmt(value) -> str:
    """Format a metric for a log line."""
    return "-" if value is None else f"{float(value):.4f}"


__all__ = ["EvaluationAgent"]
