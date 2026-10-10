"""Hyperparameter Optimization Agent (Phase 6).

The agent tunes the most promising baseline models that the Training and
Evaluation Agents already produced. It is the first agent that *searches*, and
that is exactly why it is deliberately narrow:

* only the top `top_models` baseline models of the Phase 5 ranking are tuned, so
  a student laptop never runs a search over all five models;
* every trial score comes from cross-validation **inside the pipeline**, so the
  hold-out split cannot influence the search;
* the Phase 5 baseline result is never overwritten. The report keeps
  `best_baseline_model` and `best_optimized_model` side by side, because the
  optimized cross-validation score was chosen by the search itself and the two
  are not the same kind of number.

Workflow of the agent:

    1. verify that analysis, preprocessing, model selection, training and
       evaluation all produced usable records
    2. reload the dataset and rebuild the *same* train/test split as Phase 4
    3. pick the candidate models from the Phase 5 ranking
    4. run one seeded Optuna study per candidate
    5. score each tuned pipeline once on the untouched hold-out split
    6. rank the tuned models with the project's documented rule
    7. build the structured report and update the shared state

Failures are isolated at two levels: a trial that raises is recorded and the
search continues, and a model whose search fails entirely is recorded so the next
candidate is still optimized. If every search fails, the phase reports `failed`
and the Phase 5 baseline result stays the usable one.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from tools.model_tools import ModelToolError
from tools.optimization_tools import (
    OptimizationToolError,
    build_optimization_report,
    evaluate_tuned_model,
    optimize_model,
    select_optimization_candidates,
)
from tools.preprocessing_tools import PreprocessingError, remove_duplicate_rows
from tools.training_tools import (
    build_cross_validation_config,
    describe_training_error,
    load_training_dataframe,
    prepare_features_and_target,
    split_dataset,
)
from utils import config as project_config
from utils.logger import setup_logger
from utils.state import (
    AgentState,
    AgentStatus,
    DuplicateStrategy,
    OptimizationReport,
    OptimizationResult,
    OptimizationStatus,
    TrainingStatus,
)

logger = setup_logger("agent.OptimizationAgent")


class OptimizationAgent(BaseAgent):
    """Agent that tunes the most promising baseline models with Optuna."""

    def __init__(self) -> None:
        """Initialize the Optimization Agent."""
        super().__init__(name="OptimizationAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        state: AgentState,
        *,
        top_models: int = project_config.OPTUNA_TOP_MODELS,
        n_trials: int = project_config.OPTUNA_N_TRIALS,
        seed: int = project_config.OPTUNA_RANDOM_SEED,
        timeout: Optional[int] = project_config.OPTUNA_TIMEOUT,
        cross_validation: bool = True,
    ) -> AgentState:
        """Optimize the hyperparameters of the most promising baseline models.

        Args:
            state: Shared state produced by the Evaluation Agent.
            top_models: How many baseline models to tune.
            n_trials: Trial budget per model.
            seed: Sampler and estimator seed. The same seed and data produce the
                same best configuration on every run.
            timeout: Optional wall-clock limit in seconds per model. None means no
                limit, which keeps the run reproducible.
            cross_validation: Set to False to skip the whole phase. The
                optimization cannot run without cross-validation, so this is the
                switch that documents that decision instead of failing.

        Returns:
            The updated `AgentState`, with `optimization_status` set and
            `optimization_report` filled in.

        Raises:
            PreprocessingError: A prerequisite phase did not complete. The reason
                is logged, recorded in the state and re-raised.
            OptimizationToolError: The optimization inputs are unusable. The
                reason is logged, recorded in the state and re-raised.
        """
        started = time.perf_counter()

        try:
            self._verify_settings(top_models=top_models, n_trials=n_trials, seed=seed)
            self._verify_prerequisites(state)

            if not cross_validation:
                return self._record_skip(
                    state,
                    "hyperparameter optimization was disabled by the caller "
                    "(--no-cross-validation), and tuning without cross-validation "
                    "would select parameters on the same rows it scores them on",
                )

            candidates = select_optimization_candidates(
                state.experiment_results, top_models=top_models, problem_type=state.problem_type
            )
            if not candidates:
                return self._record_skip(
                    state,
                    "no baseline model with usable metrics has a hyperparameter search "
                    "space, so there is nothing to optimize",
                )

            self.log_message(
                state,
                f"Optimizing {len(candidates)} baseline model(s) with "
                f"{project_config.OPTUNA_SAMPLER_NAME} (seed={seed}, "
                f"{n_trials} trial(s) each): {', '.join(candidates)}.",
            )

            results = self._optimize_selected_models(
                state,
                candidates,
                n_trials=n_trials,
                seed=seed,
                timeout=timeout,
                cross_validation=cross_validation,
            )
        except (PreprocessingError, ModelToolError, OptimizationToolError) as error:
            state.optimization_status = OptimizationStatus.FAILED.value
            self.log_message(state, f"Optimization aborted: {error}", level="error")
            state.add_error(str(error))
            raise
        except Exception as error:  # pragma: no cover - defensive, never hidden
            state.optimization_status = OptimizationStatus.FAILED.value
            self.log_message(
                state, f"Unexpected error during optimization: {error!r}", level="error"
            )
            state.add_error(f"Unexpected error during optimization: {error!r}")
            raise

        report = build_optimization_report(
            experiments=state.experiment_results,
            results=results,
            selected=candidates,
            n_trials=n_trials,
            top_models=top_models,
            dataset_name=state.dataset_name or state.dataset_path or "",
            problem_type=state.problem_type or "",
            target_column=state.target_column or "",
            test_samples=state.data_split.test_samples if state.data_split else 0,
            cv_n_splits=results[0].cv_n_splits if results else 0,
            cv_adjustment_reason=results[0].cv_adjustment_reason if results else "",
            baseline_recommendation=(
                state.evaluation_report.best_model if state.evaluation_report else None
            ),
        )

        state.optimization_report = report
        state.optimization_status = report.overall_status

        # The workflow time stays cumulative: every agent before this one added
        # its own wall clock, so this agent only adds what the search needed.
        state.execution_time = round(
            float(state.execution_time or 0.0) + (time.perf_counter() - started), 6
        )
        if report.best_optimized_model is None:
            self.log_message(
                state,
                "No model produced a usable tuned configuration, so the Phase 5 "
                "baseline recommendation remains the usable result of this run.",
                level="warning",
            )

        # The stage itself ran to the end; `optimization_status` carries the detail.
        state.status = AgentStatus.OPTIMIZATION_COMPLETED.value
        self._log_summary(state, report)
        return state

    # ------------------------------------------------------------------ #
    # Prerequisites
    # ------------------------------------------------------------------ #
    def _verify_settings(self, *, top_models: int, n_trials: int, seed: int) -> None:
        """Reject unusable arguments before any work starts.

        A bad argument is a caller mistake, not a search result. Validating here
        means an impossible trial budget is reported once, instead of producing one
        `failed` record per candidate that hides the single real reason.
        """
        if int(n_trials) < 1:
            raise OptimizationToolError(
                f"n_trials must be at least 1, but it is {n_trials!r}. Set "
                "utils.config.OPTUNA_N_TRIALS or pass n_trials."
            )
        if int(top_models) < 1:
            raise OptimizationToolError(
                f"top_models must be at least 1, but it is {top_models!r}. Set "
                "utils.config.OPTUNA_TOP_MODELS or pass top_models."
            )
        if seed is None:
            raise OptimizationToolError(
                "A sampler seed is required for a reproducible search, but it is None."
            )

    def _verify_prerequisites(self, state: AgentState) -> None:
        """Ensure every earlier phase left something usable to optimize.

        The agent does not repeat any earlier work. Without trained baseline
        experiments there is nothing to improve on, and quietly optimizing an
        untrained model would hide that, so the reason is raised instead.
        """
        problems: List[str] = []

        if state.training_status != TrainingStatus.COMPLETED.value:
            problems.append(
                f"the Training Agent has not been completed (status: {state.training_status})"
            )
        if not state.experiment_results:
            problems.append("no experiment was recorded by the Training Agent")
        if not state.successful_experiments:
            problems.append("no baseline experiment produced usable metrics")
        if state.preprocessing_status != "completed" or state.preprocessing_config is None:
            problems.append(
                f"the Preprocessing Agent has not been completed "
                f"(status: {state.preprocessing_status})"
            )
        if not state.dataset_path:
            problems.append("no dataset path is available to rebuild the training split")
        if not state.target_column:
            problems.append("no target column was resolved")
        if not state.problem_type:
            problems.append("no problem type was resolved")

        if problems:
            raise PreprocessingError(
                "The Optimization Agent cannot start because "
                + "; ".join(problems)
                + ". Run DataAnalysisAgent, PreprocessingAgent, ModelSelectionAgent, "
                "TrainingAgent and EvaluationAgent first."
            )

    def _record_skip(self, state: AgentState, reason: str) -> AgentState:
        """Record a skipped phase without running a single trial."""
        state.optimization_status = OptimizationStatus.SKIPPED.value
        self.log_message(state, f"Optimization skipped. {reason}", level="warning")
        return state

    # ------------------------------------------------------------------ #
    # Optimization
    # ------------------------------------------------------------------ #
    def _optimize_selected_models(
        self,
        state: AgentState,
        candidates: List[str],
        *,
        n_trials: int,
        seed: int,
        timeout: Optional[int],
        cross_validation: bool,
    ) -> List[OptimizationResult]:
        """Rebuild the training split, then optimize each candidate in turn.

        The split is rebuilt with the Phase 4 conventions instead of being read
        back from the state, because the state stores only the split *summary*,
        not the rows. Using the same `test_size` and `random_state` reproduces
        exactly the same hold-out rows, so the tuned model is compared with its
        baseline on identical data.

        Returns:
            One `OptimizationResult` per candidate, in candidate order.
        """
        prepared = self._prepare_training_data(state, cross_validation=cross_validation)
        features_train, target_train, features_test, target_test, cv_config = prepared

        by_name: Dict[str, Any] = {
            str(experiment.model_name): experiment
            for experiment in state.experiment_results
        }

        results: List[OptimizationResult] = []
        for name in candidates:
            self.log_message(
                state,
                f"Starting the {name} search with {n_trials} trial(s) over "
                f"{cv_config.n_splits} stratified fold(s), scored on the "
                f"{project_config.OPTUNA_OBJECTIVE_METRIC} of the training split.",
            )
            try:
                result = optimize_model(
                    model_name=name,
                    preprocessing_config=state.preprocessing_config,
                    features=features_train,
                    target=target_train,
                    cv_config=cv_config,
                    problem_type=state.problem_type,
                    n_trials=n_trials,
                    seed=seed,
                    timeout=timeout,
                    baseline=by_name.get(name),
                )
            except Exception as error:
                # One model must not take the phase down with it. The failure is
                # recorded as a result of its own, so the next candidate still runs
                # and the report can name what went wrong.
                message = describe_training_error(error)
                logger.warning("Optimization of %s failed: %s", name, message)
                result = OptimizationResult(
                    model_name=name,
                    status=OptimizationStatus.FAILED.value,
                    reason=message,
                    error_message=message,
                )
                self.log_message(
                    state, f"Optimization of {name} failed: {message}", level="error"
                )
                results.append(result)
                continue

            self._report_trial_failures(state, result)

            if result.status == OptimizationStatus.COMPLETED.value:
                # Only now, with the parameters fixed, is the hold-out split read.
                evaluate_tuned_model(
                    result=result,
                    preprocessing_config=state.preprocessing_config,
                    features_train=features_train,
                    target_train=target_train,
                    features_test=features_test,
                    target_test=target_test,
                )
                self.log_message(
                    state,
                    f"{name}: best cross-validation {project_config.OPTUNA_OBJECTIVE_METRIC} "
                    f"{_fmt(result.optimized_cv_f1_mean)} in trial "
                    f"{result.best_trial_number} out of {result.total_trials} "
                    f"({result.completed_trials} completed, {result.failed_trials} failed); "
                    f"baseline {_fmt(result.baseline_cv_f1_mean)}, improvement "
                    f"{_fmt(result.improvement)} ({result.improvement_direction}).",
                )
            results.append(result)
        return results

    def _report_trial_failures(self, state: AgentState, result: OptimizationResult) -> None:
        """Log every failed trial as one line, so nothing is silently dropped."""
        failed = [item for item in result.trials if item.error_message]
        for item in failed:
            self.log_message(
                state,
                f"Trial {item.trial_number} of the {result.model_name} search failed and "
                f"was excluded from the comparison: {item.error_message}",
                level="warning",
            )
        if result.failed_trials and not failed:
            self.log_message(
                state,
                f"{result.failed_trials} trial(s) of the {result.model_name} search "
                "failed without a recorded message.",
                level="warning",
            )

    def _prepare_training_data(
        self,
        state: AgentState,
        *,
        cross_validation: bool,
    ) -> tuple:
        """Reload the data and rebuild the Phase 4 split and fold configuration.

        A disabled cross-validation is **not** silently worked around: each
        selected model then receives a recorded `failed` result with the reason,
        which `build_objective` raises for. Optimizing without cross-validation
        would mean choosing parameters on the very rows that score them.

        Returns:
            A `(X_train, y_train, X_test, y_test, cv_config)` tuple.
        """
        dataframe = load_training_dataframe(state.dataset_path)
        handling = state.preprocessing_config.duplicate_handling
        if handling.strategy != DuplicateStrategy.KEEP_ALL.value:
            # Same in-memory duplicate handling as Phase 4, so the split matches
            # the one the baseline experiments were measured on. The CSV is never
            # touched.
            dataframe, removed = remove_duplicate_rows(
                dataframe, strategy=handling.strategy
            )
            logger.debug("Removed %s duplicate row(s) before the split", removed)

        features, target = prepare_features_and_target(
            dataframe, state.target_column, state.excluded_feature_names
        )
        split = split_dataset(
            features,
            target,
            test_size=state.data_split.test_size if state.data_split else project_config.TEST_SIZE,
            random_state=(
                state.data_split.random_state
                if state.data_split
                else project_config.RANDOM_STATE
            ),
        )
        if not split.stratified:
            self.log_message(
                state,
                "The rebuilt train/test split was not stratified: " + split.reason,
                level="warning",
            )

        cv_config = build_cross_validation_config(
            split.y_train,
            enabled=cross_validation,
            n_splits=project_config.CV_FOLDS,
            random_state=project_config.RANDOM_STATE,
        )
        if not cv_config.enabled:
            self.log_message(
                state,
                f"Optimization cannot run: {cv_config.adjusted_reason}. Every selected "
                "model is recorded as failed so the reason is visible in the report.",
                level="warning",
            )
        elif cv_config.adjusted_reason:
            self.log_message(
                state,
                f"The objective uses fewer folds than requested: {cv_config.adjusted_reason}.",
                level="warning",
            )
        return split.X_train, split.y_train, split.X_test, split.y_test, cv_config

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log_summary(self, state: AgentState, report: OptimizationReport) -> None:
        """Log the optimization phase as single-line messages.

        The multi-line comparison table lives in `OptimizationReport.table_text`
        and in the CLI report; repeating it in the log would interleave
        confusingly with the console output.
        """
        self.log_message(
            state,
            f"Optimization status: {report.overall_status} "
            f"({len(report.optimization_results)} model(s) searched with "
            f"{report.n_trials_per_model} trial(s) each; the search used the training "
            "split only).",
        )

        best = report.best_optimized_model
        if best is not None:
            self.log_message(
                state,
                "Best optimized model: {} (hold-out F1 {}, cross-validation F1 {} "
                "against a baseline {}, improvement {}).".format(
                    best.best_optimized_model,
                    _fmt(best.holdout_f1_score),
                    _fmt(best.optimized_cv_f1_mean),
                    _fmt(best.baseline_cv_f1_mean),
                    _fmt(best.improvement),
                ),
            )
            self.log_message(state, f"Selection reason: {best.selection_reason}")
            self.log_message(state, f"Best parameters: {best.best_params}")
            self.log_message(state, f"Performance summary: {best.performance_summary}")
        else:
            self.log_message(
                state,
                "No optimized model can be recommended; the baseline result is "
                "unchanged.",
                level="warning",
            )

        if report.best_baseline_model is not None:
            self.log_message(
                state,
                "Best baseline model (unchanged by this phase): {} (hold-out F1 {}). "
                "It stays available at {}.".format(
                    report.best_baseline_model.best_baseline_model,
                    _fmt(report.best_baseline_model.f1_score),
                    report.baseline_saved_to,
                ),
            )

        counts = report.counts
        if counts is not None:
            self.log_message(
                state,
                f"Counts: {counts.baseline_models_attempted} baseline model(s) "
                f"attempted with {counts.baseline_successful_experiments} usable "
                f"metric set(s); {counts.optimized_models_attempted} model(s) tuned "
                f"with {counts.completed_trials} completed, {counts.failed_trials} "
                f"failed and {counts.pruned_trials} pruned trial(s) "
                f"({counts.total_trials} trial(s) in total).",
            )
            self.log_message(state, counts.note)

        for observation in report.observations:
            self.log_message(state, f"Observation: {observation}")
        for note in report.notes:
            self.log_message(state, note, level="warning")
        for limitation in report.limitations:
            self.log_message(state, f"Limitation: {limitation}", level="warning")
        for recommendation in report.recommendations:
            self.log_message(state, f"Recommendation: {recommendation}")

        logger.debug(
            "Optimization finished with %d limitation(s) and %d recommendation(s)",
            len(report.limitations),
            len(report.recommendations),
        )


def _fmt(value: Optional[float]) -> str:
    """Format a metric for a log line."""
    return "-" if value is None else f"{float(value):.4f}"


__all__ = ["OptimizationAgent"]