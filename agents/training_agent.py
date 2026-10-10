"""Training Agent (Phase 4).

The agent executes one baseline experiment per selected model and records a
typed, JSON-safe result for each of them.

Pipeline of a single experiment:

    load CSV -> apply the Phase 3 duplicate strategy (in memory only)
             -> separate X and y (the target can never enter X)
             -> stratified train/test split
             -> Pipeline(preprocessor, estimator).fit(X_train, y_train)
             -> predict on X_test -> metrics + confusion matrix
             -> cross_validate(pipeline, X_train, y_train)   (leakage safe)
             -> record the experiment

Leakage prevention is structural rather than conventional: the preprocessing and
the estimator are always wrapped in one sklearn `Pipeline`, so the imputers,
encoders and scalers are fitted **inside** `fit` and therefore only ever see the
training data. Cross-validation uses the same unfitted pipeline, so every fold
refits the preprocessing on that fold's training rows.

A failure in one model never stops the workflow: the error is recorded on the
experiment, the run continues with the other models, and the failures are
reported in the comparison. Only a broken prerequisite (missing analysis, missing
preprocessing, missing target, unreadable dataset) fails the whole agent.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from agents.base_agent import BaseAgent
from tools.model_tools import ModelToolError, get_effective_parameters, get_model_definition
from tools.preprocessing_tools import (
    PreprocessingError,
    remove_duplicate_rows,
)
from tools.training_tools import (
    SplitResult,
    TrainingToolError,
    build_cross_validation_config,
    build_experiment_id,
    build_model_pipeline,
    calculate_confusion_matrix,
    count_transformed_features,
    describe_training_error,
    evaluate_classifier,
    load_training_dataframe,
    prepare_features_and_target,
    rank_experiments,
    run_cross_validation,
    select_best_experiment,
    split_dataset,
)
from utils import config as project_config
from utils.logger import setup_logger
from utils.state import (
    AgentState,
    AgentStatus,
    CrossValidationResult,
    DataSplitInfo,
    DuplicateStrategy,
    ExperimentResult,
    ExperimentStatus,
    ModelComparison,
    PreprocessingStatus,
    TrainingStatus,
)

logger = setup_logger("agent.TrainingAgent")


class TrainingAgent(BaseAgent):
    """Agent that trains, evaluates and compares the candidate models."""

    def __init__(self) -> None:
        """Initialize the Training Agent."""
        super().__init__(name="TrainingAgent")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        state: AgentState,
        *,
        test_size: float = project_config.TEST_SIZE,
        random_state: int = project_config.RANDOM_STATE,
        model_parameters: Optional[Dict[str, Dict[str, object]]] = None,
        cross_validation: bool = True,
        cross_validation_folds: int = project_config.CV_FOLDS,
    ) -> AgentState:
        """Train and evaluate every selected model.

        Args:
            state: Shared state produced by the Model Selection Agent.
            test_size: Share of rows held out for testing.
            random_state: Seed that makes the split reproducible.
            model_parameters: Optional per-model baseline parameter overrides,
                e.g. ``{"Random Forest": {"n_estimators": 50}}``.
            cross_validation: Set to False to skip cross-validation.
            cross_validation_folds: Preferred number of folds; reduced
                automatically when the smallest class is too small.

        Returns:
            The updated `AgentState`.

        Raises:
            PreprocessingError: A prerequisite phase did not complete.
            TrainingToolError: The training data could not be prepared.
        """
        started = time.perf_counter()

        try:
            self._verify_prerequisites(state)

            skip_reason = self._skip_reason(state)
            if skip_reason is not None:
                return self._record_skip(state, skip_reason)

            dataframe = load_training_dataframe(state.dataset_path)
            dataframe, removed_duplicates = self._apply_duplicate_handling(state, dataframe)
            features, target = prepare_features_and_target(
                dataframe,
                state.target_column,
                state.excluded_feature_names,
            )

            split = split_dataset(
                features,
                target,
                test_size=test_size,
                random_state=random_state,
            )
            self._record_split(state, split, removed_duplicates, test_size, random_state)

            cv_config = self._build_cv_config(
                state,
                target,
                cross_validation=cross_validation,
                cross_validation_folds=cross_validation_folds,
            )
            state.cross_validation_config = cv_config

            experiments, pipelines, cv_results = self._run_experiments(
                state,
                split,
                cv_config,
                model_parameters=model_parameters,
            )
        except (PreprocessingError, ModelToolError) as error:
            state.training_status = TrainingStatus.FAILED.value
            self.log_message(state, f"Training aborted: {error}", level="error")
            state.add_error(str(error))
            raise
        except Exception as error:
            state.training_status = TrainingStatus.FAILED.value
            self.log_message(
                state, f"Unexpected error during training: {error!r}", level="error"
            )
            state.add_error(f"Unexpected error during training: {error!r}")
            raise

        state.experiment_results = experiments
        state.experiment_count = len(experiments)
        self._finalise(state, pipelines, cv_results)

        state.execution_time = round(time.perf_counter() - started, 6)
        state.training_status = TrainingStatus.COMPLETED.value
        state.status = AgentStatus.TRAINING_COMPLETED.value
        self._log_summary(state)
        return state

    # ------------------------------------------------------------------ #
    # Prerequisites
    # ------------------------------------------------------------------ #
    def _verify_prerequisites(self, state: AgentState) -> None:
        """Ensure analysis, preprocessing and model selection are done."""
        problems: List[str] = []

        if state.dataset_shape is None or not state.column_names:
            problems.append("the Data Analysis Agent has not been completed")
        if state.preprocessing_status != PreprocessingStatus.COMPLETED.value:
            problems.append(
                f"the Preprocessing Agent has not been completed "
                f"(status: {state.preprocessing_status})"
            )
        if state.preprocessing_config is None:
            problems.append("no preprocessing configuration is available")
        if not state.target_column:
            problems.append("no target column was resolved")
        if state.model_selection_status != TrainingStatus.COMPLETED.value:
            problems.append(
                f"the Model Selection Agent has not been completed "
                f"(status: {state.model_selection_status})"
            )
        if not state.selected_models:
            problems.append("no candidate model was selected")

        if problems:
            raise PreprocessingError(
                "The Training Agent cannot start because "
                + "; ".join(problems)
                + ". Run DataAnalysisAgent, PreprocessingAgent and "
                "ModelSelectionAgent first."
            )

    def _skip_reason(self, state: AgentState) -> Optional[str]:
        """Return why training cannot run, or None when it can."""
        if state.preprocessing_config.categorical_features or state.preprocessing_config.numerical_features:
            return None
        return "the preprocessing configuration does not contain any usable feature"

    def _record_skip(self, state: AgentState, reason: str) -> AgentState:
        """Record a skipped run without training anything."""
        state.training_status = TrainingStatus.SKIPPED.value
        state.experiment_results = []
        self.log_message(state, f"Training skipped. {reason}", level="warning")
        return state

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def _apply_duplicate_handling(
        self,
        state: AgentState,
        dataframe,
    ) -> Tuple[object, int]:
        """Apply the Phase 3 duplicate strategy to the in-memory frame.

        The source CSV is never touched: `remove_duplicate_rows` returns a new
        DataFrame. The number of removed rows is returned so it can be recorded.
        """
        handling = state.preprocessing_config.duplicate_handling
        strategy = handling.strategy

        if strategy == DuplicateStrategy.KEEP_ALL.value:
            self.log_message(
                state,
                f"Duplicate handling: {handling.duplicate_rows_detected} duplicate row(s) "
                "detected and deliberately kept (strategy 'keep_all'). The metrics may be "
                "slightly biased towards repeated records.",
                level="warning",
            )
            return dataframe, 0

        deduplicated, removed = remove_duplicate_rows(dataframe, strategy=strategy)
        self.log_message(
            state,
            f"Duplicate handling: removed {removed} of {handling.duplicate_rows_detected} "
            "duplicate row(s) from the in-memory frame before splitting. "
            "The source CSV file is unchanged.",
        )
        return deduplicated, removed

    def _record_split(
        self,
        state: AgentState,
        split: SplitResult,
        removed_duplicates: int,
        test_size: float,
        random_state: int,
    ) -> None:
        """Store the split information in the state."""
        handling = state.preprocessing_config.duplicate_handling
        state.data_split = DataSplitInfo(
            test_size=test_size,
            random_state=random_state,
            stratified=split.stratified,
            stratification_reason=split.reason,
            train_samples=split.train_samples,
            test_samples=split.test_samples,
            total_samples=split.total_samples,
            class_distribution_before_split=split.class_distribution_before,
            class_distribution_train=split.class_distribution_train,
            class_distribution_test=split.class_distribution_test,
            duplicate_rows_detected=handling.duplicate_rows_detected,
            duplicate_rows_removed=removed_duplicates,
            duplicate_handling_strategy=handling.strategy,
            input_feature_count=int(split.X_train.shape[1]),
            excluded_feature_count=len(state.excluded_feature_names),
            duplicate_handling_note=handling.notes,
        )
        if not split.stratified and split.reason:
            self.log_message(state, split.reason, level="warning")

    def _build_cv_config(
        self,
        state: AgentState,
        target,
        *,
        cross_validation: bool,
        cross_validation_folds: int,
    ):
        """Build the cross-validation configuration, recording any adjustment."""
        cv_config = build_cross_validation_config(
            target,
            enabled=cross_validation,
            n_splits=cross_validation_folds,
            random_state=project_config.RANDOM_STATE,
        )
        if not cv_config.enabled:
            self.log_message(
                state,
                f"Cross-validation was skipped: {cv_config.adjusted_reason}. "
                "The single train/test evaluation is still reported.",
                level="warning",
            )
        elif cv_config.adjusted_reason:
            self.log_message(
                state,
                f"Cross-validation {cv_config.adjusted_reason}.",
                level="warning",
            )
        else:
            self.log_message(
                state,
                f"Cross-validation enabled: StratifiedKFold("
                f"n_splits={cv_config.n_splits}, shuffle=True, "
                f"random_state={cv_config.random_state}).",
            )
        return cv_config

    # ------------------------------------------------------------------ #
    # Experiments
    # ------------------------------------------------------------------ #
    def _run_experiments(
        self,
        state: AgentState,
        split: SplitResult,
        cv_config,
        *,
        model_parameters: Optional[Dict[str, Dict[str, object]]],
    ) -> Tuple[List[ExperimentResult], Dict[str, object], List[CrossValidationResult]]:
        """Train, evaluate and cross-validate every selected model.

        A failing model is recorded and skipped: one broken estimator must not
        stop the whole workflow.

        Returns:
            A `(experiments, fitted_pipelines, cross_validation_results)` tuple.
        """
        experiments: List[ExperimentResult] = []
        fitted_pipelines: Dict[str, object] = {}
        cv_results: List[CrossValidationResult] = []

        for selected in state.selected_models:
            model_name = selected.name
            overrides = (model_parameters or {}).get(model_name)
            effective_parameters = get_effective_parameters(model_name, overrides)
            experiment = ExperimentResult(
                # Deterministic id: the same model and parameters always get the
                # same id, so a repeated run produces an identical report.
                experiment_id=build_experiment_id(model_name, effective_parameters),
                model_name=model_name,
                model_type=selected.model_type,
                preprocessing_scaling=selected.scaling_required,
                parameters=effective_parameters,
                input_feature_count=int(split.X_train.shape[1]),
                train_samples=split.train_samples,
                test_samples=split.test_samples,
            )

            try:
                pipeline = self._train_single_model(state, experiment, split, overrides)
                fitted_pipelines[model_name] = pipeline

                self._evaluate_single_model(experiment, split, pipeline)
                cv_result = self._cross_validate_single_model(
                    experiment, split, pipeline, cv_config
                )
                cv_results.append(cv_result)
            except (
                TrainingToolError,
                PreprocessingError,
                ModelToolError,
                ValueError,
                TypeError,
            ) as error:
                experiment.status = ExperimentStatus.FAILED.value
                experiment.error_message = describe_training_error(error)
                self.log_message(
                    state,
                    f"Experiment {model_name} failed: {experiment.error_message}",
                    level="error",
                )
                logger.warning(
                    "Experiment %s failed: %s", model_name, experiment.error_message
                )

            experiment.total_time = round(
                experiment.training_time + experiment.prediction_time + experiment.cv_time, 6
            )
            experiments.append(experiment)

        return experiments, fitted_pipelines, cv_results

    def _train_single_model(
        self,
        state: AgentState,
        experiment: ExperimentResult,
        split: SplitResult,
        overrides: Optional[Dict[str, object]],
    ):
        """Fit one Pipeline on the training split only.

        Returns:
            The fitted sklearn `Pipeline`.
        """
        pipeline = build_model_pipeline(
            state.preprocessing_config,
            experiment.model_name,
            overrides,
        )
        definition = get_model_definition(experiment.model_name)

        self.log_message(
            state,
            f"Training {experiment.model_name} "
            f"(scale_numeric={str(definition.scaling_required).lower()}, "
            f"parameters={experiment.parameters}) on {split.train_samples} row(s).",
        )

        started = time.perf_counter()
        pipeline.fit(split.X_train, split.y_train)
        experiment.training_time = round(time.perf_counter() - started, 6)
        experiment.transformed_feature_count = count_transformed_features(pipeline)

        logger.debug(
            "Fitted %s in %.4fs -> %s features",
            experiment.model_name,
            experiment.training_time,
            experiment.transformed_feature_count,
        )
        return pipeline

    def _evaluate_single_model(
        self,
        experiment: ExperimentResult,
        split: SplitResult,
        pipeline,
    ) -> None:
        """Predict on the test split and store the metrics."""
        started = time.perf_counter()
        predictions = pipeline.predict(split.X_test)
        experiment.prediction_time = round(time.perf_counter() - started, 6)

        metrics = evaluate_classifier(split.y_test, predictions)
        experiment.accuracy = metrics["accuracy"]
        experiment.precision = metrics["precision"]
        experiment.recall = metrics["recall"]
        experiment.f1_score = metrics["f1_score"]

        # The full label set is passed explicitly so the matrix always has one
        # row and column per class of the dataset. Without it a test split that
        # happens to hold a single class would produce a 1x1 matrix, which reads
        # like a perfect prediction instead of a degenerate evaluation.
        all_labels = sorted(
            set(split.y_train.unique()) | set(split.y_test.unique()), key=str
        )
        experiment.confusion_matrix = calculate_confusion_matrix(
            split.y_test, predictions, labels=all_labels
        )

    def _cross_validate_single_model(
        self,
        experiment: ExperimentResult,
        split: SplitResult,
        pipeline,
        cv_config,
    ) -> CrossValidationResult:
        """Cross-validate the model on the training split only.

        The test split is never used here: it stays untouched for the single
        final evaluation, so the reported CV score and the reported test score
        remain independent.

        Returns:
            The `CrossValidationResult`, which is always stored - including when
            cross-validation could not run, so the reason is never lost.
        """
        if not cv_config.enabled:
            experiment.status = ExperimentStatus.PARTIAL.value
            return CrossValidationResult(
                model_name=experiment.model_name,
                n_splits=0,
                status="skipped",
                reason=cv_config.adjusted_reason or "cross-validation was disabled",
            )

        started = time.perf_counter()
        try:
            cv_result = run_cross_validation(
                pipeline,
                split.X_train,
                split.y_train,
                cv_config,
                model_name=experiment.model_name,
            )
        except (
            TrainingToolError,
            PreprocessingError,
            ModelToolError,
            ValueError,
            TypeError,
        ) as error:
            # A model that trains fine but cannot be cross-validated is reported
            # as partial: the test metrics stay valid and the reason is recorded.
            experiment.cv_time = round(time.perf_counter() - started, 6)
            experiment.status = ExperimentStatus.PARTIAL.value
            message = describe_training_error(error)
            logger.warning(
                "Cross-validation skipped for %s: %s", experiment.model_name, message
            )
            return CrossValidationResult(
                model_name=experiment.model_name,
                n_splits=0,
                status="skipped",
                reason=message,
            )

        experiment.cv_time = round(time.perf_counter() - started, 6)
        experiment.cv_accuracy_mean = cv_result.accuracy_mean
        experiment.cv_accuracy_std = cv_result.accuracy_std
        experiment.cv_f1_mean = cv_result.f1_mean
        if cv_result.status != "success":
            experiment.status = ExperimentStatus.PARTIAL.value
        return cv_result

    # ------------------------------------------------------------------ #
    # Finalisation
    # ------------------------------------------------------------------ #
    def _note_incomplete_test_split(
        self,
        state: AgentState,
        comparison: ModelComparison,
    ) -> None:
        """Warn when the test split is missing classes of the dataset.

        With a very small or very imbalanced dataset a non-stratified split can
        leave a class out of the test set. The metrics are then still correct but
        they are computed on fewer classes than the model was trained for, which
        must not be presented as a plain result.
        """
        split = state.data_split
        if split is None or not split.class_distribution_test:
            return
        all_labels = set(split.class_distribution_before_split)
        test_labels = set(split.class_distribution_test)
        if len(test_labels) >= len(all_labels):
            return

        missing = sorted(all_labels - test_labels, key=str)
        message = (
            f"The test split contains {len(test_labels)} of {len(all_labels)} "
            f"class(es); class(es) missing from it: {missing}. The hold-out "
            f"metrics of this run are computed on a reduced class set and must "
            f"not be compared with other datasets."
        )
        comparison.notes.append(message)
        self.log_message(state, message, level="warning")

    def _finalise(
        self,
        state: AgentState,
        pipelines: Dict[str, object],
        cv_results: List[CrossValidationResult],
    ) -> None:
        """Rank the experiments, pick the best model and store the artefacts."""
        state.cross_validation_results = cv_results

        comparison = rank_experiments(state.experiment_results)
        state.model_comparison = comparison

        self._note_incomplete_test_split(state, comparison)

        best = select_best_experiment(state.experiment_results)
        state.best_baseline_model = best

        if best is not None and best.model_name in pipelines:
            # Runtime object: never serialised, saved to disk in Phase 9.
            state.best_pipeline = pipelines[best.model_name]

        failures = [
            item
            for item in state.experiment_results
            if item.status == ExperimentStatus.FAILED.value
        ]
        if failures:
            self.log_message(
                state,
                "{} of {} experiment(s) failed: {}".format(
                    len(failures),
                    len(state.experiment_results),
                    ", ".join(
                        f"{item.model_name} ({item.error_message})" for item in failures
                    ),
                ),
                level="warning",
            )
        if comparison.best_model_name:
            self.log_message(
                state,
                "Best baseline model: {} (F1={}, accuracy={}).".format(
                    comparison.best_model_name,
                    _fmt(best.f1_score) if best else "-",
                    _fmt(best.accuracy) if best else "-",
                ),
            )
        else:
            self.log_message(
                state,
                "No model produced usable metrics, so no best model was selected.",
                level="warning",
            )

    def _log_summary(self, state: AgentState) -> None:
        """Log the final ranking.

        The order is logged on a single line rather than as the multi-line
        table: the full table is rendered separately (CLI report and
        `ModelComparison.table_text`) and repeating it here would interleave
        confusingly with the report in the console output.
        """
        comparison = state.model_comparison
        if comparison is None:
            return
        order = ", ".join(
            f"{entry.rank}. {entry.model_name} (f1={_fmt(entry.f1_score)})"
            for entry in comparison.entries
        )
        self.log_message(
            state,
            f"Model comparison ({comparison.successful_count} successful, "
            f"{comparison.failed_count} failed): {order or 'no ranked experiment'}.",
        )
        self.log_message(state, f"Ranking rule: {comparison.ranking_rule}")
        for note in comparison.notes:
            self.log_message(state, note, level="warning")


def _fmt(value: Optional[float]) -> str:
    """Format a metric for a log line."""
    return "-" if value is None else f"{value:.4f}"


__all__ = ["TrainingAgent"]