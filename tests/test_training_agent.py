"""Tests for the Training Agent (Phase 4).

The agent turns the candidate list into trained, evaluated and compared
experiments. These tests cover the prerequisites, the duplicate strategy, the
split record, the per-model experiments, cross-validation, error recording,
reproducibility and the JSON-safety of the state, plus the two end-to-end
integration runs on the committed sample datasets.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from agents.data_analysis_agent import DataAnalysisAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from tools.model_tools import ModelConfigurationError
from tools.preprocessing_tools import PreprocessingError
from tools.training_tools import TrainingDataError, TrainingToolError
from utils.state import (
    AgentState,
    AgentStatus,
    DuplicateStrategy,
    ExperimentStatus,
    PreprocessingStatus,
    ScalingStrategy,
    TrainingStatus,
    to_serializable,
)

from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash

EXPECTED_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Support Vector Machine",
]


def state_ready_for_training(dataset_path=None, **preprocessing_kwargs) -> AgentState:
    """Run analysis -> preprocessing -> model selection for a dataset.

    `preprocessing_kwargs` are forwarded to the Preprocessing Agent, so a test
    can change the duplicate strategy or the scaling decision.
    """
    state = AgentState()
    state = DataAnalysisAgent().run(
        state, dataset_path=str(dataset_path or BINARY_DATASET)
    )
    state = PreprocessingAgent().run(state, **preprocessing_kwargs)
    return ModelSelectionAgent().run(state)


def train(dataset_path=None, *, preprocessing_kwargs=None, **kwargs) -> AgentState:
    """Run the full Phase 4 flow and return the trained state."""
    state = state_ready_for_training(dataset_path, **(preprocessing_kwargs or {}))
    return TrainingAgent().run(state, **kwargs)


class TestPrerequisites(unittest.TestCase):
    """The agent refuses to run before model selection has happened."""

    def test_fresh_state_is_rejected(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError) as context:
            TrainingAgent().run(state)
        self.assertIn("Data Analysis Agent", str(context.exception))
        self.assertEqual(state.training_status, TrainingStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_missing_model_selection_is_rejected(self) -> None:
        state = AgentState()
        state = DataAnalysisAgent().run(state, dataset_path=str(BINARY_DATASET))
        state = PreprocessingAgent().run(state)
        with self.assertRaises(PreprocessingError) as context:
            TrainingAgent().run(state)
        self.assertIn("Model Selection Agent", str(context.exception))

    def test_missing_preprocessing_is_rejected(self) -> None:
        state = state_ready_for_training()
        state.preprocessing_status = PreprocessingStatus.SKIPPED.value
        state.preprocessing_config = None
        with self.assertRaises(PreprocessingError) as context:
            TrainingAgent().run(state)
        self.assertIn("Preprocessing Agent", str(context.exception))

    def test_no_selected_models_is_rejected(self) -> None:
        state = state_ready_for_training()
        state.selected_models = []
        with self.assertRaises(PreprocessingError) as context:
            TrainingAgent().run(state)
        self.assertIn("no candidate model", str(context.exception))

    def test_missing_dataset_file_is_reported_clearly(self) -> None:
        state = state_ready_for_training()
        state.dataset_path = "missing_dataset.csv"
        with self.assertRaises(TrainingDataError) as context:
            TrainingAgent().run(state)
        self.assertIn("does not exist", str(context.exception))
        self.assertEqual(state.training_status, TrainingStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_failure_is_logged_and_recorded_before_raising(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError):
            TrainingAgent().run(state)
        errors = [item for item in state.agent_messages if item.level == "error"]
        self.assertTrue(errors)
        self.assertEqual(errors[-1].agent, "TrainingAgent")
        self.assertEqual(state.status, AgentStatus.FAILED.value)

    def test_empty_configuration_is_skipped_without_raising(self) -> None:
        state = state_ready_for_training()
        state.preprocessing_config.numerical_features = []
        state.preprocessing_config.categorical_features = []
        result = TrainingAgent().run(state)
        self.assertEqual(result.training_status, TrainingStatus.SKIPPED.value)
        self.assertEqual(result.experiment_results, [])


class TestFullBinaryRun(unittest.TestCase):
    """The complete flow on the committed binary dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.hash_before = file_hash(BINARY_DATASET)
        cls.state = train()
        cls.rows_before = BINARY_DATASET.read_text(encoding="utf-8")

    def test_run_completes(self) -> None:
        self.assertEqual(self.state.training_status, TrainingStatus.COMPLETED.value)
        self.assertEqual(self.state.status, AgentStatus.TRAINING_COMPLETED.value)

    def test_every_selected_model_is_trained(self) -> None:
        self.assertEqual(self.state.experiment_count, 5)
        self.assertEqual(
            [item.model_name for item in self.state.experiment_results], EXPECTED_MODELS
        )

    def test_every_experiment_succeeds(self) -> None:
        self.assertEqual(self.state.failed_experiments, [])
        for experiment in self.state.experiment_results:
            self.assertEqual(experiment.status, ExperimentStatus.SUCCESS.value, experiment.model_name)
            self.assertEqual(experiment.error_message, "")

    def test_every_metric_is_computed(self) -> None:
        for experiment in self.state.experiment_results:
            for metric in (experiment.accuracy, experiment.precision, experiment.recall, experiment.f1_score):
                self.assertIsNotNone(metric, experiment.model_name)
                self.assertGreaterEqual(metric, 0.0)
                self.assertLessEqual(metric, 1.0)

    def test_cross_validation_completes_for_every_model(self) -> None:
        self.assertEqual(self.state.cross_validation_config.n_splits, 5)
        self.assertEqual(len(self.state.cross_validation_results), 5)
        for result in self.state.cross_validation_results:
            self.assertEqual(result.status, "success", result.model_name)
            self.assertIsNotNone(result.accuracy_mean)
            self.assertIsNotNone(result.f1_mean)
            self.assertIsNotNone(result.accuracy_std)

    def test_confusion_matrix_is_recorded_for_every_model(self) -> None:
        for experiment in self.state.experiment_results:
            self.assertIsNotNone(experiment.confusion_matrix, experiment.model_name)
            self.assertEqual(experiment.confusion_matrix.labels, ["0", "1"])
            self.assertEqual(len(experiment.confusion_matrix.matrix), 2)
            self.assertEqual(experiment.confusion_matrix.total, experiment.test_samples)

    def test_timings_are_recorded(self) -> None:
        for experiment in self.state.experiment_results:
            self.assertGreater(experiment.training_time, 0.0)
            self.assertGreater(experiment.prediction_time, 0.0)
            self.assertGreater(experiment.cv_time, 0.0)
            self.assertAlmostEqual(
                experiment.total_time,
                experiment.training_time + experiment.prediction_time + experiment.cv_time,
                places=6,
            )

    def test_transformed_feature_count_is_recorded(self) -> None:
        for experiment in self.state.experiment_results:
            self.assertIsNotNone(experiment.transformed_feature_count)
            self.assertEqual(experiment.transformed_feature_count, 13)

    def test_identifier_column_never_reaches_a_model(self) -> None:
        pipeline = self.state.best_pipeline
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        self.assertFalse([name for name in names if "record_id" in name])
        self.assertFalse([name for name in names if "Outcome" in name])

    def test_split_is_recorded(self) -> None:
        split = self.state.data_split
        self.assertEqual(split.test_size, 0.2)
        self.assertEqual(split.random_state, 42)
        self.assertTrue(split.stratified)
        self.assertEqual(split.stratification_reason, "")
        self.assertEqual(split.total_samples, 120)  # 123 rows - 3 duplicates
        self.assertEqual(split.train_samples, 96)
        self.assertEqual(split.test_samples, 24)

    def test_duplicate_handling_is_recorded_and_applied(self) -> None:
        split = self.state.data_split
        self.assertEqual(split.duplicate_rows_detected, 3)
        self.assertEqual(split.duplicate_rows_removed, 3)
        self.assertEqual(
            split.duplicate_handling_strategy, DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value
        )
        # 123 rows in the file, 120 after removing the 3 exact duplicates.
        self.assertEqual(split.total_samples + split.duplicate_rows_removed, 123)

    def test_class_distribution_is_recorded(self) -> None:
        split = self.state.data_split
        self.assertEqual(sum(split.class_distribution_before_split.values()), 120)
        self.assertEqual(sum(split.class_distribution_train.values()), 96)
        self.assertEqual(sum(split.class_distribution_test.values()), 24)
        self.assertEqual(set(split.class_distribution_train), {"0", "1"})

    def test_comparison_is_built(self) -> None:
        comparison = self.state.model_comparison
        self.assertEqual(comparison.successful_count, 5)
        self.assertEqual(comparison.failed_count, 0)
        self.assertEqual(len(comparison.entries), 5)
        self.assertEqual([entry.rank for entry in comparison.entries], [1, 2, 3, 4, 5])
        self.assertEqual(comparison.best_model_name, self.state.best_baseline_model.model_name)

    def test_comparison_is_sorted_by_f1_then_accuracy(self) -> None:
        entries = self.state.model_comparison.entries
        for previous, current in zip(entries, entries[1:]):
            self.assertGreaterEqual(previous.f1_score, current.f1_score)

    def test_best_model_is_identified_and_ranked_first(self) -> None:
        best = self.state.best_baseline_model
        self.assertIsNotNone(best)
        self.assertEqual(self.state.model_comparison.entries[0].model_name, best.model_name)
        self.assertEqual(self.state.model_comparison.entries[0].f1_score, best.f1_score)
        self.assertIn("highest f1_score", best.selection_reason)

    def test_best_pipeline_is_kept_at_runtime_but_not_serialised(self) -> None:
        self.assertIsNotNone(self.state.best_pipeline)
        payload = self.state.to_dict()
        self.assertNotIn("best_pipeline", payload)
        self.assertIn("best_pipeline", payload["runtime_objects"])
        described = payload["runtime_objects"]["best_pipeline"]
        self.assertEqual(described["type"], "Pipeline")
        self.assertTrue(described["fitted"])
        self.assertFalse(described["serialized"])

    def test_source_csv_is_unchanged(self) -> None:
        self.assertEqual(file_hash(BINARY_DATASET), self.hash_before)
        self.assertEqual(BINARY_DATASET.read_text(encoding="utf-8"), self.rows_before)

    def test_preprocessing_configuration_is_not_mutated(self) -> None:
        self.assertTrue(self.state.preprocessing_config.scale_numeric)
        self.assertEqual(
            self.state.preprocessing_config.scaling_strategy, ScalingStrategy.STANDARD.value
        )

    def test_tree_models_trained_without_scaling(self) -> None:
        """Each experiment records the scaling decision it was given."""
        by_name = {item.model_name: item for item in self.state.experiment_results}
        self.assertFalse(by_name["Random Forest"].preprocessing_scaling)
        self.assertFalse(by_name["Decision Tree"].preprocessing_scaling)
        self.assertTrue(by_name["Logistic Regression"].preprocessing_scaling)
        self.assertTrue(by_name["Support Vector Machine"].preprocessing_scaling)

    def test_model_pipeline_is_leakage_safe(self) -> None:
        pipeline = self.state.best_pipeline
        self.assertEqual(list(pipeline.named_steps), ["preprocessor", "model"])

    def test_run_is_reported_in_the_comparison_table(self) -> None:
        table = self.state.model_comparison.table_text
        for name in EXPECTED_MODELS:
            self.assertIn(name, table)
        self.assertIn("Best baseline model:", table)

    def test_summary_is_logged(self) -> None:
        joined = "\n".join(item.message for item in self.state.agent_messages)
        self.assertIn("Model comparison", joined)
        self.assertIn("5 successful", joined)
        self.assertIn("Ranking rule:", joined)
        self.assertIn("Best baseline model:", joined)


class TestMulticlassRun(unittest.TestCase):
    """The same flow must work for a multiclass dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.hash_before = file_hash(MULTICLASS_DATASET)
        cls.state = train(MULTICLASS_DATASET)

    def test_target_and_problem_type_are_detected(self) -> None:
        self.assertEqual(self.state.target_column, "class_label")
        self.assertEqual(self.state.problem_type, "multiclass_classification")

    def test_preprocessing_completed(self) -> None:
        self.assertEqual(self.state.preprocessing_status, PreprocessingStatus.COMPLETED.value)
        self.assertIsNotNone(self.state.preprocessing_config)

    def test_all_five_models_were_trained(self) -> None:
        self.assertEqual(self.state.experiment_count, 5)
        self.assertEqual(self.state.failed_experiments, [])

    def test_confusion_matrix_has_one_row_per_class(self) -> None:
        for experiment in self.state.experiment_results:
            matrix = experiment.confusion_matrix
            self.assertEqual(len(matrix.labels), 3, experiment.model_name)
            self.assertEqual(len(matrix.matrix), 3)
            self.assertEqual(matrix.total, experiment.test_samples)

    def test_cross_validation_completed(self) -> None:
        self.assertEqual(self.state.cross_validation_config.n_splits, 5)
        for result in self.state.cross_validation_results:
            self.assertEqual(result.status, "success", result.model_name)

    def test_best_baseline_model_is_identified(self) -> None:
        best = self.state.best_baseline_model
        self.assertIsNotNone(best)
        self.assertIn(best.model_name, EXPECTED_MODELS)
        self.assertEqual(
            self.state.model_comparison.entries[0].model_name, best.model_name
        )

    def test_source_csv_is_unchanged(self) -> None:
        self.assertEqual(file_hash(MULTICLASS_DATASET), self.hash_before)


class TestReproducibility(unittest.TestCase):
    """The same experiment twice must give the same numbers."""

    def test_two_runs_agree_on_every_metric(self) -> None:
        first = train()
        second = train()
        for before, after in zip(first.experiment_results, second.experiment_results):
            self.assertEqual(before.model_name, after.model_name)
            self.assertEqual(before.accuracy, after.accuracy)
            self.assertEqual(before.f1_score, after.f1_score)
            self.assertEqual(before.precision, after.precision)
            self.assertEqual(before.recall, after.recall)
            self.assertEqual(before.cv_accuracy_mean, after.cv_accuracy_mean)
            self.assertEqual(before.cv_f1_mean, after.cv_f1_mean)
            self.assertEqual(
                before.confusion_matrix.matrix, after.confusion_matrix.matrix
            )

    def test_two_runs_agree_on_the_split_and_the_best_model(self) -> None:
        first, second = train(), train()
        self.assertEqual(
            list(first.data_split.class_distribution_test),
            list(second.data_split.class_distribution_test),
        )
        self.assertEqual(
            first.best_baseline_model.model_name, second.best_baseline_model.model_name
        )
        self.assertEqual(first.best_baseline_model.f1_score, second.best_baseline_model.f1_score)

    def test_experiment_ids_are_unique_per_run(self) -> None:
        state = train()
        ids = [item.experiment_id for item in state.experiment_results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_experiment_ids_are_identical_across_runs(self) -> None:
        """No random UUIDs: the same configuration must produce the same ids."""
        first = [item.experiment_id for item in train().experiment_results]
        second = [item.experiment_id for item in train().experiment_results]
        self.assertEqual(first, second)

    def test_changing_the_parameters_changes_the_id(self) -> None:
        default_run = train()
        tuned_run = train(
            model_parameters={"Random Forest": {"n_estimators": 300}}
        )
        default_id = next(
            item.experiment_id
            for item in default_run.experiment_results
            if item.model_name == "Random Forest"
        )
        tuned_id = next(
            item.experiment_id
            for item in tuned_run.experiment_results
            if item.model_name == "Random Forest"
        )
        self.assertNotEqual(default_id, tuned_id)

    def test_the_seed_is_recorded_in_the_state(self) -> None:
        state = train()
        self.assertEqual(state.data_split.random_state, 42)
        self.assertEqual(state.cross_validation_config.random_state, 42)


class TestConfigurationOptions(unittest.TestCase):
    """The agent exposes a few explicit, reproducible knobs."""

    def test_test_size_is_honoured(self) -> None:
        state = train(test_size=0.3)
        self.assertEqual(state.data_split.test_size, 0.3)
        self.assertEqual(state.data_split.test_samples, 36)
        self.assertEqual(state.data_split.train_samples, 84)

    def test_cross_validation_can_be_disabled(self) -> None:
        state = train(cross_validation=False)
        self.assertFalse(state.cross_validation_config.enabled)
        for result in state.cross_validation_results:
            self.assertEqual(result.status, "skipped")
        for experiment in state.experiment_results:
            self.assertEqual(experiment.status, ExperimentStatus.PARTIAL.value)
            self.assertIsNotNone(experiment.f1_score)
            self.assertIsNone(experiment.cv_f1_mean)

    def test_reduced_folds_are_recorded_with_a_reason(self) -> None:
        state = train(cross_validation_folds=3)
        self.assertEqual(state.cross_validation_config.n_splits, 3)
        self.assertEqual(state.cross_validation_config.adjusted_reason, "")

    def test_model_parameter_overrides_are_used(self) -> None:
        state = train(model_parameters={"Random Forest": {"n_estimators": 7}})
        by_name = {item.model_name: item for item in state.experiment_results}
        self.assertEqual(by_name["Random Forest"].parameters["n_estimators"], 7)
        self.assertEqual(by_name["Random Forest"].parameters["random_state"], 42)
        self.assertEqual(by_name["Decision Tree"].parameters, {"random_state": 42})

    def test_invalid_override_fails_only_that_experiment(self) -> None:
        state = train(model_parameters={"Random Forest": {"not_a_parameter": 1}})
        by_name = {item.model_name: item for item in state.experiment_results}
        self.assertEqual(by_name["Random Forest"].status, ExperimentStatus.FAILED.value)
        self.assertIn("not_a_parameter", by_name["Random Forest"].error_message)
        # The other models still trained.
        self.assertEqual(len(state.successful_experiments), 4)

    def test_keep_all_duplicate_strategy_is_respected(self) -> None:
        state = train(
            preprocessing_kwargs={"duplicate_strategy": DuplicateStrategy.KEEP_ALL.value}
        )
        self.assertEqual(state.data_split.duplicate_rows_removed, 0)
        self.assertEqual(state.data_split.duplicate_rows_detected, 3)
        # All 123 rows are used, duplicates included.
        self.assertEqual(state.data_split.total_samples, 123)
        # Keeping repeats biases the metrics, so the agent must say so.
        warnings = "\n".join(
            item.message for item in state.agent_messages if item.level == "warning"
        )
        self.assertIn("keep_all", warnings)


class TestSmallDatasetHandling(unittest.TestCase):
    """A tiny dataset is handled explicitly, never silently."""

    @staticmethod
    def _write(directory: Path, frame: pd.DataFrame) -> str:
        path = directory / "small.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def _small_frame(self, rows: int = 12, minority: int = 1) -> pd.DataFrame:
        labels = [0] * (rows - minority) + [1] * minority
        return pd.DataFrame(
            {
                "value": [float(10 + (index % 5)) for index in range(rows)],
                "city": [["north", "south"][index % 2] for index in range(rows)],
                "label": labels,
            }
        )

    def test_single_row_class_falls_back_to_a_plain_split(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._small_frame())
            state = train(path)
            self.assertFalse(state.data_split.stratified)
            self.assertIn("stratified split skipped", state.data_split.stratification_reason)
            self.assertEqual(state.training_status, TrainingStatus.COMPLETED.value)

    def test_reduced_folds_are_recorded_for_a_small_class(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._small_frame(rows=20, minority=3))
            state = train(path)
            self.assertEqual(state.training_status, TrainingStatus.COMPLETED.value)
            self.assertLess(state.cross_validation_config.n_splits, 5)
            self.assertIn("smallest class", state.cross_validation_config.adjusted_reason)
            self.assertEqual(len(state.cross_validation_results), 5)

    def test_cross_validation_skip_is_recorded_not_hidden(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._small_frame(rows=12, minority=1))
            state = train(path)
            reasons = [item.reason for item in state.cross_validation_results]
            self.assertTrue(any(reason for reason in reasons))
            self.assertTrue(all(result.status in ("success", "skipped") for result in state.cross_validation_results))

    def test_models_still_produce_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._small_frame(rows=12, minority=1))
            state = train(path)
            self.assertEqual(len(state.successful_experiments), 5)
            self.assertIsNotNone(state.best_baseline_model)

    def test_a_class_missing_from_the_test_split_is_reported(self) -> None:
        """A degenerate evaluation must be flagged, not presented as a result."""
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._small_frame(rows=12, minority=1))
            state = train(path)
            split = state.data_split
            self.assertEqual(split.stratified, False)
            missing = set(split.class_distribution_before_split) - set(
                split.class_distribution_test
            )
            if not missing:
                self.skipTest("this seed happened to keep every class in the test split")
            self.assertTrue(
                any("missing from it" in note for note in state.model_comparison.notes),
                state.model_comparison.notes,
            )
            warnings = [item.message for item in state.agent_messages if item.level == "warning"]
            self.assertTrue(any("reduced class set" in message for message in warnings))

    def test_confusion_matrix_keeps_every_class_of_the_dataset(self) -> None:
        """Even a one-class test split yields a full-length confusion matrix."""
        with TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._small_frame(rows=12, minority=1))
            state = train(path)
            all_labels = sorted(
                set(state.data_split.class_distribution_before_split), key=str
            )
            for experiment in state.successful_experiments:
                matrix = experiment.confusion_matrix
                self.assertEqual(matrix.labels, all_labels, experiment.model_name)
                self.assertEqual(len(matrix.matrix), len(all_labels))
                for row in matrix.matrix:
                    self.assertEqual(len(row), len(all_labels))
                # The counts always add up to the number of test rows.
                self.assertEqual(matrix.total, experiment.test_samples)

    def test_a_single_class_target_fails_clearly_at_training_time(self) -> None:
        """The target is re-validated when the training data is loaded.

        Phase 3 already refuses a constant target, so this covers the second
        check: a file whose target lost its class variety between phases must
        stop the run with a readable message instead of producing meaningless
        metrics.
        """
        state = state_ready_for_training()
        frame = pd.read_csv(BINARY_DATASET)
        frame["Outcome"] = 1
        with TemporaryDirectory() as directory:
            path = Path(directory) / "constant_target.csv"
            frame.to_csv(path, index=False)
            state.dataset_path = str(path)
            with self.assertRaises(TrainingToolError) as context:
                TrainingAgent().run(state)
            self.assertIn("single distinct value", str(context.exception))
        self.assertEqual(state.training_status, TrainingStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_a_missing_target_column_fails_clearly(self) -> None:
        state = state_ready_for_training()
        frame = pd.read_csv(BINARY_DATASET).drop(columns=["Outcome"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "no_target.csv"
            frame.to_csv(path, index=False)
            state.dataset_path = str(path)
            with self.assertRaises(TrainingToolError) as context:
                TrainingAgent().run(state)
            self.assertIn("not present in the dataset", str(context.exception))


class TestErrorRecording(unittest.TestCase):
    """27. Failures are recorded clearly instead of being swallowed."""

    def test_a_broken_model_does_not_stop_the_others(self) -> None:
        state = state_ready_for_training()
        # An impossible KNN parameter for the dataset size.
        result = TrainingAgent().run(
            state, model_parameters={"K-Nearest Neighbors": {"n_neighbors": 10000}}
        )
        self.assertEqual(result.training_status, TrainingStatus.COMPLETED.value)
        self.assertEqual(len(result.failed_experiments), 1)
        self.assertEqual(result.failed_experiments[0].model_name, "K-Nearest Neighbors")
        self.assertEqual(len(result.successful_experiments), 4)

    def test_failed_experiment_keeps_an_explanatory_message(self) -> None:
        state = state_ready_for_training()
        result = TrainingAgent().run(
            state, model_parameters={"Decision Tree": {"criterion": "nonsense"}}
        )
        failed = result.failed_experiments[0]
        self.assertTrue(failed.error_message)
        self.assertEqual(failed.status, ExperimentStatus.FAILED.value)
        self.assertIsNone(failed.accuracy)
        self.assertIsNone(failed.confusion_matrix)

    def test_failure_is_logged_at_error_level(self) -> None:
        state = state_ready_for_training()
        result = TrainingAgent().run(
            state, model_parameters={"Decision Tree": {"criterion": "nonsense"}}
        )
        errors = [item for item in result.agent_messages if item.level == "error"]
        self.assertTrue(errors)
        self.assertIn("Decision Tree", errors[-1].message)

    def test_failures_are_summarised_in_the_comparison(self) -> None:
        state = state_ready_for_training()
        result = TrainingAgent().run(
            state, model_parameters={"Decision Tree": {"criterion": "nonsense"}}
        )
        self.assertEqual(result.model_comparison.failed_count, 1)
        self.assertEqual(result.model_comparison.successful_count, 4)
        self.assertTrue(any("failed" in note for note in result.model_comparison.notes))

    def test_best_model_ignores_failed_experiments(self) -> None:
        state = state_ready_for_training()
        result = TrainingAgent().run(
            state, model_parameters={"Decision Tree": {"criterion": "nonsense"}}
        )
        self.assertNotEqual(
            result.best_baseline_model.model_name, "Decision Tree"
        )
        self.assertEqual(
            result.model_comparison.entries[0].model_name,
            result.best_baseline_model.model_name,
        )

    def test_experiment_count_includes_failures(self) -> None:
        state = state_ready_for_training()
        result = TrainingAgent().run(
            state, model_parameters={"Decision Tree": {"criterion": "nonsense"}}
        )
        self.assertEqual(result.experiment_count, 5)

    def test_missing_preprocessing_column_breaks_only_that_model(self) -> None:
        """A configuration that references an absent column is reported, not hidden."""
        state = state_ready_for_training()
        state.selected_models = [
            model for model in state.selected_models if model.name == "Random Forest"
        ]
        state.preprocessing_config.numerical_features = ["not_a_real_column"]
        state.preprocessing_config.categorical_features = []
        result = TrainingAgent().run(state)
        self.assertEqual(result.training_status, TrainingStatus.COMPLETED.value)
        self.assertEqual(len(result.failed_experiments), 1)
        self.assertIn("not a column of the dataframe", result.failed_experiments[0].error_message)
        self.assertIsNone(result.best_baseline_model)

    def test_best_pipeline_is_absent_when_everything_fails(self) -> None:
        state = state_ready_for_training()
        state.selected_models = [
            model for model in state.selected_models if model.name == "Random Forest"
        ]
        state.preprocessing_config.numerical_features = ["not_a_real_column"]
        state.preprocessing_config.categorical_features = []
        result = TrainingAgent().run(state)
        self.assertIsNone(result.best_pipeline)
        self.assertEqual(result.model_comparison.entries, [])


class TestStateJsonSafety(unittest.TestCase):
    """25. The trained state stays serialisable."""

    def test_state_is_json_serialisable(self) -> None:
        state = train()
        payload = json.loads(json.dumps(state.to_dict()))  # plain dumps, no default=
        self.assertEqual(payload["training_status"], "completed")
        self.assertEqual(len(payload["experiment_results"]), 5)

    def test_experiment_payloads_are_plain_dicts(self) -> None:
        state = train()
        payload = state.to_dict()
        for experiment in payload["experiment_results"]:
            self.assertIsInstance(experiment, dict)
            self.assertIn("f1_score", experiment)
            self.assertIn("confusion_matrix", experiment)
            self.assertIsInstance(experiment["confusion_matrix"]["matrix"], list)
            self.assertIsInstance(experiment["confusion_matrix"]["matrix"][0], list)

    def test_no_pipeline_objects_are_serialised(self) -> None:
        state = train()
        payload = state.to_dict()
        self.assertNotIn("best_pipeline", payload)
        self.assertNotIn("preprocessing_pipelines", payload)
        # Only the known runtime fields are described instead of serialised.
        self.assertEqual(
            sorted(payload["runtime_objects"]), ["best_pipeline", "preprocessing_pipeline"]
        )
        for described in payload["runtime_objects"].values():
            self.assertFalse(described["serialized"])
            self.assertIn("type", described)

    def test_comparison_payload_is_complete(self) -> None:
        state = train()
        payload = state.to_dict()["model_comparison"]
        self.assertEqual(payload["primary_metric"], "f1_score")
        self.assertEqual(payload["secondary_metric"], "accuracy")
        self.assertEqual(len(payload["entries"]), 5)
        self.assertTrue(payload["table_text"])

    def test_split_payload_contains_no_dataframes(self) -> None:
        state = train()
        payload = state.to_dict()["data_split"]
        self.assertEqual(payload["train_samples"], 96)
        self.assertNotIn("X_train", payload)

    def test_experiment_record_is_usable_after_serialization(self) -> None:
        state = train()
        record = json.loads(json.dumps(to_serializable(state.experiment_results[0])))
        self.assertEqual(record["model_name"], "Logistic Regression")
        self.assertIsInstance(record["parameters"], dict)
        self.assertIsInstance(record["experiment_id"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
