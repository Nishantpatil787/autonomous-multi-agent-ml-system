"""Tests for the Evaluation Agent (Phase 5).

The agent reads the records the Training Agent produced and never trains
anything, so these tests check the orchestration contract: the prerequisites,
what lands in the shared state, that nothing was retrained, the wording rules,
and the two end-to-end integration runs on the committed sample datasets.
"""

from __future__ import annotations

import json
import re
import unittest

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from tools.evaluation_tools import EvaluationDataError
from tools.preprocessing_tools import PreprocessingError
from utils import config
from utils.state import (
    AgentState,
    AgentStatus,
    ConfusionMatrixData,
    CrossValidationResult,
    EvaluationStatus,
    ExperimentResult,
    ExperimentStatus,
    GapClassification,
    to_serializable,
)

from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash


def trained_state(dataset_path=None, *, training_kwargs=None, **preprocessing_kwargs):
    """Run analysis -> preprocessing -> model selection -> training."""
    state = AgentState()
    state = DataAnalysisAgent().run(
        state, dataset_path=str(dataset_path or BINARY_DATASET)
    )
    state = PreprocessingAgent().run(state, **preprocessing_kwargs)
    state = ModelSelectionAgent().run(state)
    return TrainingAgent().run(state, **(training_kwargs or {}))


def hand_built_state(**overrides) -> AgentState:
    """A minimal state that already looks trained, for fast unit tests."""
    state = AgentState()
    state.dataset_path = str(BINARY_DATASET)
    state.dataset_name = BINARY_DATASET.name
    state.problem_type = "binary_classification"
    state.target_column = "Outcome"
    state.training_status = "completed"
    state.experiment_count = 1
    state.experiment_results = [
        ExperimentResult(
            experiment_id="exp_a",
            model_name="Logistic Regression",
            model_type="linear",
            accuracy=0.875,
            precision=0.88,
            recall=0.87,
            f1_score=0.876,
            cv_f1_mean=0.894,
            cv_accuracy_mean=0.895,
            training_time=0.03,
            prediction_time=0.01,
            cv_time=0.2,
            total_time=0.24,
            confusion_matrix=ConfusionMatrixData(
                labels=["0", "1"], matrix=[[8, 1], [2, 13]], total=24
            ),
            status=ExperimentStatus.SUCCESS.value,
        )
    ]
    state.cross_validation_results = [
        CrossValidationResult(
            model_name="Logistic Regression",
            n_splits=5,
            status="success",
            f1_mean=0.894,
            accuracy_mean=0.895,
            accuracy_std=0.05,
            f1_std=0.04,
        )
    ]
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


# --------------------------------------------------------------------------- #
# 1: prerequisites
# --------------------------------------------------------------------------- #
class TestPrerequisites(unittest.TestCase):
    """The agent refuses to run before the Training Agent has finished."""

    def test_fresh_state_is_rejected(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError) as context:
            EvaluationAgent().run(state)
        self.assertIn("Training Agent has not been completed", str(context.exception))
        self.assertEqual(state.evaluation_status, EvaluationStatus.FAILED.value)
        self.assertTrue(state.errors)
        self.assertEqual(state.status, AgentStatus.FAILED.value)

    def test_missing_experiments_are_reported(self) -> None:
        state = AgentState()
        state.training_status = "completed"
        state.problem_type = "binary_classification"
        with self.assertRaises(PreprocessingError) as context:
            EvaluationAgent().run(state)
        self.assertIn("no experiment was recorded", str(context.exception))

    def test_failed_training_is_rejected(self) -> None:
        state = AgentState()
        state.training_status = "failed"
        state.problem_type = "binary_classification"
        with self.assertRaises(PreprocessingError):
            EvaluationAgent().run(state)

    def test_skipped_training_is_rejected(self) -> None:
        state = AgentState()
        state.training_status = "skipped"
        state.problem_type = "binary_classification"
        with self.assertRaises(PreprocessingError):
            EvaluationAgent().run(state)

    def test_missing_problem_type_is_reported(self) -> None:
        state = hand_built_state(problem_type=None)
        with self.assertRaises(PreprocessingError) as context:
            EvaluationAgent().run(state)
        self.assertIn("no problem type", str(context.exception))

    def test_failure_is_logged_and_recorded_as_a_message(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError):
            EvaluationAgent().run(state)
        self.assertTrue(
            any(item.level == "error" for item in state.agent_messages)
        )

    def test_prerequisite_failure_leaves_no_report(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError):
            EvaluationAgent().run(state)
        self.assertIsNone(state.evaluation_report)
        self.assertFalse(state.has_evaluation_report)


# --------------------------------------------------------------------------- #
# Report content
# --------------------------------------------------------------------------- #
class TestReportContent(unittest.TestCase):
    """What the agent writes into the shared state."""

    def setUp(self) -> None:
        self.state = EvaluationAgent().run(hand_built_state())
        self.report = self.state.evaluation_report

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.state.evaluation_status, EvaluationStatus.COMPLETED.value)
        self.assertEqual(self.report.overall_status, EvaluationStatus.COMPLETED.value)

    def test_workflow_status_moves_to_evaluation_completed(self) -> None:
        self.assertEqual(self.state.status, AgentStatus.EVALUATION_COMPLETED.value)

    def test_best_model_is_recommended(self) -> None:
        self.assertIsNotNone(self.report.best_model)
        self.assertEqual(self.report.best_model.best_baseline_model, "Logistic Regression")

    def test_selection_reason_is_recorded(self) -> None:
        self.assertIn("highest hold-out f1_score", self.report.selection_reason)

    def test_performance_summary_is_recorded(self) -> None:
        self.assertIn("Logistic Regression achieved", self.report.performance_summary)

    def test_model_evaluations_are_recorded(self) -> None:
        self.assertEqual(len(self.report.model_evaluations), 1)
        self.assertEqual(self.report.model_evaluations[0].rank, 1)

    def test_gap_analysis_is_recorded(self) -> None:
        gap = self.report.gap_analysis
        self.assertIsNotNone(gap)
        self.assertEqual(gap.models_with_gap, 1)
        self.assertEqual(
            self.report.model_evaluations[0].f1_gap_classification,
            GapClassification.CLOSE.value,
        )

    def test_confusion_matrix_analysis_is_recorded(self) -> None:
        self.assertEqual(len(self.report.confusion_matrix_analysis), 1)
        analysis = self.report.confusion_matrix_analysis[0]
        self.assertTrue(analysis.available)
        self.assertEqual(analysis.correct_predictions, 21)
        self.assertEqual(analysis.incorrect_predictions, 3)

    def test_execution_time_is_recorded(self) -> None:
        timing = self.report.execution_time_analysis
        self.assertIsNotNone(timing)
        self.assertAlmostEqual(timing.total_experiment_time, 0.24, places=6)
        self.assertEqual(timing.fastest_model, "Logistic Regression")

    def test_experiment_count_is_recorded(self) -> None:
        counts = self.report.experiment_count
        self.assertIsNotNone(counts)
        self.assertEqual(counts.total_experiments, 1)
        self.assertEqual(counts.baseline_models_attempted, 1)
        self.assertEqual(counts.hyperparameter_experiments, 0)

    def test_observations_limitations_and_recommendations_are_recorded(self) -> None:
        self.assertTrue(self.report.observations)
        self.assertTrue(self.report.limitations)
        self.assertTrue(self.report.recommendations)
        self.assertEqual(
            self.state.evaluation_limitations, self.report.limitations
        )

    def test_baseline_parameters_are_copied_into_the_recommendation(self) -> None:
        state = hand_built_state()
        state.experiment_results[0].parameters = {"max_iter": 1000, "random_state": 42}
        report = EvaluationAgent().run(state).evaluation_report
        self.assertEqual(
            report.best_model.parameters, {"max_iter": 1000, "random_state": 42}
        )

    def test_split_size_is_carried_into_the_report(self) -> None:
        from utils.state import DataSplitInfo

        state = hand_built_state(data_split=DataSplitInfo(test_samples=24))
        report = EvaluationAgent().run(state).evaluation_report
        self.assertEqual(report.test_samples, 24)

    def test_dataset_and_problem_type_are_carried_into_the_report(self) -> None:
        self.assertEqual(self.report.dataset_name, BINARY_DATASET.name)
        self.assertEqual(self.report.problem_type, "binary_classification")
        self.assertEqual(self.report.target_column, "Outcome")

    def test_comparison_table_is_rendered(self) -> None:
        self.assertIn("Logistic Regression", self.report.table_text)
        self.assertIn("f1 gap", self.report.table_text)

    def test_execution_time_stays_cumulative(self) -> None:
        """Phase 4 recorded the training wall clock; the agent only adds to it."""
        state = hand_built_state()
        state.execution_time = 12.5
        updated = EvaluationAgent().run(state)
        self.assertGreaterEqual(updated.execution_time, 12.5)

    def test_logging_covers_every_section(self) -> None:
        text = " ".join(item.message for item in self.state.agent_messages)
        for fragment in (
            "Evaluation status",
            "Best baseline model",
            "Hold-out vs cross-validation",
            "Confusion matrix",
            "Execution time",
            "Experiments:",
            "Limitation:",
            "Recommendation:",
        ):
            self.assertIn(fragment, text)

    def test_gap_disclaimer_is_logged(self) -> None:
        text = " ".join(item.message for item in self.state.agent_messages)
        self.assertIn(config.GAP_HEURISTIC_DISCLAIMER, text)


# --------------------------------------------------------------------------- #
# The agent must not retrain anything
# --------------------------------------------------------------------------- #
class TestNoRetraining(unittest.TestCase):
    """Phase 5 only interprets records; it never fits a model."""

    def test_no_experiment_is_added_or_removed(self) -> None:
        state = trained_state()
        before = len(state.experiment_results)
        EvaluationAgent().run(state)
        self.assertEqual(len(state.experiment_results), before)

    def test_training_status_is_untouched(self) -> None:
        state = trained_state()
        EvaluationAgent().run(state)
        self.assertEqual(state.training_status, "completed")

    def test_best_pipeline_is_untouched(self) -> None:
        state = trained_state()
        pipeline = state.best_pipeline
        EvaluationAgent().run(state)
        self.assertIs(state.best_pipeline, pipeline)

    def test_phase_four_records_are_not_mutated(self) -> None:
        state = trained_state()
        snapshot = json.dumps(to_serializable(state.experiment_results), sort_keys=True)
        EvaluationAgent().run(state)
        self.assertEqual(
            json.dumps(to_serializable(state.experiment_results), sort_keys=True),
            snapshot,
        )

    def test_evaluation_is_repeatable(self) -> None:
        """Re-running the evaluation on the same state gives the same report."""
        state = trained_state()
        first = EvaluationAgent().run(state).evaluation_report
        second = EvaluationAgent().run(state).evaluation_report
        self.assertEqual(
            json.dumps(to_serializable(first.model_evaluations), sort_keys=True),
            json.dumps(to_serializable(second.model_evaluations), sort_keys=True),
        )
        self.assertEqual(first.selection_reason, second.selection_reason)
        self.assertEqual(
            first.best_model.best_baseline_model, second.best_model.best_baseline_model
        )


# --------------------------------------------------------------------------- #
# 21: unsupported / empty / degraded states
# --------------------------------------------------------------------------- #
class TestDegradedStates(unittest.TestCase):
    """Partial runs are reported honestly instead of being hidden."""

    def test_no_best_model_when_every_experiment_failed(self) -> None:
        state = hand_built_state()
        state.experiment_results = [
            ExperimentResult(
                experiment_id="exp_broken",
                model_name="Random Forest",
                status=ExperimentStatus.FAILED.value,
                error_message="ValueError: boom",
            )
        ]
        state.cross_validation_results = []
        report = EvaluationAgent().run(state).evaluation_report
        self.assertIsNone(report.best_model)
        self.assertEqual(report.overall_status, EvaluationStatus.FAILED.value)
        self.assertIn("no baseline model is recommended", report.selection_reason.lower())
        self.assertIn("boom", json.dumps(to_serializable(report)))

    def test_failed_experiment_keeps_its_error_visible(self) -> None:
        state = hand_built_state()
        state.experiment_results.append(
            ExperimentResult(
                experiment_id="exp_broken",
                model_name="Random Forest",
                status=ExperimentStatus.FAILED.value,
                error_message="ValueError: boom",
            )
        )
        report = EvaluationAgent().run(state).evaluation_report
        self.assertEqual(report.overall_status, EvaluationStatus.PARTIAL.value)
        failed = [item for item in report.model_evaluations if item.rank is None]
        self.assertEqual(len(failed), 1)
        self.assertIn("ValueError: boom", failed[0].notes[0])

    def test_missing_cv_result_is_recorded_as_a_limitation(self) -> None:
        state = hand_built_state()
        state.cross_validation_results = []
        # Phase 4 stores the means twice; a run without cross-validation has
        # neither copy.
        state.experiment_results[0].cv_f1_mean = None
        state.experiment_results[0].cv_accuracy_mean = None
        report = EvaluationAgent().run(state).evaluation_report
        self.assertIsNone(report.model_evaluations[0].f1_gap)
        self.assertTrue(
            any("no cross-validation result" in item.lower() for item in report.limitations)
        )

    def test_missing_confusion_matrix_is_reported(self) -> None:
        state = hand_built_state()
        state.experiment_results[0].confusion_matrix = None
        report = EvaluationAgent().run(state).evaluation_report
        analysis = report.confusion_matrix_analysis[0]
        self.assertFalse(analysis.available)
        self.assertEqual(
            analysis.per_class_metrics_note, config.PER_CLASS_UNAVAILABLE_NOTE
        )
        self.assertIn(config.PER_CLASS_UNAVAILABLE_NOTE, report.limitations)

    def test_missing_model_comparison_is_rebuilt(self) -> None:
        state = trained_state()
        state.model_comparison = None
        report = EvaluationAgent().run(state).evaluation_report
        self.assertIn("rebuilt by the Evaluation Agent", report.ranking_rule)
        self.assertIsNotNone(report.best_model)

    def test_malformed_confusion_matrix_fails_loudly(self) -> None:
        state = hand_built_state()
        state.experiment_results[0].confusion_matrix = ConfusionMatrixData(
            labels=["0", "1"], matrix=[[8, 1, 0], [2, 13, 0]], total=24
        )
        with self.assertRaises(EvaluationDataError):
            EvaluationAgent().run(state)
        self.assertEqual(state.evaluation_status, EvaluationStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_cross_validation_disabled_run_is_still_evaluated(self) -> None:
        state = trained_state(training_kwargs={"cross_validation": False})
        report = EvaluationAgent().run(state).evaluation_report
        self.assertIsNotNone(report.best_model)
        self.assertIsNone(report.model_evaluations[0].f1_gap)
        self.assertEqual(
            report.model_evaluations[0].f1_gap_classification,
            GapClassification.NOT_AVAILABLE.value,
        )
        self.assertIn("models_without_gap", json.dumps(to_serializable(report.gap_analysis)))

    def test_empty_experiment_list_after_training_is_handled(self) -> None:
        """Defensive: a completed training with no record yields a skipped report."""
        state = hand_built_state()
        state.experiment_results = []
        with self.assertRaises(PreprocessingError):
            EvaluationAgent().run(state)


# --------------------------------------------------------------------------- #
# 19: JSON safety
# --------------------------------------------------------------------------- #
class TestJsonSafety(unittest.TestCase):
    """The report and the state stay serialisable without any fallback."""

    def test_report_is_json_serialisable(self) -> None:
        state = trained_state()
        report = EvaluationAgent().run(state).evaluation_report
        payload = json.dumps(to_serializable(report))  # plain dumps, no default=
        self.assertIsInstance(json.loads(payload), dict)

    def test_state_is_json_serialisable(self) -> None:
        state = EvaluationAgent().run(trained_state())
        payload = json.loads(json.dumps(state.to_dict()))
        self.assertIn("evaluation_report", payload)
        self.assertIn("evaluation_status", payload)

    def test_state_stores_no_live_object_from_phase_five(self) -> None:
        state = EvaluationAgent().run(trained_state())
        runtime = state.to_dict().get("runtime_objects", {})
        self.assertNotIn("evaluation_report", runtime)
        self.assertEqual(
            sorted(runtime), ["best_pipeline", "preprocessing_pipeline"]
        )

    def test_report_contains_no_sklearn_object(self) -> None:
        state = EvaluationAgent().run(trained_state())
        rendered = json.dumps(state.to_dict()["evaluation_report"])
        for forbidden in ("Pipeline(", "ColumnTransformer", "LogisticRegression"):
            self.assertNotIn(forbidden, rendered)


# --------------------------------------------------------------------------- #
# 19/22: reproducibility and scientific wording
# --------------------------------------------------------------------------- #
class TestReproducibilityAndWording(unittest.TestCase):
    """Two identical runs must produce identical reports and honest text."""

    def test_two_runs_agree_on_every_decision(self) -> None:
        first = EvaluationAgent().run(trained_state()).evaluation_report
        second = EvaluationAgent().run(trained_state()).evaluation_report
        self.assertEqual(first.best_model.best_baseline_model, second.best_model.best_baseline_model)
        self.assertEqual(first.ranking_rule, second.ranking_rule)
        self.assertEqual(
            [item.model_name for item in first.model_evaluations],
            [item.model_name for item in second.model_evaluations],
        )
        self.assertEqual(first.gap_analysis.summary, second.gap_analysis.summary)
        self.assertEqual(first.experiment_count.total_experiments, second.experiment_count.total_experiments)

    def test_ranking_matches_the_phase_four_winner(self) -> None:
        state = trained_state()
        report = EvaluationAgent().run(state).evaluation_report
        self.assertEqual(
            report.best_model.best_baseline_model, state.model_comparison.best_model_name
        )
        self.assertEqual(report.best_model.best_baseline_model, state.best_baseline_model.model_name)

    def test_no_overfitting_verdict_in_the_report(self) -> None:
        state = trained_state()
        report = EvaluationAgent().run(state).evaluation_report
        rendered = json.dumps(to_serializable(report)).lower()
        # Only affirmative claims are forbidden. The patterns are specific on
        # purpose: the disclaimer has to be able to say that a gap "neither
        # proves nor disproves" overfitting, so the bare word "proves" is not a
        # violation while "proves that" is.
        forbidden = (
            r"\bis overfitting\b",
            r"\bproves that\b",
            r"\bconfirms that\b",
            r"\bshows that\b",
            r"\bdefinitely\b",
            r"\bguarantee\w*\b",
            r"\bwill work perfectly\b",
            r"\bis perfectly\b",
        )
        for pattern in forbidden:
            self.assertIsNone(
                re.search(pattern, rendered), f"unsupported claim matching {pattern}"
            )

    def test_disclaimer_explicitly_denies_proof(self) -> None:
        state = trained_state()
        report = EvaluationAgent().run(state).evaluation_report
        self.assertIn(
            "neither proves nor disproves", report.gap_analysis.disclaimer
        )

    def test_report_always_carries_the_disclaimer(self) -> None:
        report = EvaluationAgent().run(trained_state()).evaluation_report
        self.assertEqual(
            report.gap_analysis.disclaimer, config.GAP_HEURISTIC_DISCLAIMER
        )

    def test_report_states_the_baseline_caveat(self) -> None:
        report = EvaluationAgent().run(trained_state()).evaluation_report
        self.assertTrue(
            any("best BASELINE model" in item for item in report.recommendations)
        )

    def test_report_states_the_missing_training_metrics(self) -> None:
        report = EvaluationAgent().run(trained_state()).evaluation_report
        self.assertIn(config.LIMITATION_NO_TRAINING_METRICS, report.limitations)

    def test_logged_messages_make_no_unsupported_claim(self) -> None:
        state = EvaluationAgent().run(trained_state())
        text = " ".join(item.message for item in state.agent_messages).lower()
        for pattern in (r"\bis overfitting\b", r"\bdefinitely\b", r"\bwill work perfectly\b"):
            self.assertIsNone(re.search(pattern, text), pattern)


# --------------------------------------------------------------------------- #
# Integration: the full workflow on the committed datasets
# --------------------------------------------------------------------------- #
class TestBinaryIntegration(unittest.TestCase):
    """analysis -> preprocessing -> selection -> training -> evaluation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.before = file_hash(BINARY_DATASET)
        cls.state = EvaluationAgent().run(trained_state(BINARY_DATASET))
        cls.report = cls.state.evaluation_report

    def test_source_dataset_is_unchanged(self) -> None:
        self.assertEqual(file_hash(BINARY_DATASET), self.before)

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.state.evaluation_status, EvaluationStatus.COMPLETED.value)

    def test_every_baseline_model_was_evaluated(self) -> None:
        self.assertEqual(len(self.report.model_evaluations), 5)
        self.assertEqual(self.report.experiment_count.baseline_models_attempted, 5)
        self.assertEqual(self.report.experiment_count.failed_experiments, 0)

    def test_a_best_model_is_generated(self) -> None:
        self.assertIsNotNone(self.report.best_model)
        self.assertEqual(
            self.report.best_model.best_baseline_model, "Logistic Regression"
        )

    def test_gap_analysis_is_generated_for_every_model(self) -> None:
        self.assertEqual(self.report.gap_analysis.models_with_gap, 5)
        for item in self.report.model_evaluations:
            self.assertIsNotNone(item.f1_gap)
            self.assertIsNotNone(item.accuracy_gap)
            self.assertNotEqual(
                item.f1_gap_classification, GapClassification.NOT_AVAILABLE.value
            )

    def test_confusion_matrix_analysis_is_generated_for_every_model(self) -> None:
        self.assertEqual(len(self.report.confusion_matrix_analysis), 5)
        for analysis in self.report.confusion_matrix_analysis:
            self.assertTrue(analysis.available)
            self.assertEqual(len(analysis.class_metrics), 2)
            self.assertEqual(
                analysis.correct_predictions + analysis.incorrect_predictions,
                analysis.total_samples,
            )

    def test_derived_per_class_metrics_reproduce_the_weighted_f1(self) -> None:
        """The derived per-class F1 values must agree with the recorded metric."""
        by_name = {item.model_name: item for item in self.report.model_evaluations}
        for analysis in self.report.confusion_matrix_analysis:
            weighted = sum(
                item.f1_score * item.support / analysis.total_samples
                for item in analysis.class_metrics
            )
            self.assertAlmostEqual(
                weighted, by_name[analysis.model_name].f1_score, places=3
            )

    def test_execution_time_is_calculated(self) -> None:
        timing = self.report.execution_time_analysis
        self.assertGreater(timing.total_experiment_time, 0.0)
        self.assertEqual(timing.timed_experiments, 5)
        self.assertIn(timing.fastest_model, {item.model_name for item in self.report.model_evaluations})
        self.assertIn(timing.slowest_model, {item.model_name for item in self.report.model_evaluations})
        self.assertLessEqual(timing.fastest_time, timing.slowest_time)

    def test_experiment_count_is_calculated(self) -> None:
        counts = self.report.experiment_count
        self.assertEqual(counts.total_experiments, 5)
        self.assertEqual(counts.successful_experiments, 5)
        self.assertEqual(counts.hyperparameter_experiments, 0)

    def test_observed_values_match_the_training_records(self) -> None:
        for experiment in self.state.experiment_results:
            evaluation = next(
                item
                for item in self.report.model_evaluations
                if item.model_name == experiment.model_name
            )
            self.assertEqual(evaluation.f1_score, experiment.f1_score)
            self.assertEqual(evaluation.accuracy, experiment.accuracy)
            self.assertEqual(evaluation.cv_f1_mean, experiment.cv_f1_mean)

    def test_best_model_matches_phase_four(self) -> None:
        self.assertEqual(
            self.report.best_model.best_baseline_model,
            self.state.best_baseline_model.model_name,
        )


class TestMulticlassIntegration(unittest.TestCase):
    """The same workflow on a three-class dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.before = file_hash(MULTICLASS_DATASET)
        cls.state = EvaluationAgent().run(trained_state(MULTICLASS_DATASET))
        cls.report = cls.state.evaluation_report

    def test_source_dataset_is_unchanged(self) -> None:
        self.assertEqual(file_hash(MULTICLASS_DATASET), self.before)

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.state.evaluation_status, EvaluationStatus.COMPLETED.value)

    def test_a_best_model_is_generated(self) -> None:
        self.assertIsNotNone(self.report.best_model)
        self.assertIn(self.report.best_model.best_baseline_model,
                      {"Decision Tree", "Random Forest"})

    def test_gap_analysis_is_generated(self) -> None:
        self.assertEqual(self.report.gap_analysis.models_with_gap, 5)

    def test_confusion_matrix_analysis_handles_three_classes(self) -> None:
        self.assertEqual(len(self.report.confusion_matrix_analysis), 5)
        for analysis in self.report.confusion_matrix_analysis:
            self.assertEqual(len(analysis.labels), 3)
            self.assertEqual(len(analysis.class_metrics), 3)
            self.assertEqual(
                sum(item.support for item in analysis.class_metrics),
                analysis.total_samples,
            )

    def test_multiclass_pattern_names_the_hardest_classes(self) -> None:
        texts = [
            analysis.error_pattern
            for analysis in self.report.confusion_matrix_analysis
            if analysis.incorrect_predictions
        ]
        self.assertTrue(texts)
        for text in texts:
            self.assertIn("classes involved in the most errors", text)

    def test_execution_time_and_count_are_calculated(self) -> None:
        self.assertGreater(self.report.execution_time_analysis.total_experiment_time, 0.0)
        self.assertEqual(self.report.experiment_count.total_experiments, 5)

    def test_ranking_matches_phase_four(self) -> None:
        self.assertEqual(
            self.report.best_model.best_baseline_model,
            self.state.best_baseline_model.model_name,
        )


if __name__ == "__main__":
    unittest.main()
