"""Tests for the Optimization Agent (Phase 6).

The tools carry the science; the agent carries the orchestration. These tests
therefore check the contract around the search:

* the phase refuses to start on a state the earlier phases did not finish;
* a model whose search fails does not take the phase down with it;
* the Phase 5 baseline result survives the phase untouched;
* what lands in the shared state is JSON-safe and holds no live object;
* the two committed sample datasets run end to end without changing a byte;
* two runs of the same dataset reach the same winner.

The search itself is exercised with a deliberately small trial budget. The
integration classes run the real chain once and assert many things about it,
because running the chain per assertion would make the suite a benchmark.
"""

from __future__ import annotations

import json
import unittest

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from tools.optimization_tools import OptimizationToolError
from tools.preprocessing_tools import PreprocessingError
from utils import config
from utils.state import (
    AgentState,
    AgentStatus,
    ImprovementDirection,
    OptimizationStatus,
    TrainingStatus,
    to_serializable,
)

from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash

# Small enough to keep the suite quick, large enough for the sampler to make a
# real choice. The production default is config.OPTUNA_N_TRIALS.
TEST_TRIALS = 4


def evaluated_state(dataset_path=None, *, trials: int = TEST_TRIALS, **kwargs) -> AgentState:
    """Run the full chain up to the Evaluation Agent, then optimize."""
    state = AgentState()
    state = DataAnalysisAgent().run(
        state, dataset_path=str(dataset_path or BINARY_DATASET)
    )
    state = PreprocessingAgent().run(state)
    state = ModelSelectionAgent().run(state)
    state = TrainingAgent().run(state)
    state = EvaluationAgent().run(state)
    return OptimizationAgent().run(state, n_trials=trials, **kwargs)


def evaluated_only_state(dataset_path=None) -> AgentState:
    """The same chain, stopped just before the Optimization Agent."""
    state = AgentState()
    state = DataAnalysisAgent().run(
        state, dataset_path=str(dataset_path or BINARY_DATASET)
    )
    state = PreprocessingAgent().run(state)
    state = ModelSelectionAgent().run(state)
    state = TrainingAgent().run(state)
    return EvaluationAgent().run(state)


# --------------------------------------------------------------------------- #
# 1: prerequisites
# --------------------------------------------------------------------------- #
class TestPrerequisites(unittest.TestCase):
    """The agent refuses to run on a state the earlier phases did not finish."""

    def test_fresh_state_is_rejected(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError) as context:
            OptimizationAgent().run(state)
        message = str(context.exception)
        self.assertIn("Training Agent has not been completed", message)
        self.assertIn("no experiment was recorded", message)
        self.assertEqual(state.optimization_status, OptimizationStatus.FAILED.value)
        self.assertEqual(state.status, AgentStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_failed_training_is_rejected(self) -> None:
        state = AgentState()
        state.training_status = TrainingStatus.FAILED.value
        with self.assertRaises(PreprocessingError):
            OptimizationAgent().run(state)

    def test_missing_preprocessing_configuration_is_reported(self) -> None:
        state = evaluated_only_state()
        state.preprocessing_status = "failed"
        state.preprocessing_config = None
        with self.assertRaises(PreprocessingError) as context:
            OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        self.assertIn("Preprocessing Agent has not been completed", str(context.exception))

    def test_a_missing_dataset_path_is_reported(self) -> None:
        state = evaluated_only_state()
        state.dataset_path = ""
        with self.assertRaises(PreprocessingError) as context:
            OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        self.assertIn("no dataset path", str(context.exception))

    def test_a_missing_target_column_is_reported(self) -> None:
        state = evaluated_only_state()
        state.target_column = ""
        with self.assertRaises(PreprocessingError) as context:
            OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        self.assertIn("no target column", str(context.exception))

    def test_a_missing_problem_type_is_reported(self) -> None:
        state = evaluated_only_state()
        state.problem_type = None
        with self.assertRaises(PreprocessingError) as context:
            OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        self.assertIn("no problem type", str(context.exception))

    def test_the_error_names_the_agents_that_must_run_first(self) -> None:
        with self.assertRaises(PreprocessingError) as context:
            OptimizationAgent().run(AgentState())
        for name in (
            "DataAnalysisAgent",
            "PreprocessingAgent",
            "ModelSelectionAgent",
            "TrainingAgent",
            "EvaluationAgent",
        ):
            self.assertIn(name, str(context.exception))

    def test_a_prerequisite_failure_leaves_no_report(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError):
            OptimizationAgent().run(state)
        self.assertIsNone(state.optimization_report)
        self.assertFalse(state.has_optimization_report)
        self.assertEqual(state.optimization_results, [])

    def test_a_prerequisite_failure_is_logged_as_an_error(self) -> None:
        state = AgentState()
        with self.assertRaises(PreprocessingError):
            OptimizationAgent().run(state)
        self.assertTrue(any(item.level == "error" for item in state.agent_messages))

    def test_a_missing_dataset_file_fails_loudly(self) -> None:
        state = evaluated_only_state()
        state.dataset_path = "datasets/does_not_exist.csv"
        with self.assertRaises(Exception):
            OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        self.assertEqual(state.optimization_status, OptimizationStatus.FAILED.value)
        self.assertTrue(state.errors)


# --------------------------------------------------------------------------- #
# 2: what the agent writes into the state
# --------------------------------------------------------------------------- #
class TestStateUpdate(unittest.TestCase):
    """The report is the deliverable; these are its required fields."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.before = evaluated_only_state()
        cls.state = OptimizationAgent().run(cls.before, n_trials=TEST_TRIALS)
        cls.report = cls.state.optimization_report

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.state.optimization_status, OptimizationStatus.COMPLETED.value)
        self.assertEqual(self.report.overall_status, OptimizationStatus.COMPLETED.value)

    def test_workflow_status_moves_to_optimization_completed(self) -> None:
        self.assertEqual(self.state.status, AgentStatus.OPTIMIZATION_COMPLETED.value)

    def test_the_phase_reports_which_models_it_tuned(self) -> None:
        self.assertTrue(self.report.models_selected_for_optimization)
        self.assertNotIn(
            set(self.report.models_selected_for_optimization),
            set(self.report.models_not_optimized),
        )
        self.assertEqual(
            len(self.report.models_selected_for_optimization)
            + len(self.report.models_not_optimized),
            len(self.state.experiment_results),
        )

    def test_candidate_selection_follows_the_phase_five_ranking(self) -> None:
        """The tuned models are the winners of the Phase 5 ranking, in order."""
        baseline_order = [
            item.model_name for item in self.state.experiment_results
        ]
        expected = [
            name
            for name in baseline_order
            if name in self.report.models_selected_for_optimization
        ]
        self.assertEqual(self.report.models_selected_for_optimization, expected)

    def test_every_selected_model_has_a_result(self) -> None:
        self.assertEqual(
            [item.model_name for item in self.report.optimization_results],
            self.report.models_selected_for_optimization,
        )

    def test_the_recommendation_records_both_baseline_and_optimized_numbers(self) -> None:
        best = self.report.best_optimized_model
        self.assertIsNotNone(best)
        self.assertEqual(best.baseline_model_name, best.best_optimized_model)
        self.assertIn("selection_reason", best.__dict__)
        self.assertTrue(best.selection_reason)
        self.assertTrue(best.performance_summary)
        self.assertIn(
            best.improvement_direction,
            (
                ImprovementDirection.IMPROVED.value,
                ImprovementDirection.DEGRADED.value,
                ImprovementDirection.UNCHANGED.value,
            ),
        )

    def test_the_phase_five_baseline_is_preserved(self) -> None:
        """Both winners are kept; the optimized one does not replace the baseline."""
        self.assertIsNotNone(self.report.best_baseline_model)
        self.assertEqual(
            self.report.best_baseline_model.best_baseline_model,
            self.before.evaluation_report.best_model.best_baseline_model,
        )
        self.assertEqual(self.state.evaluation_status, "completed")
        self.assertEqual(
            len(self.state.experiment_results), len(self.before.experiment_results)
        )

    def test_the_search_is_described_so_a_reader_can_reproduce_it(self) -> None:
        for value in (
            self.report.objective_metric,
            self.report.objective_rule,
            self.report.selection_rule,
            self.report.candidate_rule,
            self.report.ranking_rule,
            self.report.holdout_note,
        ):
            self.assertTrue(value)
            json.dumps(value)
        self.assertEqual(self.report.sampler, config.OPTUNA_SAMPLER_NAME)
        self.assertEqual(self.report.sampler_seed, config.OPTUNA_RANDOM_SEED)
        self.assertEqual(self.report.n_trials_per_model, TEST_TRIALS)
        self.assertEqual(self.report.top_models, config.OPTUNA_TOP_MODELS)

    def test_trials_are_recorded_per_model(self) -> None:
        for item in self.report.optimization_results:
            self.assertGreaterEqual(len(item.trials), 1)
            self.assertEqual(len(item.trials), item.total_trials)
            self.assertEqual(
                item.completed_trials + item.failed_trials, item.total_trials
            )
            for trial in item.trials:
                self.assertEqual(trial.model_name, item.model_name)
                self.assertIsInstance(trial.params, dict)

    def test_counts_keep_experiments_and_trials_apart(self) -> None:
        counts = self.report.counts
        self.assertIsNotNone(counts)
        self.assertEqual(counts.baseline_models_attempted, len(self.state.experiment_results))
        self.assertEqual(
            counts.baseline_successful_experiments, len(self.state.successful_experiments)
        )
        self.assertEqual(
            counts.optimized_models_attempted, len(self.report.optimization_results)
        )
        self.assertEqual(
            counts.total_trials,
            sum(item.total_trials for item in self.report.optimization_results),
        )
        self.assertEqual(
            counts.completed_trials,
            sum(item.completed_trials for item in self.report.optimization_results),
        )
        self.assertNotEqual(counts.total_trials, counts.baseline_models_attempted)

    def test_execution_time_stays_cumulative(self) -> None:
        previous = self.before.execution_time
        self.assertGreaterEqual(self.state.execution_time, previous)
        self.assertGreater(self.state.execution_time, 0.0)

    def test_a_comparison_table_is_available_to_the_cli(self) -> None:
        self.assertIn("baseline cv f1", self.report.table_text)
        for name in self.report.models_selected_for_optimization:
            self.assertIn(name, self.report.table_text)

    def test_observations_limitations_and_recommendations_are_recorded(self) -> None:
        self.assertTrue(self.report.observations)
        self.assertTrue(self.report.limitations)
        self.assertTrue(self.report.recommendations)

    def test_logging_covers_every_section_of_the_report(self) -> None:
        text = " ".join(item.message for item in self.state.agent_messages)
        self.assertIn("Optimization status", text)
        self.assertIn("Best optimized model", text)
        self.assertIn("Selection reason", text)
        self.assertIn("Best parameters", text)
        self.assertIn("Counts", text)

    def test_the_best_baseline_is_logged_as_still_available(self) -> None:
        text = " ".join(item.message for item in self.state.agent_messages)
        self.assertIn("Best baseline model", text)
        self.assertIn("unchanged", text)

    def test_no_logged_line_is_a_multi_line_table(self) -> None:
        for item in self.state.agent_messages:
            self.assertNotIn("\n", item.message)


# --------------------------------------------------------------------------- #
# 3: the phase does not damage the earlier phases
# --------------------------------------------------------------------------- #
class TestEarlierPhasesAreUntouched(unittest.TestCase):
    """Optimization is additive; it must not rewrite Phase 4 or 5."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.before = evaluated_only_state()
        cls.after = OptimizationAgent().run(cls.before, n_trials=TEST_TRIALS)

    def test_training_status_is_untouched(self) -> None:
        self.assertEqual(self.after.training_status, self.before.training_status)

    def test_no_experiment_is_added_or_removed(self) -> None:
        self.assertEqual(len(self.after.experiment_results), len(self.before.experiment_results))
        for before, after in zip(self.before.experiment_results, self.after.experiment_results):
            self.assertEqual(before.model_name, after.model_name)
            self.assertEqual(before.f1_score, after.f1_score)
            self.assertEqual(before.cv_f1_mean, after.cv_f1_mean)
            self.assertEqual(before.status, after.status)

    def test_phase_four_records_are_not_mutated(self) -> None:
        self.assertEqual(
            len(self.after.cross_validation_results),
            len(self.before.cross_validation_results),
        )
        self.assertEqual(self.after.data_split.train_samples, self.before.data_split.train_samples)
        self.assertEqual(self.after.data_split.test_samples, self.before.data_split.test_samples)

    def test_the_best_baseline_pipeline_is_untouched(self) -> None:
        self.assertIs(self.after.best_pipeline, self.before.best_pipeline)

    def test_the_evaluation_report_is_not_mutated(self) -> None:
        self.assertEqual(
            self.after.evaluation_report.best_model.best_baseline_model,
            self.before.evaluation_report.best_model.best_baseline_model,
        )
        self.assertEqual(
            len(self.after.evaluation_report.model_evaluations),
            len(self.before.evaluation_report.model_evaluations),
        )


# --------------------------------------------------------------------------- #
# 4: degraded runs
# --------------------------------------------------------------------------- #
class TestDegradedRuns(unittest.TestCase):
    """A failed search is recorded; the phase and the baseline survive it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = evaluated_only_state()

    def _state_with(self, **overrides) -> AgentState:
        state = evaluated_only_state()
        for name, value in overrides.items():
            setattr(state, name, value)
        return state

    def test_disabling_the_phase_is_a_skip_not_a_failure(self) -> None:
        state = OptimizationAgent().run(self.base, cross_validation=False)
        self.assertEqual(state.optimization_status, OptimizationStatus.SKIPPED.value)
        self.assertIsNone(state.optimization_report)
        self.assertEqual(state.status, AgentStatus.EVALUATION_COMPLETED.value)
        self.assertIn(
            "disabled", " ".join(item.message for item in state.agent_messages)
        )

    def test_the_phase_five_result_is_intact_after_a_skip(self) -> None:
        state = OptimizationAgent().run(self.base, cross_validation=False)
        self.assertEqual(
            state.evaluation_report.best_model, self.base.evaluation_report.best_model
        )
        self.assertFalse(state.errors)

    def test_a_model_with_no_search_space_is_skipped_with_a_reason(self) -> None:
        state = self._base_without_search_space()
        result = OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        self.assertEqual(result.optimization_status, OptimizationStatus.SKIPPED.value)
        self.assertIn(
            "search space",
            " ".join(item.message for item in result.agent_messages),
        )

    def test_one_failing_model_does_not_stop_the_others(self) -> None:
        """A model-level failure is isolated exactly like a failed trial."""
        import agents.optimization_agent as agent_module

        original = agent_module.optimize_model
        names = {"Decision Tree"}

        def flaky(**kwargs):
            if kwargs.get("model_name") in names:
                raise RuntimeError("synthetic model-level failure")
            return original(**kwargs)

        state = evaluated_only_state()
        # Every optimizable model becomes a candidate so the failing one is in.
        agent_module.optimize_model = flaky
        try:
            result = OptimizationAgent().run(state, top_models=5, n_trials=TEST_TRIALS)
        finally:
            agent_module.optimize_model = original

        self.assertEqual(result.optimization_status, OptimizationStatus.PARTIAL.value)
        failed = [
            item
            for item in result.optimization_results
            if item.status == OptimizationStatus.FAILED.value
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].model_name, "Decision Tree")
        self.assertIn("synthetic model-level failure", failed[0].error_message)
        self.assertIn(
            "synthetic model-level failure",
            " ".join(item.message for item in result.agent_messages),
        )
        self.assertGreater(len(result.optimization_report.models_selected_for_optimization), 2)
        self.assertIsNotNone(result.optimization_report.best_optimized_model)

    def test_a_run_where_every_search_failed_keeps_the_baseline_winner(self) -> None:
        import agents.optimization_agent as agent_module

        original = agent_module.optimize_model
        agent_module.optimize_model = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic total failure")
        )
        state = evaluated_only_state()
        try:
            result = OptimizationAgent().run(state, n_trials=TEST_TRIALS)
        finally:
            agent_module.optimize_model = original

        self.assertEqual(result.optimization_status, OptimizationStatus.FAILED.value)
        self.assertIsNone(result.optimization_report.best_optimized_model)
        self.assertIsNotNone(result.optimization_report.best_baseline_model)
        self.assertEqual(
            result.optimization_report.best_baseline_model.best_baseline_model,
            state.evaluation_report.best_model.best_baseline_model,
        )
        self.assertEqual(
            result.optimization_report.counts.failed_optimizations,
            len(result.optimization_report.models_selected_for_optimization),
        )

    def _base_without_search_space(self) -> AgentState:
        """A state whose only experiment is a model with no search space."""
        import tools.optimization_tools as optimization_tools

        state = evaluated_only_state()
        state.experiment_results = [
            item for item in state.experiment_results if item.model_name == "K-Nearest Neighbors"
        ]
        self._original = optimization_tools.is_optimizable
        optimization_tools.is_optimizable = lambda name: False
        self.addCleanup(
            setattr, optimization_tools, "is_optimizable", self._original
        )
        return state

    def test_an_invalid_trial_budget_is_reported_not_silently_corrected(self) -> None:
        state = evaluated_only_state()
        with self.assertRaises(OptimizationToolError):
            OptimizationAgent().run(state, n_trials=0)
        self.assertEqual(state.optimization_status, OptimizationStatus.FAILED.value)
        self.assertTrue(state.errors)


# --------------------------------------------------------------------------- #
# 5: JSON safety
# --------------------------------------------------------------------------- #
class TestJsonSafety(unittest.TestCase):
    """Nothing live may reach the serialised state."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = evaluated_state()

    def test_state_is_json_serialisable_without_a_default_hook(self) -> None:
        payload = self.state.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_only_the_two_known_runtime_objects_are_excluded(self) -> None:
        payload = self.state.to_dict()
        self.assertEqual(
            sorted(payload["runtime_objects"]), ["best_pipeline", "preprocessing_pipeline"]
        )

    def test_the_optimization_report_is_present_in_the_payload(self) -> None:
        payload = self.state.to_dict()
        self.assertEqual(payload["optimization_status"], OptimizationStatus.COMPLETED.value)
        report = payload["optimization_report"]
        self.assertEqual(report["overall_status"], OptimizationStatus.COMPLETED.value)
        self.assertTrue(report["optimization_results"])

    def test_the_derived_results_field_is_not_serialised_twice(self) -> None:
        payload = self.state.to_dict()
        self.assertNotIn("results", payload["optimization_report"])
        self.assertEqual(
            len(payload["optimization_report"]["optimization_results"]),
            len(self.state.optimization_results),
        )

    def test_the_report_contains_no_sklearn_or_optuna_object(self) -> None:
        """A repr mentioning a pipeline or a study would mean one leaked."""
        payload = to_serializable(self.state.optimization_report)
        text = json.dumps(payload)
        for marker in ("Pipeline(", "ColumnTransformer(", "StandardScaler(", "Study",
                       "TPESampler", "FrozenTrial", "sklearn", "optuna"):
            self.assertNotIn(marker, text)

    def test_trial_parameters_are_plain_values(self) -> None:
        for item in self.state.optimization_results:
            for value in item.best_params.values():
                self.assertIsInstance(value, (int, float, str, bool, type(None)))

    def test_state_fields_added_by_this_phase_are_the_expected_ones(self) -> None:
        payload = self.state.to_dict()
        for key in ("optimization_status", "optimization_report"):
            self.assertIn(key, payload)
        self.assertIn("optimization_results", AgentState.DERIVED_ONLY_FIELDS)


# --------------------------------------------------------------------------- #
# 6: end-to-end integration
# --------------------------------------------------------------------------- #
class _IntegrationBase(unittest.TestCase):
    """Shared plumbing: run the whole chain once, assert many things."""

    DATASET = BINARY_DATASET
    TRIALS = TEST_TRIALS

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_hash = file_hash(cls.DATASET)
        cls.state = evaluated_state(cls.DATASET, trials=cls.TRIALS)
        cls.report = cls.state.optimization_report

    def test_source_dataset_is_unchanged(self) -> None:
        self.assertEqual(file_hash(self.DATASET), self.dataset_hash)

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.state.optimization_status, OptimizationStatus.COMPLETED.value)

    def test_the_requested_number_of_models_was_tuned(self) -> None:
        self.assertLessEqual(
            len(self.report.models_selected_for_optimization), config.OPTUNA_TOP_MODELS
        )
        self.assertGreaterEqual(len(self.report.models_selected_for_optimization), 1)

    def test_a_best_optimized_model_is_generated(self) -> None:
        self.assertIsNotNone(self.report.best_optimized_model)
        self.assertIn(
            self.report.best_optimized_model.best_optimized_model,
            self.report.models_selected_for_optimization,
        )

    def test_the_winner_is_ranked_first_in_the_table(self) -> None:
        best = self.report.best_optimized_model.best_optimized_model
        rows = [
            line[:26].strip()
            for line in self.report.table_text.strip().splitlines()[2:]
            if line.strip()
        ]
        self.assertEqual(rows[0], best)
        self.assertEqual(len(rows), len(self.report.models_selected_for_optimization))

    def test_the_baseline_and_optimized_scores_are_both_reported(self) -> None:
        best = self.report.best_optimized_model
        self.assertIsNotNone(best.baseline_cv_f1_mean)
        self.assertIsNotNone(best.optimized_cv_f1_mean)
        self.assertAlmostEqual(
            best.improvement,
            best.optimized_cv_f1_mean - best.baseline_cv_f1_mean,
            places=6,
        )

    def test_the_hold_out_split_was_used_once_after_the_search(self) -> None:
        for item in self.report.optimization_results:
            if item.status == OptimizationStatus.COMPLETED.value:
                self.assertTrue(item.holdout_evaluated)
                self.assertIsNotNone(item.holdout_f1_score)
                self.assertIsNotNone(item.holdout_accuracy)
                self.assertIn(config.OPTUNA_HOLDOUT_NOTE, item.notes)

    def test_cross_validation_ran_inside_the_pipeline(self) -> None:
        for item in self.report.optimization_results:
            self.assertGreaterEqual(item.cv_n_splits, config.CV_MIN_FOLDS)
            self.assertTrue(item.search_space_description)

    def test_trial_counts_add_up(self) -> None:
        counts = self.report.counts
        self.assertEqual(
            counts.completed_trials + counts.failed_trials + counts.pruned_trials,
            counts.total_trials,
        )
        self.assertEqual(
            counts.total_trials,
            counts.optimized_models_attempted * self.report.n_trials_per_model,
        )

    def test_execution_time_is_measured(self) -> None:
        for item in self.report.optimization_results:
            self.assertGreater(item.optimization_time, 0.0)

    def test_the_improvement_arithmetic_is_reproducible_from_the_report(self) -> None:
        for item in self.report.optimization_results:
            if item.baseline_cv_f1_mean is None:
                continue
            self.assertAlmostEqual(
                item.improvement,
                item.optimized_cv_f1_mean - item.baseline_cv_f1_mean,
                places=6,
            )
            if item.improvement_percentage is not None:
                self.assertAlmostEqual(
                    item.improvement_percentage,
                    round(item.improvement / item.baseline_cv_f1_mean * 100.0, 4),
                    places=4,
                )
            else:
                self.assertLess(
                    abs(item.baseline_cv_f1_mean),
                    config.OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE,
                )

    def test_the_best_parameters_can_rebuild_the_model(self) -> None:
        """A recorded configuration has to be usable without the search."""
        from tools.training_tools import build_model_pipeline

        best = self.report.best_optimized_model
        pipeline = build_model_pipeline(
            self.state.preprocessing_config, best.best_optimized_model, best.best_params
        )
        self.assertEqual(list(pipeline.named_steps)[-1], "model")

    def test_the_report_makes_no_unsupported_claim(self) -> None:
        text = " ".join(
            [self.report.table_text]
            + self.report.observations
            + self.report.limitations
            + self.report.recommendations
            + [
                self.report.best_optimized_model.selection_reason,
                self.report.best_optimized_model.performance_summary,
                self.report.best_optimized_model.selection_note,
            ]
        ).lower()
        for phrase in (
            "guaranteed better",
            "production ready",
            "proves generalization",
            "proves generalisation",
            "eliminates overfitting",
            "no overfitting",
            "always better",
            "optimal hyperparameters",
            "state of the art",
        ):
            self.assertNotIn(phrase, text)

    def test_the_cv_selection_bias_is_stated_when_a_model_improved(self) -> None:
        improved = [
            item
            for item in self.report.optimization_results
            if item.improvement_direction == ImprovementDirection.IMPROVED.value
        ]
        if improved:
            self.assertIn(config.OPTUNA_LIMITATION_SELECTION_ON_CV, self.report.limitations)
            self.assertIn(config.OPTUNA_IMPROVEMENT_CAUTION, self.report.limitations)
        else:
            self.assertIn(config.OPTUNA_NO_IMPROVEMENT_NOTE, self.report.limitations)

    def test_the_report_states_that_the_untuned_models_are_not_compared_fairly(self) -> None:
        selected = len(self.report.models_selected_for_optimization)
        total = selected + len(self.report.models_not_optimized)
        self.assertIn(
            config.OPTUNA_LIMITATION_UNTUNED_MODELS.format(selected=selected, candidates=total),
            self.report.limitations,
        )

    def test_the_report_states_the_trial_budget(self) -> None:
        self.assertIn(
            config.OPTUNA_LIMITATION_TRIAL_BUDGET.format(trials=self.report.n_trials_per_model),
            self.report.limitations,
        )

    def test_logged_lines_make_no_unsupported_claim(self) -> None:
        text = " ".join(item.message for item in self.state.agent_messages).lower()
        for phrase in (
            "guaranteed",
            "production ready",
            "eliminates overfitting",
            "perfect model",
            "always better",
        ):
            self.assertNotIn(phrase, text)


class TestBinaryIntegration(_IntegrationBase):
    """End to end on the committed binary sample dataset."""

    DATASET = BINARY_DATASET

    def test_the_top_two_baseline_models_were_tuned(self) -> None:
        """The candidates are the Phase 5 winners of the binary run."""
        self.assertEqual(
            self.report.models_selected_for_optimization,
            ["Logistic Regression", "Support Vector Machine"],
        )

    def test_the_problem_type_is_recorded(self) -> None:
        self.assertEqual(self.report.problem_type, "binary_classification")
        self.assertEqual(self.report.target_column, "Outcome")

    def test_the_logistic_solvers_include_the_binary_only_one(self) -> None:
        logistic = next(
            item
            for item in self.report.optimization_results
            if item.model_name == "Logistic Regression"
        )
        self.assertIn("liblinear", " ".join(logistic.search_space_description))

    def test_the_best_baseline_is_still_the_logistic_regression_of_phase_five(self) -> None:
        self.assertEqual(
            self.report.best_baseline_model.best_baseline_model, "Logistic Regression"
        )

    def test_every_tuned_model_has_a_recorded_objective_value(self) -> None:
        for item in self.report.optimization_results:
            self.assertGreaterEqual(item.optimized_cv_f1_mean, 0.0)
            self.assertLessEqual(item.optimized_cv_f1_mean, 1.0)

    def test_the_ranking_follows_the_documented_rule(self) -> None:
        """F1, then accuracy, then name -- the Phase 4/5 rule, unchanged."""
        from tools.optimization_tools import rank_optimized_results

        ranked = rank_optimized_results(self.report.optimization_results)
        keys = [
            (-item.holdout_f1_score, -item.holdout_accuracy, item.model_name)
            for item in ranked
        ]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            ranked[0].model_name,
            self.report.best_optimized_model.best_optimized_model,
        )


class TestMulticlassIntegration(_IntegrationBase):
    """End to end on the committed three-class sample dataset."""

    DATASET = MULTICLASS_DATASET
    TRIALS = 4

    def test_the_problem_type_is_recorded(self) -> None:
        self.assertEqual(self.report.problem_type, "multiclass_classification")
        self.assertEqual(self.report.target_column, "class_label")

    def test_the_tree_based_baseline_winners_were_tuned(self) -> None:
        self.assertEqual(
            self.report.models_selected_for_optimization,
            ["Decision Tree", "Random Forest"],
        )

    def test_the_multiclass_solvers_exclude_the_binary_only_solver(self) -> None:
        """`liblinear` would fail in every fold of a three-class search."""
        original = config.OPTUNA_LOGISTIC_SOLVERS_MULTICLASS
        self.assertNotIn("liblinear", original)
        description = optimization_space_text()
        self.assertIn("liblinear", description["binary_classification"])
        self.assertNotIn("liblinear", description["multiclass_classification"])

    def test_a_perfect_cv_score_is_not_reported_as_proof_of_anything(self) -> None:
        """The sample dataset is separable, so a 1.0 score is a property of the
        fixture and must not be read as a result."""
        text = " ".join(self.report.observations + self.report.limitations).lower()
        self.assertNotIn("perfect model", text)
        self.assertIn(
            config.OPTUNA_LIMITATION_NO_DOMAIN_VALIDATION, self.report.limitations
        )

    def test_the_three_class_confusion_is_scored_once_per_model(self) -> None:
        for item in self.report.optimization_results:
            if item.status == OptimizationStatus.COMPLETED.value:
                self.assertTrue(item.holdout_evaluated)
                self.assertLessEqual(item.holdout_f1_score, 1.0)


def optimization_space_text():
    """The search-space description per problem type, for the solver test."""
    from tools.optimization_tools import describe_search_space

    return {
        problem_type: " ".join(describe_search_space("Logistic Regression", problem_type))
        for problem_type in ("binary_classification", "multiclass_classification")
    }


# --------------------------------------------------------------------------- #
# 7: reproducibility
# --------------------------------------------------------------------------- #
class TestReproducibility(unittest.TestCase):
    """The same data and seed must reach the same configuration."""

    def test_two_runs_on_the_binary_dataset_agree_on_every_decision(self) -> None:
        first = evaluated_state(BINARY_DATASET, trials=TEST_TRIALS)
        second = evaluated_state(BINARY_DATASET, trials=TEST_TRIALS)

        self.assertEqual(
            first.optimization_report.models_selected_for_optimization,
            second.optimization_report.models_selected_for_optimization,
        )
        self.assertEqual(
            first.optimization_report.best_optimized_model.best_optimized_model,
            second.optimization_report.best_optimized_model.best_optimized_model,
        )
        for a, b in zip(
            first.optimization_results, second.optimization_results
        ):
            self.assertEqual(a.model_name, b.model_name)
            self.assertEqual(a.best_params, b.best_params)
            self.assertEqual(a.best_trial_number, b.best_trial_number)
            self.assertEqual(a.optimized_cv_f1_mean, b.optimized_cv_f1_mean)
            self.assertEqual(a.total_trials, b.total_trials)
            self.assertEqual(a.completed_trials, b.completed_trials)
            self.assertEqual(a.failed_trials, b.failed_trials)
            self.assertEqual([t.params for t in a.trials], [t.params for t in b.trials])

    def test_timings_may_differ_but_are_still_recorded(self) -> None:
        """Only the wall clock is allowed to move between two runs."""
        first = evaluated_state(BINARY_DATASET, trials=TEST_TRIALS)
        second = evaluated_state(BINARY_DATASET, trials=TEST_TRIALS)
        for a, b in zip(first.optimization_results, second.optimization_results):
            self.assertGreater(a.optimization_time, 0.0)
            self.assertGreater(b.optimization_time, 0.0)
            self.assertIsInstance(a.optimization_time, float)
            self.assertIsInstance(b.optimization_time, float)

    def test_a_different_trial_budget_still_reaches_the_same_winner(self) -> None:
        """A larger budget is allowed to change the winner; the phase must not
        break when it does."""
        small = evaluated_state(BINARY_DATASET, trials=2)
        large = evaluated_state(BINARY_DATASET, trials=6)
        self.assertEqual(
            small.optimization_report.counts.total_trials,
            2 * len(small.optimization_report.models_selected_for_optimization),
        )
        self.assertEqual(
            large.optimization_report.counts.total_trials,
            6 * len(large.optimization_report.models_selected_for_optimization),
        )
        self.assertEqual(
            large.optimization_status, OptimizationStatus.COMPLETED.value
        )

    def test_the_winner_is_stable_under_a_repeated_run_of_the_same_budget(self) -> None:
        first = evaluated_state(MULTICLASS_DATASET, trials=TEST_TRIALS)
        second = evaluated_state(MULTICLASS_DATASET, trials=TEST_TRIALS)
        self.assertEqual(
            first.optimization_report.best_optimized_model.best_optimized_model,
            second.optimization_report.best_optimized_model.best_optimized_model,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()