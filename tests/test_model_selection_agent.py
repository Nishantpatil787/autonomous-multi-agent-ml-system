"""Tests for the Model Selection Agent (Phase 4).

The agent's whole job is to decide *which* models will be trained and to make
that decision explainable. These tests check the prerequisite verification, the
selected models, the recorded reasons, the skip behaviour and the JSON-safety of
what it writes into the shared state.
"""

from __future__ import annotations

import json
import unittest

from agents.data_analysis_agent import DataAnalysisAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from tools.model_tools import ModelConfigurationError, ModelSelectionError
from tools.preprocessing_tools import PreprocessingError
from utils.state import (
    AgentState,
    AgentStatus,
    PreprocessingStatus,
    ProblemType,
    TrainingStatus,
)

from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET

EXPECTED_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Support Vector Machine",
]

SCALING_REQUIRED = {"Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"}
SCALING_NOT_REQUIRED = {"Decision Tree", "Random Forest"}


def state_after_preprocessing(dataset_path=None) -> AgentState:
    """Run the analysis and preprocessing agents for a ready-to-use state."""
    state = AgentState()
    state = DataAnalysisAgent().run(
        state, dataset_path=str(dataset_path or BINARY_DATASET)
    )
    return PreprocessingAgent().run(state)


def bare_state() -> AgentState:
    """A fresh state with nothing done yet."""
    return AgentState()


class TestPrerequisites(unittest.TestCase):
    """The agent refuses to run before the earlier phases have completed."""

    def test_fresh_state_is_rejected(self) -> None:
        state = bare_state()
        with self.assertRaises(PreprocessingError) as context:
            ModelSelectionAgent().run(state)
        message = str(context.exception)
        self.assertIn("Data Analysis Agent", message)
        self.assertIn("Preprocessing Agent", message)
        self.assertEqual(state.model_selection_status, TrainingStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_analysis_only_is_rejected(self) -> None:
        state = DataAnalysisAgent().run(bare_state(), dataset_path=str(BINARY_DATASET))
        with self.assertRaises(PreprocessingError) as context:
            ModelSelectionAgent().run(state)
        self.assertIn("Preprocessing Agent", str(context.exception))

    def test_missing_preprocessing_config_is_rejected(self) -> None:
        state = state_after_preprocessing()
        state.preprocessing_config = None
        with self.assertRaises(PreprocessingError) as context:
            ModelSelectionAgent().run(state)
        self.assertIn("preprocessing configuration", str(context.exception))

    def test_missing_target_is_rejected(self) -> None:
        state = state_after_preprocessing()
        state.target_column = None
        with self.assertRaises(PreprocessingError) as context:
            ModelSelectionAgent().run(state)
        self.assertIn("target", str(context.exception).lower())

    def test_missing_problem_type_is_rejected(self) -> None:
        state = state_after_preprocessing()
        state.problem_type = None
        with self.assertRaises(PreprocessingError):
            ModelSelectionAgent().run(state)

    def test_skipped_preprocessing_is_rejected(self) -> None:
        state = state_after_preprocessing()
        state.preprocessing_status = PreprocessingStatus.SKIPPED.value
        state.preprocessing_config = None
        with self.assertRaises(PreprocessingError) as context:
            ModelSelectionAgent().run(state)
        self.assertIn("Preprocessing Agent", str(context.exception))

    def test_failure_is_recorded_in_the_state_before_raising(self) -> None:
        state = bare_state()
        with self.assertRaises(PreprocessingError):
            ModelSelectionAgent().run(state)
        self.assertTrue(state.errors)
        self.assertTrue(any(item.level == "error" for item in state.agent_messages))

    def test_failure_is_logged_at_error_level(self) -> None:
        state = bare_state()
        with self.assertRaises(PreprocessingError):
            ModelSelectionAgent().run(state)
        errors = [item for item in state.agent_messages if item.level == "error"]
        self.assertTrue(errors)
        self.assertEqual(errors[-1].agent, "ModelSelectionAgent")


class TestModelSelection(unittest.TestCase):
    """The happy path: five candidates for a classification problem."""

    def setUp(self) -> None:
        self.state = state_after_preprocessing()
        self.agent = ModelSelectionAgent()
        self.result = self.agent.run(self.state)

    def test_all_five_models_are_selected(self) -> None:
        self.assertEqual(self.result.selected_model_names, EXPECTED_MODELS)

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.result.model_selection_status, TrainingStatus.COMPLETED.value)
        self.assertEqual(self.result.status, AgentStatus.MODELS_SELECTED.value)

    def test_selected_models_are_typed_records(self) -> None:
        for model in self.result.selected_models:
            self.assertTrue(model.name)
            self.assertTrue(model.model_type)
            self.assertIsInstance(model.scaling_required, bool)
            self.assertTrue(model.selection_reason)
            self.assertIsInstance(model.default_parameters, dict)

    def test_scaling_requirements_are_recorded(self) -> None:
        self.assertEqual(
            set(self.result.scaling_required_model_names), SCALING_REQUIRED
        )
        self.assertEqual(
            set(self.result.scaling_optional_model_names), SCALING_NOT_REQUIRED
        )

    def test_baseline_parameters_are_recorded(self) -> None:
        by_name = {model.name: model.default_parameters for model in self.result.selected_models}
        self.assertEqual(by_name["Logistic Regression"]["max_iter"], 1000)
        self.assertEqual(by_name["Random Forest"]["n_estimators"], 100)
        self.assertEqual(by_name["K-Nearest Neighbors"]["n_neighbors"], 5)
        self.assertFalse(by_name["Support Vector Machine"]["probability"])
        for parameters in by_name.values():
            if "random_state" in parameters:
                self.assertEqual(parameters["random_state"], 42)

    def test_no_model_is_instantiated_or_trained(self) -> None:
        """Selection is a decision step; training happens in the next agent."""
        self.assertIsNone(self.result.best_pipeline)
        self.assertEqual(self.result.experiment_results, [])

    def test_selection_is_deterministic(self) -> None:
        again = ModelSelectionAgent().run(state_after_preprocessing())
        self.assertEqual(self.result.selected_model_names, again.selected_model_names)

    def test_summary_records_the_decision(self) -> None:
        summary = self.result.model_selection_summary
        self.assertEqual(summary.status, TrainingStatus.COMPLETED.value)
        self.assertEqual(summary.selected_count, 5)
        self.assertEqual(summary.selected_models, EXPECTED_MODELS)
        self.assertEqual(summary.problem_type, self.result.problem_type)
        self.assertEqual(summary.target_column, self.result.target_column)
        self.assertIn("Logistic Regression", summary.summary_text)
        self.assertIn("f1_score", summary.summary_text)

    def test_summary_documents_the_scaling_plan(self) -> None:
        summary = self.result.model_selection_summary
        self.assertEqual(set(summary.scaling_required_models), SCALING_REQUIRED)
        self.assertEqual(set(summary.scaling_optional_models), SCALING_NOT_REQUIRED)

    def test_decisions_are_logged_for_explainability(self) -> None:
        texts = [item.message for item in self.result.agent_messages]
        joined = "\n".join(texts)
        for name in EXPECTED_MODELS:
            self.assertIn(f"Selected {name}", joined)
        self.assertIn("Scaling plan", joined)
        self.assertIn("Ranking rule", joined)
        self.assertIn("f1_score", joined)

    def test_run_returns_the_same_state_object(self) -> None:
        self.assertIs(self.result, self.state)


class TestSubsetSelection(unittest.TestCase):
    """A caller may narrow the candidate list."""

    def test_only_the_requested_models_are_selected(self) -> None:
        state = ModelSelectionAgent().run(
            state_after_preprocessing(), model_names=["Random Forest"]
        )
        self.assertEqual(state.selected_model_names, ["Random Forest"])

    def test_excluded_models_are_left_out(self) -> None:
        state = ModelSelectionAgent().run(
            state_after_preprocessing(), exclude_models=["Support Vector Machine"]
        )
        self.assertNotIn("Support Vector Machine", state.selected_model_names)
        self.assertEqual(len(state.selected_model_names), 4)

    def test_excluded_models_are_recorded_with_a_reason(self) -> None:
        state = ModelSelectionAgent().run(
            state_after_preprocessing(), exclude_models=["K-Nearest Neighbors"]
        )
        self.assertIn(
            "excluded by the caller", state.model_selection_summary.excluded_models["K-Nearest Neighbors"]
        )

    def test_unknown_model_name_is_rejected(self) -> None:
        with self.assertRaises(ModelConfigurationError) as context:
            ModelSelectionAgent().run(
                state_after_preprocessing(), model_names=["Deep Belief Network"]
            )
        self.assertIn("Deep Belief Network", str(context.exception))
        self.assertIn("Supported models", str(context.exception))

    def test_excluding_everything_is_rejected(self) -> None:
        with self.assertRaises(ModelSelectionError):
            ModelSelectionAgent().run(state_after_preprocessing(), exclude_models=EXPECTED_MODELS)


class TestMulticlassSelection(unittest.TestCase):
    """Multiclass datasets get the same candidate list."""

    def test_multiclass_selects_the_same_models(self) -> None:
        state = ModelSelectionAgent().run(state_after_preprocessing(MULTICLASS_DATASET))
        self.assertEqual(state.problem_type, ProblemType.MULTICLASS_CLASSIFICATION.value)
        self.assertEqual(state.selected_model_names, EXPECTED_MODELS)


class TestSkipBehaviour(unittest.TestCase):
    """An unsupported problem type is skipped with a reason, not guessed."""

    def _state_with_problem(self, problem_type: str) -> AgentState:
        state = state_after_preprocessing()
        state.problem_type = problem_type
        return state

    def test_regression_is_skipped_with_a_reason(self) -> None:
        state = self._state_with_problem(ProblemType.REGRESSION.value)
        result = ModelSelectionAgent().run(state)
        self.assertEqual(result.model_selection_status, TrainingStatus.SKIPPED.value)
        self.assertEqual(result.selected_models, [])
        self.assertIn("not supported", result.model_selection_summary.status_reason)
        self.assertIn("classification", result.model_selection_summary.status_reason)

    def test_uncertain_problem_type_is_skipped(self) -> None:
        state = self._state_with_problem(ProblemType.UNSUPPORTED_OR_UNCERTAIN.value)
        result = ModelSelectionAgent().run(state)
        self.assertEqual(result.model_selection_status, TrainingStatus.SKIPPED.value)
        self.assertTrue(result.model_selection_summary.status_reason)

    def test_skip_does_not_raise(self) -> None:
        state = self._state_with_problem(ProblemType.REGRESSION.value)
        result = ModelSelectionAgent().run(state)
        self.assertIsNotNone(result.model_selection_summary)
        self.assertEqual(result.errors, [])

    def test_skip_is_logged_as_a_warning(self) -> None:
        state = self._state_with_problem(ProblemType.REGRESSION.value)
        result = ModelSelectionAgent().run(state)
        warnings = [item for item in result.agent_messages if item.level == "warning"]
        self.assertTrue(warnings)
        self.assertIn("skipped", warnings[-1].message.lower())

    def test_empty_configuration_is_skipped(self) -> None:
        state = state_after_preprocessing()
        state.preprocessing_config.numerical_features = []
        state.preprocessing_config.categorical_features = []
        result = ModelSelectionAgent().run(state)
        self.assertEqual(result.model_selection_status, TrainingStatus.SKIPPED.value)
        self.assertIn("usable feature", result.model_selection_summary.status_reason)


class TestStateJsonSafety(unittest.TestCase):
    """The state stays serialisable after model selection."""

    def test_state_is_json_serialisable(self) -> None:
        state = ModelSelectionAgent().run(state_after_preprocessing())
        payload = json.loads(json.dumps(state.to_dict()))  # plain dumps, no default=
        self.assertEqual(
            payload["selected_models"][0]["name"], "Logistic Regression"
        )
        self.assertEqual(payload["model_selection_status"], "completed")
        self.assertEqual(payload["model_selection_summary"]["selected_count"], 5)

    def test_selected_models_are_dicts_in_the_payload(self) -> None:
        state = ModelSelectionAgent().run(state_after_preprocessing())
        payload = state.to_dict()
        for item in payload["selected_models"]:
            self.assertIsInstance(item, dict)
            self.assertEqual(sorted(item)[:3], ["default_parameters", "model_type", "name"])

    def test_no_live_estimators_reach_the_state(self) -> None:
        state = ModelSelectionAgent().run(state_after_preprocessing())
        payload = state.to_dict()
        self.assertNotIn("best_pipeline", payload)
        self.assertEqual(payload["experiment_results"], [])

    def test_selection_twice_produces_the_same_payload(self) -> None:
        first = ModelSelectionAgent().run(state_after_preprocessing()).to_dict()
        second = ModelSelectionAgent().run(state_after_preprocessing()).to_dict()
        self.assertEqual(
            first["selected_models"], second["selected_models"]
        )
        self.assertEqual(
            first["model_selection_summary"]["selected_models"],
            second["model_selection_summary"]["selected_models"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
