"""Tests for the controlled model registry (Phase 4).

Covers the 27 required checks for model selection: registry content, definition
validity, problem-type support, scaling recommendations and error handling for
invalid input.
"""

from __future__ import annotations

import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tools.model_tools import (
    MODEL_REGISTRY,
    SUPPORTED_PROBLEM_TYPES,
    ModelConfigurationError,
    ModelSelectionError,
    build_estimator,
    build_model_registry,
    config_for_scaling,
    get_effective_parameters,
    get_excluded_model_reasons,
    get_model_definition,
    get_supported_models,
    select_candidate_models,
    supports_problem_type,
    validate_model_configuration,
)
from utils import config
from utils.state import ModelType, PreprocessingConfig, ProblemType, ScalingStrategy

EXPECTED_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Support Vector Machine",
]

EXPECTED_ESTIMATORS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Support Vector Machine": SVC,
}

EXPECTED_SCALING = {
    "Logistic Regression": True,
    "Decision Tree": False,
    "Random Forest": False,
    "K-Nearest Neighbors": True,
    "Support Vector Machine": True,
}


def sample_config() -> PreprocessingConfig:
    """A minimal valid preprocessing configuration for the scaling tests."""
    return PreprocessingConfig(
        numerical_features=["age", "bmi"],
        categorical_features=["gender"],
        scale_numeric=True,
        scaling_strategy=ScalingStrategy.STANDARD.value,
    )


class TestRegistryContents(unittest.TestCase):
    """1. The registry contains exactly the five supported models."""

    def test_registry_contains_all_five_supported_models(self) -> None:
        self.assertEqual(get_supported_models(), EXPECTED_MODELS)

    def test_build_model_registry_returns_the_same_entries(self) -> None:
        self.assertEqual(list(build_model_registry()), EXPECTED_MODELS)

    def test_registry_is_keyed_by_model_name(self) -> None:
        for name in EXPECTED_MODELS:
            self.assertIn(name, MODEL_REGISTRY)
            self.assertEqual(MODEL_REGISTRY[name].name, name)


class TestModelDefinitions(unittest.TestCase):
    """2. Every model definition is complete and internally consistent."""

    def test_definitions_are_valid(self) -> None:
        for name in EXPECTED_MODELS:
            definition = get_model_definition(name)
            self.assertTrue(definition.name, f"{name} has no name")
            self.assertIn(definition.model_type, [item.value for item in ModelType])
            self.assertTrue(definition.description, f"{name} has no description")
            self.assertIsInstance(definition.scaling_required, bool)
            self.assertTrue(definition.default_parameters, f"{name} has no parameters")
            self.assertTrue(definition.supported_problem_types)

    def test_model_types_are_correct(self) -> None:
        self.assertEqual(get_model_definition("Logistic Regression").model_type, ModelType.LINEAR.value)
        self.assertEqual(get_model_definition("Decision Tree").model_type, ModelType.TREE_BASED.value)
        self.assertEqual(get_model_definition("Random Forest").model_type, ModelType.ENSEMBLE.value)
        self.assertEqual(
            get_model_definition("K-Nearest Neighbors").model_type, ModelType.INSTANCE_BASED.value
        )
        self.assertEqual(
            get_model_definition("Support Vector Machine").model_type, ModelType.KERNEL_BASED.value
        )

    def test_default_parameters_match_the_project_configuration(self) -> None:
        self.assertEqual(
            get_model_definition("Logistic Regression").default_parameters,
            config.LOGISTIC_REGRESSION_PARAMS,
        )
        self.assertEqual(
            get_model_definition("Random Forest").default_parameters,
            config.RANDOM_FOREST_PARAMS,
        )
        self.assertEqual(
            get_model_definition("K-Nearest Neighbors").default_parameters,
            config.KNN_PARAMS,
        )
        self.assertEqual(
            get_model_definition("Support Vector Machine").default_parameters,
            config.SVM_PARAMS,
        )

    def test_estimator_factories_build_the_expected_classes(self) -> None:
        for name, estimator_class in EXPECTED_ESTIMATORS.items():
            self.assertIsInstance(build_estimator(name), estimator_class)

    def test_estimator_defaults_are_reproducible(self) -> None:
        # random_state must be pinned wherever the estimator supports it.
        self.assertEqual(build_estimator("Logistic Regression").random_state, config.RANDOM_STATE)
        self.assertEqual(build_estimator("Decision Tree").random_state, config.RANDOM_STATE)
        self.assertEqual(build_estimator("Random Forest").random_state, config.RANDOM_STATE)
        self.assertEqual(build_estimator("Support Vector Machine").random_state, config.RANDOM_STATE)
        self.assertEqual(build_estimator("Random Forest").n_estimators, 100)
        self.assertEqual(build_estimator("K-Nearest Neighbors").n_neighbors, 5)
        self.assertEqual(build_estimator("Logistic Regression").max_iter, 1000)
        self.assertFalse(build_estimator("Support Vector Machine").probability)


class TestScalingRecommendation(unittest.TestCase):
    """4. Scaling recommendations are correct for each model."""

    def test_scaling_recommendation_is_correct(self) -> None:
        for name, expected in EXPECTED_SCALING.items():
            self.assertEqual(get_model_definition(name).scaling_required, expected, name)

    def test_tree_models_do_not_require_scaling(self) -> None:
        self.assertFalse(get_model_definition("Decision Tree").scaling_required)
        self.assertFalse(get_model_definition("Random Forest").scaling_required)

    def test_distance_and_margin_models_require_scaling(self) -> None:
        for name in ("Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"):
            self.assertTrue(get_model_definition(name).scaling_required, name)


class TestScalingVariants(unittest.TestCase):
    """Model-specific configurations are derived without mutating the original."""

    def test_scaled_variant_sets_scale_numeric(self) -> None:
        variant = config_for_scaling(sample_config(), True)
        self.assertTrue(variant.scale_numeric)
        self.assertEqual(variant.scaling_strategy, ScalingStrategy.STANDARD.value)

    def test_unscaled_variant_clears_scaling(self) -> None:
        variant = config_for_scaling(sample_config(), False)
        self.assertFalse(variant.scale_numeric)
        self.assertEqual(variant.scaling_strategy, ScalingStrategy.NONE.value)

    def test_original_configuration_is_never_mutated(self) -> None:
        base = sample_config()
        config_for_scaling(base, False)
        config_for_scaling(base, True)
        self.assertTrue(base.scale_numeric)
        self.assertEqual(base.scaling_strategy, ScalingStrategy.STANDARD.value)

    def test_variant_is_a_copy_not_the_same_object(self) -> None:
        base = sample_config()
        variant = config_for_scaling(base, False)
        self.assertIsNot(variant, base)
        self.assertIsNot(variant.numerical_features, base.numerical_features)

    def test_variant_keeps_the_feature_decisions(self) -> None:
        variant = config_for_scaling(sample_config(), False)
        self.assertEqual(variant.numerical_features, ["age", "bmi"])
        self.assertEqual(variant.categorical_features, ["gender"])

    def test_variant_describes_its_steps(self) -> None:
        variant = config_for_scaling(sample_config(), False)
        self.assertFalse(any("StandardScaler" in step for step in variant.steps[0].transformers))
        scaled = config_for_scaling(sample_config(), True)
        self.assertTrue(any("StandardScaler" in step for step in scaled.steps[0].transformers))

    def test_missing_configuration_raises(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            config_for_scaling(None, True)


class TestProblemTypeSupport(unittest.TestCase):
    """3. Only supported problem types are accepted."""

    def test_all_models_support_classification(self) -> None:
        for problem_type in (
            ProblemType.BINARY_CLASSIFICATION.value,
            ProblemType.MULTICLASS_CLASSIFICATION.value,
        ):
            self.assertEqual(list(SUPPORTED_PROBLEM_TYPES), [
                ProblemType.BINARY_CLASSIFICATION.value,
                ProblemType.MULTICLASS_CLASSIFICATION.value,
            ])
            for name in EXPECTED_MODELS:
                self.assertTrue(
                    supports_problem_type(name, problem_type),
                    f"{name} should support {problem_type}",
                )

    def test_regression_is_not_supported(self) -> None:
        for name in EXPECTED_MODELS:
            self.assertFalse(supports_problem_type(name, ProblemType.REGRESSION.value))
        with self.assertRaises(ModelConfigurationError):
            validate_model_configuration("Random Forest", ProblemType.REGRESSION.value)

    def test_uncertain_problem_type_is_not_supported(self) -> None:
        self.assertFalse(
            supports_problem_type("Random Forest", ProblemType.UNSUPPORTED_OR_UNCERTAIN.value)
        )

    def test_validate_accepts_a_valid_combination(self) -> None:
        self.assertIsNone(
            validate_model_configuration(
                "Logistic Regression", ProblemType.BINARY_CLASSIFICATION.value
            )
        )

    def test_validate_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            validate_model_configuration(
                "Logistic Regression",
                ProblemType.BINARY_CLASSIFICATION.value,
                {"nonsense": 1},
            )

    def test_validate_rejects_missing_problem_type(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            validate_model_configuration("Random Forest", None)


class TestCandidateSelection(unittest.TestCase):
    """Candidate selection is deterministic and records its reasons."""

    def test_selection_returns_all_models_for_classification(self) -> None:
        for problem_type in (
            ProblemType.BINARY_CLASSIFICATION.value,
            ProblemType.MULTICLASS_CLASSIFICATION.value,
        ):
            selected = select_candidate_models(problem_type)
            self.assertEqual([item.name for item in selected], EXPECTED_MODELS)

    def test_selection_records_scaling_requirement(self) -> None:
        selected = {item.name: item.scaling_required for item in select_candidate_models("binary_classification")}
        self.assertEqual(selected, EXPECTED_SCALING)

    def test_selection_records_a_reason_for_every_model(self) -> None:
        for item in select_candidate_models("binary_classification"):
            self.assertTrue(item.selection_reason)
            self.assertIn(item.model_type, [model.value for model in ModelType])
            self.assertTrue(item.default_parameters)

    def test_selection_can_be_limited_to_a_subset(self) -> None:
        selected = select_candidate_models(
            "binary_classification", ["Random Forest", "Decision Tree"]
        )
        self.assertEqual([item.name for item in selected], ["Decision Tree", "Random Forest"])

    def test_selection_can_exclude_models(self) -> None:
        selected = select_candidate_models(
            "binary_classification", exclude=["Support Vector Machine"]
        )
        self.assertNotIn("Support Vector Machine", [item.name for item in selected])
        self.assertEqual(len(selected), 4)

    def test_selection_follows_registry_order_not_caller_order(self) -> None:
        selected = select_candidate_models(
            "binary_classification", ["Support Vector Machine", "Decision Tree"]
        )
        self.assertEqual(
            [item.name for item in selected],
            ["Decision Tree", "Support Vector Machine"],
        )

    def test_selection_is_deterministic(self) -> None:
        first = select_candidate_models("binary_classification")
        second = select_candidate_models("binary_classification")
        self.assertEqual([item.name for item in first], [item.name for item in second])

    def test_unsupported_problem_type_raises(self) -> None:
        with self.assertRaises(ModelSelectionError):
            select_candidate_models(ProblemType.REGRESSION.value)
        with self.assertRaises(ModelSelectionError):
            select_candidate_models(None)

    def test_excluding_every_model_raises(self) -> None:
        with self.assertRaises(ModelSelectionError):
            select_candidate_models("binary_classification", exclude=EXPECTED_MODELS)

    def test_excluded_model_reasons_are_reported(self) -> None:
        reasons = get_excluded_model_reasons(ProblemType.REGRESSION.value)
        self.assertEqual(sorted(reasons), sorted(EXPECTED_MODELS))
        self.assertIn("regression", reasons["Random Forest"])


class TestInvalidInputHandling(unittest.TestCase):
    """5. Invalid model names are rejected loudly."""

    def test_unknown_model_name_raises(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            get_model_definition("Deep Belief Network")
        with self.assertRaises(ModelConfigurationError):
            build_estimator("Deep Belief Network")

    def test_unknown_model_name_in_selection_raises(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            select_candidate_models("binary_classification", exclude=["NotAModel"])

    def test_error_message_lists_the_supported_models(self) -> None:
        with self.assertRaises(ModelConfigurationError) as context:
            get_model_definition("nope")
        for name in EXPECTED_MODELS:
            self.assertIn(name, str(context.exception))

    def test_effective_parameters_merge_overrides(self) -> None:
        merged = get_effective_parameters("Random Forest", {"n_estimators": 10})
        self.assertEqual(merged["n_estimators"], 10)
        self.assertEqual(merged["random_state"], config.RANDOM_STATE)
        self.assertEqual(get_effective_parameters("Random Forest"), config.RANDOM_FOREST_PARAMS)

    def test_default_parameters_are_not_shared_between_calls(self) -> None:
        first = get_effective_parameters("Decision Tree")
        first["random_state"] = 999
        self.assertEqual(get_model_definition("Decision Tree").default_parameters["random_state"],
                         config.RANDOM_STATE)

    def test_estimators_are_independent_instances(self) -> None:
        first = build_estimator("Random Forest")
        second = build_estimator("Random Forest")
        self.assertIsNot(first, second)

    def test_two_random_forests_with_same_seed_predict_identically(self) -> None:
        frame = pd.DataFrame(
            {"a": [float(index % 7) for index in range(40)],
             "b": [float(index % 3) for index in range(40)]}
        )
        target = pd.Series([index % 2 for index in range(40)])
        first = build_estimator("Random Forest").fit(frame, target)
        second = build_estimator("Random Forest").fit(frame, target)
        self.assertEqual(list(first.predict(frame)), list(second.predict(frame)))


if __name__ == "__main__":
    unittest.main(verbosity=2)