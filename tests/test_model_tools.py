"""Tests for model registry, estimator factory, and model configuration tools."""

import unittest
from sklearn.base import BaseEstimator
from tools import model_tools


class TestModelTools(unittest.TestCase):
    """Unit tests for tools/model_tools.py."""

    def test_01_registry_contains_all_five_models(self) -> None:
        """Verify registry contains all 5 supported models."""
        expected_models = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "k_nearest_neighbors",
            "support_vector_machine",
        ]
        registry = model_tools.build_model_registry()
        for m in expected_models:
            self.assertIn(m, registry)

    def test_02_model_definitions_are_valid(self) -> None:
        """Verify model definitions have required keys and instantiate valid estimators."""
        for key in model_tools.get_supported_models():
            definition = model_tools.get_model_definition(key)
            self.assertIn("display_name", definition)
            self.assertIn("scaling_required", definition)
            self.assertIn("default_params", definition)
            self.assertIn("factory", definition)

            estimator = model_tools.build_estimator(key)
            self.assertIsInstance(estimator, BaseEstimator)

    def test_03_supported_classification_problem_types_respected(self) -> None:
        """Verify models match binary and multiclass classification tasks."""
        binary_models = model_tools.get_supported_models("binary_classification")
        multi_models = model_tools.get_supported_models("multiclass_classification")
        self.assertEqual(len(binary_models), 5)
        self.assertEqual(len(multi_models), 5)

        # Regression is not supported
        reg_models = model_tools.get_supported_models("regression")
        self.assertEqual(len(reg_models), 0)

    def test_04_scaling_recommendations_are_correct(self) -> None:
        """Verify scaling recommendations: DT and RF False, LR, KNN, and SVM True."""
        self.assertTrue(model_tools.get_model_definition("logistic_regression")["scaling_required"])
        self.assertFalse(model_tools.get_model_definition("decision_tree")["scaling_required"])
        self.assertFalse(model_tools.get_model_definition("random_forest")["scaling_required"])
        self.assertTrue(model_tools.get_model_definition("k_nearest_neighbors")["scaling_required"])
        self.assertTrue(model_tools.get_model_definition("support_vector_machine")["scaling_required"])

    def test_05_invalid_model_names_handled(self) -> None:
        """Verify that invalid model names raise KeyError or validation failure."""
        with self.assertRaises(KeyError):
            model_tools.get_model_definition("deep_neural_network_unknown")

        valid, msg = model_tools.validate_model_configuration("fake_model")
        self.assertFalse(valid)
        self.assertIn("not in the controlled registry", msg)


if __name__ == "__main__":
    unittest.main()
