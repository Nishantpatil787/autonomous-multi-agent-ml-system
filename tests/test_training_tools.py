"""Tests for tools/training_tools.py."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tools import preprocessing_tools, training_tools

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
SAMPLE_CSV = DATASETS_DIR / "sample_classification.csv"


class TestTrainingTools(unittest.TestCase):
    """Unit tests for tools/training_tools.py."""

    def test_01_load_training_dataframe_and_duplicates(self) -> None:
        """Verify dataset loading and in-memory duplicate row removal."""
        df_clean, dups_removed = training_tools.load_training_dataframe(
            SAMPLE_CSV,
            duplicate_strategy="remove_before_training",
        )
        self.assertEqual(dups_removed, 2)
        self.assertEqual(len(df_clean), 58)

        # Retain duplicates if strategy is 'keep'
        df_raw, dups_kept = training_tools.load_training_dataframe(
            SAMPLE_CSV,
            duplicate_strategy="keep",
        )
        self.assertEqual(dups_kept, 0)
        self.assertEqual(len(df_raw), 60)

    def test_02_prepare_features_and_target(self) -> None:
        """Verify target is removed from feature matrix and excluded features are dropped."""
        df = pd.DataFrame({
            "age": [25, 30],
            "salary": [50000, 60000],
            "record_id": ["id1", "id2"],
            "Outcome": [0, 1],
        })
        X, y = training_tools.prepare_features_and_target(
            df,
            target_column="Outcome",
            excluded_features=["record_id"],
        )
        self.assertNotIn("Outcome", X.columns)
        self.assertNotIn("record_id", X.columns)
        self.assertIn("age", X.columns)
        self.assertIn("salary", X.columns)
        self.assertEqual(list(y), [0, 1])

    def test_03_split_dataset_stratified(self) -> None:
        """Verify stratified split preserves label proportions."""
        X = pd.DataFrame({"feat": range(20)})
        y = pd.Series([0] * 10 + [1] * 10)
        X_train, X_test, y_train, y_test, is_stratified, _ = training_tools.split_dataset(
            X, y, test_size=0.2, random_state=42
        )
        self.assertTrue(is_stratified)
        self.assertEqual(len(X_train), 16)
        self.assertEqual(len(X_test), 4)
        # Stratification test: 2 zeros, 2 ones in test set
        self.assertEqual(int((y_test == 0).sum()), 2)
        self.assertEqual(int((y_test == 1).sum()), 2)

    def test_04_split_dataset_fallback_tiny_class(self) -> None:
        """Verify safe fallback to unstratified split when a class has only 1 sample."""
        X = pd.DataFrame({"feat": range(10)})
        y = pd.Series([0] * 9 + [1] * 1)  # class 1 has only 1 sample
        X_train, X_test, y_train, y_test, is_stratified, reason = training_tools.split_dataset(
            X, y, test_size=0.2, random_state=42
        )
        self.assertFalse(is_stratified)
        self.assertIn("Non-stratified split", reason)
        self.assertEqual(len(X_train) + len(X_test), 10)

    def test_05_build_model_pipeline(self) -> None:
        """Verify pipeline creation with model-specific scaling rules."""
        base_config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["age"],
            selected_categorical_features=[],
            excluded_features=[],
            exclusion_reasons={},
        )
        # Tree model -> scaling False
        rf_pipe, rf_scale = training_tools.build_model_pipeline(base_config, "random_forest")
        self.assertIsInstance(rf_pipe, Pipeline)
        self.assertFalse(rf_scale)

        # Logistic Regression -> scaling True
        lr_pipe, lr_scale = training_tools.build_model_pipeline(base_config, "logistic_regression")
        self.assertIsInstance(lr_pipe, Pipeline)
        self.assertTrue(lr_scale)

    def test_06_evaluate_classifier(self) -> None:
        """Verify metric calculations for binary classification."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        metrics = training_tools.evaluate_classifier(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)

    def test_07_calculate_confusion_matrix(self) -> None:
        """Verify confusion matrix generation and label ordering."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        cm = training_tools.calculate_confusion_matrix(y_true, y_pred)
        self.assertIn("matrix", cm)
        self.assertIn("labels", cm)
        self.assertEqual(cm["labels"], ["0", "1"])
        self.assertEqual(len(cm["matrix"]), 2)

    def test_08_run_cross_validation(self) -> None:
        """Verify cross validation executes over the pipeline and returns mean/std metrics."""
        base_config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["val"],
            selected_categorical_features=[],
            excluded_features=[],
            exclusion_reasons={},
        )
        pipeline, _ = training_tools.build_model_pipeline(base_config, "decision_tree")
        X = pd.DataFrame({"val": range(30)})
        y = pd.Series([0] * 15 + [1] * 15)

        cv_res = training_tools.run_cross_validation(pipeline, X, y, n_splits=3, random_state=42)
        self.assertEqual(cv_res["status"], "completed")
        self.assertEqual(cv_res["cv_folds"], 3)
        self.assertIsNotNone(cv_res["cv_accuracy_mean"])
        self.assertIsNotNone(cv_res["cv_f1_mean"])

    def test_09_run_cross_validation_small_class(self) -> None:
        """Verify cross validation safely handles very small class by reducing folds or skipping."""
        base_config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["val"],
            selected_categorical_features=[],
            excluded_features=[],
            exclusion_reasons={},
        )
        pipeline, _ = training_tools.build_model_pipeline(base_config, "decision_tree")
        X = pd.DataFrame({"val": range(10)})
        y = pd.Series([0] * 9 + [1] * 1)  # only 1 sample in class 1

        cv_res = training_tools.run_cross_validation(pipeline, X, y, n_splits=5)
        self.assertEqual(cv_res["status"], "skipped")
        self.assertIsNone(cv_res["cv_f1_mean"])

    def test_10_rank_experiments(self) -> None:
        """Verify experiment ranking by F1-score descending and Accuracy descending."""
        exps = [
            {"model_name": "model_b", "f1_score": 0.80, "accuracy": 0.85, "precision": 0.80, "recall": 0.80, "training_time": 0.1, "preprocessing_scaling": False, "status": "success"},
            {"model_name": "model_a", "f1_score": 0.90, "accuracy": 0.88, "precision": 0.90, "recall": 0.90, "training_time": 0.1, "preprocessing_scaling": True, "status": "success"},
            {"model_name": "model_c", "f1_score": 0.80, "accuracy": 0.82, "precision": 0.80, "recall": 0.80, "training_time": 0.1, "preprocessing_scaling": False, "status": "success"},
        ]
        ranked, best = training_tools.rank_experiments(exps)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["model_name"], "model_a")
        self.assertEqual(ranked[1]["model_name"], "model_b")  # 0.80 F1, 0.85 Acc > 0.82 Acc
        self.assertEqual(ranked[2]["model_name"], "model_c")
        self.assertEqual(best["model_name"], "model_a")

    def test_11_no_data_leakage_in_training(self) -> None:
        """Explicitly verify that pipeline fitting during training learns statistics only from train split."""
        # Train data has mean = 20.0
        # Test data has extreme mean = 5000.0
        X_train = pd.DataFrame({"num": [10.0, 20.0, 30.0]})
        y_train = pd.Series([0, 1, 0])

        X_test = pd.DataFrame({"num": [4990.0, 5000.0, 5010.0]})
        y_test = pd.Series([1, 1, 1])

        base_config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["num"],
            selected_categorical_features=[],
            excluded_features=[],
            exclusion_reasons={},
        )
        pipeline, _ = training_tools.build_model_pipeline(base_config, "logistic_regression")

        # Fit strictly on train
        pipeline.fit(X_train, y_train)

        # Check fitted scaler mean in the pipeline
        preprocessor = pipeline.named_steps["preprocessor"]
        fitted_scaler = preprocessor.named_transformers_["numerical"].named_steps["scaler"]

        # Mean must be strictly 20.0 (X_train mean), completely untainted by X_test (5000.0)
        self.assertEqual(float(fitted_scaler.mean_[0]), 20.0)

        # Predict on X_test works without error
        preds = pipeline.predict(X_test)
        self.assertEqual(len(preds), 3)


if __name__ == "__main__":
    unittest.main()
