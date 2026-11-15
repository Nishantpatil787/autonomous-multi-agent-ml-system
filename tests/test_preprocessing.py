"""Comprehensive unittest test suite for the Preprocessing Agent and tools."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from agents.data_analysis_agent import DataAnalysisAgent
from agents.preprocessing_agent import PreprocessingAgent
from tools import data_tools, preprocessing_tools
from utils.state import AgentState, WorkflowStatus

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
SAMPLE_BINARY_CSV = DATASETS_DIR / "sample_classification.csv"
SAMPLE_MULTI_CSV = DATASETS_DIR / "sample_multiclass.csv"
SAMPLE_EDGE_CSV = DATASETS_DIR / "sample_edge_cases.csv"


class TestPreprocessing(unittest.TestCase):
    """Test suite verifying Preprocessing Agent, tool functions, and leakage prevention."""

    def test_01_numerical_median_imputation(self) -> None:
        """Test that numerical pipeline imputes missing values using the median."""
        pipe = preprocessing_tools.build_numerical_pipeline(impute_strategy="median", scale=False)
        X = pd.DataFrame({"num": [10.0, 20.0, np.nan, 30.0, 100.0]})
        # Median of [10, 20, 30, 100] is 25.0
        transformed = pipe.fit_transform(X)
        self.assertEqual(transformed[2, 0], 25.0)

    def test_02_categorical_most_frequent_imputation(self) -> None:
        """Test that categorical pipeline imputes missing values using the most frequent category."""
        pipe = preprocessing_tools.build_categorical_pipeline(impute_strategy="most_frequent")
        X = pd.DataFrame({"cat": ["apple", "apple", "banana", np.nan]})
        transformed = pipe.fit_transform(X)
        # Should have 2 unique categories: apple, banana. The missing should be imputed as 'apple'.
        # apple is index 0 -> [1, 0]
        self.assertEqual(transformed.shape[1], 2)
        # 4th sample was missing, now imputed as 'apple' -> [1.0, 0.0]
        self.assertEqual(transformed[3, 0], 1.0)
        self.assertEqual(transformed[3, 1], 0.0)

    def test_03_one_hot_encoding(self) -> None:
        """Test that categorical pipeline correctly applies OneHotEncoding."""
        pipe = preprocessing_tools.build_categorical_pipeline()
        X = pd.DataFrame({"cat": ["Red", "Green", "Blue", "Red"]})
        transformed = pipe.fit_transform(X)
        self.assertEqual(transformed.shape, (4, 3))
        # Each row should sum to 1.0
        for row in transformed:
            self.assertEqual(np.sum(row), 1.0)

    def test_04_unknown_category_handling(self) -> None:
        """Test that unseen categories in test data do not raise errors (handle_unknown='ignore')."""
        pipe = preprocessing_tools.build_categorical_pipeline(handle_unknown="ignore")
        X_train = pd.DataFrame({"cat": ["low", "medium", "high"]})
        pipe.fit(X_train)

        X_test = pd.DataFrame({"cat": ["unknown_cat", "low"]})
        transformed_test = pipe.transform(X_test)
        # Unknown category should become all zeros [0, 0, 0]
        self.assertEqual(np.sum(transformed_test[0]), 0.0)
        # Known 'low' should have a one
        self.assertEqual(np.sum(transformed_test[1]), 1.0)

    def test_05_scaling_configuration(self) -> None:
        """Test scaling behavior: StandardScaler applies zero-mean, unit-variance when scale=True."""
        pipe_scaled = preprocessing_tools.build_numerical_pipeline(scale=True)
        X = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        scaled_out = pipe_scaled.fit_transform(X)
        self.assertAlmostEqual(float(scaled_out.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(scaled_out.std()), 1.0, places=5)

        pipe_unscaled = preprocessing_tools.build_numerical_pipeline(scale=False)
        unscaled_out = pipe_unscaled.fit_transform(X)
        self.assertEqual(unscaled_out[0, 0], 10.0)

    def test_06_identifier_exclusion(self) -> None:
        """Test that identifier columns such as record_id are excluded from model features."""
        filter_res = preprocessing_tools.filter_usable_features(
            feature_names=["record_id", "age", "salary"],
            numerical_features=["age", "salary"],
            categorical_features=["record_id"],
            total_rows=40,
        )
        self.assertIn("record_id", filter_res["excluded_features"])
        self.assertIn("age", filter_res["selected_numerical_features"])
        self.assertIn("Identifier-like", filter_res["exclusion_reasons"]["record_id"])

    def test_07_constant_column_exclusion(self) -> None:
        """Test that constant columns flagged in Phase 2 are excluded."""
        filter_res = preprocessing_tools.filter_usable_features(
            feature_names=["const_col", "age"],
            numerical_features=["const_col", "age"],
            categorical_features=[],
            data_quality_issues=[{"type": "constant_column", "column": "const_col"}],
            total_rows=40,
        )
        self.assertIn("const_col", filter_res["excluded_features"])
        self.assertIn("const_col", filter_res["constant_columns_excluded"])
        self.assertIn("age", filter_res["selected_numerical_features"])

    def test_08_datetime_exclusion(self) -> None:
        """Test that datetime columns are excluded in this phase."""
        filter_res = preprocessing_tools.filter_usable_features(
            feature_names=["date_registered", "income"],
            numerical_features=["income"],
            categorical_features=["date_registered"],
            total_rows=40,
        )
        self.assertIn("date_registered", filter_res["excluded_features"])
        self.assertIn("date_registered", filter_res["datetime_columns_excluded"])

    def test_09_high_cardinality_handling(self) -> None:
        """Test that categorical columns with cardinality exceeding max_cardinality are excluded."""
        cat_stats = {"high_card_code": {"unique_count": 80}}
        filter_res = preprocessing_tools.filter_usable_features(
            feature_names=["high_card_code", "category"],
            numerical_features=[],
            categorical_features=["high_card_code", "category"],
            categorical_statistics=cat_stats,
            max_cardinality=50,
            total_rows=100,
        )
        self.assertIn("high_card_code", filter_res["excluded_features"])
        self.assertIn("high_card_code", filter_res["high_cardinality_columns_excluded"])
        self.assertIn("category", filter_res["selected_categorical_features"])

    def test_10_duplicate_handling(self) -> None:
        """Test that duplicate row detection and strategy are recorded."""
        state = AgentState(
            dataset_path=str(SAMPLE_BINARY_CSV),
            workflow_status=WorkflowStatus.ANALYSIS_COMPLETED,
            analysis_summary={"dummy": "summary"},
            feature_names=["age", "salary", "department"],
            numerical_features=["age", "salary"],
            categorical_features=["department"],
            duplicate_count=2,
            dataset_shape=(60, 4),
        )
        agent = PreprocessingAgent()
        updated_state = agent.run(state, duplicate_strategy="remove_before_training")
        self.assertEqual(updated_state.duplicate_handling_strategy, "remove_before_training")
        self.assertEqual(updated_state.duplicate_rows_detected, 2)
        self.assertEqual(updated_state.duplicate_rows_removed, 2)

    def test_11_column_transformer_creation(self) -> None:
        """Test ColumnTransformer building and reconstitution from configuration dictionary."""
        config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["age", "salary"],
            selected_categorical_features=["department"],
            excluded_features=["record_id"],
            exclusion_reasons={"record_id": "identifier"},
            numerical_imputation_strategy="median",
            categorical_imputation_strategy="most_frequent",
            scale_numeric=True,
        )
        transformer = preprocessing_tools.recreate_column_transformer_from_config(config)
        self.assertIsInstance(transformer, ColumnTransformer)
        self.assertEqual(len(transformer.transformers), 2)

    def test_12_agent_state_update(self) -> None:
        """Test that PreprocessingAgent updates AgentState with complete metadata."""
        # Run analysis first
        analysis_agent = DataAnalysisAgent()
        state = AgentState(dataset_path=str(SAMPLE_BINARY_CSV))
        state = analysis_agent.run(state)

        # Run preprocessing
        prep_agent = PreprocessingAgent()
        state = prep_agent.run(state)

        self.assertEqual(state.workflow_status, WorkflowStatus.PREPROCESSING_COMPLETED)
        self.assertTrue(len(state.selected_numerical_features) > 0)
        self.assertTrue(len(state.selected_categorical_features) > 0)
        self.assertEqual(state.numerical_imputation_strategy, "median")
        self.assertEqual(state.categorical_imputation_strategy, "most_frequent")
        self.assertEqual(state.scaling_strategy, "standard_scaler")
        self.assertTrue(state.scale_numeric)
        self.assertIn("selected_features_count", state.preprocessing_summary)

    def test_13_missing_analysis_state_error(self) -> None:
        """Test that PreprocessingAgent halts with FAILED status if analysis was not run."""
        state = AgentState(dataset_path=str(SAMPLE_BINARY_CSV))
        prep_agent = PreprocessingAgent()
        updated_state = prep_agent.run(state)

        self.assertEqual(updated_state.workflow_status, WorkflowStatus.FAILED)
        self.assertIn("Data Analysis has not been completed", updated_state.error_message)

    def test_14_invalid_preprocessing_configuration(self) -> None:
        """Test that validate_preprocessing_config catches invalid parameters."""
        # Invalid numerical strategy
        valid, msg = preprocessing_tools.validate_preprocessing_config({
            "numerical_imputation_strategy": "invalid_strategy",
            "selected_numerical_features": ["f1"],
            "selected_categorical_features": [],
        })
        self.assertFalse(valid)
        self.assertIn("Unsupported numerical", msg)

        # Empty features
        valid_empty, msg_empty = preprocessing_tools.validate_preprocessing_config({
            "numerical_imputation_strategy": "median",
            "categorical_imputation_strategy": "most_frequent",
            "selected_numerical_features": [],
            "selected_categorical_features": [],
        })
        self.assertFalse(valid_empty)
        self.assertIn("No usable features", msg_empty)

    def test_15_pipeline_creation_serializable(self) -> None:
        """Test that pipeline config is pure serializable dict without complex live objects."""
        config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["a"],
            selected_categorical_features=["b"],
            excluded_features=[],
            exclusion_reasons={},
        )
        self.assertIsInstance(config, dict)
        self.assertIn("selected_numerical_features", config)
        self.assertEqual(config["numerical_imputation_strategy"], "median")

    def test_16_pipeline_can_fit_on_training_data(self) -> None:
        """Test that ColumnTransformer fits cleanly on a training split."""
        df = data_tools.load_csv(SAMPLE_BINARY_CSV)
        X = df.drop(columns=["Outcome"])

        config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["age", "salary", "credit_score"],
            selected_categorical_features=["department", "education"],
            excluded_features=[],
            exclusion_reasons={},
            scale_numeric=True,
        )
        transformer = preprocessing_tools.recreate_column_transformer_from_config(config)

        # Fit on training split
        X_train = X.iloc[:45]
        transformer.fit(X_train)
        self.assertTrue(hasattr(transformer, "transformers_"))

    def test_17_pipeline_can_transform_test_data(self) -> None:
        """Test that fitted ColumnTransformer transforms unseen test data without errors."""
        df = data_tools.load_csv(SAMPLE_BINARY_CSV)
        X = df.drop(columns=["Outcome"])

        config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["age", "salary", "credit_score"],
            selected_categorical_features=["department", "education"],
            excluded_features=[],
            exclusion_reasons={},
            scale_numeric=True,
        )
        transformer = preprocessing_tools.recreate_column_transformer_from_config(config)

        X_train = X.iloc[:45]
        X_test = X.iloc[45:]

        transformer.fit(X_train)
        X_test_trans = transformer.transform(X_test)
        self.assertIsInstance(X_test_trans, np.ndarray)
        self.assertEqual(X_test_trans.shape[0], 15)
        # Check no NaN values in transformed test data
        self.assertFalse(np.isnan(X_test_trans).any())

    def test_18_no_data_leakage(self) -> None:
        """Explicitly verify that pipeline fitting is computed strictly on X_train and not test data."""
        # Create a synthetic case with distinct statistics
        # Train data: mean = 10
        # Test data: extreme outlier mean = 1000
        train_df = pd.DataFrame({"feat": [8.0, 10.0, 12.0]})
        test_df = pd.DataFrame({"feat": [990.0, 1000.0, 1010.0]})

        config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=["feat"],
            selected_categorical_features=[],
            excluded_features=[],
            exclusion_reasons={},
            scale_numeric=True,
        )
        transformer = preprocessing_tools.recreate_column_transformer_from_config(config)

        # FIT ONLY ON TRAIN
        transformer.fit(train_df)

        # Extract the scaler fitted mean
        fitted_scaler = transformer.named_transformers_["numerical"].named_steps["scaler"]
        train_fitted_mean = fitted_scaler.mean_[0]

        # The scaler mean MUST match train mean (10.0), unaffected by test data
        self.assertEqual(train_fitted_mean, 10.0)

        # Transform test data
        transformed_test = transformer.transform(test_df)

        # If test data had leaked during fit, fitted_scaler.mean_ would be near ~505.
        # Verify fitted mean remained strictly 10.0
        self.assertEqual(fitted_scaler.mean_[0], 10.0)
        # Test sample 1000 scaled with train mean 10 and std sqrt(8/3) ~ 1.633 is ~ 606
        self.assertTrue(transformed_test[1, 0] > 500.0)


if __name__ == "__main__":
    unittest.main()
