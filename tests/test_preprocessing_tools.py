"""Tests for the reusable preprocessing tools (Phase 3)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from tests.helpers import (
    frame_clean_categorical,
    frame_clean_numerical,
    frame_identifier_only,
    frame_mixed,
    frame_with_constant_and_empty,
    frame_with_duplicates,
    frame_with_high_cardinality,
    write_csv,
)
from tools.data_tools import detect_data_quality_issues, load_csv
from tools.preprocessing_tools import (
    PreprocessingConfigError,
    build_categorical_pipeline,
    build_column_transformer,
    build_numerical_pipeline,
    decide_duplicate_handling,
    describe_pipeline_steps,
    estimate_onehot_feature_count,
    estimate_transformed_feature_count,
    identify_constant_columns,
    identify_empty_columns,
    identify_high_cardinality_columns,
    identify_identifier_columns,
    identify_unsupported_columns,
    plan_preprocessing,
    remove_duplicate_rows,
    select_model_features,
    validate_preprocessing_config,
)
from utils.state import (
    DuplicateStrategy,
    ExclusionReason,
    PreprocessingConfig,
    ScalingStrategy,
)


def _excluded_map(config_record: PreprocessingConfig) -> dict:
    """Return {column: reason} for quick assertions."""
    return {item.column: item.reason for item in config_record.excluded_features}


class TestNumericalPipeline(unittest.TestCase):
    """1 & 5. Numerical median imputation and scaling configuration."""

    def test_default_imputation_is_median(self) -> None:
        pipeline = build_numerical_pipeline()
        self.assertIsInstance(pipeline, Pipeline)
        imputer = pipeline.named_steps["imputer"]
        self.assertIsInstance(imputer, SimpleImputer)
        self.assertEqual(imputer.strategy, "median")

    def test_scaling_is_enabled_by_default(self) -> None:
        pipeline = build_numerical_pipeline()
        self.assertIn("scaler", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["scaler"], StandardScaler)

    def test_scaling_can_be_disabled(self) -> None:
        pipeline = build_numerical_pipeline(scale_numeric=False)
        self.assertNotIn("scaler", pipeline.named_steps)
        self.assertEqual(list(pipeline.named_steps), ["imputer"])

    def test_standard_scaler_produces_zero_mean_unit_variance(self) -> None:
        frame = frame_clean_numerical()
        pipeline = build_numerical_pipeline()
        transformed = pipeline.fit_transform(frame[["feature_a", "feature_b"]])
        self.assertTrue(np.allclose(transformed.mean(axis=0), 0.0, atol=1e-9))
        self.assertTrue(np.allclose(transformed.std(axis=0), 1.0, atol=1e-9))

    def test_invalid_numerical_imputation_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            build_numerical_pipeline(imputation="not_a_strategy")

    def test_invalid_scaling_strategy_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            build_numerical_pipeline(scaling_strategy="min_max")


class TestCategoricalPipeline(unittest.TestCase):
    """2, 3 & 4. Categorical imputation, one-hot encoding, unknown categories."""

    def test_default_imputation_is_most_frequent(self) -> None:
        pipeline = build_categorical_pipeline()
        imputer = pipeline.named_steps["imputer"]
        self.assertEqual(imputer.strategy, "most_frequent")

    def test_one_hot_encoder_configuration(self) -> None:
        pipeline = build_categorical_pipeline()
        encoder = pipeline.named_steps["encoder"]
        self.assertIsInstance(encoder, OneHotEncoder)
        self.assertEqual(encoder.handle_unknown, "ignore")

    def test_one_hot_encoding_expands_categories(self) -> None:
        frame = frame_clean_categorical()
        pipeline = build_categorical_pipeline()
        transformed = pipeline.fit_transform(frame[["colour", "size"]])
        # 3 colours + 3 sizes
        self.assertEqual(transformed.shape[1], 6)

    def test_most_frequent_imputation_uses_training_mode(self) -> None:
        frame = pd.DataFrame({"colour": ["red", "red", "blue", None]})
        pipeline = build_categorical_pipeline()
        pipeline.fit(frame)
        imputed = pipeline.named_steps["imputer"].transform(frame)
        self.assertEqual(list(imputed.ravel()), ["red", "red", "blue", "red"])

    def test_unknown_category_is_encoded_as_all_zero_row(self) -> None:
        train = pd.DataFrame({"colour": ["red", "red", "blue", "green"]})
        pipeline = build_categorical_pipeline()
        pipeline.fit(train)
        transformed = pipeline.transform(pd.DataFrame({"colour": ["purple"]}))
        self.assertEqual(transformed.shape[0], 1)
        # Unknown categories are ignored, so the row is entirely zero.
        self.assertAlmostEqual(float(transformed.toarray().sum() if hasattr(transformed, "toarray")
                                    else transformed.sum()), 0.0)

    def test_unknown_categories_never_raise(self) -> None:
        train = pd.DataFrame({"colour": ["red", "blue"]})
        pipeline = build_categorical_pipeline()
        pipeline.fit(train)
        unseen = pd.DataFrame({"colour": ["purple", "orange", "cyan"]})
        # Would raise if handle_unknown were not "ignore".
        transformed = pipeline.transform(unseen)
        self.assertEqual(transformed.shape[0], 3)

    def test_handle_unknown_must_be_ignore(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            build_categorical_pipeline(handle_unknown="error")


class TestColumnIdentification(unittest.TestCase):
    """6, 7, 8 & 9. Identifier, constant, datetime and high-cardinality columns."""

    def setUp(self) -> None:
        self.frame = frame_mixed()

    def test_identify_empty_columns(self) -> None:
        frame = frame_with_constant_and_empty()
        self.assertEqual(identify_empty_columns(frame), ["empty_col"])

    def test_identify_constant_columns(self) -> None:
        self.assertEqual(identify_constant_columns(self.frame), ["constant_col"])

    def test_identify_identifier_columns(self) -> None:
        identifiers = identify_identifier_columns(self.frame, target_column="churn")
        # record_id is a unique integer id; signup_date also holds one value per
        # row. Both are near-unique, so both are reported here. The planner
        # resolves the reason later (a datetime column is reported as an
        # unsupported dtype, not as an identifier).
        self.assertIn("record_id", identifiers)
        self.assertEqual(set(identifiers), {"record_id", "signup_date"})

    def test_identifier_detection_is_ratio_based_not_type_based(self) -> None:
        # 'age' is integer too, but it is not unique per row, so it is kept.
        frame = pd.DataFrame(
            {"record_id": range(20), "age": [20 + index % 5 for index in range(20)]}
        )
        self.assertEqual(identify_identifier_columns(frame), ["record_id"])

    def test_identify_unsupported_datetime_columns(self) -> None:
        unsupported = identify_unsupported_columns(self.frame, target_column="churn")
        self.assertEqual(unsupported, ["signup_date"])

    def test_identify_high_cardinality_columns(self) -> None:
        frame = frame_with_high_cardinality()
        flagged = identify_high_cardinality_columns(frame, target_column="label")
        self.assertEqual(flagged, ["free_text"])

    def test_estimate_onehot_feature_count(self) -> None:
        frame = frame_clean_categorical()
        # 3 colours + 3 sizes; missing values are imputed before encoding and
        # unseen categories do not create extra columns.
        self.assertEqual(estimate_onehot_feature_count(frame, ["colour", "size"]), 6)

    def test_select_model_features_excludes_target_and_excluded(self) -> None:
        numerical, categorical = select_model_features(
            self.frame, target_column="churn", excluded_columns=["record_id", "constant_col"]
        )
        self.assertEqual(numerical, ["age", "bmi"])
        self.assertEqual(categorical, ["city", "signup_date"])


class TestDuplicateHandling(unittest.TestCase):
    """10. Duplicate handling is configurable and never silent."""

    def test_default_strategy_plans_removal(self) -> None:
        handling = decide_duplicate_handling(5)
        self.assertEqual(handling.strategy, DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value)
        self.assertEqual(handling.duplicate_rows_detected, 5)
        self.assertEqual(handling.duplicate_rows_to_remove, 5)
        self.assertEqual(handling.duplicate_rows_removed, 0)
        self.assertIn("never modified", handling.notes)

    def test_keep_all_strategy(self) -> None:
        handling = decide_duplicate_handling(5, strategy=DuplicateStrategy.KEEP_ALL.value)
        self.assertEqual(handling.duplicate_rows_to_remove, 0)
        self.assertIn("deliberately kept", handling.notes)

    def test_invalid_strategy_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            decide_duplicate_handling(1, strategy="drop_everything")

    def test_remove_duplicate_rows_returns_new_frame(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        deduplicated, removed = remove_duplicate_rows(frame)
        self.assertEqual(removed, 3)
        self.assertEqual(deduplicated.shape[0], 20)
        # The input frame is untouched.
        self.assertEqual(frame.shape[0], 23)

    def test_remove_duplicate_rows_keep_all(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        deduplicated, removed = remove_duplicate_rows(
            frame, strategy=DuplicateStrategy.KEEP_ALL.value
        )
        self.assertEqual(removed, 0)
        self.assertEqual(deduplicated.shape[0], 23)

    def test_removal_matches_pandas_drop_duplicates(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        deduplicated, removed = remove_duplicate_rows(frame)
        self.assertEqual(deduplicated.shape[0], frame.drop_duplicates().shape[0])
        self.assertEqual(removed, frame.shape[0] - deduplicated.shape[0])


class TestColumnTransformerCreation(unittest.TestCase):
    """11, 15 & 16. ColumnTransformer creation, pipeline creation and fitting."""

    def setUp(self) -> None:
        self.frame = frame_mixed()
        self.issues = detect_data_quality_issues(self.frame, target_column="churn")
        self.config = plan_preprocessing(
            self.frame, target_column="churn", data_quality_issues=self.issues
        )

    def test_column_transformer_created_with_two_branches(self) -> None:
        transformer = build_column_transformer(self.config)
        self.assertIsInstance(transformer, ColumnTransformer)
        names = [name for name, _, _ in transformer.transformers]
        self.assertEqual(names, ["numerical", "categorical"])

    def test_branch_columns_match_the_configuration(self) -> None:
        transformer = build_column_transformer(self.config)
        branches = {name: columns for name, _, columns in transformer.transformers}
        self.assertEqual(branches["numerical"], self.config.numerical_features)
        self.assertEqual(branches["categorical"], self.config.categorical_features)

    def test_remainder_is_drop(self) -> None:
        transformer = build_column_transformer(self.config)
        self.assertEqual(transformer.remainder, "drop")

    def test_created_pipeline_is_unfitted(self) -> None:
        transformer = build_column_transformer(self.config)
        with self.assertRaises(NotFittedError):
            check_is_fitted(transformer)

    def test_fit_scope_is_recorded(self) -> None:
        self.assertEqual(self.config.fit_scope, "training_data_only")

    def test_pipeline_fits_and_transforms_training_data(self) -> None:
        transformer = build_column_transformer(self.config)
        features = self.frame.drop(columns=["churn"])
        transformed = transformer.fit_transform(features)
        self.assertIsNotNone(transformed)
        self.assertGreater(transformed.shape[0], 0)
        # 2 numerical columns + 4 encoded values of 'city'
        self.assertEqual(transformed.shape[1], 2 + 4)

    def test_pipeline_transforms_test_data(self) -> None:
        train = self.frame.iloc[:30].drop(columns=["churn"])
        test = self.frame.iloc[30:].drop(columns=["churn"])
        transformer = build_column_transformer(self.config)
        transformer.fit(train)
        transformed = transformer.transform(test)
        self.assertEqual(transformed.shape[0], test.shape[0])
        self.assertEqual(transformed.shape[1], 2 + 4)

    def test_excluded_columns_are_not_passed_to_the_transformer(self) -> None:
        transformer = build_column_transformer(self.config)
        # Pass the full frame, including the excluded columns and the target.
        transformed = transformer.fit_transform(self.frame)
        feature_names = list(transformer.get_feature_names_out())
        joined = " ".join(feature_names)
        for excluded in ("record_id", "constant_col", "signup_date", "churn"):
            self.assertNotIn(excluded, joined)
        # Only numerical age/bmi and encoded city survive.
        self.assertEqual(transformed.shape[1], 2 + 4)

    def test_describe_pipeline_steps(self) -> None:
        steps = describe_pipeline_steps(self.config)
        names = [step.name for step in steps]
        self.assertEqual(names, ["numerical", "categorical"])
        self.assertIn("SimpleImputer(strategy='median')", steps[0].transformers)
        self.assertIn("StandardScaler()", steps[0].transformers)
        self.assertTrue(
            any("handle_unknown='ignore'" in item for item in steps[1].transformers)
        )

    def test_estimate_transformed_feature_count(self) -> None:
        estimated = estimate_transformed_feature_count(self.frame, self.config)
        transformer = build_column_transformer(self.config)
        actual = transformer.fit_transform(self.frame.drop(columns=["churn"])).shape[1]
        self.assertEqual(estimated, actual)

    def test_scaling_can_be_disabled_in_the_transformer(self) -> None:
        config_record = plan_preprocessing(
            self.frame, target_column="churn", data_quality_issues=self.issues,
            scale_numeric=False,
        )
        transformer = build_column_transformer(config_record)
        self.assertFalse(config_record.scale_numeric)
        self.assertEqual(config_record.scaling_strategy, ScalingStrategy.NONE.value)
        # Inspect the unfitted branch pipelines.
        branches = {name: pipeline for name, pipeline, _ in transformer.transformers}
        self.assertNotIn("scaler", branches["numerical"].named_steps)


class TestPlanPreprocessing(unittest.TestCase):
    """Planning decisions recorded in the configuration."""

    def test_plan_excludes_identifier_constant_and_datetime(self) -> None:
        frame = frame_mixed()
        config_record = plan_preprocessing(frame, target_column="churn")
        excluded = _excluded_map(config_record)
        self.assertEqual(excluded["record_id"], ExclusionReason.IDENTIFIER_LIKE.value)
        self.assertEqual(excluded["constant_col"], ExclusionReason.CONSTANT_COLUMN.value)
        self.assertEqual(excluded["signup_date"], ExclusionReason.UNSUPPORTED_TYPE.value)

    def test_plan_keeps_usable_features(self) -> None:
        config_record = plan_preprocessing(frame_mixed(), target_column="churn")
        self.assertEqual(config_record.numerical_features, ["age", "bmi"])
        self.assertEqual(config_record.categorical_features, ["city"])

    def test_plan_excludes_high_cardinality_column(self) -> None:
        config_record = plan_preprocessing(
            frame_with_high_cardinality(), target_column="label"
        )
        excluded = _excluded_map(config_record)
        self.assertEqual(excluded["free_text"], ExclusionReason.HIGH_CARDINALITY.value)
        self.assertEqual(config_record.numerical_features, ["amount"])

    def test_plan_excludes_empty_and_constant_columns(self) -> None:
        config_record = plan_preprocessing(
            frame_with_constant_and_empty(), target_column="label"
        )
        excluded = _excluded_map(config_record)
        self.assertEqual(excluded["empty_col"], ExclusionReason.EMPTY_COLUMN.value)
        self.assertEqual(excluded["constant_col"], ExclusionReason.CONSTANT_COLUMN.value)

    def test_plan_records_strategies(self) -> None:
        config_record = plan_preprocessing(frame_mixed(), target_column="churn")
        self.assertEqual(config_record.numerical_imputation, "median")
        self.assertEqual(config_record.categorical_imputation, "most_frequent")
        self.assertEqual(config_record.encoding, "one_hot")
        self.assertEqual(config_record.encoding_handle_unknown, "ignore")
        self.assertTrue(config_record.scale_numeric)
        self.assertEqual(config_record.scaling_strategy, ScalingStrategy.STANDARD.value)
        self.assertEqual(config_record.remainder, "drop")

    def test_plan_uses_quality_issues_from_phase_two(self) -> None:
        frame = frame_mixed()
        issues = detect_data_quality_issues(frame, target_column="churn")
        config_record = plan_preprocessing(
            frame, target_column="churn", data_quality_issues=issues
        )
        self.assertEqual(_excluded_map(config_record)["record_id"],
                         ExclusionReason.IDENTIFIER_LIKE.value)

    def test_plan_without_any_usable_feature_raises(self) -> None:
        frame = frame_identifier_only()
        with self.assertRaises(PreprocessingConfigError):
            plan_preprocessing(frame, target_column="label")

    def test_numeric_only_dataset_has_no_categorical_branch(self) -> None:
        frame = frame_clean_numerical()
        frame["label"] = [index % 2 for index in range(frame.shape[0])]
        config_record = plan_preprocessing(frame, target_column="label")
        self.assertEqual(config_record.categorical_features, [])
        transformer = build_column_transformer(config_record)
        self.assertEqual([name for name, _, _ in transformer.transformers], ["numerical"])

    def test_categorical_only_dataset_has_no_numerical_branch(self) -> None:
        frame = frame_clean_categorical()
        config_record = plan_preprocessing(frame, target_column="label")
        self.assertEqual(config_record.numerical_features, [])
        transformer = build_column_transformer(config_record)
        self.assertEqual([name for name, _, _ in transformer.transformers], ["categorical"])
        transformed = transformer.fit_transform(frame.drop(columns=["label"]))
        self.assertEqual(transformed.shape[1], 6)


class TestConfigValidation(unittest.TestCase):
    """14. Invalid preprocessing configurations are rejected."""

    def _config(self, **overrides) -> PreprocessingConfig:
        base = {
            "numerical_features": ["age"],
            "categorical_features": ["city"],
            "excluded_features": [],
            "target_column": "churn",
        }
        base.update(overrides)
        return PreprocessingConfig(**base)

    def test_valid_configuration_passes(self) -> None:
        self.assertIsNone(validate_preprocessing_config(self._config()))

    def test_no_features_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(numerical_features=[], categorical_features=[]))

    def test_overlapping_feature_groups_raise(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(categorical_features=["age", "city"]))

    def test_excluded_column_still_selected_raises(self) -> None:
        from utils.state import ExcludedFeature

        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(
                self._config(
                    excluded_features=[
                        ExcludedFeature(column="age", reason=ExclusionReason.IDENTIFIER_LIKE.value)
                    ]
                )
            )

    def test_invalid_numerical_imputation_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(numerical_imputation="magic"))

    def test_invalid_categorical_imputation_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(categorical_imputation="magic"))

    def test_invalid_encoding_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(encoding="ordinal"))

    def test_handle_unknown_must_be_ignore(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(encoding_handle_unknown="error"))

    def test_invalid_scaling_strategy_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(scaling_strategy="robust"))

    def test_inconsistent_scaling_configuration_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(
                self._config(scale_numeric=True, scaling_strategy=ScalingStrategy.NONE.value)
            )

    def test_passthrough_remainder_raises(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(remainder="passthrough"))

    def test_invalid_duplicate_strategy_raises(self) -> None:
        from utils.state import DuplicateHandling

        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(
                self._config(duplicate_handling=DuplicateHandling(strategy="nonsense"))
            )

    def test_features_missing_from_dataframe_raise(self) -> None:
        frame = frame_clean_numerical()
        with self.assertRaises(PreprocessingConfigError):
            validate_preprocessing_config(self._config(), dataframe=frame)

    def test_invalid_configuration_is_rejected_by_the_builder(self) -> None:
        with self.assertRaises(PreprocessingConfigError):
            build_column_transformer(self._config(numerical_features=[], categorical_features=[]))


class TestDataLeakage(unittest.TestCase):
    """18. The pipeline must never be fitted on data outside the training split."""

    def setUp(self) -> None:
        # The second half is shifted on purpose: if the pipeline were fitted on
        # the full dataset, the learned statistics would not match the training
        # split and this test would fail.
        rows = 60
        frame = pd.DataFrame(
            {
                "record_id": range(1, rows + 1),
                "age": [float(20 + (index % 30)) for index in range(rows)],
                "bmi": [20.0 + (index % 10) for index in range(rows)],
                "city": ["north", "south"] * (rows // 2),
                "churn": [index % 2 for index in range(rows)],
            }
        )
        # Shift bmi in the test half so its statistics differ clearly from the
        # training half. age is left untouched so it stays a normal feature
        # (a column that is unique per row would be excluded as an identifier).
        frame.loc[rows // 2 :, "bmi"] = frame.loc[rows // 2 :, "bmi"] + 500.0
        frame.loc[rows // 2 :, "city"] = "west"

        self.frame = frame
        self.config = plan_preprocessing(frame, target_column="churn")
        self.train = frame.iloc[: rows // 2].drop(columns=["churn"])
        self.test = frame.iloc[rows // 2 :].drop(columns=["churn"])

    def test_pipeline_is_not_fitted_before_explicit_fit(self) -> None:
        transformer = build_column_transformer(self.config)
        with self.assertRaises(NotFittedError):
            check_is_fitted(transformer)

    def test_imputer_statistics_come_from_training_data_only(self) -> None:
        transformer = build_column_transformer(self.config)
        transformer.fit(self.train)

        imputer = transformer.named_transformers_["numerical"].named_steps["imputer"]
        expected = self.train["age"].median()
        self.assertAlmostEqual(float(imputer.statistics_[0]), float(expected), places=6)

    def test_scaler_statistics_come_from_training_data_only(self) -> None:
        transformer = build_column_transformer(self.config)
        transformer.fit(self.train)

        scaler = transformer.named_transformers_["numerical"].named_steps["scaler"]
        self.assertTrue(
            np.allclose(scaler.mean_, self.train[["age", "bmi"]].mean().to_numpy())
        )

    def test_statistics_differ_from_full_dataset_statistics(self) -> None:
        transformer = build_column_transformer(self.config)
        transformer.fit(self.train)

        scaler = transformer.named_transformers_["numerical"].named_steps["scaler"]
        full_dataset_mean = self.frame[["age", "bmi"]].mean().to_numpy()
        # If the pipeline had been fitted on everything, these would be equal.
        self.assertFalse(np.allclose(scaler.mean_, full_dataset_mean))

    def test_encoder_categories_come_from_training_data_only(self) -> None:
        transformer = build_column_transformer(self.config)
        transformer.fit(self.train)

        encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
        categories = set(encoder.categories_[0])
        # 'west' only exists in the test split, so it cannot be a learned category.
        self.assertEqual(categories, {"north", "south"})
        self.assertNotIn("west", categories)

    def test_unseen_test_category_is_transformed_not_fitted(self) -> None:
        transformer = build_column_transformer(self.config)
        transformer.fit(self.train)
        transformed = transformer.transform(self.test)
        self.assertEqual(transformed.shape[0], self.test.shape[0])


class TestSourceDataIsNeverModified(unittest.TestCase):
    """The preprocessing decisions must not change the file on disk."""

    def test_planning_does_not_modify_the_dataframe(self) -> None:
        frame = frame_mixed()
        before_shape = frame.shape
        before_missing = int(frame.isna().sum().sum())
        plan_preprocessing(frame, target_column="churn")
        self.assertEqual(frame.shape, before_shape)
        self.assertEqual(int(frame.isna().sum().sum()), before_missing)
        self.assertIn("record_id", frame.columns)

    def test_planning_does_not_modify_the_csv_file(self) -> None:
        from tests.helpers import file_hash

        with TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp), "mixed.csv", frame_mixed())
            before = file_hash(path)
            plan_preprocessing(load_csv(path), target_column="churn")
            self.assertEqual(file_hash(path), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)