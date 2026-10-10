"""Tests for the reusable training and evaluation tools (Phase 4).

Covers required checks 6-26: loading, X/y separation, splitting and its safe
fallback, pipeline construction and fitting, missing values, categorical and
unknown categories, identifier exclusion, duplicate strategy, metrics, confusion
matrix, cross-validation (including its small-class handling), JSON-safe
experiment records and baseline ranking.

The leakage tests are the important ones: they use a dataset whose *test* rows
have statistics that differ sharply from the training rows, then assert that the
fitted imputer/scaler/encoder learned from the training rows only.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.preprocessing_tools import plan_preprocessing, remove_duplicate_rows
from tools.training_tools import (
    DatasetTooSmallError,
    TrainingDataError,
    build_cross_validation_config,
    build_experiment_id,
    build_model_pipeline,
    build_preprocessing_variants,
    calculate_confusion_matrix,
    class_distribution,
    count_transformed_features,
    determine_safe_n_splits,
    evaluate_classifier,
    load_training_dataframe,
    prepare_features_and_target,
    rank_experiments,
    run_cross_validation,
    select_best_experiment,
    split_dataset,
)
from utils.state import (
    BestModelInfo,
    ConfusionMatrixData,
    ExperimentResult,
    ExperimentStatus,
    ModelComparison,
    PreprocessingConfig,
    ScalingStrategy,
    to_serializable,
)

from tests.helpers import (
    BINARY_DATASET,
    frame_mixed,
    frame_with_duplicates,
    write_csv,
)


def plan(frame: pd.DataFrame, target: str = "churn", **kwargs) -> PreprocessingConfig:
    """Plan a preprocessing configuration for a synthetic frame."""
    return plan_preprocessing(frame, target_column=target, **kwargs)


def make_training_frame(rows: int = 60) -> pd.DataFrame:
    """A learnable frame with one identifier, one numerical and one categorical feature.

    The identifier column is unique per row on purpose, so the Phase 3
    exclusion rules must drop it before it can reach a model.
    """
    return pd.DataFrame(
        {
            "record_id": [f"R{index:04d}" for index in range(rows)],
            "age": [20 + (index % 30) for index in range(rows)],
            "city": [["north", "south", "east"][index % 3] for index in range(rows)],
            "churn": [index % 2 for index in range(rows)],
        }
    )


def split_for(frame: pd.DataFrame, target: str = "churn", test_size: float = 0.2):
    """Plan, separate and split a frame, returning (config, X, y, split)."""
    config_record = plan(frame, target=target)
    X, y = prepare_features_and_target(
        frame, target, [item.column for item in config_record.excluded_features]
    )
    split = split_dataset(X, y, test_size=test_size)
    return config_record, X, y, split


def _branch(pipeline, name: str, fitted: bool = False):
    """Return a ColumnTransformer branch by name.

    Args:
        pipeline: A model Pipeline.
        name: Branch name, "numerical" or "categorical".
        fitted: Read the fitted branches (`transformers_`) instead of the spec.
    """
    transformer = pipeline.named_steps["preprocessor"]
    spec = transformer.transformers_ if fitted else transformer.transformers
    for branch_name, branch, _ in spec:
        if branch_name == name:
            return branch
    raise AssertionError(f"branch '{name}' not found")


def _numerical_steps(pipeline):
    """Return the step names of the (possibly unfitted) numerical branch."""
    return list(_branch(pipeline, "numerical").named_steps)


class TestLoadTrainingDataframe(unittest.TestCase):
    """6. Dataset loading works and never hides an error."""

    def test_loads_the_committed_sample_dataset(self) -> None:
        frame = load_training_dataframe(str(BINARY_DATASET))
        self.assertEqual(frame.shape[0], 123)
        self.assertIn("Outcome", frame.columns)

    def test_missing_file_raises_a_clear_error(self) -> None:
        with self.assertRaises(TrainingDataError) as context:
            load_training_dataframe("definitely_missing.csv")
        self.assertIn("does not exist", str(context.exception))

    def test_directory_instead_of_csv_raises(self) -> None:
        with self.assertRaises(TrainingDataError):
            load_training_dataframe(str(BINARY_DATASET.parent))

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(TrainingDataError) as context:
                load_training_dataframe(str(path))
            self.assertIn("empty", str(context.exception).lower())

    def test_loading_does_not_modify_the_source_file(self) -> None:
        from tests.helpers import file_hash

        before = file_hash(BINARY_DATASET)
        load_training_dataframe(str(BINARY_DATASET))
        self.assertEqual(file_hash(BINARY_DATASET), before)


class TestFeatureTargetSeparation(unittest.TestCase):
    """7-8. The target is separated and can never enter X."""

    def setUp(self) -> None:
        self.frame = make_training_frame()

    def test_target_and_features_are_returned(self) -> None:
        X, y = prepare_features_and_target(self.frame, "churn", ["record_id"])
        self.assertEqual(X.shape[0], self.frame.shape[0])
        self.assertEqual(y.name, "churn")

    def test_target_never_enters_the_feature_matrix(self) -> None:
        X, y = prepare_features_and_target(self.frame, "churn", ["record_id"])
        self.assertNotIn("churn", X.columns)
        self.assertEqual(y.name, "churn")
        self.assertEqual(len(y), self.frame.shape[0])

    def test_excluded_columns_are_dropped(self) -> None:
        X, _ = prepare_features_and_target(self.frame, "churn", ["record_id"])
        self.assertNotIn("record_id", X.columns)

    def test_missing_target_raises_a_clear_error(self) -> None:
        with self.assertRaises(TrainingDataError) as context:
            prepare_features_and_target(self.frame, "not_a_column", [])
        self.assertIn("not present", str(context.exception))

    def test_no_target_name_raises(self) -> None:
        with self.assertRaises(TrainingDataError):
            prepare_features_and_target(self.frame, None, [])

    def test_missing_target_values_raise(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "churn"] = None
        with self.assertRaises(TrainingDataError) as context:
            prepare_features_and_target(frame, "churn", ["record_id"])
        self.assertIn("missing", str(context.exception).lower())

    def test_constant_target_raises(self) -> None:
        frame = self.frame.copy()
        frame["churn"] = 1
        with self.assertRaises(TrainingDataError) as context:
            prepare_features_and_target(frame, "churn", ["record_id"])
        self.assertIn("single distinct value", str(context.exception))

    def test_no_feature_left_raises(self) -> None:
        with self.assertRaises(TrainingDataError) as context:
            prepare_features_and_target(
                self.frame, "churn", [item for item in self.frame.columns if item != "churn"]
            )
        self.assertIn("No feature column", str(context.exception))

    def test_duplicate_target_in_exclusions_does_not_break(self) -> None:
        X, _ = prepare_features_and_target(self.frame, "churn", ["churn", "record_id"])
        self.assertNotIn("churn", X.columns)

    def test_class_distribution_is_json_safe(self) -> None:
        _, y = prepare_features_and_target(self.frame, "churn", ["record_id"])
        distribution = class_distribution(y)
        self.assertEqual(distribution, {"0": 30, "1": 30})
        json.dumps(distribution)


class TestSplitDataset(unittest.TestCase):
    """9-11. Splitting works, stratifies when possible and falls back safely."""

    def setUp(self) -> None:
        self.frame = make_training_frame(60)
        self.config, self.X, self.y, self.split = split_for(self.frame)

    def test_split_produces_the_requested_sizes(self) -> None:
        self.assertEqual(self.split.train_samples, 48)
        self.assertEqual(self.split.test_samples, 12)
        self.assertEqual(self.split.total_samples, 60)
        self.assertEqual(self.split.train_samples + self.split.test_samples, 60)

    def test_splits_do_not_overlap(self) -> None:
        train_ids = set(self.split.X_train.index)
        test_ids = set(self.split.X_test.index)
        self.assertFalse(train_ids & test_ids)

    def test_targets_stay_aligned_with_features(self) -> None:
        for features, target in (
            (self.split.X_train, self.split.y_train),
            (self.split.X_test, self.split.y_test),
        ):
            self.assertEqual(list(features.index), list(target.index))

    def test_stratification_is_used_when_possible(self) -> None:
        self.assertTrue(self.split.stratified)
        self.assertEqual(self.split.reason, "")

    def test_stratification_preserves_the_class_ratio(self) -> None:
        self.assertEqual(self.split.class_distribution_train, {"0": 24, "1": 24})
        self.assertEqual(self.split.class_distribution_test, {"0": 6, "1": 6})

    def test_split_is_reproducible(self) -> None:
        again = split_dataset(self.X, self.y, test_size=0.2, random_state=42)
        self.assertEqual(list(self.split.X_test.index), list(again.X_test.index))

    def test_split_changes_with_a_different_seed(self) -> None:
        again = split_dataset(self.X, self.y, test_size=0.2, random_state=7)
        self.assertNotEqual(list(self.split.X_test.index), list(again.X_test.index))

    def test_fallback_for_a_very_small_class(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [20 + (index % 7) for index in range(20)],
                "churn": [0] * 19 + [1],
            }
        )
        X, y = prepare_features_and_target(frame, "churn", [])
        split = split_dataset(X, y, test_size=0.25)
        self.assertFalse(split.stratified)
        self.assertIn("stratified split skipped", split.reason)
        self.assertIn("smallest class", split.reason)
        self.assertEqual(split.train_samples + split.test_samples, 20)

    def test_fallback_for_a_tiny_dataset(self) -> None:
        frame = pd.DataFrame(
            {"age": [1.0, 2.0, 3.0, 4.0], "churn": [0, 1, 0, 1]}
        )
        X, y = prepare_features_and_target(frame, "churn", [])
        split = split_dataset(X, y, test_size=0.5)
        self.assertFalse(split.stratified)
        self.assertIn("plain random split", split.reason)

    def test_stratification_can_be_disabled(self) -> None:
        split = split_dataset(self.X, self.y, test_size=0.2, stratify_if_possible=False)
        self.assertFalse(split.stratified)

    def test_single_row_dataset_raises(self) -> None:
        frame = pd.DataFrame({"age": [1.0], "churn": [0]})
        with self.assertRaises(DatasetTooSmallError):
            split_dataset(frame[["age"]], frame["churn"], test_size=0.2)

    def test_impossible_test_size_raises_a_clear_error(self) -> None:
        with self.assertRaises(DatasetTooSmallError) as context:
            split_dataset(self.X, self.y, test_size=0.99)
        self.assertIn("not possible", str(context.exception))


class TestPipelineConstruction(unittest.TestCase):
    """12-14. The Pipeline holds preprocessing + model and transforms test data."""

    def setUp(self) -> None:
        self.frame = make_training_frame(60)
        self.config, self.X, self.y, self.split = split_for(self.frame)

    def test_pipeline_has_exactly_two_named_steps(self) -> None:
        pipeline = build_model_pipeline(self.config, "Random Forest")
        self.assertEqual(
            list(pipeline.named_steps), ["preprocessor", "model"]
        )

    def test_preprocessor_is_a_column_transformer(self) -> None:
        from sklearn.compose import ColumnTransformer

        pipeline = build_model_pipeline(self.config, "Random Forest")
        self.assertIsInstance(pipeline.named_steps["preprocessor"], ColumnTransformer)

    def test_model_step_is_the_requested_estimator(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        pipeline = build_model_pipeline(self.config, "Random Forest")
        self.assertIsInstance(pipeline.named_steps["model"], RandomForestClassifier)

    def test_pipeline_starts_unfitted(self) -> None:
        pipeline = build_model_pipeline(self.config, "Decision Tree")
        self.assertIs(count_transformed_features(pipeline), None)

    def test_pipeline_fits_on_the_training_split(self) -> None:
        pipeline = build_model_pipeline(self.config, "Random Forest")
        pipeline.fit(self.split.X_train, self.split.y_train)
        predictions = pipeline.predict(self.split.X_test)
        self.assertEqual(len(predictions), self.split.test_samples)
        self.assertEqual(count_transformed_features(pipeline), 4)

    def test_test_data_is_transformed_through_the_fitted_pipeline(self) -> None:
        pipeline = build_model_pipeline(self.config, "Random Forest")
        pipeline.fit(self.split.X_train, self.split.y_train)
        # The transformer has only seen the training rows, so its statistics
        # must match the training split and never the test split.
        transformer = pipeline.named_steps["preprocessor"]
        learned_median = transformer.named_transformers_["numerical"].named_steps["imputer"].statistics_
        self.assertEqual(
            list(learned_median), [float(self.split.X_train["age"].median())]
        )

    def test_pipeline_does_not_modify_the_input_frames(self) -> None:
        before_train = self.split.X_train.copy()
        before_test = self.split.X_test.copy()
        pipeline = build_model_pipeline(self.config, "Random Forest")
        pipeline.fit(self.split.X_train, self.split.y_train)
        pipeline.predict(self.split.X_test)
        pd.testing.assert_frame_equal(before_train, self.split.X_train)
        pd.testing.assert_frame_equal(before_test, self.split.X_test)

    def test_pipeline_handles_missing_values(self) -> None:
        frame = make_training_frame(60)
        frame.loc[[0, 1, 2], "age"] = None
        config_record, X, y, split = split_for(frame)
        pipeline = build_model_pipeline(config_record, "Random Forest")
        pipeline.fit(split.X_train, split.y_train)
        predictions = pipeline.predict(split.X_test)
        self.assertEqual(len(predictions), split.test_samples)
        self.assertTrue(set(predictions).issubset(set(y)))

    def test_pipeline_handles_categorical_features(self) -> None:
        pipeline = build_model_pipeline(self.config, "Random Forest")
        pipeline.fit(self.split.X_train, self.split.y_train)
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        self.assertIn("categorical__city_north", names)
        self.assertIn("categorical__city_south", names)
        self.assertIn("numerical__age", names)

    def test_pipeline_handles_unknown_categories_in_test_data(self) -> None:
        config_record, X, y, split = split_for(self.frame)
        unknown = split.X_test.copy()
        unknown.loc[unknown.index[0], "city"] = "atlantis"
        pipeline = build_model_pipeline(config_record, "Logistic Regression")
        pipeline.fit(split.X_train, split.y_train)
        predictions = pipeline.predict(unknown)
        self.assertEqual(len(predictions), len(unknown))

    def test_scaled_model_receives_a_scaler(self) -> None:
        steps = _numerical_steps(build_model_pipeline(self.config, "Logistic Regression"))
        self.assertIn("scaler", steps)

    def test_tree_model_does_not_receive_a_scaler(self) -> None:
        steps = _numerical_steps(build_model_pipeline(self.config, "Random Forest"))
        self.assertNotIn("scaler", steps)

    def test_knn_and_svm_also_receive_a_scaler(self) -> None:
        for name in ("K-Nearest Neighbors", "Support Vector Machine"):
            self.assertIn(
                "scaler", _numerical_steps(build_model_pipeline(self.config, name)), name
            )

    def test_scaling_variant_does_not_change_the_stored_configuration(self) -> None:
        self.assertTrue(self.config.scale_numeric)
        build_model_pipeline(self.config, "Random Forest")
        self.assertTrue(self.config.scale_numeric)
        self.assertEqual(self.config.scaling_strategy, ScalingStrategy.STANDARD.value)

    def test_preprocessing_variants_return_config_and_transformer(self) -> None:
        variant, transformer = build_preprocessing_variants(self.config, False)
        self.assertFalse(variant.scale_numeric)
        branch_names = [name for name, _, _ in transformer.transformers]
        self.assertEqual(branch_names, ["numerical", "categorical"])

    def test_unusable_configuration_raises(self) -> None:
        from tools.model_tools import ModelConfigurationError

        with self.assertRaises(ModelConfigurationError):
            build_model_pipeline(None, "Random Forest")


class TestLeakagePrevention(unittest.TestCase):
    """Preprocessing must learn from the training split only, never the test.

    The split is built **by hand** instead of with `split_dataset`, so the two
    halves are guaranteed to be far apart statistically. If the pipeline were
    ever fitted on the whole dataset, every learned statistic in these tests
    would move towards the test range and the assertions would fail loudly.
    """

    @staticmethod
    def train_frame() -> pd.DataFrame:
        """Six training rows with small values (10.0 - 20.0)."""
        return pd.DataFrame(
            {
                "value": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )

    @staticmethod
    def test_frame() -> pd.DataFrame:
        """Twenty test rows with very large values (1000.0 - 1009.0, repeated).

        The values repeat on purpose: a column that is unique per row would be
        excluded as an identifier, which is the correct Phase 3 behaviour but
        would make this fixture useless for comparing statistics.
        """
        return pd.DataFrame(
            {
                "value": [1000.0 + (index % 10) for index in range(20)],
                "label": [index % 2 for index in range(20)],
            }
        )

    @classmethod
    def whole_frame(cls) -> pd.DataFrame:
        """The concatenation of both halves, i.e. what a leak would fit on."""
        return pd.concat([cls.train_frame(), cls.test_frame()], ignore_index=True)

    def test_fixture_is_strongly_biased(self) -> None:
        """Guard: the two halves must be far apart or the other tests are void."""
        train, test = self.train_frame(), self.test_frame()
        self.assertGreater(float(test["value"].mean()) - float(train["value"].mean()), 900.0)
        self.assertGreater(
            float(self.whole_frame()["value"].median()) - float(train["value"].median()), 500.0
        )

    def test_configuration_keeps_the_feature(self) -> None:
        """Guard: the feature must survive the Phase 3 exclusion rules."""
        config_record = plan_preprocessing(self.whole_frame(), target_column="label")
        self.assertEqual(config_record.numerical_features, ["value"])

    def test_imputer_learns_from_training_rows_only(self) -> None:
        train, test = self.train_frame(), self.test_frame()
        whole = self.whole_frame()

        # One missing training value so an imputation statistic exists.
        train = train.copy()
        train.loc[train.index[0], "value"] = None

        config_record = plan_preprocessing(whole, target_column="label")
        pipeline = build_model_pipeline(config_record, "Random Forest")
        pipeline.fit(train, train["label"])

        learned = float(
            pipeline.named_steps["preprocessor"].named_transformers_["numerical"]
            .named_steps["imputer"].statistics_[0]
        )
        # Fitted on the training rows: the median of the six training values.
        self.assertAlmostEqual(learned, float(train["value"].median()), places=6)
        # Fitting on the whole dataset would have produced the median of the
        # combined frame, which is completely different. That is the leak.
        self.assertGreater(float(whole["value"].median()) - learned, 500.0)
        self.assertLess(learned, 100.0)

    def test_scaler_learns_from_training_rows_only(self) -> None:
        train, test, whole = self.train_frame(), self.test_frame(), self.whole_frame()

        config_record = plan_preprocessing(whole, target_column="label", scale_numeric=True)
        pipeline = build_model_pipeline(config_record, "Logistic Regression")
        pipeline.fit(train, train["label"])

        scaler = pipeline.named_steps["preprocessor"].named_transformers_["numerical"] \
            .named_steps["scaler"]
        learned_mean = float(scaler.mean_[0])

        self.assertAlmostEqual(learned_mean, float(train["value"].mean()), places=6)
        self.assertAlmostEqual(learned_mean, 15.0, places=6)
        # A leak would push this towards the combined mean (~780) or the test mean.
        self.assertGreater(float(test["value"].mean()) - learned_mean, 900.0)
        self.assertGreater(float(whole["value"].mean()) - learned_mean, 500.0)

    def test_encoder_learns_categories_from_training_rows_only(self) -> None:
        train = pd.DataFrame(
            {
                "city": ["north", "south"] * 3,
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "city": ["atlantis"] * 10,
                "label": [0, 1] * 5,
            }
        )
        whole = pd.concat([train, test], ignore_index=True)

        config_record = plan_preprocessing(whole, target_column="label")
        pipeline = build_model_pipeline(config_record, "Random Forest")
        pipeline.fit(train, train["label"])

        encoder = _branch(pipeline, "categorical", fitted=True)
        encoder_names = list(encoder.get_feature_names_out(["city"]))
        self.assertTrue(any("north" in name for name in encoder_names))
        self.assertTrue(any("south" in name for name in encoder_names))
        # "atlantis" only exists in the test half, so the encoder cannot know it.
        self.assertFalse([name for name in encoder_names if "atlantis" in name])

    def test_categorical_imputer_uses_the_training_mode(self) -> None:
        train = pd.DataFrame(
            {
                "city": ["north"] * 4 + [None, "south"],
                "label": [0, 1, 0, 1, 0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "city": ["atlantis"] * 4,
                "label": [0, 1, 0, 1],
            }
        )
        whole = pd.concat([train, test], ignore_index=True)

        config_record = plan_preprocessing(whole, target_column="label")
        pipeline = build_model_pipeline(config_record, "Random Forest")
        pipeline.fit(train, train["label"])

        imputer = _branch(pipeline, "categorical", fitted=True).named_steps["imputer"]
        self.assertEqual(list(imputer.statistics_), ["north"])

    def test_standardised_training_rows_have_zero_mean(self) -> None:
        train = self.train_frame()
        config_record = plan_preprocessing(
            self.whole_frame(), target_column="label", scale_numeric=True
        )
        pipeline = build_model_pipeline(config_record, "Logistic Regression")
        pipeline.fit(train, train["label"])

        transformed_train = pipeline.named_steps["preprocessor"].transform(train)
        # Only true if the scaler was fitted on these very rows.
        self.assertAlmostEqual(float(np.mean(transformed_train)), 0.0, places=6)
        self.assertAlmostEqual(float(np.std(transformed_train)), 1.0, places=6)


class TestDuplicateStrategy(unittest.TestCase):
    """19. The Phase 3 duplicate strategy is respected."""

    def test_remove_exact_duplicates_removes_rows(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        deduplicated, removed = remove_duplicate_rows(frame, strategy="remove_exact_duplicates")
        self.assertEqual(removed, 3)
        self.assertEqual(deduplicated.shape[0], 20)

    def test_keep_all_retains_every_row(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        kept, removed = remove_duplicate_rows(frame, strategy="keep_all")
        self.assertEqual(removed, 0)
        self.assertEqual(kept.shape[0], 23)

    def test_duplicate_removal_does_not_modify_the_source_frame(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        before = frame.copy()
        remove_duplicate_rows(frame, strategy="remove_exact_duplicates")
        pd.testing.assert_frame_equal(before, frame)

    def test_duplicate_removal_does_not_touch_the_csv_file(self) -> None:
        from tests.helpers import file_hash

        frame = frame_with_duplicates(rows=20, duplicates=3)
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(Path(directory), "dups.csv", frame)
            before = file_hash(path)
            reloaded = load_training_dataframe(str(path))
            remove_duplicate_rows(reloaded, strategy="remove_exact_duplicates")
            self.assertEqual(file_hash(path), before)


class TestIdentifierExclusion(unittest.TestCase):
    """18. Identifier columns stay out of the feature matrix."""

    def test_identifier_column_is_excluded_from_the_pipeline(self) -> None:
        frame = make_training_frame(60)
        config_record = plan(frame, target="churn")
        self.assertIn("record_id", [item.column for item in config_record.excluded_features])

        X, _ = prepare_features_and_target(
            frame, "churn", [item.column for item in config_record.excluded_features]
        )
        self.assertNotIn("record_id", X.columns)

        pipeline = build_model_pipeline(config_record, "Random Forest")
        pipeline.fit(X, frame["churn"])
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        self.assertFalse([name for name in names if "record_id" in name])

    def test_feature_names_exclude_the_target(self) -> None:
        frame = make_training_frame(60)
        config_record = plan(frame, target="churn")
        X, y = prepare_features_and_target(
            frame, "churn", [item.column for item in config_record.excluded_features]
        )
        pipeline = build_model_pipeline(config_record, "Random Forest")
        pipeline.fit(X, y)
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        self.assertFalse([name for name in names if "churn" in name])


class TestMetricsAndConfusionMatrix(unittest.TestCase):
    """20-21. Metrics and confusion matrix are produced."""

    def test_perfect_predictions_score_one(self) -> None:
        metrics = evaluate_classifier([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_metrics_are_computed_for_a_mixed_prediction(self) -> None:
        # y_true = [0, 1, 0, 1, 0, 1], y_pred = [0, 0, 1, 1, 0, 0]
        # 3 of 6 correct; both classes have precision 0.5, so the weighted
        # precision and recall are 0.5 as well.
        metrics = evaluate_classifier([0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1_score"], (0.5714285714 + 0.4) / 2, places=6)
        self.assertLess(metrics["f1_score"], metrics["accuracy"])

    def test_metrics_use_weighted_average(self) -> None:
        multiclass = evaluate_classifier([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], average="weighted")
        macro = evaluate_classifier([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 1, 2], average="weighted")
        self.assertEqual(set(multiclass), {"accuracy", "precision", "recall", "f1_score"})
        self.assertGreater(multiclass["f1_score"], macro["f1_score"])

    def test_zero_division_keeps_metrics_defined(self) -> None:
        # The model never predicts class 1: precision/recall must stay defined.
        metrics = evaluate_classifier([0, 0, 1, 1], [0, 0, 0, 0])
        for name, value in metrics.items():
            self.assertIsInstance(value, float)
            self.assertFalse(np.isnan(value))

    def test_metrics_are_floats_between_zero_and_one(self) -> None:
        metrics = evaluate_classifier([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])
        for value in metrics.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_confusion_matrix_shape_and_total(self) -> None:
        matrix = calculate_confusion_matrix([0, 1, 0, 1, 1], [0, 1, 1, 1, 1])
        self.assertEqual(matrix.labels, ["0", "1"])
        # Row 0 (true 0): one predicted 0, one predicted 1.
        # Row 1 (true 1): none predicted 0, all three predicted 1.
        self.assertEqual(matrix.matrix, [[1, 1], [0, 3]])
        self.assertEqual(matrix.total, 5)

    def test_confusion_matrix_labels_are_not_duplicated(self) -> None:
        matrix = calculate_confusion_matrix([0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(len(matrix.labels), len(set(matrix.labels)))
        self.assertEqual(len(matrix.matrix), 2)

    def test_confusion_matrix_is_json_safe(self) -> None:
        matrix = calculate_confusion_matrix(
            ["b", "a", "c", "c"], ["b", "a", "c", "a"]
        )
        self.assertIsInstance(matrix, ConfusionMatrixData)
        payload = {"labels": matrix.labels, "matrix": matrix.matrix, "total": matrix.total}
        json.dumps(payload)
        self.assertTrue(all(isinstance(item, str) for item in matrix.labels))

    def test_confusion_matrix_for_a_predicted_only_class(self) -> None:
        matrix = calculate_confusion_matrix([0, 1, 1, 1], [1, 1, 1, 1])
        self.assertEqual(sorted(matrix.labels), ["0", "1"])
        self.assertEqual(matrix.total, 4)

    def test_explicit_labels_are_respected(self) -> None:
        matrix = calculate_confusion_matrix([0, 1], [0, 1], labels=[1, 0])
        self.assertEqual(matrix.labels, ["1", "0"])


class TestCrossValidation(unittest.TestCase):
    """22-24. Cross-validation runs through the Pipeline and adapts to small data."""

    def setUp(self) -> None:
        self.frame = make_training_frame(60)
        self.config, self.X, self.y, self.split = split_for(self.frame)

    def test_default_config_uses_five_stratified_folds(self) -> None:
        cv_config = build_cross_validation_config(self.y)
        self.assertTrue(cv_config.enabled)
        self.assertEqual(cv_config.n_splits, 5)
        self.assertTrue(cv_config.shuffle)
        self.assertEqual(cv_config.random_state, 42)
        self.assertEqual(cv_config.adjusted_reason, "")
        self.assertEqual(cv_config.scoring, {"accuracy": "accuracy", "f1": "f1_weighted"})

    def test_cross_validation_returns_mean_and_std(self) -> None:
        pipeline = build_model_pipeline(self.config, "Random Forest")
        cv_config = build_cross_validation_config(self.split.y_train)
        result = run_cross_validation(
            pipeline, self.split.X_train, self.split.y_train, cv_config, "Random Forest"
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.n_splits, 5)
        self.assertEqual(len(result.fold_accuracies), 5)
        self.assertIsNotNone(result.accuracy_mean)
        self.assertIsNotNone(result.accuracy_std)
        self.assertIsNotNone(result.f1_mean)
        self.assertGreaterEqual(result.accuracy_std, 0.0)

    def test_cross_validation_uses_the_pipeline(self) -> None:
        """The unfitted pipeline is passed in, so each fold refits preprocessing."""
        pipeline = build_model_pipeline(self.config, "Logistic Regression")
        self.assertIs(count_transformed_features(pipeline), None)
        cv_config = build_cross_validation_config(self.split.y_train)
        run_cross_validation(
            pipeline, self.split.X_train, self.split.y_train, cv_config, "Logistic Regression"
        )
        # cross_validate clones the estimator, so the caller's pipeline is still
        # unfitted: cross-validation cannot leak state back out of the folds.
        self.assertIs(count_transformed_features(pipeline), None)

    def test_cross_validation_is_reproducible(self) -> None:
        cv_config = build_cross_validation_config(self.split.y_train)
        first = run_cross_validation(
            build_model_pipeline(self.config, "Random Forest"),
            self.split.X_train, self.split.y_train, cv_config, "Random Forest",
        )
        second = run_cross_validation(
            build_model_pipeline(self.config, "Random Forest"),
            self.split.X_train, self.split.y_train, cv_config, "Random Forest",
        )
        self.assertEqual(first.accuracy_mean, second.accuracy_mean)
        self.assertEqual(first.f1_mean, second.f1_mean)

    def test_cross_validation_never_sees_the_test_split(self) -> None:
        """CV over the training rows must ignore the extreme test rows entirely.

        If cross-validation were fitted on the whole dataset the extreme rows
        would leak into every fold; here they are simply not passed in, so the
        scores stay in the normal range instead of collapsing.
        """
        whole = TestLeakagePrevention.whole_frame()
        train = TestLeakagePrevention.train_frame()
        test = TestLeakagePrevention.test_frame()

        config_record = plan_preprocessing(whole, target_column="label", scale_numeric=True)
        cv_config = build_cross_validation_config(train["label"], n_splits=3)
        result = run_cross_validation(
            build_model_pipeline(config_record, "Logistic Regression"),
            train,
            train["label"],
            cv_config,
            "Logistic Regression",
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.n_splits, 3)
        for value in result.fold_accuracies:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        # The caller only ever passed training rows to the CV call.
        self.assertEqual(len(train), 6)
        self.assertGreater(float(test["value"].mean()), 1000.0)

    def test_disabled_cross_validation_is_reported(self) -> None:
        cv_config = build_cross_validation_config(self.y, enabled=False)
        self.assertFalse(cv_config.enabled)
        result = run_cross_validation(
            build_model_pipeline(self.config, "Random Forest"),
            self.split.X_train,
            self.split.y_train,
            cv_config,
            "Random Forest",
        )
        self.assertEqual(result.status, "skipped")
        self.assertTrue(result.reason)
        self.assertIsNone(result.accuracy_mean)

    def test_folds_are_reduced_for_a_small_class(self) -> None:
        target = pd.Series([0] * 13 + [1] * 3)
        folds, reason = determine_safe_n_splits(target, requested=5)
        self.assertEqual(folds, 3)
        self.assertIn("reduced from 5 to 3", reason)

        cv_config = build_cross_validation_config(target)
        self.assertEqual(cv_config.n_splits, 3)
        self.assertIn("smallest class", cv_config.adjusted_reason)

    def test_folds_are_reduced_to_two_as_the_minimum(self) -> None:
        target = pd.Series([0] * 9 + [1] * 2)
        folds, reason = determine_safe_n_splits(target, requested=5)
        self.assertEqual(folds, 2)
        self.assertIn("reduced from 5 to 2", reason)

    def test_cross_validation_runs_with_reduced_folds(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [20 + (index % 9) for index in range(15)],
                "churn": [0] * 12 + [1] * 3,
            }
        )
        config_record = plan_preprocessing(frame, target_column="churn")
        X, y = prepare_features_and_target(frame, "churn", [])
        split = split_dataset(X, y, test_size=0.3)
        cv_config = build_cross_validation_config(split.y_train)
        result = run_cross_validation(
            build_model_pipeline(config_record, "Random Forest"),
            split.X_train,
            split.y_train,
            cv_config,
            "Random Forest",
        )
        # The fold count can never exceed the smallest class in the training split.
        smallest_class = min(split.class_distribution_train.values())
        self.assertEqual(result.status, "success")
        self.assertEqual(result.n_splits, smallest_class)
        self.assertLess(result.n_splits, 5)
        self.assertEqual(len(result.fold_accuracies), smallest_class)

    def test_impossible_cross_validation_raises_a_clear_error(self) -> None:
        target = pd.Series([0] * 9 + [1])
        with self.assertRaises(DatasetTooSmallError) as context:
            determine_safe_n_splits(target, requested=5)
        self.assertIn("Cross-validation is impossible", str(context.exception))

    def test_impossible_cross_validation_disables_cv_instead_of_failing(self) -> None:
        """A one-sample class is reported, not raised, so the run continues."""
        target = pd.Series([0] * 9 + [1])
        cv_config = build_cross_validation_config(target)
        self.assertFalse(cv_config.enabled)
        self.assertEqual(cv_config.n_splits, 0)
        self.assertIn("Cross-validation is impossible", cv_config.adjusted_reason)
        self.assertTrue(cv_config.adjusted_reason)

    def test_a_disabled_configuration_still_scores_the_training_data(self) -> None:
        frame = make_training_frame(60)
        config_record, X, y, split = split_for(frame)
        cv_config = build_cross_validation_config(pd.Series([0] * 5 + [1]))
        result = run_cross_validation(
            build_model_pipeline(config_record, "Random Forest"),
            split.X_train,
            split.y_train,
            cv_config,
            "Random Forest",
        )
        self.assertEqual(result.status, "skipped")
        self.assertTrue(result.reason)
        self.assertIsNone(result.accuracy_mean)

    def test_safe_folds_are_kept_when_they_are_possible(self) -> None:
        target = pd.Series([0] * 20 + [1] * 20)
        folds, reason = determine_safe_n_splits(target, requested=5)
        self.assertEqual(folds, 5)
        self.assertEqual(reason, "")


class TestExperimentRecords(unittest.TestCase):
    """25. Experiment records are JSON-safe."""

    @staticmethod
    def experiment(
        name: str, f1: float, accuracy: float, status: str = "success"
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=f"exp_{name}",
            model_name=name,
            model_type="linear",
            accuracy=accuracy,
            f1_score=f1,
            status=status,
        )

    def test_experiment_record_is_json_safe(self) -> None:
        record = self.experiment("Random Forest", 0.9, 0.91)
        record.parameters = {"n_estimators": 100, "random_state": 42}
        record.confusion_matrix = ConfusionMatrixData(
            labels=["0", "1"], matrix=[[5, 1], [0, 6]], total=12
        )
        payload = json.loads(json.dumps(to_serializable(record)))
        self.assertEqual(payload["model_name"], "Random Forest")
        self.assertEqual(payload["confusion_matrix"]["matrix"], [[5, 1], [0, 6]])
        self.assertEqual(payload["confusion_matrix"]["labels"], ["0", "1"])

    def test_experiment_record_holds_only_plain_values(self) -> None:
        record = self.experiment("Random Forest", 0.9, 0.91)
        record.parameters = {"n_estimators": 100, "random_state": 42}
        record.confusion_matrix = ConfusionMatrixData(labels=["0", "1"], matrix=[[1, 0], [0, 1]])
        json.dumps(to_serializable(record))  # must not raise

        allowed = (bool, dict, float, int, list, str, type(None), ConfusionMatrixData)
        for name, value in record.__dict__.items():
            self.assertIsInstance(
                value, allowed, f"{name} holds an unexpected type: {type(value)}"
            )

    def test_failed_experiment_keeps_its_error_message(self) -> None:
        record = self.experiment("Support Vector Machine", 0.0, 0.0, ExperimentStatus.FAILED.value)
        record.error_message = "ValueError: boom"
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error_message, "ValueError: boom")

    def test_partial_experiment_records_missing_cv_values_as_none(self) -> None:
        record = self.experiment("K-Nearest Neighbors", 0.7, 0.7, ExperimentStatus.PARTIAL.value)
        self.assertIsNone(record.cv_f1_mean)
        json.dumps(to_serializable(record))


class TestExperimentIdentity(unittest.TestCase):
    """Experiment ids must be stable, unique and readable."""

    def test_id_is_deterministic(self) -> None:
        first = build_experiment_id(
            "Random Forest", {"n_estimators": 100, "random_state": 42}
        )
        second = build_experiment_id(
            "Random Forest", {"random_state": 42, "n_estimators": 100}
        )
        self.assertEqual(first, second)

    def test_id_reads_like_the_model(self) -> None:
        identifier = build_experiment_id("Random Forest", {"n_estimators": 100})
        self.assertTrue(identifier.startswith("exp_random_forest_"))
        self.assertEqual(len(identifier), len("exp_random_forest_") + 6)

    def test_different_parameters_give_different_ids(self) -> None:
        self.assertNotEqual(
            build_experiment_id("Random Forest", {"n_estimators": 100}),
            build_experiment_id("Random Forest", {"n_estimators": 300}),
        )

    def test_different_models_give_different_ids(self) -> None:
        self.assertNotEqual(
            build_experiment_id("Random Forest", {}),
            build_experiment_id("Decision Tree", {}),
        )

    def test_id_without_parameters_is_stable(self) -> None:
        self.assertEqual(
            build_experiment_id("K-Nearest Neighbors", None), "exp_k-nearest_neighbors_default"
        )

    def test_id_is_json_safe(self) -> None:
        identifier = build_experiment_id("Support Vector Machine", {"probability": False})
        self.assertIsInstance(identifier, str)
        json.dumps(identifier)


class TestRankingAndComparison(unittest.TestCase):
    """26. Baseline ranking and the best-model rule work as documented."""

    @staticmethod
    def experiment(name: str, f1: float, accuracy: float, status: str = "success") -> ExperimentResult:
        return ExperimentResult(
            experiment_id=f"exp_{name}",
            model_name=name,
            model_type="linear",
            accuracy=accuracy,
            f1_score=f1,
            cv_f1_mean=f1 - 0.01,
            cv_accuracy_mean=accuracy - 0.01,
            status=status,
        )

    def test_ranking_uses_f1_first(self) -> None:
        experiments = [
            self.experiment("Random Forest", 0.80, 0.95),
            self.experiment("Logistic Regression", 0.90, 0.85),
        ]
        comparison = rank_experiments(experiments)
        self.assertEqual(
            [entry.model_name for entry in comparison.entries],
            ["Logistic Regression", "Random Forest"],
        )

    def test_accuracy_breaks_ties(self) -> None:
        experiments = [
            self.experiment("Decision Tree", 0.80, 0.70),
            self.experiment("Random Forest", 0.80, 0.90),
        ]
        comparison = rank_experiments(experiments)
        self.assertEqual(comparison.best_model_name, "Random Forest")

    def test_full_tie_is_resolved_deterministically_by_name(self) -> None:
        experiments = [
            self.experiment("Random Forest", 1.0, 1.0),
            self.experiment("Decision Tree", 1.0, 1.0),
        ]
        self.assertEqual(rank_experiments(experiments).best_model_name, "Decision Tree")

    def test_failed_experiments_are_excluded_from_the_ranking(self) -> None:
        experiments = [
            self.experiment("Random Forest", 0.9, 0.9),
            self.experiment("Support Vector Machine", 0.0, 0.0, "failed"),
        ]
        comparison = rank_experiments(experiments)
        self.assertEqual(comparison.successful_count, 1)
        self.assertEqual(comparison.failed_count, 1)
        self.assertEqual(len(comparison.entries), 1)
        self.assertTrue(any("failed" in note for note in comparison.notes))

    def test_partial_experiments_are_still_ranked(self) -> None:
        experiments = [self.experiment("K-Nearest Neighbors", 0.75, 0.75, "partial")]
        comparison = rank_experiments(experiments)
        self.assertEqual(comparison.successful_count, 1)
        self.assertTrue(any("Cross-validation was skipped" in note for note in comparison.notes))

    def test_ranks_are_sequential(self) -> None:
        experiments = [self.experiment(f"Model{index}", 0.5 + index / 100, 0.5) for index in range(4)]
        comparison = rank_experiments(experiments)
        self.assertEqual([entry.rank for entry in comparison.entries], [1, 2, 3, 4])

    def test_ranking_rule_is_documented(self) -> None:
        comparison = rank_experiments([self.experiment("Random Forest", 0.9, 0.9)])
        self.assertIn("f1_score", comparison.ranking_rule)
        self.assertIn("accuracy", comparison.ranking_rule)
        self.assertIn("BASELINE", comparison.ranking_rule)

    def test_comparison_table_is_rendered(self) -> None:
        comparison = rank_experiments(
            [self.experiment("Random Forest", 0.9, 0.91), self.experiment("Decision Tree", 0.7, 0.72)]
        )
        self.assertIn("Random Forest", comparison.table_text)
        self.assertIn("Best baseline model: Random Forest", comparison.table_text)
        # header + separator + two models + separator + best-model line
        self.assertEqual(len(comparison.table_text.splitlines()), 6)

    def test_comparison_table_shows_a_dash_for_missing_metrics(self) -> None:
        record = self.experiment("Random Forest", 0.9, 0.91, "partial")
        comparison = rank_experiments([record])
        self.assertIn("-", comparison.table_text)

    def test_best_model_record_is_complete(self) -> None:
        experiments = [
            self.experiment("Random Forest", 0.88, 0.89),
            self.experiment("Decision Tree", 0.80, 0.81),
        ]
        best = select_best_experiment(experiments)
        self.assertIsInstance(best, BestModelInfo)
        self.assertEqual(best.model_name, "Random Forest")
        self.assertEqual(best.f1_score, 0.88)
        self.assertEqual(best.accuracy, 0.89)
        self.assertEqual(best.experiment_id, "exp_Random Forest")
        self.assertIn("highest f1_score", best.selection_reason)

    def test_best_model_is_none_when_nothing_succeeded(self) -> None:
        experiments = [self.experiment("Random Forest", 0.0, 0.0, "failed")]
        self.assertIsNone(select_best_experiment(experiments))

    def test_best_model_is_none_for_an_empty_list(self) -> None:
        self.assertIsNone(select_best_experiment([]))
        comparison = rank_experiments([])
        self.assertIsInstance(comparison, ModelComparison)
        self.assertEqual(comparison.entries, [])

    def test_custom_metrics_can_drive_the_ranking(self) -> None:
        experiments = [
            self.experiment("A", 0.5, 0.95),
            self.experiment("B", 0.5, 0.99),
        ]
        comparison = rank_experiments(experiments, primary_metric="accuracy", secondary_metric="f1_score")
        self.assertEqual(comparison.best_model_name, "B")

    def test_real_end_to_end_metrics_for_every_model(self) -> None:
        """Each supported model trains and produces valid metrics on one frame."""
        from tools.model_tools import get_supported_models

        frame = make_training_frame(60)
        config_record, X, y, split = split_for(frame)
        cv_config = build_cross_validation_config(split.y_train)

        for name in get_supported_models():
            pipeline = build_model_pipeline(config_record, name)
            pipeline.fit(split.X_train, split.y_train)
            predictions = pipeline.predict(split.X_test)
            metrics = evaluate_classifier(split.y_test, predictions)
            cv_result = run_cross_validation(
                pipeline, split.X_train, split.y_train, cv_config, name
            )
            self.assertGreaterEqual(metrics["accuracy"], 0.0, name)
            self.assertLessEqual(metrics["accuracy"], 1.0, name)
            self.assertIsNotNone(cv_result.accuracy_mean, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
