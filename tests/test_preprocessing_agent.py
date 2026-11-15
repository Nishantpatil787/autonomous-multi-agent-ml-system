"""Tests for the Preprocessing Agent (Phase 3).

Focus: the agent must consume the Phase 2 state, record typed decisions, build
an unfitted pipeline, refuse to run without an analysis, and never modify the
source dataset.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from agents.data_analysis_agent import DataAnalysisAgent
from agents.preprocessing_agent import PreprocessingAgent
from tests.helpers import (
    BINARY_DATASET,
    MULTICLASS_DATASET,
    file_hash,
    frame_clean_categorical,
    frame_clean_numerical,
    frame_mixed,
    frame_with_constant_and_empty,
    frame_with_duplicates,
    frame_with_high_cardinality,
    write_csv,
)
from tools.data_tools import DataToolError
from tools.preprocessing_tools import AnalysisNotReadyError, PreprocessingConfigError
from utils.state import (
    AgentState,
    AgentStatus,
    DuplicateStrategy,
    ExclusionReason,
    PreprocessingStatus,
    ProblemType,
    ScalingStrategy,
)


def analyse_then_preprocess(dataset_path, **kwargs) -> AgentState:
    """Run the Data Analysis Agent and feed its state to the Preprocessing Agent."""
    state = DataAnalysisAgent().run(AgentState(), dataset_path=dataset_path)
    return PreprocessingAgent().run(state, **kwargs)


class TestAgentOnSampleDataset(unittest.TestCase):
    """12. AgentState update on the bundled sample datasets."""

    def setUp(self) -> None:
        self.state = analyse_then_preprocess(BINARY_DATASET)

    def test_status_is_completed(self) -> None:
        self.assertEqual(self.state.preprocessing_status, PreprocessingStatus.COMPLETED.value)
        self.assertEqual(self.state.status, AgentStatus.PREPROCESSING_COMPLETED.value)
        self.assertEqual(self.state.errors, [])

    def test_identifier_column_is_excluded(self) -> None:
        self.assertNotIn("record_id", self.state.preprocessing_numerical_features)
        self.assertNotIn("record_id", self.state.preprocessing_categorical_features)
        excluded = self.state.excluded_features_by_reason
        self.assertIn("record_id", excluded[ExclusionReason.IDENTIFIER_LIKE.value])

    def test_selected_numerical_features(self) -> None:
        self.assertEqual(self.state.preprocessing_numerical_features, ["age", "bmi", "glucose", "blood_pressure"])

    def test_selected_categorical_features(self) -> None:
        self.assertEqual(self.state.preprocessing_categorical_features, ["gender", "region", "smoking"])

    def test_exclusion_reason_is_recorded_with_detail(self) -> None:
        item = self.state.excluded_features[0]
        self.assertEqual(item.column, "record_id")
        self.assertIn("identifier", item.detail)
        self.assertEqual(item.reason, ExclusionReason.IDENTIFIER_LIKE.value)

    def test_imputation_and_encoding_strategies_recorded(self) -> None:
        config = self.state.preprocessing_config
        self.assertEqual(config.numerical_imputation, "median")
        self.assertEqual(config.categorical_imputation, "most_frequent")
        self.assertEqual(config.encoding, "one_hot")
        self.assertEqual(config.encoding_handle_unknown, "ignore")

    def test_scaling_strategy_recorded(self) -> None:
        config = self.state.preprocessing_config
        self.assertTrue(config.scale_numeric)
        self.assertEqual(config.scaling_strategy, ScalingStrategy.STANDARD.value)

    def test_duplicate_handling_recorded(self) -> None:
        handling = self.state.preprocessing_config.duplicate_handling
        self.assertEqual(handling.strategy, DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value)
        self.assertEqual(handling.duplicate_rows_detected, self.state.duplicate_count)
        self.assertEqual(handling.duplicate_rows_removed, 0)
        self.assertEqual(handling.duplicate_rows_to_remove, self.state.duplicate_count)

    def test_preprocessing_steps_are_typed(self) -> None:
        names = [step.name for step in self.state.preprocessing_steps]
        self.assertEqual(names, ["numerical", "categorical"])
        self.assertTrue(self.state.preprocessing_steps[0].transformers)

    def test_fit_scope_is_recorded(self) -> None:
        self.assertEqual(self.state.preprocessing_config.fit_scope, "training_data_only")

    def test_summary_is_populated(self) -> None:
        summary = self.state.preprocessing_summary
        self.assertTrue(summary.pipeline_ready)
        self.assertEqual(summary.status, PreprocessingStatus.COMPLETED.value)
        self.assertEqual(summary.numerical_feature_count, 4)
        self.assertEqual(summary.categorical_feature_count, 3)
        self.assertEqual(summary.used_feature_count, 7)
        self.assertEqual(summary.excluded_feature_count, 1)
        self.assertEqual(summary.duplicate_rows_detected, 3)
        self.assertIn("median", summary.summary_text)

    def test_pipeline_is_created_and_unfitted(self) -> None:
        pipeline = self.state.preprocessing_pipeline
        self.assertIsInstance(pipeline, ColumnTransformer)
        with self.assertRaises(NotFittedError):
            check_is_fitted(pipeline)

    def test_state_is_json_serialisable(self) -> None:
        payload = self.state.to_dict()
        # No live scikit-learn object may end up in the serialised state.
        json.dumps(payload)
        self.assertNotIn("preprocessing_pipeline", payload)
        self.assertEqual(payload["runtime_objects"]["preprocessing_pipeline"]["type"], "ColumnTransformer")
        self.assertFalse(payload["runtime_objects"]["preprocessing_pipeline"]["fitted"])

    def test_config_alone_can_rebuild_the_pipeline(self) -> None:
        from tools.preprocessing_tools import build_column_transformer

        rebuilt = build_column_transformer(self.state.preprocessing_config)
        self.assertIsInstance(rebuilt, ColumnTransformer)
        self.assertEqual(
            [name for name, _, _ in rebuilt.transformers],
            [name for name, _, _ in self.state.preprocessing_pipeline.transformers],
        )

    def test_agent_logged_its_decisions(self) -> None:
        messages = [m.message for m in self.state.agent_messages if m.agent == "PreprocessingAgent"]
        self.assertTrue(any("Numerical imputation" in item for item in messages))
        self.assertTrue(any("handle_unknown='ignore'" in item for item in messages))
        self.assertTrue(any("NOT fitted" in item for item in messages))

    def test_source_dataset_is_unchanged(self) -> None:
        before = file_hash(BINARY_DATASET)
        analyse_then_preprocess(BINARY_DATASET)
        self.assertEqual(file_hash(BINARY_DATASET), before)


class TestAgentOnMulticlassDataset(unittest.TestCase):
    """The agent works the same way for a multiclass target."""

    def setUp(self) -> None:
        self.state = analyse_then_preprocess(MULTICLASS_DATASET)

    def test_multiclass_pipeline_is_built(self) -> None:
        self.assertEqual(self.state.preprocessing_status, PreprocessingStatus.COMPLETED.value)
        self.assertEqual(self.state.target_column, "class_label")
        self.assertEqual(
            self.state.preprocessing_numerical_features, ["length_cm", "weight_g", "humidity_pct"]
        )
        self.assertEqual(
            self.state.preprocessing_categorical_features, ["texture", "colour"]
        )

    def test_sample_id_is_excluded_as_identifier(self) -> None:
        excluded = self.state.excluded_features_by_reason
        self.assertIn("sample_id", excluded[ExclusionReason.IDENTIFIER_LIKE.value])

    def test_missing_values_are_handled_by_the_configured_pipeline(self) -> None:
        # weight_g and humidity_pct contain missing values and are kept because
        # the imputer inside the pipeline deals with them.
        self.assertEqual(self.state.total_missing_values, 6)
        self.assertIn("weight_g", self.state.preprocessing_numerical_features)
        imputer = self.state.preprocessing_pipeline.transformers[0][1].named_steps["imputer"]
        self.assertEqual(imputer.strategy, "median")


class TestAgentEdgeCases(unittest.TestCase):
    """6, 7, 8, 9, 10. Exclusions and duplicate handling through the agent."""

    def _run_on(self, frame: pd.DataFrame, **kwargs) -> AgentState:
        with TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp), "case.csv", frame)
            return analyse_then_preprocess(path, **kwargs)

    def test_constant_column_is_excluded(self) -> None:
        state = self._run_on(frame_mixed())
        excluded = state.excluded_features_by_reason
        self.assertIn("constant_col", excluded[ExclusionReason.CONSTANT_COLUMN.value])

    def test_empty_column_is_excluded(self) -> None:
        state = self._run_on(frame_with_constant_and_empty())
        excluded = state.excluded_features_by_reason
        self.assertIn("empty_col", excluded[ExclusionReason.EMPTY_COLUMN.value])

    def test_datetime_column_is_excluded(self) -> None:
        # A CSV read never produces datetime dtypes, so the loader is patched to
        # hand over the in-memory frame that has a real datetime column.
        frame = frame_mixed()
        with TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp), "case.csv", frame)
            state = DataAnalysisAgent().run(AgentState(), dataset_path=path)
            with patch("agents.preprocessing_agent.load_csv", return_value=frame):
                state = PreprocessingAgent().run(state)

        excluded = state.excluded_features_by_reason
        self.assertIn("signup_date", excluded[ExclusionReason.UNSUPPORTED_TYPE.value])
        self.assertNotIn("signup_date", state.preprocessing_numerical_features)
        self.assertNotIn("signup_date", state.preprocessing_categorical_features)
        detail = next(
            item.detail for item in state.excluded_features if item.column == "signup_date"
        )
        self.assertIn("date features", detail)

    def test_high_cardinality_column_is_excluded(self) -> None:
        state = self._run_on(frame_with_high_cardinality())
        excluded = state.excluded_features_by_reason
        self.assertIn("free_text", excluded[ExclusionReason.HIGH_CARDINALITY.value])

    def test_high_cardinality_exclusion_is_logged_as_warning(self) -> None:
        state = self._run_on(frame_with_high_cardinality())
        warnings = [
            m.message
            for m in state.agent_messages
            if m.agent == "PreprocessingAgent" and m.level == "warning"
        ]
        self.assertTrue(any("free_text" in item for item in warnings))

    def test_duplicate_handling_can_keep_all(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        state = self._run_on(frame, duplicate_strategy=DuplicateStrategy.KEEP_ALL.value)
        handling = state.preprocessing_config.duplicate_handling
        self.assertEqual(handling.strategy, DuplicateStrategy.KEEP_ALL.value)
        self.assertEqual(handling.duplicate_rows_to_remove, 0)

    def test_duplicate_handling_defaults_to_removal(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        state = self._run_on(frame)
        handling = state.preprocessing_config.duplicate_handling
        self.assertEqual(handling.duplicate_rows_to_remove, 3)
        self.assertEqual(handling.duplicate_rows_removed, 0)

    def test_invalid_duplicate_strategy_fails_loudly(self) -> None:
        frame = frame_with_duplicates(rows=20, duplicates=3)
        with TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp), "case.csv", frame)
            state = DataAnalysisAgent().run(AgentState(), dataset_path=path)
            with self.assertRaises(PreprocessingConfigError):
                PreprocessingAgent().run(state, duplicate_strategy="drop_everything")
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_numerical_only_dataset(self) -> None:
        frame = frame_clean_numerical()
        frame["label"] = [index % 2 for index in range(frame.shape[0])]
        state = self._run_on(frame)
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.COMPLETED.value)
        self.assertEqual(state.preprocessing_categorical_features, [])
        self.assertEqual(
            [name for name, _, _ in state.preprocessing_pipeline.transformers], ["numerical"]
        )

    def test_categorical_only_dataset(self) -> None:
        state = self._run_on(frame_clean_categorical())
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.COMPLETED.value)
        self.assertEqual(state.preprocessing_numerical_features, [])
        self.assertEqual(
            [name for name, _, _ in state.preprocessing_pipeline.transformers], ["categorical"]
        )

    def test_scaling_can_be_disabled(self) -> None:
        state = self._run_on(frame_mixed(), scale_numeric=False)
        config = state.preprocessing_config
        self.assertFalse(config.scale_numeric)
        self.assertEqual(config.scaling_strategy, ScalingStrategy.NONE.value)
        branches = {name: pipe for name, pipe, _ in state.preprocessing_pipeline.transformers}
        self.assertNotIn("scaler", branches["numerical"].named_steps)

    def test_imputation_strategies_can_be_overridden(self) -> None:
        state = self._run_on(frame_mixed(), numerical_imputation="mean")
        self.assertEqual(state.preprocessing_config.numerical_imputation, "mean")

    def test_invalid_imputation_override_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp), "case.csv", frame_mixed())
            state = DataAnalysisAgent().run(AgentState(), dataset_path=path)
            with self.assertRaises(PreprocessingConfigError):
                PreprocessingAgent().run(state, numerical_imputation="magic")


class TestAgentGuards(unittest.TestCase):
    """13. Missing analysis state and other error paths."""

    def test_missing_analysis_state_raises(self) -> None:
        state = AgentState()
        with self.assertRaises(AnalysisNotReadyError):
            PreprocessingAgent().run(state, dataset_path=BINARY_DATASET)
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.FAILED.value)
        self.assertTrue(state.errors)

    def test_missing_dataset_path_raises(self) -> None:
        state = AgentState()
        with self.assertRaises(AnalysisNotReadyError):
            PreprocessingAgent().run(state)
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.FAILED.value)

    def test_missing_file_raises_and_is_recorded(self) -> None:
        state = DataAnalysisAgent().run(AgentState(), dataset_path=BINARY_DATASET)
        # Break the path after the analysis so only the preprocessing read fails.
        state.dataset_path = "datasets/does_not_exist.csv"
        with self.assertRaises(DataToolError):
            PreprocessingAgent().run(state)
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.FAILED.value)
        self.assertTrue(state.errors)
        self.assertIn("does not exist", state.errors[0])

    def test_regression_target_is_skipped_with_reason(self) -> None:
        with TemporaryDirectory() as tmp:
            frame = pd.DataFrame(
                {
                    "age": [float(20 + (index % 10)) for index in range(40)],
                    "income": [10.5 + index * 2.25 for index in range(40)],
                }
            )
            path = write_csv(Path(tmp), "regression.csv", frame)
            state = DataAnalysisAgent().run(
                AgentState(), dataset_path=path, target_column="income"
            )
            self.assertEqual(state.problem_type, ProblemType.REGRESSION.value)
            state = PreprocessingAgent().run(state)

        self.assertEqual(state.preprocessing_status, PreprocessingStatus.SKIPPED.value)
        self.assertIsNone(state.preprocessing_pipeline)
        self.assertIsNone(state.preprocessing_config)
        self.assertIn("regression", state.preprocessing_summary.summary_text.lower())
        self.assertTrue(
            any(m.level == "warning" for m in state.agent_messages if m.agent == "PreprocessingAgent")
        )

    def test_missing_target_is_skipped_with_reason(self) -> None:
        with TemporaryDirectory() as tmp:
            frame = pd.DataFrame(
                {f"field_{index}": [f"{index}-{row}" for row in range(30)] for index in range(4)}
            )
            path = write_csv(Path(tmp), "no_target.csv", frame)
            state = analyse_then_preprocess(path)

        self.assertEqual(state.preprocessing_status, PreprocessingStatus.SKIPPED.value)
        self.assertIsNone(state.preprocessing_pipeline)
        self.assertIn("target", state.preprocessing_summary.summary_text.lower())

    def test_agent_is_deterministic(self) -> None:
        first = analyse_then_preprocess(BINARY_DATASET)
        second = analyse_then_preprocess(BINARY_DATASET)
        self.assertEqual(
            first.preprocessing_config.numerical_features,
            second.preprocessing_config.numerical_features,
        )
        self.assertEqual(
            first.preprocessing_config.categorical_features,
            second.preprocessing_config.categorical_features,
        )
        self.assertEqual(
            first.preprocessing_summary.summary_text,
            second.preprocessing_summary.summary_text,
        )

    def test_rerunning_the_agent_is_stable(self) -> None:
        state = analyse_then_preprocess(BINARY_DATASET)
        before = state.preprocessing_config
        state = PreprocessingAgent().run(state)
        self.assertEqual(state.preprocessing_status, PreprocessingStatus.COMPLETED.value)
        self.assertEqual(state.preprocessing_config.numerical_features, before.numerical_features)
        self.assertEqual(state.preprocessing_config.categorical_features, before.categorical_features)


class TestPipelineUsability(unittest.TestCase):
    """16, 17 & 18. The built pipeline is usable by later training code."""

    def setUp(self) -> None:
        self.state = analyse_then_preprocess(BINARY_DATASET)
        self.frame = pd.read_csv(BINARY_DATASET)

    def _split(self):
        train = self.frame.iloc[:100]
        test = self.frame.iloc[100:]
        return (
            train.drop(columns=[self.state.target_column]),
            test.drop(columns=[self.state.target_column]),
            train[self.state.target_column],
            test[self.state.target_column],
        )

    def test_pipeline_fits_and_transforms_training_split(self) -> None:
        from tools.preprocessing_tools import remove_duplicate_rows

        X_train, _, y_train, _ = self._split()
        X_train, _ = remove_duplicate_rows(X_train)
        pipeline = self.state.preprocessing_pipeline
        transformed = pipeline.fit_transform(X_train)
        self.assertEqual(transformed.shape[0], len(y_train))
        # 4 numerical + 2 + 4 + 3 encoded categorical values
        self.assertEqual(transformed.shape[1], 13)

    def test_pipeline_transforms_unseen_test_rows(self) -> None:
        from tools.preprocessing_tools import remove_duplicate_rows

        X_train, X_test, _, _ = self._split()
        X_train, _ = remove_duplicate_rows(X_train)
        pipeline = self.state.preprocessing_pipeline
        pipeline.fit(X_train)
        transformed = pipeline.transform(X_test)
        self.assertEqual(transformed.shape[0], X_test.shape[0])
        self.assertEqual(transformed.shape[1], 13)

    def test_imputer_uses_training_median_not_full_dataset_median(self) -> None:
        from tools.preprocessing_tools import remove_duplicate_rows

        X_train, X_test, _, _ = self._split()
        X_train, _ = remove_duplicate_rows(X_train)
        pipeline = self.state.preprocessing_pipeline
        pipeline.fit(X_train)

        imputer = pipeline.named_transformers_["numerical"].named_steps["imputer"]
        age_index = self.state.preprocessing_numerical_features.index("age")
        self.assertAlmostEqual(
            float(imputer.statistics_[age_index]), float(X_train["age"].median()), places=6
        )
        full_median = float(self.frame["age"].median())
        self.assertNotAlmostEqual(float(imputer.statistics_[age_index]), full_median, places=1)

    def test_encoder_did_not_learn_from_the_test_split(self) -> None:
        from tools.preprocessing_tools import remove_duplicate_rows

        X_train, X_test, _, _ = self._split()
        X_train, _ = remove_duplicate_rows(X_train)
        # Introduce a category that only exists in the test split.
        X_test = X_test.copy()
        X_test.loc[X_test.index[0], "region"] = "atlantis"
        pipeline = self.state.preprocessing_pipeline
        pipeline.fit(X_train)

        encoder = pipeline.named_transformers_["categorical"].named_steps["encoder"]
        region_index = self.state.preprocessing_categorical_features.index("region")
        self.assertNotIn("atlantis", set(encoder.categories_[region_index]))
        # ...and transforming it still works.
        self.assertEqual(pipeline.transform(X_test).shape[0], X_test.shape[0])

    def test_scaler_learned_training_statistics(self) -> None:
        from tools.preprocessing_tools import remove_duplicate_rows

        X_train, _, _, _ = self._split()
        X_train, _ = remove_duplicate_rows(X_train)
        pipeline = self.state.preprocessing_pipeline
        pipeline.fit(X_train)

        numerical = self.state.preprocessing_numerical_features
        # The scaler runs after the imputer, so its mean must match the mean of
        # the imputed training column, not the raw column.
        branch = pipeline.named_transformers_["numerical"]
        imputed = branch.named_steps["imputer"].transform(X_train[numerical])
        self.assertTrue(np.allclose(branch.named_steps["scaler"].mean_, imputed.mean(axis=0)))

    def test_target_column_is_never_a_feature(self) -> None:
        features = (
            self.state.preprocessing_numerical_features
            + self.state.preprocessing_categorical_features
        )
        self.assertNotIn(self.state.target_column, features)


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (used in the scaler assertions)

    unittest.main(verbosity=2)