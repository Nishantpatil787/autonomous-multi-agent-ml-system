"""Tests for the Data Analysis Agent (Phase 2).

Focus: the agent must fill the shared `AgentState` with structured data, log
its operations, never modify the source dataset and fail loudly on bad input.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.data_analysis_agent import DataAnalysisAgent
from tests.helpers import (
    BINARY_DATASET,
    BINARY_EXPECTED,
    MULTICLASS_DATASET,
    MULTICLASS_EXPECTED,
    file_hash,
    write_csv,
)
from tools.data_tools import DataToolError, DatasetNotFoundError, InvalidTargetColumnError
from utils.state import AgentState, AgentStatus, MissingValueReport, ProblemType


class TestAgentOnBinaryDataset(unittest.TestCase):
    """11. AgentState update on the binary sample dataset."""

    def setUp(self) -> None:
        self.agent = DataAnalysisAgent()
        self.state = self.agent.run(AgentState(), dataset_path=BINARY_DATASET)

    def test_dataset_identity_and_shape(self) -> None:
        self.assertEqual(self.state.dataset_name, "sample_classification.csv")
        self.assertEqual(self.state.dataset_shape.rows, BINARY_EXPECTED["rows"])
        self.assertEqual(self.state.dataset_shape.columns, BINARY_EXPECTED["columns"])
        self.assertEqual(len(self.state.column_names), BINARY_EXPECTED["columns"])
        self.assertEqual(len(self.state.column_dtypes), BINARY_EXPECTED["columns"])

    def test_feature_lists(self) -> None:
        self.assertEqual(self.state.numerical_features, BINARY_EXPECTED["numerical_features"])
        self.assertEqual(self.state.categorical_features, BINARY_EXPECTED["categorical_features"])
        self.assertEqual(self.state.unsupported_columns, [])
        self.assertNotIn("Outcome", self.state.feature_names)

    def test_missing_values_state(self) -> None:
        self.assertIsInstance(self.state.missing_values, MissingValueReport)
        self.assertEqual(self.state.missing_values.total, BINARY_EXPECTED["missing_total"])
        self.assertEqual(self.state.missing_values.per_column, BINARY_EXPECTED["missing_by_column"])
        self.assertEqual(self.state.total_missing_values, BINARY_EXPECTED["missing_total"])

    def test_duplicate_count_state(self) -> None:
        self.assertEqual(self.state.duplicate_count, BINARY_EXPECTED["duplicates"])

    def test_automatic_target_detection_state(self) -> None:
        self.assertEqual(self.state.target_column, BINARY_EXPECTED["target"])
        self.assertEqual(self.state.target_detection_method, "automatic")
        self.assertIn(self.state.target_detection_confidence, {"high", "medium", "low"})
        self.assertTrue(self.state.target_candidates)
        self.assertIn("heuristic", self.state.target_detection_reason.lower())

    def test_binary_problem_type_state(self) -> None:
        self.assertEqual(self.state.problem_type, ProblemType.BINARY_CLASSIFICATION.value)
        self.assertTrue(self.state.problem_type_reason)
        self.assertEqual(len(self.state.class_distribution), 2)

    def test_class_distribution_state(self) -> None:
        total = sum(entry.count for entry in self.state.class_distribution.values())
        self.assertEqual(total, BINARY_EXPECTED["rows"])
        self.assertAlmostEqual(
            sum(entry.percentage for entry in self.state.class_distribution.values()),
            100.0,
            places=1,
        )
        for entry in self.state.class_distribution.values():
            self.assertIsInstance(entry.count, int)
            self.assertIsInstance(entry.percentage, float)

    def test_statistics_state(self) -> None:
        self.assertEqual(
            set(self.state.numerical_statistics), set(self.state.numerical_features)
        )
        self.assertEqual(
            set(self.state.categorical_statistics), set(self.state.categorical_features)
        )
        age = self.state.numerical_statistics["age"]
        self.assertEqual(age.count, BINARY_EXPECTED["rows"] - BINARY_EXPECTED["missing_by_column"]["age"])

    def test_data_quality_state(self) -> None:
        kinds = {issue.kind for issue in self.state.data_quality_issues}
        self.assertIn("missing_values", kinds)
        self.assertIn("duplicate_rows", kinds)
        self.assertIn("high_cardinality_column", kinds)
        for issue in self.state.data_quality_issues:
            self.assertIn(issue.severity, {"info", "warning", "error"})
            self.assertTrue(issue.message)

    def test_status_and_messages(self) -> None:
        self.assertEqual(self.state.status, AgentStatus.ANALYSIS_COMPLETED.value)
        self.assertEqual(self.state.errors, [])
        self.assertTrue(self.state.agent_messages)
        self.assertTrue(
            all(message.agent == "DataAnalysisAgent" for message in self.state.agent_messages)
        )
        self.assertTrue(all(hasattr(message, "timestamp") for message in self.state.agent_messages))

    def test_summary_state(self) -> None:
        summary = self.state.analysis_summary
        self.assertIsNotNone(summary)
        self.assertEqual(summary.rows, BINARY_EXPECTED["rows"])
        self.assertEqual(summary.target_column, BINARY_EXPECTED["target"])
        self.assertEqual(summary.problem_type, ProblemType.BINARY_CLASSIFICATION.value)
        self.assertEqual(summary.issue_count, len(self.state.data_quality_issues))
        self.assertIn("sample_classification.csv", summary.summary_text)

    def test_source_dataset_not_modified(self) -> None:
        # The hash was already produced by load_csv in setUp; re-running the
        # agent must leave the file byte-identical.
        before = file_hash(BINARY_DATASET)
        self.agent.run(AgentState(), dataset_path=BINARY_DATASET)
        self.assertEqual(file_hash(BINARY_DATASET), before)

    def test_state_is_json_serialisable(self) -> None:
        # No `default=str` fallback is needed: to_dict already converts every
        # value into a JSON-safe type.
        payload = json.dumps(self.state.to_dict())
        self.assertIn("sample_classification.csv", payload)


class TestAgentOnMulticlassDataset(unittest.TestCase):
    """8. Multiclass classification detection through the agent."""

    def setUp(self) -> None:
        self.state = DataAnalysisAgent().run(AgentState(), dataset_path=MULTICLASS_DATASET)

    def test_multiclass_detected(self) -> None:
        self.assertEqual(self.state.problem_type, ProblemType.MULTICLASS_CLASSIFICATION.value)
        self.assertEqual(self.state.target_column, MULTICLASS_EXPECTED["target"])
        self.assertEqual(self.state.target_detection_method, "automatic")

    def test_class_distribution_has_three_classes(self) -> None:
        self.assertEqual(len(self.state.class_distribution), MULTICLASS_EXPECTED["class_count"])
        # Duplicated rows are part of the file, so every row is counted.
        total = sum(entry.count for entry in self.state.class_distribution.values())
        self.assertEqual(total, MULTICLASS_EXPECTED["rows"])

    def test_multiclass_missing_and_duplicates(self) -> None:
        self.assertEqual(self.state.total_missing_values, MULTICLASS_EXPECTED["missing_total"])
        self.assertEqual(self.state.duplicate_count, MULTICLASS_EXPECTED["duplicates"])


class TestAgentTargetHandling(unittest.TestCase):
    """6. Explicit vs automatic target handling."""

    def test_explicit_target_is_respected(self) -> None:
        state = DataAnalysisAgent().run(
            AgentState(), dataset_path=BINARY_DATASET, target_column="gender"
        )
        self.assertEqual(state.target_column, "gender")
        self.assertEqual(state.target_detection_method, "explicit")
        self.assertEqual(state.target_detection_confidence, "high")
        self.assertEqual(state.problem_type, ProblemType.BINARY_CLASSIFICATION.value)
        self.assertIn("explicitly provided", state.target_detection_reason)

    def test_explicit_target_column_is_not_a_feature(self) -> None:
        state = DataAnalysisAgent().run(
            AgentState(), dataset_path=BINARY_DATASET, target_column="gender"
        )
        self.assertNotIn("gender", state.numerical_features)
        self.assertNotIn("gender", state.categorical_features)

    def test_dataset_path_taken_from_state(self) -> None:
        state = AgentState(dataset_path=str(BINARY_DATASET))
        result = DataAnalysisAgent().run(state)
        self.assertEqual(result.dataset_name, "sample_classification.csv")

    def test_explicit_target_survives_in_state(self) -> None:
        state = AgentState(dataset_path=str(BINARY_DATASET), target_column="gender")
        result = DataAnalysisAgent().run(state)
        self.assertEqual(result.target_column, "gender")


class TestAgentErrorHandling(unittest.TestCase):
    """12. Invalid input handling: errors are logged, recorded and re-raised."""

    def test_invalid_path_raises_and_records_error(self) -> None:
        state = AgentState()
        with self.assertRaises(DatasetNotFoundError):
            DataAnalysisAgent().run(state, dataset_path="datasets/no_such_file.csv")

        self.assertEqual(state.status, AgentStatus.FAILED.value)
        self.assertTrue(state.errors)
        self.assertIn("no_such_file.csv", state.errors[0])
        self.assertTrue(
            any(message.level == "error" for message in state.agent_messages)
        )

    def test_missing_path_argument_raises(self) -> None:
        state = AgentState()
        with self.assertRaises(DataToolError):
            DataAnalysisAgent().run(state)
        self.assertEqual(state.status, AgentStatus.FAILED.value)

    def test_invalid_explicit_target_raises(self) -> None:
        state = AgentState()
        with self.assertRaises(InvalidTargetColumnError):
            DataAnalysisAgent().run(state, dataset_path=BINARY_DATASET, target_column="ghost")
        self.assertEqual(state.status, AgentStatus.FAILED.value)

    def test_unsupported_file_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.xlsx"
            path.write_text("not really a spreadsheet", encoding="utf-8")
            state = AgentState()
            with self.assertRaises(DataToolError):
                DataAnalysisAgent().run(state, dataset_path=path)
            self.assertTrue(state.errors)

    def test_empty_csv_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(DataToolError):
                DataAnalysisAgent().run(AgentState(), dataset_path=path)

    def test_agent_does_not_swallow_errors(self) -> None:
        # A failure must always surface to the caller.
        with self.assertRaises(DataToolError):
            DataAnalysisAgent().run(AgentState(), dataset_path="missing/dataset.csv")


class TestAgentWithoutUsableTarget(unittest.TestCase):
    """Uncertain target must not be silently treated as classification."""

    def test_no_detectable_target_is_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame(
                {f"field_{index}": [f"{index}-{row}" for row in range(40)] for index in range(4)}
            )
            path = write_csv(Path(tmp), "no_target.csv", frame)
            state = DataAnalysisAgent().run(AgentState(), dataset_path=path)

        self.assertIsNone(state.target_column)
        self.assertEqual(state.target_detection_method, "not_detected")
        self.assertEqual(state.target_detection_confidence, "none")
        self.assertEqual(state.problem_type, ProblemType.UNSUPPORTED_OR_UNCERTAIN.value)
        self.assertEqual(state.class_distribution, {})
        self.assertTrue(state.target_candidates)

    def test_regression_target_is_reported_as_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            size = 40
            frame = pd.DataFrame(
                {
                    "age": list(range(20, 20 + size)),
                    # Continuous target: clearly a regression problem.
                    "income": [10.5 + index * 2.25 for index in range(size)],
                }
            )
            path = write_csv(Path(tmp), "regression.csv", frame)
            state = DataAnalysisAgent().run(AgentState(), dataset_path=path, target_column="income")

        self.assertEqual(state.problem_type, ProblemType.REGRESSION.value)
        self.assertEqual(state.class_distribution, {})
        kinds = {issue.kind for issue in state.data_quality_issues}
        self.assertIn("unsupported_problem_type", kinds)
        self.assertTrue(
            any(message.level == "warning" for message in state.agent_messages)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)