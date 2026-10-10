"""Tests for the reusable data tools (Phase 2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers import (
    BINARY_DATASET,
    BINARY_EXPECTED,
    MULTICLASS_DATASET,
    MULTICLASS_EXPECTED,
    issue_kinds,
    write_csv,
)
from tools.data_tools import (
    DatasetLoadError,
    DatasetNotFoundError,
    EmptyDatasetError,
    FeatureTypes,
    InvalidTargetColumnError,
    UnsupportedFileFormatError,
    analyze_dataset,
    analyze_missing_values,
    analyze_target,
    calculate_class_distribution,
    compute_categorical_statistics,
    compute_numerical_statistics,
    count_duplicates,
    detect_data_quality_issues,
    detect_feature_types,
    detect_problem_type,
    detect_target_column,
    get_column_info,
    get_dataset_shape,
    load_csv,
)
from utils.state import ProblemType


class TestLoading(unittest.TestCase):
    """1. CSV loading and invalid input handling."""

    def test_load_csv_returns_dataframe(self) -> None:
        frame = load_csv(BINARY_DATASET)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame.shape, (BINARY_EXPECTED["rows"], BINARY_EXPECTED["columns"]))

    def test_load_csv_accepts_string_path(self) -> None:
        frame = load_csv(str(BINARY_DATASET))
        self.assertEqual(len(frame), BINARY_EXPECTED["rows"])

    def test_load_csv_missing_path_raises(self) -> None:
        with self.assertRaises(DatasetNotFoundError):
            load_csv("datasets/this_file_does_not_exist.csv")

    def test_load_csv_empty_path_raises(self) -> None:
        with self.assertRaises(DatasetNotFoundError):
            load_csv("")

    def test_load_csv_directory_raises(self) -> None:
        with self.assertRaises(DatasetNotFoundError):
            load_csv(BINARY_DATASET.parent)

    def test_load_csv_unsupported_extension_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.txt"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(UnsupportedFileFormatError):
                load_csv(path)

    def test_load_csv_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(EmptyDatasetError):
                load_csv(path)

    def test_load_csv_header_only_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "header_only.csv"
            path.write_text("a,b,c\n", encoding="utf-8")
            with self.assertRaises(EmptyDatasetError):
                load_csv(path)

    def test_load_csv_corrupt_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.csv"
            path.write_text('a,b\n"unclosed,1\n2,3,4,5\n', encoding="utf-8")
            with self.assertRaises((DatasetLoadError, EmptyDatasetError)):
                load_csv(path)

    def test_load_csv_does_not_modify_source(self) -> None:
        before = BINARY_DATASET.stat().st_mtime_ns
        frame = load_csv(BINARY_DATASET)
        frame["age"] = 0  # in-memory modification must not touch the file
        self.assertEqual(BINARY_DATASET.stat().st_mtime_ns, before)


class TestStructure(unittest.TestCase):
    """2. Shape and column information."""

    def setUp(self) -> None:
        self.frame = load_csv(BINARY_DATASET)

    def test_get_dataset_shape(self) -> None:
        shape = get_dataset_shape(self.frame)
        self.assertEqual(shape.rows, BINARY_EXPECTED["rows"])
        self.assertEqual(shape.columns, BINARY_EXPECTED["columns"])
        self.assertIn("rows", str(shape))

    def test_get_column_info(self) -> None:
        info = get_column_info(self.frame)
        self.assertEqual(len(info), BINARY_EXPECTED["columns"])
        self.assertEqual(info["age"], "float64")
        self.assertEqual(info["gender"], str(self.frame["gender"].dtype))


class TestFeatureTypes(unittest.TestCase):
    """3. Numerical / categorical detection."""

    def test_detect_feature_types_on_sample(self) -> None:
        frame = load_csv(BINARY_DATASET)
        types = detect_feature_types(frame, exclude_columns=["Outcome"])
        self.assertIsInstance(types, FeatureTypes)
        self.assertEqual(types.numerical_features, BINARY_EXPECTED["numerical_features"])
        self.assertEqual(types.categorical_features, BINARY_EXPECTED["categorical_features"])
        self.assertEqual(types.unsupported_columns, [])

    def test_boolean_columns_are_categorical(self) -> None:
        frame = pd.DataFrame({"flag": [True, False, True], "value": [1.0, 2.0, 3.0]})
        types = detect_feature_types(frame)
        self.assertEqual(types.categorical_features, ["flag"])
        self.assertEqual(types.numerical_features, ["value"])

    def test_datetime_columns_are_unsupported(self) -> None:
        frame = pd.DataFrame(
            {
                "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "value": [1, 2, 3],
            }
        )
        types = detect_feature_types(frame)
        self.assertEqual(types.unsupported_columns, ["when"])
        self.assertEqual(types.numerical_features, ["value"])

    def test_feature_groups_are_disjoint(self) -> None:
        frame = load_csv(MULTICLASS_DATASET)
        types = detect_feature_types(frame)
        all_columns = (
            types.numerical_features + types.categorical_features + types.unsupported_columns
        )
        self.assertEqual(len(all_columns), len(set(all_columns)))
        self.assertEqual(len(all_columns), frame.shape[1])


class TestMissingAndDuplicates(unittest.TestCase):
    """4 & 5. Missing values and duplicate rows."""

    def setUp(self) -> None:
        self.frame = load_csv(BINARY_DATASET)

    def test_analyze_missing_values(self) -> None:
        report = analyze_missing_values(self.frame)
        self.assertEqual(report.total, BINARY_EXPECTED["missing_total"])
        self.assertEqual(report.affected_columns, 4)
        self.assertEqual(report.per_column, BINARY_EXPECTED["missing_by_column"])
        self.assertAlmostEqual(
            report.per_column_percentage["age"],
            round(6 / BINARY_EXPECTED["rows"] * 100, 2),
            places=2,
        )

    def test_missing_report_ignores_clean_columns(self) -> None:
        report = analyze_missing_values(self.frame)
        self.assertNotIn("glucose", report.per_column)
        self.assertNotIn("glucose", report.per_column_percentage)

    def test_count_duplicates(self) -> None:
        self.assertEqual(count_duplicates(self.frame), BINARY_EXPECTED["duplicates"])

    def test_count_duplicates_on_clean_frame(self) -> None:
        clean = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self.assertEqual(count_duplicates(clean), 0)

    def test_no_rows_or_columns_are_dropped(self) -> None:
        report = analyze_missing_values(self.frame)
        self.assertEqual(self.frame.shape, (BINARY_EXPECTED["rows"], BINARY_EXPECTED["columns"]))
        self.assertEqual(report.total, 17)


class TestStatistics(unittest.TestCase):
    """6 & 7. Numerical and categorical statistics."""

    def test_numerical_statistics(self) -> None:
        frame = load_csv(BINARY_DATASET)
        stats = compute_numerical_statistics(frame, ["age", "glucose"])
        self.assertEqual(set(stats), {"age", "glucose"})

        age = stats["age"]
        expected = frame["age"].dropna()
        self.assertEqual(age.count, int(expected.count()))
        self.assertAlmostEqual(age.mean, expected.mean(), places=6)
        self.assertAlmostEqual(age.std, expected.std(), places=6)
        self.assertAlmostEqual(age.minimum, expected.min(), places=6)
        self.assertAlmostEqual(age.maximum, expected.max(), places=6)

    def test_numerical_statistics_single_value_has_no_std(self) -> None:
        frame = pd.DataFrame({"only": [7.0, None, None]})
        stats = compute_numerical_statistics(frame, ["only"])
        self.assertEqual(stats["only"].count, 1)
        self.assertIsNone(stats["only"].std)
        self.assertEqual(stats["only"].mean, 7.0)

    def test_numerical_statistics_identical_values_have_zero_std(self) -> None:
        frame = pd.DataFrame({"only": [7.0, 7.0, None]})
        stats = compute_numerical_statistics(frame, ["only"])
        self.assertEqual(stats["only"].std, 0.0)

    def test_categorical_statistics(self) -> None:
        frame = load_csv(BINARY_DATASET)
        stats = compute_categorical_statistics(frame, ["gender", "region"], top_n=3)
        self.assertEqual(stats["gender"].unique_count, 2)
        self.assertEqual(len(stats["region"].top_values), 3)
        self.assertAlmostEqual(sum(stats["gender"].top_values_percentage.values()), 100.0, places=1)


class TestTargetDetection(unittest.TestCase):
    """6. Target column detection."""

    def setUp(self) -> None:
        self.frame = load_csv(BINARY_DATASET)

    def test_explicit_target(self) -> None:
        detection = detect_target_column(self.frame, explicit_target="Outcome")
        self.assertEqual(detection.column, "Outcome")
        self.assertEqual(detection.method, "explicit")
        self.assertEqual(detection.confidence, "high")
        self.assertEqual(detection.candidates, [])

    def test_explicit_target_invalid_raises(self) -> None:
        with self.assertRaises(InvalidTargetColumnError):
            detect_target_column(self.frame, explicit_target="not_a_column")

    def test_automatic_detection_finds_target(self) -> None:
        detection = detect_target_column(self.frame)
        self.assertEqual(detection.column, BINARY_EXPECTED["target"])
        self.assertEqual(detection.method, "automatic")
        self.assertIn(detection.confidence, {"high", "medium", "low"})
        self.assertTrue(detection.candidates)
        self.assertEqual(detection.candidates[0].column, BINARY_EXPECTED["target"])

    def test_automatic_detection_is_heuristic_and_reports_uncertainty(self) -> None:
        detection = detect_target_column(self.frame)
        self.assertIn("heuristic", detection.reason.lower() + " heuristic")
        self.assertIn("confirm", detection.reason.lower())

    def test_identifier_column_is_not_selected(self) -> None:
        # record_id is unique per row and must be rejected as a target.
        detection = detect_target_column(self.frame)
        self.assertNotEqual(detection.column, "record_id")

    def test_detection_without_confident_candidate(self) -> None:
        frame = pd.DataFrame(
            {f"col_{index}": [f"{index}-{row}" for row in range(30)] for index in range(4)}
        )
        detection = detect_target_column(frame)
        self.assertIsNone(detection.column)
        self.assertEqual(detection.method, "not_detected")
        self.assertEqual(detection.confidence, "none")
        self.assertTrue(detection.candidates)

    def test_multiclass_automatic_detection(self) -> None:
        frame = load_csv(MULTICLASS_DATASET)
        detection = detect_target_column(frame)
        self.assertEqual(detection.column, MULTICLASS_EXPECTED["target"])
        self.assertEqual(detection.method, "automatic")


class TestProblemType(unittest.TestCase):
    """7 & 8. Binary, multiclass and unsupported problem types."""

    def test_binary_classification_detected(self) -> None:
        frame = load_csv(BINARY_DATASET)
        analysis = analyze_target(frame, "Outcome")
        self.assertEqual(analysis.problem_type, ProblemType.BINARY_CLASSIFICATION)
        self.assertTrue(analysis.is_supported)
        self.assertEqual(analysis.class_count, 2)
        self.assertEqual(analysis.missing_count, 0)

    def test_multiclass_classification_detected(self) -> None:
        frame = load_csv(MULTICLASS_DATASET)
        analysis = analyze_target(frame, "class_label")
        self.assertEqual(analysis.problem_type, ProblemType.MULTICLASS_CLASSIFICATION)
        self.assertTrue(analysis.is_supported)
        self.assertEqual(analysis.class_count, MULTICLASS_EXPECTED["class_count"])

    def test_binary_categorical_target(self) -> None:
        frame = pd.DataFrame({"status": ["yes", "no", "yes", "no", "yes", "no"]})
        result = detect_problem_type(frame["status"])
        self.assertEqual(result.problem_type, ProblemType.BINARY_CLASSIFICATION)

    def test_boolean_target_is_binary(self) -> None:
        frame = pd.DataFrame({"flag": [True, False] * 10})
        result = detect_problem_type(frame["flag"])
        self.assertEqual(result.problem_type, ProblemType.BINARY_CLASSIFICATION)

    def test_continuous_target_is_regression(self) -> None:
        frame = pd.DataFrame({"price": [1.5, 2.75, 3.1, 4.9, 5.25, 6.4, 7.8, 8.9]})
        result = detect_problem_type(frame["price"])
        self.assertEqual(result.problem_type, ProblemType.REGRESSION)
        self.assertFalse(result.is_supported)
        self.assertFalse(result.is_classification)
        self.assertIn("regression", result.reason.lower())

    def test_integer_valued_regression_target(self) -> None:
        # Integers with a high unique ratio are still a regression target.
        frame = pd.DataFrame({"value": list(range(100))})
        result = detect_problem_type(frame["value"])
        self.assertEqual(result.problem_type, ProblemType.REGRESSION)

    def test_high_cardinality_text_target_is_uncertain(self) -> None:
        frame = pd.DataFrame({"text": [f"id-{index}" for index in range(60)]})
        result = detect_problem_type(frame["text"])
        self.assertEqual(result.problem_type, ProblemType.UNSUPPORTED_OR_UNCERTAIN)
        self.assertFalse(result.is_supported)

    def test_constant_target_is_uncertain(self) -> None:
        frame = pd.DataFrame({"constant": ["a"] * 30})
        result = detect_problem_type(frame["constant"])
        self.assertEqual(result.problem_type, ProblemType.UNSUPPORTED_OR_UNCERTAIN)

    def test_all_missing_target_is_uncertain(self) -> None:
        frame = pd.DataFrame({"empty": [None] * 10})
        result = detect_problem_type(frame["empty"])
        self.assertEqual(result.problem_type, ProblemType.UNSUPPORTED_OR_UNCERTAIN)
        self.assertEqual(result.class_count, 0)

    def test_analyze_target_invalid_column_raises(self) -> None:
        frame = load_csv(BINARY_DATASET)
        with self.assertRaises(InvalidTargetColumnError):
            analyze_target(frame, "missing_column")


class TestClassDistribution(unittest.TestCase):
    """9. Class distribution."""

    def test_binary_distribution(self) -> None:
        frame = load_csv(BINARY_DATASET)
        distribution = calculate_class_distribution(frame["Outcome"])
        self.assertEqual(set(distribution), {"0", "1"})
        total = sum(entry.count for entry in distribution.values())
        self.assertEqual(total, BINARY_EXPECTED["rows"])
        self.assertAlmostEqual(
            sum(entry.percentage for entry in distribution.values()), 100.0, places=1
        )

    def test_distribution_counts_match_pandas(self) -> None:
        frame = load_csv(BINARY_DATASET)
        distribution = calculate_class_distribution(frame["Outcome"])
        for label, entry in distribution.items():
            # Labels are stringified in the structured output.
            self.assertEqual(entry.count, int((frame["Outcome"] == int(label)).sum()))

    def test_distribution_ignores_missing_values(self) -> None:
        series = pd.Series(["a", "b", None, "a"])
        distribution = calculate_class_distribution(series)
        self.assertEqual(distribution["a"].count, 2)
        self.assertEqual(distribution["a"].percentage, 66.67)

    def test_multiclass_distribution_has_three_classes(self) -> None:
        frame = load_csv(MULTICLASS_DATASET)
        distribution = calculate_class_distribution(frame["class_label"])
        self.assertEqual(len(distribution), MULTICLASS_EXPECTED["class_count"])
        self.assertAlmostEqual(
            sum(entry.percentage for entry in distribution.values()), 100.0, places=1
        )

    def test_empty_series_distribution(self) -> None:
        self.assertEqual(calculate_class_distribution(pd.Series([], dtype="object")), {})


class TestDataQuality(unittest.TestCase):
    """10. Data quality checks."""

    def test_issues_on_sample_dataset(self) -> None:
        frame = load_csv(BINARY_DATASET)
        issues = detect_data_quality_issues(frame, target_column="Outcome")
        kinds = issue_kinds(issues)

        self.assertEqual(kinds["missing_values"], 4)
        self.assertEqual(kinds["duplicate_rows"], 1)
        self.assertEqual(kinds["high_cardinality_column"], 1)

    def test_constant_column_detected(self) -> None:
        frame = pd.DataFrame({"constant": ["same"] * 10, "value": range(10)})
        issues = detect_data_quality_issues(frame)
        kinds = issue_kinds(issues)
        self.assertEqual(kinds["constant_column"], 1)
        constant_issue = next(i for i in issues if i.kind == "constant_column")
        self.assertEqual(constant_issue.column, "constant")

    def test_empty_column_detected(self) -> None:
        frame = pd.DataFrame({"empty": [None] * 8, "value": range(8)})
        issues = detect_data_quality_issues(frame)
        kinds = issue_kinds(issues)
        self.assertEqual(kinds["empty_column"], 1)
        self.assertEqual(issues[0].severity, "error")

    def test_high_missing_column_has_higher_severity(self) -> None:
        frame = pd.DataFrame(
            {
                "mostly_missing": [None] * 8 + [1.0, 2.0],
                "slightly_missing": [None] + list(range(9)),
            }
        )
        issues = detect_data_quality_issues(frame)
        severities = {issue.column: issue.severity for issue in issues if issue.kind == "missing_values"}
        self.assertEqual(severities["mostly_missing"], "error")
        self.assertEqual(severities["slightly_missing"], "info")

    def test_unsupported_dtype_reported(self) -> None:
        frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01"] * 5), "value": range(5)})
        issues = detect_data_quality_issues(frame)
        self.assertIn("unsupported_column_type", issue_kinds(issues))

    def test_rare_class_reported(self) -> None:
        series = pd.Series(["a"] * 20 + ["b"] * 19 + ["c"])
        distribution = calculate_class_distribution(series)
        issues = detect_data_quality_issues(
            pd.DataFrame({"target": series}), target_column="target",
            class_distribution=distribution,
        )
        self.assertIn("rare_class", issue_kinds(issues))

    def test_imbalanced_classes_reported(self) -> None:
        series = pd.Series(["a"] * 95 + ["b"] * 5)
        distribution = calculate_class_distribution(series)
        issues = detect_data_quality_issues(
            pd.DataFrame({"target": series}),
            target_column="target",
            class_distribution=distribution,
        )
        imbalance = [issue for issue in issues if issue.kind == "imbalanced_classes"]
        self.assertEqual(len(imbalance), 1)
        self.assertEqual(imbalance[0].details["majority_class"], "a")

    def test_moderately_skewed_classes_not_reported_as_imbalanced(self) -> None:
        series = pd.Series(["a"] * 62 + ["b"] * 38)
        distribution = calculate_class_distribution(series)
        issues = detect_data_quality_issues(
            pd.DataFrame({"target": series}),
            target_column="target",
            class_distribution=distribution,
        )
        self.assertNotIn("imbalanced_classes", issue_kinds(issues))

    def test_issues_do_not_modify_dataframe(self) -> None:
        frame = load_csv(BINARY_DATASET)
        before_shape = frame.shape
        before_columns = list(frame.columns)
        before_missing = int(frame.isna().sum().sum())
        detect_data_quality_issues(frame, target_column="Outcome")
        self.assertEqual(frame.shape, before_shape)
        self.assertEqual(list(frame.columns), before_columns)
        self.assertEqual(int(frame.isna().sum().sum()), before_missing)

    def test_issues_sorted_by_severity(self) -> None:
        frame = load_csv(BINARY_DATASET)
        issues = detect_data_quality_issues(frame, target_column="Outcome")
        order = {"error": 0, "warning": 1, "info": 2}
        ranks = [order[issue.severity] for issue in issues]
        self.assertEqual(ranks, sorted(ranks))

    def test_no_issues_for_clean_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = 30
            frame = pd.DataFrame(
                {
                    "age": [20 + (index % 10) * 3 for index in range(rows)],
                    "city": [["amsterdam", "berlin", "cairo"][index % 3] for index in range(rows)],
                    "target": [index % 2 for index in range(rows)],
                }
            )
            path = write_csv(Path(tmp), "clean.csv", frame)
            loaded = load_csv(path)
            issues = detect_data_quality_issues(
                loaded,
                target_column="target",
                class_distribution=calculate_class_distribution(loaded["target"]),
            )
            self.assertEqual(issues, [])


class TestAggregateAnalysis(unittest.TestCase):
    """End-to-end tool pipeline."""

    def test_analyze_dataset_on_binary_sample(self) -> None:
        frame = load_csv(BINARY_DATASET)
        analysis = analyze_dataset(frame)

        self.assertEqual(analysis.shape.rows, BINARY_EXPECTED["rows"])
        self.assertEqual(analysis.shape.columns, BINARY_EXPECTED["columns"])
        self.assertEqual(analysis.target_detection.column, "Outcome")
        self.assertEqual(analysis.target_analysis.problem_type, ProblemType.BINARY_CLASSIFICATION)
        self.assertEqual(analysis.missing_values.total, BINARY_EXPECTED["missing_total"])
        self.assertEqual(analysis.duplicate_count, BINARY_EXPECTED["duplicates"])
        self.assertNotIn("Outcome", analysis.feature_types.numerical_features)
        self.assertNotIn("Outcome", analysis.feature_types.categorical_features)

    def test_analyze_dataset_with_explicit_target(self) -> None:
        frame = load_csv(BINARY_DATASET)
        analysis = analyze_dataset(frame, explicit_target="smoking")
        self.assertEqual(analysis.target_detection.method, "explicit")
        # 'smoking' has three values, so the task becomes multiclass.
        self.assertEqual(
            analysis.target_analysis.problem_type, ProblemType.MULTICLASS_CLASSIFICATION
        )

    def test_analyze_dataset_without_target(self) -> None:
        frame = pd.DataFrame({f"c{i}": [f"{i}_{row}" for row in range(20)] for i in range(3)})
        analysis = analyze_dataset(frame)
        self.assertIsNone(analysis.target_detection.column)
        self.assertIsNone(analysis.target_analysis)


if __name__ == "__main__":
    unittest.main(verbosity=2)