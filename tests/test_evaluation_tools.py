"""Tests for the reusable evaluation tools (Phase 5).

The module under test contains no training code, so every case here is a pure
function call on hand-built records. That is deliberate: the tests below pin the
numbers (gap arithmetic, confusion-matrix counts, timings, counts) and the
wording rules (no unsupported overfitting claim, observed vs interpreted).
"""

from __future__ import annotations

import json
import re
import unittest

from tools.evaluation_tools import (
    EvaluationDataError,
    analyze_confusion_matrix,
    analyze_execution_time,
    analyze_performance_gaps,
    build_evaluation_report,
    build_evaluation_table,
    calculate_cv_holdout_gap,
    classify_performance_gap,
    compare_models,
    count_experiments,
    generate_evaluation_summary,
    generate_limitations,
    generate_model_recommendation,
    rank_models,
)
from tools.training_tools import rank_experiments
from utils import config
from utils.state import (
    ConfusionMatrixData,
    CrossValidationResult,
    EvaluationStatus,
    ExperimentResult,
    ExperimentStatus,
    GapClassification,
    ModelEvaluation,
    to_serializable,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def experiment(
    model_name: str = "Logistic Regression",
    *,
    accuracy: float | None = 0.875,
    precision: float | None = 0.88,
    recall: float | None = 0.87,
    f1_score: float | None = 0.876,
    cv_f1_mean: float | None = 0.894,
    cv_accuracy_mean: float | None = 0.895,
    status: str = ExperimentStatus.SUCCESS.value,
    confusion: ConfusionMatrixData | None = None,
    training_time: float = 0.03,
    prediction_time: float = 0.01,
    cv_time: float = 0.20,
    model_type: str = "linear",
) -> ExperimentResult:
    """Build one `ExperimentResult` with sensible defaults."""
    return ExperimentResult(
        experiment_id=f"exp_{model_name.lower().replace(' ', '_')}",
        model_name=model_name,
        model_type=model_type,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        cv_f1_mean=cv_f1_mean,
        cv_accuracy_mean=cv_accuracy_mean,
        training_time=training_time,
        prediction_time=prediction_time,
        cv_time=cv_time,
        total_time=round(training_time + prediction_time + cv_time, 6),
        confusion_matrix=confusion,
        status=status,
    )


def cv_result(
    model_name: str = "Logistic Regression",
    *,
    f1_mean: float | None = 0.894,
    accuracy_mean: float | None = 0.895,
    status: str = "success",
    n_splits: int = 5,
) -> CrossValidationResult:
    """Build one `CrossValidationResult`."""
    return CrossValidationResult(
        model_name=model_name,
        n_splits=n_splits if status == "success" else 0,
        status=status,
        accuracy_mean=accuracy_mean,
        f1_mean=f1_mean,
        accuracy_std=0.05 if status == "success" else None,
        f1_std=0.04 if status == "success" else None,
    )


BINARY_MATRIX = ConfusionMatrixData(
    labels=["0", "1"],
    matrix=[[8, 1], [2, 13]],
    total=24,
)

# Rows are true labels, columns are predicted labels (scikit-learn convention).
#   alpha: 6 correct
#   beta : 1 predicted as alpha, 5 correct, 1 predicted as gamma
#   gamma: 2 predicted as alpha, 1 predicted as beta, 3 correct
MULTICLASS_MATRIX = ConfusionMatrixData(
    labels=["alpha", "beta", "gamma"],
    matrix=[
        [6, 0, 0],
        [1, 5, 1],
        [2, 1, 3],
    ],
    total=19,
)

# alpha is classified perfectly, beta and gamma each contribute two errors.
PERFECT_CLASS_MATRIX = ConfusionMatrixData(
    labels=["alpha", "beta", "gamma"],
    matrix=[
        [6, 0, 0],
        [0, 6, 1],
        [0, 1, 4],
    ],
    total=18,
)


# --------------------------------------------------------------------------- #
# 2, 3, 7: gap calculation
# --------------------------------------------------------------------------- #
class TestGapCalculation(unittest.TestCase):
    """The signed difference between the hold-out score and the CV mean."""

    def test_f1_gap_is_holdout_minus_cv(self) -> None:
        self.assertAlmostEqual(calculate_cv_holdout_gap(0.876, 0.894), -0.018, places=6)

    def test_accuracy_gap_is_holdout_minus_cv(self) -> None:
        self.assertAlmostEqual(calculate_cv_holdout_gap(0.90, 0.85), 0.05, places=6)

    def test_positive_gap_keeps_its_sign(self) -> None:
        self.assertGreater(calculate_cv_holdout_gap(0.95, 0.90), 0)

    def test_negative_gap_keeps_its_sign(self) -> None:
        self.assertLess(calculate_cv_holdout_gap(0.90, 0.95), 0)

    def test_zero_gap(self) -> None:
        self.assertEqual(calculate_cv_holdout_gap(1.0, 1.0), 0.0)

    def test_missing_cv_value_gives_no_gap(self) -> None:
        self.assertIsNone(calculate_cv_holdout_gap(0.876, None))

    def test_missing_holdout_value_gives_no_gap(self) -> None:
        self.assertIsNone(calculate_cv_holdout_gap(None, 0.894))

    def test_both_missing_gives_no_gap(self) -> None:
        self.assertIsNone(calculate_cv_holdout_gap(None, None))


# --------------------------------------------------------------------------- #
# 4, 5, 6: gap classification
# --------------------------------------------------------------------------- #
class TestGapClassification(unittest.TestCase):
    """Close / moderate / substantial boundaries of the reporting heuristic."""

    def test_small_gap_is_close(self) -> None:
        self.assertIs(classify_performance_gap(0.018), GapClassification.CLOSE)

    def test_tiny_gap_is_close(self) -> None:
        self.assertIs(classify_performance_gap(-0.001), GapClassification.CLOSE)

    def test_medium_gap_is_moderate(self) -> None:
        self.assertIs(classify_performance_gap(0.05), GapClassification.MODERATE)

    def test_medium_negative_gap_is_moderate(self) -> None:
        self.assertIs(classify_performance_gap(-0.05), GapClassification.MODERATE)

    def test_large_gap_is_substantial(self) -> None:
        self.assertIs(classify_performance_gap(0.10), GapClassification.SUBSTANTIAL)

    def test_large_negative_gap_is_substantial(self) -> None:
        self.assertIs(classify_performance_gap(-0.1019), GapClassification.SUBSTANTIAL)

    def test_exact_close_boundary_is_moderate(self) -> None:
        # |gap| < 0.03 is close, so exactly 0.03 is already moderate.
        self.assertIs(
            classify_performance_gap(0.03), GapClassification.MODERATE
        )

    def test_just_below_close_boundary_is_close(self) -> None:
        self.assertIs(classify_performance_gap(0.0299), GapClassification.CLOSE)

    def test_missing_gap_is_not_available(self) -> None:
        self.assertIs(classify_performance_gap(None), GapClassification.NOT_AVAILABLE)

    def test_thresholds_are_configurable(self) -> None:
        self.assertIs(
            classify_performance_gap(0.05, close_threshold=0.1, substantial_threshold=0.2),
            GapClassification.CLOSE,
        )

    def test_contradictory_thresholds_raise(self) -> None:
        with self.assertRaises(EvaluationDataError):
            classify_performance_gap(0.05, close_threshold=0.5, substantial_threshold=0.1)


# --------------------------------------------------------------------------- #
# 8, 9, 10, 11: confusion matrix analysis
# --------------------------------------------------------------------------- #
class TestConfusionMatrixAnalysis(unittest.TestCase):
    """Counts, ratios and the derived per-class metrics."""

    def test_binary_counts_correct_and_incorrect(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX, model_name="Logistic Regression")
        self.assertTrue(analysis.available)
        self.assertEqual(analysis.total_samples, 24)
        self.assertEqual(analysis.correct_predictions, 21)
        self.assertEqual(analysis.incorrect_predictions, 3)

    def test_binary_accuracy_from_matrix(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        self.assertAlmostEqual(analysis.accuracy_from_matrix, 21 / 24, places=6)

    def test_binary_true_positives_are_on_the_diagonal(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        by_label = {item.label: item for item in analysis.class_metrics}
        self.assertEqual(by_label["0"].true_positives, 8)
        self.assertEqual(by_label["1"].true_positives, 13)

    def test_binary_false_positives_and_negatives(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        by_label = {item.label: item for item in analysis.class_metrics}
        # With "0" as the positive class: column 0 holds the predicted-0 counts,
        # so the 2 samples of class 1 predicted as 0 are false positives.
        self.assertEqual(by_label["0"].false_positives, 2)
        self.assertEqual(by_label["0"].false_negatives, 1)
        self.assertEqual(by_label["0"].true_negatives, 13)
        self.assertEqual(by_label["1"].false_positives, 1)
        self.assertEqual(by_label["1"].false_negatives, 2)
        self.assertEqual(by_label["1"].true_negatives, 8)

    def test_binary_support_sums_to_the_total(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        self.assertEqual(sum(item.support for item in analysis.class_metrics), 24)

    def test_derived_per_class_precision_and_recall(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        by_label = {item.label: item for item in analysis.class_metrics}
        # precision = TP / (TP + FP), recall = TP / (TP + FN)
        self.assertAlmostEqual(by_label["0"].precision, 8 / 10, places=6)
        self.assertAlmostEqual(by_label["0"].recall, 8 / 9, places=6)
        self.assertAlmostEqual(by_label["1"].precision, 13 / 14, places=6)
        self.assertAlmostEqual(by_label["1"].recall, 13 / 15, places=6)
        self.assertTrue(by_label["0"].derived)

    def test_derived_f1_is_the_harmonic_mean(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        item = next(entry for entry in analysis.class_metrics if entry.label == "0")
        expected = 2 * (8 / 10) * (8 / 9) / ((8 / 10) + (8 / 9))
        self.assertAlmostEqual(item.f1_score, expected, places=5)

    def test_derived_weighted_metrics_match_phase_four(self) -> None:
        """The derived per-class values reproduce the weighted metrics.

        This is the strongest check that deriving per-class numbers from the
        stored matrix is legitimate: the weighted average of the derived values
        must equal the weighted metric Phase 4 computed with scikit-learn.
        """
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        total = analysis.total_samples
        weighted_f1 = sum(
            item.f1_score * item.support / total for item in analysis.class_metrics
        )
        weighted_precision = sum(
            item.precision * item.support / total for item in analysis.class_metrics
        )
        weighted_recall = sum(
            item.recall * item.support / total for item in analysis.class_metrics
        )
        self.assertAlmostEqual(weighted_f1, 0.8761, places=3)
        self.assertAlmostEqual(weighted_precision, 0.8804, places=3)
        self.assertAlmostEqual(weighted_recall, 0.875, places=3)

    def test_binary_error_pattern_mentions_the_dominant_direction(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        self.assertIn(
            "false positives (2) exceed false negatives (1)", analysis.error_pattern
        )

    def test_binary_error_pattern_names_the_label_it_read_as_positive(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        self.assertIn("Reading the label '0' as the positive class", analysis.error_pattern)

    def test_binary_error_pattern_avoids_domain_claims(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        lowered = analysis.error_pattern.lower()
        for forbidden in ("patient", "disease", "risk", "fraud", "customer"):
            self.assertNotIn(forbidden, lowered)

    def test_multiclass_counts(self) -> None:
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX, model_name="KNN")
        self.assertEqual(analysis.total_samples, 19)
        self.assertEqual(analysis.correct_predictions, 14)
        self.assertEqual(analysis.incorrect_predictions, 5)

    def test_multiclass_per_class_counts(self) -> None:
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX)
        by_label = {item.label: item for item in analysis.class_metrics}
        self.assertEqual(by_label["alpha"].support, 6)
        self.assertEqual(by_label["alpha"].false_positives, 3)
        self.assertEqual(by_label["alpha"].false_negatives, 0)
        self.assertEqual(by_label["beta"].support, 7)
        self.assertEqual(by_label["beta"].false_positives, 1)
        self.assertEqual(by_label["beta"].false_negatives, 2)
        self.assertEqual(by_label["gamma"].support, 6)
        self.assertEqual(by_label["gamma"].false_positives, 1)
        self.assertEqual(by_label["gamma"].false_negatives, 3)

    def test_multiclass_error_counts_are_not_double_counted(self) -> None:
        """Every mistake is one false negative *and* one false positive.

        The reported total must therefore be the off-diagonal count, not the sum
        of the per-class false negatives and false positives.
        """
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX)
        summed = sum(
            item.false_negatives + item.false_positives for item in analysis.class_metrics
        )
        self.assertEqual(summed, 2 * analysis.incorrect_predictions)
        self.assertIn("misclassified 5 hold-out", analysis.error_pattern)

    def test_multiclass_identifies_classes_with_most_errors(self) -> None:
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX)
        # gamma: 3 missed + 1 wrongly assigned, then alpha and beta with 3 each.
        self.assertEqual(
            analysis.classes_with_most_errors, ["gamma", "alpha", "beta"]
        )

    def test_perfect_class_is_not_listed_among_the_error_classes(self) -> None:
        analysis = analyze_confusion_matrix(PERFECT_CLASS_MATRIX)
        self.assertNotIn("alpha", analysis.classes_with_most_errors)
        self.assertEqual(analysis.classes_with_most_errors, ["beta", "gamma"])

    def test_multiclass_identifies_dominant_confusion_pairs(self) -> None:
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX)
        pairs = {
            (pair.true_label, pair.predicted_label): pair.count
            for pair in analysis.dominant_confusion_pairs
        }
        # Four off-diagonal cells exist, but only TOP_CONFUSION_PAIRS are kept.
        self.assertEqual(pairs[("gamma", "alpha")], 2)
        self.assertEqual(pairs[("beta", "alpha")], 1)
        self.assertEqual(pairs[("beta", "gamma")], 1)
        self.assertNotIn(("gamma", "beta"), pairs)
        # Pairs are ordered by descending count.
        counts = [pair.count for pair in analysis.dominant_confusion_pairs]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_multiclass_pattern_states_no_domain_interpretation(self) -> None:
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX)
        self.assertIn("No domain interpretation is added", analysis.error_pattern)

    def test_perfect_matrix_reports_no_error_pattern(self) -> None:
        perfect = ConfusionMatrixData(labels=["a", "b"], matrix=[[5, 0], [0, 5]], total=10)
        analysis = analyze_confusion_matrix(perfect)
        self.assertEqual(analysis.incorrect_predictions, 0)
        self.assertIn("classified correctly", analysis.error_pattern)

    def test_single_label_matrix_is_handled(self) -> None:
        single = ConfusionMatrixData(labels=["only"], matrix=[[4]], total=4)
        analysis = analyze_confusion_matrix(single)
        self.assertEqual(analysis.correct_predictions, 4)
        self.assertIn("single label", analysis.error_pattern)

    def test_zero_division_yields_none_instead_of_nan(self) -> None:
        # Class "a" is never true and never predicted: 0/0 everywhere.
        matrix = ConfusionMatrixData(
            labels=["a", "b"], matrix=[[0, 0], [0, 3]], total=3
        )
        analysis = analyze_confusion_matrix(matrix)
        by_label = {item.label: item for item in analysis.class_metrics}
        self.assertIsNone(by_label["a"].precision)
        self.assertIsNone(by_label["a"].recall)
        self.assertIsNone(by_label["a"].f1_score)

    def test_empty_matrix_is_reported(self) -> None:
        empty = ConfusionMatrixData(labels=["a", "b"], matrix=[[0, 0], [0, 0]], total=0)
        analysis = analyze_confusion_matrix(empty)
        self.assertEqual(analysis.total_samples, 0)
        self.assertIsNone(analysis.accuracy_from_matrix)
        self.assertTrue(any("empty" in note for note in analysis.notes))

    def test_top_pairs_limit_is_respected(self) -> None:
        analysis = analyze_confusion_matrix(MULTICLASS_MATRIX, top_pairs=1)
        self.assertEqual(len(analysis.dominant_confusion_pairs), 1)

    def test_labels_are_strings(self) -> None:
        numeric = ConfusionMatrixData(labels=[0, 1], matrix=[[1, 0], [0, 1]], total=2)
        analysis = analyze_confusion_matrix(numeric)
        self.assertEqual(analysis.labels, ["0", "1"])
        json.dumps(to_serializable(analysis))


# --------------------------------------------------------------------------- #
# 18: missing confusion matrix
# --------------------------------------------------------------------------- #
class TestMissingConfusionMatrix(unittest.TestCase):
    """A missing matrix is reported, never replaced by invented numbers."""

    def test_missing_matrix_returns_unavailable_record(self) -> None:
        analysis = analyze_confusion_matrix(None, model_name="Decision Tree")
        self.assertFalse(analysis.available)
        self.assertEqual(analysis.model_name, "Decision Tree")
        self.assertEqual(analysis.total_samples, 0)
        self.assertEqual(analysis.correct_predictions, 0)
        self.assertEqual(analysis.incorrect_predictions, 0)
        self.assertIsNone(analysis.accuracy_from_matrix)
        self.assertEqual(analysis.class_metrics, [])
        self.assertTrue(analysis.notes)

    def test_missing_matrix_records_the_per_class_note(self) -> None:
        analysis = analyze_confusion_matrix(None)
        self.assertEqual(
            analysis.per_class_metrics_note, config.PER_CLASS_UNAVAILABLE_NOTE
        )

    def test_available_matrix_records_the_derivation_note(self) -> None:
        analysis = analyze_confusion_matrix(BINARY_MATRIX)
        self.assertEqual(
            analysis.per_class_metrics_note, config.PER_CLASS_DERIVATION_NOTE
        )

    def test_malformed_matrix_raises(self) -> None:
        ragged = ConfusionMatrixData(labels=["a", "b"], matrix=[[1, 0, 0], [0, 1, 0]])
        with self.assertRaises(EvaluationDataError):
            analyze_confusion_matrix(ragged)

    def test_wrong_row_count_raises(self) -> None:
        wrong = ConfusionMatrixData(labels=["a", "b"], matrix=[[1, 0]])
        with self.assertRaises(EvaluationDataError):
            analyze_confusion_matrix(wrong)

    def test_negative_count_raises(self) -> None:
        negative = ConfusionMatrixData(labels=["a", "b"], matrix=[[1, 0], [0, -1]])
        with self.assertRaises(EvaluationDataError):
            analyze_confusion_matrix(negative)

    def test_non_numeric_count_raises(self) -> None:
        text = ConfusionMatrixData(labels=["a", "b"], matrix=[[1, "x"], [0, 1]])
        with self.assertRaises(EvaluationDataError):
            analyze_confusion_matrix(text)


# --------------------------------------------------------------------------- #
# 12, 13: comparison and ranking
# --------------------------------------------------------------------------- #
class TestModelRanking(unittest.TestCase):
    """The Phase 5 ranking must agree with the documented Phase 4 rule."""

    def test_rank_uses_f1_first(self) -> None:
        evaluations = compare_models(
            [
                experiment("Logistic Regression", f1_score=0.70, accuracy=0.99),
                experiment("Decision Tree", f1_score=0.90, accuracy=0.50),
            ]
        )
        ranked = rank_models(evaluations)
        self.assertEqual(ranked[0].model_name, "Decision Tree")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].rank, 2)

    def test_rank_falls_back_to_accuracy_on_equal_f1(self) -> None:
        evaluations = compare_models(
            [
                experiment("Logistic Regression", f1_score=0.90, accuracy=0.70),
                experiment("Decision Tree", f1_score=0.90, accuracy=0.95),
            ]
        )
        ranked = rank_models(evaluations)
        self.assertEqual(ranked[0].model_name, "Decision Tree")

    def test_tie_is_broken_by_model_name(self) -> None:
        evaluations = compare_models(
            [
                experiment("Random Forest", f1_score=1.0, accuracy=1.0),
                experiment("Decision Tree", f1_score=1.0, accuracy=1.0),
            ]
        )
        ranked = rank_models(evaluations)
        self.assertEqual(ranked[0].model_name, "Decision Tree")
        self.assertEqual(ranked[1].model_name, "Random Forest")

    def test_failed_experiment_is_ranked_last_without_a_rank(self) -> None:
        failed = experiment("Random Forest", status=ExperimentStatus.FAILED.value,
                            f1_score=None, accuracy=None, cv_f1_mean=None)
        evaluations = compare_models(
            [failed, experiment("Decision Tree", f1_score=0.9)]
        )
        ranked = rank_models(evaluations)
        self.assertEqual(ranked[0].model_name, "Decision Tree")
        self.assertEqual(ranked[-1].model_name, "Random Forest")
        self.assertIsNone(ranked[-1].rank)

    def test_every_experiment_produces_exactly_one_evaluation(self) -> None:
        experiments = [
            experiment("Logistic Regression"),
            experiment("Decision Tree"),
            experiment("Random Forest", status=ExperimentStatus.FAILED.value),
        ]
        self.assertEqual(len(compare_models(experiments)), 3)

    def test_evaluation_agrees_with_phase_four_ranking(self) -> None:
        """Phase 5 must not announce a different winner than Phase 4."""
        experiments = [
            experiment("Logistic Regression", f1_score=0.8761, accuracy=0.875),
            experiment("Support Vector Machine", f1_score=0.8357, accuracy=0.8333),
            experiment("K-Nearest Neighbors", f1_score=0.7935, accuracy=0.7917),
        ]
        phase_four = rank_experiments(experiments)
        phase_five = rank_models(compare_models(experiments))
        self.assertEqual(phase_four.best_model_name, phase_five[0].model_name)
        self.assertEqual(
            [entry.model_name for entry in phase_four.entries],
            [item.model_name for item in phase_five],
        )

    def test_ranking_is_deterministic_across_calls(self) -> None:
        experiments = [
            experiment("K-Nearest Neighbors", f1_score=0.5),
            experiment("Decision Tree", f1_score=0.5),
            experiment("Logistic Regression", f1_score=0.5),
        ]
        first = [item.model_name for item in rank_models(compare_models(experiments))]
        second = [item.model_name for item in rank_models(compare_models(experiments))]
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_empty_input_is_handled(self) -> None:
        self.assertEqual(compare_models([]), [])
        self.assertEqual(rank_models([]), [])
        self.assertIn("no experiment", build_evaluation_table([]).lower())


# --------------------------------------------------------------------------- #
# 17: missing cross-validation results
# --------------------------------------------------------------------------- #
class TestMissingCrossValidation(unittest.TestCase):
    """No CV record means no gap, and the report says so."""

    def test_gap_is_none_without_cv_result(self) -> None:
        evaluations = compare_models(
            [experiment("Decision Tree", cv_f1_mean=None, cv_accuracy_mean=None)]
        )
        item = evaluations[0]
        self.assertIsNone(item.f1_gap)
        self.assertIsNone(item.accuracy_gap)
        self.assertEqual(item.f1_gap_classification, GapClassification.NOT_AVAILABLE.value)

    def test_partial_status_when_cv_is_missing(self) -> None:
        evaluations = compare_models([experiment("Decision Tree", cv_f1_mean=None)])
        self.assertEqual(evaluations[0].evaluation_status, EvaluationStatus.PARTIAL.value)

    def test_cv_record_supplies_the_fold_deviation(self) -> None:
        evaluations = compare_models(
            [experiment("Decision Tree", cv_f1_mean=None, cv_accuracy_mean=None)],
            [cv_result("Decision Tree", f1_mean=0.80, accuracy_mean=0.79)],
        )
        item = evaluations[0]
        self.assertAlmostEqual(item.f1_gap, 0.076, places=6)
        self.assertIsNotNone(item.cv_f1_std)
        self.assertEqual(item.cv_n_splits, 5)
        self.assertEqual(item.evaluation_status, EvaluationStatus.COMPLETED.value)

    def test_skipped_cv_record_is_not_used(self) -> None:
        evaluations = compare_models(
            [experiment("Decision Tree", cv_f1_mean=None, cv_accuracy_mean=None)],
            [cv_result("Decision Tree", status="skipped", f1_mean=None, accuracy_mean=None)],
        )
        self.assertIsNone(evaluations[0].f1_gap)

    def test_gap_analysis_lists_models_without_gap(self) -> None:
        evaluations = rank_models(
            compare_models(
                [
                    experiment("Decision Tree", cv_f1_mean=None, cv_accuracy_mean=None),
                    experiment("Random Forest"),
                ]
            )
        )
        analysis = analyze_performance_gaps(evaluations)
        self.assertEqual(analysis.models_without_gap, ["Decision Tree"])
        self.assertEqual(analysis.models_with_gap, 1)
        self.assertIn("Random Forest", analysis.close_agreement_models)

    def test_gap_analysis_without_any_cv_result(self) -> None:
        evaluations = rank_models(
            compare_models(
                [
                    experiment("Decision Tree", cv_f1_mean=None, cv_accuracy_mean=None),
                    experiment("Random Forest", cv_f1_mean=None, cv_accuracy_mean=None),
                ]
            )
        )
        analysis = analyze_performance_gaps(evaluations)
        self.assertEqual(analysis.models_with_gap, 0)
        self.assertEqual(len(analysis.models_without_gap), 2)
        self.assertIn("No F1-score gap could be computed", analysis.summary)
        self.assertIsNone(analysis.largest_absolute_gap)


# --------------------------------------------------------------------------- #
# Gap analysis aggregation
# --------------------------------------------------------------------------- #
class TestGapAnalysis(unittest.TestCase):
    """The aggregate view keeps the direction and stays non-committal."""

    def _evaluations(self) -> list:
        return rank_models(
            compare_models(
                [
                    experiment("A", f1_score=0.90, accuracy=0.90, cv_f1_mean=0.89,
                               cv_accuracy_mean=0.90),
                    experiment("B", f1_score=0.80, accuracy=0.80, cv_f1_mean=0.85,
                               cv_accuracy_mean=0.85),
                    experiment("C", f1_score=0.70, accuracy=0.70, cv_f1_mean=0.50,
                               cv_accuracy_mean=0.50),
                ]
            )
        )

    def test_close_models_are_grouped(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations())
        self.assertEqual(analysis.close_agreement_models, ["A"])

    def test_moderate_models_are_grouped(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations())
        self.assertEqual(analysis.moderate_difference_models, ["B"])

    def test_substantial_models_are_grouped(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations())
        self.assertEqual(analysis.substantial_difference_models, ["C"])

    def test_largest_gap_is_tracked(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations())
        # C: hold-out 0.70 minus CV 0.50 = +0.20, the largest absolute gap.
        self.assertAlmostEqual(analysis.largest_absolute_gap, 0.20, places=6)
        self.assertEqual(analysis.largest_absolute_gap_model, "C")

    def test_thresholds_and_rule_are_recorded(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations())
        self.assertEqual(analysis.close_threshold, config.GAP_CLOSE_THRESHOLD)
        self.assertEqual(analysis.substantial_threshold, config.GAP_SUBSTANTIAL_THRESHOLD)
        self.assertEqual(analysis.rule, config.GAP_CLASSIFICATION_RULE)
        self.assertEqual(analysis.disclaimer, config.GAP_HEURISTIC_DISCLAIMER)

    def test_accuracy_metric_can_be_aggregated(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations(), metric="accuracy")
        self.assertEqual(analysis.metric, "accuracy")
        self.assertEqual(analysis.models_with_gap, 3)

    def test_substantial_gap_uses_the_cautious_wording(self) -> None:
        analysis = analyze_performance_gaps(self._evaluations())
        self.assertEqual(analysis.interpretation, config.OVERFITTING_CAUTION_NOTE)

    def test_close_gaps_are_described_without_overfitting_claims(self) -> None:
        evaluations = rank_models(
            compare_models([experiment("A", f1_score=0.90, cv_f1_mean=0.90)])
        )
        analysis = analyze_performance_gaps(evaluations)
        self.assertIn("close-agreement", analysis.interpretation.lower())
        self.assertNotIn("overfitting", analysis.interpretation.lower())

    def test_per_model_interpretation_separates_numbers_from_wording(self) -> None:
        item = rank_models(compare_models([experiment()]))[0]
        self.assertIn("hold-out score is 0.8760", item.gap_interpretation)
        self.assertIn("0.0180 lower than", item.gap_interpretation)
        self.assertIn("close", item.gap_interpretation)
        self.assertNotIn("is overfitting", item.gap_interpretation)


# --------------------------------------------------------------------------- #
# 14: execution time analysis
# --------------------------------------------------------------------------- #
class TestExecutionTimeAnalysis(unittest.TestCase):
    """Totals, averages and the fastest/slowest successful model."""

    def test_total_and_average_time(self) -> None:
        experiments = [
            experiment("A", training_time=0.1, prediction_time=0.1, cv_time=0.2),
            experiment("B", training_time=0.3, prediction_time=0.1, cv_time=0.4),
        ]
        analysis = analyze_execution_time(experiments)
        # 0.4 s and 0.8 s of total experiment time.
        self.assertAlmostEqual(analysis.total_experiment_time, 1.2, places=6)
        self.assertAlmostEqual(analysis.average_experiment_time, 0.6, places=6)

    def test_fastest_and_slowest_model(self) -> None:
        experiments = [
            experiment("Fast", training_time=0.01, prediction_time=0.0, cv_time=0.0),
            experiment("Medium", training_time=0.10, prediction_time=0.0, cv_time=0.0),
            experiment("Slow", training_time=1.00, prediction_time=0.0, cv_time=0.0),
        ]
        analysis = analyze_execution_time(experiments)
        self.assertEqual(analysis.fastest_model, "Fast")
        self.assertAlmostEqual(analysis.fastest_time, 0.01, places=6)
        self.assertEqual(analysis.slowest_model, "Slow")
        self.assertAlmostEqual(analysis.slowest_time, 1.0, places=6)

    def test_time_split_is_reported_per_phase(self) -> None:
        analysis = analyze_execution_time(
            [experiment("A", training_time=0.2, prediction_time=0.05, cv_time=0.75)]
        )
        self.assertAlmostEqual(analysis.total_training_time, 0.2, places=6)
        self.assertAlmostEqual(analysis.total_prediction_time, 0.05, places=6)
        self.assertAlmostEqual(analysis.total_cross_validation_time, 0.75, places=6)

    def test_failed_experiments_are_not_timed(self) -> None:
        experiments = [
            experiment("A", training_time=0.2, prediction_time=0.0, cv_time=0.0),
            experiment("Broken", status=ExperimentStatus.FAILED.value, f1_score=None,
                       accuracy=None, cv_f1_mean=None, training_time=5.0,
                       prediction_time=0.0, cv_time=0.0),
        ]
        analysis = analyze_execution_time(experiments)
        self.assertEqual(analysis.timed_experiments, 1)
        self.assertAlmostEqual(analysis.total_experiment_time, 0.2, places=6)

    def test_ties_are_broken_deterministically(self) -> None:
        experiments = [
            experiment("Zebra", training_time=0.1, prediction_time=0.0, cv_time=0.0),
            experiment("Alpha", training_time=0.1, prediction_time=0.0, cv_time=0.0),
        ]
        analysis = analyze_execution_time(experiments)
        self.assertEqual(analysis.fastest_model, "Alpha")
        self.assertEqual(analysis.slowest_model, "Zebra")

    def test_empty_input_does_not_raise(self) -> None:
        analysis = analyze_execution_time([])
        self.assertEqual(analysis.timed_experiments, 0)
        self.assertEqual(analysis.total_experiment_time, 0.0)
        self.assertEqual(analysis.fastest_model, "")

    def test_single_run_disclaimer_is_recorded(self) -> None:
        analysis = analyze_execution_time([experiment()])
        self.assertIn(config.LIMITATION_SINGLE_RUN_TIMING, analysis.notes)


# --------------------------------------------------------------------------- #
# 15, 16: experiment counts and failed experiments
# --------------------------------------------------------------------------- #
class TestExperimentCounts(unittest.TestCase):
    """Models evaluated once, not hyperparameter trials."""

    def test_counts_of_a_clean_run(self) -> None:
        counts = count_experiments([experiment("A"), experiment("B"), experiment("C")])
        self.assertEqual(counts.baseline_models_attempted, 3)
        self.assertEqual(counts.successful_experiments, 3)
        self.assertEqual(counts.partial_experiments, 0)
        self.assertEqual(counts.failed_experiments, 0)
        self.assertEqual(counts.total_experiments, 3)

    def test_failed_experiment_is_counted_and_named(self) -> None:
        counts = count_experiments(
            [
                experiment("A"),
                experiment("Broken", status=ExperimentStatus.FAILED.value,
                           f1_score=None, accuracy=None),
            ]
        )
        self.assertEqual(counts.failed_experiments, 1)
        self.assertEqual(counts.successful_experiments, 1)
        self.assertEqual(counts.total_experiments, 2)
        self.assertEqual(counts.failed_model_names, ["Broken"])

    def test_partial_experiment_is_counted_separately(self) -> None:
        counts = count_experiments(
            [
                experiment("A"),
                experiment("B", status=ExperimentStatus.PARTIAL.value, cv_f1_mean=None),
            ]
        )
        self.assertEqual(counts.partial_experiments, 1)
        self.assertEqual(counts.successful_experiments, 1)
        self.assertEqual(counts.failed_experiments, 0)

    def test_hyperparameter_experiments_are_zero(self) -> None:
        counts = count_experiments([experiment("A")])
        self.assertEqual(counts.hyperparameter_experiments, 0)

    def test_note_separates_models_from_hyperparameter_trials(self) -> None:
        note = count_experiments([experiment("A")]).note
        self.assertIn("not hyperparameter trials", note)

    def test_unknown_status_is_counted_as_failed(self) -> None:
        counts = count_experiments([experiment("A"), experiment("Weird", status="mystery")])
        self.assertEqual(counts.failed_experiments, 1)
        self.assertIn("unknown status", counts.note)

    def test_empty_run(self) -> None:
        counts = count_experiments([])
        self.assertEqual(counts.total_experiments, 0)
        self.assertEqual(counts.baseline_models_attempted, 0)


# --------------------------------------------------------------------------- #
# 19, 21: report assembly and empty / unsupported state
# --------------------------------------------------------------------------- #
class TestEvaluationReport(unittest.TestCase):
    """The assembled report, including the degraded cases."""

    def _report(self, **kwargs):
        experiments = kwargs.pop("experiments", None)
        if experiments is None:
            experiments = [experiment("Logistic Regression", confusion=BINARY_MATRIX)]
        return build_evaluation_report(
            experiments=experiments,
            cross_validation_results=kwargs.pop("cv", ()),
            **kwargs,
        )

    def test_report_is_json_safe(self) -> None:
        report = self._report()
        payload = json.dumps(to_serializable(report))  # plain dumps: no default=
        restored = json.loads(payload)
        self.assertEqual(restored["best_model"]["best_baseline_model"], "Logistic Regression")

    def test_report_contains_every_required_section(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.best_model)
        self.assertTrue(report.selection_reason)
        self.assertTrue(report.performance_summary)
        self.assertEqual(len(report.model_evaluations), 1)
        self.assertIsNotNone(report.gap_analysis)
        self.assertEqual(len(report.confusion_matrix_analysis), 1)
        self.assertIsNotNone(report.execution_time_analysis)
        self.assertIsNotNone(report.experiment_count)
        self.assertTrue(report.observations)
        self.assertTrue(report.limitations)
        self.assertTrue(report.recommendations)

    def test_report_keeps_no_runtime_object(self) -> None:
        report = self._report()
        self.assertNotIn("pipeline", str(report).lower())
        self.assertNotIn("Pipeline(", str(report))

    def test_report_status_completed_when_every_experiment_worked(self) -> None:
        report = self._report()
        self.assertEqual(report.overall_status, EvaluationStatus.COMPLETED.value)

    def test_report_status_partial_when_an_experiment_failed(self) -> None:
        report = self._report(
            experiments=[
                experiment("Logistic Regression"),
                experiment("Broken", status=ExperimentStatus.FAILED.value,
                           f1_score=None, accuracy=None, cv_f1_mean=None),
            ]
        )
        self.assertEqual(report.overall_status, EvaluationStatus.PARTIAL.value)
        self.assertEqual(report.experiment_count.failed_experiments, 1)

    def test_report_status_failed_when_no_experiment_produced_metrics(self) -> None:
        report = self._report(
            experiments=[
                experiment("Broken", status=ExperimentStatus.FAILED.value,
                           f1_score=None, accuracy=None, cv_f1_mean=None)
            ]
        )
        self.assertEqual(report.overall_status, EvaluationStatus.FAILED.value)
        self.assertIsNone(report.best_model)
        self.assertIn("no baseline model is recommended", report.selection_reason.lower())

    def test_report_status_skipped_without_experiments(self) -> None:
        report = self._report(experiments=[])
        self.assertEqual(report.overall_status, EvaluationStatus.SKIPPED.value)
        self.assertIsNone(report.best_model)
        self.assertTrue(any("no experiment" in note.lower() for note in report.notes))

    def test_report_of_an_empty_run_has_no_invented_numbers(self) -> None:
        report = self._report(experiments=[])
        self.assertEqual(report.experiment_count.total_experiments, 0)
        self.assertEqual(report.execution_time_analysis.total_experiment_time, 0.0)
        self.assertEqual(report.confusion_matrix_analysis, [])

    def test_report_uses_the_supplied_comparison_rule(self) -> None:
        comparison = rank_experiments([experiment()])
        report = self._report(model_comparison=comparison)
        self.assertEqual(report.ranking_rule, comparison.ranking_rule)

    def test_report_rebuilds_the_ranking_rule_when_missing(self) -> None:
        report = self._report(model_comparison=None)
        self.assertIn("rebuilt by the Evaluation Agent", report.ranking_rule)
        self.assertTrue(any("rebuilt" in note for note in report.notes))

    def test_report_carries_the_baseline_parameters(self) -> None:
        report = self._report(parameters_by_model={"Logistic Regression": {"max_iter": 1000}})
        self.assertEqual(report.best_model.parameters, {"max_iter": 1000})

    def test_report_records_the_split_size(self) -> None:
        self.assertEqual(self._report(test_samples=24).test_samples, 24)

    def test_small_test_split_adds_a_limitation(self) -> None:
        report = self._report(test_samples=8)
        self.assertTrue(
            any("hold-out split contains 8" in item for item in report.limitations)
        )

    def test_report_is_reproducible(self) -> None:
        experiments = [experiment("A", confusion=BINARY_MATRIX), experiment("B")]
        first = to_serializable(build_evaluation_report(experiments=experiments))
        second = to_serializable(build_evaluation_report(experiments=experiments))
        # Timings are identical here because the records are hand-built.
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_table_contains_the_required_columns(self) -> None:
        report = self._report()
        table = report.table_text
        for column in ("rank", "model", "acc", "prec", "rec", "f1", "cv acc", "cv f1",
                       "f1 gap", "acc gap", "train s", "total s", "status"):
            self.assertIn(column, table)


# --------------------------------------------------------------------------- #
# 20: recommendation and summary
# --------------------------------------------------------------------------- #
class TestRecommendation(unittest.TestCase):
    """The recommendation follows the ranking and claims nothing extra."""

    def test_best_model_recommendation(self) -> None:
        ranked = rank_models(
            compare_models(
                [
                    experiment("Logistic Regression", f1_score=0.8761, accuracy=0.875),
                    experiment("Decision Tree", f1_score=0.7536, accuracy=0.75),
                ]
            )
        )
        recommendation = generate_model_recommendation(ranked)
        self.assertEqual(recommendation.best_baseline_model, "Logistic Regression")
        self.assertIn("highest hold-out f1_score (0.8761)", recommendation.selection_reason)
        self.assertIn("2 successful baseline experiment(s)", recommendation.selection_reason)

    def test_tie_is_explained_in_the_reason(self) -> None:
        ranked = rank_models(
            compare_models(
                [
                    experiment("Decision Tree", f1_score=1.0, accuracy=1.0),
                    experiment("Random Forest", f1_score=1.0, accuracy=1.0),
                ]
            )
        )
        recommendation = generate_model_recommendation(ranked)
        self.assertIn("tie on f1_score", recommendation.selection_reason)

    def test_no_recommendation_without_metrics(self) -> None:
        self.assertIsNone(generate_model_recommendation([]))

    def test_reason_stays_inside_the_available_evidence(self) -> None:
        ranked = rank_models(compare_models([experiment("A")]))
        reason = generate_model_recommendation(ranked).selection_reason
        self.assertNotIn("robust", reason.lower())
        self.assertNotIn("best possible", reason.lower())
        self.assertNotIn("real-world", reason.lower())

    def test_summary_contains_the_measured_numbers(self) -> None:
        ranked = rank_models(
            compare_models([experiment(f1_score=0.8761, accuracy=0.875, cv_f1_mean=0.894)])
        )
        summary = generate_evaluation_summary(ranked)
        self.assertIn("hold-out accuracy of 0.8750", summary)
        self.assertIn("F1-score of 0.8761", summary)
        self.assertIn("cross-validation mean F1-score", summary)
        self.assertIn("0.0179 lower than", summary)

    def test_summary_uses_the_fold_count_from_the_cv_record(self) -> None:
        ranked = rank_models(
            compare_models([experiment()], [cv_result()])
        )
        self.assertIn("over 5 stratified fold(s)", generate_evaluation_summary(ranked))

    def test_summary_omits_cross_validation_when_absent(self) -> None:
        ranked = rank_models(compare_models([experiment(cv_f1_mean=None, cv_accuracy_mean=None)]))
        summary = generate_evaluation_summary(ranked)
        self.assertNotIn("cross-validation mean", summary)

    def test_summary_without_metrics(self) -> None:
        self.assertIn("No model", generate_evaluation_summary([]))

    def test_limitations_always_include_the_baseline_caveats(self) -> None:
        ranked = rank_models(compare_models([experiment()]))
        limitations = generate_limitations(
            model_evaluations=ranked,
            gap_analysis=analyze_performance_gaps(ranked),
            confusion_analyses=[analyze_confusion_matrix(BINARY_MATRIX)],
            execution_time=analyze_execution_time([experiment()]),
            test_samples=24,
        )
        self.assertIn(config.LIMITATION_NO_TUNING, limitations)
        self.assertIn(config.LIMITATION_NO_TRAINING_METRICS, limitations)
        self.assertIn(config.LIMITATION_SINGLE_SPLIT, limitations)

    def test_limitations_report_a_missing_confusion_matrix(self) -> None:
        ranked = rank_models(compare_models([experiment()]))
        limitations = generate_limitations(
            model_evaluations=ranked,
            gap_analysis=analyze_performance_gaps(ranked),
            confusion_analyses=[analyze_confusion_matrix(None)],
            execution_time=None,
            test_samples=24,
        )
        self.assertIn(config.PER_CLASS_UNAVAILABLE_NOTE, limitations)


# --------------------------------------------------------------------------- #
# 22: scientific wording
# --------------------------------------------------------------------------- #
class TestScientificWording(unittest.TestCase):
    """The report must never overstate what a measurement shows."""

    # The patterns are specific on purpose: the disclaimer must be able to say
    # that a gap "neither proves nor disproves" overfitting, so the bare word
    # "proves" is not a violation while "proves that" is.
    FORBIDDEN = (
        r"\bis overfitting\b",
        r"\bproves that\b",
        r"\bconfirms that\b",
        r"\bshows that\b",
        r"\bdefinitely\b",
        r"\bguarantee\w*\b",
        r"\bwill work perfectly\b",
        r"\bis perfectly\b",
        r"\bperfect model\b",
        r"\bstate of the art\b",
    )

    def _report(self, **kwargs):
        return build_evaluation_report(**kwargs)

    def test_observations_make_no_forbidden_claim(self) -> None:
        report = self._report(
            experiments=[
                experiment("A", f1_score=0.90, cv_f1_mean=0.50),
                experiment("B", f1_score=0.80, cv_f1_mean=0.79),
            ]
        )
        text = " ".join(
            report.observations + report.limitations + report.recommendations
        ).lower()
        for pattern in self.FORBIDDEN:
            self.assertIsNone(
                re.search(pattern, text), f"unsupported claim matching {pattern}"
            )

    def test_substantial_gap_is_not_called_overfitting(self) -> None:
        report = self._report(
            experiments=[experiment("A", f1_score=0.95, cv_f1_mean=0.50)]
        )
        self.assertEqual(report.gap_analysis.close_agreement_models, [])
        self.assertEqual(
            report.gap_analysis.substantial_difference_models, ["A"]
        )
        self.assertIn("warrants further investigation", report.gap_analysis.interpretation)
        self.assertNotIn("overfitting", report.gap_analysis.interpretation.lower())

    def test_disclaimer_is_always_attached_to_the_gap_analysis(self) -> None:
        report = self._report(experiments=[experiment()])
        self.assertEqual(report.gap_analysis.disclaimer, config.GAP_HEURISTIC_DISCLAIMER)

    def test_close_gap_states_the_numbers_and_the_threshold(self) -> None:
        report = self._report(experiments=[experiment("A", f1_score=0.876, cv_f1_mean=0.894)])
        interpretation = report.model_evaluations[0].gap_interpretation
        self.assertIn("hold-out score is 0.8760", interpretation)
        self.assertIn("0.0180 lower than", interpretation)
        self.assertIn("close", interpretation)

    def test_error_patterns_do_not_invent_domain_consequences(self) -> None:
        report = self._report(
            experiments=[experiment("A", confusion=MULTICLASS_MATRIX)]
        )
        text = report.confusion_matrix_analysis[0].error_pattern.lower()
        for phrase in ("patient", "disease", "customer", "fraud", "churn"):
            self.assertNotIn(phrase, text)

    def test_training_metrics_are_not_claimed(self) -> None:
        report = self._report(experiments=[experiment()])
        joined = " ".join(report.observations).lower()
        self.assertNotIn("train accuracy", joined)
        self.assertNotIn("training accuracy", joined)
        self.assertNotIn("memoris", joined)


if __name__ == "__main__":
    unittest.main()
