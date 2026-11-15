"""Tests for tools/evaluation_tools.py."""

import unittest
from tools import evaluation_tools


class TestEvaluationTools(unittest.TestCase):
    """Unit tests for tools/evaluation_tools.py."""

    def test_01_gap_calculation_positive_and_negative(self) -> None:
        """Verify signed and absolute gap calculations for positive and negative differences."""
        # Positive gap (hold-out > CV)
        gap, gap_abs = evaluation_tools.calculate_gap(0.85, 0.80)
        self.assertEqual(gap, 0.05)
        self.assertEqual(gap_abs, 0.05)

        # Negative gap (hold-out < CV)
        gap_neg, gap_neg_abs = evaluation_tools.calculate_gap(0.80, 0.85)
        self.assertEqual(gap_neg, -0.05)
        self.assertEqual(gap_neg_abs, 0.05)

        # Missing values
        gap_none, gap_none_abs = evaluation_tools.calculate_gap(0.85, None)
        self.assertIsNone(gap_none)
        self.assertIsNone(gap_none_abs)

    def test_02_gap_classifications(self) -> None:
        """Verify close, moderate, substantial, and not applicable gap classifications."""
        # Close: < 0.03
        self.assertEqual(evaluation_tools.classify_performance_gap(0.015), "close agreement")
        self.assertEqual(evaluation_tools.classify_performance_gap(0.029), "close agreement")

        # Moderate: 0.03 <= gap < 0.10
        self.assertEqual(evaluation_tools.classify_performance_gap(0.03), "moderate difference")
        self.assertEqual(evaluation_tools.classify_performance_gap(0.08), "moderate difference")

        # Substantial: >= 0.10
        self.assertEqual(evaluation_tools.classify_performance_gap(0.10), "substantial difference")
        self.assertEqual(evaluation_tools.classify_performance_gap(0.25), "substantial difference")

        # Not applicable: None
        self.assertEqual(evaluation_tools.classify_performance_gap(None), "not applicable")

    def test_03_scientific_wording_no_unsupported_overfitting_claims(self) -> None:
        """Verify that gap interpretations use cautious scientific language without claims of definitive overfitting."""
        interp_close = evaluation_tools.interpret_gap("model_a", "F1-score", 0.85, 0.84, 0.01, "close agreement")
        self.assertIn("close agreement", interp_close)
        self.assertNotIn("definitely", interp_close.lower())
        self.assertNotIn("overfitting", interp_close.lower())

        interp_sub = evaluation_tools.interpret_gap("model_b", "F1-score", 0.70, 0.90, -0.20, "substantial difference")
        self.assertIn("substantial difference", interp_sub)
        self.assertIn("may indicate sensitivity to the selected split", interp_sub)
        self.assertIn("warrants further investigation", interp_sub)
        self.assertNotIn("is overfitting", interp_sub.lower())
        self.assertNotIn("definitely", interp_sub.lower())

    def test_04_confusion_matrix_binary_analysis(self) -> None:
        """Verify binary confusion matrix calculation of TP, FP, TN, FN and error pattern."""
        # 2x2 matrix: [[TN=10, FP=2], [FN=4, TP=14]]
        cm = [[10, 2], [4, 14]]
        labels = ["0", "1"]
        analysis = evaluation_tools.analyze_confusion_matrix(cm, labels, "logistic_regression")

        self.assertTrue(analysis["is_binary"])
        self.assertEqual(analysis["total_samples"], 30)
        self.assertEqual(analysis["correct_predictions"], 24)
        self.assertEqual(analysis["incorrect_predictions"], 6)
        self.assertEqual(analysis["true_negatives"], 10)
        self.assertEqual(analysis["false_positives"], 2)
        self.assertEqual(analysis["false_negatives"], 4)
        self.assertEqual(analysis["true_positives"], 14)
        self.assertEqual(analysis["accuracy"], 0.8)
        self.assertIn("False negatives (4) exceed false positives (2)", analysis["dominant_error_pattern"])

    def test_05_confusion_matrix_multiclass_analysis(self) -> None:
        """Verify multiclass confusion matrix calculation and dominant confusion pairs."""
        # 3x3 matrix
        cm = [
            [5, 2, 0],  # Class A: 5 correct, 2 confused as B
            [1, 6, 1],  # Class B: 6 correct, 1 confused as A, 1 as C
            [0, 3, 4],  # Class C: 4 correct, 3 confused as B
        ]
        labels = ["ClassA", "ClassB", "ClassC"]
        analysis = evaluation_tools.analyze_confusion_matrix(cm, labels, "random_forest")

        self.assertFalse(analysis["is_binary"])
        self.assertEqual(analysis["total_samples"], 22)
        self.assertEqual(analysis["correct_predictions"], 15)
        self.assertEqual(analysis["incorrect_predictions"], 7)
        self.assertIn("ClassC", analysis["class_error_counts"])
        self.assertEqual(analysis["class_error_counts"]["ClassC"], 3)
        self.assertTrue(len(analysis["dominant_confusion_pairs"]) > 0)
        # Top confusion pair should be ClassC -> ClassB (count 3)
        self.assertEqual(analysis["dominant_confusion_pairs"][0]["actual"], "ClassC")
        self.assertEqual(analysis["dominant_confusion_pairs"][0]["predicted"], "ClassB")
        self.assertEqual(analysis["dominant_confusion_pairs"][0]["count"], 3)

    def test_06_missing_confusion_matrix_handling(self) -> None:
        """Verify graceful handling of empty or missing confusion matrix."""
        analysis = evaluation_tools.analyze_confusion_matrix([], [], "dummy_model")
        self.assertEqual(analysis["total_samples"], 0)
        self.assertEqual(analysis["correct_predictions"], 0)
        self.assertEqual(analysis["incorrect_predictions"], 0)
        self.assertIn("No confusion matrix data available", analysis["dominant_error_pattern"])

    def test_07_execution_time_analysis(self) -> None:
        """Verify aggregation of total and average experiment times, fastest and slowest models."""
        exps = [
            {"model_name": "mod_fast", "training_time": 0.05, "total_time": 0.06, "status": "success"},
            {"model_name": "mod_slow", "training_time": 0.50, "total_time": 0.55, "status": "success"},
            {"model_name": "mod_mid", "training_time": 0.20, "total_time": 0.22, "status": "success"},
            {"model_name": "mod_fail", "training_time": 0.0, "total_time": 0.0, "status": "failed"},
        ]
        time_analysis = evaluation_tools.analyze_execution_time(exps)
        self.assertEqual(time_analysis["total_experiments"], 4)
        self.assertEqual(time_analysis["successful_experiments"], 3)
        self.assertEqual(time_analysis["failed_experiments"], 1)
        self.assertEqual(time_analysis["fastest_model"], "mod_fast")
        self.assertEqual(time_analysis["slowest_model"], "mod_slow")
        self.assertEqual(time_analysis["fastest_time"], 0.05)
        self.assertEqual(time_analysis["slowest_time"], 0.50)
        self.assertAlmostEqual(time_analysis["total_experiment_time"], 0.83, places=2)

    def test_08_model_ranking_and_tie_breaking(self) -> None:
        """Verify model ranking primarily by F1-score and tie-breaking by accuracy."""
        exps = [
            {"model_name": "model_tie_low_acc", "f1_score": 0.85, "accuracy": 0.80, "cv_f1_mean": 0.82, "status": "success"},
            {"model_name": "model_best", "f1_score": 0.90, "accuracy": 0.90, "cv_f1_mean": 0.89, "status": "success"},
            {"model_name": "model_tie_high_acc", "f1_score": 0.85, "accuracy": 0.86, "cv_f1_mean": 0.84, "status": "success"},
            {"model_name": "model_failed", "f1_score": 0.0, "accuracy": 0.0, "status": "failed", "error_message": "error"},
        ]
        ranked = evaluation_tools.compare_and_rank_models(exps)
        self.assertEqual(len(ranked), 4)
        self.assertEqual(ranked[0]["model_name"], "model_best")
        self.assertEqual(ranked[0]["rank"], 1)
        # Tie-breaker: model_tie_high_acc (0.86 acc) beats model_tie_low_acc (0.80 acc)
        self.assertEqual(ranked[1]["model_name"], "model_tie_high_acc")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[2]["model_name"], "model_tie_low_acc")
        self.assertEqual(ranked[2]["rank"], 3)
        # Failed model should have rank None and be last
        self.assertIsNone(ranked[3]["rank"])
        self.assertEqual(ranked[3]["status"], "failed")

    def test_09_generate_model_recommendation(self) -> None:
        """Verify recommendation extraction for best baseline model."""
        ranked = [
            {"model_name": "rf", "f1_score": 0.88, "accuracy": 0.89, "precision": 0.88, "recall": 0.88, "status": "success", "training_time": 0.2, "scaling_used": False},
            {"model_name": "lr", "f1_score": 0.82, "accuracy": 0.83, "precision": 0.82, "recall": 0.82, "status": "success", "training_time": 0.1, "scaling_used": True},
        ]
        best_name, reason, summary = evaluation_tools.generate_model_recommendation(ranked)
        self.assertEqual(best_name, "rf")
        self.assertIn("highest hold-out F1-score (0.8800)", reason)
        self.assertEqual(summary["best_model"], "rf")
        self.assertEqual(summary["f1_score"], 0.88)


if __name__ == "__main__":
    unittest.main()
