"""Reusable model evaluation, diagnostic gap analysis, and report generation tools.

Consumes Phase 4 experiment results to compute hold-out vs CV gaps,
analyze confusion matrices, profile execution times, and synthesize
objective academic findings without retraining models.
"""

from typing import Any, Dict, List, Optional, Tuple
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("EvaluationTools")


def calculate_gap(
    holdout_val: Optional[float],
    cv_val: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute signed gap (holdout - cv) and absolute gap.

    Returns:
        Tuple of (signed_gap, absolute_gap) or (None, None) if either is missing.
    """
    if holdout_val is None or cv_val is None:
        return None, None

    gap = round(float(holdout_val) - float(cv_val), 4)
    gap_abs = round(abs(gap), 4)
    return gap, gap_abs


def classify_performance_gap(
    gap_abs: Optional[float],
    close_threshold: Optional[float] = None,
    moderate_threshold: Optional[float] = None,
) -> str:
    """Classify the magnitude of gap between hold-out and cross-validation using reporting heuristics.

    Reporting Heuristic Thresholds:
      - ABS(gap) < close_threshold: "close agreement"
      - close_threshold <= ABS(gap) < moderate_threshold: "moderate difference"
      - ABS(gap) >= moderate_threshold: "substantial difference"
    """
    if gap_abs is None:
        return "not applicable"

    close_t = close_threshold if close_threshold is not None else Config.GAP_THRESHOLD_CLOSE
    mod_t = moderate_threshold if moderate_threshold is not None else Config.GAP_THRESHOLD_MODERATE

    if gap_abs < close_t:
        return "close agreement"
    elif gap_abs < mod_t:
        return "moderate difference"
    else:
        return "substantial difference"


def interpret_gap(
    model_name: str,
    metric_name: str,
    holdout_val: float,
    cv_val: Optional[float],
    gap: Optional[float],
    classification: str,
) -> str:
    """Formulate a scientifically guarded interpretation of the metric discrepancy."""
    if cv_val is None or gap is None:
        return f"{model_name}: Cross-validation {metric_name} was not available for comparison."

    if classification == "close agreement":
        return (
            f"{model_name}: The hold-out {metric_name} ({holdout_val:.4f}) is in close agreement with "
            f"cross-validation {metric_name} ({cv_val:.4f}), differing by {gap:+.4f}."
        )
    elif classification == "moderate difference":
        return (
            f"{model_name}: A moderate difference of {gap:+.4f} was observed between hold-out "
            f"({holdout_val:.4f}) and cross-validation ({cv_val:.4f}) {metric_name}. "
            "This suggests mild sensitivity to the specific partition split."
        )
    else:
        return (
            f"{model_name}: A substantial difference of {gap:+.4f} was observed between hold-out "
            f"({holdout_val:.4f}) and cross-validation ({cv_val:.4f}) {metric_name}. "
            "This difference may indicate sensitivity to the selected split and warrants further investigation."
        )


def analyze_confusion_matrix(
    cm_data: List[List[int]],
    labels: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Perform diagnostic analysis on a confusion matrix."""
    if not cm_data or not isinstance(cm_data, list) or len(cm_data) == 0:
        return {
            "model_name": model_name,
            "total_samples": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0.0,
            "is_binary": False,
            "dominant_error_pattern": "No confusion matrix data available.",
            "dominant_confusion_pairs": [],
            "class_error_counts": {},
        }

    n_classes = len(cm_data)
    total_samples = int(sum(sum(row) for row in cm_data))
    correct_predictions = int(sum(cm_data[i][i] for i in range(n_classes) if i < len(cm_data[i])))
    incorrect_predictions = total_samples - correct_predictions
    accuracy = round(correct_predictions / total_samples, 4) if total_samples > 0 else 0.0

    is_binary = n_classes == 2

    # Binary classification breakdown
    tp: Optional[int] = None
    fp: Optional[int] = None
    tn: Optional[int] = None
    fn: Optional[int] = None
    pattern_desc = ""

    dominant_pairs: List[Dict[str, Any]] = []
    class_errors: Dict[str, int] = {}

    if is_binary and len(cm_data[0]) >= 2 and len(cm_data[1]) >= 2:
        tn = int(cm_data[0][0])
        fp = int(cm_data[0][1])
        fn = int(cm_data[1][0])
        tp = int(cm_data[1][1])

        if fn > fp:
            pattern_desc = (
                f"False negatives ({fn}) exceed false positives ({fp}), indicating the model "
                "misses more positive samples than it incorrectly labels as positive."
            )
        elif fp > fn:
            pattern_desc = (
                f"False positives ({fp}) exceed false negatives ({fn}), indicating the model "
                "predicts the positive class more frequently than actual positives appear."
            )
        elif fp == fn and fp > 0:
            pattern_desc = (
                f"Errors are evenly distributed between false positives ({fp}) and false negatives ({fn})."
            )
        else:
            pattern_desc = "Zero classification errors observed on the evaluation split."

    else:
        # Multiclass breakdown
        for i in range(n_classes):
            lbl = labels[i] if i < len(labels) else str(i)
            row_sum = sum(cm_data[i])
            diag_val = cm_data[i][i] if i < len(cm_data[i]) else 0
            class_errors[lbl] = int(row_sum - diag_val)

            for j in range(len(cm_data[i])):
                if i != j and cm_data[i][j] > 0:
                    pred_lbl = labels[j] if j < len(labels) else str(j)
                    dominant_pairs.append({
                        "actual": lbl,
                        "predicted": pred_lbl,
                        "count": int(cm_data[i][j]),
                    })

        dominant_pairs.sort(key=lambda p: p["count"], reverse=True)
        if dominant_pairs:
            top_pair = dominant_pairs[0]
            worst_class = max(class_errors.items(), key=lambda item: item[1])[0]
            pattern_desc = (
                f"Class '{worst_class}' exhibited the highest error count ({class_errors[worst_class]}). "
                f"The most frequent confusion was actual '{top_pair['actual']}' predicted as '{top_pair['predicted']}' "
                f"({top_pair['count']} instance(s))."
            )
        else:
            pattern_desc = "Zero classification errors observed across all classes."

    return {
        "model_name": model_name,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy": accuracy,
        "is_binary": is_binary,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "dominant_error_pattern": pattern_desc,
        "dominant_confusion_pairs": dominant_pairs,
        "class_error_counts": class_errors,
    }


def analyze_execution_time(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate training and total execution durations across experiments."""
    total_experiments = len(experiments)
    successful = [exp for exp in experiments if exp.get("status") == "success"]
    failed = [exp for exp in experiments if exp.get("status") != "success"]

    total_time = round(sum(exp.get("total_time", 0.0) for exp in experiments), 4)
    avg_time = round(total_time / total_experiments, 4) if total_experiments > 0 else 0.0

    fastest_model: Optional[str] = None
    fastest_time = 0.0
    slowest_model: Optional[str] = None
    slowest_time = 0.0

    if successful:
        sorted_by_time = sorted(successful, key=lambda x: x.get("training_time", 0.0))
        fastest_exp = sorted_by_time[0]
        slowest_exp = sorted_by_time[-1]

        fastest_model = fastest_exp["model_name"]
        fastest_time = round(fastest_exp.get("training_time", 0.0), 4)
        slowest_model = slowest_exp["model_name"]
        slowest_time = round(slowest_exp.get("training_time", 0.0), 4)

    return {
        "total_experiment_time": total_time,
        "average_experiment_time": avg_time,
        "fastest_model": fastest_model,
        "fastest_time": fastest_time,
        "slowest_model": slowest_model,
        "slowest_time": slowest_time,
        "baseline_models_attempted": total_experiments,
        "successful_experiments": len(successful),
        "failed_experiments": len(failed),
        "total_experiments": total_experiments,
    }


def compare_and_rank_models(
    experiments: List[Dict[str, Any]],
    close_threshold: Optional[float] = None,
    moderate_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build a comprehensive model comparison table with hold-out metrics, CV metrics, and gap heuristics."""
    successful = [exp for exp in experiments if exp.get("status") == "success"]
    failed = [exp for exp in experiments if exp.get("status") != "success"]

    # Rank primarily by F1-score, secondarily by accuracy, tertiarily by CV F1
    def sort_key(e: Dict[str, Any]) -> Tuple[float, float, float]:
        f1 = e.get("f1_score", 0.0) or 0.0
        acc = e.get("accuracy", 0.0) or 0.0
        cv_f1 = e.get("cv_f1_mean", 0.0) or 0.0
        return (f1, acc, cv_f1)

    successful.sort(key=sort_key, reverse=True)

    ranked_list: List[Dict[str, Any]] = []
    for rank, exp in enumerate(successful, start=1):
        holdout_f1 = exp.get("f1_score", 0.0)
        cv_f1 = exp.get("cv_f1_mean")
        f1_gap, f1_gap_abs = calculate_gap(holdout_f1, cv_f1)

        holdout_acc = exp.get("accuracy", 0.0)
        cv_acc = exp.get("cv_accuracy_mean")
        acc_gap, acc_gap_abs = calculate_gap(holdout_acc, cv_acc)

        ranked_list.append({
            "rank": rank,
            "model_name": exp["model_name"],
            "accuracy": holdout_acc,
            "precision": exp.get("precision", 0.0),
            "recall": exp.get("recall", 0.0),
            "f1_score": holdout_f1,
            "cv_f1_mean": cv_f1,
            "cv_accuracy_mean": cv_acc,
            "f1_gap": f1_gap,
            "f1_gap_classification": classify_performance_gap(f1_gap_abs, close_threshold, moderate_threshold),
            "accuracy_gap": acc_gap,
            "accuracy_gap_classification": classify_performance_gap(acc_gap_abs, close_threshold, moderate_threshold),
            "training_time": exp.get("training_time", 0.0),
            "total_time": exp.get("total_time", 0.0),
            "scaling_used": exp.get("preprocessing_scaling", False),
            "status": "success",
        })

    for exp in failed:
        ranked_list.append({
            "rank": None,
            "model_name": exp["model_name"],
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "cv_f1_mean": None,
            "cv_accuracy_mean": None,
            "f1_gap": None,
            "f1_gap_classification": "not applicable",
            "accuracy_gap": None,
            "accuracy_gap_classification": "not applicable",
            "training_time": exp.get("training_time", 0.0),
            "total_time": exp.get("total_time", 0.0),
            "scaling_used": exp.get("preprocessing_scaling", False),
            "status": "failed",
            "error_message": exp.get("error_message"),
        })

    return ranked_list


def generate_model_recommendation(
    ranked_evaluations: List[Dict[str, Any]],
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Determine best baseline model and construct objective selection justification."""
    successful = [m for m in ranked_evaluations if m.get("status") == "success"]
    if not successful:
        return (
            None,
            "No candidate models succeeded during training; baseline recommendation cannot be made.",
            {},
        )

    best = successful[0]
    best_name = best["model_name"]
    f1 = best["f1_score"]
    acc = best["accuracy"]

    reason = (
        f"Selected as top baseline model based on highest hold-out F1-score ({f1:.4f}) "
        f"and accuracy ({acc:.4f}) among all evaluated candidates."
    )

    summary = {
        "best_model": best_name,
        "f1_score": f1,
        "accuracy": acc,
        "precision": best["precision"],
        "recall": best["recall"],
        "cv_f1_mean": best.get("cv_f1_mean"),
        "cv_accuracy_mean": best.get("cv_accuracy_mean"),
        "training_time": best["training_time"],
        "scaling_used": best["scaling_used"],
    }
    return best_name, reason, summary


def generate_evaluation_report(
    ranked_models: List[Dict[str, Any]],
    gap_analyses: List[Dict[str, Any]],
    cm_analyses: Dict[str, Dict[str, Any]],
    time_analysis: Dict[str, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """Compile structured, academic evaluation report dictionary."""
    best_name, selection_reason, perf_summary = generate_model_recommendation(ranked_models)

    observations: List[str] = []
    if best_name:
        observations.append(
            f"Model '{best_name}' attained the strongest baseline performance with F1-score of "
            f"{perf_summary.get('f1_score', 0.0):.4f} on the held-out test split."
        )

    # Observations on gaps
    for gap in gap_analyses:
        model = gap["model_name"]
        f1_cls = gap["f1_classification"]
        if f1_cls == "close agreement":
            observations.append(
                f"Model '{model}' showed close agreement between cross-validation and hold-out F1-scores."
            )
        elif f1_cls == "substantial difference":
            observations.append(
                f"Model '{model}' showed a substantial difference between cross-validation and hold-out F1-scores "
                f"({gap.get('f1_gap'):+.4f}), suggesting potential sensitivity to partition variance."
            )

    limitations = [
        "Evaluations reflect default baseline hyperparameters; models have not yet undergone hyperparameter optimization.",
        "Candidate pool is currently restricted to 5 controlled scikit-learn classification algorithms.",
        "Hold-out evaluations are based on a single stratified 80/20 train/test partition.",
        "Per-class precision and recall curves are not calculated in this phase; weighted averages are reported.",
    ]

    recommendations = [
        f"Proceed to Phase 6 (Hyperparameter Optimization) targeting the top candidate models (e.g. '{best_name}').",
        "Explore Optuna Bayesian search over bounded hyperparameter spaces to refine model performance.",
        "Investigate feature importance analysis on tree-based models to understand key drivers.",
    ]

    return {
        "overall_status": "completed" if best_name else "failed",
        "best_model": best_name or "None",
        "selection_reason": selection_reason,
        "performance_summary": perf_summary,
        "model_evaluations": ranked_models,
        "gap_analysis": gap_analyses,
        "confusion_matrix_analyses": cm_analyses,
        "execution_time_analysis": time_analysis,
        "experiment_count": {
            "baseline_models_attempted": time_analysis["baseline_models_attempted"],
            "successful_experiments": time_analysis["successful_experiments"],
            "failed_experiments": time_analysis["failed_experiments"],
            "total_experiments": time_analysis["total_experiments"],
        },
        "observations": observations,
        "limitations": limitations,
        "recommendations": recommendations,
    }
