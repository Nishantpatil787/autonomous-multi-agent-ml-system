"""Autonomous Multi-Agent ML Workflow Benchmark Wrapper for Phase 11.

Reuses the existing Phase 7 LangGraph multi-agent workflow (run_graph)
without altering agent behavior or search logic. Measures execution time,
extracts model selection and holdout metrics, and counts total experiments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from graph.workflow import run_graph
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("AutonomousBenchmarkWrapper")


@dataclass
class AutonomousResult:
    """Structured output from the Autonomous Multi-Agent workflow execution."""

    dataset_path: str
    target_column: str
    problem_type: str
    train_samples: int
    test_samples: int
    feature_count: int
    selected_model: str
    selected_display_name: str
    selected_params: Dict[str, Any]
    holdout_accuracy: float
    holdout_precision: float
    holdout_recall: float
    holdout_f1: float
    cv_f1_mean: float
    cv_accuracy_mean: float
    total_runtime_seconds: float
    baseline_runtime_seconds: float
    optimization_runtime_seconds: float
    baseline_experiment_count: int
    optimization_trial_count: int
    experiment_count: int
    optimization_executed: bool
    mlflow_run_id: Optional[str] = None
    confusion_matrix: List[List[int]] = field(default_factory=list)
    confusion_matrix_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "feature_count": self.feature_count,
            "selected_model": self.selected_model,
            "selected_display_name": self.selected_display_name,
            "selected_params": self.selected_params,
            "holdout_accuracy": self.holdout_accuracy,
            "holdout_precision": self.holdout_precision,
            "holdout_recall": self.holdout_recall,
            "holdout_f1": self.holdout_f1,
            "cv_f1_mean": self.cv_f1_mean,
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "total_runtime_seconds": round(self.total_runtime_seconds, 4),
            "baseline_runtime_seconds": round(self.baseline_runtime_seconds, 4),
            "optimization_runtime_seconds": round(self.optimization_runtime_seconds, 4),
            "baseline_experiment_count": self.baseline_experiment_count,
            "optimization_trial_count": self.optimization_trial_count,
            "experiment_count": self.experiment_count,
            "optimization_executed": self.optimization_executed,
            "mlflow_run_id": self.mlflow_run_id,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": self.confusion_matrix_labels,
        }


class AutonomousBenchmarkWrapper:
    """Benchmark wrapper around the existing Phase 7 LangGraph workflow."""

    def __init__(
        self,
        enable_optimization: bool = True,
        enable_llm: bool = False,
        max_retries: int = Config.LANGGRAPH_MAX_RETRIES,
        n_trials: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> None:
        self.enable_optimization = enable_optimization
        self.enable_llm = enable_llm
        self.max_retries = max_retries
        self.n_trials = n_trials
        self.top_n = top_n

    def run(
        self,
        dataset_path: Union[str, Path],
        target_column: Optional[str] = None,
    ) -> AutonomousResult:
        """Run the autonomous multi-agent workflow and extract benchmark telemetry.

        Args:
            dataset_path: Path to dataset CSV.
            target_column: Optional target column override.

        Returns:
            AutonomousResult containing performance metrics, runtime, and trial counts.
        """
        start_time = time.perf_counter()
        csv_path = str(Path(dataset_path).resolve())

        # Reuse existing LangGraph orchestration
        workflow_res = run_graph(
            dataset_path=csv_path,
            target_column=target_column,
            enable_optimization=self.enable_optimization,
            max_retries=self.max_retries,
            enable_llm=self.enable_llm,
            n_trials=self.n_trials,
            top_n=self.top_n,
        )

        total_runtime = time.perf_counter() - start_time
        state = workflow_res.final_agent_state

        if not state:
            raise RuntimeError(f"Autonomous workflow returned no final state: {workflow_res.error_message}")

        # Extract selected model & metrics
        # If optimization completed and selected a model, use it; else baseline
        is_opt = bool(state.best_optimized_model and workflow_res.optimization_executed)
        chosen_model = state.best_optimized_model if is_opt else (state.best_baseline_model or "unknown")

        metrics_dict = state.best_optimized_metrics if is_opt else state.best_baseline_metrics
        holdout_acc = float(metrics_dict.get("accuracy", 0.0))
        holdout_prec = float(metrics_dict.get("precision", 0.0))
        holdout_rec = float(metrics_dict.get("recall", 0.0))
        holdout_f1 = float(metrics_dict.get("f1_score", 0.0))

        # Extract cross-validation score
        cv_f1 = 0.0
        cv_acc = 0.0
        if is_opt and state.optimization_results and chosen_model in state.optimization_results:
            opt_data = state.optimization_results[chosen_model]
            cv_f1 = float(opt_data.get("best_cv_f1", 0.0))
            cv_acc = float(opt_data.get("holdout_accuracy", holdout_acc))
        else:
            # Baseline CV from experiment results
            for exp in state.experiment_results:
                if exp.get("model_name", "").lower() == chosen_model.lower():
                    cv_f1 = float(exp.get("cv_f1_mean") or 0.0)
                    cv_acc = float(exp.get("cv_accuracy_mean") or 0.0)
                    break

        # Best hyperparameters
        params = {}
        if is_opt and state.best_hyperparameters and chosen_model in state.best_hyperparameters:
            params = state.best_hyperparameters[chosen_model]

        # Experiment counts
        base_exp_count = int(state.experiment_count or len(state.experiment_results))
        opt_trials = int(state.hyperparameter_experiments or state.successful_trials)
        total_exp = base_exp_count + opt_trials

        # Timing estimates from state telemetry
        baseline_time = 0.0
        if state.execution_time_analysis:
            baseline_time = float(state.execution_time_analysis.get("total_experiment_time", 0.0))
        opt_time = max(0.0, total_runtime - baseline_time)

        # Confusion matrix
        cm_matrix = []
        cm_labels = []
        if state.confusion_matrix_analyses and chosen_model in state.confusion_matrix_analyses:
            cm_dict = state.confusion_matrices.get(chosen_model, {})
            cm_matrix = cm_dict.get("matrix", [])
            cm_labels = cm_dict.get("labels", [])

        split_info = state.split_info or {}
        train_samples = split_info.get("train_samples", 0)
        test_samples = split_info.get("test_samples", 0)

        return AutonomousResult(
            dataset_path=csv_path,
            target_column=state.target_column or "",
            problem_type=state.problem_type or "",
            train_samples=train_samples,
            test_samples=test_samples,
            feature_count=len(state.processed_feature_names or state.selected_numerical_features + state.selected_categorical_features),
            selected_model=chosen_model,
            selected_display_name=chosen_model.replace("_", " ").title(),
            selected_params=params,
            holdout_accuracy=holdout_acc,
            holdout_precision=holdout_prec,
            holdout_recall=holdout_rec,
            holdout_f1=holdout_f1,
            cv_f1_mean=cv_f1,
            cv_accuracy_mean=cv_acc,
            total_runtime_seconds=total_runtime,
            baseline_runtime_seconds=baseline_time,
            optimization_runtime_seconds=opt_time,
            baseline_experiment_count=base_exp_count,
            optimization_trial_count=opt_trials,
            experiment_count=total_exp,
            optimization_executed=workflow_res.optimization_executed,
            mlflow_run_id=workflow_res.mlflow_run_id,
            confusion_matrix=cm_matrix,
            confusion_matrix_labels=cm_labels,
        )
