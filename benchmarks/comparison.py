"""Comparison analysis, reporting, artifact generation, and plotting for Phase 11.

Calculates mathematical differences between Conventional and Autonomous ML workflows,
serializes results to JSON and CSV, and generates publication-ready comparison charts.
Optionally logs comparison metrics to MLflow if enabled.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure matplotlib uses non-interactive backend and writable cache
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.autonomous_workflow import AutonomousBenchmarkWrapper, AutonomousResult
from benchmarks.benchmark_config import (
    BENCHMARK_DATASETS,
    BenchmarkConfig,
    BenchmarkDatasetSpec,
    get_dataset_hash,
)
from benchmarks.conventional_workflow import ConventionalMLWorkflow, ConventionalResult
from tools.mlflow_tools import MLflowTracker
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("BenchmarkComparison")


@dataclass
class DatasetComparisonResult:
    """Comparison analysis for a single benchmark dataset."""

    dataset_name: str
    dataset_path: str
    dataset_hash: str
    task_type: str
    conventional: ConventionalResult
    autonomous: AutonomousResult
    f1_difference: float
    accuracy_difference: float
    precision_difference: float
    recall_difference: float
    cv_f1_difference: float
    runtime_difference: float
    experiment_difference: int
    effort_automation_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison to a JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "dataset_hash": self.dataset_hash,
            "task_type": self.task_type,
            "conventional": self.conventional.to_dict(),
            "autonomous": self.autonomous.to_dict(),
            "comparison": {
                "f1_difference": round(self.f1_difference, 4),
                "accuracy_difference": round(self.accuracy_difference, 4),
                "precision_difference": round(self.precision_difference, 4),
                "recall_difference": round(self.recall_difference, 4),
                "cv_f1_difference": round(self.cv_f1_difference, 4),
                "runtime_difference": round(self.runtime_difference, 4),
                "experiment_difference": self.experiment_difference,
                "effort_automation_ratio": round(self.effort_automation_ratio, 4),
            },
        }


def compute_dataset_comparison(
    conventional_res: ConventionalResult,
    autonomous_res: AutonomousResult,
    dataset_spec: BenchmarkDatasetSpec,
) -> DatasetComparisonResult:
    """Calculate exact mathematical differences between conventional and autonomous results."""
    d_f1 = autonomous_res.holdout_f1 - conventional_res.holdout_f1
    d_acc = autonomous_res.holdout_accuracy - conventional_res.holdout_accuracy
    d_prec = autonomous_res.holdout_precision - conventional_res.holdout_precision
    d_rec = autonomous_res.holdout_recall - conventional_res.holdout_recall
    d_cv_f1 = autonomous_res.cv_f1_mean - conventional_res.cv_f1_mean
    d_runtime = autonomous_res.total_runtime_seconds - conventional_res.total_runtime_seconds
    d_exp = autonomous_res.experiment_count - conventional_res.experiment_count

    # Effort automation proxy: ratio of autonomous trials to conventional manual configurations
    effort_ratio = (
        autonomous_res.experiment_count / conventional_res.experiment_count
        if conventional_res.experiment_count > 0
        else 1.0
    )

    d_hash = get_dataset_hash(dataset_spec.file_path)

    return DatasetComparisonResult(
        dataset_name=dataset_spec.name,
        dataset_path=str(dataset_spec.file_path),
        dataset_hash=d_hash,
        task_type=dataset_spec.problem_type,
        conventional=conventional_res,
        autonomous=autonomous_res,
        f1_difference=d_f1,
        accuracy_difference=d_acc,
        precision_difference=d_prec,
        recall_difference=d_rec,
        cv_f1_difference=d_cv_f1,
        runtime_difference=d_runtime,
        experiment_difference=d_exp,
        effort_automation_ratio=effort_ratio,
    )


class BenchmarkRunner:
    """Orchestrates comparative benchmarking between Conventional and Autonomous workflows."""

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()
        self.conventional_workflow = ConventionalMLWorkflow(
            random_state=self.config.random_state,
            test_size=self.config.test_size,
            cv_folds=self.config.cv_folds,
        )
        self.autonomous_wrapper = AutonomousBenchmarkWrapper(
            enable_optimization=True,
            enable_llm=False,
        )

    def run_benchmark_for_dataset(
        self, dataset_spec: BenchmarkDatasetSpec
    ) -> DatasetComparisonResult:
        """Execute both workflows on a single dataset and return the comparison."""
        logger.info("Executing Conventional Workflow on: %s", dataset_spec.name)
        conv_res = self.conventional_workflow.run(
            dataset_path=dataset_spec.file_path,
            target_column=dataset_spec.target_column,
        )

        logger.info("Executing Autonomous Multi-Agent Workflow on: %s", dataset_spec.name)
        auto_res = self.autonomous_wrapper.run(
            dataset_path=dataset_spec.file_path,
            target_column=dataset_spec.target_column,
        )

        return compute_dataset_comparison(conv_res, auto_res, dataset_spec)

    def run_all(self) -> Dict[str, DatasetComparisonResult]:
        """Execute comparative benchmarks across all standard project datasets."""
        results: Dict[str, DatasetComparisonResult] = {}
        for key, spec in BENCHMARK_DATASETS.items():
            logger.info("Starting Phase 11 benchmark for: %s (%s)", spec.name, key)
            results[key] = self.run_benchmark_for_dataset(spec)

        # Generate output artifacts
        self.save_json_report(results)
        self.save_csv_report(results)
        self.generate_plots(results)

        # Log to MLflow if enabled
        if self.config.enable_mlflow:
            self._log_mlflow_benchmark(results)

        return results

    def save_json_report(self, results: Dict[str, DatasetComparisonResult]) -> Path:
        """Save benchmark comparison metrics to JSON format."""
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": "11",
            "benchmark_title": "Conventional vs Autonomous Multi-Agent ML Workflow Comparison",
            "datasets": {k: v.to_dict() for k, v in results.items()},
        }
        with open(self.config.json_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved benchmark JSON to: %s", self.config.json_output_path)
        return self.config.json_output_path

    def save_csv_report(self, results: Dict[str, DatasetComparisonResult]) -> Path:
        """Save tabular comparison summary to CSV format."""
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "Dataset",
            "Task Type",
            "Workflow",
            "Selected Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "CV F1-Score",
            "Runtime (s)",
            "Experiments",
        ]

        rows = []
        for key, comp in results.items():
            # Conventional row
            rows.append({
                "Dataset": comp.dataset_name,
                "Task Type": comp.task_type,
                "Workflow": "Conventional",
                "Selected Model": comp.conventional.selected_display_name,
                "Accuracy": f"{comp.conventional.holdout_accuracy:.4f}",
                "Precision": f"{comp.conventional.holdout_precision:.4f}",
                "Recall": f"{comp.conventional.holdout_recall:.4f}",
                "F1-Score": f"{comp.conventional.holdout_f1:.4f}",
                "CV F1-Score": f"{comp.conventional.cv_f1_mean:.4f}",
                "Runtime (s)": f"{comp.conventional.total_runtime_seconds:.2f}",
                "Experiments": comp.conventional.experiment_count,
            })
            # Autonomous row
            rows.append({
                "Dataset": comp.dataset_name,
                "Task Type": comp.task_type,
                "Workflow": "Autonomous Multi-Agent",
                "Selected Model": comp.autonomous.selected_display_name,
                "Accuracy": f"{comp.autonomous.holdout_accuracy:.4f}",
                "Precision": f"{comp.autonomous.holdout_precision:.4f}",
                "Recall": f"{comp.autonomous.holdout_recall:.4f}",
                "F1-Score": f"{comp.autonomous.holdout_f1:.4f}",
                "CV F1-Score": f"{comp.autonomous.cv_f1_mean:.4f}",
                "Runtime (s)": f"{comp.autonomous.total_runtime_seconds:.2f}",
                "Experiments": comp.autonomous.experiment_count,
            })

        with open(self.config.csv_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Saved benchmark CSV to: %s", self.config.csv_output_path)
        return self.config.csv_output_path

    def generate_plots(self, results: Dict[str, DatasetComparisonResult]) -> None:
        """Generate publication-ready comparison bar charts for F1, runtime, and experiment counts."""
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        dataset_labels = [r.dataset_name.replace(".csv", "") for r in results.values()]
        x = np.arange(len(dataset_labels))
        width = 0.35

        # 1. F1-Score Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        conv_f1 = [r.conventional.holdout_f1 for r in results.values()]
        auto_f1 = [r.autonomous.holdout_f1 for r in results.values()]

        rects1 = ax.bar(x - width / 2, conv_f1, width, label="Conventional", color="#4C72B0")
        rects2 = ax.bar(x + width / 2, auto_f1, width, label="Autonomous Multi-Agent", color="#55A868")

        ax.set_ylabel("Holdout F1-Score (Weighted)")
        ax.set_title("Holdout F1-Score: Conventional vs Autonomous ML Workflow")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(self.config.chart_f1_path, dpi=200)
        plt.close(fig)

        # 2. Runtime Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        conv_rt = [r.conventional.total_runtime_seconds for r in results.values()]
        auto_rt = [r.autonomous.total_runtime_seconds for r in results.values()]

        rects1 = ax.bar(x - width / 2, conv_rt, width, label="Conventional", color="#4C72B0")
        rects2 = ax.bar(x + width / 2, auto_rt, width, label="Autonomous Multi-Agent", color="#C44E52")

        ax.set_ylabel("Execution Time (Seconds)")
        ax.set_title("Total Runtime: Conventional vs Autonomous ML Workflow")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(self.config.chart_runtime_path, dpi=200)
        plt.close(fig)

        # 3. Experiment Count Comparison Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        conv_exp = [r.conventional.experiment_count for r in results.values()]
        auto_exp = [r.autonomous.experiment_count for r in results.values()]

        rects1 = ax.bar(x - width / 2, conv_exp, width, label="Conventional", color="#4C72B0")
        rects2 = ax.bar(x + width / 2, auto_exp, width, label="Autonomous Multi-Agent", color="#8172B2")

        ax.set_ylabel("Number of Evaluated Configurations / Experiments")
        ax.set_title("Experiment Count: Conventional vs Autonomous ML Workflow")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        fig.savefig(self.config.chart_experiments_path, dpi=200)
        plt.close(fig)

        logger.info("Saved comparison plots to %s", self.config.reports_dir)

    def _log_mlflow_benchmark(self, results: Dict[str, DatasetComparisonResult]) -> None:
        """Log benchmark comparison metrics to local MLflow store."""
        try:
            tracker = MLflowTracker(
                enabled=True,
                tracking_uri=Config.MLFLOW_TRACKING_URI,
                experiment_name=Config.MLFLOW_EXPERIMENT_NAME,
            )
            run_id = tracker.start_workflow_run(
                dataset_path="benchmark_comparison",
                enable_optimization=True,
                enable_llm=False,
            )
            if not run_id:
                return

            tracker.log_reproducibility()

            for key, comp in results.items():
                prefix = f"bench_{key}_"
                metrics = {
                    f"{prefix}conv_f1": comp.conventional.holdout_f1,
                    f"{prefix}conv_acc": comp.conventional.holdout_accuracy,
                    f"{prefix}conv_runtime": comp.conventional.total_runtime_seconds,
                    f"{prefix}conv_experiments": comp.conventional.experiment_count,
                    f"{prefix}auto_f1": comp.autonomous.holdout_f1,
                    f"{prefix}auto_acc": comp.autonomous.holdout_accuracy,
                    f"{prefix}auto_runtime": comp.autonomous.total_runtime_seconds,
                    f"{prefix}auto_experiments": comp.autonomous.experiment_count,
                    f"{prefix}diff_f1": comp.f1_difference,
                    f"{prefix}diff_acc": comp.accuracy_difference,
                    f"{prefix}diff_runtime": comp.runtime_difference,
                }
                for m_name, m_val in metrics.items():
                    try:
                        tracker._client.log_metric(run_id, m_name, float(m_val))
                    except Exception:  # noqa: BLE001
                        pass

                try:
                    tracker._client.set_tag(run_id, f"{prefix}conv_model", comp.conventional.selected_model)
                    tracker._client.set_tag(run_id, f"{prefix}auto_model", comp.autonomous.selected_model)
                    tracker._client.set_tag(run_id, f"{prefix}sha256", comp.dataset_hash)
                except Exception:  # noqa: BLE001
                    pass

            tracker.end_run("FINISHED")
            logger.info("Logged Phase 11 benchmark comparison to MLflow (run_id=%s)", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] Failed to log benchmark run (ignored): %s", exc)
