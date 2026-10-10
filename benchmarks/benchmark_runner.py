"""Runner that executes and compares the two Phase 11 workflows.

The runner performs the controlled experiment:

1. validates the committed fixtures (SHA-256),
2. runs the conventional workflow (manual grid) on a dataset,
3. runs the autonomous workflow (the existing ``run_workflow`` with LLM off,
   Optuna on, persistence redirected to a scratch directory and tracking off)
   on the *same* dataset — the two sides therefore share dataset bytes, the
   analysis function, the preprocessing plan builder, the split function and
   the metric definitions,
4. compares the winners under the project's own ranking rule,
5. records the result in the dedicated benchmark MLflow store (best-effort,
   never failing the benchmark),
6. writes ``reports/phase11_comparison.json``, ``.csv`` and three PNG charts.

Timing methodology (identical on both sides): ``time.perf_counter()`` is read
once before and once after the machine-learning work of each workflow. Report
writing, chart rendering and MLflow recording happen *after* the stopwatch is
stopped and never count toward either runtime.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from benchmarks import benchmark_config
from benchmarks.conventional_workflow import ConventionalWorkflowResult
from benchmarks.conventional_workflow import _sha256 as _sha256_file
from benchmarks.conventional_workflow import run_conventional_workflow
from graph.workflow import run_workflow
from tools.model_tools import get_supported_models
from utils import config
from utils.state import FinalModelSource
from utils.state import FinalModelSelection
from utils.state import OptimizationCounts
from utils.state import PersistenceReport

# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowBenchmarkResult:
    """The measured outcome of one workflow on one dataset."""

    workflow: str  # "conventional" | "autonomous"
    dataset_key: str
    dataset_sha: str = ""
    selected_model: str = ""
    selection_source: str = ""  # "grid" | "baseline" | "optimized"
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    experiments: int = 0
    baseline_experiments: int = 0
    grid_experiments: int = 0
    trials: int = 0
    target_column: str = ""
    problem_type: str = ""
    feature_count: int = 0
    train_samples: int = 0
    test_samples: int = 0
    split_stratified: bool = True
    split_reason: str = ""
    stage_times: Dict[str, float] = field(default_factory=dict)
    node_durations: Dict[str, float] = field(default_factory=dict)
    status: str = "success"
    notes: List[str] = field(default_factory=list)

    def summary_row(self) -> Dict[str, Any]:
        """A flat, JSON/CSV friendly row for this workflow result."""
        return {
            "dataset": self.dataset_key,
            "workflow": self.workflow,
            "model": self.selected_model,
            "selection_source": self.selection_source,
            "f1_score": _nan_to_none(self.f1_score),
            "accuracy": _nan_to_none(self.accuracy),
            "precision": _nan_to_none(self.precision),
            "recall": _nan_to_none(self.recall),
            "cv_f1_mean": _nan_to_none(self.cv_f1_mean),
            "runtime_seconds": round(float(self.runtime_seconds), 6),
            "experiments": int(self.experiments),
            "baseline_experiments": int(self.baseline_experiments),
            "grid_experiments": int(self.grid_experiments),
            "trials": int(self.trials),
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "feature_count": int(self.feature_count),
            "train_samples": int(self.train_samples),
            "test_samples": int(self.test_samples),
            "status": self.status,
        }


@dataclass
class DatasetComparison:
    """The comparison of both workflows on one dataset."""

    dataset_key: str
    dataset_sha256: str
    conventional: WorkflowBenchmarkResult
    autonomous: WorkflowBenchmarkResult
    delta_f1: Optional[float] = None
    delta_accuracy: Optional[float] = None
    delta_cv_f1: Optional[float] = None
    delta_runtime: float = 0.0
    delta_experiments: int = 0
    f1_relative_percent: Optional[float] = None
    verdict: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSuiteResult:
    """The aggregate of one benchmark run (one or two datasets)."""

    datasets: List[DatasetComparison] = field(default_factory=list)
    generated_at: str = ""
    timing_method: str = "time.perf_counter() around the ML work of each workflow"
    experiment_definition: str = benchmark_config.EXPERIMENT_DEFINITION
    split: Dict[str, Any] = field(
        default_factory=lambda: {
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "stratify_if_possible": True,
        }
    )
    conventional_grid_size: int = benchmark_config.conventional_grid_size()
    autonomous_n_trials: int = benchmark_config.AUTONOMOUS_N_TRIALS
    candidate_families: List[str] = field(default_factory=list)
    mlflow: Dict[str, Any] = field(default_factory=dict)
    files_written: List[str] = field(default_factory=list)

    def comparison_for(self, dataset_key: str) -> Optional[DatasetComparison]:
        return next((d for d in self.datasets if d.dataset_key == dataset_key), None)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _nan_to_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN guard for JSON serialisability
        return None
    return round(number, 6)


def _json_safe(value: Any) -> Any:
    """Recursively turn a result tree into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    return str(value)


# --------------------------------------------------------------------------- #
# Autonomous side: reuse the existing workflow, keep every side effect out
# --------------------------------------------------------------------------- #
def _run_autonomous_side(
    dataset_key: str,
    dataset_path: str,
    *,
    n_trials: int,
    scratch_dir: Optional[str] = None,
) -> WorkflowBenchmarkResult:
    """Run the real workflow (LLM off, Optuna on) and extract its outcome.

    Persistence is pointed at a scratch directory and tracking is disabled, so
    the run never writes a model into ``models/`` and never touches the guarded
    canonical MLflow store — yet `FinalModelSelection` is still computed
    authoritatively because the persistence node runs.
    """
    started = time.perf_counter()
    result = run_workflow(
        dataset_path=dataset_path,
        enable_optimization=True,
        cross_validation=True,
        n_trials=n_trials,
        enable_llm=False,
        enable_persistence=True,
        persistence_output_dir=scratch_dir,
        enable_tracking=False,
    )
    runtime_seconds = round(time.perf_counter() - started, 6)

    state = result.final_agent_state
    record = WorkflowBenchmarkResult(
        workflow="autonomous",
        dataset_key=dataset_key,
        status="success",
        runtime_seconds=runtime_seconds,
        node_durations={
            item.node: round(float(item.duration), 6) for item in result.node_records
        },
    )
    record.notes.append(
        "autonomous side = existing run_workflow (LLM off, Optuna on, "
        "persistence to scratch, tracking off); FinalModelSelection computed by "
        "the persistence node"
    )
    if result.workflow_status.value not in ("completed",):
        record.status = result.workflow_status.value
        record.notes.append(f"workflow_status={result.workflow_status.value}")

    eval_report = state.evaluation_report
    opt_report = state.optimization_report
    persist_report: Optional[PersistenceReport] = state.persistence_report
    selection: Optional[FinalModelSelection] = (
        persist_report.selection if persist_report is not None else None
    )
    if persist_report is None or selection is None:
        raise RuntimeError(
            "The autonomous workflow produced no FinalModelSelection; cannot "
            "build a fair comparison."
        )

    # --- dataset / target facts -------------------------------------------- #
    record.target_column = (
        persist_report.target_column or state.target_column or ""
    )
    record.problem_type = (
        persist_report.task_type
        or (eval_report.problem_type if eval_report else "")
        or (state.problem_type or "")
    )
    record.feature_count = int(persist_report.feature_count or 0)
    split_facts = _split_facts_from_state(state)
    record.train_samples = split_facts["train_samples"]
    record.test_samples = split_facts["test_samples"]
    record.split_stratified = split_facts["stratified"]
    record.split_reason = split_facts["reason"]

    # --- the winning model and its metrics --------------------------------- #
    record.selected_model = selection.model_name
    source = selection.selection_source
    if not isinstance(source, str):
        source = source.value
    record.selection_source = source
    if source == FinalModelSource.OPTIMIZED.value:
        record.f1_score = _nan_to_none(persist_report.selected_f1_score)
        record.accuracy = _nan_to_none(persist_report.selected_accuracy)
        record.cv_f1_mean = _nan_to_none(selection.optimized_cv_f1_mean)
        record.parameters = dict(selection.best_params or {})
    else:
        record.f1_score = _nan_to_none(persist_report.selected_f1_score)
        record.accuracy = _nan_to_none(persist_report.selected_accuracy)
        record.cv_f1_mean = _nan_to_none(selection.baseline_cv_f1_mean)
        record.parameters = dict(selection.best_params or {})
    record.notes.append(f"FinalModelSelection.source={record.selection_source}")

    # precision/recall are Phase-5-only facts: present for a baseline winner,
    # absent (None) for an optimised winner, which Phase 6 does not record.
    if record.selection_source == FinalModelSource.BASELINE.value and eval_report:
        for evaluation in eval_report.model_evaluations:
            if evaluation.model_name == record.selected_model:
                record.precision = _nan_to_none(evaluation.precision)
                record.recall = _nan_to_none(evaluation.recall)
                break

    # --- experiment counts -------------------------------------------------- #
    # One experiment = one unique model configuration evaluated once; trial
    # counts come straight from the optimisation report so the comparison never
    # re-counts work the pipeline already counted.
    baseline_experiments = 0
    if eval_report and eval_report.experiment_count:
        baseline_experiments = int(eval_report.experiment_count.total_experiments)
    trials = 0
    counts: Optional[OptimizationCounts] = opt_report.counts if opt_report else None
    if counts is not None:
        trials = int(counts.total_trials)
    record.baseline_experiments = baseline_experiments
    record.trials = trials
    record.experiments = baseline_experiments + trials
    record.notes.append(
        f"experiments = baseline({baseline_experiments}) + optuna trials({trials})"
    )
    return record


def _split_facts_from_state(state: Any) -> Dict[str, Any]:
    """Recover the split facts recorded by the training node, with safe fallbacks."""
    facts = {"train_samples": 0, "test_samples": 0, "stratified": True, "reason": ""}
    data_split = getattr(state, "data_split", None)
    if data_split is not None:
        facts["train_samples"] = int(getattr(data_split, "train_samples", 0) or 0)
        facts["test_samples"] = int(getattr(data_split, "test_samples", 0) or 0)
        facts["stratified"] = bool(getattr(data_split, "stratified", True))
        facts["reason"] = str(
            getattr(data_split, "stratification_reason", "") or ""
        )
    return facts


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def _build_comparison(
    dataset_key: str,
    benchmark_sha256: str,
    conventional: ConventionalWorkflowResult,
    autonomous: WorkflowBenchmarkResult,
) -> DatasetComparison:
    """Assemble one dataset comparison from the two measured sides."""
    conventional_record = _conventional_to_record(dataset_key, conventional)
    delta_f1: Optional[float] = None
    if conventional_record.f1_score is not None and autonomous.f1_score is not None:
        delta_f1 = round(autonomous.f1_score - conventional_record.f1_score, 6)
    delta_accuracy: Optional[float] = None
    if (
        conventional_record.accuracy is not None
        and autonomous.accuracy is not None
    ):
        delta_accuracy = round(
            autonomous.accuracy - conventional_record.accuracy, 6
        )
    delta_cv_f1: Optional[float] = None
    if (
        conventional_record.cv_f1_mean is not None
        and autonomous.cv_f1_mean is not None
    ):
        delta_cv_f1 = round(
            autonomous.cv_f1_mean - conventional_record.cv_f1_mean, 6
        )

    f1_relative: Optional[float] = None
    if delta_f1 is not None and conventional_record.f1_score:
        f1_relative = round(
            100.0 * delta_f1 / conventional_record.f1_score, 2
        )

    notes: List[str] = []
    if delta_f1 is not None and delta_f1 >= 0:
        verdict = (
            f"autonomous matched or exceeded conventional F1 "
            f"(delta_f1={delta_f1:+.4f})"
        )
    elif delta_f1 is not None:
        verdict = (
            f"conventional scored higher F1 (delta_f1={delta_f1:+.4f})"
        )
    else:
        verdict = "no comparable F1 could be computed"
    notes.append(
        f"autonomous explored {autonomous.experiments} experiments "
        f"({autonomous.baseline_experiments} baseline + {autonomous.trials} trials) "
        f"vs conventional {conventional_record.experiments} manual configurations"
    )
    notes.append(
        f"runtime is workload cost, not quality: conventional={conventional_record.runtime_seconds:.3f}s "
        f"autonomous={autonomous.runtime_seconds:.3f}s"
    )

    return DatasetComparison(
        dataset_key=dataset_key,
        dataset_sha256=benchmark_sha256,
        conventional=conventional_record,
        autonomous=autonomous,
        delta_f1=delta_f1,
        delta_accuracy=delta_accuracy,
        delta_cv_f1=delta_cv_f1,
        delta_runtime=round(
            autonomous.runtime_seconds - conventional_record.runtime_seconds, 6
        ),
        delta_experiments=autonomous.experiments - conventional_record.experiments,
        f1_relative_percent=f1_relative,
        verdict=verdict,
        notes=notes,
    )


def _conventional_to_record(
    dataset_key: str, result: ConventionalWorkflowResult
) -> WorkflowBenchmarkResult:
    """Turn the conventional workflow result into the comparison record shape."""
    best = result.best
    best_experiment = result.best_experiment
    record = WorkflowBenchmarkResult(
        workflow="conventional",
        dataset_key=dataset_key,
        selected_model=best.model_name if best else "",
        selection_source="grid",
        f1_score=_nan_to_none(result.best_f1_score),
        accuracy=_nan_to_none(result.best_accuracy),
        cv_f1_mean=_nan_to_none(result.best_cv_f1_mean),
        precision=_nan_to_none(best_experiment.precision if best_experiment else None),
        recall=_nan_to_none(best_experiment.recall if best_experiment else None),
        parameters=dict(result.best_parameters or {}),
        runtime_seconds=result.runtime_seconds,
        experiments=result.experiment_count,
        baseline_experiments=0,
        grid_experiments=result.experiment_count,
        trials=0,
        target_column=result.target_column,
        problem_type=result.problem_type,
        feature_count=result.feature_count,
        train_samples=result.train_samples,
        test_samples=result.test_samples,
        split_stratified=result.split_stratified,
        split_reason=result.split_reason,
        stage_times=dict(result.stage_times),
        status="success",
        notes=[
            "conventional side = manual grid over the shared registry using "
            "tools only (no agents, no LangGraph, no LLM, no Optuna)",
            f"selection_source=grid, best_configuration={result.selection_reason}",
        ],
    )
    record.dataset_sha = result.dataset_sha256
    return record


# --------------------------------------------------------------------------- #
# MLflow recording (dedicated store; best-effort, never fails the benchmark)
# --------------------------------------------------------------------------- #
def record_benchmark_mLflow(
    comparison: DatasetComparison,
    *,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Record one dataset comparison in the dedicated benchmark MLflow store.

    The store is the *benchmark* store (``experiments/tracking_benchmark/``), a
    different file from the canonical store. Any failure — including a blocked
    dependency or an unreachable store — is caught and reported instead of
    failing the benchmark.

    Returns:
        A status dictionary with ``recorded``, ``tracking_uri`` and, on
        failure, the error message.
    """
    tracking_uri = tracking_uri or benchmark_config.BENCHMARK_MLFLOW_TRACKING_URI
    experiment_name = experiment_name or benchmark_config.BENCHMARK_MLFLOW_EXPERIMENT_NAME
    outcome: Dict[str, Any] = {
        "recorded": False,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_ids": [],
    }
    try:
        import mlflow.client

        client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                experiment_name,
                tags={**benchmark_config.BENCHMARK_MLFLOW_TAGS, "kind": "benchmark"},
            )
        else:
            experiment_id = experiment.experiment_id

        for workflow_record in (comparison.conventional, comparison.autonomous):
            run = client.create_run(
                experiment_id,
                tags={
                    **benchmark_config.BENCHMARK_MLFLOW_TAGS,
                    "dataset": comparison.dataset_key,
                    "workflow": workflow_record.workflow,
                },
            )
            run_id = run.info.run_id
            for name, value in {
                "dataset": comparison.dataset_key,
                "workflow": workflow_record.workflow,
                "selected_model": workflow_record.selected_model,
                "selection_source": workflow_record.selection_source,
                "split_test_size": str(config.TEST_SIZE),
                "split_random_state": str(config.RANDOM_STATE),
                "dataset_sha256": comparison.dataset_sha256,
            }.items():
                client.log_param(run_id, name, value)
            for name, value in {
                "f1_score": workflow_record.f1_score,
                "accuracy": workflow_record.accuracy,
                "precision": workflow_record.precision,
                "recall": workflow_record.recall,
                "cv_f1_mean": workflow_record.cv_f1_mean,
                "runtime_seconds": workflow_record.runtime_seconds,
                "experiments": float(workflow_record.experiments),
            }.items():
                if value is not None:
                    client.log_metric(run_id, name, value)
            client.set_terminated(run_id, status="FINISHED")
            outcome["run_ids"].append(run_id)
        outcome["recorded"] = bool(outcome["run_ids"])
    except Exception as error:  # noqa: BLE001 - MLflow must never fail the benchmark
        outcome["recorded"] = False
        outcome["error"] = f"{type(error).__name__}: {error}"[:500]
    return outcome


# --------------------------------------------------------------------------- #
# Report artifacts
# --------------------------------------------------------------------------- #
def _fit_dataset_key(dataset_path: Path) -> str:
    """Return the benchmark key for a fixture path, else the stem."""
    for key, path in benchmark_config.DATASET_FILES.items():
        if path.resolve() == dataset_path.resolve():
            return key
    return dataset_path.parent.name or dataset_path.stem


def _write_json(suite: BenchmarkSuiteResult, path: Path) -> Path:
    payload = {
        "metadata": {
            "phase": 11,
            "timing_method": suite.timing_method,
            "experiment_definition": suite.experiment_definition,
            "split": suite.split,
            "conventional_grid_size": suite.conventional_grid_size,
            "conventional_grid": benchmark_config.build_conventional_grid(),
            "autonomous_n_trials": suite.autonomous_n_trials,
            "candidate_families": suite.candidate_families,
            "generated_at": suite.generated_at,
            "mlflow": suite.mlflow,
            "files_written": suite.files_written,
        },
        "datasets": {
            comparison.dataset_key: {
                "dataset_sha256": comparison.dataset_sha256,
                "verdict": comparison.verdict,
                "delta": {
                    "f1": comparison.delta_f1,
                    "accuracy": comparison.delta_accuracy,
                    "cv_f1": comparison.delta_cv_f1,
                    "f1_relative_percent": comparison.f1_relative_percent,
                    "runtime_seconds": comparison.delta_runtime,
                    "experiments": comparison.delta_experiments,
                },
                "conventional": comparison.conventional.summary_row(),
                "autonomous": comparison.autonomous.summary_row(),
                "conventional_details": {
                    "stage_times": comparison.conventional.stage_times,
                    "selection_reason": comparison.conventional.notes,
                    "parameters": comparison.conventional.parameters,
                },
                "autonomous_details": {
                    "node_durations": comparison.autonomous.node_durations,
                    "parameters": comparison.autonomous.parameters,
                },
                "notes": comparison.notes,
            }
            for comparison in suite.datasets
        },
    }
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _write_csv(suite: BenchmarkSuiteResult, path: Path) -> Path:
    columns = [
        "dataset",
        "workflow",
        "model",
        "selection_source",
        "f1_score",
        "accuracy",
        "precision",
        "recall",
        "cv_f1_mean",
        "runtime_seconds",
        "experiments",
        "baseline_experiments",
        "grid_experiments",
        "trials",
        "target_column",
        "problem_type",
        "feature_count",
        "train_samples",
        "test_samples",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for comparison in suite.datasets:
            writer.writerow(comparison.conventional.summary_row())
            writer.writerow(comparison.autonomous.summary_row())
    return path


def _write_charts(suite: BenchmarkSuiteResult, output_dir: Path) -> List[Path]:
    """Render the three comparison charts. Never raises (charts are best-effort)."""
    written: List[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        datasets = [comparison.dataset_key for comparison in suite.datasets]
        labels = ["Conventional", "Autonomous"]

        f1_scores = [
            (
                comparison.conventional.f1_score or 0.0,
                comparison.autonomous.f1_score or 0.0,
            )
            for comparison in suite.datasets
        ]
        _bar_chart(
            plt,
            x_labels=datasets,
            series=zip(labels, zip(*f1_scores)),
            title="Phase 11: weighted F1 by workflow",
            ylabel="weighted F1",
            output=output_dir / "phase11_f1_comparison.png",
        )
        written.append(output_dir / "phase11_f1_comparison.png")

        runtimes = [
            (
                comparison.conventional.runtime_seconds,
                comparison.autonomous.runtime_seconds,
            )
            for comparison in suite.datasets
        ]
        _bar_chart(
            plt,
            x_labels=datasets,
            series=zip(labels, zip(*runtimes)),
            title="Phase 11: wall-clock runtime by workflow",
            ylabel="seconds (time.perf_counter)",
            output=output_dir / "phase11_runtime_comparison.png",
        )
        written.append(output_dir / "phase11_runtime_comparison.png")

        # The chart deliberately uses a log scale: the autonomous side explores
        # roughly 2x the configurations, and linear bars would hide both counts.
        experiments = [
            (
                float(comparison.conventional.experiments),
                float(comparison.autonomous.experiments),
            )
            for comparison in suite.datasets
        ]
        _bar_chart(
            plt,
            x_labels=datasets,
            series=zip(labels, zip(*experiments)),
            title="Phase 11: experiments evaluated by workflow",
            ylabel="experiments (log scale)",
            output=output_dir / "phase11_experiment_comparison.png",
            log_scale=True,
        )
        written.append(output_dir / "phase11_experiment_comparison.png")
    except Exception as error:  # noqa: BLE001 - charts are a report nicety
        suite.mlflow["charts_error"] = f"{type(error).__name__}: {error}"
        for comparison in suite.datasets:
            comparison.notes.append(f"charts could not be rendered: {error}")
    return written


def _bar_chart(
    plt: Any,
    *,
    x_labels: Sequence[str],
    series: Any,
    title: str,
    ylabel: str,
    output: Path,
    log_scale: bool = False,
) -> None:
    """Draw a simple grouped bar chart and save it to `output`."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    import numpy as np

    positions = np.arange(len(x_labels))
    width = 0.35
    for index, (name, values) in enumerate(series):
        values = [float(v) for v in values]
        axis.bar(
            positions + index * width,
            values,
            width,
            label=name,
        )
    axis.set_xticks(positions + width / 2)
    axis.set_xticklabels(x_labels)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    if log_scale and all(v > 0 for values in [v for _, v in series] for v in values):
        axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Top-level entry points
# --------------------------------------------------------------------------- #
def run_dataset_benchmark(
    dataset_key: str,
    *,
    n_trials: int = benchmark_config.AUTONOMOUS_N_TRIALS,
    benchmark_mLflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    scratch_dir: Optional[str] = None,
) -> DatasetComparison:
    """Run both workflows on one dataset and build the comparison."""
    dataset_path = benchmark_config.resolve_dataset_path(dataset_key)
    benchmark_sha256 = _sha256_file(dataset_path)

    conventional = run_conventional_workflow(dataset_path)
    auton_scratch = scratch_dir or tempfile.mkdtemp(prefix="phase11_artifacts_")
    autonomous = _run_autonomous_side(
        dataset_key,
        str(dataset_path),
        n_trials=n_trials,
        scratch_dir=auton_scratch,
    )

    # Both sides carry the same dataset fingerprint for the symmetric record.
    autonomous.dataset_sha = benchmark_sha256

    comparison = _build_comparison(
        dataset_key, benchmark_sha256, conventional, autonomous
    )

    if benchmark_mLflow:
        comparison.notes.append(
            "mlflow_recording="
            + json.dumps(
                _json_safe(
                    record_benchmark_mLflow(
                        comparison,
                        tracking_uri=tracking_uri,
                        experiment_name=experiment_name,
                    )
                )
            )
        )
    return comparison


def run_benchmark(
    datasets: Optional[Sequence[str]] = None,
    *,
    n_trials: int = benchmark_config.AUTONOMOUS_N_TRIALS,
    benchmark_mLflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    scratch_dir: Optional[str] = None,
    output_dir: Any = None,
    write_reports: bool = True,
) -> BenchmarkSuiteResult:
    """Run the controlled benchmark on one or both committed datasets.

    Args:
        datasets: Dataset keys to benchmark; None means both.
        n_trials: Optuna trial budget for the autonomous side (defaults to the
            project's own ``OPTUNA_N_TRIALS``).
        benchmark_mLflow: Whether to record results in the dedicated benchmark
            MLflow store (best-effort).
        tracking_uri / experiment_name: Overrides for the MLflow recording.
        scratch_dir: Directory for the autonomous persistence artifacts (defaults
            to a temporary directory).
        output_dir: Where the JSON/CSV/PNG reports are written (defaults to
            ``reports/``).
        write_reports: Whether to write the comparison files at all.

    Returns:
        A `BenchmarkSuiteResult` with one entry per dataset.
    """
    import datetime

    dataset_keys = list(datasets or benchmark_config.ALL_DATASET_KEYS)
    output_dir = Path(output_dir or benchmark_config.REPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = BenchmarkSuiteResult()
    suite.generated_at = datetime.datetime.now().isoformat(timespec="seconds")
    suite.candidate_families = list(get_supported_models())

    for dataset_key in dataset_keys:
        comparison = run_dataset_benchmark(
            dataset_key,
            n_trials=n_trials,
            benchmark_mLflow=benchmark_mLflow,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            scratch_dir=scratch_dir,
        )
        suite.datasets.append(comparison)

    suite.mlflow["enabled"] = bool(benchmark_mLflow)
    suite.mlflow["tracking_uri"] = (
        tracking_uri or benchmark_config.BENCHMARK_MLFLOW_TRACKING_URI
    )
    suite.mlflow["experiment_name"] = (
        experiment_name or benchmark_config.BENCHMARK_MLFLOW_EXPERIMENT_NAME
    )
    suite.mlflow["store_error"] = ""

    written: List[str] = []
    if write_reports:
        csv_path = output_dir / "phase11_comparison.csv"
        json_path = output_dir / "phase11_comparison.json"
        _write_csv(suite, csv_path)
        written.append(str(csv_path))
        written.extend(str(p) for p in _write_charts(suite, output_dir))
        # The JSON is written last so its ``files_written`` metadata describes
        # the whole set of outputs, itself excluded.
        suite.files_written = list(written)
        _write_json(suite, json_path)
        written.append(str(json_path))
    suite.files_written = written
    return suite


# --------------------------------------------------------------------------- #
# Human-readable rendering (used by the CLI and tests)
# --------------------------------------------------------------------------- #
def format_comparison_text(comparison: DatasetComparison) -> str:
    """Render one dataset comparison as a fixed-width text block."""
    rows = [
        f"== benchmark dataset: {comparison.dataset_key} "
        f"(sha256 {comparison.dataset_sha256[:12]}…) ==",
        "",
        f"{'metric':<18}{'conventional':>14}{'autonomous':>14}{'delta':>14}",
        "-" * 60,
    ]
    pairs = [
        ("weighted F1", comparison.conventional.f1_score, comparison.autonomous.f1_score, comparison.delta_f1),
        ("accuracy", comparison.conventional.accuracy, comparison.autonomous.accuracy, comparison.delta_accuracy),
        ("CV F1 (mean)", comparison.conventional.cv_f1_mean, comparison.autonomous.cv_f1_mean, comparison.delta_cv_f1),
    ]
    for label, conv, auto, delta in pairs:
        rows.append(
            f"{label:<18}"
            f"{'-' if conv is None else f'{conv:.6f}':>14}"
            f"{'-' if auto is None else f'{auto:.6f}':>14}"
            f"{'-' if delta is None else f'{delta:+.6f}':>14}"
        )
    rows.append(
        f"{'runtime (s)':<18}"
        f"{comparison.conventional.runtime_seconds:>14.3f}"
        f"{comparison.autonomous.runtime_seconds:>14.3f}"
        f"{comparison.delta_runtime:>+14.3f}"
    )
    rows.append(
        f"{'experiments':<18}"
        f"{comparison.conventional.experiments:>14d}"
        f"{comparison.autonomous.experiments:>14d}"
        f"{comparison.delta_experiments:>+14d}"
    )
    rows.append("-" * 60)
    rows.append(
        f"conventional best: {comparison.conventional.selected_model} "
        f"(params {comparison.conventional.parameters})"
    )
    rows.append(
        f"autonomous best:   {comparison.autonomous.selected_model} "
        f"(source={comparison.autonomous.selection_source})"
    )
    rows.append(f"verdict: {comparison.verdict}")
    for note in comparison.notes:
        rows.append(f"note:   {note}")
    return "\n".join(rows)