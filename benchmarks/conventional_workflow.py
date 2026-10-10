"""The conventional (manual-style) workflow measured by the Phase 11 benchmark.

This module implements the "traditional manual ML process" side of the
comparison. It is deliberately a plain script: it chains the project's reusable
**tool** functions (``data_tools``, ``preprocessing_tools``, ``training_tools``)
the same way a data scientist would run them by hand in a notebook, and it
completely avoids the autonomous machinery:

- it never instantiates an agent (no ``agents.*`` import),
- it never builds or runs a LangGraph workflow (no ``graph.*`` import),
- it never calls an LLM (no ``llm.*`` import),
- it never uses Optuna (no ``optimization_tools`` / ``optuna`` import),
- it performs a small, fully predetermined manual grid instead of a search.

Fairness with the autonomous side is structural, not promised in prose:

- the very same committed dataset (SHA-256 checked) is analysed with the very
  same ``data_tools.analyze_dataset`` function,
- the preprocessing plan comes from the very same
  ``preprocessing_tools.plan_preprocessing`` defaults,
- the split uses the very same ``training_tools.split_dataset`` defaults
  (``TEST_SIZE``, ``RANDOM_STATE``, stratification), so both sides train and
  score on the same rows,
- the candidate families are the five models of the shared registry, and the
  manual grid is derived from the Phase 6 search-space bounds,
- the metrics come from ``training_tools.evaluate_classifier`` (weighted
  averages) and the final selection reuses ``select_best_experiment``, i.e. the
  project's documented ``f1 desc / accuracy desc / name asc`` ranking rule.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks import benchmark_config
from tools import data_tools, preprocessing_tools, training_tools
from tools.model_tools import get_model_definition, get_supported_models
from utils.state import BestModelInfo, ExperimentResult, ExperimentStatus


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ConventionalWorkflowError(RuntimeError):
    """Raised when the conventional workflow cannot proceed."""


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class ConventionalWorkflowResult:
    """Everything the conventional workflow produced for one dataset."""

    dataset_name: str
    dataset_path: str
    dataset_sha256: str
    target_column: str
    problem_type: str
    problem_type_reason: str
    feature_count: int
    train_samples: int
    test_samples: int
    split_stratified: bool
    split_reason: str
    candidate_families: List[str] = field(default_factory=list)
    grid_configurations: int = 0
    experiments: List[ExperimentResult] = field(default_factory=list)
    successful_experiments: int = 0
    failed_experiments: int = 0
    best: Optional[BestModelInfo] = None
    best_experiment: Optional[ExperimentResult] = None
    selection_reason: str = ""
    stage_times: Dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    @property
    def experiment_count(self) -> int:
        """Total experiments: one per evaluated configuration, CV folds excluded."""
        return len(self.experiments)

    @property
    def best_f1_score(self) -> Optional[float]:
        return self.best.f1_score if self.best else None

    @property
    def best_accuracy(self) -> Optional[float]:
        return self.best.accuracy if self.best else None

    @property
    def best_cv_f1_mean(self) -> Optional[float]:
        return self.best.cv_f1_mean if self.best else None

    @property
    def best_parameters(self) -> Dict[str, Any]:
        return self.best.parameters if self.best else {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file (bytes, not newline-normalised)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_one_experiment(
    *,
    preprocessing_config: Any,
    split: Any,
    cv_config: Any,
    model_name: str,
    parameters: Dict[str, Any],
) -> ExperimentResult:
    """Train, evaluate (and cross-validate) one manual configuration.

    Returns:
        An `ExperimentResult` mirroring exactly what the autonomous baseline
        stage records for one experiment: deterministic id, hold-out metrics,
        CV means and per-step timings. Failures are recorded on the record, so
        one broken configuration never aborts the whole manual pass.
    """
    definition = get_model_definition(model_name)
    experiment = ExperimentResult(
        experiment_id=training_tools.build_experiment_id(model_name, parameters),
        model_name=model_name,
        model_type=definition.model_type,
        preprocessing_scaling=definition.scaling_required,
        parameters=dict(parameters),
        input_feature_count=int(split.X_train.shape[1]),
        train_samples=split.train_samples,
        test_samples=split.test_samples,
    )
    try:
        pipeline = training_tools.build_model_pipeline(
            preprocessing_config, model_name, parameters
        )

        started = time.perf_counter()
        pipeline.fit(split.X_train, split.y_train)
        experiment.training_time = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        predictions = pipeline.predict(split.X_test)
        experiment.prediction_time = round(time.perf_counter() - started, 6)

        metrics = training_tools.evaluate_classifier(split.y_test, predictions)
        experiment.accuracy = metrics["accuracy"]
        experiment.precision = metrics["precision"]
        experiment.recall = metrics["recall"]
        experiment.f1_score = metrics["f1_score"]

        if cv_config.enabled:
            started = time.perf_counter()
            cv_result = training_tools.run_cross_validation(
                pipeline,
                split.X_train,
                split.y_train,
                cv_config,
                model_name=model_name,
            )
            experiment.cv_time = round(time.perf_counter() - started, 6)
            experiment.cv_accuracy_mean = cv_result.accuracy_mean
            experiment.cv_f1_mean = cv_result.f1_mean
        else:
            experiment.status = ExperimentStatus.PARTIAL.value
    except Exception as error:  # noqa: BLE001 - recorded per experiment on purpose
        experiment.status = ExperimentStatus.FAILED.value
        experiment.error_message = (
            f"{type(error).__name__}: {error}".replace("\n", " ")[:500]
        )
    experiment.total_time = round(
        experiment.training_time + experiment.prediction_time + experiment.cv_time, 6
    )
    return experiment


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _select_grid_best(
    experiments: List[ExperimentResult],
) -> "tuple[Optional[BestModelInfo], Optional[ExperimentResult]]":
    """Select the best *configuration* with the project's documented ranking rule.

    The project's ``select_best_experiment`` collapses candidates by model
    *name*, which is exactly right for the Phase 4 baseline stage (one
    experiment per family) but wrong for the manual grid, where one family
    contributes several configurations. ``rank_experiments`` implements the
    same documented rule - f1 descending, accuracy descending, stable registry
    order as the final tie-break - at the granularity of individual
    experiments, which is the granularity a manual pass needs: the winner is
    one concrete configuration with one concrete parameter set.

    Args:
        experiments: The `ExperimentResult` records of the manual grid.

    Returns:
        A ``(best_info, best_experiment)`` pair, or ``(None, None)`` when no
        experiment produced usable metrics.
    """
    comparison = training_tools.rank_experiments(experiments)
    if not comparison.entries:
        return None, None
    primary = comparison.primary_metric
    secondary = comparison.secondary_metric
    top = comparison.entries[0]

    def matches(experiment: ExperimentResult) -> bool:
        if getattr(experiment, "status", None) not in (
            ExperimentStatus.SUCCESS.value,
            ExperimentStatus.PARTIAL.value,
        ):
            return False
        if getattr(experiment, primary, None) != getattr(top, primary, None):
            return False
        if getattr(experiment, secondary, None) != getattr(top, secondary, None):
            return False
        return True

    winner = next((experiment for experiment in experiments if matches(experiment)), None)
    if winner is None:
        return None, None
    best = BestModelInfo(
        model_name=winner.model_name,
        experiment_id=winner.experiment_id,
        f1_score=winner.f1_score,
        accuracy=winner.accuracy,
        cv_f1_mean=winner.cv_f1_mean,
        preprocessing_scaling=winner.preprocessing_scaling,
        parameters=dict(winner.parameters),
        selection_reason=(
            f"highest {primary} ({getattr(top, primary, None):.4f}) among "
            f"{comparison.successful_count} successful grid configuration(s); "
            f"tie-break on {secondary}"
        ),
    )
    return best, winner


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #
def run_conventional_workflow(
    dataset_path: Any,
    *,
    cross_validation: bool = True,
    grid: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> ConventionalWorkflowResult:
    """Run the conventional workflow on one dataset and return the full result.

    Args:
        dataset_path: Path to a CSV fixture (usually a benchmark dataset key's
            resolved fixture path).
        cross_validation: Whether each configuration is cross-validated (True
            mirrors the autonomous workflow, which cross-validates its models).
        grid: Manual configuration grid overriding the predefined benchmark one.
            Every grid entry is a complete parameter dict for
            ``training_tools.build_model_pipeline``.

    Returns:
        A `ConventionalWorkflowResult` with the grid size, per-configuration
        experiments, the best model under the project's selection rule, and the
        measured runtime.

    Raises:
        ConventionalWorkflowError: The dataset has no usable target or no
            usable configuration was produced.
    """
    started = time.perf_counter()

    dataset_path = Path(dataset_path)
    dataset_sha256 = _sha256(dataset_path)

    # --- 1. Load + 2. analysis (same tools the Data Analysis Agent uses) ----- #
    dataframe = data_tools.load_csv(str(dataset_path))
    stage_mark = time.perf_counter()
    analysis = data_tools.analyze_dataset(dataframe)
    analysis_time = round(time.perf_counter() - stage_mark, 6)

    target_column = analysis.target_detection.column
    target_analysis = analysis.target_analysis
    if not target_column or target_analysis is None:
        raise ConventionalWorkflowError(
            f"No usable target column was detected in '{dataset_path.name}'."
        )
    problem_type = target_analysis.problem_type.value
    if not target_analysis.is_supported:
        raise ConventionalWorkflowError(
            f"Target '{target_column}' is not a supported classification target "
            f"({target_analysis.problem_type_reason})."
        )

    # --- 3. Preprocessing plan (the Preprocessing Agent's defaults) ---------- #
    stage_mark = time.perf_counter()
    preprocessing_config = preprocessing_tools.plan_preprocessing(
        dataframe,
        target_column=target_column,
        duplicate_count=analysis.duplicate_count,
        data_quality_issues=analysis.data_quality_issues,
        unsupported_columns=analysis.feature_types.unsupported_columns,
    )
    planning_time = round(time.perf_counter() - stage_mark, 6)

    # --- 4. Duplicate handling (the Training Agent's step) ------------------- #
    handling_strategy = preprocessing_config.duplicate_handling.strategy
    dataframe, _removed = preprocessing_tools.remove_duplicate_rows(
        dataframe, strategy=handling_strategy
    )

    # --- 5. Features / target + 6. the project's exact split ----------------- #
    excluded_columns = [item.column for item in preprocessing_config.excluded_features]
    features, target = training_tools.prepare_features_and_target(
        dataframe, target_column, excluded_columns=excluded_columns
    )
    stage_mark = time.perf_counter()
    split = training_tools.split_dataset(features, target)
    split_time = round(time.perf_counter() - stage_mark, 6)

    # --- 7. Cross-validation configuration ----------------------------------- #
    cv_config = training_tools.build_cross_validation_config(
        target, enabled=cross_validation
    )

    # --- 8. The manual grid of experiments ----------------------------------- #
    resolution = grid if grid is not None else benchmark_config.build_conventional_grid()
    candidate_families = get_supported_models()
    stage_mark = time.perf_counter()
    experiments: List[ExperimentResult] = []
    for model_name in candidate_families:
        for parameters in resolution.get(model_name, []):
            experiments.append(
                _run_one_experiment(
                    preprocessing_config=preprocessing_config,
                    split=split,
                    cv_config=cv_config,
                    model_name=model_name,
                    parameters=parameters,
                )
            )
    experiments_time = round(time.perf_counter() - stage_mark, 6)

    # --- 9. Selection with the project's own ranking rule -------------------- #
    best, best_experiment = _select_grid_best(experiments)

    successful = sum(1 for e in experiments if e.status == ExperimentStatus.SUCCESS.value)
    result = ConventionalWorkflowResult(
        dataset_name=dataset_path.name,
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        target_column=target_column,
        problem_type=problem_type,
        problem_type_reason=target_analysis.problem_type_reason,
        feature_count=int(split.X_train.shape[1]),
        train_samples=split.train_samples,
        test_samples=split.test_samples,
        split_stratified=split.stratified,
        split_reason=split.reason,
        candidate_families=list(candidate_families),
        grid_configurations=len(experiments),
        experiments=experiments,
        successful_experiments=successful,
        failed_experiments=len(experiments) - successful,
        best=best,
        best_experiment=best_experiment,
        selection_reason=(
            best.selection_reason
            if best is not None
            else "No experiment produced metrics."
        ),
        stage_times={
            "analysis": analysis_time,
            "preprocessing_planning": planning_time,
            "splitting": split_time,
            "experiments": experiments_time,
        },
        runtime_seconds=round(time.perf_counter() - started, 6),
    )
    return result