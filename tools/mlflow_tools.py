"""Local MLflow experiment tracking (Phase 10).

What this module is: an observability layer. It records decisions that the
deterministic workflow has *already* made and hands them to MLflow.

What this module is not: a decision maker. Nothing here selects a model, tunes a
hyperparameter, routes the graph, enables optimization, decides whether an
artifact is written or tells the LLM layer what to do. The rule the whole
design follows is the one below::

    Model Selection Agent -> decides, then MLflow logs the decision

    MLflow -> decides                       # never, and not even a hook for it

Consequences of that rule, all deliberate:

* The module imports **nothing** from ``agents/`` or ``graph/``. The graph calls
  into it with finished data, so tracking cannot introduce a circular import or
  reach back and influence a stage.
* No ML logic is reimplemented here. Nothing recomputes a metric, re-reads the
  dataset or re-runs Data Analysis. Every value is read from the ``AgentState``
  the workflow already produced, which is also why Phase 10 cannot change a
  Phase 1-9 result: it only reads.
* It needs no server. The tracking URI is a local directory under
  ``experiments/`` by default.

Failure isolation is the other half of the design. A tracking failure must never
turn a successful machine learning run into a failed one, so every public
function here catches its own errors, emits one ``logger.warning`` and returns a
recorded status. :class:`TrackingError` is the single typed exception, it is
raised only by the private helpers the public functions are built from, and it
never escapes a public function.

The read side (:func:`experiment_summary`, :func:`recent_runs`) exists for the
dashboard and obeys the same rule: it reports what is in the tracking store and
raises nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils import config
from utils.logger import setup_logger

logger = setup_logger("tools.mlflow_tools")


# --------------------------------------------------------------------------- #
# Errors and statuses
# --------------------------------------------------------------------------- #


class TrackingError(Exception):
    """Raised when a tracking operation fails.

    This is the only exception type this module raises. It exists so that the
    public functions below have something concrete to catch and turn into a
    recorded status; callers of the public API never see it.
    """


class TrackingStatus(str, Enum):
    """What happened to the tracking of one workflow run.

    ``RECORDED`` and ``FAILED`` are deliberately separate from the machine
    learning outcome. A run can be ``RECORDED`` while the workflow was
    ``partial``, and the workflow can be ``completed`` while tracking was
    ``FAILED``; collapsing the two would hide exactly the information this layer
    exists to provide.
    """

    #: Tracking was switched off by configuration or by ``--no-mlflow``.
    DISABLED = "disabled"
    #: The mlflow package is not installed, so nothing was attempted.
    UNAVAILABLE = "unavailable"
    #: The run was created and everything this module tried was logged.
    RECORDED = "recorded"
    #: The run was started but at least one logging call failed.
    FAILED = "failed"


@dataclass
class TrackingSummary:
    """The outcome of one tracking attempt.

    Attributes:
        status: What happened to the tracking of the run.
        run_id: The MLflow run id, when a run was created.
        experiment_name: The experiment the run belongs to.
        tracking_uri: The tracking store that was used.
        reason: Why the status is not ``RECORDED``. Empty when it is.
        failures: Every individual logging call that failed, in order.
    """

    status: TrackingStatus = TrackingStatus.DISABLED
    run_id: str = ""
    experiment_name: str = ""
    tracking_uri: str = ""
    reason: str = ""
    failures: List[str] = field(default_factory=list)

    @property
    def recorded(self) -> bool:
        """Whether a run exists and nothing that was attempted failed."""
        return self.status is TrackingStatus.RECORDED

    @property
    def usable(self) -> bool:
        """Whether a reader can look this run up."""
        return bool(self.run_id)


# --------------------------------------------------------------------------- #
# MLflow availability (lazy, never imported at module import time)
# --------------------------------------------------------------------------- #

#: The imported module, cached after the first successful import. ``None`` until
#: then, and after an explicit ``reset_mlflow_cache()`` in a test.
_MLFLOW: Any = None
_MLFLOW_IMPORT_FAILED = False


def reset_mlflow_cache() -> None:
    """Forget the cached mlflow import.

    Only useful to a test that has to observe the import-failed path after a
    successful import, because the real module is cached for the process.
    """
    global _MLFLOW, _MLFLOW_IMPORT_FAILED
    _MLFLOW = None
    _MLFLOW_IMPORT_FAILED = False


def import_mlflow() -> Any:
    """Import mlflow on demand and cache the module.

    The import is deferred rather than done at module import time for two
    reasons: the module must import cleanly on a machine where mlflow is not
    installed, and importing mlflow costs about a second, which would be paid by
    every command that touches this file.

    Returns:
        The mlflow module.

    Raises:
        TrackingError: mlflow is not installed, or failed to import.
    """
    global _MLFLOW, _MLFLOW_IMPORT_FAILED
    if _MLFLOW is not None:
        return _MLFLOW
    if _MLFLOW_IMPORT_FAILED:
        raise TrackingError("mlflow was already found to be unimportable")
    try:
        import mlflow  # noqa: PLC0415 - deliberately deferred
    except Exception as error:  # noqa: BLE001 - any import failure is the same
        _MLFLOW_IMPORT_FAILED = True
        raise TrackingError(
            f"mlflow is not importable ({type(error).__name__}: {error})"
        ) from error
    _MLFLOW = mlflow
    return _MLFLOW


def mlflow_available() -> bool:
    """Whether mlflow can be imported right now.

    Returns:
        True when the import succeeds, False when it does not. Never raises: a
        caller asking this question is asking about availability, and an
        exception would be the wrong answer.
    """
    try:
        import_mlflow()
    except TrackingError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Value coercion
# --------------------------------------------------------------------------- #


def param_value(value: Any) -> Optional[str]:
    """Coerce one value into something MLflow accepts as a parameter.

    MLflow parameters are strings, numbers or booleans, with a length limit. A
    run that logged an unconvertible value would raise and, through the
    failure-isolation rules, silently lose every later parameter - so the
    conversion is explicit and total.

    Args:
        value: Any value produced by the workflow.

    Returns:
        A string, number or boolean, or None when the value cannot be
        represented. None means "do not log this", never a silent empty string.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value or value in (
            float("inf"),
            float("-inf"),
        )):
            # NaN and the infinities are valid floats but are not meaningful
            # metrics and cannot be compared, so they are treated as absent.
            return None
        return value
    if isinstance(value, str):
        return value[: config.MLFLOW_MAX_PARAM_VALUE]
    if isinstance(value, (list, tuple, set)):
        try:
            text = json.dumps(
                sorted(value, key=str) if isinstance(value, set) else list(value),
                default=str,
            )
        except (TypeError, ValueError):
            text = str(value)
        return text[: config.MLFLOW_MAX_PARAM_VALUE]
    if isinstance(value, Mapping):
        try:
            text = json.dumps(dict(value), default=str, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)
        return text[: config.MLFLOW_MAX_PARAM_VALUE]
    return str(value)[: config.MLFLOW_MAX_PARAM_VALUE]


def metric_value(value: Any) -> Optional[float]:
    """Coerce one value into a float MLflow will accept as a metric.

    Args:
        value: Any value produced by the workflow.

    Returns:
        The value as a float, or None when it is absent or not a finite number.
        A metric that was never measured is logged as absent, never as 0.0,
        because a zero would be read as a real measurement.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def slug(text: str, limit: int = 200) -> str:
    """Turn an arbitrary string into a stable MLflow entity key.

    MLflow caps a key at 250 characters and rejects the ones containing a
    newline, so a free-text reason string cannot be used verbatim.

    Args:
        text: The text to convert.
        limit: The maximum length of the result.

    Returns:
        A key-safe string. Empty input returns an empty string rather than a
        placeholder, so the caller can decide to skip it.
    """
    cleaned = " ".join(str(text or "").split())
    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    return cleaned[:limit]


# --------------------------------------------------------------------------- #
# Pure extractors: AgentState -> plain data
# --------------------------------------------------------------------------- #
#
# These are deliberately plain functions over finished results with no mlflow
# involvement whatsoever. That is what makes the interesting half of Phase 10
# testable without a tracking store at all, and it is also the structural proof
# that this layer reads decisions rather than making them.
#
# Every accessor is defensive. A workflow can stop early, in which case most of
# these fields are absent, and a tracking layer must describe what did happen
# without inventing what did not.


def _state_of(source: Any) -> Any:
    """Return the AgentState from either an AgentState or a GraphState.

    Args:
        source: An ``AgentState``, a ``GraphState`` mapping, or None.

    Returns:
        The ``AgentState``, or None when there is none to read.
    """
    if source is None:
        return None
    agent_state = getattr(source, "agent_state", None)
    if agent_state is not None:
        return agent_state
    if isinstance(source, Mapping):
        return source.get("agent_state")
    return source


def _get(container: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dataclass or a mapping.

    Args:
        container: The object or mapping to read.
        name: The field name.
        default: What to return when the field is absent or None.

    Returns:
        The field value, or the default.
    """
    if container is None:
        return default
    if isinstance(container, Mapping):
        value = container.get(name, default)
    else:
        value = getattr(container, name, default)
    return default if value is None else value


def file_sha256(path: Any) -> str:
    """Return the SHA-256 of a file, or an empty string when it cannot be read.

    Recorded because a metric without the data that produced it cannot be
    reproduced, and because it is the cheapest available way to say "the same
    input produced this number".

    Args:
        path: The file to hash.

    Returns:
        The lowercase hex digest, or "" when the file is missing or unreadable.
        An unreadable dataset is not a tracking failure worth failing a run over.
    """
    if path is None:
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    try:
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as error:
        logger.warning(
            "%s could not hash %s: %s",
            config.MLFLOW_LOG_PREFIX,
            target.name,
            error,
        )
        return ""


def dataset_record(state: Any, dataset_path: Any = None) -> Dict[str, Any]:
    """Describe the dataset the workflow analysed.

    Metadata only. No row, no column value and no raw frame is recorded: the
    purpose is to make a run reproducible, and a copy of the input would be a
    duplication of a file that already lives in ``datasets/`` rather than a
    reproducibility aid.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.
        dataset_path: The dataset file, when the caller knows it.

    Returns:
        A flat mapping of dataset facts. Absent facts are simply not present, so
        a consumer can tell "not measured" from "measured as zero".
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return {}

    path = dataset_path if dataset_path is not None else _get(agent_state, "dataset_path")
    shape = _get(agent_state, "dataset_shape")
    used_feature_count = _get(
        _get(agent_state, "preprocessing_summary"), "used_feature_count"
    )

    record: Dict[str, Any] = {
        "dataset.name": _get(agent_state, "dataset_name"),
        "dataset.task_type": _get(agent_state, "problem_type"),
        "dataset.target_column": _get(agent_state, "target_column"),
        "dataset.target_detection_method": _get(
            agent_state, "target_detection_method"
        ),
        "dataset.target_detection_confidence": _get(
            agent_state, "target_detection_confidence"
        ),
        "dataset.rows": _get(shape, "rows"),
        "dataset.columns": _get(shape, "columns"),
        "dataset.missing_values": _get(agent_state, "total_missing_values"),
        "dataset.duplicate_rows": _get(agent_state, "duplicate_count"),
        "dataset.candidate_numerical_features": len(
            _get(agent_state, "numerical_features") or []
        ),
        "dataset.candidate_categorical_features": len(
            _get(agent_state, "categorical_features") or []
        ),
        "dataset.used_features": used_feature_count,
        "dataset.issue_count": _get(
            _get(agent_state, "analysis_summary"), "issue_count"
        ),
        "dataset.sha256": file_sha256(path),
    }
    return {key: value for key, value in record.items() if value is not None}


def preprocessing_record(state: Any) -> Dict[str, Any]:
    """Describe the preprocessing the workflow chose.

    Every value is the decision string or flag Phases 3 recorded. No fitted
    transformer, no column transformer object and no imputation value is logged:
    a parameter has to be readable in the MLflow UI, and the fitted state is
    already inside ``models/best_model.joblib``.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.

    Returns:
        A flat mapping of preprocessing decisions and counts.
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return {}
    pre_config = _get(agent_state, "preprocessing_config")
    summary = _get(agent_state, "preprocessing_summary")
    if pre_config is None and summary is None:
        return {}

    duplicates = _get(pre_config, "duplicate_handling")
    record: Dict[str, Any] = {
        "preprocessing.numerical_imputation": _get(pre_config, "numerical_imputation"),
        "preprocessing.categorical_imputation": _get(
            pre_config, "categorical_imputation"
        ),
        "preprocessing.encoding": _get(pre_config, "encoding"),
        "preprocessing.encoding_handle_unknown": _get(
            pre_config, "encoding_handle_unknown"
        ),
        "preprocessing.encoding_drop": _get(pre_config, "encoding_drop"),
        "preprocessing.scale_numeric": _get(pre_config, "scale_numeric"),
        "preprocessing.scaling_strategy": _get(pre_config, "scaling_strategy"),
        "preprocessing.remainder": _get(pre_config, "remainder"),
        "preprocessing.fit_scope": _get(pre_config, "fit_scope"),
        "preprocessing.duplicate_strategy": _get(duplicates, "strategy"),
        "preprocessing.duplicate_rows_detected": _get(
            duplicates, "duplicate_rows_detected"
        ),
        "preprocessing.duplicate_rows_removed": _get(
            duplicates, "duplicate_rows_removed"
        ),
        "preprocessing.numerical_features": len(
            _get(pre_config, "numerical_features") or []
        ),
        "preprocessing.categorical_features": len(
            _get(pre_config, "categorical_features") or []
        ),
        "preprocessing.used_feature_count": _get(summary, "used_feature_count"),
        "preprocessing.excluded_feature_count": _get(
            summary, "excluded_feature_count"
        ),
        "preprocessing.estimated_output_features": _get(
            summary, "estimated_output_feature_count"
        ),
        "preprocessing.pipeline_ready": _get(summary, "pipeline_ready"),
    }

    excluded = _get(pre_config, "excluded_features") or []
    if excluded:
        # The reason matters for reproduction, so it is recorded per feature
        # rather than collapsed into a count that hides which column went where.
        for item in excluded:
            column = slug(str(_get(item, "column", "")))
            if column:
                record[f"preprocessing.excluded.{column}"] = str(
                    _get(item, "reason", "")
                )
    return {key: value for key, value in record.items() if value is not None}


def comparison_rows(state: Any) -> List[Dict[str, Any]]:
    """Return one flat row per evaluated baseline model.

    This is the machine-readable model comparison. It exists as an artifact
    rather than as parameters because there is no way to create one MLflow run
    per model without turning a five-model comparison into a wall of runs that
    each say almost nothing.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.

    Returns:
        One dictionary per model, ordered by recorded rank when there is one.
        Empty when the evaluation stage never ran.
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return []
    report = _get(agent_state, "evaluation_report")

    rows: List[Dict[str, Any]] = []
    for entry in _get(report, "model_evaluations") or []:
        rows.append(
            {
                "rank": _get(entry, "rank"),
                "model_name": _get(entry, "model_name"),
                "model_type": _get(entry, "model_type"),
                "experiment_id": _get(entry, "experiment_id"),
                "experiment_status": _get(entry, "experiment_status"),
                "evaluation_status": _get(entry, "evaluation_status"),
                "scaling_required": _get(entry, "preprocessing_scaling"),
                "holdout_accuracy": _get(entry, "accuracy"),
                "holdout_precision": _get(entry, "precision"),
                "holdout_recall": _get(entry, "recall"),
                "holdout_f1_score": _get(entry, "f1_score"),
                "cv_accuracy_mean": _get(entry, "cv_accuracy_mean"),
                "cv_f1_mean": _get(entry, "cv_f1_mean"),
                "cv_n_splits": _get(entry, "cv_n_splits"),
                "f1_gap": _get(entry, "f1_gap"),
                "f1_gap_classification": _get(entry, "f1_gap_classification"),
                "training_time_seconds": _get(entry, "training_time"),
                "total_time_seconds": _get(entry, "total_time"),
            }
        )

    if not rows:
        # The evaluation report is the richer source, but a workflow can stop
        # before it is built while the training stage already recorded a
        # comparison. Reading that instead keeps the artifact populated whenever
        # any model was trained.
        for entry in (
            _get(_get(agent_state, "model_comparison"), "entries") or []
        ):
            rows.append(
                {
                    "rank": _get(entry, "rank"),
                    "model_name": _get(entry, "model_name"),
                    "model_type": _get(entry, "model_type"),
                    "experiment_id": None,
                    "experiment_status": None,
                    "evaluation_status": None,
                    "scaling_required": _get(entry, "preprocessing_scaling"),
                    "holdout_accuracy": _get(entry, "accuracy"),
                    "holdout_precision": _get(entry, "precision"),
                    "holdout_recall": _get(entry, "recall"),
                    "holdout_f1_score": _get(entry, "f1_score"),
                    "cv_accuracy_mean": _get(entry, "cv_accuracy_mean"),
                    "cv_f1_mean": _get(entry, "cv_f1_mean"),
                    "cv_n_splits": None,
                    "f1_gap": None,
                    "f1_gap_classification": None,
                    "training_time_seconds": None,
                    "total_time_seconds": _get(entry, "total_time"),
                }
            )

    def sort_key(row: Dict[str, Any]) -> Tuple[int, float, str]:
        rank = row.get("rank")
        f1 = row.get("holdout_f1_score")
        return (
            rank if isinstance(rank, int) else 10_000,
            -(f1 if isinstance(f1, (int, float)) else -1.0),
            str(row.get("model_name") or ""),
        )

    return sorted(rows, key=sort_key)


def evaluation_record(state: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Describe the Phase 5 evaluation outcome.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.

    Returns:
        A ``(params, metrics)`` pair. Params are the textual decisions - which
        model was recommended and under which rule - and metrics are the
        numbers. They are kept apart because MLflow treats the two differently:
        a param is an immutable fact about the run, a metric is a measurement.
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return {}, {}
    report = _get(agent_state, "evaluation_report")
    if report is None:
        return {}, {}

    best = _get(report, "best_model")
    split = _get(agent_state, "data_split")
    cv_config = _get(agent_state, "cross_validation_config")
    gap = _get(report, "gap_analysis")

    # ``ModelRecommendation`` carries the numbers but not the gap category,
    # which Phase 5 stores per model on ``ModelEvaluation``. The category of the
    # *selected* model is the one worth recording, so it is read from the row
    # that recommended the same model rather than being approximated.
    selected_name = _get(best, "best_baseline_model")
    selected_evaluation = next(
        (
            entry
            for entry in _get(report, "model_evaluations") or []
            if _get(entry, "model_name") == selected_name
        ),
        None,
    )

    params: Dict[str, Any] = {
        "evaluation.status": _get(report, "overall_status"),
        "evaluation.ranking_rule": _get(report, "ranking_rule"),
        "evaluation.selected_model": selected_name,
        "evaluation.selected_model_type": _get(best, "model_type"),
        "evaluation.selection_reason": _get(best, "selection_reason"),
        "evaluation.test_size": _get(split, "test_size"),
        "evaluation.stratified": _get(split, "stratified"),
        "evaluation.test_samples": _get(split, "test_samples"),
        "evaluation.train_samples": _get(split, "train_samples"),
        "evaluation.total_samples": _get(split, "total_samples"),
        "evaluation.duplicate_rows_removed": _get(split, "duplicate_rows_removed"),
        "evaluation.cv_enabled": _get(cv_config, "enabled"),
        "evaluation.cv_n_splits": _get(cv_config, "n_splits"),
        "evaluation.cv_scoring": _get(cv_config, "scoring"),
        "evaluation.cv_shuffle": _get(cv_config, "shuffle"),
        "evaluation.gap_metric": _get(gap, "metric"),
        "evaluation.gap_classification": _get(
            selected_evaluation, "f1_gap_classification"
        ),
        "evaluation.selected_model_needs_scaling": _get(
            best, "preprocessing_scaling"
        ),
    }

    metrics: Dict[str, Any] = {
        "evaluation.holdout_accuracy": _get(best, "accuracy"),
        "evaluation.holdout_precision": _get(best, "precision"),
        "evaluation.holdout_recall": _get(best, "recall"),
        "evaluation.holdout_f1_score": _get(best, "f1_score"),
        "evaluation.cv_f1_mean": _get(best, "cv_f1_mean"),
        "evaluation.cv_accuracy_mean": _get(best, "cv_accuracy_mean"),
        "evaluation.generalization_gap": _get(best, "f1_gap"),
        "evaluation.selected_training_seconds": _get(best, "total_time"),
        "evaluation.gap_largest_absolute": _get(gap, "largest_absolute_gap"),
        "evaluation.closest_threshold": _get(gap, "close_threshold"),
        "evaluation.substantial_threshold": _get(gap, "substantial_threshold"),
    }
    return (
        {key: value for key, value in params.items() if value is not None},
        {key: value for key, value in metrics.items() if value is not None},
    )


def confusion_matrix_record(state: Any) -> Dict[str, Any]:
    """Return the confusion matrix of the selected baseline, if one was stored.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.

    Returns:
        A mapping with the labels and the matrix, or an empty mapping when no
        matrix was recorded. A model that failed to evaluate has no matrix, and
        an absent matrix is reported as absent rather than as an empty grid that
        would draw as a blank figure.
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return {}
    analyses = _get(_get(agent_state, "evaluation_report"), "confusion_matrix_analysis")
    for analysis in analyses or []:
        if not _get(analysis, "available"):
            continue
        labels = [str(label) for label in (_get(analysis, "labels") or [])]
        return {
            "model_name": _get(analysis, "model_name"),
            "labels": labels,
            "total_samples": _get(analysis, "total_samples"),
            "correct_predictions": _get(analysis, "correct_predictions"),
            "incorrect_predictions": _get(analysis, "incorrect_predictions"),
            "accuracy_from_matrix": _get(analysis, "accuracy_from_matrix"),
            "class_metrics": [
                {
                    "label": _get(item, "label"),
                    "support": _get(item, "support"),
                    "precision": _get(item, "precision"),
                    "recall": _get(item, "recall"),
                    "f1_score": _get(item, "f1_score"),
                }
                for item in (_get(analysis, "class_metrics") or [])
            ],
            "dominant_confusion_pairs": [
                {
                    "true_label": _get(pair, "true_label"),
                    "predicted_label": _get(pair, "predicted_label"),
                    "count": _get(pair, "count"),
                }
                for pair in (_get(analysis, "dominant_confusion_pairs") or [])
            ],
        }
    return {}


def optimization_record(state: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Describe the Phase 6 hyperparameter search.

    MLflow does not run the search. Optuna remains the optimization engine and
    every number here was produced by it; this only copies the results into the
    tracking run.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.

    Returns:
        A ``(params, metrics, trials)`` triple. ``trials`` is the per-trial
        table, kept out of ``params`` because a 20-trial search with a large
        search space would otherwise put hundreds of characters into a field
        meant to be read in a table.
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return {}, {}, {}
    report = _get(agent_state, "optimization_report")
    if report is None:
        return {}, {}, {}

    status = _get(agent_state, "optimization_status")
    best = _get(report, "best_optimized_model")

    params: Dict[str, Any] = {
        "optimization.status": status,
        "optimization.overall_status": _get(report, "overall_status"),
        "optimization.was_performed": _get(report, "overall_status") == "completed",
        "optimization.sampler": _get(report, "sampler"),
        "optimization.sampler_seed": _get(report, "sampler_seed"),
        "optimization.objective_metric": _get(report, "objective_metric"),
        "optimization.objective_rule": _get(report, "objective_rule"),
        "optimization.selection_rule": _get(report, "selection_rule"),
        "optimization.ranking_rule": _get(report, "ranking_rule"),
        "optimization.cv_n_splits": _get(report, "cv_n_splits"),
        "optimization.n_trials_per_model": _get(report, "n_trials_per_model"),
        "optimization.top_models": _get(report, "top_models"),
        "optimization.best_model": _get(best, "best_optimized_model"),
        "optimization.best_model_type": _get(best, "model_type"),
        "optimization.best_trial_number": _get(best, "best_trial_number"),
        "optimization.total_trials": _get(best, "total_trials"),
        "optimization.best_params": _get(best, "best_params"),
        "optimization.selection_reason": _get(best, "selection_reason"),
        # The baseline is a ModelRecommendation dataclass, so its name has to be
        # pulled out rather than stringified: its repr is a whole dataclass dump
        # and would be truncated into an unreadable parameter.
        "optimization.baseline_model_name": _get(
            _get(report, "best_baseline_model"), "best_baseline_model"
        ),
        "optimization.models_searched": list(
            _get(report, "models_selected_for_optimization") or []
        ),
        "optimization.models_skipped": list(
            _get(report, "models_not_optimized") or []
        ),
    }

    metrics: Dict[str, Any] = {
        "optimization.baseline_cv_f1_mean": _get(best, "baseline_cv_f1_mean"),
        "optimization.optimized_cv_f1_mean": _get(best, "optimized_cv_f1_mean"),
        "optimization.baseline_holdout_f1_score": _get(
            best, "baseline_holdout_f1_score"
        ),
        "optimization.holdout_f1_score": _get(best, "holdout_f1_score"),
        "optimization.holdout_f1_delta": _get(best, "holdout_f1_delta"),
        "optimization.improvement": _get(best, "improvement"),
        "optimization.improvement_percentage": _get(
            best, "improvement_percentage"
        ),
        "optimization.best_trial_number": _get(best, "best_trial_number"),
        "optimization.total_trials": _get(best, "total_trials"),
        "optimization.time_seconds": _get(best, "optimization_time"),
    }

    trials: Dict[str, Any] = {"trials": []}
    for result in _get(report, "optimization_results") or []:
        for trial in _get(result, "trials") or []:
            trials["trials"].append(
                {
                    "model_name": _get(result, "model_name"),
                    "trial_number": _get(trial, "trial_number"),
                    "state": _get(trial, "state"),
                    "value": _get(trial, "objective_value"),
                    "duration_seconds": _get(trial, "duration"),
                    "params": dict(_get(trial, "params") or {}),
                }
            )
    return (
        {key: value for key, value in params.items() if value is not None},
        {key: value for key, value in metrics.items() if value is not None},
        trials,
    )


def final_model_record(state: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Describe the model Phase 9 selected and persisted.

    This reads the existing ``FinalModelSelection`` and ``PersistenceReport``.
    Phase 10 does not re-select anything, so there is deliberately no second
    selection policy here and no new ranking rule.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.

    Returns:
        A ``(params, metrics)`` pair about the persisted model.
    """
    agent_state = _state_of(state)
    if agent_state is None:
        return {}, {}
    selection = _get(
        agent_state, "final_model_selection"
    ) or _get(_get(agent_state, "persistence_report"), "selection")
    report = _get(agent_state, "persistence_report")

    params: Dict[str, Any] = {
        "final_model.name": _get(selection, "model_name"),
        "final_model.model_type": _get(selection, "model_type"),
        "final_model.selection_source": _get(selection, "selection_source"),
        "final_model.optimization_used": _get(selection, "optimization_used"),
        "final_model.optimization_status": _get(selection, "optimization_status"),
        "final_model.baseline_model_name": _get(selection, "baseline_model_name"),
        "final_model.optimized_model_name": _get(selection, "optimized_model_name"),
        "final_model.improvement_direction": _get(
            selection, "improvement_direction"
        ),
        "final_model.best_params": _get(selection, "best_params"),
        "persistence.status": _get(agent_state, "persistence_status"),
        "persistence.overall_status": _get(report, "overall_status"),
        "persistence.task_type": _get(report, "task_type"),
        "persistence.final_fit_strategy": _get(report, "final_fit_strategy"),
        "persistence.schema_version": _get(report, "schema_version"),
        # File *names* only, never a path. An absolute output directory would
        # leak the developer who ran it and would make two runs on different
        # machines look different for no useful reason.
        "persistence.model_artifact": _get(report, "model_artifact"),
        "persistence.metadata_artifact": _get(report, "metadata_artifact"),
        "persistence.results_artifact": _get(report, "results_artifact"),
    }

    metrics: Dict[str, Any] = {
        "final_model.holdout_f1_score": _get(report, "selected_f1_score"),
        "final_model.holdout_accuracy": _get(report, "selected_accuracy"),
        "final_model.feature_count": _get(report, "feature_count"),
        "final_model.class_count": _get(report, "class_count"),
        "final_model.baseline_f1_score": _get(selection, "baseline_f1_score"),
        "final_model.optimized_f1_score": _get(selection, "optimized_f1_score"),
        "final_model.improvement": _get(selection, "improvement"),
    }
    return (
        # An empty string is dropped rather than logged: Phase 9 records
        # "no optimized model" as an empty name, and logging that as a parameter
        # would read in the MLflow UI as a model literally named "".
        {key: value for key, value in params.items() if value not in (None, "")},
        {key: value for key, value in metrics.items() if value is not None},
    )


def timing_record(result: Any) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return the per-stage and total durations of a finished workflow.

    Phase 7's ``_execute_node`` already times every node with
    ``time.perf_counter`` and stores the result in ``NodeRecord.duration``, so
    this is a read of an existing measurement. No timing instrumentation was
    added and no agent logic was touched.

    Args:
        result: A :class:`WorkflowResult`, or a ``GraphState``.

    Returns:
        A ``(node_metrics, summary_metrics)`` pair.
    """
    records = []
    if result is not None and not isinstance(result, Mapping):
        records = list(getattr(result, "node_records", None) or [])
    elif isinstance(result, Mapping):
        records = list(result.get("node_records") or [])

    node_metrics: Dict[str, float] = {}
    for record in records:
        node = str(_get(record, "node", "") or "")
        duration = metric_value(_get(record, "duration"))
        if not node or duration is None:
            continue
        node_metrics[f"timing.node.{slug(node)}.seconds"] = duration

    agent_state = _state_of(result)
    summary: Dict[str, float] = {}
    total = metric_value(_get(agent_state, "execution_time"))
    if total is not None:
        summary["timing.workflow.total_seconds"] = total
    if node_metrics:
        # The sum is not the total: retries and the graph's own overhead sit
        # between the nodes, so this is recorded as its own number rather than
        # being presented as a second way of reading the total.
        summary["timing.node_sum_seconds"] = sum(node_metrics.values())
        summary["timing.node_count"] = float(len(node_metrics))
    return node_metrics, summary


def reproducibility_record(state: Any, dataset_path: Any = None) -> Dict[str, Any]:
    """Build the reproducibility record for a run.

    The version list is a fixed whitelist rather than a dump of the environment.
    A full ``pip freeze`` would record whatever else the developer happened to
    have installed, which is noise for reproduction and a disclosure risk for
    anything installed that was not meant to be.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.
        dataset_path: The dataset file, when the caller knows it.

    Returns:
        A mapping with the seeds, the versions and the platform.
    """
    agent_state = _state_of(state)
    split = _get(agent_state, "data_split")
    cv_config = _get(agent_state, "cross_validation_config")
    optimization = _get(agent_state, "optimization_report")
    path = (
        dataset_path
        if dataset_path is not None
        else _get(agent_state, "dataset_path")
    )

    versions: Dict[str, str] = {"python": platform.python_version()}
    for name in config.MLFLOW_VERSION_WHITELIST:
        if name == "python":
            continue
        try:
            versions[name] = importlib_metadata.version(name)
        except Exception:  # noqa: BLE001 - a missing extra is not a failure
            versions[name] = "not installed"

    seeds = {
        "split.random_state": _get(split, "random_state"),
        "cv.random_state": _get(cv_config, "random_state"),
        "optuna.sampler_seed": _get(optimization, "sampler_seed"),
    }
    return {
        "seeds": {key: value for key, value in seeds.items() if value is not None},
        "versions": versions,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "dataset_sha256": file_sha256(path),
        "config": {
            "PERSISTENCE_ENABLED": bool(getattr(config, "PERSISTENCE_ENABLED", False)),
            "LANGGRAPH_ENABLE_OPTIMIZATION": bool(
                getattr(config, "LANGGRAPH_ENABLE_OPTIMIZATION", False)
            ),
            "MLFLOW_EXPERIMENT_NAME": config.MLFLOW_EXPERIMENT_NAME,
            "MLFLOW_LOG_MODEL_ARTIFACT": bool(config.MLFLOW_LOG_MODEL_ARTIFACT),
            "OPTUNA_N_TRIALS": getattr(config, "OPTUNA_N_TRIALS", None),
        },
    }


# --------------------------------------------------------------------------- #
# Run tags
# --------------------------------------------------------------------------- #


def run_tags(
    state: Any,
    result: Any = None,
    *,
    workflow_status: str = "",
    enable_optimization: bool = False,
    enable_llm: bool = False,
    enable_persistence: bool = False,
) -> Dict[str, str]:
    """Build the tags describing one workflow run.

    Tags are the right place for the handful of facts that identify a run at a
    glance; everything quantitative belongs in parameters or metrics.

    Args:
        state: The ``AgentState``, or a ``GraphState`` holding one.
        result: The :class:`WorkflowResult`, when there is one.
        workflow_status: The derived workflow status, as a string.
        enable_optimization: Whether the optimization stage was permitted.
        enable_llm: Whether the optional interpretation stage was enabled.
        enable_persistence: Whether the persistence stage was permitted.

    Returns:
        A mapping of tag name to tag value.
    """
    agent_state = _state_of(state)
    tags: Dict[str, str] = {
        "project": "autonomous-multi-agent-ml",
        "workflow": "langgraph",
        "phase": "10",
        "task_type": str(_get(agent_state, "problem_type") or "unknown"),
        "dataset": str(_get(agent_state, "dataset_name") or "unknown"),
        "execution_mode": "graph",
        "optimization_enabled": str(bool(enable_optimization)).lower(),
        "llm_enabled": str(bool(enable_llm)).lower(),
        "persistence_enabled": str(bool(enable_persistence)).lower(),
        "workflow_status": str(workflow_status or "unknown"),
    }
    if result is not None and not isinstance(result, Mapping):
        tags["llm_completed_roles"] = str(
            len(list(getattr(result, "llm_completed_roles", None) or []))
        )
        tags["total_retries"] = str(int(getattr(result, "total_retries", 0) or 0))
        tags["stopped_early"] = str(bool(getattr(result, "stopped_early", False))).lower()
    return {key: str(value)[: config.MLFLOW_MAX_TAG_VALUE] for key, value in tags.items()}


# --------------------------------------------------------------------------- #
# The tracked run
# --------------------------------------------------------------------------- #


class TrackedRun:
    """One MLflow run that records a workflow run and then survives its failure.

    The class exists so that the tracking calls are made in exactly one place
    and each one is individually guarded. A failure while logging a parameter
    cannot prevent the metrics from being logged, and cannot prevent the run
    from being closed - which is what makes "a tracking failure never affects
    the machine learning result" true rather than merely intended.

    Every method returns a bool so that :func:`track_workflow_run` can build an
    honest list of what went wrong.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Configure the run without contacting MLflow yet.

        Args:
            enabled: Whether to attempt tracking at all.
            tracking_uri: Override for the tracking store.
            experiment_name: Override for the experiment name.
        """
        self.enabled = bool(enabled)
        self.tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
        self.run_id = ""
        # Set by track_workflow_run once the finished state is known, so that the
        # run is named after the dataset it actually processed.
        self.run_name_override = ""
        self._mlflow: Any = None
        self._failures: List[str] = []

    # -- store bootstrap ---------------------------------------------------- #

    def _ensure_store(self) -> None:
        """Create the tracking directory if a local file store needs one.

        A SQLite URI creates its own database file, but only if the directory
        exists, so the parent is made first. Creating it here rather than in the
        configuration keeps a read-only consumer - the dashboard - from
        creating anything.
        """
        if self._mlflow is None:
            return
        if not self.tracking_uri.lower().startswith("sqlite:"):
            return
        try:
            path = _sqlite_database_path(self.tracking_uri)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.warning(
                "%s the tracking directory could not be created: %s",
                config.MLFLOW_LOG_PREFIX,
                error,
            )

    def _client(self) -> Any:
        """Return an ``MlflowClient`` bound to this run's tracking URI.

        Returns:
            The client. Raises whatever the client raises; every caller is
            already inside a guard.
        """
        from mlflow.tracking import MlflowClient  # noqa: PLC0415 - with mlflow

        return MlflowClient(tracking_uri=self.tracking_uri)

    def make_run_name(self, state: Any = None, workflow_status: str = "") -> str:
        """Build a readable run name.

        MLflow's default is a random word triple, which makes a history list
        unreadable. The name is derived from the dataset and the task type, both
        of which are already known, and falls back to a stable default when the
        workflow never got far enough to identify them.

        Args:
            state: The finished state, when there is one.
            workflow_status: The derived workflow status.

        Returns:
            A run name such as ``sample_classification/binary``.
        """
        if self.run_name_override:
            return self.run_name_override
        agent_state = _state_of(state)
        dataset = str(_get(agent_state, "dataset_name", "") or "").strip()
        task = str(_get(agent_state, "problem_type", "") or "").strip()
        parts = [part for part in (dataset or "workflow", task) if part]
        if workflow_status:
            parts.append(workflow_status)
        return slug("-".join(parts) or "workflow", limit=120) or "workflow"

    # -- lifecycle ---------------------------------------------------------- #

    def start(self, tags: Mapping[str, str]) -> bool:
        """Point MLflow at the local store and open a run.

        Args:
            tags: The run tags, from :func:`run_tags`.

        Returns:
            True when a run is open.
        """
        if not self.enabled:
            return False
        try:
            self._mlflow = import_mlflow()
            self._ensure_store()
            self._mlflow.set_tracking_uri(self.tracking_uri)
            # MlflowClient rather than the fluent API: its signatures are the
            # ones that stayed stable across the MLflow 2.x to 3.x boundary,
            # while the fluent search_runs changed shape outright. The run itself
            # still has to be started fluently, because that is what puts it on
            # the active-run stack that log_param and log_metric read from.
            client = self._client()
            experiment = client.get_experiment_by_name(self.experiment_name)
            experiment_id = (
                client.create_experiment(self.experiment_name)
                if experiment is None
                else experiment.experiment_id
            )
            active = self._mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.make_run_name(),
                tags={key: str(value) for key, value in tags.items()},
            )
            self.run_id = str(getattr(active.info, "run_id", "") or "")
            logger.info(
                "%s run %s started in experiment '%s' (%s)",
                config.MLFLOW_LOG_PREFIX,
                self.run_id or "?",
                self.experiment_name,
                self.tracking_uri,
            )
            return bool(self.run_id)
        except TrackingError as error:
            self._fail("start", str(error))
            return False
        except Exception as error:  # noqa: BLE001 - never escapes
            self._fail("start", f"{type(error).__name__}: {error}")
            return False

    def end(self, workflow_status: str) -> bool:
        """Close the run, mapping the workflow status onto MLflow's own.

        MLflow has three terminal states and no notion of "partial", so a
        workflow that completed its required work with a degradable stage
        missing is recorded as FINISHED and the exact value is kept in the
        ``workflow_status`` tag. Collapsing it would lose the distinction
        between a clean run and one that skipped optimization.

        Args:
            workflow_status: The derived workflow status string.

        Returns:
            True when the run was closed cleanly.
        """
        if self._mlflow is None:
            return False
        try:
            status = self._run_status(workflow_status)
            self._mlflow.end_run(status=status)
            logger.info(
                "%s run %s finished (workflow_status=%s, mlflow_status=%s)",
                config.MLFLOW_LOG_PREFIX,
                self.run_id or "?",
                workflow_status,
                # end_run returns None; the mapped status is the fact worth logging.
                status,
            )
            return True
        except Exception as error:  # noqa: BLE001 - never escapes
            self._fail("end", f"{type(error).__name__}: {error}")
            return False

    @staticmethod
    def _run_status(workflow_status: str) -> str:
        """Translate a workflow status into an MLflow terminal status.

        Args:
            workflow_status: The derived workflow status.

        Returns:
            The MLflow status name: ``"FAILED"`` only for a failed workflow,
            ``"FINISHED`` otherwise.
        """
        return "FAILED" if str(workflow_status).lower() == "failed" else "FINISHED"

    # -- logging ------------------------------------------------------------ #

    def _fail(self, step: str, detail: str) -> None:
        """Record one failed tracking call.

        Args:
            step: Which operation failed.
            detail: The reason.
        """
        message = f"{step}: {detail}"
        self._failures.append(message)
        logger.warning(
            "%s tracking step '%s' failed and was skipped: %s",
            config.MLFLOW_LOG_PREFIX,
            step,
            detail,
        )

    def log_params(self, params: Mapping[str, Any]) -> int:
        """Log a flat mapping of parameters.

        Parameters are immutable inside an MLflow run, so logging the same key
        twice raises. Duplicates are dropped here rather than caught from the
        client, which keeps the first value - the one the workflow actually
        decided.

        Args:
            params: The parameters to log.

        Returns:
            The number of parameters logged.
        """
        if self._mlflow is None:
            return 0
        logged = 0
        seen: set = set()
        for raw_key, raw_value in dict(params).items():
            key = slug(str(raw_key), limit=250)
            value = param_value(raw_value)
            if not key or value is None or key in seen:
                continue
            seen.add(key)
            try:
                self._mlflow.log_param(key, value)
                logged += 1
            except Exception as error:  # noqa: BLE001 - one bad param, not a lost run
                self._fail(f"param {key}", f"{type(error).__name__}: {error}")
        return logged

    def log_metrics(self, metrics: Mapping[str, Any]) -> int:
        """Log a flat mapping of metrics.

        Args:
            metrics: The metrics to log.

        Returns:
            The number of metrics logged.
        """
        if self._mlflow is None:
            return 0
        logged = 0
        for raw_key, raw_value in dict(metrics).items():
            key = slug(str(raw_key), limit=250)
            value = metric_value(raw_value)
            if not key or value is None:
                continue
            try:
                self._mlflow.log_metric(key, value)
                logged += 1
            except Exception as error:  # noqa: BLE001 - one bad metric, not a lost run
                self._fail(f"metric {key}", f"{type(error).__name__}: {error}")
        return logged

    def set_tags(self, tags: Mapping[str, Any]) -> int:
        """Set tags on the run.

        Unlike parameters, tags may be set repeatedly and the last value wins,
        so no duplicate handling is needed.

        Args:
            tags: The tags to set.

        Returns:
            The number of tags set.
        """
        if self._mlflow is None:
            return 0
        logged = 0
        for raw_key, raw_value in dict(tags).items():
            key = slug(str(raw_key), limit=250)
            if not key:
                continue
            try:
                self._mlflow.set_tag(
                    key, str(raw_value)[: config.MLFLOW_MAX_TAG_VALUE]
                )
                logged += 1
            except Exception as error:  # noqa: BLE001 - never escapes
                self._fail(f"tag {key}", f"{type(error).__name__}: {error}")
        return logged

    def log_json(self, name: str, payload: Any) -> bool:
        """Write a JSON document into the run's artifact directory.

        ``mlflow.log_dict`` is used rather than writing a temporary file and
        calling ``log_artifact``: the payload never touches the filesystem of
        this project, which means logging an artifact cannot disturb ``models/``
        or ``datasets/`` even in principle.

        Args:
            name: The artifact file name.
            payload: A JSON-serialisable object.

        Returns:
            True when the document was written.
        """
        if self._mlflow is None:
            return False
        if not name:
            return False
        try:
            self._mlflow.log_dict(
                _jsonable(payload), artifact_file=name, run_id=self.run_id or None
            )
            return True
        except Exception as error:  # noqa: BLE001 - never escapes
            self._fail(f"artifact {name}", f"{type(error).__name__}: {error}")
            return False

    def log_pipeline(self, pipeline: Any, artifact_path: str) -> bool:
        """Copy the fitted pipeline into the run's artifact directory.

        Off by default (``MLFLOW_LOG_MODEL_ARTIFACT`` is False). Phase 9's
        ``models/best_model.joblib`` stays the primary deployment artifact: a
        second serialized copy would be a second thing to keep in sync and a
        second serialization format the project would otherwise not depend on.
        The metadata and metrics describing the model are logged either way.

        Args:
            pipeline: The fitted pipeline.
            artifact_path: Where to put it inside the run.

        Returns:
            True when it was logged, False when it was disabled or failed.
        """
        if not config.MLFLOW_LOG_MODEL_ARTIFACT or self._mlflow is None:
            return False
        if pipeline is None:
            return False
        try:
            import joblib  # noqa: PLC0415 - only needed on this opt-in path

            handle, temporary = tempfile.mkstemp(prefix="mlflow-model-", suffix=".joblib")
            os.close(handle)
            try:
                joblib.dump(pipeline, temporary)
                self._mlflow.log_artifact(
                    temporary, artifact_path=artifact_path, run_id=self.run_id or None
                )
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
            return True
        except Exception as error:  # noqa: BLE001 - never escapes
            self._fail(f"model artifact {artifact_path}", f"{type(error).__name__}: {error}")
            return False


def _jsonable(value: Any) -> Any:
    """Make a value safe to hand to ``mlflow.log_dict``.

    Args:
        value: Any structure produced by the extractors.

    Returns:
        A structure containing only dicts, lists, strings, numbers, booleans and
        None. A value that cannot be represented becomes its ``repr``, which is
        lossy but visible - better than an artifact that fails to write.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        try:
            return _jsonable(item())
        except Exception:  # noqa: BLE001 - fall through to the generic path
            pass
    try:
        from utils.state import to_serializable  # noqa: PLC0415 - avoids a cycle at import

        return _jsonable(to_serializable(value))
    except Exception:  # noqa: BLE001 - never fails
        return repr(value)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def track_workflow_run(
    state: Any = None,
    result: Any = None,
    *,
    dataset_path: Any = None,
    workflow_status: str = "",
    enable_optimization: bool = False,
    enable_llm: bool = False,
    enable_persistence: bool = False,
    enabled: Optional[bool] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> TrackingSummary:
    """Record one finished workflow run in MLflow.

    This is the only function the workflow calls. It never raises: a tracking
    problem is reported through the returned :class:`TrackingSummary` and one
    warning in the log, and the machine learning result is untouched.

    Args:
        state: The finished ``AgentState`` or ``GraphState``.
        result: The :class:`WorkflowResult`, when there is one.
        dataset_path: The dataset file, when the caller knows it.
        workflow_status: The derived workflow status, as a string.
        enable_optimization: Whether the optimization stage was permitted.
        enable_llm: Whether the optional interpretation stage was enabled.
        enable_persistence: Whether the persistence stage was permitted.
        enabled: Override for ``config.MLFLOW_ENABLED``.
        tracking_uri: Override for the tracking store.
        experiment_name: Override for the experiment name.

    Returns:
        A :class:`TrackingSummary` describing what was recorded.
    """
    if enabled is None:
        enabled = bool(config.MLFLOW_ENABLED)
    summary = TrackingSummary(
        experiment_name=experiment_name or config.MLFLOW_EXPERIMENT_NAME,
        tracking_uri=tracking_uri or config.MLFLOW_TRACKING_URI,
    )
    if not enabled:
        summary.status = TrackingStatus.DISABLED
        summary.reason = "MLflow tracking is disabled by configuration."
        logger.info("%s tracking disabled; the run was not recorded", config.MLFLOW_LOG_PREFIX)
        return summary

    tracked = TrackedRun(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    tracked.run_name_override = tracked.make_run_name(state, workflow_status)
    summary.experiment_name = tracked.experiment_name
    summary.tracking_uri = tracked.tracking_uri

    if not mlflow_available():
        summary.status = TrackingStatus.UNAVAILABLE
        summary.reason = (
            "mlflow is not installed, so no experiment was created. Install it with "
            "'pip install mlflow' or turn tracking off explicitly with --no-mlflow."
        )
        logger.warning("%s %s", config.MLFLOW_LOG_PREFIX, summary.reason)
        return summary

    tags = run_tags(
        state,
        result,
        workflow_status=workflow_status,
        enable_optimization=enable_optimization,
        enable_llm=enable_llm,
        enable_persistence=enable_persistence,
    )
    if not tracked.start(tags):
        summary.status = TrackingStatus.FAILED
        summary.reason = "The MLflow run could not be started."
        summary.failures = list(tracked._failures)
        return summary

    summary.run_id = tracked.run_id

    try:
        _log_workflow_payload(
            tracked,
            state=state,
            result=result,
            dataset_path=dataset_path,
            workflow_status=workflow_status,
        )
    except Exception as error:  # noqa: BLE001 - an extractor's bug must not escape
        # Each individual logging call is already guarded, so reaching here means
        # a bug in one of the pure extractors rather than a tracking outage. It
        # is still recorded as a tracking failure and never raised: the caller
        # asked for observability and must get its machine learning result
        # either way.
        logger.warning(
            "%s tracking step 'payload' failed and was skipped: %s: %s. The "
            "machine learning results are unaffected.",
            config.MLFLOW_LOG_PREFIX,
            type(error).__name__,
            error,
        )
        tracked._failures.append(f"payload: {type(error).__name__}: {error}")
    finally:
        tracked.end(workflow_status)

    summary.failures = list(tracked._failures)
    if summary.failures:
        summary.status = TrackingStatus.FAILED
        summary.reason = (
            f"{len(summary.failures)} tracking operation(s) failed; the machine "
            "learning results are unaffected."
        )
    else:
        summary.status = TrackingStatus.RECORDED
        summary.reason = ""
    return summary


def _log_workflow_payload(
    tracked: TrackedRun,
    *,
    state: Any,
    result: Any,
    dataset_path: Any,
    workflow_status: str,
) -> None:
    """Log everything about one workflow run.

    Split out from :func:`track_workflow_run` so that the sequence of logging
    calls is visible in one place, and so a failure in the middle of it cannot
    skip the closing of the run.

    Args:
        tracked: The open run.
        state: The finished state.
        result: The finished workflow result.
        dataset_path: The dataset file, when known.
        workflow_status: The derived workflow status.
    """
    params: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}

    params.update(dataset_record(state, dataset_path))
    params.update(preprocessing_record(state))

    rows = comparison_rows(state)
    best_row = rows[0] if rows else {}
    if best_row:
        params["comparison.model_count"] = len(rows)
        params["comparison.best_model_name"] = best_row.get("model_name")
        params["comparison.ranking_rule"] = _get(
            _get(_state_of(state), "evaluation_report"), "ranking_rule"
        )

    evaluation_params, evaluation_metrics = evaluation_record(state)
    params.update(evaluation_params)
    metrics.update(evaluation_metrics)

    optimization_params, optimization_metrics, trials = optimization_record(state)
    params.update(optimization_params)
    metrics.update(optimization_metrics)

    final_params, final_metrics = final_model_record(state)
    params.update(final_params)
    metrics.update(final_metrics)

    node_timing, summary_timing = timing_record(result)
    metrics.update(node_timing)
    metrics.update(summary_timing)

    metrics["run.models_compared"] = float(len(rows))

    tracked.log_params(params)
    tracked.log_metrics(metrics)

    # The exact workflow status also travels as a tag, because MLflow's own
    # terminal status cannot express "partial".
    tracked.set_tags({"workflow_status": workflow_status or "unknown"})

    matrix = confusion_matrix_record(state)
    if matrix:
        tracked.log_json(config.MLFLOW_ARTIFACT_CONFUSION_MATRIX, matrix)
    if rows:
        tracked.log_json(
            config.MLFLOW_ARTIFACT_MODEL_COMPARISON,
            {"ranking": "recorded rank, then hold-out f1 descending", "models": rows},
        )
    if trials.get("trials"):
        tracked.log_json(config.MLFLOW_ARTIFACT_OPTIMIZATION_TRIALS, trials)

    dataset_info = dataset_record(state, dataset_path)
    if dataset_info:
        tracked.log_json(config.MLFLOW_ARTIFACT_DATASET, dataset_info)
    pre_config = preprocessing_record(state)
    if pre_config:
        tracked.log_json(config.MLFLOW_ARTIFACT_PREPROCESSING, pre_config)

    reproducibility = reproducibility_record(state, dataset_path)
    tracked.log_json(
        config.MLFLOW_ARTIFACT_REPRODUCIBILITY,
        {**reproducibility, "workflow_status": workflow_status},
    )


def tracking_configuration() -> Dict[str, Any]:
    """Describe how tracking is configured right now.

    Used by the dashboard and by the tests, so that "is tracking on?" is
    answered by one function rather than by three different guesses.

    Returns:
        The enabled flag, the tracking URI, the experiment name and whether the
        mlflow package can be imported.
    """
    return {
        "enabled": bool(config.MLFLOW_ENABLED),
        "tracking_uri": config.MLFLOW_TRACKING_URI,
        "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
        "available": mlflow_available(),
        "history_limit": int(config.MLFLOW_HISTORY_LIMIT),
    }


# --------------------------------------------------------------------------- #
# Read side, for the dashboard
# --------------------------------------------------------------------------- #
#
# Nothing below starts a run, writes a run or touches the machine learning
# pipeline. It only reads what a previous run recorded, so a dashboard rerun can
# neither train a model nor add a run to the history it is displaying.


def _resolve_experiment_id(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Optional[str]:
    """Return the id of an experiment, if it already exists.

    Args:
        tracking_uri: The tracking store to read. Defaults to the configured one.
        experiment_name: The experiment to find. Defaults to the configured one.

    Returns:
        The experiment id, or None when the experiment has never been created or
        the tracking store cannot be read. Both are ordinary states for a project
        whose tracking is off, so neither raises.
    """
    try:
        import_mlflow()
        from mlflow.tracking import MlflowClient  # noqa: PLC0415 - with mlflow

        client = MlflowClient(
            tracking_uri=tracking_uri or config.MLFLOW_TRACKING_URI
        )
        experiment = client.get_experiment_by_name(
            experiment_name or config.MLFLOW_EXPERIMENT_NAME
        )
    except TrackingError:
        return None
    except Exception as error:  # noqa: BLE001 - a read failure is reportable
        logger.warning(
            "%s the tracking store could not be read: %s: %s",
            config.MLFLOW_LOG_PREFIX,
            type(error).__name__,
            error,
        )
        return None
    if experiment is None:
        return None
    return str(getattr(experiment, "experiment_id", "") or "") or None


def _sqlite_database_path(tracking_uri: str) -> Optional[Path]:
    """Return the file a ``sqlite:`` tracking URI points at.

    Args:
        tracking_uri: The tracking URI.

    Returns:
        The database path, or None when the URI is not a local SQLite one. A
        remote or ``file:`` URI is deliberately not parsed: MLflow owns those
        formats, and guessing at them here would be a second, wrong
        implementation of the same thing.
    """
    remainder = str(tracking_uri or "")[len("sqlite:") :]
    if not remainder.startswith("///"):
        return None
    raw = remainder[len("///") :]
    if not raw:
        return None
    # After the three slashes, what is left is a filesystem path. On Windows the
    # usual form is ``sqlite:///C:/dir/db.db`` (a drive letter, no leading
    # slash); on POSIX it is ``sqlite:////abs/dir/db.db``. Forward slashes are
    # accepted by pathlib on every platform, so no substitution is needed.
    return Path(raw)


def recent_runs(
    limit: Optional[int] = None,
    *,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the most recent tracked runs, newest first.

    Bounded by ``MLFLOW_HISTORY_LIMIT`` because the dashboard is a report, not an
    MLflow replacement, and an unbounded table would slow the page down the more
    experiments a project accumulated.

    Args:
        limit: How many runs to return. None uses the configured limit.
        tracking_uri: Override for the tracking store. This exists for tests that
            must read a temporary store rather than the project's own; the
            dashboard never passes it, so a page render cannot be pointed at an
            arbitrary server.
        experiment_name: Override for the experiment name.

    Returns:
        A list of small dictionaries, one per run. Empty when tracking is off,
        unavailable, or nothing has been recorded yet. Never raises.
    """
    maximum = int(limit if limit is not None else config.MLFLOW_HISTORY_LIMIT)
    if maximum <= 0 or not config.MLFLOW_ENABLED:
        return []
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
    experiment_id = _resolve_experiment_id(uri, name)
    if experiment_id is None:
        return []

    try:
        import_mlflow()
        from mlflow.tracking import MlflowClient  # noqa: PLC0415 - with mlflow

        # MlflowClient, not mlflow.search_runs: the fluent function returns a
        # pandas DataFrame in MLflow 3.x, which would make pandas a hard
        # dependency of a module that must work when tracking is unavailable.
        client = MlflowClient(tracking_uri=uri)
        found = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=maximum,
        )
        runs = list(getattr(found, "runs", []) or found or [])[:maximum]
    except Exception as error:  # noqa: BLE001 - a read failure is reportable
        logger.warning(
            "%s the run history could not be read: %s: %s",
            config.MLFLOW_LOG_PREFIX,
            type(error).__name__,
            error,
        )
        return []

    rows: List[Dict[str, Any]] = []
    for run in runs:
        data = getattr(run, "data", None)
        tags = dict(getattr(data, "tags", None) or {})
        params = dict(getattr(data, "params", None) or {})
        row_metrics = dict(getattr(data, "metrics", None) or {})
        # start_time is epoch milliseconds and lives on RunInfo, not RunData.
        timestamp = getattr(getattr(run, "info", None), "start_time", None)
        rows.append(
            {
                "run_id": str(getattr(run.info, "run_id", "") or ""),
                "run_name": str(getattr(run.info, "run_name", "") or ""),
                "status": str(getattr(run.info, "status", "") or ""),
                "start_time": _format_timestamp(timestamp),
                "workflow_status": tags.get("workflow_status", ""),
                "task_type": tags.get("task_type", ""),
                "dataset": tags.get("dataset", ""),
                "optimization_enabled": tags.get("optimization_enabled", ""),
                "persistence_enabled": tags.get("persistence_enabled", ""),
                "model": params.get("final_model.name", ""),
                "f1_score": _optional_float(row_metrics.get("final_model.holdout_f1_score")),
                "accuracy": _optional_float(row_metrics.get("final_model.holdout_accuracy")),
                "optimization_status": params.get("optimization.status", ""),
            }
        )
    return rows


def experiment_summary(
    *,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarise the tracking state for the dashboard.

    This is the read-side contract the dashboard depends on. It always returns a
    mapping with the same keys, so the view never has to guess whether a field
    is missing or whether tracking was merely switched off.

    Args:
        tracking_uri: Override for the tracking store. The dashboard never passes
            this; it exists so a test can read a temporary store.
        experiment_name: Override for the experiment name.

    Returns:
        A mapping with ``available``, ``enabled``, ``installed``,
        ``tracking_uri``, ``experiment_name``, ``latest`` (the newest run or an
        empty mapping) and ``runs`` (the bounded history). Never raises.
    """
    configuration = tracking_configuration()
    summary: Dict[str, Any] = {
        "available": False,
        "enabled": configuration["enabled"],
        "installed": configuration["available"],
        "tracking_uri": tracking_uri or configuration["tracking_uri"],
        "experiment_name": experiment_name or configuration["experiment_name"],
        "latest": {},
        "runs": [],
        "note": config.DASHBOARD_NO_EXPERIMENT_TRACKING,
    }
    if not configuration["enabled"]:
        return summary
    if not configuration["available"]:
        summary["note"] = (
            "MLflow tracking is unavailable or disabled because the mlflow package "
            "is not installed. Every other section is read from the Phase 9 "
            "artifacts and is unaffected."
        )
        return summary

    runs = recent_runs(tracking_uri=tracking_uri, experiment_name=experiment_name)
    summary["runs"] = runs
    if not runs:
        summary["note"] = (
            "MLflow tracking is enabled and no run has been recorded yet. Run "
            "'python app.py --graph' to create the first one."
        )
        return summary

    summary["available"] = True
    summary["latest"] = runs[0]
    summary["note"] = ""
    return summary


def _optional_float(value: Any) -> Optional[float]:
    """Parse a metric value that may be absent.

    Args:
        value: The raw value from the tracking store.

    Returns:
        The float, or None when it is missing or not a number.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_timestamp(value: Any) -> str:
    """Render an MLflow timestamp as an ISO-8601 UTC string.

    Args:
        value: The timestamp from the tracking store.

    Returns:
        An ISO-8601 string, or an empty string when there is no usable
        timestamp. Never raises, because a dashboard must not die on a
        formatting problem in a value it only displays.
    """
    if value is None:
        return ""
    try:
        from datetime import datetime, timezone  # noqa: PLC0415 - tiny and local

        seconds = int(value) / 1000.0
        return (
            datetime.fromtimestamp(seconds, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except Exception:  # noqa: BLE001 - display only
        return ""


__all__ = [
    "TrackedRun",
    "TrackingError",
    "TrackingStatus",
    "TrackingSummary",
    "comparison_rows",
    "confusion_matrix_record",
    "dataset_record",
    "evaluation_record",
    "experiment_summary",
    "file_sha256",
    "final_model_record",
    "import_mlflow",
    "metric_value",
    "mlflow_available",
    "optimization_record",
    "param_value",
    "preprocessing_record",
    "recent_runs",
    "reproducibility_record",
    "reset_mlflow_cache",
    "run_tags",
    "slug",
    "timing_record",
    "track_workflow_run",
    "tracking_configuration",
]