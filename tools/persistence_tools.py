"""Reusable model persistence operations (Phase 9).

Everything that reads or writes a model artifact lives here, so the Persistence
Agent stays a thin orchestration layer and the dashboard has exactly one place to
go for "load the model this project saved".

Three rules shape this module, and all three are load-bearing:

**Persistence adds no machine learning.** The pipeline that gets written is the
one Phase 4 or Phase 6 already fitted and scored. This module selects between two
candidates using the *existing* ranking metrics (``f1_score`` then ``accuracy``,
the rule Phase 4, Phase 5 and Phase 6 already use), never recomputes a metric and
never re-runs a search. :func:`select_final_model` is the only place a choice is
made, and the reason it made it is recorded next to it.

**The final fit happens after the final selection.** :func:`build_final_pipeline`
either reuses the pipeline that is already fitted or rebuilds the estimator from
the selected parameters and refits it on the same training split Phase 4 used.
The hold-out rows are never fitted on, so the metrics recorded in the metadata
still describe the model that was saved.

**Only the project's own artifact is ever loaded.** :func:`load_persisted_model`
derives its path from :mod:`utils.config` and accepts no caller-supplied file.
A ``.joblib`` file is a pickle, and unpickling executes arbitrary code, so a
dashboard that accepted an uploaded model would be an arbitrary-code-execution
hole. This module is therefore also the only place in the project that calls
``joblib.load``.

Writes are atomic: the payload goes to a temporary file in the destination
directory and is then moved into place, so a crash mid-write can never leave a
half-written artifact behind for the dashboard to load.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib

from tools import model_tools
from tools.optimization_tools import build_tuned_pipeline
from tools.preprocessing_tools import remove_duplicate_rows
from tools.training_tools import (
    build_model_pipeline,
    load_training_dataframe,
    prepare_features_and_target,
    split_dataset,
)
from utils import config
from utils.logger import setup_logger
from utils.state import (
    FinalModelSelection,
    FinalModelSource,
    OptimizationStatus,
    to_serializable,
)

logger = setup_logger("tools.persistence_tools")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class PersistenceError(Exception):
    """Base class for model persistence errors."""


class NoModelToPersistError(PersistenceError):
    """Raised when the state carries no model that could be persisted.

    This is a deterministic problem: there is nothing to write, and writing
    something else would be a lie about the run.
    """


class ArtifactWriteError(PersistenceError):
    """Raised when an artifact could not be written."""


class ArtifactReadError(PersistenceError):
    """Raised when an artifact is missing, unreadable or malformed."""


# --------------------------------------------------------------------------- #
# Local records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArtifactPaths:
    """The three files Phase 9 writes, resolved to absolute paths.

    The directory is resolved once, at construction, so every writer and reader in
    a run agrees on the destination even if the configuration changed midway.
    """

    directory: Path
    model: Path
    metadata: Path
    results: Path

    def relative_names(self) -> Dict[str, str]:
        """Return the artifact names, without the directory prefix.

        The state stores these instead of the absolute paths so a report written
        on one machine stays meaningful on another.

        Returns:
            A mapping of ``"model"``, ``"metadata"`` and ``"results"`` to the file
            name each one uses inside the output directory.
        """
        return {
            "model": self.model.name,
            "metadata": self.metadata.name,
            "results": self.results.name,
        }


@dataclass(frozen=True)
class PersistedModel:
    """A loaded artifact: the fitted pipeline and the metadata describing it."""

    pipeline: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    paths: Optional[ArtifactPaths] = None

    @property
    def model_name(self) -> str:
        """Registry name of the persisted model, e.g. ``"Random Forest"``."""
        return str(self.metadata.get("model_name", "") or "")

    @property
    def target_column(self) -> str:
        """Name of the column the model predicts."""
        return str(self.metadata.get("target_column", "") or "")

    @property
    def class_labels(self) -> List[str]:
        """Class labels the model was trained on, in the stored order."""
        labels = self.metadata.get("class_labels") or []
        return [str(item) for item in labels]

    @property
    def numerical_features(self) -> List[str]:
        """Numerical feature names, in the order the preprocessor expects."""
        return [str(item) for item in (self.metadata.get("numerical_features") or [])]

    @property
    def categorical_features(self) -> List[str]:
        """Categorical feature names, in the order the preprocessor expects."""
        return [str(item) for item in (self.metadata.get("categorical_features") or [])]

    @property
    def feature_columns(self) -> List[str]:
        """Every column the pipeline consumes, numerical ones first."""
        return self.numerical_features + self.categorical_features

    @property
    def supports_probability(self) -> bool:
        """True when the saved estimator can return class probabilities."""
        estimator = getattr(self.pipeline, "named_steps", {}).get("model")
        return hasattr(estimator, "predict_proba")


@dataclass(frozen=True)
class FinalPipeline:
    """The pipeline that will be written, and how it was obtained."""

    pipeline: Any
    strategy: str
    note: str = ""
    train_samples: int = 0


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def resolve_output_dir(output_dir: Optional[Path] = None) -> Path:
    """Return the directory the artifacts are written to.

    Args:
        output_dir: Explicit directory, or None to use the configured one.

    Returns:
        The resolved output directory. The directory itself is **not** created
        here: creating it is the writer's job, so a read-only check cannot have
        the side effect of making a directory.
    """
    if output_dir is not None:
        return Path(output_dir).resolve()
    if config.PERSISTENCE_OUTPUT_DIR is not None:
        return Path(config.PERSISTENCE_OUTPUT_DIR).resolve()
    return Path(config.MODELS_DIR).resolve()


def artifact_paths(output_dir: Optional[Path] = None) -> ArtifactPaths:
    """Return the three artifact paths for one output directory.

    Args:
        output_dir: Explicit output directory, or None for the configured one.

    Returns:
        The resolved :class:`ArtifactPaths`.
    """
    directory = resolve_output_dir(output_dir)
    return ArtifactPaths(
        directory=directory,
        model=directory / config.PERSISTENCE_MODEL_FILENAME,
        metadata=directory / config.PERSISTENCE_METADATA_FILENAME,
        results=directory / config.PERSISTENCE_RESULTS_FILENAME,
    )


def artifacts_exist(output_dir: Optional[Path] = None) -> bool:
    """Return True when both the pipeline and its metadata are on disk.

    A pipeline without metadata would leave the dashboard unable to describe what
    it loaded, so the pair is the unit of "an artifact exists".

    Args:
        output_dir: Explicit output directory, or None for the configured one.

    Returns:
        True when the model file and the metadata file both exist.
    """
    paths = artifact_paths(output_dir)
    return paths.model.is_file() and paths.metadata.is_file()


# --------------------------------------------------------------------------- #
# Candidate resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One model that could become the final model, with the scores it has."""

    model_name: str
    model_type: str = ""
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when the candidate carries the primary metric.

        A candidate without it cannot be ranked under the project's rule, and
        ranking it anyway would mean inventing a score.
        """
        return bool(self.model_name) and self.f1_score is not None

    def ranking_key(self) -> Tuple[float, float, str]:
        """Return the project's documented sort key for this candidate.

        Primary metric descending, secondary metric descending, name ascending.
        This is the same rule as :func:`tools.training_tools.rank_experiments` and
        :func:`tools.optimization_tools.rank_optimized_results`; the values are
        negated so a plain tuple comparison picks the winner.
        """
        return (
            -float(self.f1_score if self.f1_score is not None else 0.0),
            -float(self.accuracy if self.accuracy is not None else 0.0),
            str(self.model_name),
        )


def resolve_model_type(agent_state: Any, model_name: str) -> str:
    """Return the registry family of a model name.

    ``BestModelInfo`` (Phase 4/5) does not carry the family, so the name is
    resolved from the training experiments, which do. The model registry is the
    last resort, so an unknown name yields an empty string rather than a guess.

    Args:
        agent_state: The authoritative machine learning state.
        model_name: Registry name, e.g. ``"Support Vector Machine"``.

    Returns:
        The :class:`~utils.state.ModelType` value, or an empty string.
    """
    for experiment in getattr(agent_state, "experiment_results", None) or []:
        if str(getattr(experiment, "model_name", "")) == str(model_name):
            model_type = str(getattr(experiment, "model_type", "") or "")
            if model_type:
                return model_type
    try:
        return str(model_tools.get_model_definition(str(model_name)).model_type)
    except Exception:  # noqa: BLE001 - an unknown name is reported, not raised
        return ""


def baseline_candidate(agent_state: Any) -> Optional[Candidate]:
    """Return the best baseline model as a final-model candidate.

    Reads only what Phase 4/5 already recorded: ``best_baseline_model`` and, when
    present, the Phase 5 recommendation. Nothing is recomputed.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        A :class:`Candidate`, or None when no baseline produced metrics.
    """
    best = getattr(agent_state, "best_baseline_model", None)
    if best is None or not str(getattr(best, "model_name", "")):
        return None

    name = str(best.model_name)
    accuracy = getattr(best, "accuracy", None)
    cv_f1 = getattr(best, "cv_f1_mean", None)
    recommendation = None
    report = getattr(agent_state, "evaluation_report", None)
    if report is not None:
        recommendation = report.best_model

    return Candidate(
        model_name=name,
        model_type=(
            str(getattr(recommendation, "model_type", "") or "")
            if recommendation is not None and str(recommendation.model_type)
            else resolve_model_type(agent_state, name)
        ),
        f1_score=getattr(best, "f1_score", None),
        accuracy=accuracy,
        cv_f1_mean=cv_f1,
        parameters=dict(getattr(best, "parameters", None) or {}),
    )


def optimization_result_for(agent_state: Any, model_name: str) -> Optional[Any]:
    """Return the Phase 6 ``OptimizationResult`` for one model name.

    The Phase 6 *recommendation* (``OptimizedModelRecommendation``) is a summary and
    deliberately carries only the numbers needed to explain the choice. The full
    result record holds more - most importantly ``holdout_accuracy`` - so the two
    are joined here whenever a metric is needed.

    Args:
        agent_state: The authoritative machine learning state.
        model_name: Registry name of the tuned model.

    Returns:
        The matching `OptimizationResult`, or None when the model was not tuned.
    """
    report = getattr(agent_state, "optimization_report", None)
    if report is None:
        return None
    for item in getattr(report, "optimization_results", None) or []:
        if str(getattr(item, "model_name", "")) == str(model_name):
            return item
    return None


def optimized_candidate(agent_state: Any) -> Optional[Candidate]:
    """Return the best tuned model as a final-model candidate.

    Reads the Phase 6 recommendation and joins it with the matching
    ``OptimizationResult``, because the recommendation alone carries no
    ``holdout_accuracy`` and the documented ranking needs that value as its
    secondary metric. A missing recommendation is the normal "optimization did not
    produce a usable model" answer rather than an error: it is only built when at
    least one model was tuned *and* scored on the hold-out split.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        A :class:`Candidate`, or None when optimization produced no usable winner.
    """
    report = getattr(agent_state, "optimization_report", None)
    if report is None:
        return None
    winner = getattr(report, "best_optimized_model", None)
    if winner is None or not str(getattr(winner, "best_optimized_model", "")):
        return None

    name = str(winner.best_optimized_model)
    detail = optimization_result_for(agent_state, name)
    accuracy = getattr(detail, "holdout_accuracy", None)
    if accuracy is None:
        # An older or partial Phase 6 record: fall back to the accuracy the same
        # model reached as a baseline, which is at least a real measurement rather
        # than an invented one. The reason is reported by `select_final_model`.
        accuracy = next(
            (
                getattr(experiment, "accuracy", None)
                for experiment in getattr(agent_state, "experiment_results", None) or []
                if str(getattr(experiment, "model_name", "")) == name
            ),
            None,
        )

    return Candidate(
        model_name=name,
        model_type=str(getattr(winner, "model_type", "") or ""),
        f1_score=getattr(winner, "holdout_f1_score", None),
        accuracy=accuracy,
        cv_f1_mean=getattr(winner, "optimized_cv_f1_mean", None),
        parameters=dict(getattr(winner, "best_params", None) or {}),
    )


def select_final_model(agent_state: Any) -> FinalModelSelection:
    """Choose which model to persist, deterministically.

    The policy, in full:

    1. Collect the two candidates that already exist: the Phase 4/5 baseline
       winner and the Phase 6 tuned winner.
    2. If the tuned winner is missing or carries no hold-out F1, persist the
       baseline. This is the disabled, skipped and failed search case.
    3. Otherwise compare the two with the project's own ranking key - hold-out
       ``f1_score`` descending, then hold-out ``accuracy`` descending, then the
       model name ascending - and persist the better one.

    No new metric, threshold or tie-break is introduced: the key is the same one
    :func:`tools.training_tools.rank_experiments` and
    :func:`tools.optimization_tools.rank_optimized_results` use.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        The recorded :class:`~utils.state.FinalModelSelection`.

    Raises:
        NoModelToPersistError: Neither candidate carries the primary metric, so
            there is nothing that could honestly be written to disk.
    """
    baseline = baseline_candidate(agent_state)
    optimized = optimized_candidate(agent_state)

    optimization_status = str(getattr(agent_state, "optimization_status", "") or "")
    if optimized is not None and not optimized.usable:
        logger.warning(
            "%s the tuned model '%s' has no hold-out f1_score, so the baseline is used",
            config.PERSISTENCE_LOG_PREFIX,
            optimized.model_name,
        )
        optimized = None

    usable_baseline = baseline if baseline is not None and baseline.usable else None
    usable_optimized = optimized if optimized is not None and optimized.usable else None

    if usable_baseline is None and usable_optimized is None:
        raise NoModelToPersistError(
            "There is no model to persist: the state carries neither a usable best "
            "baseline model nor a usable tuned model with a hold-out f1_score. The "
            "Evaluation Agent has to complete before a model can be saved."
        )

    selection = FinalModelSelection(
        selection_rule=config.PERSISTENCE_SELECTION_RULE,
        optimization_used=usable_optimized is not None,
        optimization_status=optimization_status,
    )

    if usable_baseline is not None:
        selection.baseline_model_name = usable_baseline.model_name
        selection.baseline_f1_score = usable_baseline.f1_score
        selection.baseline_accuracy = usable_baseline.accuracy
        selection.baseline_cv_f1_mean = usable_baseline.cv_f1_mean
    if usable_optimized is not None:
        selection.optimized_model_name = usable_optimized.model_name
        selection.optimized_f1_score = usable_optimized.f1_score
        selection.optimized_accuracy = usable_optimized.accuracy
        selection.optimized_cv_f1_mean = usable_optimized.cv_f1_mean

    if usable_optimized is None:
        winner = usable_baseline
        source = FinalModelSource.BASELINE
        selection.optimization_used = False
    elif usable_baseline is None:
        winner = usable_optimized
        source = FinalModelSource.OPTIMIZED
    elif usable_optimized.ranking_key() <= usable_baseline.ranking_key():
        winner = usable_optimized
        source = FinalModelSource.OPTIMIZED
    else:
        winner = usable_baseline
        source = FinalModelSource.BASELINE
        selection.best_params = dict(usable_optimized.parameters)

    selection.model_name = winner.model_name
    selection.model_type = winner.model_type or resolve_model_type(agent_state, winner.model_name)
    selection.selection_source = source.value
    selection.best_params = dict(winner.parameters)
    if source is FinalModelSource.BASELINE:
        selection.optimization_used = False

    selection.selection_reason = _selection_reason(selection, baseline, optimized)

    if selection.optimization_used:
        improvement = None
        if (
            selection.optimized_f1_score is not None
            and selection.baseline_cv_f1_mean is not None
        ):
            improvement = round(
                float(selection.optimized_cv_f1_mean) - float(selection.baseline_cv_f1_mean),
                6,
            )
        selection.improvement = improvement
        report = getattr(agent_state, "optimization_report", None)
        winner_record = None
        if report is not None:
            for item in getattr(report, "optimization_results", None) or []:
                if str(getattr(item, "model_name", "")) == selection.optimized_model_name:
                    winner_record = item
                    break
        if winner_record is not None:
            selection.improvement_direction = str(
                getattr(winner_record, "improvement_direction", "") or ""
            ) or selection.improvement_direction
            selection.optimization_used = True

    return selection


def _selection_reason(
    selection: FinalModelSelection,
    baseline: Optional[Candidate],
    optimized: Optional[Candidate],
) -> str:
    """Write the one-line justification for the recorded choice.

    Args:
        selection: The selection being built.
        baseline: The baseline candidate, or None.
        optimized: The tuned candidate, or None.

    Returns:
        A sentence naming the rule, the winner and the score it beat.
    """
    if selection.selection_source == FinalModelSource.OPTIMIZED.value:
        base = (
            f"optimized {selection.model_name} (hold-out f1_score "
            f"{_fmt(selection.optimized_f1_score)}, accuracy "
            f"{_fmt(selection.optimized_accuracy)})"
        )
        if selection.baseline_model_name:
            base += (
                f" beat the best baseline {selection.baseline_model_name} "
                f"(hold-out f1_score {_fmt(selection.baseline_f1_score)})"
            )
        return (
            f"{config.PERSISTENCE_NOTE_OPTIMIZED} Selected: {base}. Rule: primary "
            f"{config.BEST_MODEL_PRIMARY_METRIC} descending, secondary "
            f"{config.BEST_MODEL_SECONDARY_METRIC} descending, model name ascending."
        )

    reason = config.PERSISTENCE_NOTE_BASELINE_FALLBACK
    if optimized is None and selection.optimization_status:
        reason += f" Optimization status: {selection.optimization_status}."
    if baseline is not None:
        reason += (
            f" Selected: baseline {baseline.model_name} (hold-out f1_score "
            f"{_fmt(baseline.f1_score)})."
        )
    return reason


# --------------------------------------------------------------------------- #
# Final fit
# --------------------------------------------------------------------------- #
def rebuild_training_split(agent_state: Any) -> Tuple[Any, Any]:
    """Rebuild the training split Phase 4 used, without touching the hold-out rows.

    The dataset is re-read from disk, the recorded duplicate strategy is applied
    to the in-memory frame, the target and the excluded columns are removed, and
    the split is repeated with the configured test size and seed. The result is
    therefore row-for-row the split the recorded metrics belong to.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        An ``(X_train, y_train)`` tuple.

    Raises:
        PersistenceError: The dataset is missing, has no target, or is too small.
    """
    dataset_path = getattr(agent_state, "dataset_path", None)
    if not dataset_path:
        raise PersistenceError(
            "The state carries no dataset path, so the training split cannot be "
            "rebuilt for the final fit."
        )

    dataframe = load_training_dataframe(str(dataset_path))

    strategy = ""
    config_record = getattr(agent_state, "preprocessing_config", None)
    if config_record is not None and config_record.duplicate_handling is not None:
        strategy = str(config_record.duplicate_handling.strategy)
    if strategy and strategy != "keep_all":
        dataframe, _removed = remove_duplicate_rows(dataframe, strategy=strategy)

    features, target = prepare_features_and_target(
        dataframe,
        getattr(agent_state, "target_column", None),
        getattr(agent_state, "excluded_feature_names", None),
    )
    split = getattr(agent_state, "data_split", None)
    split_result = split_dataset(
        features,
        target,
        test_size=(
            getattr(split, "test_size", None)
            if split is not None and getattr(split, "test_size", None) is not None
            else config.TEST_SIZE
        ),
        random_state=(
            getattr(split, "random_state", None)
            if split is not None and getattr(split, "random_state", None) is not None
            else config.RANDOM_STATE
        ),
    )
    return split_result.X_train, split_result.y_train


def build_final_pipeline(
    agent_state: Any,
    selection: FinalModelSelection,
) -> FinalPipeline:
    """Produce the deployable pipeline for the selected model.

    Two strategies, and only two:

    ``reused_fitted_pipeline``
        The Phase 4 baseline winner already has a fitted pipeline on the state
        (``best_pipeline``). It was fitted on the training split alone, so it is
        saved exactly as it was, and the recorded hold-out metrics keep describing
        the model that was written.

    ``rebuilt_and_refitted_pipeline``
        There is no fitted pipeline for the selection - always the case for a
        tuned model, because Phase 6 keeps only the winning parameters - so the
        estimator is rebuilt from the Phase 3 preprocessing definition plus the
        selected parameters and refitted on the training split.

    This runs *after* :func:`select_final_model` and never before it, and it never
    changes the selection, the optimization result or any recorded metric.

    Args:
        agent_state: The authoritative machine learning state.
        selection: The recorded final model choice.

    Returns:
        A :class:`FinalPipeline` holding the fitted pipeline.

    Raises:
        PersistenceError: The preprocessing configuration is missing, the split
            cannot be rebuilt, or the refit raised.
    """
    if selection.selection_source == FinalModelSource.BASELINE.value:
        existing = getattr(agent_state, "best_pipeline", None)
        if (
            existing is not None
            and str(getattr(agent_state, "best_baseline_model", None).model_name)
            == selection.model_name
        ):
            return FinalPipeline(
                pipeline=existing,
                strategy="reused_fitted_pipeline",
                note=(
                    "The fitted baseline pipeline from the Training Agent was saved "
                    "unchanged. It was fitted on the training split only."
                ),
            )

    config_record = getattr(agent_state, "preprocessing_config", None)
    if config_record is None:
        raise PersistenceError(
            "No preprocessing configuration is available, so the selected model "
            "cannot be rebuilt."
        )

    parameters = dict(selection.best_params or {})
    if selection.selection_source == FinalModelSource.OPTIMIZED.value:
        # The tuned parameters are used exactly as Phase 6 recorded them, with the
        # fixed estimator seed reapplied by the same helper the search used.
        pipeline = build_tuned_pipeline(config_record, selection.model_name, parameters)
        strategy = "rebuilt_and_refitted_pipeline"
        note = (
            "The tuned estimator was rebuilt from the winning Optuna parameters and "
            "refitted on the training split."
        )
    else:
        pipeline = build_model_pipeline(config_record, selection.model_name, parameters)
        strategy = "rebuilt_and_refitted_pipeline"
        note = (
            "The baseline estimator was rebuilt from the recorded parameters and "
            "refitted on the training split."
        )

    X_train, y_train = rebuild_training_split(agent_state)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as error:  # noqa: BLE001 - reported with full context
        raise PersistenceError(
            f"The final fit of '{selection.model_name}' failed on the training split: "
            f"{type(error).__name__}: {error}"
        ) from error

    return FinalPipeline(
        pipeline=pipeline,
        strategy=strategy,
        note=f"{note} {config.PERSISTENCE_NOTE_FINAL_FIT}",
        train_samples=int(getattr(X_train, "shape", (0,))[0]),
    )


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _class_labels(agent_state: Any, model_name: str) -> List[str]:
    """Return the class labels for the persisted model.

    The stored confusion matrix of the selected experiment is preferred because it
    carries the exact order scikit-learn used. The class distribution of the whole
    dataset is the fallback, sorted so the answer stays deterministic.

    Args:
        agent_state: The authoritative machine learning state.
        model_name: The selected model name.

    Returns:
        The class labels as strings, possibly empty.
    """
    for experiment in getattr(agent_state, "experiment_results", None) or []:
        if str(getattr(experiment, "model_name", "")) != str(model_name):
            continue
        matrix = getattr(experiment, "confusion_matrix", None)
        labels = list(getattr(matrix, "labels", None) or []) if matrix is not None else []
        if labels:
            return [str(item) for item in labels]

    distribution = getattr(agent_state, "class_distribution", None) or {}
    return sorted(str(key) for key in distribution)


def _confusion_matrix(
    agent_state: Any,
    selection: FinalModelSelection,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return the stored confusion matrix of the persisted model, plus a note.

    Only Phase 4 stores a confusion matrix, so one exists for a baseline selection
    and never for an optimized one. Rather than recomputing it - which would be a
    second evaluation of a model the project evaluated once on purpose - the
    optimized case returns None together with a note that says exactly why.

    Args:
        agent_state: The authoritative machine learning state.
        selection: The recorded final model choice.

    Returns:
        An ``(matrix, note)`` tuple. ``matrix`` is a dictionary with ``labels``,
        ``matrix`` and ``total``, or None when no matrix describes this model.
        ``note`` is empty when a matrix was found and explains its absence
        otherwise.
    """
    if selection.selection_source == FinalModelSource.OPTIMIZED.value:
        return None, (
            "The persisted model is an optimized one. Phase 6 records the hold-out "
            "f1_score and accuracy of a tuned model but no confusion matrix, and "
            "persistence never re-scores, so there is no matrix to store."
        )

    experiment = experiment_for(agent_state, selection.model_name)
    if experiment is None:
        return None, "No training experiment was recorded for this model."
    matrix = getattr(experiment, "confusion_matrix", None)
    if matrix is None:
        return None, "The training experiment recorded no confusion matrix."
    return (
        {
            "labels": [str(item) for item in (getattr(matrix, "labels", None) or [])],
            "matrix": [
                [int(value) for value in row]
                for row in (getattr(matrix, "matrix", None) or [])
            ],
            "total": int(getattr(matrix, "total", 0) or 0),
        },
        "",
    )


def experiment_for(agent_state: Any, model_name: str) -> Optional[Any]:
    """Return the Phase 4 ``ExperimentResult`` for one model name.

    Args:
        agent_state: The authoritative machine learning state.
        model_name: Registry name of the model.

    Returns:
        The matching `ExperimentResult`, or None when the model was never trained.
    """
    for experiment in getattr(agent_state, "experiment_results", None) or []:
        if str(getattr(experiment, "model_name", "")) == str(model_name):
            return experiment
    return None


def _metrics_for(
    agent_state: Any,
    selection: FinalModelSelection,
) -> Tuple[Dict[str, Optional[float]], str]:
    """Return the hold-out metrics of the persisted model, plus their source.

    The two cases are genuinely different and are never mixed:

    * a **baseline** selection was measured by Phase 4, so all four metrics exist;
    * an **optimized** selection was measured by Phase 6, which records the
      hold-out ``f1_score`` and ``accuracy`` of a tuned model and nothing else.
      Precision, recall and a confusion matrix for a *tuned* model do not exist
      anywhere in the project, so they stay None rather than being filled in with
      the untuned model's numbers, which would describe a different model.

    Args:
        agent_state: The authoritative machine learning state.
        selection: The recorded final model choice.

    Returns:
        An ``(metrics, source)`` tuple. ``metrics`` has ``accuracy``, ``precision``,
        ``recall`` and ``f1_score``; ``source`` names the phase that measured them.
    """
    empty: Dict[str, Optional[float]] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
    }

    if selection.selection_source == FinalModelSource.OPTIMIZED.value:
        detail = optimization_result_for(agent_state, selection.model_name)
        if detail is None:
            return empty, "unavailable"
        return (
            {
                "accuracy": getattr(detail, "holdout_accuracy", None),
                "precision": None,
                "recall": None,
                "f1_score": getattr(detail, "holdout_f1_score", None),
            },
            "optimization_holdout",
        )

    experiment = experiment_for(agent_state, selection.model_name)
    if experiment is None:
        return empty, "unavailable"
    return (
        {
            "accuracy": getattr(experiment, "accuracy", None),
            "precision": getattr(experiment, "precision", None),
            "recall": getattr(experiment, "recall", None),
            "f1_score": getattr(experiment, "f1_score", None),
        },
        "evaluation_holdout",
    )


def _cv_metrics_for(agent_state: Any, selection: FinalModelSelection) -> Dict[str, Any]:
    """Return the cross-validation metrics of the persisted model.

    A tuned model is described by the cross-validated value the search maximized
    (``optimized_cv_*``); a baseline model by the Phase 4 cross-validation. The two
    are never mixed, because the cross-validation mean of the tuned configuration
    is the quantity the search optimized and is therefore optimistically biased.

    Args:
        agent_state: The authoritative machine learning state.
        selection: The recorded final model choice.

    Returns:
        A dictionary with ``cv_f1_mean``, ``cv_accuracy_mean``, ``cv_f1_std``,
        ``cv_n_splits`` and ``source``. Values that were never recorded are None.
    """
    folds = 0
    cv_config = getattr(agent_state, "cross_validation_config", None)
    if cv_config is not None:
        folds = int(getattr(cv_config, "n_splits", 0) or 0)

    if selection.selection_source == FinalModelSource.OPTIMIZED.value:
        detail = optimization_result_for(agent_state, selection.model_name)
        if detail is None:
            return {
                "cv_f1_mean": selection.optimized_cv_f1_mean,
                "cv_accuracy_mean": None,
                "cv_f1_std": None,
                "cv_n_splits": folds,
                "source": "optimized",
            }
        return {
            "cv_f1_mean": getattr(detail, "optimized_cv_f1_mean", None),
            "cv_accuracy_mean": getattr(detail, "optimized_cv_accuracy_mean", None),
            "cv_f1_std": None,
            "cv_n_splits": int(getattr(detail, "cv_n_splits", 0) or folds),
            "source": "optimized",
        }

    experiment = experiment_for(agent_state, selection.model_name)
    if experiment is None:
        return {
            "cv_f1_mean": selection.baseline_cv_f1_mean,
            "cv_accuracy_mean": None,
            "cv_f1_std": None,
            "cv_n_splits": folds,
            "source": "baseline",
        }
    return {
        "cv_f1_mean": getattr(experiment, "cv_f1_mean", None),
        "cv_accuracy_mean": getattr(experiment, "cv_accuracy_mean", None),
        "cv_f1_std": getattr(experiment, "cv_f1_std", None),
        "cv_n_splits": folds,
        "source": "baseline",
    }


def json_safe(value: Any) -> Any:
    """Convert a value into something ``json.dumps`` accepts without a hook.

    :func:`utils.state.to_serializable` handles dataclasses, enums and containers,
    but a NumPy scalar such as ``int64`` is neither a Python ``int`` nor a
    dataclass, so it would end up as the *string* ``"5"``. A hyperparameter
    recorded as ``"5"`` instead of ``5`` would change the metadata's meaning, so
    NumPy values are converted to their Python equivalents first.

    Args:
        value: Any value taken from the state or from Optuna.

    Returns:
        A JSON-serialisable value.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value

    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        try:
            return json_safe(value.item())
        except Exception:  # noqa: BLE001 - fall through to the generic path
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist())
        except Exception:  # noqa: BLE001 - fall through to the generic path
            pass
    return to_serializable(value)


def library_versions() -> Dict[str, str]:
    """Return the versions of the libraries that shaped the artifact.

    A persisted pipeline is only reproducible in an environment that matches it,
    so the versions travel with the model instead of living only in the reader's
    memory.

    Returns:
        A mapping of library name to version string.
    """
    import sklearn

    return {
        "scikit-learn": str(sklearn.__version__),
        "joblib": str(joblib.__version__),
    }


def build_metadata(
    agent_state: Any,
    selection: FinalModelSelection,
    final: FinalPipeline,
) -> Dict[str, Any]:
    """Build the JSON document that describes the persisted model.

    Everything in the document was produced by Phases 2-6. Phase 9 adds the
    selection, the artifact names and the final-fit strategy, and nothing else:
    there is no live estimator, no dataset row and no absolute path in here.

    Args:
        agent_state: The authoritative machine learning state.
        selection: The recorded final model choice.
        final: The pipeline that will be written, and how it was obtained.

    Returns:
        A JSON-serialisable dictionary.
    """
    config_record = getattr(agent_state, "preprocessing_config", None)
    numerical = list(getattr(agent_state, "preprocessing_numerical_features", None) or [])
    categorical = list(getattr(agent_state, "preprocessing_categorical_features", None) or [])
    labels = _class_labels(agent_state, selection.model_name)
    dataset_name = str(getattr(agent_state, "dataset_name", "") or "")
    split = getattr(agent_state, "data_split", None)
    metrics, metrics_source = _metrics_for(agent_state, selection)
    matrix, matrix_note = _confusion_matrix(agent_state, selection)

    notes: List[str] = [
        config.PERSISTENCE_NOTE_NO_METRICS,
        f"Persisted after model selection; the final fit exists only to produce this "
        f"artifact ({selection.selection_source} model).",
    ]
    if selection.selection_source == FinalModelSource.BASELINE.value:
        notes.append(config.PERSISTENCE_NOTE_BASELINE_FALLBACK)

    metadata: Dict[str, Any] = {
        "schema_version": int(config.PERSISTENCE_SCHEMA_VERSION),
        "artifact": config.PERSISTENCE_MODEL_FILENAME,
        "produced_by": "agents/persistence_agent.py",
        "pipeline_steps": ["preprocessor", "model"],
        "model_name": selection.model_name,
        "model_type": selection.model_type,
        "selection_source": selection.selection_source,
        "selection_reason": selection.selection_reason,
        "selection_rule": selection.selection_rule,
        "optimization_used": bool(selection.optimization_used),
        "optimization_status": selection.optimization_status,
        "best_params": json_safe(dict(selection.best_params or {})),
        "baseline_model_name": selection.baseline_model_name,
        "baseline_f1_score": selection.baseline_f1_score,
        "optimized_model_name": selection.optimized_model_name,
        "optimized_f1_score": selection.optimized_f1_score,
        "task_type": str(getattr(agent_state, "problem_type", "") or ""),
        "problem_type_reason": str(getattr(agent_state, "problem_type_reason", "") or ""),
        "target_column": str(getattr(agent_state, "target_column", "") or ""),
        "dataset_name": dataset_name,
        "class_labels": labels,
        "class_count": len(labels),
        "feature_columns": numerical + categorical,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "feature_count": len(numerical) + len(categorical),
        "excluded_features": [
            {"column": item.column, "reason": item.reason}
            for item in getattr(agent_state, "excluded_features", None) or []
        ],
        "metrics": json_safe(metrics),
        "metrics_source": metrics_source,
        "cv_metrics": json_safe(_cv_metrics_for(agent_state, selection)),
        "confusion_matrix": json_safe(matrix),
        "confusion_matrix_note": matrix_note,
        "final_fit": {
            "strategy": final.strategy,
            "note": final.note,
            "train_samples": final.train_samples,
        },
        "split": json_safe(
            {
                "test_size": float(config.TEST_SIZE),
                "random_state": int(config.RANDOM_STATE),
                "stratified": bool(getattr(split, "stratified", True)),
                "train_samples": int(getattr(split, "train_samples", 0) or 0),
                "test_samples": int(getattr(split, "test_samples", 0) or 0),
            }
        ),
        "preprocessing": json_safe(
            {
                "numerical_imputation": getattr(config_record, "numerical_imputation", ""),
                "categorical_imputation": getattr(config_record, "categorical_imputation", ""),
                "encoding": getattr(config_record, "encoding", ""),
                "scale_numeric": bool(getattr(config_record, "scale_numeric", False)),
                "scaling_strategy": getattr(config_record, "scaling_strategy", ""),
                "duplicate_handling": getattr(
                    getattr(config_record, "duplicate_handling", None), "strategy", ""
                ),
                "fit_scope": getattr(config_record, "fit_scope", ""),
            }
        ),
        "library_versions": library_versions(),
        "limitations": [
            config.PERSISTENCE_LIMITATION_SINGLE_SPLIT,
            config.PERSISTENCE_LIMITATION_NO_UPLOAD,
        ],
        "notes": notes,
    }
    return json_safe(metadata)


# --------------------------------------------------------------------------- #
# Run results bundle (what the dashboard reads for the report sections)
# --------------------------------------------------------------------------- #
def _baseline_rows(agent_state: Any) -> List[Dict[str, Any]]:
    """Return the Phase 4 comparison table as JSON-safe rows.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        One dictionary per ranked baseline model.
    """
    comparison = getattr(agent_state, "model_comparison", None)
    rows: List[Dict[str, Any]] = []
    if comparison is None:
        return rows
    for entry in getattr(comparison, "entries", None) or []:
        rows.append(
            {
                "model_name": entry.model_name,
                "model_type": entry.model_type,
                "rank": entry.rank,
                "status": "baseline",
                "accuracy": entry.accuracy,
                "precision": entry.precision,
                "recall": entry.recall,
                "f1_score": entry.f1_score,
                "cv_f1_mean": entry.cv_f1_mean,
                "cv_accuracy_mean": entry.cv_accuracy_mean,
                "preprocessing_scaling": entry.preprocessing_scaling,
            }
        )
    return rows


def _optimized_rows(agent_state: Any) -> List[Dict[str, Any]]:
    """Return the Phase 6 comparison as JSON-safe rows.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        One dictionary per tuned model.
    """
    rows: List[Dict[str, Any]] = []
    for item in getattr(agent_state, "optimization_results", None) or []:
        rows.append(
            {
                "model_name": item.model_name,
                "model_type": item.model_type,
                "status": "optimized",
                "optimization_status": item.status,
                "baseline_f1_score": item.baseline_holdout_f1_score,
                "optimized_f1_score": item.holdout_f1_score,
                "baseline_cv_f1_mean": item.baseline_cv_f1_mean,
                "optimized_cv_f1_mean": item.optimized_cv_f1_mean,
                "improvement": item.improvement,
                "improvement_percentage": item.improvement_percentage,
                "improvement_direction": item.improvement_direction,
                "best_params": json_safe(dict(item.best_params or {})),
                "best_trial_number": item.best_trial_number,
                "total_trials": item.total_trials,
                "completed_trials": item.completed_trials,
                "failed_trials": item.failed_trials,
                "reason": item.reason,
                "error_message": item.error_message,
            }
        )
    return rows


def _evaluation_section(agent_state: Any) -> Dict[str, Any]:
    """Return the Phase 5 evaluation results as a JSON-safe mapping.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        The evaluation section, empty when the Evaluation Agent produced nothing.
    """
    report = getattr(agent_state, "evaluation_report", None)
    if report is None:
        return {}

    best = report.best_model
    gap = report.gap_analysis
    return {
        "overall_status": report.overall_status,
        "ranking_rule": report.ranking_rule,
        "test_samples": report.test_samples,
        "problem_type": report.problem_type,
        "best_model": (
            {
                "model_name": best.best_baseline_model,
                "model_type": best.model_type,
                "selection_reason": best.selection_reason,
                "accuracy": best.accuracy,
                "precision": best.precision,
                "recall": best.recall,
                "f1_score": best.f1_score,
                "cv_accuracy_mean": best.cv_accuracy_mean,
            }
            if best is not None
            else None
        ),
        "models": [
            {
                "model_name": item.model_name,
                "model_type": item.model_type,
                "rank": item.rank,
                "status": item.evaluation_status,
                "accuracy": item.accuracy,
                "precision": item.precision,
                "recall": item.recall,
                "f1_score": item.f1_score,
                "cv_accuracy_mean": item.cv_accuracy_mean,
                "cv_f1_mean": None,
                "accuracy_gap": item.accuracy_gap,
                "accuracy_gap_classification": item.accuracy_gap_classification,
                "gap_interpretation": item.gap_interpretation,
                "notes": list(item.notes),
            }
            for item in report.model_evaluations
        ],
        "gap_analysis": (
            {
                "metric": gap.metric,
                "rule": gap.rule,
                "close_threshold": gap.close_threshold,
                "substantial_threshold": gap.substantial_threshold,
                "disclaimer": gap.disclaimer,
                "largest_absolute_gap": gap.largest_absolute_gap,
                "largest_absolute_gap_model": gap.largest_absolute_gap_model,
                "models_with_gap": gap.models_with_gap,
                "models_without_gap": list(gap.models_without_gap),
                "close_agreement_models": list(gap.close_agreement_models),
                "moderate_difference_models": list(gap.moderate_difference_models),
                "substantial_difference_models": list(gap.substantial_difference_models),
                "summary": gap.summary,
                "interpretation": gap.interpretation,
            }
            if gap is not None
            else None
        ),
        "observations": list(report.observations),
        "limitations": list(report.limitations),
        "recommendations": list(report.recommendations),
        "notes": list(report.notes),
    }


def _dataset_section(agent_state: Any) -> Dict[str, Any]:
    """Return the Phase 2 data analysis as a JSON-safe mapping.

    Only aggregates are written: row and column counts, missing-value counts and
    the class distribution. No dataset row is ever persisted.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        The dataset section.
    """
    summary = getattr(agent_state, "analysis_summary", None)
    missing = getattr(agent_state, "missing_values", None)
    shape = getattr(agent_state, "dataset_shape", None)

    return {
        "name": str(getattr(agent_state, "dataset_name", "") or ""),
        "rows": int(getattr(summary, "rows", 0) or getattr(shape, "rows", 0) or 0),
        "columns": int(getattr(summary, "columns", 0) or getattr(shape, "columns", 0) or 0),
        "numerical_feature_count": int(getattr(summary, "numerical_feature_count", 0) or 0),
        "categorical_feature_count": int(
            getattr(summary, "categorical_feature_count", 0) or 0
        ),
        "total_missing_values": int(
            getattr(summary, "total_missing_values", None)
            if getattr(summary, "total_missing_values", None) is not None
            else getattr(missing, "total", 0) or 0
        ),
        "affected_missing_columns": int(getattr(missing, "affected_columns", 0) or 0),
        "duplicate_count": int(getattr(summary, "duplicate_count", 0) or 0),
        "target_column": str(getattr(agent_state, "target_column", "") or ""),
        "target_detection_method": str(getattr(agent_state, "target_detection_method", "") or ""),
        "target_detection_confidence": str(
            getattr(agent_state, "target_detection_confidence", "") or ""
        ),
        "problem_type": str(getattr(agent_state, "problem_type", "") or ""),
        "problem_type_reason": str(getattr(agent_state, "problem_type_reason", "") or ""),
        "issue_count": int(getattr(summary, "issue_count", 0) or 0),
        "issues_by_severity": dict(getattr(summary, "issues_by_severity", None) or {}),
        "numerical_features": list(getattr(agent_state, "numerical_features", None) or []),
        "categorical_features": list(
            getattr(agent_state, "categorical_features", None) or []
        ),
        "excluded_features": [
            {"column": item.column, "reason": item.reason}
            for item in getattr(agent_state, "excluded_features", None) or []
        ],
        "numerical_statistics": {
            name: {
                "mean": getattr(item, "mean", None),
                "std": getattr(item, "std", None),
                "minimum": getattr(item, "minimum", None),
                "maximum": getattr(item, "maximum", None),
            }
            for name, item in (
                getattr(agent_state, "numerical_statistics", None) or {}
            ).items()
        },
        "categorical_statistics": {
            name: {
                "unique_count": getattr(item, "unique_count", None),
                "top_values": dict(getattr(item, "top_values", None) or {}),
            }
            for name, item in (
                getattr(agent_state, "categorical_statistics", None) or {}
            ).items()
        },
        "class_distribution": {
            str(label): {
                "count": getattr(item, "count", 0),
                "percentage": getattr(item, "percentage", 0.0),
            }
            for label, item in (
                getattr(agent_state, "class_distribution", None) or {}
            ).items()
        },
        "summary_text": str(getattr(summary, "summary_text", "") or ""),
    }


def _optimization_section(agent_state: Any) -> Dict[str, Any]:
    """Return the Phase 6 summary as a JSON-safe mapping.

    Args:
        agent_state: The authoritative machine learning state.

    Returns:
        The optimization section, empty when optimization never ran.
    """
    report = getattr(agent_state, "optimization_report", None)
    if report is None:
        return {}

    best = report.best_optimized_model
    return {
        "overall_status": report.overall_status,
        "performed": report.overall_status == OptimizationStatus.COMPLETED.value,
        "objective_metric": report.objective_metric,
        "objective_rule": report.objective_rule,
        "selection_rule": report.selection_rule,
        "candidate_rule": report.candidate_rule,
        "ranking_rule": report.ranking_rule,
        "n_trials_per_model": report.n_trials_per_model,
        "top_models": report.top_models,
        "sampler": report.sampler,
        "sampler_seed": report.sampler_seed,
        "cv_n_splits": report.cv_n_splits,
        "holdout_note": report.holdout_note,
        "models_selected_for_optimization": list(report.models_selected_for_optimization),
        "models_not_optimized": list(report.models_not_optimized),
        "best_optimized_model": (
            {
                "model_name": best.best_optimized_model,
                "model_type": best.model_type,
                "baseline_model_name": best.baseline_model_name,
                "selection_reason": best.selection_reason,
                "performance_summary": best.performance_summary,
                "baseline_cv_f1_mean": best.baseline_cv_f1_mean,
                "optimized_cv_f1_mean": best.optimized_cv_f1_mean,
                "improvement": best.improvement,
                "improvement_percentage": best.improvement_percentage,
                "improvement_direction": best.improvement_direction,
                "baseline_holdout_f1_score": best.baseline_holdout_f1_score,
                "holdout_f1_score": best.holdout_f1_score,
                "holdout_f1_delta": best.holdout_f1_delta,
                "best_params": json_safe(dict(best.best_params or {})),
                "best_trial_number": best.best_trial_number,
                "total_trials": best.total_trials,
            }
            if best is not None
            else None
        ),
        "observations": list(report.observations),
        "limitations": list(report.limitations),
        "recommendations": list(report.recommendations),
        "notes": list(report.notes),
    }


def build_run_results(
    agent_state: Any,
    llm_results: Optional[Sequence[Any]] = None,
    *,
    selection: Optional[FinalModelSelection] = None,
) -> Dict[str, Any]:
    """Build the JSON document the dashboard reads its report sections from.

    The bundle is a *read model*, not a second source of truth: every value in it
    was already produced by Phases 2-6 and copied here. Nothing is recomputed, and
    no live object is included.

    Args:
        agent_state: The authoritative machine learning state.
        llm_results: Optional Phase 8 interpretation records. They live in the
            graph state rather than on the ``AgentState``, so they are passed in.
        selection: The recorded final model choice, added when available.

    Returns:
        A JSON-serialisable dictionary.
    """
    payload: Dict[str, Any] = {
        "schema_version": int(config.PERSISTENCE_SCHEMA_VERSION),
        "produced_by": "agents/persistence_agent.py",
        "dataset": _dataset_section(agent_state),
        "baseline_models": _baseline_rows(agent_state),
        "optimized_models": _optimized_rows(agent_state),
        "evaluation": _evaluation_section(agent_state),
        "optimization": _optimization_section(agent_state),
        "preprocessing": {
            "status": str(getattr(agent_state, "preprocessing_status", "") or ""),
            "numerical_features": list(
                getattr(agent_state, "preprocessing_numerical_features", None) or []
            ),
            "categorical_features": list(
                getattr(agent_state, "preprocessing_categorical_features", None) or []
            ),
            "steps": [
                {
                    "name": step.name,
                    "applies_to": step.applies_to,
                    "transformers": list(step.transformers),
                }
                for step in getattr(agent_state, "preprocessing_steps", None) or []
            ],
            "summary": str(
                getattr(getattr(agent_state, "preprocessing_summary", None), "summary_text", "")
                or ""
            ),
        },
        "llm": {
            "enabled": bool(llm_results),
            "results": [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in (llm_results or [])
            ],
        },
        "notes": [config.PERSISTENCE_LIMITATION_NO_RETRAIN],
    }
    if selection is not None:
        metrics, metrics_source = _metrics_for(agent_state, selection)
        matrix, matrix_note = _confusion_matrix(agent_state, selection)
        payload["final_model"] = {
            "model_name": selection.model_name,
            "model_type": selection.model_type,
            "selection_source": selection.selection_source,
            "selection_reason": selection.selection_reason,
            "selection_rule": selection.selection_rule,
            "optimization_used": selection.optimization_used,
            "optimization_status": selection.optimization_status,
            "baseline_model_name": selection.baseline_model_name,
            "baseline_f1_score": selection.baseline_f1_score,
            "baseline_accuracy": selection.baseline_accuracy,
            "optimized_model_name": selection.optimized_model_name,
            "optimized_f1_score": selection.optimized_f1_score,
            "optimized_accuracy": selection.optimized_accuracy,
            "best_params": json_safe(dict(selection.best_params or {})),
            "metrics": json_safe(metrics),
            "metrics_source": metrics_source,
            "cv_metrics": json_safe(_cv_metrics_for(agent_state, selection)),
            "confusion_matrix": json_safe(matrix),
            "confusion_matrix_note": matrix_note,
        }
    return json_safe(payload)


# --------------------------------------------------------------------------- #
# Atomic writing and reading
# --------------------------------------------------------------------------- #
def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` without ever leaving a partial file behind.

    The payload goes to a temporary file **in the destination directory** (so the
    final move stays on one filesystem and is therefore atomic) and is then
    renamed over the target. A crash mid-write leaves the previous artifact
    intact; a crash after the rename leaves the new one intact. There is no moment
    in between where a half-written file exists at ``path``.

    Args:
        path: Destination file.
        data: Bytes to write.

    Raises:
        ArtifactWriteError: The directory could not be created or the write failed.
    """
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactWriteError(
            f"Could not create the output directory '{destination.parent}': {error}"
        ) from error

    handle = None
    temporary: Optional[str] = None
    try:
        # `delete=False` so the handle stays open while joblib writes to it.
        file_descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        handle = os.fdopen(file_descriptor, "wb")
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary, str(destination))
        temporary = None
    except OSError as error:
        raise ArtifactWriteError(f"Could not write '{destination}': {error}") from error
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:  # pragma: no cover - the write already failed
                pass
        if temporary is not None and os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass


def write_json(path: Path, payload: Dict[str, Any]) -> int:
    """Write a JSON document atomically and return its size in bytes.

    Args:
        path: Destination file.
        payload: A JSON-serialisable mapping.

    Returns:
        The number of bytes written.

    Raises:
        ArtifactWriteError: The payload is not JSON-serialisable, or the write
            failed. A non-serialisable payload is reported rather than written
            with a ``default=`` hook, because that hook is exactly what would let
            a live object leak into a file that is meant to describe one.
    """
    try:
        text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ArtifactWriteError(
            f"The metadata for '{path.name}' is not JSON serialisable: {error}"
        ) from error
    data = text.encode("utf-8")
    atomic_write_bytes(path, data)
    return len(data)


def save_pipeline(pipeline: Any, path: Path) -> int:
    """Write a fitted pipeline to disk atomically.

    Args:
        pipeline: The fitted scikit-learn ``Pipeline``.
        path: Destination file.

    Returns:
        The number of bytes written.

    Raises:
        ArtifactWriteError: The pipeline could not be serialised or written.
    """
    try:
        import io

        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer, compress=config.PERSISTENCE_JOBLIB_COMPRESS)
    except Exception as error:  # noqa: BLE001 - reported with full context
        raise ArtifactWriteError(
            f"The fitted pipeline could not be serialised for '{Path(path).name}': "
            f"{type(error).__name__}: {error}"
        ) from error
    data = buffer.getvalue()
    atomic_write_bytes(Path(path), data)
    return len(data)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON document written by this project.

    Args:
        path: The file to read.

    Returns:
        The parsed mapping.

    Raises:
        ArtifactReadError: The file is missing, is not valid JSON, or is not a
            JSON object.
    """
    target = Path(path)
    if not target.is_file():
        raise ArtifactReadError(f"Artifact not found: {target.name}")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        raise ArtifactReadError(f"Could not read '{target.name}': {error}") from error
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise ArtifactReadError(
            f"'{target.name}' is not valid JSON, so it cannot be trusted: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ArtifactReadError(
            f"'{target.name}' must contain a JSON object, found "
            f"{type(payload).__name__}."
        )
    return payload


def validate_schema(payload: Dict[str, Any], *, name: str) -> None:
    """Check that a document declares a schema version this project understands.

    Args:
        payload: The parsed document.
        name: File name, used in the error message.

    Raises:
        ArtifactReadError: The version is missing or newer than this project.
            Guessing at an unknown layout would produce a wrong reading rather
            than an honest "not available".
    """
    version = payload.get("schema_version")
    if version is None:
        raise ArtifactReadError(
            f"'{name}' does not declare a schema_version, so it cannot be trusted."
        )
    try:
        number = int(version)
    except (TypeError, ValueError) as error:
        raise ArtifactReadError(
            f"'{name}' declares a non-numeric schema_version {version!r}."
        ) from error
    if number > int(config.PERSISTENCE_SCHEMA_VERSION):
        raise ArtifactReadError(
            f"'{name}' was written by a newer version of the project "
            f"(schema_version {number} > {config.PERSISTENCE_SCHEMA_VERSION}). "
            "Re-run the workflow to rewrite the artifacts."
        )


def load_pipeline(path: Path) -> Any:
    """Load a pipeline this project wrote.

    This is the **only** place in the project that calls ``joblib.load``. A
    ``.joblib`` file is a pickle: reading one executes whatever code it contains,
    so the accepted path is fixed by configuration and never comes from a user
    upload.

    Args:
        path: The artifact to load. Callers pass a path derived from
            :func:`artifact_paths`.

    Returns:
        The unpickled object.

    Raises:
        ArtifactReadError: The file is missing or could not be unpickled.
    """
    target = Path(path)
    if not target.is_file():
        raise ArtifactReadError(f"Artifact not found: {target.name}")
    try:
        return joblib.load(target)
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        raise ArtifactReadError(
            f"'{target.name}' could not be loaded: {type(error).__name__}: {error}"
        ) from error


def load_persisted_model(
    output_dir: Optional[Path] = None,
) -> Optional[PersistedModel]:
    """Load the artifact this project wrote, or None when there is none.

    The dashboard calls exactly this. It takes an *output directory* rather than a
    model file on purpose: there is no code path through which a caller can ask
    this project to load an arbitrary file.

    Args:
        output_dir: Directory holding the artifacts, or None for the configured one.

    Returns:
        The loaded :class:`PersistedModel`, or None when the artifacts are absent.

    Raises:
        ArtifactReadError: An artifact exists but cannot be read or understood.
    """
    paths = artifact_paths(output_dir)
    if not artifacts_exist(output_dir):
        return None

    metadata = read_json(paths.metadata)
    validate_schema(metadata, name=paths.metadata.name)

    recorded = str(metadata.get("artifact", "") or "")
    if recorded and recorded != paths.model.name:
        # The metadata points at a different file than the one that sits next to
        # it. Something replaced part of the artifact set, so the pair cannot be
        # trusted to belong together.
        raise ArtifactReadError(
            f"'{paths.metadata.name}' describes '{recorded}' but "
            f"'{paths.model.name}' was found next to it. The artifact set is "
            "inconsistent; re-run the workflow."
        )

    pipeline = load_pipeline(paths.model)
    return PersistedModel(pipeline=pipeline, metadata=metadata, paths=paths)


def load_run_results(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the dashboard read model, or an empty mapping when it is absent.

    A missing results bundle is not an error: the dashboard can still show the
    model, its metadata and a prediction, and says so for the sections it cannot
    fill.

    Args:
        output_dir: Directory holding the artifacts, or None for the configured one.

    Returns:
        The parsed mapping, empty when the file does not exist.

    Raises:
        ArtifactReadError: The file exists but cannot be read or understood.
    """
    paths = artifact_paths(output_dir)
    if not paths.results.is_file():
        return {}
    payload = read_json(paths.results)
    validate_schema(payload, name=paths.results.name)
    return payload


def write_artifacts(
    agent_state: Any,
    selection: FinalModelSelection,
    final: FinalPipeline,
    *,
    output_dir: Optional[Path] = None,
    llm_results: Optional[Sequence[Any]] = None,
) -> ArtifactPaths:
    """Write the three artifacts and return where they went.

    The order matters: the model file is written **first**, so a reader can never
    find metadata describing a model that is not there yet.

    Args:
        agent_state: The authoritative machine learning state.
        selection: The recorded final model choice.
        final: The pipeline to persist.
        output_dir: Directory to write into, or None for the configured one.
        llm_results: Optional Phase 8 interpretation records.

    Returns:
        The resolved :class:`ArtifactPaths`.

    Raises:
        ArtifactWriteError: One of the three writes failed.
    """
    paths = artifact_paths(output_dir)
    size = save_pipeline(final.pipeline, paths.model)

    metadata = build_metadata(agent_state, selection, final)
    metadata["artifact_size_bytes"] = int(size)
    write_json(paths.metadata, metadata)

    write_json(
        paths.results,
        build_run_results(agent_state, llm_results, selection=selection),
    )

    logger.info(
        "%s saved %s (%d bytes), %s and %s to %s",
        config.PERSISTENCE_LOG_PREFIX,
        paths.model.name,
        size,
        paths.metadata.name,
        paths.results.name,
        paths.directory,
    )
    return paths


def _fmt(value: Optional[float]) -> str:
    """Format a metric for a sentence, showing '-' when it is missing."""
    return "-" if value is None else f"{float(value):.4f}"


__all__ = [
    "ArtifactPaths",
    "ArtifactReadError",
    "ArtifactWriteError",
    "Candidate",
    "FinalPipeline",
    "NoModelToPersistError",
    "PersistedModel",
    "PersistenceError",
    "artifact_paths",
    "artifacts_exist",
    "baseline_candidate",
    "build_final_pipeline",
    "build_metadata",
    "build_run_results",
    "build_tuned_pipeline",
    "experiment_for",
    "json_safe",
    "library_versions",
    "load_persisted_model",
    "load_pipeline",
    "load_run_results",
    "optimized_candidate",
    "optimization_result_for",
    "read_json",
    "rebuild_training_split",
    "resolve_model_type",
    "resolve_output_dir",
    "save_pipeline",
    "select_final_model",
    "validate_schema",
    "write_artifacts",
    "write_json",
]
