"""Reusable hyperparameter optimization operations (Phase 6).

This module owns every Optuna interaction and every calculation around it, so
the Optimization Agent stays a thin orchestration layer. The most important
invariant of this module is the **leakage rule**, which extends the Phase 3/4
one:

    the objective receives the *unfitted* `preprocessor -> estimator` Pipeline
    together with the TRAINING rows only

    cross_validate(pipeline, X_train, y_train, ...)  ->  every fold clones the
                                                       pipeline and refits the
                                                       imputer, encoder, scaler
                                                       and estimator on that
                                                       fold's training rows

The hold-out split is never passed into the objective, so it cannot influence
which hyperparameters are suggested, which configurations are compared, or which
models are selected for tuning. It is scored once afterwards, and only to report
the tuned model next to its baseline.

Three further rules are enforced structurally rather than by convention:

* **Controlled spaces.** A search space exists only for models in the Phase 4
  registry, and it is a small explicit list per model. Nothing is generated from
  a string and no estimator parameter is sampled that scikit-learn would not
  accept.
* **Reproducibility.** The sampler is seeded, the estimators receive a fixed
  `random_state` and cross-validation uses the Phase 4 fold conventions, so two
  runs on the same data reach the same best configuration.
* **Isolated failures.** A trial that raises is recorded as a failed trial and
  the search continues; a model whose search fails entirely is recorded and the
  next selected model is still optimized.

Nothing in this module writes to the source CSV file.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from tools import model_tools
from tools.model_tools import ModelToolError, get_model_definition
from tools.preprocessing_tools import PreprocessingError
from tools.training_tools import TrainingToolError, build_model_pipeline
from utils import config
from utils.logger import setup_logger
from utils.state import (
    CrossValidationConfig,
    ExperimentResult,
    ExperimentStatus,
    ImprovementDirection,
    ModelRecommendation,
    OptimizationCounts,
    OptimizationResult,
    OptimizationStatus,
    OptimizedModelRecommendation,
    OptimizationReport,
    PreprocessingConfig,
    ProblemType,
    TrialRecord,
    TrialState,
)

logger = setup_logger("tools.optimization_tools")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class OptimizationToolError(Exception):
    """Base class for hyperparameter optimization errors."""


class OptimizationDataError(OptimizationToolError):
    """Raised when the optimization inputs cannot be used.

    This covers a malformed trial record, an unsupported model and an unusable
    cross-validation configuration. It is deliberately distinct from the
    per-trial failure path: a bad *input* is a configuration error that the
    caller must see, while a bad *trial* is a normal search outcome.
    """


# Optuna prints a full traceback for every failed trial. The failure is already
# recorded on the `TrialRecord` and logged once by this module, so Optuna's own
# reporting is quietened to avoid printing the same error twice.
optuna.logging.set_verbosity(getattr(optuna.logging, config.OPTUNA_LOG_LEVEL))


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def get_optimizable_models() -> List[str]:
    """Return the models that have a Phase 6 search space, in registry order.

    Every name is also a Phase 4 registry name: a model cannot be optimized
    unless it could also be trained as a baseline.
    """
    return [
        name
        for name in model_tools.get_supported_models()
        if _search_space_kind(name) is not None
    ]


def is_optimizable(model_name: str) -> bool:
    """Return whether a model has a Phase 6 search space.

    This is a predicate, so it is total: a name the registry does not know
    returns False instead of raising. A caller that needs the error (for example
    `describe_search_space`) gets it from `_require_kind`, which raises with the
    full list of supported models. Returning False here also keeps candidate
    selection safe if a state ever carries an unfamiliar model name.
    """
    return _search_space_kind(model_name) is not None


def _search_space_kind(model_name: str) -> Optional[str]:
    """Return the internal search-space identifier of a model.

    An unknown name returns None: `is_optimizable` must be able to answer for
    any string, and the loud error belongs to the callers that need one.
    """
    try:
        definition = get_model_definition(model_name)
    except ModelToolError:
        return None
    return definition.name if definition.name in _SEARCH_SPACE_KINDS else None


# Internal identifier per optimizable model. It is never exposed in a report;
# reports use the model display name.
_SEARCH_SPACE_KINDS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "K-Nearest Neighbors": "knn",
    "Support Vector Machine": "svm",
}


def describe_search_space(model_name: str, problem_type: Optional[str]) -> List[str]:
    """Describe the search space of one model in plain, readable strings.

    The description is what the report shows, so it states the *range* of each
    tunable parameter instead of just its name. `problem_type` matters for
    Logistic Regression, because not every solver supports multiclass targets.

    Args:
        model_name: Display name from the registry.
        problem_type: Value from `ProblemType`.

    Returns:
        One description line per tunable parameter.

    Raises:
        OptimizationDataError: The model has no search space, or the problem
            type is not a supported classification problem.
    """
    kind = _require_kind(model_name, problem_type)
    solvers = _logistic_solvers(problem_type)

    descriptions = {
        "logistic_regression": [
            f"C: log-uniform between {config.OPTUNA_LOGISTIC_C_RANGE[0]:g} and "
            f"{config.OPTUNA_LOGISTIC_C_RANGE[1]:g}",
            f"solver: one of {list(solvers)}",
            f"max_iter: fixed at {config.OPTUNA_LOGISTIC_MAX_ITER} "
            "(optimiser convergence is not tuned)",
        ],
        "decision_tree": [
            f"max_depth: integer between {config.OPTUNA_TREE_MAX_DEPTH_RANGE[0]} "
            f"and {config.OPTUNA_TREE_MAX_DEPTH_RANGE[1]}",
            "min_samples_split: integer between "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[0]} and "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[1]}",
            "min_samples_leaf: integer between "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[0]} and "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[1]}",
            f"criterion: one of {list(config.OPTUNA_TREE_CRITERIA)}",
        ],
        "random_forest": [
            "n_estimators: integer between "
            f"{config.OPTUNA_FOREST_N_ESTIMATORS_RANGE[0]} and "
            f"{config.OPTUNA_FOREST_N_ESTIMATORS_RANGE[1]} in steps of "
            f"{config.OPTUNA_FOREST_N_ESTIMATORS_STEP}",
            f"max_depth: one of {list(config.OPTUNA_FOREST_MAX_DEPTH_CHOICES)} "
            "(None means no depth limit)",
            "min_samples_split: integer between "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[0]} and "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[1]}",
            "min_samples_leaf: integer between "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[0]} and "
            f"{config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[1]}",
            f"max_features: one of {list(config.OPTUNA_FOREST_MAX_FEATURES_CHOICES)}",
        ],
        "knn": [
            "n_neighbors: integer between "
            f"{config.OPTUNA_KNN_N_NEIGHBORS_RANGE[0]} and "
            f"{config.OPTUNA_KNN_N_NEIGHBORS_RANGE[1]}",
            f"weights: one of {list(config.OPTUNA_KNN_WEIGHTS_CHOICES)}",
            f"metric: one of {list(config.OPTUNA_KNN_METRIC_CHOICES)}",
        ],
        "svm": [
            f"C: log-uniform between {config.OPTUNA_SVM_C_RANGE[0]:g} and "
            f"{config.OPTUNA_SVM_C_RANGE[1]:g}",
            f"kernel: one of {list(config.OPTUNA_SVM_KERNEL_CHOICES)}",
            "gamma: log-uniform between "
            f"{config.OPTUNA_SVM_GAMMA_RANGE[0]:g} and "
            f"{config.OPTUNA_SVM_GAMMA_RANGE[1]:g}, suggested only for a kernel "
            "that uses it (the linear kernel ignores gamma)",
        ],
    }
    return descriptions[kind]


def suggest_parameters(
    trial: "optuna.Trial",
    model_name: str,
    problem_type: Optional[str],
) -> Dict[str, Any]:
    """Ask Optuna for one parameter set for one trial.

    Every value is drawn from the controlled bounds in `config`, so the
    suggestion is always a configuration the registry estimator accepts.

    Args:
        trial: The Optuna trial to suggest from.
        model_name: Display name from the registry.
        problem_type: Value from `ProblemType`.

    Returns:
        The suggested parameters, ready to be passed to `build_estimator`.

    Raises:
        OptimizationDataError: The model has no search space, or the problem
            type is not supported.
    """
    kind = _require_kind(model_name, problem_type)

    if kind == "logistic_regression":
        low, high = config.OPTUNA_LOGISTIC_C_RANGE
        return {
            "C": trial.suggest_float("C", low, high, log=True),
            "solver": trial.suggest_categorical("solver", list(_logistic_solvers(problem_type))),
            "max_iter": config.OPTUNA_LOGISTIC_MAX_ITER,
        }

    if kind == "decision_tree":
        return {
            "max_depth": trial.suggest_int(
                "max_depth", config.OPTUNA_TREE_MAX_DEPTH_RANGE[0], config.OPTUNA_TREE_MAX_DEPTH_RANGE[1]
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[0],
                config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[1],
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[0],
                config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[1],
            ),
            "criterion": trial.suggest_categorical(
                "criterion", list(config.OPTUNA_TREE_CRITERIA)
            ),
        }

    if kind == "random_forest":
        low, high = config.OPTUNA_FOREST_N_ESTIMATORS_RANGE
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", low, high, step=config.OPTUNA_FOREST_N_ESTIMATORS_STEP
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", list(config.OPTUNA_FOREST_MAX_DEPTH_CHOICES)
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[0],
                config.OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE[1],
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[0],
                config.OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE[1],
            ),
            "max_features": trial.suggest_categorical(
                "max_features", list(config.OPTUNA_FOREST_MAX_FEATURES_CHOICES)
            ),
        }

    if kind == "knn":
        return {
            "n_neighbors": trial.suggest_int(
                "n_neighbors",
                config.OPTUNA_KNN_N_NEIGHBORS_RANGE[0],
                config.OPTUNA_KNN_N_NEIGHBORS_RANGE[1],
            ),
            "weights": trial.suggest_categorical(
                "weights", list(config.OPTUNA_KNN_WEIGHTS_CHOICES)
            ),
            "metric": trial.suggest_categorical(
                "metric", list(config.OPTUNA_KNN_METRIC_CHOICES)
            ),
        }

    # Support Vector Machine. gamma is suggested only for a kernel that reads it:
    # asking for a parameter the chosen kernel would ignore would spend trials on
    # a value that cannot change the model.
    kernel = trial.suggest_categorical("kernel", list(config.OPTUNA_SVM_KERNEL_CHOICES))
    low, high = config.OPTUNA_SVM_C_RANGE
    parameters: Dict[str, Any] = {
        "C": trial.suggest_float("C", low, high, log=True),
        "kernel": kernel,
    }
    if kernel == "linear":
        parameters["gamma"] = config.OPTUNA_SVM_LINEAR_GAMMA
    else:
        gamma_low, gamma_high = config.OPTUNA_SVM_GAMMA_RANGE
        parameters["gamma"] = trial.suggest_float(
            "gamma", gamma_low, gamma_high, log=config.OPTUNA_SVM_GAMMA_LOG
        )
    return parameters


def _logistic_solvers(problem_type: Optional[str]) -> Tuple[str, ...]:
    """Return the Logistic Regression solvers valid for a problem type.

    The choice is restricted on purpose. scikit-learn's `liblinear` solver
    supports binary targets only: with a multiclass target it raises inside
    every CV fold, which would turn a legal hyperparameter choice into a wall of
    failed trials. Restricting the categorical list per problem type keeps every
    suggested configuration valid instead of relying on the trial failure path.
    """
    if problem_type == ProblemType.BINARY_CLASSIFICATION.value:
        return config.OPTUNA_LOGISTIC_SOLVERS_BINARY
    if problem_type == ProblemType.MULTICLASS_CLASSIFICATION.value:
        return config.OPTUNA_LOGISTIC_SOLVERS_MULTICLASS
    raise OptimizationDataError(
        f"Problem type '{problem_type}' is not supported for hyperparameter "
        "optimization. Supported: "
        f"{[ProblemType.BINARY_CLASSIFICATION.value, ProblemType.MULTICLASS_CLASSIFICATION.value]}"
    )


def _require_kind(model_name: str, problem_type: Optional[str]) -> str:
    """Return the search-space identifier, validating model and problem type."""
    if not is_optimizable(model_name):
        raise OptimizationDataError(
            f"Model '{model_name}' has no hyperparameter search space. Optimizable "
            f"models: {get_optimizable_models()}"
        )
    # Validates the problem type as well.
    _logistic_solvers(problem_type)
    return _SEARCH_SPACE_KINDS[model_name]


def _with_fixed_seed(model_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Return the parameters with the fixed estimator seed applied.

    `random_state` is not a searched dimension -- leaving it to the registry
    default would make reproducibility depend on a value that lives in another
    module and can change without anyone touching the search space. Overriding
    it here means a recorded `best_params` dict always reproduces the model that
    produced its score. Deterministic estimators (K-Nearest Neighbors) receive no
    seed because their constructor has no such parameter.
    """
    seeded = dict(parameters)
    if "random_state" in get_model_definition(model_name).default_parameters:
        seeded["random_state"] = config.OPTUNA_RANDOM_SEED
    return seeded


def build_tuned_pipeline(
    preprocessing_config: PreprocessingConfig,
    model_name: str,
    best_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Rebuild the pipeline for a tuned model from its winning parameters.

    Phase 6 keeps the winning parameter dict but not the fitted estimator, because
    a fitted pipeline is a live object that must not sit on the shared state. This
    function is the public way back from a recorded ``best_params`` to an unfitted
    pipeline: it applies the same fixed estimator seed that :func:`_with_fixed_seed`
    applied during the search and then defers to the Phase 4 builder, so a
    reconstructed model is the model that produced the recorded score.

    It is used by the Model Persistence Agent (Phase 9) to produce the deployable
    artifact, and by nothing else in Phase 6, so Phase 6 behaviour is unchanged.

    Args:
        preprocessing_config: Configuration from the Preprocessing Agent.
        model_name: Registry name of the tuned model.
        best_params: The winning parameters, e.g. ``best_params`` of an
            `OptimizationResult`. None or an empty mapping uses the registry
            defaults, with the seed reapplied.

    Returns:
        An unfitted `preprocessor -> estimator` `Pipeline`.

    Raises:
        OptimizationDataError: The model name is unknown to the registry.
    """
    parameters = _with_fixed_seed(str(model_name), dict(best_params or {}))
    return build_model_pipeline(preprocessing_config, str(model_name), parameters)


# --------------------------------------------------------------------------- #
# Study construction
# --------------------------------------------------------------------------- #
def build_sampler(seed: int = config.OPTUNA_RANDOM_SEED) -> "optuna.samplers.BaseSampler":
    """Build the deterministic Optuna sampler.

    A seeded TPESampler is what makes the search reproducible: with the same
    seed Optuna suggests the same sequence of configurations, so the best
    configuration found is the same on every run.

    Args:
        seed: Sampler seed.

    Returns:
        A seeded `TPESampler`.

    Raises:
        OptimizationDataError: The sampler name in the configuration is unknown.
    """
    name = config.OPTUNA_SAMPLER_NAME.lower()
    if name != "tpe":
        raise OptimizationDataError(
            f"Unsupported Optuna sampler '{config.OPTUNA_SAMPLER_NAME}'. "
            "Supported: 'tpe'."
        )
    return optuna.samplers.TPESampler(seed=int(seed))


def build_study(
    model_name: str,
    *,
    seed: int = config.OPTUNA_RANDOM_SEED,
) -> "optuna.Study":
    """Create the in-memory Optuna study for one model.

    The study uses the default in-memory storage on purpose: it is discarded when
    the agent finishes, and a live `Study` never reaches the shared state. MLflow
    tracking (Phase 10) will replace this storage later.

    Args:
        model_name: Display name, used to make the study name readable.
        seed: Sampler seed.

    Returns:
        A new `optuna.Study` in the maximizing direction.
    """
    study_name = "opt_" + "_".join(str(model_name).lower().split())
    return optuna.create_study(
        study_name=study_name,
        direction=config.OPTUNA_DIRECTION,
        sampler=build_sampler(seed),
    )


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def build_objective(
    *,
    model_name: str,
    preprocessing_config: PreprocessingConfig,
    features: pd.DataFrame,
    target: pd.Series,
    cv_config: CrossValidationConfig,
    problem_type: Optional[str],
) -> Any:
    """Build the Optuna objective for one model.

    The returned callable receives the **training** rows only. For every trial it
    builds a fresh unfitted Pipeline, suggests a configuration, and cross-validates
    it with the Phase 4 conventions (`StratifiedKFold`, `shuffle=True`,
    `random_state=RANDOM_STATE`).

    Because the pipeline is unfitted and `cross_validate` clones it per fold, the
    imputer, encoder, scaler and estimator are refitted on each fold's training
    rows. Preprocessing is therefore *inside* the cross-validation, not applied to
    the dataset beforehand.

    Args:
        model_name: Display name from the registry.
        preprocessing_config: Phase 3 configuration, used by value only.
        features: **Training** feature frame (the hold-out rows are not passed).
        target: **Training** target.
        cv_config: Fold configuration from the Phase 4 helper.
        problem_type: Value from `ProblemType`, used to restrict the solvers.

    Returns:
        A callable suitable for `study.optimize`.

    Raises:
        OptimizationDataError: Cross-validation cannot run safely on this data.
    """
    # Validated before the first trial instead of on every trial: an unsupported
    # problem type would otherwise fail every configuration identically and waste
    # the whole budget before saying anything.
    _require_kind(model_name, problem_type)

    if not cv_config.enabled or cv_config.n_splits < config.CV_MIN_FOLDS:
        raise OptimizationDataError(
            "Hyperparameter optimization needs a usable cross-validation, but the "
            "configuration is disabled: "
            f"{cv_config.adjusted_reason or 'no reason recorded'}. Optimizing on a "
            "single split would tune the model on the same rows it is scored on, so "
            "the search is not run."
        )

    # The Phase 4 helper protects against a class that is too small to stratify,
    # but a target that holds a single class passes it and still cannot produce a
    # meaningful F1: every fold would score 0.0 or 1.0 by accident, and the search
    # would happily maximise the accident.
    distinct = sorted({str(item) for item in pd.unique(target.to_numpy())})
    if len(distinct) < 2:
        raise OptimizationDataError(
            "Hyperparameter optimization needs at least two classes in the training "
            f"target, but it contains {len(distinct)} ({', '.join(distinct) or 'none'}). "
            "F1 cannot be scored on a single-class target, so the search is not run."
        )

    splitter = StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=cv_config.shuffle,
        random_state=cv_config.random_state,
    )

    def objective(trial: "optuna.Trial") -> float:
        """Score one suggested configuration by cross-validated weighted F1."""
        parameters = suggest_parameters(trial, model_name, problem_type)
        pipeline = build_model_pipeline(
            preprocessing_config, model_name, _with_fixed_seed(model_name, parameters)
        )
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring=config.OPTUNA_OBJECTIVE_METRIC,
            n_jobs=1,             # sequential keeps the search reproducible
            error_score="raise",  # a broken configuration fails the trial loudly
        )
        return float(sum(scores["test_score"]) / len(scores["test_score"]))

    return objective


# --------------------------------------------------------------------------- #
# Trial records and search execution
# --------------------------------------------------------------------------- #
def build_trial_record(
    frozen_trial: Any,
    model_name: str,
    error_messages: Optional[Dict[int, str]] = None,
) -> TrialRecord:
    """Reduce one Optuna `FrozenTrial` to a JSON-safe `TrialRecord`.

    The live trial object is not stored: it holds a reference to the study
    storage. Only the trial number, the suggested parameters, the objective value,
    the state, the duration and a recorded error message survive.

    Args:
        frozen_trial: `optuna.trial.FrozenTrial` from a finished study.
        model_name: Display name, stored on the record.
        error_messages: Trial number to error message, filled in by
            `optimize_model` because Optuna does not expose the exception.

    Returns:
        A `TrialRecord`.

    Raises:
        OptimizationDataError: The trial cannot be read.
    """
    try:
        state = frozen_trial.state.name
        params = dict(frozen_trial.params)
        number = int(frozen_trial.number)
    except AttributeError as error:
        raise OptimizationDataError(
            f"Not a usable Optuna trial object: {type(frozen_trial).__name__}"
        ) from error

    value = frozen_trial.value if state == TrialState.COMPLETE.value else None
    if value is not None:
        value = float(value)

    duration = 0.0
    try:
        if frozen_trial.duration is not None:
            duration = round(float(frozen_trial.duration.total_seconds()), 6)
    except Exception:  # pragma: no cover - defensive, never hidden
        logger.debug("Could not read the duration of trial %s", number)
        duration = 0.0

    # A failed trial must always carry a reason. Optuna records the state but not
    # the exception, so `error_messages` is the only source; when even that is
    # missing the record says so explicitly instead of looking like a completed
    # trial with no score.
    message = (error_messages or {}).get(number, "")
    if state == TrialState.FAIL.value and not message:
        message = (
            "the trial failed and no error message was captured "
            "(Optuna does not expose the exception)"
        )

    return TrialRecord(
        trial_number=number,
        model_name=model_name,
        params=params,
        objective_value=value,
        state=state,
        duration=duration,
        error_message=message,
    )


def summarize_trials(trials: Sequence[TrialRecord]) -> Dict[str, int]:
    """Count trials by state.

    Returns:
        A dictionary with `total`, `completed`, `failed` and `pruned` counts.
    """
    completed = sum(1 for item in trials if item.state == TrialState.COMPLETE.value)
    failed = sum(1 for item in trials if item.state == TrialState.FAIL.value)
    pruned = sum(1 for item in trials if item.state == TrialState.PRUNED.value)
    return {
        "total": len(trials),
        "completed": completed,
        "failed": failed,
        "pruned": pruned,
    }


def optimize_model(
    *,
    model_name: str,
    preprocessing_config: PreprocessingConfig,
    features: pd.DataFrame,
    target: pd.Series,
    cv_config: CrossValidationConfig,
    problem_type: Optional[str],
    n_trials: int = config.OPTUNA_N_TRIALS,
    seed: int = config.OPTUNA_RANDOM_SEED,
    timeout: Optional[int] = config.OPTUNA_TIMEOUT,
    baseline: Optional[ExperimentResult] = None,
) -> OptimizationResult:
    """Optimize one model and return its `OptimizationResult`.

    Failure isolation works at two levels:

    * a trial that raises is marked failed by Optuna (`catch=(Exception,)`), its
      message is recorded on the `TrialRecord` and the search continues;
    * a search that produces no completed trial at all returns a result with
      status `failed` and the reason, so the caller can continue with the next
      model instead of aborting the phase.

    Args:
        model_name: Display name from the registry.
        preprocessing_config: Phase 3 configuration (used by value only).
        features: **Training** feature frame.
        target: **Training** target.
        cv_config: Fold configuration for the training split.
        problem_type: Value from `ProblemType`.
        n_trials: Trial budget for this model. Must be at least 1; a smaller
            value is a caller mistake and raises rather than being rounded up.
        seed: Sampler and estimator seed.
        timeout: Optional wall-clock limit in seconds.
        baseline: The Phase 4 `ExperimentResult`, used to fill in the baseline
            side of the comparison.

    Returns:
        An `OptimizationResult`, never None. A failed search is a result with
        status `failed`, so the caller never has to special-case it.

    Raises:
        OptimizationToolError: The model is not optimizable, or the trial budget
            is smaller than one.
    """
    # Validated before anything is built: an unknown model or an unsupported
    # problem type is a caller error, not a search result.
    _require_kind(model_name, problem_type)
    definition = get_model_definition(model_name)
    trial_budget = int(n_trials)
    if trial_budget < 1:
        # Silently running one trial instead of zero would report a search that
        # the caller never asked for.
        raise OptimizationToolError(
            f"The trial budget for {model_name} must be at least 1, but it is "
            f"{n_trials!r}. Set utils.config.OPTUNA_N_TRIALS or pass n_trials."
        )
    result = OptimizationResult(
        model_name=model_name,
        model_type=definition.model_type,
        objective_metric=config.OPTUNA_OBJECTIVE_METRIC,
        cv_n_splits=cv_config.n_splits if cv_config.enabled else 0,
        cv_adjustment_reason=cv_config.adjusted_reason,
        search_space_description=describe_search_space(model_name, problem_type),
    )
    if baseline is not None:
        result.baseline_cv_f1_mean = baseline.cv_f1_mean
        result.baseline_cv_accuracy_mean = baseline.cv_accuracy_mean
        result.baseline_holdout_f1_score = baseline.f1_score
        result.baseline_holdout_accuracy = baseline.accuracy
        result.baseline_parameters = dict(baseline.parameters or {})

    try:
        objective = build_objective(
            model_name=model_name,
            preprocessing_config=preprocessing_config,
            features=features,
            target=target,
            cv_config=cv_config,
            problem_type=problem_type,
        )
    except OptimizationDataError as error:
        result.status = OptimizationStatus.FAILED.value
        result.reason = str(error)
        result.error_message = str(error)
        logger.warning("Optimization of %s not started: %s", model_name, error)
        return result

    # The message is captured here because Optuna records the state of a failed
    # trial but not the exception that caused it. Wrapping the objective is the
    # only way to keep the reason instead of logging an empty failure.
    error_messages: Dict[int, str] = {}

    def recording_objective(trial: "optuna.Trial") -> float:
        try:
            return objective(trial)
        except Exception as error:
            error_messages[int(trial.number)] = describe_optimization_error(error)
            raise

    study = build_study(model_name, seed=seed)

    started = time.perf_counter()
    try:
        study.optimize(
            recording_objective,
            n_trials=trial_budget,
            timeout=timeout,
            # Isolation: a broken configuration fails its own trial and the
            # search continues. Every message is recorded above, so this does not
            # hide anything.
            catch=(Exception,),
            show_progress_bar=config.OPTUNA_SHOW_PROGRESS_BAR,
        )
    except Exception as error:
        # Reaching this means Optuna itself could not continue (for example an
        # unusable storage). The reason is reported instead of being swallowed.
        result.status = OptimizationStatus.FAILED.value
        result.reason = describe_optimization_error(error)
        result.error_message = result.reason
        result.optimization_time = round(time.perf_counter() - started, 6)
        logger.warning("Optimization of %s stopped early: %s", model_name, result.reason)
        return result
    result.optimization_time = round(time.perf_counter() - started, 6)

    result.trials = [
        build_trial_record(item, model_name, error_messages) for item in study.trials
    ]
    counts = summarize_trials(result.trials)
    result.total_trials = counts["total"]
    result.completed_trials = counts["completed"]
    result.failed_trials = counts["failed"]
    result.pruned_trials = counts["pruned"]

    if not counts["completed"]:
        result.status = OptimizationStatus.FAILED.value
        result.reason = (
            f"every one of the {counts['total']} trial(s) failed, so no configuration "
            "could be scored"
        )
        first_error = next(
            (item.error_message for item in result.trials if item.error_message), ""
        )
        result.error_message = (
            f"{result.reason}. First recorded error: {first_error}"
            if first_error
            else result.reason
        )
        logger.warning("Optimization of %s produced no usable trial", model_name)
        return result

    best = study.best_trial
    result.status = OptimizationStatus.COMPLETED.value
    result.best_params = dict(best.params)
    result.best_trial_number = int(best.number)
    result.optimized_cv_f1_mean = float(study.best_value)

    # The accuracy of the winning configuration is measured with the same folds so
    # the baseline/optimized pair can also be compared on the secondary metric.
    result.optimized_cv_accuracy_mean = _score_winning_configuration(
        model_name=model_name,
        preprocessing_config=preprocessing_config,
        parameters=result.best_params,
        features=features,
        target=target,
        cv_config=cv_config,
    )

    improvement, percentage, note = calculate_improvement(
        result.baseline_cv_f1_mean, result.optimized_cv_f1_mean
    )
    result.improvement = improvement
    result.improvement_percentage = percentage
    result.improvement_note = note
    result.improvement_direction = classify_improvement(improvement)

    if counts["failed"]:
        result.notes.append(
            f"{counts['failed']} of {counts['total']} trial(s) failed and were excluded "
            "from the comparison; each failure is recorded with its error message."
        )
    result.notes.append(config.OPTUNA_TRIAL_BUDGET_NOTE.format(trials=counts["total"]))
    result.notes.append(config.OPTUNA_SELECTION_NOTE)

    logger.info(
        "Optimized %s: best CV %s after %d trial(s) (%d failed)",
        model_name,
        _fmt(result.optimized_cv_f1_mean),
        counts["total"],
        counts["failed"],
    )
    return result


def _score_winning_configuration(
    *,
    model_name: str,
    preprocessing_config: PreprocessingConfig,
    parameters: Dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    cv_config: CrossValidationConfig,
) -> Optional[float]:
    """Return the mean cross-validated accuracy of the winning configuration.

    The search optimizes F1, but the project's ranking rule uses accuracy as the
    secondary metric. Re-scoring the *already selected* configuration with the
    same folds adds no selection risk: the parameters are fixed at this point, and
    the training rows are still the only rows used.
    """
    try:
        pipeline = build_model_pipeline(
            preprocessing_config, model_name, _with_fixed_seed(model_name, parameters)
        )
        splitter = StratifiedKFold(
            n_splits=cv_config.n_splits,
            shuffle=cv_config.shuffle,
            random_state=cv_config.random_state,
        )
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring="accuracy",
            n_jobs=1,
            error_score="raise",
        )
    except (PreprocessingError, ModelToolError, TrainingToolError, ValueError, TypeError) as error:
        logger.warning(
            "Could not cross-validate the accuracy of the tuned %s: %s",
            model_name,
            error,
        )
        return None
    return round(float(sum(scores["test_score"]) / len(scores["test_score"])), 6)


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def select_optimization_candidates(
    experiments: Sequence[Any],
    top_models: int = config.OPTUNA_TOP_MODELS,
    problem_type: Optional[str] = None,
) -> List[str]:
    """Choose the models to optimize, deterministically.

    The rule is the Phase 5 ranking rule applied to the baseline experiments:
    only successful (or partial) experiments with metrics are considered, they are
    sorted by F1 descending, then accuracy descending, then model name ascending.
    The first `top_models` names are selected.

    Ties therefore resolve to the same model on every run and on every machine,
    and a model without a search space is skipped instead of being silently
    attempted.

    Args:
        experiments: `ExperimentResult` records from the Training Agent.
        top_models: How many models to optimize. Values below 1 are treated as 1,
            because a run that optimizes nothing cannot produce a comparison.
        problem_type: Optional problem type; when given, models that do not
            support it are excluded.

    Returns:
        The selected model names, in ranking order.
    """
    limit = max(1, int(top_models or 1))
    usable = [
        item
        for item in experiments or []
        if getattr(item, "status", "")
        in (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
        and getattr(item, "f1_score", None) is not None
    ]
    if problem_type:
        usable = [
            item
            for item in usable
            if model_tools.supports_problem_type(str(item.model_name), problem_type)
        ]

    ordered = sorted(
        usable,
        key=lambda item: (
            -float(getattr(item, "f1_score", 0.0) or 0.0),
            -float(getattr(item, "accuracy", 0.0) or 0.0),
            str(getattr(item, "model_name", "")),
        ),
    )

    selected: List[str] = []
    for item in ordered:
        name = str(item.model_name)
        if is_optimizable(name) and name not in selected:
            selected.append(name)
        if len(selected) >= limit:
            break
    return selected


def build_candidate_selection_rule(
    top_models: int = config.OPTUNA_TOP_MODELS,
    problem_type: Optional[str] = None,
) -> str:
    """Describe the candidate selection rule in one readable sentence."""
    return (
        f"the top {max(1, int(top_models or 1))} baseline model(s) of the Phase 5 "
        f"ranking for {problem_type or 'the detected problem type'}: only successful "
        "experiments with metrics are considered, ordered by f1_score (descending), "
        "then accuracy (descending), then model name (ascending); a model without a "
        "search space is skipped"
    )


# --------------------------------------------------------------------------- #
# Improvement
# --------------------------------------------------------------------------- #
def calculate_improvement(
    baseline_value: Optional[float],
    optimized_value: Optional[float],
) -> Tuple[Optional[float], Optional[float], str]:
    """Compute the absolute and relative improvement of a tuned model.

    The percentage is only reported when the denominator is safe. A baseline of
    exactly zero (or below `OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE`) makes the
    ratio undefined, and dividing by it would produce an infinite number or a
    division error; in that case the percentage is withheld and the reason is
    returned so it can be recorded instead of being guessed.

    Args:
        baseline_value: The baseline score.
        optimized_value: The optimized score.

    Returns:
        An `(improvement, percentage, note)` tuple. `improvement` and `percentage`
        are None when the corresponding value cannot be computed.
    """
    if baseline_value is None or optimized_value is None:
        return None, None, (
            "no improvement could be computed because one of the two cross-validation "
            "scores is missing"
        )

    improvement = float(optimized_value) - float(baseline_value)

    denominator = abs(float(baseline_value))
    if denominator < config.OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE:
        return improvement, None, (
            "the percentage improvement is not reported because the baseline score is "
            f"{_fmt(baseline_value)}, which would make the ratio meaningless; the "
            "absolute difference is reported instead"
        )

    percentage = round(improvement / denominator * 100.0, 4)
    return improvement, percentage, (
        f"percentage relative to the baseline cross-validation score of "
        f"{_fmt(baseline_value)}"
    )


def classify_improvement(improvement: Optional[float]) -> str:
    """Classify a measured improvement.

    This describes a **measurement**, not a verdict. A degraded score is a normal
    outcome of a small search and is never turned into a claim about the model.
    """
    if improvement is None:
        return ImprovementDirection.NOT_AVAILABLE.value
    if improvement > 0:
        return ImprovementDirection.IMPROVED.value
    if improvement < 0:
        return ImprovementDirection.DEGRADED.value
    return ImprovementDirection.UNCHANGED.value


# --------------------------------------------------------------------------- #
# Hold-out evaluation of the tuned pipeline
# --------------------------------------------------------------------------- #
def evaluate_tuned_model(
    *,
    result: OptimizationResult,
    preprocessing_config: PreprocessingConfig,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
) -> None:
    """Measure one tuned model once on the untouched hold-out split.

    This runs **after** the search: the hyperparameters are already fixed, so the
    hold-out split cannot influence them. The split is used to report the tuned
    model next to its baseline, and the result is stored on the `result` so the
    difference between the two hold-out scores is visible too.

    Nothing is fitted on the test rows: the pipeline is fitted on the training
    rows and only transforms the test rows, exactly as in Phase 4.

    Args:
        result: The `OptimizationResult` to update in place.
        preprocessing_config: Phase 3 configuration.
        features_train: Training feature frame.
        target_train: Training target.
        features_test: Hold-out feature frame.
        target_test: Hold-out target.
    """
    from tools.training_tools import evaluate_classifier  # local import: avoids a cycle

    if result.status != OptimizationStatus.COMPLETED.value:
        return
    if not features_test.shape[0]:
        result.notes.append(
            "The hold-out split is empty, so the tuned model was not evaluated on it."
        )
        return

    started = time.perf_counter()
    try:
        pipeline = build_model_pipeline(
            preprocessing_config, result.model_name, _with_fixed_seed(result.model_name, result.best_params)
        )
        pipeline.fit(features_train, target_train)
        predictions = pipeline.predict(features_test)
        metrics = evaluate_classifier(target_test, predictions)
    except (
        PreprocessingError,
        ModelToolError,
        TrainingToolError,
        ValueError,
        TypeError,
    ) as error:
        # A tuned model that cannot be scored on the hold-out split does not
        # invalidate the search result, so the reason is recorded and the
        # cross-validation numbers stay usable.
        message = describe_optimization_error(error)
        result.notes.append(
            f"The tuned {result.model_name} could not be evaluated on the hold-out "
            f"split: {message}"
        )
        logger.warning("Hold-out evaluation of %s failed: %s", result.model_name, message)
        return

    result.holdout_evaluated = True
    result.holdout_f1_score = round(float(metrics["f1_score"]), 6)
    result.holdout_accuracy = round(float(metrics["accuracy"]), 6)
    if result.baseline_holdout_f1_score is not None:
        result.holdout_f1_delta = round(
            float(result.holdout_f1_score) - float(result.baseline_holdout_f1_score), 6
        )
    result.notes.append(config.OPTUNA_HOLDOUT_NOTE)
    logger.debug(
        "Evaluated tuned %s on %d hold-out row(s) in %.4fs",
        result.model_name,
        features_test.shape[0],
        time.perf_counter() - started,
    )


# --------------------------------------------------------------------------- #
# Ranking of optimized models
# --------------------------------------------------------------------------- #
OPTIMIZED_RANKING_RULE = (
    "primary: hold-out f1_score (descending); secondary: hold-out accuracy "
    "(descending); model name (ascending) as the final tie-break; models without "
    "hold-out metrics are excluded"
)


def rank_optimized_results(results: Sequence[OptimizationResult]) -> List[OptimizationResult]:
    """Rank the tuned models with the project's documented ranking rule.

    The rule is the Phase 4 / Phase 5 one - F1, then accuracy, then model name -
    so the baseline ranking and the optimized ranking are directly comparable.
    Applying it to the hold-out score means the hold-out split takes part in the
    *final choice of winner*, which is stated in the report and in
    `OPTUNA_LIMITATION_HOLDOUT_RANKING`; it never takes part in the search.

    Args:
        results: `OptimizationResult` records.

    Returns:
        The usable results, best first. Failed or skipped results are excluded.

    Raises:
        OptimizationDataError: A record is missing both scores, so it cannot be
            ranked; this would silently produce a wrong order.
    """
    usable = [
        item
        for item in results or []
        if item.status == OptimizationStatus.COMPLETED.value
        and item.holdout_f1_score is not None
    ]
    for item in usable:
        if item.holdout_accuracy is None:
            raise OptimizationDataError(
                f"'{item.model_name}' has a hold-out F1 score but no hold-out accuracy, "
                "so the ranking rule cannot be applied to it."
            )
    return sorted(
        usable,
        key=lambda item: (
            -float(item.holdout_f1_score or 0.0),
            -float(item.holdout_accuracy or 0.0),
            str(item.model_name),
        ),
    )


def build_optimization_table(results: Sequence[OptimizationResult]) -> str:
    """Render the baseline-vs-optimized comparison as a fixed-width table."""
    header = (
        f"{'model':<26}{'baseline cv f1':>16}{'optimized cv f1':>17}{'improvement':>13}"
        f"{'baseline holdout':>17}{'optimized holdout':>19}{'trials':>8}{'failed':>8}"
    )
    separator = "-" * len(header)
    rows = [header, separator]
    for item in results or []:
        rows.append(
            f"{item.model_name:<26}{_fmt(item.baseline_cv_f1_mean):>16}"
            f"{_fmt(item.optimized_cv_f1_mean):>17}{_fmt(item.improvement):>13}"
            f"{_fmt(item.baseline_holdout_f1_score):>17}"
            f"{_fmt(item.holdout_f1_score):>19}{str(item.total_trials):>8}"
            f"{str(item.failed_trials):>8}"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Experiment / trial counts
# --------------------------------------------------------------------------- #
def count_optimization_work(
    experiments: Sequence[Any],
    results: Sequence[OptimizationResult],
) -> OptimizationCounts:
    """Count baseline experiments and optimization trials separately.

    A *model experiment* is one model trained once with default parameters. An
    Optuna *trial* is one hyperparameter configuration scored by cross-validation
    on the training split. The two are different quantities: Phase 4 evaluated 5
    models, Phase 6 may have tried 40 configurations for 2 of them. Adding them
    into one "experiments" number would describe neither, so both are reported.

    Args:
        experiments: `ExperimentResult` records from the Training Agent.
        results: `OptimizationResult` records from the optimization step.

    Returns:
        An `OptimizationCounts` record.
    """
    attempts = len(experiments or [])
    usable = sum(
        1
        for item in experiments or []
        if getattr(item, "status", "")
        in (ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value)
    )
    completed = 0
    failed = 0
    pruned = 0
    failed_optimizations: List[str] = []
    models_with_failed_trials: List[str] = []

    for item in results or []:
        completed += item.completed_trials
        failed += item.failed_trials
        pruned += item.pruned_trials
        if item.status == OptimizationStatus.COMPLETED.value:
            if item.failed_trials:
                models_with_failed_trials.append(item.model_name)
        elif item.model_name:
            failed_optimizations.append(item.model_name)

    return OptimizationCounts(
        baseline_models_attempted=attempts,
        baseline_successful_experiments=usable,
        baseline_failed_experiments=attempts - usable,
        optimized_models_attempted=len(results or []),
        successful_optimizations=sum(
            1
            for item in results or []
            if item.status == OptimizationStatus.COMPLETED.value
        ),
        failed_optimizations=len(failed_optimizations),
        completed_trials=completed,
        failed_trials=failed,
        pruned_trials=pruned,
        total_trials=completed + failed + pruned,
        failed_trial_models=models_with_failed_trials,
        failed_model_names=failed_optimizations,
        note=(
            f"{usable} of {attempts} baseline model(s) produced usable metrics, and "
            f"{len(results or [])} of them were tuned with "
            f"{completed + failed + pruned} hyperparameter configuration(s) evaluated "
            f"({completed} completed, {failed} failed, {pruned} pruned). A model "
            "experiment and an Optuna trial are different quantities and are not "
            "added together."
        ),
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _overall_status(results: Sequence[OptimizationResult]) -> str:
    """Derive the phase status from the per-model results.

    A failed *trial* is a normal search outcome and does not make the run
    partial; only a model whose search produced nothing usable does.
    """
    items = list(results or [])
    if not items:
        return OptimizationStatus.SKIPPED.value
    completed = sum(
        1 for item in items if item.status == OptimizationStatus.COMPLETED.value
    )
    if completed == 0:
        return OptimizationStatus.FAILED.value
    if completed == len(items):
        return OptimizationStatus.COMPLETED.value
    return OptimizationStatus.PARTIAL.value


def generate_optimization_summary(
    results: Sequence[OptimizationResult],
    baseline: Optional[ModelRecommendation] = None,
) -> str:
    """Write a single-paragraph, single-line summary of the optimization."""
    items = list(results or [])
    completed = [item for item in items if item.status == OptimizationStatus.COMPLETED.value]
    if not completed:
        return (
            "No model produced a usable tuned configuration, so the Phase 5 baseline "
            "result remains the only usable result of this run."
        )

    improved = sum(
        1 for item in completed if item.improvement_direction == ImprovementDirection.IMPROVED.value
    )
    best = rank_optimized_results(completed)
    winner = best[0] if best else None
    runner_up = best[1] if len(best) > 1 else None
    margin = (
        f", a margin of {abs(winner.holdout_f1_score - runner_up.holdout_f1_score):.4f} "
        f"over {runner_up.model_name}"
        if runner_up is not None
        else ""
    )
    baseline_phrase = (
        f"the best baseline was {baseline.best_baseline_model} with a hold-out F1 of "
        f"{_fmt(baseline.f1_score)}"
        if baseline is not None
        else "no baseline recommendation was available"
    )
    return (
        f"{len(completed)} of {len(items)} optimized model(s) produced a usable "
        f"configuration, {improved} of them with a higher cross-validation F1 than their "
        f"baseline; the best optimized model is {winner.model_name} with a hold-out F1 "
        f"of {_fmt(winner.holdout_f1_score)} (cross-validation F1 "
        f"{_fmt(winner.optimized_cv_f1_mean)} against a baseline "
        f"{_fmt(winner.baseline_cv_f1_mean)}, improvement {_fmt(winner.improvement)}"
        f"{margin}); {baseline_phrase}."
    )


def generate_optimized_recommendation(
    ranked: Sequence[OptimizationResult],
) -> Optional[OptimizedModelRecommendation]:
    """Build the recommendation for the best tuned model.

    Returns:
        An `OptimizedModelRecommendation`, or None when nothing was optimized.
    """
    if not ranked:
        return None
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    reason = (
        f"highest hold-out f1_score ({_fmt(winner.holdout_f1_score)}) among "
        f"{len(ranked)} tuned model(s); ties are broken on accuracy"
    )
    if runner_up is not None:
        reason += (
            f", ahead of {runner_up.model_name} "
            f"({_fmt(runner_up.holdout_f1_score)})"
        )
    return OptimizedModelRecommendation(
        best_optimized_model=winner.model_name,
        model_type=winner.model_type,
        baseline_model_name=winner.model_name,
        selection_reason=reason,
        performance_summary=(
            f"{winner.model_name} reached a cross-validation F1 of "
            f"{_fmt(winner.optimized_cv_f1_mean)} against a baseline "
            f"{_fmt(winner.baseline_cv_f1_mean)} (improvement "
            f"{_fmt(winner.improvement)}) and a hold-out F1 of "
            f"{_fmt(winner.holdout_f1_score)} against a baseline "
            f"{_fmt(winner.baseline_holdout_f1_score)} (difference "
            f"{_fmt(winner.holdout_f1_delta)})."
        ),
        baseline_cv_f1_mean=winner.baseline_cv_f1_mean,
        optimized_cv_f1_mean=winner.optimized_cv_f1_mean,
        improvement=winner.improvement,
        improvement_percentage=winner.improvement_percentage,
        improvement_direction=winner.improvement_direction,
        baseline_holdout_f1_score=winner.baseline_holdout_f1_score,
        holdout_f1_score=winner.holdout_f1_score,
        holdout_f1_delta=winner.holdout_f1_delta,
        best_params=dict(winner.best_params),
        best_trial_number=winner.best_trial_number,
        total_trials=winner.total_trials,
        cv_n_splits=winner.cv_n_splits,
        optimization_time=winner.optimization_time,
        selection_note=config.OPTUNA_SELECTION_NOTE,
    )


def generate_optimization_observations(
    *,
    results: Sequence[OptimizationResult],
    ranked: Sequence[OptimizationResult],
    counts: Optional[OptimizationCounts],
    selected: Sequence[str],
    n_trials: int,
) -> List[str]:
    """List what was measured, as measurements.

    Every entry states a number. Interpretations belong in the recommendations,
    and hypotheses belong in the notes, so an observation can never be read as a
    conclusion.
    """
    observations: List[str] = []
    completed = [
        item for item in results if item.status == OptimizationStatus.COMPLETED.value
    ]
    observations.append(
        f"{len(selected)} of the baseline models were selected for optimization "
        f"({', '.join(selected) if selected else 'none'}) with a budget of {n_trials} "
        "trial(s) each."
    )
    for item in results:
        if item.status != OptimizationStatus.COMPLETED.value:
            observations.append(
                f"{item.model_name}: optimization produced no usable configuration "
                f"({item.reason or item.error_message or 'no reason recorded'})."
            )
            continue
        observations.append(
            f"{item.model_name}: cross-validation F1 {_fmt(item.optimized_cv_f1_mean)} "
            f"against a baseline {_fmt(item.baseline_cv_f1_mean)} (improvement "
            f"{_fmt(item.improvement)}, direction {item.improvement_direction}), "
            f"{item.completed_trials} completed and {item.failed_trials} failed "
            f"trial(s) out of {item.total_trials}, best trial "
            f"{item.best_trial_number}, {_fmt(item.optimization_time)} s."
        )
        if item.holdout_evaluated:
            observations.append(
                f"{item.model_name}: hold-out F1 {_fmt(item.holdout_f1_score)} against a "
                f"baseline {_fmt(item.baseline_holdout_f1_score)} (difference "
                f"{_fmt(item.holdout_f1_delta)}); the hold-out split was scored once, "
                "after the search."
            )
        else:
            observations.append(
                f"{item.model_name}: no hold-out metrics were measured for the tuned "
                "configuration."
            )
    if ranked:
        spread = float(ranked[0].holdout_f1_score or 0.0) - float(ranked[-1].holdout_f1_score or 0.0)
        observations.append(
            f"The hold-out F1 spread between the best and the weakest tuned model is "
            f"{spread:.4f}."
        )
    if counts is not None:
        observations.append(counts.note)
    return observations


def generate_optimization_limitations(
    *,
    results: Sequence[OptimizationResult],
    ranked: Sequence[OptimizationResult],
    counts: Optional[OptimizationCounts],
    n_trials: int,
    test_samples: int,
    candidate_total: int,
    selected_count: int,
) -> List[str]:
    """List what this optimization run cannot claim.

    The list is data-driven: a limitation appears only when it applies to the run
    that was actually executed.
    """
    completed = [
        item for item in results if item.status == OptimizationStatus.COMPLETED.value
    ]
    limitations = [
        config.OPTUNA_LIMITATION_TRIAL_BUDGET.format(trials=n_trials),
        config.OPTUNA_LIMITATION_NO_REPEATED_SEARCH,
    ]

    improved = [
        item
        for item in completed
        if item.improvement_direction == ImprovementDirection.IMPROVED.value
    ]
    if improved:
        limitations.append(config.OPTUNA_LIMITATION_SELECTION_ON_CV)
        limitations.append(config.OPTUNA_IMPROVEMENT_CAUTION)

    if ranked:
        # The ranking rule reads the hold-out scores, so the winner's hold-out
        # numbers are not an untouched estimate any more. Stating it is the
        # difference between a comparison and an overclaim.
        limitations.append(config.OPTUNA_LIMITATION_HOLDOUT_RANKING)
        limitations.append(config.OPTUNA_LIMITATION_SINGLE_SPLIT)

    if test_samples and test_samples < 20:
        limitations.append(
            config.OPTUNA_LIMITATION_SMALL_TEST_SPLIT.format(
                test_samples=test_samples, step=1 / test_samples
            )
        )

    if candidate_total > selected_count:
        limitations.append(
            config.OPTUNA_LIMITATION_UNTUNED_MODELS.format(
                selected=selected_count, candidates=candidate_total
            )
        )

    if any(
        item.improvement_direction
        in (ImprovementDirection.DEGRADED.value, ImprovementDirection.UNCHANGED.value)
        for item in completed
    ):
        limitations.append(config.OPTUNA_NO_IMPROVEMENT_NOTE)

    if any(item.optimization_time > 0 for item in completed):
        limitations.append(config.OPTUNA_SINGLE_RUN_TIMING_NOTE)

    if counts is not None and counts.failed_trials:
        limitations.append(
            f"{counts.failed_trials} trial(s) failed and produced no score; they are "
            "counted and listed with their error messages instead of being ignored."
        )
    if counts is not None and counts.failed_optimizations:
        limitations.append(
            f"{counts.failed_optimizations} selected model(s) could not be optimized at "
            "all, so the comparison covers fewer models than were selected."
        )

    limitations.append(config.OPTUNA_LIMITATION_NO_TRAINING_METRICS)
    limitations.append(config.OPTUNA_LIMITATION_NO_DOMAIN_VALIDATION)
    return limitations


def generate_optimization_recommendations(
    *,
    ranked: Sequence[OptimizationResult],
    recommendation: Optional[OptimizedModelRecommendation],
    results: Sequence[OptimizationResult],
    counts: Optional[OptimizationCounts],
    baseline: Optional[ModelRecommendation],
) -> List[str]:
    """Suggest the next concrete steps, bounded by what Phase 6 can support."""
    items: List[str] = []

    if recommendation is not None and baseline is not None:
        items.append(
            f"Keep both results: {baseline.best_baseline_model} is the best BASELINE "
            f"model and {recommendation.best_optimized_model} is the best OPTIMIZED "
            "model. The baseline is not replaced, because the optimized score was "
            "selected by the search itself."
        )
        items.append(
            f"Use {recommendation.best_optimized_model} with the reported parameters as "
            "the reference configuration for the next phase, and re-run the whole "
            "pipeline whenever the model or the data changes so the comparison stays "
            "reproducible."
        )
    elif baseline is not None:
        items.append(
            f"No tuned model can be recommended yet, so {baseline.best_baseline_model} "
            "remains the model to use."
        )
    else:
        items.append(
            "No model can be recommended: at least one model must produce usable "
            "metrics first."
        )

    degraded = [
        item.model_name
        for item in results
        if item.improvement_direction == ImprovementDirection.DEGRADED.value
    ]
    if degraded:
        items.append(
            "The cross-validation score did not improve for "
            + ", ".join(degraded)
            + ". This is a normal outcome on a small dataset; raise the trial budget "
            "before concluding that tuning cannot help."
        )
    unchanged = [
        item.model_name
        for item in results
        if item.improvement_direction == ImprovementDirection.UNCHANGED.value
    ]
    if unchanged:
        items.append(
            "The cross-validation score is unchanged for "
            + ", ".join(unchanged)
            + ", which suggests the default parameters were already at the top of the "
            "explored region."
        )

    if counts is not None and counts.failed_trials:
        items.append(
            "Inspect the recorded trial failures before trusting the search: a "
            "configuration that fails during cross-validation points at a boundary in "
            "the search space rather than at the data."
        )

    items.append(
        "Re-run the optimization with a different Optuna sampler seed to see whether "
        "the same configuration is found again; a single seeded search is not "
        "statistically reliable."
    )
    return items


def build_optimization_report(
    *,
    experiments: Sequence[Any],
    results: Sequence[OptimizationResult],
    selected: Sequence[str],
    n_trials: int = config.OPTUNA_N_TRIALS,
    top_models: int = config.OPTUNA_TOP_MODELS,
    dataset_name: str = "",
    problem_type: str = "",
    target_column: str = "",
    test_samples: int = 0,
    cv_n_splits: int = 0,
    cv_adjustment_reason: str = "",
    baseline_recommendation: Optional[ModelRecommendation] = None,
) -> OptimizationReport:
    """Assemble the complete Phase 6 optimization report.

    This is the single entry point the Optimization Agent calls once every
    selected model has been searched. It performs no fitting and no search; it
    only ranks, counts, compares and words the results.

    Args:
        experiments: `ExperimentResult` records from the Training Agent.
        results: `OptimizationResult` records, one per selected model.
        selected: Model names chosen for optimization.
        n_trials: Trial budget used per model.
        top_models: How many models were requested.
        dataset_name: Name of the dataset, for the report header.
        problem_type: Detected problem type.
        target_column: Target column name.
        test_samples: Size of the hold-out split, used by the limitations.
        cv_n_splits: Number of folds the objective used.
        cv_adjustment_reason: Why the fold count was reduced, if it was.
        baseline_recommendation: The Phase 5 recommendation, preserved so the
            baseline and the optimized winner are both visible.

    Returns:
        An `OptimizationReport`. A run without a usable optimized model still
        returns a report (status `skipped` or `failed`) with the reason recorded.
    """
    ranked = rank_optimized_results(results)
    counts = count_optimization_work(experiments, results)
    recommendation = generate_optimized_recommendation(ranked)
    selected_names = [str(name) for name in selected or []]
    attempted = {str(item.model_name) for item in results or []}

    notes: List[str] = []
    not_optimized = [
        str(item.model_name)
        for item in experiments or []
        if str(getattr(item, "model_name", "")) not in selected_names
    ]
    if not_optimized:
        notes.append(
            "These baseline models kept their default parameters: "
            + ", ".join(not_optimized)
            + ". Their scores are not a fair comparison with a tuned model."
        )
    if selected_names and not attempted:
        notes.append(
            "The selected models produced no result at all, so no tuned model can be "
            "compared with the baseline."
        )

    if recommendation is not None:
        recommendation.limitations = [
            config.OPTUNA_LIMITATION_TRIAL_BUDGET.format(trials=n_trials),
            config.OPTUNA_LIMITATION_SELECTION_ON_CV,
        ]

    report = OptimizationReport(
        overall_status=_overall_status(results),
        dataset_name=str(dataset_name or ""),
        problem_type=str(problem_type or ""),
        target_column=str(target_column or ""),
        n_trials_per_model=int(n_trials),
        top_models=int(top_models),
        sampler=config.OPTUNA_SAMPLER_NAME,
        sampler_seed=int(config.OPTUNA_RANDOM_SEED),
        objective_metric=config.OPTUNA_OBJECTIVE_METRIC,
        objective_rule=(
            f"the objective is the mean {config.OPTUNA_OBJECTIVE_METRIC} of a "
            "StratifiedKFold cross-validation on the TRAINING split, maximizing it; the "
            "hold-out split is not part of the objective"
        ),
        selection_rule=config.OPTUNA_SELECTION_NOTE,
        candidate_rule=build_candidate_selection_rule(top_models, problem_type or None),
        ranking_rule=OPTIMIZED_RANKING_RULE,
        cv_n_splits=int(cv_n_splits),
        cv_adjustment_reason=str(cv_adjustment_reason or ""),
        holdout_note=config.OPTUNA_HOLDOUT_NOTE,
        selection_note=config.OPTUNA_SELECTION_NOTE,
        models_selected_for_optimization=selected_names,
        models_not_optimized=not_optimized,
        optimization_results=list(results or []),
        best_optimized_model=recommendation,
        best_baseline_model=baseline_recommendation,
        baseline_saved_to=(
            "evaluation_report.best_model"
            if baseline_recommendation is not None
            else "not available"
        ),
        counts=counts,
        observations=generate_optimization_observations(
            results=results,
            ranked=ranked,
            counts=counts,
            selected=selected_names,
            n_trials=n_trials,
        ),
        limitations=generate_optimization_limitations(
            results=results,
            ranked=ranked,
            counts=counts,
            n_trials=n_trials,
            test_samples=test_samples,
            candidate_total=len(experiments or []),
            selected_count=len(selected_names),
        ),
        recommendations=generate_optimization_recommendations(
            ranked=ranked,
            recommendation=recommendation,
            results=results,
            counts=counts,
            baseline=baseline_recommendation,
        ),
        notes=notes,
    )
    # The table is built from the ranked list, not from the candidate order, so
    # the first row is the model the recommendation names. A table in a
    # different order from the ranking is the kind of thing a reader silently
    # misreads.
    report.table_text = build_optimization_table(ranked if ranked else results)
    logger.debug(
        "Built optimization report: status=%s, %d result(s)",
        report.overall_status,
        len(results or []),
    )
    return report


# --------------------------------------------------------------------------- #
# Error helpers
# --------------------------------------------------------------------------- #
def describe_optimization_error(error: Exception) -> str:
    """Turn an exception into a single-line, loggable message."""
    return f"{type(error).__name__}: {error}"


def _fmt(value: Optional[float]) -> str:
    """Format a metric for a table or a sentence, showing '-' when it is missing."""
    return "-" if value is None else f"{float(value):.4f}"


__all__ = [
    "OPTIMIZED_RANKING_RULE",
    "OptimizationDataError",
    "OptimizationToolError",
    "build_candidate_selection_rule",
    "build_objective",
    "build_optimization_report",
    "build_optimization_table",
    "build_sampler",
    "build_study",
    "build_trial_record",
    "calculate_improvement",
    "classify_improvement",
    "count_optimization_work",
    "describe_optimization_error",
    "describe_search_space",
    "evaluate_tuned_model",
    "generate_optimization_limitations",
    "generate_optimization_observations",
    "generate_optimization_recommendations",
    "generate_optimization_summary",
    "generate_optimized_recommendation",
    "get_optimizable_models",
    "is_optimizable",
    "optimize_model",
    "rank_optimized_results",
    "select_optimization_candidates",
    "suggest_parameters",
    "summarize_trials",
]