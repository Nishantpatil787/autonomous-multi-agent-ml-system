"""Controlled model registry (Phase 4).

This module is the **only** place where estimators are created. The system never
generates model classes dynamically and never accepts an arbitrary class from a
user or from an LLM: a model must be present in this registry to be used.

A registry entry pairs a scikit-learn estimator factory with a serialisable
description (`ModelDefinition`) so the same information can be stored in the
shared state and shown in reports without touching the live objects.

Each entry also states whether the model benefits from standardised numerical
features. Distance- and margin-based models (Logistic Regression, KNN, SVM) do;
tree-based models do not, which is why the Training Agent builds two
preprocessing variants (with and without scaling) from the same Phase 3
configuration instead of mutating it.

Model selection here is **deterministic**: no random search, no LLM and no
tuning. The parameters in this file are reproducible baseline values only;
hyperparameter optimization belongs to Phase 6.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import config
from utils.logger import setup_logger
from utils.state import (
    ModelDefinition,
    ModelType,
    PreprocessingConfig,
    ProblemType,
    ScalingStrategy,
    SelectedModel,
)

logger = setup_logger("tools.model_tools")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ModelToolError(Exception):
    """Base class for model registry errors."""


class ModelConfigurationError(ModelToolError):
    """Raised when a model name, problem type or parameter set is invalid."""


class ModelSelectionError(ModelToolError):
    """Raised when the candidate models cannot be determined."""


# Problem types the Phase 4 models can be used for.
SUPPORTED_PROBLEM_TYPES: tuple = (
    ProblemType.BINARY_CLASSIFICATION.value,
    ProblemType.MULTICLASS_CLASSIFICATION.value,
)


# --------------------------------------------------------------------------- #
# Registry construction
# --------------------------------------------------------------------------- #
def build_model_registry() -> Dict[str, ModelDefinition]:
    """Build the controlled registry of supported candidate models.

    Returns:
        A mapping of model name to `ModelDefinition`. The definition holds the
        documented default parameters; the estimator itself is created on demand
        by `build_estimator`, so no live object is ever stored.

    Note:
        The estimator class is captured in a private factory attribute
        (`_factory`) which is never serialised and never reaches `to_dict()`.
    """
    registry: Dict[str, ModelDefinition] = {}

    definitions: List[ModelDefinition] = [
        ModelDefinition(
            name="Logistic Regression",
            model_type=ModelType.LINEAR.value,
            scaling_required=True,
            default_parameters=dict(config.LOGISTIC_REGRESSION_PARAMS),
            supported_problem_types=list(SUPPORTED_PROBLEM_TYPES),
            description=(
                "Linear model with a sigmoid activation. Fast, interpretable and "
                "a good first baseline for binary and multiclass classification."
            ),
        ),
        ModelDefinition(
            name="Decision Tree",
            model_type=ModelType.TREE_BASED.value,
            scaling_required=False,
            default_parameters=dict(config.DECISION_TREE_PARAMS),
            supported_problem_types=list(SUPPORTED_PROBLEM_TYPES),
            description=(
                "Single tree splitting the feature space with thresholds. Does "
                "not need scaling because splits are scale invariant."
            ),
        ),
        ModelDefinition(
            name="Random Forest",
            model_type=ModelType.ENSEMBLE.value,
            scaling_required=False,
            default_parameters=dict(config.RANDOM_FOREST_PARAMS),
            supported_problem_types=list(SUPPORTED_PROBLEM_TYPES),
            description=(
                "Bagged ensemble of decision trees. Usually the strongest "
                "baseline here and robust to unscaled features."
            ),
        ),
        ModelDefinition(
            name="K-Nearest Neighbors",
            model_type=ModelType.INSTANCE_BASED.value,
            scaling_required=True,
            default_parameters=dict(config.KNN_PARAMS),
            supported_problem_types=list(SUPPORTED_PROBLEM_TYPES),
            description=(
                "Classifies a sample by the majority label of its k closest "
                "neighbours. Highly sensitive to feature scale."
            ),
        ),
        ModelDefinition(
            name="Support Vector Machine",
            model_type=ModelType.KERNEL_BASED.value,
            scaling_required=True,
            default_parameters=dict(config.SVM_PARAMS),
            supported_problem_types=list(SUPPORTED_PROBLEM_TYPES),
            description=(
                "Maximum margin classifier with an RBF kernel. Effective in "
                "high dimensions but the slowest baseline on large datasets."
            ),
        ),
    ]

    factories: Dict[str, Callable[[], Any]] = {
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "K-Nearest Neighbors": KNeighborsClassifier,
        "Support Vector Machine": SVC,
    }

    for definition in definitions:
        definition._factory = factories[definition.name]  # type: ignore[attr-defined]
        registry[definition.name] = definition

    logger.debug("Model registry built with %d entries: %s", len(registry), list(registry))
    return registry


# The registry is immutable in practice: it is built once and only read.
MODEL_REGISTRY: Dict[str, ModelDefinition] = build_model_registry()


# --------------------------------------------------------------------------- #
# Lookups and construction
# --------------------------------------------------------------------------- #
def get_supported_models() -> List[str]:
    """Return the names of every supported model, in a stable order."""
    return list(MODEL_REGISTRY)


def get_model_definition(model_name: str) -> ModelDefinition:
    """Return the registry definition of one model.

    Args:
        model_name: Display name, e.g. "Random Forest".

    Returns:
        The `ModelDefinition`.

    Raises:
        ModelConfigurationError: The model is not in the registry.
    """
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as error:
        raise ModelConfigurationError(
            f"Unknown model '{model_name}'. Supported models: {get_supported_models()}"
        ) from error


def build_estimator(model_name: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """Create an unfitted estimator instance for a supported model.

    Only models present in the controlled registry can be built; the estimator
    class is never taken from user input.

    Args:
        model_name: Display name from the registry.
        parameters: Optional overrides for the baseline parameters.

    Returns:
        A fresh, unfitted scikit-learn classifier.

    Raises:
        ModelConfigurationError: Unknown model or unsupported parameter.
    """
    definition = get_model_definition(model_name)
    factory = _get_factory(model_name)

    merged = dict(definition.default_parameters)
    if parameters:
        merged.update(parameters)
    return factory(**merged)


def get_effective_parameters(model_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the baseline parameters merged with the given overrides."""
    definition = get_model_definition(model_name)
    merged = dict(definition.default_parameters)
    if parameters:
        merged.update(parameters)
    return merged


def supports_problem_type(model_name: str, problem_type: Optional[str]) -> bool:
    """Return whether a model can be used for the given problem type."""
    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        return False
    return problem_type in get_model_definition(model_name).supported_problem_types


def get_supported_parameters(model_name: str) -> List[str]:
    """Return the parameter names a model accepts.

    The signature is inspected instead of instantiating the estimator, so an
    invalid parameter is reported as a configuration error rather than as an
    opaque `TypeError` from scikit-learn.

    Args:
        model_name: Display name from the registry.

    Returns:
        The accepted parameter names, sorted.
    """
    factory = _get_factory(model_name)
    parameters = inspect.signature(factory.__init__).parameters
    return sorted(
        name for name, parameter in parameters.items()
        if name != "self" and parameter.kind is not parameter.VAR_KEYWORD
    )


def _get_factory(model_name: str) -> Callable[[], Any]:
    """Return the estimator factory of a registry entry."""
    definition = get_model_definition(model_name)
    factory: Optional[Callable[[], Any]] = getattr(definition, "_factory", None)
    if factory is None:  # pragma: no cover - registry is built with factories
        raise ModelConfigurationError(f"Model '{model_name}' has no estimator factory.")
    return factory


def validate_model_configuration(
    model_name: str,
    problem_type: Optional[str],
    parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate that a model can be trained for a problem type.

    Args:
        model_name: Display name from the registry.
        problem_type: Value from `ProblemType`.
        parameters: Optional parameter overrides.

    Raises:
        ModelConfigurationError: The model is unknown, does not support the
            problem type, or the parameters are not accepted.
    """
    definition = get_model_definition(model_name)

    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ModelConfigurationError(
            f"Problem type '{problem_type}' is not supported in Phase 4. "
            f"Supported: {list(SUPPORTED_PROBLEM_TYPES)}"
        )
    if problem_type not in definition.supported_problem_types:
        raise ModelConfigurationError(
            f"Model '{model_name}' does not support problem type '{problem_type}'."
        )

    effective = get_effective_parameters(model_name, parameters)
    accepted = get_supported_parameters(model_name)
    unknown = sorted(set(effective) - set(accepted))
    if unknown:
        raise ModelConfigurationError(
            f"Model '{model_name}' does not accept parameter(s): {unknown}. "
            f"Accepted: {accepted}"
        )


# --------------------------------------------------------------------------- #
# Scaling-aware preprocessing variants
# --------------------------------------------------------------------------- #
def config_for_scaling(
    base_config: PreprocessingConfig,
    scale_numeric: bool,
) -> PreprocessingConfig:
    """Return a copy of a preprocessing configuration for a scaling variant.

    The Phase 3 configuration is treated as immutable: this function never
    mutates it, it returns a new `PreprocessingConfig`. That keeps the original
    decision in the state consistent while still letting the Training Agent run
    tree-based models without a scaler.

    Args:
        base_config: The configuration produced by the Preprocessing Agent.
        scale_numeric: Whether numerical features must be scaled.

    Returns:
        A new, internally validated `PreprocessingConfig`.

    Raises:
        ModelConfigurationError: The base configuration is missing or broken.
    """
    # Imported here to avoid a circular import at module load time.
    from tools.preprocessing_tools import (  # noqa: WPS433 (runtime import is deliberate)
        describe_pipeline_steps,
        validate_preprocessing_config,
    )

    if base_config is None:
        raise ModelConfigurationError(
            "A preprocessing configuration is required to build a model pipeline."
        )

    variant = PreprocessingConfig(
        numerical_features=list(base_config.numerical_features),
        categorical_features=list(base_config.categorical_features),
        excluded_features=list(base_config.excluded_features),
        target_column=base_config.target_column,
        numerical_imputation=base_config.numerical_imputation,
        categorical_imputation=base_config.categorical_imputation,
        encoding=base_config.encoding,
        encoding_handle_unknown=base_config.encoding_handle_unknown,
        encoding_drop=base_config.encoding_drop,
        encoding_sparse_output=base_config.encoding_sparse_output,
        scale_numeric=bool(scale_numeric),
        scaling_strategy=(
            ScalingStrategy.STANDARD.value if scale_numeric else ScalingStrategy.NONE.value
        ),
        duplicate_handling=base_config.duplicate_handling,
        remainder=base_config.remainder,
        fit_scope=base_config.fit_scope,
    )
    variant.steps = describe_pipeline_steps(variant)

    try:
        validate_preprocessing_config(variant)
    except Exception as error:  # converted into a registry-level error
        raise ModelConfigurationError(
            f"Could not derive a scaling={scale_numeric} preprocessing variant: {error}"
        ) from error
    return variant


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def select_candidate_models(
    problem_type: Optional[str],
    model_names: Optional[Sequence[str]] = None,
    *,
    exclude: Optional[Sequence[str]] = None,
) -> List[SelectedModel]:
    """Choose the candidate models for a problem type.

    The rule is transparent and deterministic: every registered model whose
    `supported_problem_types` contains the problem type is selected. No model is
    added, ranked by score or invented at this stage.

    Args:
        problem_type: Value from `ProblemType`.
        model_names: Optional subset of registry names to consider.
        exclude: Model names to leave out (recorded as a reason).

    Returns:
        The selected models, always in **registry order**, so the result does not
        depend on the order in which a caller lists the candidates.

    Raises:
        ModelSelectionError: The problem type is unsupported or every candidate
            was excluded.
    """
    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ModelSelectionError(
            f"Problem type '{problem_type}' is not supported for model selection in Phase 4. "
            f"Supported: {list(SUPPORTED_PROBLEM_TYPES)}"
        )

    excluded_names = {name for name in (exclude or [])}
    for name in excluded_names:
        # Validate the name so a typo is reported instead of silently ignored.
        get_model_definition(name)

    considered = set(model_names) if model_names is not None else set(get_supported_models())
    unknown_names = sorted(considered - set(get_supported_models()))
    if unknown_names:
        raise ModelConfigurationError(
            f"Unknown model(s) {unknown_names}. Supported models: {get_supported_models()}"
        )

    selected: List[SelectedModel] = []
    # Iterate the registry, not the caller's list, so the order is canonical.
    for name in get_supported_models():
        if name not in considered or name in excluded_names:
            continue
        definition = get_model_definition(name)
        if problem_type not in definition.supported_problem_types:
            continue
        selected.append(
            SelectedModel(
                name=definition.name,
                model_type=definition.model_type,
                scaling_required=definition.scaling_required,
                selection_reason=(
                    f"Supported baseline for {problem_type}; "
                    f"{'uses' if definition.scaling_required else 'does not need'} "
                    f"feature scaling."
                ),
                default_parameters=dict(definition.default_parameters),
            )
        )

    if not selected:
        raise ModelSelectionError(
            f"No candidate model left for problem type '{problem_type}'. "
            f"Excluded models: {sorted(excluded_names)}"
        )

    logger.info(
        "Selected %d candidate model(s) for %s: %s",
        len(selected),
        problem_type,
        [item.name for item in selected],
    )
    return selected


def get_excluded_model_reasons(
    problem_type: Optional[str],
    model_names: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Explain, per model, why it was not selected."""
    considered = list(model_names) if model_names is not None else get_supported_models()
    reasons: Dict[str, str] = {}
    for name in considered:
        definition = get_model_definition(name)
        if problem_type not in SUPPORTED_PROBLEM_TYPES:
            reasons[name] = (
                f"problem type '{problem_type}' is not supported in Phase 4"
            )
        elif problem_type not in definition.supported_problem_types:
            reasons[name] = f"does not support problem type '{problem_type}'"
    return reasons