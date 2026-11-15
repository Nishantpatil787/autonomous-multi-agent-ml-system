"""Controlled model registry and estimator factory for classification models.

Defines reproducible defaults, scaling requirements, and model factories
without dynamic or unconstrained code execution.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.config import Config
from utils.logger import get_logger

logger = get_logger("ModelTools")


def _create_logistic_regression(**params: Any) -> LogisticRegression:
    default = {"max_iter": 1000, "random_state": Config.RANDOM_STATE}
    default.update(params)
    return LogisticRegression(**default)


def _create_decision_tree(**params: Any) -> DecisionTreeClassifier:
    default = {"random_state": Config.RANDOM_STATE}
    default.update(params)
    return DecisionTreeClassifier(**default)


def _create_random_forest(**params: Any) -> RandomForestClassifier:
    default = {"n_estimators": 100, "random_state": Config.RANDOM_STATE}
    default.update(params)
    return RandomForestClassifier(**default)


def _create_knn(**params: Any) -> KNeighborsClassifier:
    default = {"n_neighbors": 5}
    default.update(params)
    return KNeighborsClassifier(**default)


def _create_svm(**params: Any) -> SVC:
    default = {"random_state": Config.RANDOM_STATE, "probability": False}
    default.update(params)
    return SVC(**default)


# Controlled model registry mapping model identifiers to metadata and factory functions
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "scaling_required": True,
        "default_params": {"max_iter": 1000, "random_state": Config.RANDOM_STATE},
        "supported_problem_types": [
            "binary_classification",
            "multiclass_classification",
        ],
        "factory": _create_logistic_regression,
        "description": "Linear classification model using logistic sigmoid loss.",
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "scaling_required": False,
        "default_params": {"random_state": Config.RANDOM_STATE},
        "supported_problem_types": [
            "binary_classification",
            "multiclass_classification",
        ],
        "factory": _create_decision_tree,
        "description": "Non-parametric tree-based classifier splitting on feature thresholds.",
    },
    "random_forest": {
        "display_name": "Random Forest",
        "scaling_required": False,
        "default_params": {"n_estimators": 100, "random_state": Config.RANDOM_STATE},
        "supported_problem_types": [
            "binary_classification",
            "multiclass_classification",
        ],
        "factory": _create_random_forest,
        "description": "Ensemble of randomized decision trees using bagging and feature subsampling.",
    },
    "k_nearest_neighbors": {
        "display_name": "K-Nearest Neighbors",
        "scaling_required": True,
        "default_params": {"n_neighbors": 5},
        "supported_problem_types": [
            "binary_classification",
            "multiclass_classification",
        ],
        "factory": _create_knn,
        "description": "Instance-based classifier determining classes by Euclidean proximity.",
    },
    "support_vector_machine": {
        "display_name": "Support Vector Machine",
        "scaling_required": True,
        "default_params": {"random_state": Config.RANDOM_STATE, "probability": False},
        "supported_problem_types": [
            "binary_classification",
            "multiclass_classification",
        ],
        "factory": _create_svm,
        "description": "Maximum-margin hyperplane classifier using kernel transformations.",
    },
}


def build_model_registry() -> Dict[str, Dict[str, Any]]:
    """Return serializable metadata describing all supported models in the registry."""
    return {
        key: {
            "display_name": spec["display_name"],
            "scaling_required": spec["scaling_required"],
            "default_params": spec["default_params"],
            "supported_problem_types": spec["supported_problem_types"],
            "description": spec["description"],
        }
        for key, spec in MODEL_REGISTRY.items()
    }


def get_supported_models(problem_type: Optional[str] = None) -> List[str]:
    """Retrieve model keys matching a specific problem type (or all if None)."""
    if not problem_type:
        return list(MODEL_REGISTRY.keys())

    return [
        key
        for key, spec in MODEL_REGISTRY.items()
        if problem_type in spec["supported_problem_types"]
    ]


def get_model_definition(model_key: str) -> Dict[str, Any]:
    """Retrieve specification for a registered model."""
    normalized_key = model_key.lower().strip()
    if normalized_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unsupported model '{model_key}'. Supported models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[normalized_key]


def build_estimator(
    model_key: str, param_overrides: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """Instantiate a scikit-learn estimator using the controlled factory.

    Args:
        model_key: Identifier of the model in the registry.
        param_overrides: Optional keyword argument overrides.

    Returns:
        Configured un-fitted BaseEstimator instance.
    """
    model_def = get_model_definition(model_key)
    factory: Callable[..., BaseEstimator] = model_def["factory"]
    params = param_overrides or {}
    return factory(**params)


def validate_model_configuration(model_key: str, problem_type: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that a model key is registered and compatible with the problem type."""
    normalized = model_key.lower().strip()
    if normalized not in MODEL_REGISTRY:
        return False, f"Model '{model_key}' is not in the controlled registry."

    if problem_type:
        supported = MODEL_REGISTRY[normalized]["supported_problem_types"]
        if problem_type not in supported:
            return (
                False,
                f"Model '{model_key}' does not support problem type '{problem_type}'. Supported types: {supported}",
            )

    return True, "Model configuration is valid."
