"""Configuration settings for the Autonomous Multi-Agent Machine Learning System."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Attempt to load local environment variables if python-dotenv is installed
try:
    from dotenv import load_dotenv

    # Always load from project root (the directory containing this file's parent)
    _project_root = Path(__file__).resolve().parent.parent
    _env_file = _project_root / ".env"
    load_dotenv(dotenv_path=_env_file, override=True)
except ImportError:
    pass


@dataclass(frozen=True)
class AppConfig:
    """Central application configuration."""

    # Project directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATASETS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "datasets")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    REPORTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports")
    EXPERIMENTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "experiments")

    # Reproducibility and Training defaults
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
    CV_FOLDS: int = int(os.getenv("CV_FOLDS", "5"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Scope for Classification
    SUPPORTED_PROBLEM_TYPES: List[str] = field(
        default_factory=lambda: [
            "binary_classification",
            "multiclass_classification",
        ]
    )

    INITIAL_CLASSIFICATION_MODELS: List[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "k_nearest_neighbors",
            "support_vector_machine",
        ]
    )

    # Preprocessing Defaults (Phase 3)
    DEFAULT_NUMERICAL_IMPUTATION: str = os.getenv("DEFAULT_NUMERICAL_IMPUTATION", "median")
    DEFAULT_CATEGORICAL_IMPUTATION: str = os.getenv("DEFAULT_CATEGORICAL_IMPUTATION", "most_frequent")
    DEFAULT_ENCODING: str = "one_hot"
    DEFAULT_SCALE_NUMERIC: bool = os.getenv("DEFAULT_SCALE_NUMERIC", "true").lower() == "true"
    DEFAULT_DUPLICATE_STRATEGY: str = "remove_before_training"
    MAX_CATEGORICAL_CARDINALITY: int = int(os.getenv("MAX_CATEGORICAL_CARDINALITY", "50"))
    HIGH_CARDINALITY_RATIO_THRESHOLD: float = float(os.getenv("HIGH_CARDINALITY_RATIO_THRESHOLD", "0.85"))

    # Evaluation Reporting Heuristic Thresholds (Phase 5)
    GAP_THRESHOLD_CLOSE: float = float(os.getenv("GAP_THRESHOLD_CLOSE", "0.03"))
    GAP_THRESHOLD_MODERATE: float = float(os.getenv("GAP_THRESHOLD_MODERATE", "0.10"))

    # Hyperparameter Optimization defaults (Phase 6)
    OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", "20"))
    OPTUNA_RANDOM_SEED: int = int(os.getenv("OPTUNA_RANDOM_SEED", "42"))
    OPTUNA_TOP_MODELS: int = int(os.getenv("OPTUNA_TOP_MODELS", "2"))
    OPTUNA_TIMEOUT_SECONDS: Optional[int] = (
        int(os.getenv("OPTUNA_TIMEOUT_SECONDS", "300"))
        if os.getenv("OPTUNA_TIMEOUT_SECONDS") != "none"
        else None
    )
    DEFAULT_OPTUNA_TRIALS: int = OPTUNA_N_TRIALS
    DEFAULT_OPTUNA_TIMEOUT_SECONDS: int = 300

    # LangGraph Orchestration defaults (Phase 7)
    LANGGRAPH_MAX_RETRIES: int = int(os.getenv("LANGGRAPH_MAX_RETRIES", "1"))
    LANGGRAPH_ENABLE_OPTIMIZATION: bool = (
        os.getenv("LANGGRAPH_ENABLE_OPTIMIZATION", "true").lower() == "true"
    )
    LANGGRAPH_STOP_ON_FAILURE: bool = (
        os.getenv("LANGGRAPH_STOP_ON_FAILURE", "true").lower() == "true"
    )

    # Local LLM defaults (Phase 8 - Optional via Ollama)
    OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "15.0"))
    OLLAMA_MAX_RETRIES: int = int(os.getenv("OLLAMA_MAX_RETRIES", "1"))
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

    # MLflow Experiment Tracking (Phase 10 - Optional, local-first)
    # Default: disabled.  Enable via env var: MLFLOW_ENABLED=true
    MLFLOW_ENABLED: bool = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "experiments/mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "autonomous_ml_system"
    )

    def ensure_directories(self) -> None:
        """Create necessary project directories if they do not exist."""
        for path in [
            self.DATASETS_DIR,
            self.MODELS_DIR,
            self.REPORTS_DIR,
            self.EXPERIMENTS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
Config = AppConfig()
Config.ensure_directories()
