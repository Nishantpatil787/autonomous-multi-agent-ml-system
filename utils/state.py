"""Shared state definition for the Autonomous Multi-Agent Machine Learning System.

This module defines the structured communication contract passed between
the orchestrator and specialized agents (Data Analysis, Preprocessing,
Model Selection, Training, Optimization, Evaluation).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class WorkflowStatus(str, Enum):
    """Execution status of the multi-agent workflow."""

    INITIALIZED = "initialized"
    ANALYZING = "analyzing"
    ANALYSIS_COMPLETED = "analysis_completed"
    PREPROCESSING = "preprocessing"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    SELECTING_MODELS = "selecting_models"
    MODELS_SELECTED = "models_selected"
    TRAINING = "training"
    TRAINING_COMPLETED = "training_completed"
    OPTIMIZING = "optimizing"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    EVALUATING = "evaluating"
    EVALUATION_COMPLETED = "evaluation_completed"
    COMPLETED = "completed"
    FAILED = "failed"


class ProblemType(str, Enum):
    """Detected machine learning task type."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNSUPPORTED_OR_UNCERTAIN = "unsupported_or_uncertain"
    UNKNOWN = "unknown"


class GapClassification(str, Enum):
    """Reporting heuristic classification for hold-out vs cross-validation metric gaps."""

    CLOSE = "close agreement"
    MODERATE = "moderate difference"
    SUBSTANTIAL = "substantial difference"
    NOT_APPLICABLE = "not applicable"


@dataclass
class AgentLogEntry:
    """Record of an individual action or message from an agent."""

    agent_name: str
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    status: str = "info"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingDecisions:
    """Structured record of decisions determined by the Preprocessing Agent."""

    original_feature_count: int = 0
    selected_numerical_features: List[str] = field(default_factory=list)
    selected_categorical_features: List[str] = field(default_factory=list)
    excluded_features: List[str] = field(default_factory=list)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    numerical_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"
    encoding_strategy: str = "one_hot"
    scaling_strategy: str = "standard_scaler"
    scale_numeric: bool = True
    duplicate_handling_strategy: str = "remove_before_training"
    duplicate_rows_detected: int = 0
    duplicate_rows_removed: int = 0
    constant_columns_excluded: List[str] = field(default_factory=list)
    high_cardinality_columns_excluded: List[str] = field(default_factory=list)
    datetime_columns_excluded: List[str] = field(default_factory=list)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Typed record of an individual model training experiment."""

    experiment_id: str
    model_name: str
    preprocessing_scaling: bool
    parameters: Dict[str, Any] = field(default_factory=dict)
    train_samples: int = 0
    test_samples: int = 0
    feature_count_raw: int = 0
    feature_count_processed: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    cv_folds: int = 0
    training_time: float = 0.0
    prediction_time: float = 0.0
    total_time: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    confusion_matrix_labels: List[str] = field(default_factory=list)
    status: str = "success"
    error_message: Optional[str] = None


@dataclass
class GapAnalysis:
    """Hold-out versus cross-validation gap analysis record."""

    model_name: str
    holdout_f1: float
    cv_f1: Optional[float]
    f1_gap: Optional[float]
    f1_gap_abs: Optional[float]
    f1_classification: str
    holdout_accuracy: float
    cv_accuracy: Optional[float]
    accuracy_gap: Optional[float]
    accuracy_gap_abs: Optional[float]
    accuracy_classification: str
    interpretation: str


@dataclass
class ConfusionMatrixAnalysis:
    """Diagnostic analysis of prediction confusion matrix."""

    model_name: str
    total_samples: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    is_binary: bool
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_negatives: Optional[int] = None
    dominant_error_pattern: str = ""
    dominant_confusion_pairs: List[Dict[str, Any]] = field(default_factory=list)
    class_error_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ExecutionTimeAnalysis:
    """Aggregated execution telemetry across all experiments."""

    total_experiment_time: float = 0.0
    average_experiment_time: float = 0.0
    fastest_model: Optional[str] = None
    fastest_time: float = 0.0
    slowest_model: Optional[str] = None
    slowest_time: float = 0.0
    baseline_models_attempted: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    total_experiments: int = 0


@dataclass
class ModelEvaluation:
    """Structured evaluation record for an individual candidate model."""

    rank: int
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_f1_mean: Optional[float] = None
    cv_accuracy_mean: Optional[float] = None
    f1_gap: Optional[float] = None
    accuracy_gap: Optional[float] = None
    training_time: float = 0.0
    total_time: float = 0.0
    scaling_used: bool = False
    status: str = "success"


@dataclass
class EvaluationReport:
    """Complete structured synthesis produced by the Evaluation Agent."""

    overall_status: str
    best_model: str
    selection_reason: str
    performance_summary: Dict[str, Any]
    model_evaluations: List[Dict[str, Any]]
    gap_analysis: List[Dict[str, Any]]
    confusion_matrix_analyses: Dict[str, Dict[str, Any]]
    execution_time_analysis: Dict[str, Any]
    experiment_count: Dict[str, int]
    observations: List[str]
    limitations: List[str]
    recommendations: List[str]


@dataclass
class AgentState:
    """Central shared state passed between all agents in the workflow."""

    # Dataset metadata
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_shape: Optional[Tuple[int, int]] = None
    column_names: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    # Target and task metadata
    target_column: Optional[str] = None
    target_detection_method: Optional[str] = None  # "explicit" or "heuristic"
    problem_type: Optional[str] = None

    # Analysis Agent outputs (Phase 2)
    missing_values: Dict[str, Any] = field(default_factory=dict)
    missing_value_summary: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    duplicate_rows_count: int = 0
    class_distribution: Dict[str, Any] = field(default_factory=dict)
    numerical_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categorical_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    analysis_report: Dict[str, Any] = field(default_factory=dict)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)

    # Preprocessing Agent outputs (Phase 3)
    original_feature_count: int = 0
    selected_numerical_features: List[str] = field(default_factory=list)
    selected_categorical_features: List[str] = field(default_factory=list)
    excluded_features: List[str] = field(default_factory=list)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    numerical_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"
    encoding_strategy: str = "one_hot"
    scaling_strategy: str = "standard_scaler"
    scale_numeric: bool = True
    duplicate_handling_strategy: str = "remove_before_training"
    duplicate_rows_detected: int = 0
    duplicate_rows_removed: int = 0
    constant_columns_excluded: List[str] = field(default_factory=list)
    high_cardinality_columns_excluded: List[str] = field(default_factory=list)
    datetime_columns_excluded: List[str] = field(default_factory=list)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_decisions: Dict[str, Any] = field(default_factory=dict)
    preprocessing_summary: Dict[str, Any] = field(default_factory=dict)
    pipeline_path: Optional[str] = None
    processed_feature_names: List[str] = field(default_factory=list)

    # Model Selection Agent outputs (Phase 4)
    selected_models: List[str] = field(default_factory=list)
    model_selection_reasoning: Dict[str, str] = field(default_factory=dict)
    model_selection_summary: Dict[str, Any] = field(default_factory=dict)
    selected_model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Training Agent outputs (Phase 4)
    experiment_results: List[Dict[str, Any]] = field(default_factory=list)
    model_comparison: List[Dict[str, Any]] = field(default_factory=list)
    best_baseline_model: Optional[str] = None
    best_model_name: Optional[str] = None
    best_baseline_metrics: Dict[str, float] = field(default_factory=dict)
    training_results: Dict[str, Any] = field(default_factory=dict)
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    trained_model_paths: Dict[str, str] = field(default_factory=dict)
    split_info: Dict[str, Any] = field(default_factory=dict)
    cv_config: Dict[str, Any] = field(default_factory=dict)
    experiment_count: int = 0

    # Evaluation Agent outputs (Phase 5)
    evaluation_report: Dict[str, Any] = field(default_factory=dict)
    evaluation_status: str = "pending"
    model_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    gap_analyses: List[Dict[str, Any]] = field(default_factory=list)
    confusion_matrix_analyses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time_analysis: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrices: Dict[str, Any] = field(default_factory=dict)
    final_summary_report: Dict[str, Any] = field(default_factory=dict)

    # Hyperparameter Optimization Agent outputs (Phase 6)
    optimization_results: Dict[str, Any] = field(default_factory=dict)
    best_hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_status: str = "pending"
    optimization_comparison: List[Dict[str, Any]] = field(default_factory=list)
    best_optimized_model: Optional[str] = None
    best_optimized_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_summary: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_experiments: int = 0
    successful_trials: int = 0
    failed_trials: int = 0

    # Local LLM Explanations & Suggestions (Phase 8 - Optional)
    llm_enabled: bool = False
    llm_status: str = "disabled"
    data_analysis_explanation: Dict[str, Any] = field(default_factory=dict)
    evaluation_explanation: Dict[str, Any] = field(default_factory=dict)
    optimization_plan_suggestion: Dict[str, Any] = field(default_factory=dict)
    optimization_explanation: Dict[str, Any] = field(default_factory=dict)
    llm_logs: List[Dict[str, Any]] = field(default_factory=list)

    # Execution telemetry & audit trail
    workflow_status: WorkflowStatus = WorkflowStatus.INITIALIZED
    execution_time_seconds: float = 0.0
    agent_logs: List[AgentLogEntry] = field(default_factory=list)
    error_message: Optional[str] = None

    # Model Persistence outputs (Phase 9)
    persisted_model_path: Optional[str] = None
    persisted_metadata_path: Optional[str] = None
    persisted_model_metadata: Dict[str, Any] = field(default_factory=dict)
    persistence_status: str = "pending"

    def add_log(
        self,
        agent_name: str,
        message: str,
        status: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an audit log entry from an agent."""
        entry = AgentLogEntry(
            agent_name=agent_name,
            message=message,
            status=status,
            details=details or {},
        )
        self.agent_logs.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a plain dictionary for serialization or LangGraph integration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Reconstruct state from a dictionary."""
        data_copy = data.copy()
        raw_logs = data_copy.pop("agent_logs", [])
        logs = [
            entry if isinstance(entry, AgentLogEntry) else AgentLogEntry(**entry)
            for entry in raw_logs
        ]
        status = data_copy.pop("workflow_status", WorkflowStatus.INITIALIZED)
        if isinstance(status, str):
            status = WorkflowStatus(status)
        return cls(workflow_status=status, agent_logs=logs, **data_copy)
