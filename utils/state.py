"""Shared state and typed data contracts for multi-agent communication.

Every record used to move information between agents is defined here so that
`utils` stays the lowest layer of the project (agents and tools depend on
`utils`, never the other way around).

This module contains:
    * Enums describing the ML task, agent status and preprocessing choices.
    * Small, typed result records produced by the Data Analysis Agent (Phase 2),
      the Preprocessing Agent (Phase 3), the Model Selection and Training Agents
      (Phase 4), the Evaluation Agent (Phase 5) and the Hyperparameter
      Optimization Agent (Phase 6).
    * The `AgentState` object that flows through the whole workflow.

Design rules applied here:
    * Structured, typed fields instead of unstructured strings.
    * JSON-friendly primitives only (int / float / str / bool / None), so the
      state can be serialised for reports and the future dashboard.
    * Fields belonging to later phases (training, tuning) are kept, but they
      stay empty until the corresponding phase is implemented.
    * Runtime objects (scikit-learn pipelines, models) are never part of the
      serialised state. They are described by `runtime_objects` instead and can
      always be rebuilt from the stored configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ProblemType(str, Enum):
    """Machine learning problem type detected for the target column.

    Only classification is supported in the current phase. `REGRESSION` is
    reported when the target looks continuous so the reason is never hidden.
    """

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNSUPPORTED_OR_UNCERTAIN = "unsupported_or_uncertain"


class TargetDetectionMethod(str, Enum):
    """How the target column was chosen."""

    EXPLICIT = "explicit"
    AUTOMATIC = "automatic"
    NOT_DETECTED = "not_detected"


class Confidence(str, Enum):
    """Confidence level of a heuristic decision."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Severity(str, Enum):
    """Severity of a data quality issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AgentStatus(str, Enum):
    """Workflow status stored in `AgentState.status`.

    Intermediate stages use "<stage>_completed"; "completed" is reserved for
    the end of the full workflow.
    """

    INITIALIZED = "initialized"
    RUNNING = "running"
    ANALYSIS_COMPLETED = "analysis_completed"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    MODELS_SELECTED = "models_selected"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    COMPLETED = "completed"
    FAILED = "failed"


class PreprocessingStatus(str, Enum):
    """Outcome of the Preprocessing Agent."""

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExclusionReason(str, Enum):
    """Why a column was excluded from the model features."""

    EMPTY_COLUMN = "empty_column"
    CONSTANT_COLUMN = "constant_column"
    IDENTIFIER_LIKE = "identifier_like"
    HIGH_CARDINALITY = "high_cardinality"
    UNSUPPORTED_TYPE = "unsupported_type"


class DuplicateStrategy(str, Enum):
    """What to do with fully duplicated rows."""

    REMOVE_EXACT_DUPLICATES = "remove_exact_duplicates"
    KEEP_ALL = "keep_all"


class ScalingStrategy(str, Enum):
    """Scaling applied to numerical features."""

    STANDARD = "standard"
    NONE = "none"


class ModelType(str, Enum):
    """Family a candidate model belongs to."""

    LINEAR = "linear"
    TREE_BASED = "tree_based"
    INSTANCE_BASED = "instance_based"
    KERNEL_BASED = "kernel_based"
    ENSEMBLE = "ensemble"


class TrainingStatus(str, Enum):
    """Outcome of the Training Agent."""

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExperimentStatus(str, Enum):
    """Outcome of a single model experiment.

    `partial` means the model was trained but cross-validation could not run
    (for example a class with too few samples). The metrics are still valid.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class EvaluationStatus(str, Enum):
    """Outcome of the Evaluation Agent.

    `completed`  - every attempted experiment produced usable metrics;
    `partial`    - at least one experiment was evaluated, but some failed or
                   produced hold-out metrics without cross-validation;
    `skipped`    - training finished but nothing was recorded to evaluate;
    `failed`     - no experiment produced usable metrics.
    """

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class GapClassification(str, Enum):
    """How a measured hold-out vs cross-validation difference is worded.

    These are **reporting categories**, not statistical verdicts. A
    `substantial` gap never means "the model is overfitting"; it only means the
    two numbers differ enough to be worth reporting under the configured
    thresholds (see `config.GAP_CLASSIFICATION_RULE`).
    """

    CLOSE = "close"
    MODERATE = "moderate"
    SUBSTANTIAL = "substantial"
    NOT_AVAILABLE = "not_available"


class OptimizationStatus(str, Enum):
    """Outcome of the Hyperparameter Optimization Agent.

    `completed`  - every selected model was tuned and produced a usable result;
    `partial`    - at least one model was tuned, but at least one selection or
                   optimization failed (a failed trial alone does not make the
                   run partial, because that is a normal search outcome);
    `skipped`    - the agent was reached but there was nothing to optimize;
    `failed`     - no model produced a usable optimized configuration, so the
                   Phase 5 baseline result stays the only usable result.
    """

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class TrialState(str, Enum):
    """Outcome of a single Optuna trial.

    These mirror the states Optuna reports for a finished trial. `complete` means
    the objective produced a score; `fail` means the trial raised and its error
    message is stored next to it. A failed trial is a normal search outcome, not
    a reason to abandon the study.
    """

    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


class ImprovementDirection(str, Enum):
    """Whether tuning moved the cross-validation score up, down or not at all.

    `improved`, `degraded` and `unchanged` describe a **measurement**, not a
    verdict: a degraded score is a normal outcome of a small random search and is
    reported as such. `not_available` means one of the two scores is missing, so
    no difference could be computed and none was invented.
    """

    IMPROVED = "improved"
    DEGRADED = "degraded"
    UNCHANGED = "unchanged"
    NOT_AVAILABLE = "not_available"


# --------------------------------------------------------------------------- #
# Data analysis records
# --------------------------------------------------------------------------- #
@dataclass
class DatasetShape:
    """Number of rows and columns in the dataset."""

    rows: int
    columns: int

    def __str__(self) -> str:
        return f"{self.rows} rows x {self.columns} columns"


@dataclass
class MissingValueReport:
    """Missing value statistics.

    `per_column` and `per_column_percentage` only contain columns that
    actually have at least one missing value.
    """

    total: int
    total_percentage: float
    affected_columns: int
    per_column: Dict[str, int] = field(default_factory=dict)
    per_column_percentage: Dict[str, float] = field(default_factory=dict)


@dataclass
class NumericalStats:
    """Basic descriptive statistics for one numerical column."""

    column: str
    count: int
    mean: Optional[float]
    std: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]


@dataclass
class CategoricalStats:
    """Value distribution for one categorical column."""

    column: str
    unique_count: int
    top_values: Dict[str, int] = field(default_factory=dict)
    top_values_percentage: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClassDistributionEntry:
    """Count and share of a single target class."""

    count: int
    percentage: float


@dataclass
class DataQualityIssue:
    """A potential data quality problem.

    Issues are only reported, never fixed automatically. Removal or
    imputation decisions belong to the Preprocessing Agent (Phase 3).
    """

    kind: str
    severity: str
    message: str
    column: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetCandidate:
    """One scored candidate considered during automatic target detection."""

    column: str
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class TargetDetection:
    """Result of target column resolution."""

    column: Optional[str]
    method: str
    confidence: str
    reason: str
    candidates: List[TargetCandidate] = field(default_factory=list)


@dataclass
class AgentMessage:
    """A status message emitted by an agent."""

    agent: str
    message: str
    level: str = "info"
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnalysisSummary:
    """Compact, display-ready summary of the Data Analysis Agent result."""

    dataset_name: Optional[str]
    rows: int
    columns: int
    numerical_feature_count: int
    categorical_feature_count: int
    total_missing_values: int
    duplicate_count: int
    target_column: Optional[str]
    target_detection_method: str
    problem_type: Optional[str]
    problem_type_reason: str
    issue_count: int
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    summary_text: str = ""


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def to_serializable(value: Any) -> Any:
    """Convert nested dataclasses/containers into JSON-safe values.

    Dataclasses become dictionaries, enums become their value, and anything
    that is not a JSON primitive (for example a scikit-learn estimator) becomes
    a short string, so the result can always be passed to `json.dumps`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_serializable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return str(value)


def describe_runtime_object(obj: Any) -> Dict[str, Any]:
    """Describe a runtime object without serialising its internals.

    Only the class name, module and whether the object has already been fitted
    are reported. This keeps the shared state small and safe to export while
    still telling the user what kind of object the pipeline is.
    """
    description: Dict[str, Any] = {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "fitted": False,
        "serialized": False,
    }
    try:
        # Imported lazily so this module stays usable without scikit-learn.
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(obj)
        description["fitted"] = True
    except ImportError:  # pragma: no cover - scikit-learn is a project dependency
        pass
    except Exception:
        description["fitted"] = False
    return description


# --------------------------------------------------------------------------- #
# Preprocessing records (Phase 3)
# --------------------------------------------------------------------------- #
@dataclass
class ExcludedFeature:
    """A column that is not used as a model feature, and why."""

    column: str
    reason: str  # ExclusionReason value
    detail: str = ""


@dataclass
class DuplicateHandling:
    """Decision taken for fully duplicated rows.

    Duplicates are removed from the in-memory training frame by the training
    code (Phase 4) using `remove_duplicate_rows`, never from the source file.
    """

    strategy: str = DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value
    duplicate_rows_detected: int = 0
    duplicate_rows_removed: int = 0
    duplicate_rows_to_remove: int = 0
    removal_stage: str = "training_data_preparation"
    notes: str = ""


@dataclass
class PreprocessingStep:
    """Human readable description of one branch of the pipeline."""

    name: str
    applies_to: str
    transformers: List[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    """Fully serialisable description of the preprocessing decisions.

    This is the single source of truth: the scikit-learn pipeline is always
    rebuildable from this record with `build_column_transformer(config)`.
    """

    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    excluded_features: List[ExcludedFeature] = field(default_factory=list)
    target_column: Optional[str] = None

    # Strategies
    numerical_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    encoding: str = "one_hot"
    encoding_handle_unknown: str = "ignore"
    encoding_drop: Optional[str] = None
    encoding_sparse_output: bool = True
    scale_numeric: bool = True
    scaling_strategy: str = ScalingStrategy.STANDARD.value
    duplicate_handling: DuplicateHandling = field(default_factory=DuplicateHandling)

    # Pipeline behaviour
    remainder: str = "drop"
    # Hard guarantee: the pipeline is only ever fitted on training data.
    fit_scope: str = "training_data_only"
    steps: List[PreprocessingStep] = field(default_factory=list)


@dataclass
class PreprocessingSummary:
    """Display-ready summary of the preprocessing decisions."""

    status: str = PreprocessingStatus.NOT_STARTED.value
    status_reason: str = ""
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    original_feature_count: int = 0
    numerical_feature_count: int = 0
    categorical_feature_count: int = 0
    used_feature_count: int = 0
    excluded_feature_count: int = 0
    excluded_by_reason: Dict[str, int] = field(default_factory=dict)
    estimated_output_feature_count: int = 0
    duplicate_rows_detected: int = 0
    duplicate_strategy: str = DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value
    pipeline_ready: bool = False
    summary_text: str = ""


# --------------------------------------------------------------------------- #
# Model selection and training records (Phase 4)
# --------------------------------------------------------------------------- #
@dataclass
class ModelDefinition:
    """Static, JSON-safe description of one supported candidate model.

    This mirrors the entry of the model registry in `tools/model_tools.py`. The
    state keeps a serialisable copy so reports never have to touch the registry
    (which holds live factory callables).
    """

    name: str
    model_type: str  # ModelType value
    scaling_required: bool
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    supported_problem_types: List[str] = field(default_factory=list)
    description: str = ""
    requires_probability: bool = False


@dataclass
class SelectedModel:
    """A candidate model chosen for this dataset, with the reason why."""

    name: str
    model_type: str
    scaling_required: bool
    selection_reason: str = ""
    default_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSelectionSummary:
    """Display-ready summary of the model selection step."""

    status: str = "not_started"
    status_reason: str = ""
    problem_type: Optional[str] = None
    target_column: Optional[str] = None
    selected_count: int = 0
    selected_models: List[str] = field(default_factory=list)
    scaling_required_models: List[str] = field(default_factory=list)
    scaling_optional_models: List[str] = field(default_factory=list)
    excluded_models: Dict[str, str] = field(default_factory=dict)
    summary_text: str = ""


@dataclass
class DataSplitInfo:
    """How the dataset was divided for training and testing."""

    test_size: float = 0.2
    random_state: int = 42
    stratified: bool = True
    stratification_reason: str = ""
    train_samples: int = 0
    test_samples: int = 0
    total_samples: int = 0
    class_distribution_before_split: Dict[str, int] = field(default_factory=dict)
    class_distribution_train: Dict[str, int] = field(default_factory=dict)
    class_distribution_test: Dict[str, int] = field(default_factory=dict)
    duplicate_rows_detected: int = 0
    duplicate_rows_removed: int = 0
    duplicate_handling_strategy: str = DuplicateStrategy.REMOVE_EXACT_DUPLICATES.value
    input_feature_count: int = 0
    excluded_feature_count: int = 0
    duplicate_handling_note: str = ""


@dataclass
class CrossValidationConfig:
    """Baseline cross-validation settings (Phase 4)."""

    enabled: bool = True
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    scoring: Dict[str, str] = field(default_factory=dict)
    adjusted_reason: str = ""


@dataclass
class CrossValidationResult:
    """Cross-validation outcome for one experiment (JSON-safe)."""

    model_name: str = ""
    n_splits: int = 0
    status: str = "success"
    reason: str = ""
    accuracy_mean: Optional[float] = None
    accuracy_std: Optional[float] = None
    f1_mean: Optional[float] = None
    f1_std: Optional[float] = None
    fold_accuracies: List[float] = field(default_factory=list)


@dataclass
class ConfusionMatrixData:
    """Confusion matrix as plain numbers, ready for a future chart."""

    labels: List[str] = field(default_factory=list)
    matrix: List[List[int]] = field(default_factory=list)
    total: int = 0


@dataclass
class ExperimentResult:
    """Result of training and evaluating one candidate model.

    The fitted pipeline is deliberately **not** part of this record: it is kept
    separately on `AgentState.best_pipeline` and never serialised.
    """

    experiment_id: str = ""
    model_name: str = ""
    model_type: str = ""
    preprocessing_scaling: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_feature_count: int = 0
    transformed_feature_count: Optional[int] = None
    train_samples: int = 0
    test_samples: int = 0
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    cv_time: float = 0.0
    total_time: float = 0.0
    confusion_matrix: Optional[ConfusionMatrixData] = None
    status: str = ExperimentStatus.SUCCESS.value
    error_message: str = ""


@dataclass
class ModelComparisonEntry:
    """One row of the model comparison table."""

    model_name: str = ""
    model_type: str = ""
    preprocessing_scaling: bool = True
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    cv_accuracy_mean: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    total_time: float = 0.0
    rank: Optional[int] = None


@dataclass
class ModelComparison:
    """Structured comparison of every successful experiment."""

    ranking_rule: str = ""
    primary_metric: str = "f1_score"
    secondary_metric: str = "accuracy"
    entries: List[ModelComparisonEntry] = field(default_factory=list)
    successful_count: int = 0
    failed_count: int = 0
    best_model_name: Optional[str] = None
    table_text: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class BestModelInfo:
    """The current best *baseline* model (Phase 4, no tuning yet)."""

    model_name: str = ""
    experiment_id: str = ""
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    preprocessing_scaling: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""


# --------------------------------------------------------------------------- #
# Evaluation records (Phase 5)
# --------------------------------------------------------------------------- #
@dataclass
class ClassMetrics:
    """Per-class precision / recall / F1 for one label.

    Derived from the stored confusion matrix of the hold-out test split: with
    `precision = TP / (TP + FP)` and `recall = TP / (TP + FN)` the four counts
    in the matrix are sufficient, so no model has to be retrained.

    `true_negatives`, `false_positives` and `false_negatives` are counted from
    the point of view of **this** label as the positive class. The project has
    no domain metadata, so no label is assumed to be the "interesting" one.
    """

    label: str = ""
    support: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    derived: bool = False
    note: str = ""


@dataclass
class ConfusionPair:
    """One off-diagonal confusion cell: `true_label` predicted as `predicted_label`."""

    true_label: str = ""
    predicted_label: str = ""
    count: int = 0


@dataclass
class ConfusionMatrixAnalysis:
    """Interpretation of one hold-out confusion matrix.

    Purely descriptive: counts, ratios and the dominant error pattern. Because
    the project holds no domain information, the analysis never states what a
    mistake would mean in the real world.
    """

    model_name: str = ""
    available: bool = False
    labels: List[str] = field(default_factory=list)
    total_samples: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0
    accuracy_from_matrix: Optional[float] = None
    class_metrics: List[ClassMetrics] = field(default_factory=list)
    dominant_confusion_pairs: List[ConfusionPair] = field(default_factory=list)
    classes_with_most_errors: List[str] = field(default_factory=list)
    error_pattern: str = ""
    per_class_metrics_note: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class ModelEvaluation:
    """Everything the Evaluation Agent knows about one baseline model.

    The hold-out metrics, the cross-validation means and the gap between them
    are kept side by side so that an *observation* (a number) is never mixed up
    with an *interpretation* (`gap_interpretation`) or a *hypothesis*
    (`notes`).
    """

    rank: Optional[int] = None
    model_name: str = ""
    model_type: str = ""
    experiment_id: str = ""
    experiment_status: str = ""
    evaluation_status: str = EvaluationStatus.NOT_STARTED.value
    preprocessing_scaling: bool = True
    # Observed: hold-out test split
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    # Observed: cross-validation on the training split
    cv_accuracy_mean: Optional[float] = None
    cv_accuracy_std: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None
    cv_n_splits: int = 0
    # Observed: holdout - cross-validation
    f1_gap: Optional[float] = None
    accuracy_gap: Optional[float] = None
    # Interpretation: reporting category of the gaps above
    f1_gap_classification: str = GapClassification.NOT_AVAILABLE.value
    accuracy_gap_classification: str = GapClassification.NOT_AVAILABLE.value
    gap_interpretation: str = ""
    # Observed: wall-clock time of the experiment
    training_time: float = 0.0
    total_time: float = 0.0
    # Hypothesis / caveats
    notes: List[str] = field(default_factory=list)


@dataclass
class GapAnalysis:
    """Aggregate view of the hold-out vs cross-validation differences."""

    close_threshold: float = 0.0
    substantial_threshold: float = 0.0
    rule: str = ""
    disclaimer: str = ""
    models_with_gap: int = 0
    models_without_gap: List[str] = field(default_factory=list)
    close_agreement_models: List[str] = field(default_factory=list)
    moderate_difference_models: List[str] = field(default_factory=list)
    substantial_difference_models: List[str] = field(default_factory=list)
    metric: str = "f1"
    largest_absolute_gap: Optional[float] = None
    largest_absolute_gap_model: str = ""
    summary: str = ""
    interpretation: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class ExecutionTimeAnalysis:
    """Wall-clock cost of the baseline experiments."""

    total_experiment_time: float = 0.0
    average_experiment_time: float = 0.0
    total_training_time: float = 0.0
    total_prediction_time: float = 0.0
    total_cross_validation_time: float = 0.0
    fastest_model: str = ""
    fastest_time: float = 0.0
    slowest_model: str = ""
    slowest_time: float = 0.0
    timed_experiments: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class ExperimentCount:
    """How much work the baseline stage actually performed.

    `total_experiments` counts *models evaluated once each*, which is a
    different quantity from the number of hyperparameter configurations tried.
    Phase 5 performs no search, so `hyperparameter_experiments` is 0 by
    definition; it is stored explicitly because the Phase 11 comparison of the
    conventional and the autonomous workflow needs both numbers.
    """

    baseline_models_attempted: int = 0
    successful_experiments: int = 0
    partial_experiments: int = 0
    failed_experiments: int = 0
    total_experiments: int = 0
    hyperparameter_experiments: int = 0
    failed_model_names: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ModelRecommendation:
    """The recommended baseline model, with the evidence that supports it."""

    best_baseline_model: str = ""
    experiment_id: str = ""
    model_type: str = ""
    selection_reason: str = ""
    performance_summary: str = ""
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    cv_f1_mean: Optional[float] = None
    cv_accuracy_mean: Optional[float] = None
    f1_gap: Optional[float] = None
    total_time: float = 0.0
    preprocessing_scaling: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix_error_pattern: str = ""
    limitations: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Structured evaluation of the baseline models (Phase 5).

    The report is JSON-safe: it holds numbers, strings and nested dataclasses
    only, so it can be written to a file or sent to the future dashboard.
    There is no live scikit-learn object anywhere in it.
    """

    overall_status: str = EvaluationStatus.NOT_STARTED.value
    dataset_name: str = ""
    problem_type: str = ""
    target_column: str = ""
    test_samples: int = 0
    ranking_rule: str = ""
    best_model: Optional[ModelRecommendation] = None
    selection_reason: str = ""
    performance_summary: str = ""
    model_evaluations: List[ModelEvaluation] = field(default_factory=list)
    gap_analysis: Optional[GapAnalysis] = None
    confusion_matrix_analysis: List[ConfusionMatrixAnalysis] = field(default_factory=list)
    execution_time_analysis: Optional[ExecutionTimeAnalysis] = None
    experiment_count: Optional[ExperimentCount] = None
    observations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    table_text: str = ""
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Optimization records (Phase 6)
# --------------------------------------------------------------------------- #
@dataclass
class TrialRecord:
    """One Optuna trial, reduced to JSON-safe primitives.

    The full Optuna `Trial` object is deliberately **not** stored: it holds live
    references to the storage and to the estimator that was built. Only the
    information the report needs is kept, so the shared state stays serialisable
    and comparable between runs.
    """

    trial_number: int = 0
    model_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    objective_value: Optional[float] = None
    state: str = TrialState.COMPLETE.value
    duration: float = 0.0
    error_message: str = ""


@dataclass
class OptimizationResult:
    """Outcome of tuning one baseline model.

    The baseline and the optimized numbers are kept side by side and are never
    merged: `baseline_cv_f1_mean` comes from the Phase 4 cross-validation on the
    training split, `optimized_cv_f1_mean` is the value the search maximized, and
    the hold-out fields were measured once *after* the search. The improvement is
    always a cross-validation improvement, because that is the only quantity the
    search could legitimately influence.
    """

    model_name: str = ""
    model_type: str = ""
    status: str = OptimizationStatus.NOT_STARTED.value
    reason: str = ""

    # Observed: the Phase 4 baseline, kept unchanged.
    baseline_cv_f1_mean: Optional[float] = None
    baseline_cv_accuracy_mean: Optional[float] = None
    baseline_holdout_f1_score: Optional[float] = None
    baseline_holdout_accuracy: Optional[float] = None
    baseline_parameters: Dict[str, Any] = field(default_factory=dict)

    # Observed: the tuned configuration found by the search.
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_trial_number: Optional[int] = None
    optimized_cv_f1_mean: Optional[float] = None
    optimized_cv_accuracy_mean: Optional[float] = None

    # Observed: measured once on the untouched hold-out split, after the search.
    holdout_evaluated: bool = False
    holdout_f1_score: Optional[float] = None
    holdout_accuracy: Optional[float] = None
    holdout_f1_delta: Optional[float] = None

    # Derived: optimized - baseline, on cross-validated F1.
    improvement: Optional[float] = None
    improvement_percentage: Optional[float] = None
    improvement_direction: str = ImprovementDirection.NOT_AVAILABLE.value
    improvement_note: str = ""

    # Observed: what the search did.
    cv_n_splits: int = 0
    cv_adjustment_reason: str = ""
    total_trials: int = 0
    completed_trials: int = 0
    failed_trials: int = 0
    pruned_trials: int = 0
    search_space_description: List[str] = field(default_factory=list)
    objective_metric: str = ""
    optimization_time: float = 0.0
    error_message: str = ""
    trials: List[TrialRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class OptimizationCounts:
    """How much work the baseline stage and the optimization stage each did.

    A *model experiment* is one model trained once with default parameters; an
    Optuna *trial* is one hyperparameter configuration evaluated by
    cross-validation. Phase 6 established that these are different quantities, so
    they are counted separately here and never summed into one number.

    `total_trials` is `completed_trials + failed_trials + pruned_trials`, which is
    what `n_trials` asked for unless the search stopped early.
    """

    baseline_models_attempted: int = 0
    baseline_successful_experiments: int = 0
    baseline_failed_experiments: int = 0
    optimized_models_attempted: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    completed_trials: int = 0
    failed_trials: int = 0
    pruned_trials: int = 0
    total_trials: int = 0
    failed_trial_models: List[str] = field(default_factory=list)
    failed_model_names: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class OptimizedModelRecommendation:
    """The best tuned model, with the baseline evidence it is compared against.

    The Phase 5 baseline recommendation is **not** replaced: both are stored, so
    a reader can always see what the untuned default achieved and what the
    search found on top of it.
    """

    best_optimized_model: str = ""
    model_type: str = ""
    baseline_model_name: str = ""
    selection_reason: str = ""
    performance_summary: str = ""
    baseline_cv_f1_mean: Optional[float] = None
    optimized_cv_f1_mean: Optional[float] = None
    improvement: Optional[float] = None
    improvement_percentage: Optional[float] = None
    improvement_direction: str = ImprovementDirection.NOT_AVAILABLE.value
    baseline_holdout_f1_score: Optional[float] = None
    holdout_f1_score: Optional[float] = None
    holdout_f1_delta: Optional[float] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_trial_number: Optional[int] = None
    total_trials: int = 0
    cv_n_splits: int = 0
    optimization_time: float = 0.0
    selection_note: str = ""
    limitations: List[str] = field(default_factory=list)


@dataclass
class OptimizationReport:
    """Structured report of the hyperparameter optimization phase (Phase 6).

    JSON-safe by construction: numbers, strings, lists and nested dataclasses
    only. No Optuna `Study`, no `Trial` and no scikit-learn estimator is ever
    stored here - the live objects stay in the agent's local scope and the state
    only describes them.
    """

    overall_status: str = OptimizationStatus.NOT_STARTED.value
    dataset_name: str = ""
    problem_type: str = ""
    target_column: str = ""
    n_trials_per_model: int = 0
    top_models: int = 0
    sampler: str = ""
    sampler_seed: int = 0
    objective_metric: str = ""
    objective_rule: str = ""
    selection_rule: str = ""
    candidate_rule: str = ""
    ranking_rule: str = ""
    cv_n_splits: int = 0
    cv_adjustment_reason: str = ""
    holdout_note: str = ""
    selection_note: str = ""
    models_selected_for_optimization: List[str] = field(default_factory=list)
    models_not_optimized: List[str] = field(default_factory=list)
    optimization_results: List[OptimizationResult] = field(default_factory=list)
    best_optimized_model: Optional[OptimizedModelRecommendation] = None
    best_baseline_model: Optional[ModelRecommendation] = None
    baseline_saved_to: str = ""
    counts: Optional[OptimizationCounts] = None
    observations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    table_text: str = ""
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Model persistence records (Phase 9)
# --------------------------------------------------------------------------- #


class FinalModelSource(str, Enum):
    """Which earlier phase produced the model that was persisted.

    This is a label, not a judgement. ``baseline`` means the Phase 4/5 winner was
    persisted, which is what happens when optimization was disabled, skipped or
    produced no usable winner. ``optimized`` means the Phase 6 winner scored at
    least as high as that baseline under the project's existing ranking rule.
    """

    OPTIMIZED = "optimized"
    BASELINE = "baseline"


class PersistenceStatus(str, Enum):
    """Outcome of the Model Persistence Agent.

    ``skipped`` and ``failed`` are separate on purpose: "there was nothing worth
    saving" and "saving failed" call for different responses from the caller.
    """

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class FinalModelSelection:
    """The deterministic choice of the model that gets persisted.

    Both candidates and the scores they were *already* given are kept, so a
    reader can check the decision instead of taking it on trust. Phase 9
    computed nothing here: ``baseline_*`` comes from the Phase 4/5 experiment
    and ``optimized_*`` from the Phase 6 result.
    """

    model_name: str = ""
    model_type: str = ""
    selection_source: str = FinalModelSource.BASELINE.value
    selection_reason: str = ""
    selection_rule: str = ""
    baseline_model_name: str = ""
    baseline_f1_score: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    baseline_cv_f1_mean: Optional[float] = None
    optimized_model_name: str = ""
    optimized_f1_score: Optional[float] = None
    optimized_accuracy: Optional[float] = None
    optimized_cv_f1_mean: Optional[float] = None
    optimization_used: bool = False
    optimization_status: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    improvement: Optional[float] = None
    improvement_direction: str = ImprovementDirection.NOT_AVAILABLE.value


@dataclass
class PersistenceReport:
    """What the Model Persistence Agent wrote, and where.

    The three artifact names are stored **relative to the output directory**, so
    the report never carries an absolute path from one machine into another. The
    live pipeline is deliberately absent: it was written to disk and is not part
    of the state.
    """

    overall_status: str = PersistenceStatus.NOT_STARTED.value
    dataset_name: str = ""
    target_column: str = ""
    task_type: str = ""
    selection: Optional[FinalModelSelection] = None
    model_artifact: str = ""
    metadata_artifact: str = ""
    results_artifact: str = ""
    final_fit_strategy: str = ""
    final_fit_note: str = ""
    #: The hold-out scores of the model that was actually written. They are copied
    #: from the winning record rather than re-read, so a reader never has to work
    #: out which of the two candidate scores belongs to the saved model.
    selected_f1_score: Optional[float] = None
    selected_accuracy: Optional[float] = None
    feature_count: int = 0
    class_count: int = 0
    schema_version: int = 1
    artifact_size_bytes: int = 0
    observations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #
@dataclass
class AgentState:
    """Shared state object passed between agents.

    Agents read the fields they need and update the fields they produce. Every
    stage writes a message to `agent_messages` so the workflow stays traceable.
    """

    # ---- Dataset identification -------------------------------------------- #
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_shape: Optional[DatasetShape] = None
    column_names: List[str] = field(default_factory=list)
    column_dtypes: Dict[str, str] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    unsupported_columns: List[str] = field(default_factory=list)

    # ---- Target and problem type ------------------------------------------ #
    # Allowed values: ProblemType values, e.g. "binary_classification".
    target_column: Optional[str] = None
    # Allowed values: TargetDetectionMethod values, e.g. "explicit".
    target_detection_method: Optional[str] = None
    target_detection_confidence: Optional[str] = None
    target_detection_reason: str = ""
    target_candidates: List[TargetCandidate] = field(default_factory=list)
    problem_type: Optional[str] = None
    problem_type_reason: str = ""

    # ---- Data analysis results -------------------------------------------- #
    total_missing_values: int = 0
    missing_values: Optional[MissingValueReport] = None
    duplicate_count: int = 0
    numerical_statistics: Dict[str, NumericalStats] = field(default_factory=dict)
    categorical_statistics: Dict[str, CategoricalStats] = field(default_factory=dict)
    class_distribution: Dict[str, ClassDistributionEntry] = field(default_factory=dict)
    data_quality_issues: List[DataQualityIssue] = field(default_factory=list)
    analysis_summary: Optional[AnalysisSummary] = None

    # ---- Preprocessing decisions (Phase 3) -------------------------------- #
    # `preprocessing_config` is the single source of truth; the convenience
    # properties below are read-only views on it.
    preprocessing_status: str = PreprocessingStatus.NOT_STARTED.value
    preprocessing_config: Optional[PreprocessingConfig] = None
    preprocessing_summary: Optional[PreprocessingSummary] = None
    preprocessing_steps: List[PreprocessingStep] = field(default_factory=list)
    # Runtime object: an *unfitted* sklearn ColumnTransformer. It is never
    # serialised into the state (see `to_dict`) and is always rebuildable from
    # `preprocessing_config`.
    preprocessing_pipeline: Optional[Any] = None

    # ---- Model selection (Phase 4) ---------------------------------------- #
    # The live estimator classes live in the registry (tools/model_tools.py);
    # only serialisable copies of the definitions are kept in the state.
    model_selection_status: str = TrainingStatus.NOT_STARTED.value
    selected_models: List[SelectedModel] = field(default_factory=list)
    model_selection_summary: Optional[ModelSelectionSummary] = None

    # ---- Training (Phase 4) ----------------------------------------------- #
    training_status: str = TrainingStatus.NOT_STARTED.value
    data_split: Optional[DataSplitInfo] = None
    cross_validation_config: Optional[CrossValidationConfig] = None
    cross_validation_results: List[CrossValidationResult] = field(default_factory=list)
    experiment_results: List[ExperimentResult] = field(default_factory=list)
    model_comparison: Optional[ModelComparison] = None
    best_baseline_model: Optional[BestModelInfo] = None
    # Runtime object: the fitted Pipeline of the best baseline model. It is
    # never serialised (see `to_dict`) and Phase 9 saves it to disk. Each
    # experiment keeps its own pipeline instance, because scikit-learn fits the
    # steps in place and sharing one transformer between models would leak
    # fitted state from one experiment into the next.
    best_pipeline: Optional[Any] = None

    # ---- Evaluation (Phase 5) ---------------------------------------------- #
    # The Evaluation Agent only interprets the records above: it never trains a
    # model and never adds a live scikit-learn object to the state.
    evaluation_status: str = EvaluationStatus.NOT_STARTED.value
    evaluation_report: Optional[EvaluationReport] = None

    # ---- Hyperparameter optimization (Phase 6) ----------------------------- #
    # The Optimization Agent tunes the most promising baseline models. The live
    # Optuna Study and the tuned pipelines exist only inside the agent while it
    # runs: only the JSON-safe `TrialRecord`s and the summary survive here.
    optimization_status: str = OptimizationStatus.NOT_STARTED.value
    optimization_report: Optional[OptimizationReport] = None

    # ---- Model persistence (Phase 9) --------------------------------------- #
    # Phase 9 writes what Phases 4-6 already produced and reads it back for the
    # dashboard. It adds no machine learning behaviour: no new metric, no new
    # ranking rule and no live scikit-learn object beyond the fitted pipeline
    # that is written to disk here and never held on the state.
    persistence_status: str = PersistenceStatus.NOT_STARTED.value
    persistence_report: Optional[PersistenceReport] = None

    # ---- Workflow tracking ------------------------------------------------- #
    execution_time: float = 0.0
    experiment_count: int = 0
    agent_messages: List[AgentMessage] = field(default_factory=list)
    status: str = AgentStatus.INITIALIZED.value
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    # Runtime objects are described, never serialised
    # ------------------------------------------------------------------ #
    RUNTIME_ONLY_FIELDS: tuple = (
        "preprocessing_pipeline",
        "best_pipeline",
    )

    # ------------------------------------------------------------------ #
    # Fields exposed as a read-only property on top of another field.
    # They are skipped by `to_dict` so the same records are not serialised
    # twice; the owning record stays the single source of truth.
    # ------------------------------------------------------------------ #
    DERIVED_ONLY_FIELDS: tuple = (
        "optimization_results",
    )

    # ------------------------------------------------------------------ #
    # Read-only views on the preprocessing configuration
    # ------------------------------------------------------------------ #
    @property
    def preprocessing_numerical_features(self) -> List[str]:
        """Numerical features selected for the model."""
        if not self.preprocessing_config:
            return []
        return list(self.preprocessing_config.numerical_features)

    @property
    def preprocessing_categorical_features(self) -> List[str]:
        """Categorical features selected for the model."""
        if not self.preprocessing_config:
            return []
        return list(self.preprocessing_config.categorical_features)

    @property
    def excluded_features(self) -> List[ExcludedFeature]:
        """Columns excluded from the model features, with their reasons."""
        if not self.preprocessing_config:
            return []
        return list(self.preprocessing_config.excluded_features)

    @property
    def excluded_feature_names(self) -> List[str]:
        """Names of the excluded columns."""
        return [item.column for item in self.excluded_features]

    @property
    def excluded_features_by_reason(self) -> Dict[str, List[str]]:
        """Group the excluded columns by their exclusion reason."""
        grouped: Dict[str, List[str]] = {}
        for item in self.excluded_features:
            grouped.setdefault(item.reason, []).append(item.column)
        return grouped

    # ------------------------------------------------------------------ #
    # Read-only views on the Phase 4 model selection and training results
    # ------------------------------------------------------------------ #
    @property
    def selected_model_names(self) -> List[str]:
        """Names of the candidate models chosen for this dataset."""
        return [model.name for model in self.selected_models]

    @property
    def scaling_required_model_names(self) -> List[str]:
        """Selected models that should run with scaling enabled."""
        return [model.name for model in self.selected_models if model.scaling_required]

    @property
    def scaling_optional_model_names(self) -> List[str]:
        """Selected models that can run without scaling (tree based)."""
        return [model.name for model in self.selected_models if not model.scaling_required]

    @property
    def successful_experiments(self) -> List[ExperimentResult]:
        """Experiments that produced usable metrics (success or partial)."""
        usable = {ExperimentStatus.SUCCESS.value, ExperimentStatus.PARTIAL.value}
        return [item for item in self.experiment_results if item.status in usable]

    @property
    def failed_experiments(self) -> List[ExperimentResult]:
        """Experiments that could not be completed."""
        return [
            item
            for item in self.experiment_results
            if item.status == ExperimentStatus.FAILED.value
        ]

    @property
    def has_best_baseline_model(self) -> bool:
        """True when a best baseline model has been identified."""
        return self.best_baseline_model is not None

    @property
    def has_evaluation_report(self) -> bool:
        """True when the Evaluation Agent produced a report."""
        return self.evaluation_report is not None

    @property
    def evaluation_limitations(self) -> List[str]:
        """Limitations recorded by the Evaluation Agent."""
        if self.evaluation_report is None:
            return []
        return list(self.evaluation_report.limitations)

    # ------------------------------------------------------------------ #
    # Read-only views on the Phase 6 optimization results
    # ------------------------------------------------------------------ #
    @property
    def optimization_results(self) -> List[OptimizationResult]:
        """Per-model optimization results, in the order the agent produced them.

        `optimization_report` is the single source of truth, so this is a view on
        it rather than a second copy that could drift out of sync.
        """
        if self.optimization_report is None:
            return []
        return list(self.optimization_report.optimization_results)

    @property
    def has_optimization_report(self) -> bool:
        """True when the Optimization Agent produced a report."""
        return self.optimization_report is not None

    @property
    def optimization_limitations(self) -> List[str]:
        """Limitations recorded by the Optimization Agent."""
        if self.optimization_report is None:
            return []
        return list(self.optimization_report.limitations)

    @property
    def successful_optimizations(self) -> List[OptimizationResult]:
        """Optimized models that produced a usable tuned configuration."""
        return [
            item
            for item in self.optimization_results
            if item.status == OptimizationStatus.COMPLETED.value
        ]

    # ------------------------------------------------------------------ #
    # Read-only views on the Phase 9 persistence results
    # ------------------------------------------------------------------ #
    @property
    def has_persistence_report(self) -> bool:
        """True when the Model Persistence Agent produced a report."""
        return self.persistence_report is not None

    @property
    def final_model_selection(self) -> Optional[FinalModelSelection]:
        """The recorded final model choice, or None when nothing was persisted."""
        if self.persistence_report is None:
            return None
        return self.persistence_report.selection

    @property
    def model_artifact_name(self) -> Optional[str]:
        """Name of the persisted pipeline file, or None when none was written."""
        if self.persistence_report is None:
            return None
        return self.persistence_report.model_artifact or None

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a JSON-safe dictionary.

        Runtime objects (scikit-learn pipelines, models) are never embedded.
        They are described under `runtime_objects` so the state can safely be
        written to a report or a JSON file.
        """
        data: Dict[str, Any] = {}
        for item in fields(self):
            if item.name in self.RUNTIME_ONLY_FIELDS:
                continue
            if item.name in self.DERIVED_ONLY_FIELDS:
                continue
            data[item.name] = to_serializable(getattr(self, item.name))

        runtime_objects = {
            name: describe_runtime_object(getattr(self, name))
            for name in self.RUNTIME_ONLY_FIELDS
            if getattr(self, name) is not None
        }
        if runtime_objects:
            data["runtime_objects"] = runtime_objects
        return data

    def add_message(self, agent: str, message: str, level: str = "info") -> None:
        """Record a status message emitted by an agent."""
        self.agent_messages.append(
            AgentMessage(agent=agent, message=message, level=level, timestamp=time.time())
        )

    def add_error(self, error: str) -> None:
        """Record an error and move the workflow to the failed state."""
        self.errors.append(error)
        self.status = AgentStatus.FAILED.value

    def issues_by_severity(self) -> Dict[str, int]:
        """Count data quality issues per severity level."""
        counts = {severity.value: 0 for severity in Severity}
        for issue in self.data_quality_issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts

    def get_execution_time(self) -> float:
        """Return the elapsed workflow time in seconds."""
        if self.execution_time == 0.0 and self.status in (
            AgentStatus.COMPLETED.value,
            AgentStatus.FAILED.value,
        ):
            self.execution_time = time.time() - self.start_time
        return self.execution_time