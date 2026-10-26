"""Centralised configuration for the project.

All tunable thresholds live here so that the agents and tools stay free of
magic numbers and dataset-specific assumptions. Values can be overridden with
environment variables where it makes sense (see LOG_LEVEL).

Phase 1: base project settings.
Phase 2: data analysis thresholds (added for the Data Analysis Agent).
Phase 3: preprocessing settings (added for the Preprocessing Agent).
Phase 4: model and training settings (added for the Model Selection and
    Training Agents).
Phase 5: evaluation thresholds (added for the Evaluation Agent).
Phase 6: hyperparameter optimization settings and search-space bounds (added
    for the Hyperparameter Optimization Agent).
Phase 7: LangGraph orchestration settings (added for the workflow graph). The
    graph only coordinates the existing agents; it adds no machine-learning
    behaviour of its own.
Phase 8: optional local LLM settings (Ollama). Every value here is optional: the
    layer is disabled by default, it never becomes the authority for a
    measurable decision, and every one of its failures degrades to a recorded
    status instead of stopping the machine learning pipeline.
Phase 9: model persistence and dashboard settings. Phase 9 does not add any
    machine learning behaviour: it writes the pipeline that Phases 4-6 already
    produced, and reads it back for a dashboard. It introduces no new metric, no
    new ranking rule and no second evaluation path.
Phase 10: local MLflow experiment tracking. MLflow is an observability layer
    only. It records decisions the workflow has already made and is never asked
    for one, it needs no server, and every failure it can produce degrades to a
    recorded status instead of affecting the machine learning result.
"""

import os
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in (DATASETS_DIR, MODELS_DIR, REPORTS_DIR, EXPERIMENTS_DIR, LOGS_DIR):
    _dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env.example"

# --------------------------------------------------------------------------- #
# Data processing defaults
# --------------------------------------------------------------------------- #
# Shared, reproducible defaults. RANDOM_STATE / TEST_SIZE / CV_FOLDS are reused
# by the Phase 4 training agent.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# File formats the system can read at the moment.
SUPPORTED_DATA_FORMATS: List[str] = [".csv"]

# --------------------------------------------------------------------------- #
# Data analysis thresholds (Phase 2)
# --------------------------------------------------------------------------- #

# A target column with more distinct values than this is not treated as a
# classification label.
MAX_CLASSES_FOR_CLASSIFICATION: int = 20

# Ratio of distinct values to rows above which a column is considered
# high-cardinality (identifier-like) instead of a class label.
CLASSIFICATION_MAX_UNIQUE_RATIO: float = 0.10

# Very small datasets are exempt from the ratio test, otherwise toy datasets
# with few rows would always look "high cardinality".
MIN_ROWS_FOR_UNIQUE_RATIO_CHECK: int = 20

# A class with fewer samples than this is reported as a rare class.
RARE_CLASS_MIN_SAMPLES: int = 2

# High-cardinality column detection.
HIGH_CARDINALITY_RATIO: float = 0.50
HIGH_CARDINALITY_MIN_UNIQUE: int = 20
# A column whose values are (almost) unique per row behaves like an identifier.
IDENTIFIER_UNIQUE_RATIO: float = 0.95

# Missing value severity thresholds (fraction of column values).
HIGH_MISSING_RATIO_WARNING: float = 0.20
HIGH_MISSING_RATIO_ERROR: float = 0.50

# A class is reported as dominating the target above this percentage.
IMBALANCE_MAJORITY_PERCENTAGE: float = 70.0

# Number of most frequent categorical values kept in the statistics report.
TOP_CATEGORICAL_VALUES: int = 5

# Heuristic target-column detection. These are conventions only: the detector
# reports a confidence level and the user can always supply the target.
TARGET_NAME_HINTS: List[str] = [
    "target", "label", "outcome", "class", "class_label", "response",
    "result", "status", "diagnosis", "survived", "prediction", "dependent",
    "y", "churn", "default", "fraud", "category",
]
TARGET_NAME_PENALTIES: List[str] = [
    "id", "index", "uuid", "key", "no", "num", "number", "row", "timestamp",
    "date", "time", "unnamed",
]
# Minimum score required before automatic detection is accepted.
TARGET_DETECTION_MIN_SCORE: float = 2.5
# Score boundaries used to convert a raw score into a confidence level.
TARGET_DETECTION_HIGH_SCORE: float = 6.0
TARGET_DETECTION_MEDIUM_SCORE: float = 4.0
# How many scored candidates to keep for transparency in reports/UI.
TARGET_DETECTION_TOP_CANDIDATES: int = 5

# --------------------------------------------------------------------------- #
# Preprocessing settings (Phase 3)
# --------------------------------------------------------------------------- #
# Median imputation is the default for numerical columns because it is much
# more robust to outliers than the mean.
NUMERICAL_IMPUTATION_STRATEGY = "median"
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"
ENCODING_STRATEGY = "one_hot"
# Unseen categories in test data must not break the pipeline.
ENCODING_HANDLE_UNKNOWN = "ignore"
ENCODING_SPARSE_OUTPUT = True

# Scaling is configurable so that both distance-based models (Logistic
# Regression, KNN, SVM) and tree-based models can be supported later by the
# Model Selection Agent, without changing this architecture.
SCALE_NUMERIC = True
SCALING_STRATEGY = "standard"

# Exact duplicate rows are removed from the in-memory training frame, never
# from the source file.
DUPLICATE_STRATEGY = "remove_exact_duplicates"

# A categorical column is excluded instead of one-hot encoded when its encoded
# width becomes unreasonable.
HIGH_CARDINALITY_MAX_UNIQUE_RATIO = 0.50
HIGH_CARDINALITY_MAX_ENCODED_FEATURES = 50

# The pipeline is only ever fitted on training data.
PREPROCESSING_FIT_SCOPE = "training_data_only"
PREPROCESSING_REMAINDER = "drop"

# --------------------------------------------------------------------------- #
# Classification models (Phase 4)
# --------------------------------------------------------------------------- #
# Display names kept for backwards compatibility. The real registry, with the
# estimator factories and their baseline parameters, lives in
# tools/model_tools.py - nothing is generated dynamically.
CLASSIFICATION_MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "K-Nearest Neighbors": "knn",
    "Support Vector Machine": "svm",
}

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1_score"]

# Baseline (untuned) parameters. Hyperparameter search is Phase 6.
LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}
DECISION_TREE_PARAMS: dict = {
    "random_state": RANDOM_STATE,
}
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
}
KNN_PARAMS: dict = {"n_neighbors": 5}
SVM_PARAMS: dict = {
    # probability=True trains an internal 5-fold Platt scaling, which is slow
    # and unused here because the experiments only need hard labels.
    "probability": False,
    "random_state": RANDOM_STATE,
}

# --------------------------------------------------------------------------- #
# Training and evaluation (Phase 4)
# --------------------------------------------------------------------------- #
# TEST_SIZE and CV_FOLDS are defined in the shared defaults block above.
# Metrics use weighted averages so they stay meaningful for multiclass and for
# imbalanced binary targets. zero_division=0 keeps them defined when a model
# never predicts a rare class.
METRIC_AVERAGE = "weighted"
ZERO_DIVISION = 0

# Baseline cross-validation. The number of folds is reduced automatically when
# a class has too few samples to support 5 stratified folds.
CV_SHUFFLE = True
CV_MIN_FOLDS = 2
# Below this many samples the dataset is too small for a meaningful split.
MIN_SAMPLES_FOR_STRATIFIED_SPLIT = 10
MIN_SAMPLES_PER_CLASS_FOR_STRATIFICATION = 2

# Ranking rule for the baseline comparison.
BEST_MODEL_PRIMARY_METRIC = "f1_score"
BEST_MODEL_SECONDARY_METRIC = "accuracy"

# --------------------------------------------------------------------------- #
# Evaluation thresholds (Phase 5)
# --------------------------------------------------------------------------- #
# The Evaluation Agent does not train anything: it interprets the records the
# Training Agent already produced. The numbers below only decide *how a measured
# difference is worded* in the report.

# Difference between the hold-out score and the cross-validation mean, reported
# as `holdout - cv`. Classification boundaries:
#     |gap| <  GAP_CLOSE_THRESHOLD                              -> close
#     GAP_CLOSE_THRESHOLD <= |gap| < GAP_SUBSTANTIAL_THRESHOLD -> moderate
#     |gap| >= GAP_SUBSTANTIAL_THRESHOLD                       -> substantial
GAP_CLOSE_THRESHOLD = 0.03
GAP_SUBSTANTIAL_THRESHOLD = 0.10

GAP_CLASSIFICATION_RULE = (
    f"gap = holdout - cross-validation; |gap| < {GAP_CLOSE_THRESHOLD} is 'close', "
    f"{GAP_CLOSE_THRESHOLD} <= |gap| < {GAP_SUBSTANTIAL_THRESHOLD} is 'moderate', "
    f"|gap| >= {GAP_SUBSTANTIAL_THRESHOLD} is 'substantial'"
)

# Repeated verbatim in the report so the heuristics can never be mistaken for a
# statistical test.
GAP_HEURISTIC_DISCLAIMER = (
    "Reporting heuristic, not a statistical test: the hold-out/cross-validation gap "
    "is measured on one split of a small dataset, so its size alone neither proves "
    "nor disproves overfitting or instability."
)

OVERFITTING_CAUTION_NOTE = (
    "A large difference between cross-validation and hold-out performance may indicate "
    "sensitivity to the selected split and warrants further investigation."
)

# Number of confusion pairs / hardest classes named in the report.
TOP_CONFUSION_PAIRS = 3
TOP_ERROR_CLASSES = 3

# Per-class metrics: Phase 4 stored weighted metrics and one confusion matrix.
# Per-class precision/recall/F1 are derived from that matrix when it is present,
# so no model has to be retrained. These strings document both cases.
PER_CLASS_DERIVATION_NOTE = (
    "Per-class precision, recall and F1 were derived from the stored hold-out "
    "confusion matrix; no model was retrained."
)
PER_CLASS_UNAVAILABLE_NOTE = (
    "Per-class precision/recall not available from Phase-4 records."
)

# Limitations the Evaluation Agent always reports about a baseline run.
LIMITATION_SINGLE_SPLIT = (
    "The hold-out result comes from a single train/test split (one random_state), so it "
    "is one estimate of generalisation, not a repeated-measures estimate."
)
LIMITATION_NO_TRAINING_METRICS = (
    "Phase 4 did not record training-set metrics, so a train-versus-test gap cannot be "
    "reported and no conclusion about overfitting can be drawn from this run."
)
LIMITATION_NO_TUNING = (
    "All models use baseline (untuned) parameters; the ranking describes the default "
    "configuration of each algorithm, not the best configuration it could reach."
)
LIMITATION_SMALL_TEST_SPLIT = (
    "The hold-out split contains {test_samples} row(s), so a single sample changes the "
    "accuracy by roughly {step:.4f} and the reported metrics are coarse."
)
LIMITATION_SINGLE_RUN_TIMING = (
    "Execution times come from one run on one machine without repetitions, so they are "
    "indicative only and must not be read as a benchmark."
)
LIMITATION_NO_TUNING_RECOMMENDATION = (
    "Hyperparameter optimization (Phase 6) can change this ranking; the selection below "
    "is the best BASELINE model."
)

# --------------------------------------------------------------------------- #
# Hyperparameter optimization (Phase 6)
# --------------------------------------------------------------------------- #
# The Hyperparameter Optimization Agent tunes only the most promising baseline
# models that Phase 5 ranked. Two models x 20 trials x 5 CV folds stays a few
# seconds on a student laptop, which is the point: this project has to be
# reproducible on modest hardware, so the defaults are deliberately small and
# every one of them can be overridden through the agent's arguments.

# How many models from the Phase 5 ranking are tuned. The remaining models keep
# their baseline parameters, so the report can never claim a fair comparison
# between a tuned model and an untuned one.
OPTUNA_TOP_MODELS = 2

# Trials per model. This is the trial budget, not the number of models.
OPTUNA_N_TRIALS = 20

# Optuna is seeded, so two runs with the same data and configuration suggest the
# same parameters and reach the same best trial. The seed also fixes the
# `random_state` the tuned estimators receive, so the whole search is
# reproducible rather than only its sampling step.
OPTUNA_RANDOM_SEED = RANDOM_STATE

# Optional wall-clock limit in seconds. None means "no timeout", which keeps the
# run deterministic: a timeout would cut the search off at a machine-dependent
# moment and make the result depend on how fast the laptop happens to be.
OPTUNA_TIMEOUT: Optional[int] = None

# TPE (Tree-structured Parzen Estimator) is Optuna's default sampler and is a
# reasonable choice for small, low-dimensional spaces.
OPTUNA_SAMPLER_NAME = "tpe"

# "maximize" because the objective is the cross-validated F1 score.
OPTUNA_DIRECTION = "maximize"

# The objective is the Phase 4 weighted F1, so an optimized model is directly
# comparable with the baseline cross-validated score. Phase 4 already made that
# choice for imbalanced and multiclass targets; changing it here would make the
# baseline/optimized comparison meaningless.
OPTUNA_OBJECTIVE_METRIC = "f1_weighted"

# Optuna is chatty by default and prints a full traceback for every failed
# trial. The project logger owns the console output and records the failure of
# every trial itself (see `optimization_tools.optimize_model`), so Optuna's own
# reporting is quietened to avoid printing the same error twice. Nothing is
# hidden: each failure ends up in the trial records and in the report.
OPTUNA_SHOW_PROGRESS_BAR = False
OPTUNA_LOG_LEVEL = "ERROR"

# Objective round and sampling parameters.

# Search-space bounds. The ranges are wide enough to matter and small enough to
# stay explainable and fast. Nothing here is dataset-specific.
OPTUNA_LOGISTIC_C_RANGE = (1e-3, 1e2)
# Convergence is not tuned: the baseline uses 1000, and a saga/lbfgs fit on a
# standardised matrix converges well inside 2000 iterations. Keeping max_iter
# fixed removes a dimension that would only add trials without adding insight.
OPTUNA_LOGISTIC_MAX_ITER = 2000
OPTUNA_LOGISTIC_SOLVERS_BINARY = ("lbfgs", "liblinear")
OPTUNA_LOGISTIC_SOLVERS_MULTICLASS = ("lbfgs", "saga", "newton-cg")

OPTUNA_TREE_MAX_DEPTH_RANGE = (1, 20)
OPTUNA_TREE_MIN_SAMPLES_SPLIT_RANGE = (2, 20)
OPTUNA_TREE_MIN_SAMPLES_LEAF_RANGE = (1, 10)
OPTUNA_TREE_CRITERIA = ("gini", "entropy", "log_loss")

OPTUNA_FOREST_N_ESTIMATORS_RANGE = (10, 200)
OPTUNA_FOREST_N_ESTIMATORS_STEP = 10
# None means "grow trees without a depth limit"; it is a legitimate value for
# max_depth, so it is offered explicitly instead of being hidden behind a
# sentinel the caller would have to translate.
OPTUNA_FOREST_MAX_DEPTH_CHOICES = (None, 3, 5, 8, 12, 20)
OPTUNA_FOREST_MAX_FEATURES_CHOICES = ("sqrt", "log2", None)

OPTUNA_KNN_N_NEIGHBORS_RANGE = (1, 30)
OPTUNA_KNN_WEIGHTS_CHOICES = ("uniform", "distance")
OPTUNA_KNN_METRIC_CHOICES = ("minkowski", "euclidean", "manhattan")

OPTUNA_SVM_C_RANGE = (1e-2, 1e2)
OPTUNA_SVM_KERNEL_CHOICES = ("rbf", "linear", "poly")
# gamma is ignored by the linear kernel, so it is only suggested when the chosen
# kernel can use it. Suggesting it unconditionally would waste trials and report
# a parameter that has no effect on the model.
OPTUNA_SVM_GAMMA_RANGE = (1e-4, 1e1)
OPTUNA_SVM_GAMMA_LOG = True
# "scale" is scikit-learn's documented default and is the value recorded for a
# linear kernel, which ignores gamma anyway.
OPTUNA_SVM_LINEAR_GAMMA = "scale"

# A percentage improvement is only reported when the baseline denominator is
# large enough to make the ratio meaningful. This is the safety floor: below it
# the percentage is withheld and the absolute improvement is reported instead.
OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE = 1e-9

# --------------------------------------------------------------------------- #
# Optimization wording (Phase 6)
# --------------------------------------------------------------------------- #
# Reported with every run so a tuned score is never mistaken for a guarantee.
OPTUNA_IMPROVEMENT_CAUTION = (
    "A higher cross-validation score after tuning means a better configuration was "
    "found for this search budget. It does not guarantee better performance on unseen "
    "data, and it does not remove the risk of overfitting the cross-validation folds."
)

OPTUNA_HOLDOUT_NOTE = (
    "Hold-out metrics were measured once, after the hyperparameters were selected, on "
    "the split that was never used for the search. They are the independent comparison "
    "with the baseline, not a guarantee of future performance."
)

OPTUNA_NO_IMPROVEMENT_NOTE = (
    "No cross-validation improvement was measured for this model. On a small dataset a "
    "search that small often cannot beat the default parameters, and that outcome is "
    "reported rather than hidden."
)

OPTUNA_SELECTION_NOTE = (
    "The hyperparameters were chosen using the training split only, through "
    "cross-validation. The hold-out split was not used to search, to suggest "
    "hyperparameters or to choose which models to optimize."
)

OPTUNA_TRIAL_BUDGET_NOTE = (
    "With {trials} trial(s) per model the search covers only a small part of the space, "
    "so the best configuration found is the best of the configurations tried, not a "
    "proven optimum."
)

OPTUNA_SINGLE_RUN_TIMING_NOTE = (
    "Optimization times come from one run on one machine without repetitions and are "
    "indicative only, so they must not be read as a benchmark."
)

# Limitations the Optimization Agent always reports.
OPTUNA_LIMITATION_TRIAL_BUDGET = (
    "Each model was tuned with only {trials} trial(s) on a single dataset. A search that "
    "small can miss a better region of the space, so the reported configuration is the "
    "best of the configurations tried rather than a proven optimum."
)
OPTUNA_LIMITATION_NO_REPEATED_SEARCH = (
    "The search was run once with a single sampler seed. Repeated searches with "
    "different seeds were not compared, so the stability of the search result itself is "
    "unknown."
)
OPTUNA_LIMITATION_SELECTION_ON_CV = (
    "The reported optimized cross-validation score is the value that was maximized, so "
    "it is optimistically biased; the hold-out score is the independent check."
)
OPTUNA_LIMITATION_HOLDOUT_RANKING = (
    "The tuned models are ranked with the project's documented rule (F1, then "
    "accuracy), which reads the hold-out scores, so the winner's hold-out numbers are "
    "no longer an untouched estimate of generalisation. The hyperparameters themselves "
    "were chosen on the training split only."
)
OPTUNA_LIMITATION_SINGLE_SPLIT = (
    "Every number here comes from a single train/test split, so the hold-out comparison "
    "is one estimate and not a repeated-measures estimate."
)
OPTUNA_LIMITATION_SMALL_TEST_SPLIT = (
    "The hold-out split contains {test_samples} row(s), so a single sample changes the "
    "accuracy by roughly {step:.4f} and the reported metrics are coarse."
)
OPTUNA_LIMITATION_UNTUNED_MODELS = (
    "Only the top {selected} of {candidates} baseline model(s) were tuned. The remaining "
    "models still use their default parameters, so their scores are not a fair "
    "comparison with a tuned model."
)
OPTUNA_LIMITATION_NO_TRAINING_METRICS = (
    "Training-set metrics are still not recorded, so a train-versus-test gap cannot be "
    "reported and no conclusion about overfitting can be drawn from this run."
)
OPTUNA_LIMITATION_NO_DOMAIN_VALIDATION = (
    "The datasets are small synthetic samples with no domain meaning attached to the "
    "labels, so nothing in this run shows how the models would behave on real data."
)

# --------------------------------------------------------------------------- #
# Phase 7: LangGraph orchestration
# --------------------------------------------------------------------------- #
# The graph coordinates the six existing agents. It introduces no machine
# learning logic, so every number it reports still comes from the agents.

# How many times a failed node is retried before the workflow stops. 1 means
# "try once, then retry once, then give up". 0 disables retrying entirely.
LANGGRAPH_MAX_RETRIES = 1

# Master switch for the Optuna search when the graph decides whether to
# optimize. Set to False (or pass --no-optimize) for the evaluation-only
# workflow. This mirrors the Phase 6 CLI flag and never overrides an explicit
# caller argument.
LANGGRAPH_ENABLE_OPTIMIZATION = True

# When True a required node that ultimately fails ends the workflow instead of
# continuing into the next node with incomplete state. Phase 7 deliberately
# refuses to continue past a failure: the downstream agents would raise anyway.
LANGGRAPH_STOP_ON_FAILURE = True

# Hard ceiling on graph steps. A routing bug can never turn into an unbounded
# run: the graph stops and reports the breach instead of looping forever.
LANGGRAPH_MAX_STEPS = 40

# Prefix for the one-line node logs, e.g. "[GRAPH] Training -> COMPLETED".
LANGGRAPH_LOG_PREFIX = "[GRAPH]"

# Error types that are NOT retried. These are the project's own error taxonomy,
# raised for a deterministic reason (missing file, unmet prerequisite, invalid
# configuration). Repeating the same call with the same inputs would produce the
# same error, so retrying them would only waste time. Any other exception is
# treated as unexpected and is retried. Names are used instead of the classes so
# that utils.config keeps no dependency on the tools package.
LANGGRAPH_NON_RETRYABLE_ERRORS = (
    "DataToolError",
    "PreprocessingError",
    "ModelToolError",
    "TrainingToolError",
    "EvaluationToolError",
    "OptimizationToolError",
)

# Executable names that would mean an LLM had been added to the runtime. Kept as
# data so tests can assert the project never depends on one. Phase 8 does not
# change this list: it reaches Ollama through the standard library's
# `urllib.request`, so none of these modules is ever imported and the graph's
# routing decisions stay free of any language model.
LANGGRAPH_FORBIDDEN_LLM_DEPENDENCIES = (
    "ollama",
    "openai",
    "google.generativeai",
    "anthropic",
    "cohere",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_community",
)

LANGGRAPH_ROUTING_RULE = (
    "Optimization runs only when every measurable condition holds: optimization is "
    "enabled by configuration, the evaluation phase finished successfully, at least one "
    "baseline experiment produced usable metrics, and the optimization agent has not "
    "already run in this workflow. The decision is computed from state values only, so "
    "it is deterministic, and no language model is involved at any point."
)

LANGGRAPH_RETRY_RULE = (
    "A node that raises an unexpected error is retried at most {max_retries} time(s). "
    "Errors raised by the project's own error taxonomy (missing file, unmet prerequisite, "
    "invalid configuration) are not retried, because they are deterministic and would "
    "fail identically on a second attempt. Retrying is bounded, so no node can loop."
)

LANGGRAPH_LIMITATION_NO_LLM = (
    "Every routing decision is a comparison of recorded state values. The optional Phase 8 "
    "local LLM explains results after a decision has been recorded, but it never influences "
    "the route, so the workflow stays reproducible either way; a run with --llm produces the "
    "same machine learning numbers as a run without it."
)
LANGGRAPH_LIMITATION_STEP_LIMIT = (
    "The graph stops if it exceeds {max_steps} step(s). That bound is a safety net against "
    "a routing bug rather than a normal operating limit; a healthy run uses far fewer "
    "steps than the ceiling."
)
LANGGRAPH_RECOMMENDATION_NO_LLM = (
    "Keep the default routing: it is inspectable and reproducible. The Phase 8 local LLM "
    "reads a summarised context and proposes a plan; the measurable conditions that guard "
    "this graph stay in charge of stopping the run and of accepting any result."
)

# --------------------------------------------------------------------------- #
# Phase 8: optional local LLM (Ollama)
# --------------------------------------------------------------------------- #
# Phase 8 adds an explanation and planning layer that is OFF by default. Nothing
# in Phases 1-7 changes: without --llm the project never opens a socket, never
# builds a prompt and never touches this module.
#
# The architectural rule that makes the layer safe: the deterministic pipeline
# stays authoritative. The language model may explain, interpret and propose, but
# it never decides whether the data is valid, whether training succeeded, what a
# score is, whether the search succeeded, or whether the workflow should
# continue. Those remain comparisons of recorded values (see graph/routing.py).

# Master switch. False means the whole layer is inert and no client is built.
OLLAMA_ENABLED = False

# The local Ollama server. This is a loopback address on purpose: the project
# must never be able to send project data to a remote host. `LLM_REMOTE_HOSTS_*
# below is what enforces that. Read from the environment like the other Phase 8
# knobs (see .env.example): the client still refuses to build for any non-loopback
# host, so this only changes *which* local address is used.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Local open-source model. Deliberately small: this is a student project on a
# laptop, so the default must not require a GPU and must not be auto-downloaded.
#
# The value was chosen from the lightweight, openly licensed models that Ollama
# ships (`ollama pull <name>`), balancing three constraints: a small memory
# footprint for CPU-only hardware, reliable JSON output (the parser requires it),
# and a permissive licence. It is only a DEFAULT: the actual model must already
# be present locally, the project never downloads one, and an unavailable model is
# reported as such rather than silently substituted.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Wall-clock budget for one HTTP call. Small local models answer in seconds on
# CPU; a short timeout is what keeps a stalled request from stalling the workflow.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

# One retry for a transient failure (connection reset, 5xx, timeout). The same
# budget logic as the graph's node retry: bounded, so no call can loop.
OLLAMA_MAX_RETRIES = 1

# 0.0 keeps the model as close to deterministic as a local LLM gets. It still is
# not a guarantee, which is another reason the layer explains rather than decides.
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# Ollama's `format` option makes the *server* constrain generation to valid JSON.
# This is a first line of defence only: the parser validates the text itself,
# because a model can still return a syntactically valid but wrong object.
OLLAMA_JSON_FORMAT = True

# One controlled repair request when the first response is not usable JSON. It is
# bounded to a single attempt so a malformed model cannot produce a retry storm.
LLM_ALLOW_ONE_REPAIR = True

# Maximum number of characters kept from a raw model response for diagnostics. A
# response is never stored in the state: this bound only protects logs.
LLM_RAW_RESPONSE_LOG_LIMIT = 400

# Maximum number of list items sent per context field, and the maximum number of
# context fields, so a large dataset cannot produce an unbounded prompt.
LLM_CONTEXT_MAX_LIST_ITEMS = 12
LLM_CONTEXT_MAX_FIELDS = 40

# Hosts the client is allowed to contact. Enforced in llm/client.py. A remote
# host would break the local-only guarantee, so a non-loopback base URL is
# rejected rather than merely warned about.
LLM_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")

# Environment variable names that must never reach the model. Declared as data so
# the client can assert on them and tests can check the filter directly.
LLM_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "SESSION",
    "AUTH",
)

# Prefix for the one-line LLM log entries, e.g. "[LLM] Data Analysis -> COMPLETED".
LLM_LOG_PREFIX = "[LLM]"

# --------------------------------------------------------------------------- #
# Phase 8: wording
# --------------------------------------------------------------------------- #
# Reported with every run so a generated explanation is never mistaken for a
# measurement. The wording is deliberately the same discipline Phases 5 and 6
# already established: an observation is separated from an interpretation, and an
# interpretation is separated from a hypothesis.

LLM_ROLE_STATEMENT = (
    "The language model is an optional assistant. It explains and interprets results that "
    "the deterministic pipeline already measured. It does not measure, train, validate, "
    "decide or control the workflow."
)

LLM_AUTHORITY_RULE = (
    "The deterministic pipeline is the only authority for measurable claims. Every number "
    "in this run was computed by the pipeline itself; the language model only describes, "
    "interprets and proposes. It cannot change a metric, a ranking, a model, a dataset, a "
    "search space or a routing decision."
)

# Phrases a generated explanation must never contain, because each of them turns
# a measured result into an unearned guarantee. The parser checks the model's
# text for these and records a violation instead of passing the claim through.
LLM_FORBIDDEN_CLAIMS = (
    "proves overfitting",
    "proves that",
    "guarantees",
    "guaranteed",
    "definitely generalizes",
    "will work perfectly",
    "production ready",
    "production-ready",
    "certainly achieves",
    "is the optimal model",
)

# The accepted replacements, offered to the model so a forbidden phrase has a
# documented alternative rather than merely being banned.
LLM_ALLOWED_CLAIM_WORDS = (
    "observed",
    "measured",
    "within this experiment",
    "may indicate",
    "could warrant further investigation",
    "this run",
)

LLM_CONFIDENCE_LEVELS = ("low", "medium", "high")

LLM_NO_METRIC_INVENTION = (
    "Never invent a metric, a value, a dataset, an experiment, a model or a result. If a "
    "number is not present in the supplied context, it does not exist for this run and must "
    "not be mentioned at all."
)

LLM_NO_STATISTICAL_CLAIM = (
    "Never claim statistical proof. A single train/test split and a cross-validation mean "
    "do not establish significance, and a difference between them is a reported observation "
    "rather than a statistical test."
)

LLM_NO_RETRAINING = (
    "Never train, fit, tune or re-run anything, never propose executable code, and never "
    "define or alter a hyperparameter search space. Any optimisation suggestion is a "
    "priority hint for the existing Optuna search, which remains in full control."
)

LLM_NO_DATA_MODIFICATION = (
    "Never suggest changing, cleaning or modifying the dataset values, and never suggest "
    "ignoring or overriding a validation failure or a recorded agent failure."
)

LLM_OBSERVATION_SEPARATION = (
    "Separate the three registers explicitly. OBSERVATION is a measured value from the "
    "supplied context. INTERPRETATION is a plausible reading of an observation. HYPOTHESIS "
    "is a possible explanation that still needs investigation. Never present a hypothesis "
    "as a fact and never present an interpretation as a measurement."
)

LLM_PRIVACY_RULE = (
    "You receive a summarised, non-sensitive context only: no raw rows, no credentials, no "
    "environment variables and no file contents. Do not ask for, reconstruct or guess data "
    "that was not supplied."
)

LLM_FALLBACK_NOTE = (
    "The local language model was unavailable, so this section contains no generated "
    "interpretation. Every measured result in the pipeline is unaffected: this layer only "
    "ever adds explanation."
)

LLM_DISABLED_NOTE = (
    "The optional local language model is disabled for this run (OLLAMA_ENABLED = False, or "
    "--llm was not passed). The core machine learning pipeline does not require an LLM."
)

LLM_LOCALITY_NOTE = (
    "The model runs locally through Ollama on this machine. No project data, prompt or "
    "response leaves the host, and no API key is configured or required."
)

LLM_MODEL_UNAVAILABLE_NOTE = (
    "The configured model is not installed locally. Pull it yourself with "
    "'ollama pull <model>' if you want generated explanations; the project never downloads "
    "a model automatically."
)

LLM_LIMITATION_GENERATED = (
    "The explanations, interpretations and suggestions below were generated by a local "
    "language model and are not measurements. They may be wrong, they are not verified, and "
    "nothing in the pipeline was changed because of them."
)

LLM_LIMITATION_LOCAL_MODEL = (
    "A small locally hosted model is used so the project runs on modest hardware. Its "
    "reasoning quality is well below a hosted model, so its text should be read as a "
    "starting point for review rather than as an expert conclusion."
)

LLM_LIMITATION_SINGLE_SPLIT = (
    "The context sent to the model describes one dataset and one run. The model cannot see "
    "the raw data, so it cannot verify anything it says and must not be treated as a "
    "second opinion on the measurements."
)

LLM_LIMITATION_OPTUNA_AUTHORITY = (
    "An optimisation plan is a priority hint only. Optuna searched its own pre-defined "
    "spaces and the deterministic candidate rules chose which models were tuned, so a "
    "suggestion that was not followed changed no result."
)

LLM_LIMITATION_NO_CLAIM_CHECK = (
    "Responses are requested as JSON, parsed, and checked against the expected field names, "
    "and phrasing that would overstate certainty is recorded as a violation. The model "
    "output is not fact-checked against the measurements, which is not possible locally "
    "without re-running the pipeline."
)

LLM_RECOMMENDATION_REVIEW = (
    "Read the generated text as a draft for review. When it disagrees with a measured "
    "value, the measurement is correct and the text is wrong."
)

# --------------------------------------------------------------------------- #
# Phase 9: model persistence and dashboard
# --------------------------------------------------------------------------- #
# Phase 9 does two things: it writes the fitted pipeline to disk, and it reads
# that file back for a Streamlit dashboard. It adds no machine learning logic.
#
# The rule that keeps the addition honest: the persisted pipeline is the one
# Phases 4-6 already fitted and scored. Phase 9 never re-selects a model with a
# new criterion, never re-runs a search, never recomputes a metric and never
# touches the hold-out split.

# The three artifacts Phase 9 writes. The names are stable so the dashboard and
# the README can refer to them without importing the code that creates them.
PERSISTENCE_MODEL_FILENAME = "best_model.joblib"
PERSISTENCE_METADATA_FILENAME = "best_model_metadata.json"
PERSISTENCE_RESULTS_FILENAME = "run_results.json"

# Bumped only when the *shape* of a metadata or results document changes in a
# way a reader would notice. The dashboard refuses a document whose version it
# does not understand instead of guessing.
PERSISTENCE_SCHEMA_VERSION = 1

# The directory the artifacts are written to. ``None`` means
# ``config.MODELS_DIR``; an explicit directory is what lets a test write into a
# temporary folder instead of the project tree.
PERSISTENCE_OUTPUT_DIR: Optional[Path] = None

# Whether the graph persists the model by default. True means a plain
# ``app.py --graph`` run leaves a usable artifact behind, which is what the
# dashboard needs. ``--no-save-model`` turns it off for the same reason
# ``--no-optimize`` exists: to run the pipeline without producing extra output.
PERSISTENCE_ENABLED = True

# Prefix for the one-line persistence logs, e.g. "[PERSIST] Saved best_model.joblib".
PERSISTENCE_LOG_PREFIX = "[PERSIST]"

# ``compress`` shrinks the artifact noticeably and costs a little CPU. It is the
# right trade for a student laptop that will store a few small pipelines.
PERSISTENCE_JOBLIB_COMPRESS = 3

# The final-model selection rule. It is *not* a new ranking policy: it reuses
# ``BEST_MODEL_PRIMARY_METRIC`` (f1_score) and ``BEST_MODEL_SECONDARY_METRIC``
# (accuracy), which are the metrics the Phase 4, Phase 5 and Phase 6 rankings
# already use, and it never sees a metric those rankings do not record.
PERSISTENCE_SELECTION_RULE = (
    "Final model selection reuses the project's existing ranking: primary "
    "{primary} (descending), secondary {secondary} (descending), then the model "
    "name (ascending) as the final tie-break. The optimized candidate is the Phase 6 "
    "winner (best_optimized_model) and the baseline candidate is the Phase 4/5 winner "
    "(best_baseline_model); both are compared on the hold-out split they were already "
    "scored on. When optimization was disabled, skipped or produced no usable winner, "
    "the baseline candidate is the final model. No new metric and no new threshold is "
    "introduced here."
).format(
    primary=BEST_MODEL_PRIMARY_METRIC, secondary=BEST_MODEL_SECONDARY_METRIC
)

PERSISTENCE_NOTE_FINAL_FIT = (
    "The final fit happens after the final model has been selected, and only to produce "
    "a deployable artifact. When the selected pipeline is already fitted it is reused "
    "unchanged; otherwise the estimator is rebuilt from the selected parameters and the "
    "Phase 3 preprocessing definition, and refitted on the same training split "
    "(test_size={test_size}, random_state={random_state}) that Phases 4-6 used. The "
    "hold-out rows are never used for fitting or tuning, so the metrics recorded in the "
    "metadata still describe the model that was saved."
).format(test_size=TEST_SIZE, random_state=RANDOM_STATE)

PERSISTENCE_NOTE_BASELINE_FALLBACK = (
    "The best valid baseline model is persisted because no usable optimized model was "
    "available. The Phase 5 evaluation result is preserved unchanged."
)

PERSISTENCE_NOTE_OPTIMIZED = (
    "The optimized model is persisted because it scored at least as high as the best "
    "baseline on the hold-out split under the project's existing ranking rule."
)

PERSISTENCE_NOTE_NO_METRICS = (
    "The recorded metrics are the ones measured before persistence. Persistence "
    "recomputes nothing."
)

PERSISTENCE_LIMITATION_SINGLE_SPLIT = (
    "One train/test split was used. The hold-out score is a single estimate and no "
    "repeated validation was performed, so it carries the usual uncertainty of a "
    "single estimate."
)

PERSISTENCE_LIMITATION_NO_UPLOAD = (
    "The dashboard loads exactly one model file: the artifact this project wrote. "
    "It offers no way to upload a .joblib or .pkl file, because loading one executes "
    "arbitrary code inside the Python process and is not something a local dashboard "
    "can make safe. CSV datasets are accepted for training in the Train a Model "
    "section; they are data, not code."
)

PERSISTENCE_LIMITATION_NO_RETRAIN = (
    "The dashboard starts a workflow only when you press Start ML Workflow. Viewing "
    "the page, switching sections or making a prediction never trains, tunes or "
    "re-evaluates a model, and every displayed number still comes from the artifacts "
    "a workflow wrote."
)

DASHBOARD_TITLE = "Autonomous ML System"
DASHBOARD_TAGLINE = "Build, compare, and optimize machine learning models."
DASHBOARD_SUBTITLE = (
    "Data Analysis - Preprocessing - Model Selection - Training - Evaluation - "
    "Optimization - Model Persistence"
)
DASHBOARD_NO_ARTIFACT = "No persisted model is available yet."
DASHBOARD_NO_ARTIFACT_GUIDANCE = (
    "Open the Train a Model section, upload a CSV dataset and select its target "
    "column, or run the workflow from the command line: python app.py --graph"
)
DASHBOARD_INVALID_ARTIFACT = (
    "The persisted artifact exists but could not be read. The machine learning results "
    "are not affected; run the workflow again to rewrite the artifacts."
)
DASHBOARD_NO_CONFUSION_MATRIX = (
    "No confusion matrix was stored for the selected model, so nothing can be drawn "
    "here. This is a missing artifact, not an error."
)
DASHBOARD_NO_OPTIMIZATION = (
    "Hyperparameter optimization was not performed for this run, so there is nothing to "
    "compare. The persisted model is the best baseline."
)
DASHBOARD_NO_LLM = (
    "No local language model interpretation was stored for this run. The optional Ollama "
    "layer is off by default and the dashboard works without it."
)
DASHBOARD_NO_PREDICTION = (
    "No prediction was produced. Fill in the feature form and press Predict."
)
DASHBOARD_UNTRUSTED_ARTIFACT = (
    "This dashboard accepts CSV datasets for training and never accepts a model "
    "file. Loading a .joblib or .pkl file executes arbitrary code, so the only model "
    "it loads is the artifact this project wrote at the configured path."
)
DASHBOARD_NO_EXPERIMENT_TRACKING = (
    "MLflow tracking is unavailable or disabled. Every other section is read from the "
    "Phase 9 artifacts and is unaffected."
)
DASHBOARD_NO_BENCHMARK = "No Phase 11 benchmark results are available yet."

# Dashboard training workflow (Phase 9 UI extension). These strings are shown in
# the interface; the limits themselves live in tools/training_tools.py.
DASHBOARD_TRAIN_NOTE = (
    "Training runs the existing multi-agent pipeline: data analysis, preprocessing, "
    "model selection, training, evaluation, optional Optuna optimization and model "
    "persistence. The dashboard does not implement a second training path."
)
DASHBOARD_UPLOAD_NOTE = (
    "Uploaded CSV files are saved under datasets/uploads/ so the run can be "
    "reproduced. Only the first five rows are previewed, and the file stays on this "
    "machine. The directory is ignored by Git."
)
DASHBOARD_FAILED_RUN_NOTE = (
    "The most recent training attempt did not complete. The results shown below are "
    "still the last successful run; they are not from the failed attempt."
)

# --------------------------------------------------------------------------- #
# Phase 11: conventional-vs-autonomous benchmark
# --------------------------------------------------------------------------- #
# The benchmark is intentionally *not* run from the dashboard: the dashboard is
# a read-only viewer and its job is identical whether the comparison files exist
# or not. When reports/phase11_comparison.json is missing the section says so
# with DASHBOARD_NO_BENCHMARK; when present it renders the recorded numbers.

# --------------------------------------------------------------------------- #
# Phase 10: MLflow experiment tracking
# --------------------------------------------------------------------------- #
# MLflow records decisions that the deterministic workflow has already made. It
# never selects a model, never triggers a search, never routes the graph and
# never decides whether an artifact is written. Every failure below degrades to
# a recorded status; none of them can turn a successful machine learning run
# into a failed one.

# The directory the local tracking store lives in. It sits under
# ``config.EXPERIMENTS_DIR`` rather than ``models/`` or ``datasets/`` so that
# generated tracking state can never be mistaken for a project artifact and can
# be ignored as one line in ``.gitignore``.
MLFLOW_TRACKING_DIR: Path = EXPERIMENTS_DIR / "tracking"

# The SQLite file backing the local tracking store.
MLFLOW_TRACKING_DB: Path = MLFLOW_TRACKING_DIR / "mlflow.db"


def _sqlite_uri(path: Path) -> str:
    """Build the ``sqlite:///`` tracking URI for a database file.

    MLflow needs the ``sqlite`` scheme spelled out: a bare ``file://`` URI is
    routed to the filesystem store instead, which MLflow 3.x refuses to write to.
    The path is normalised to forward slashes because that is what the SQLAlchemy
    URI parser expects on every platform, including Windows.

    Args:
        path: The SQLite file.

    Returns:
        A tracking URI such as ``sqlite:///C:/projects/app/experiments/tracking/mlflow.db``.
    """
    return "sqlite:///" + path.as_posix()


# The tracking URI MLflow is pointed at. Local-first by default: a SQLite file
# means no server, no account, no network and no daemon. The environment variable
# exists for a developer who wants to point a throwaway experiment at their own
# local ``mlflow server``; nothing in this project requires one.
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", _sqlite_uri(MLFLOW_TRACKING_DB)
)

# The experiment every workflow run is recorded in. One experiment for the whole
# project, not one per dataset or per model: the runs inside it are already
# separated by their tags and parameters, and one run per model would turn a
# five-model comparison into hundreds of near-empty runs.
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "autonomous-multi-agent-ml"
)

# Whether tracking is on by default. True means a plain ``app.py --graph`` run
# records itself; ``--no-mlflow`` and ``MLFLOW_ENABLED=false`` turn it off for
# the same reason ``--no-optimize`` exists - to run the pipeline without
# producing extra output.
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Prefix for the one-line tracking logs, e.g.
# "[MLFLOW] recorded run 4f2a... in experiment autonomous-multi-agent-ml".
MLFLOW_LOG_PREFIX = "[MLFLOW]"

# Seconds to allow for a single tracking call before giving up on it. Tracking
# writes to a local directory, so this only matters if that directory is on a
# slow or stalled filesystem; the default is generous but finite because a hang
# would stall the workflow that is merely being observed.
MLFLOW_OPERATION_TIMEOUT = float(os.getenv("MLFLOW_OPERATION_TIMEOUT", "20"))

# MLflow truncates a parameter value at 6000 characters and a tag value at 8000.
# Values are cut here instead so that a long reason string is logged in a
# predictable, documented way rather than rejected by the tracking client.
MLFLOW_MAX_PARAM_VALUE = 6000
MLFLOW_MAX_TAG_VALUE = 8000

# Whether to copy the fitted pipeline into the run's artifact directory. False
# by default, and deliberately: ``models/best_model.joblib`` remains the primary
# deployment artifact, a second copy would be a second thing to keep in sync,
# and ``mlflow.sklearn.log_model`` would pin a second serialization format the
# project does not otherwise depend on. What is recorded instead is the
# metadata and metrics that describe the persisted model.
MLFLOW_LOG_MODEL_ARTIFACT = False

# How many recent runs the dashboard's history view lists. Bounded on purpose:
# the dashboard is a report, not an MLflow replacement, and an unbounded table
# would make the page slower the more experiments a project accumulated.
MLFLOW_HISTORY_LIMIT = 10

# Sub-directory names inside a run's artifact directory. Named here so the tests
# and the report can refer to them without repeating string literals.
MLFLOW_ARTIFACT_DATASET = "dataset_summary.json"
MLFLOW_ARTIFACT_PREPROCESSING = "preprocessing_config.json"
MLFLOW_ARTIFACT_MODEL_COMPARISON = "model_comparison.json"
MLFLOW_ARTIFACT_CONFUSION_MATRIX = "confusion_matrix.json"
MLFLOW_ARTIFACT_OPTIMIZATION_TRIALS = "optimization_trials.json"
MLFLOW_ARTIFACT_REPRODUCIBILITY = "reproducibility.json"

# The whitelist of distributions whose versions are recorded. Deliberately a
# closed list: dumping ``pip freeze`` would record the whole environment,
# including anything a developer happened to have installed, which is noise at
# best and a disclosure risk at worst.
MLFLOW_VERSION_WHITELIST = (
    "python",
    "pandas",
    "numpy",
    "scikit-learn",
    "optuna",
    "langgraph",
    "mlflow",
    "joblib",
    "streamlit",
)

# MLflow 3.x prints a suggestion to load a tracing skill the first time it is
# imported. It is addressed to an MLflow developer, not to a user running a CSV
# through a workflow, and it would otherwise appear on every single command.
# ``setdefault`` leaves an explicit choice by the developer alone.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"