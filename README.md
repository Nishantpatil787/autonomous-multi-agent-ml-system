# Autonomous Multi-Agent System for Machine Learning Model Development and Optimization

## Overview

This project is an open-source, local-first autonomous multi-agent system that
develops a machine learning model from a tabular CSV dataset with minimal human
intervention. Specialised agents — data analysis, preprocessing, model
selection, training, evaluation and optimization — are coordinated by a
LangGraph state machine, an optional local LLM layer explains completed stages,
final models are persisted as complete fitted pipelines, every graph run is
tracked in a local MLflow store, and a read-only Streamlit dashboard presents
the whole result.

Phases 1–11 (foundation through the conventional-vs-autonomous benchmark) are
complete; Phase 12 added the final research documentation and packaging
(master report, architecture/workflow/comparison diagrams, structure, setup,
demo, results and reproducibility documents).

## Features

- **Automated data analysis** — types, missing values, duplicates, statistics,
  class distribution, target resolution and problem-type detection, all with
  recorded reasoning.
- **Leakage-free preprocessing** — exclusions with reasons, imputation,
  encoding and scaling built as one reproducible `ColumnTransformer`.
- **Controlled model selection** — five candidate families (Logistic
  Regression, Decision Tree, Random Forest, KNN, SVM) from a fixed registry.
- **Deterministic training and evaluation** — stratified split and
  5-fold CV at `random_state = 42`, weighted metrics, confusion matrices and a
  generalization-gap analysis on recorded results only.
- **Autonomous hyperparameter optimization** — Optuna (TPE, bounded spaces,
  seeded, 20 trials per tuned model) evaluated on the training part only.
- **LangGraph orchestration** — explicit routing, a deterministic
  `Should Optimize?` decision, bounded retries and recorded failure handling.
- **Optional local LLM layer** — Ollama over loopback, strict JSON,
  explanation-only; the pipeline works identically without it.
- **Model persistence** — fitted pipeline + metadata + results JSON, atomic
  writes, a single safe `joblib.load` site.
- **Read-only Streamlit dashboard** — 12 numbered sections, never trains.
- **Local MLflow tracking** — experiments, runs, parameters, metrics and history
  in a local SQLite store; failure-isolated and opt-out with `--no-mlflow`.
- **Controlled benchmark** — conventional (21-config grid) vs autonomous on the
  same datasets, split and metrics, with JSON/CSV/PNG outputs and a dashboard
  section, guarded so tests/benchmarks can never mutate canonical artifacts.

## Project Title
Autonomous Multi-Agent System for Machine Learning Model Development and Optimization

## Problem Statement
Machine learning model development usually requires repeated manual work such as data analysis, model selection, and hyperparameter tuning. This project proposes an autonomous multi-agent framework that automates these tasks with minimal human intervention.

## Project Objective
Build a working open-source, local-first autonomous multi-agent machine learning system that can take a tabular dataset from the user and autonomously perform major stages of a conventional machine learning workflow. The system reduces manual effort using specialized agents coordinated through an orchestrator.

## Proposed Architecture
The system follows a modular multi-agent architecture where each agent has a specific responsibility. Agents communicate through a structured shared state object. The high-level workflow is:

1. User uploads dataset
2. Orchestrator Agent (controls workflow and state)
3. Data Analysis Agent (inspects data and determines problem type) — **implemented (Phase 2)**
4. Preprocessing Agent (cleans and transforms data) — **implemented (Phase 3)**
5. Model Selection Agent (selects candidate models) — **implemented (Phase 4)**
6. Training Agent (trains and evaluates models) — **implemented (Phase 4)**
7. Hyperparameter Optimization Agent (optimizes promising models) — Phase 6
8. Evaluation Agent (compares results and selects best model) — Phase 5
9. Final Report / Best Model — Phase 5 / 9

Code layout:
- `agents/` — orchestration logic of each agent (what to do, in which order)
- `tools/` — reusable, testable data operations (how the data work is done)
- `utils/` — shared typed state, configuration and logging (the data contracts)
- `datasets/`, `models/`, `reports/`, `experiments/` — data and output folders

## Agents

| Agent | Responsibility | Status |
|---|---|---|
| **Orchestrator Agent** | Controls workflow, maintains shared state, decides next agent, handles failures, determines completion. | Placeholder (Phase 7) |
| **Data Analysis Agent** | Loads dataset, detects feature types, missing values, duplicates, statistics, class distribution, resolves the target column and detects the problem type. | **Implemented (Phase 2)** |
| **Preprocessing Agent** | Decides which columns become features, handles missing values, encodes categoricals, scales numericals, builds a reproducible sklearn ColumnTransformer. | **Implemented (Phase 3)** |
| **Model Selection Agent** | Selects candidate models from a controlled predefined set (LR, DT, RF, KNN, SVM). | **Implemented (Phase 4)** |
| **Training Agent** | Trains candidates with a train/test split and cross-validation, records metrics (Accuracy, Precision, Recall, F1-score) and timing. | **Implemented (Phase 4)** |
| **Evaluation Agent** | Compares models, builds the confusion matrix, computes metrics, records execution time and experiment count, identifies the best model. | **Implemented (Phase 5)** |
| **Hyperparameter Optimization Agent** | Selects the best baseline models, searches their hyperparameters with Optuna using cross-validated F1 as the objective, and compares the tuned models against the untuned baselines. | **Implemented (Phase 6)** |

## Phase 2: Data Analysis Agent

The agent is read-only: the source dataset is never modified, no rows or columns
are dropped, and no preprocessing or training is performed.

**Capabilities**
1. Dataset shape (rows, columns), column names and pandas dtypes.
2. Feature type detection: numerical, categorical (including booleans) and
   unsupported types such as datetime, which are reported instead of guessed.
3. Missing values: total, per column counts and percentages.
4. Duplicate rows.
5. Basic numerical statistics: count, mean, standard deviation, minimum, maximum.
6. Categorical statistics: unique counts and the most frequent values.
7. Target column resolution:
   - `explicit` when the user provides the column;
   - `automatic` otherwise, using a transparent heuristic score built from
     column-name conventions, data type, cardinality and completeness.
   Automatic detection is a **suggestion only**: the agent returns a
   confidence level, the reason and the full ranking of candidates, and the
   user can always override the target. When no column is convincing, the
   target stays unresolved instead of being guessed.
8. Problem type detection: `binary_classification`, `multiclass_classification`,
   `regression` (detected but not supported yet) or
   `unsupported_or_uncertain`. Classification is never silently assumed.
9. Class distribution: class label, sample count and percentage.
10. Data quality checks (reported only, never fixed here): missing values,
    duplicate rows, constant columns, empty columns, high-cardinality or
    identifier-like columns, unsupported column types, rare target classes and
    strongly imbalanced classes.

**Structured output** (mirrors the requested schema):

```json
{
  "dataset_shape": {"rows": 123, "columns": 9},
  "numerical_features": ["record_id", "age", "bmi", "glucose", "blood_pressure"],
  "categorical_features": ["gender", "region", "smoking"],
  "missing_values": {"total": 17, "affected_columns": 4},
  "duplicate_count": 3,
  "target_column": "Outcome",
  "target_detection_method": "automatic",
  "target_detection_confidence": "high",
  "problem_type": "binary_classification",
  "class_distribution": {
    "1": {"count": 76, "percentage": 61.79},
    "0": {"count": 47, "percentage": 38.21}
  },
  "data_quality_issues": []
}
```

## Phase 3: Preprocessing Agent

The agent consumes the Phase 2 `AgentState` and produces a **reproducible,
unfitted** scikit-learn `ColumnTransformer`. It never fits the pipeline, never
trains a model and never modifies the source dataset.

**Decisions recorded in the state**

| Decision | Default | Reason |
|---|---|---|
| Numerical imputation | `SimpleImputer(strategy="median")` | Median is far more robust to outliers than the mean. |
| Categorical imputation | `SimpleImputer(strategy="most_frequent")` | Keeps the mode of the *training* split. |
| Encoding | `OneHotEncoder(handle_unknown="ignore")` | Unseen test categories become an all-zero row instead of an error. |
| Scaling | `StandardScaler()`, `scale_numeric=True` | Configurable, so tree-based models can run without scaling later. |
| Remainder | `drop` | Excluded columns can never leak into the features. |
| Duplicates | `remove_exact_duplicates` (configurable to `keep_all`) | Applied to the in-memory frame by `remove_duplicate_rows`, never to the CSV file. |

**Feature selection** — columns are excluded with one clear reason each, in this
precedence order (first match wins):

1. `empty_column` — every value is missing
2. `constant_column` — a single distinct value
3. `unsupported_type` — datetime / complex dtypes (date features come later)
4. `identifier_like` — near-unique values (e.g. `record_id`, 98% unique). The
   decision is **ratio based, not type based**, so ordinary integer features are
   never dropped just because they are integers.
5. `high_cardinality` — one-hot encoding would create an unreasonable number of
   columns (thresholds live in `utils/config.py`, not as magic numbers)

**Pipeline architecture**

```
ColumnTransformer(remainder="drop")
├── numerical  : SimpleImputer(strategy="median")      -> StandardScaler()
└── categorical: SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore")
```

**Data leakage prevention**

`build_column_transformer(config)` always returns an **unfitted** pipeline, and
`fit_scope="training_data_only"` is stored in the state. Later training code must
do exactly this:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = build_column_transformer(state.preprocessing_config)  # unfitted
X_train_t = pipeline.fit_transform(X_train)   # medians/modes/means from TRAIN only
X_test_t  = pipeline.transform(X_test)        # never fitted on test data
```

On the sample dataset this is verifiable: the learned median for `glucose` is
`133.0` (training split) while the full-dataset median is `131.0`.

**Serialization safety**

The state holds two things: `preprocessing_config` (a fully JSON-safe dataclass,
the single source of truth) and `preprocessing_pipeline` (the live, unfitted
sklearn object). `AgentState.to_dict()` never serialises the live object — it
reports it under `runtime_objects` as `{"type": "ColumnTransformer", "fitted":
false, ...}`, and the pipeline can always be rebuilt from the configuration with
`build_column_transformer(config)`.

**Structured output**

```json
{
  "preprocessing_status": "completed",
  "preprocessing_config": {
    "numerical_features": ["age", "bmi", "glucose", "blood_pressure"],
    "categorical_features": ["gender", "region", "smoking"],
    "excluded_features": [
      {"column": "record_id", "reason": "identifier_like", "detail": "..."}
    ],
    "numerical_imputation": "median",
    "categorical_imputation": "most_frequent",
    "encoding": "one_hot",
    "encoding_handle_unknown": "ignore",
    "scale_numeric": true,
    "scaling_strategy": "standard",
    "duplicate_handling": {"strategy": "remove_exact_duplicates", "duplicate_rows_detected": 3},
    "remainder": "drop",
    "fit_scope": "training_data_only"
  },
  "runtime_objects": {
    "preprocessing_pipeline": {"type": "ColumnTransformer", "fitted": false}
  }
}
```

**When preprocessing is skipped (not silently)**

If no target was resolved, or the problem type is not binary/multiclass
classification, the agent sets `preprocessing_status = "skipped"`, stores the
reason in `preprocessing_summary.status_reason`, logs a warning and builds no
pipeline. Regression is therefore detected but not silently preprocessed.

## Phase 4: Model Selection Agent and Training Agent

Phase 4 turns the preprocessing *configuration* into real experiments. It adds
five baseline classification models, trains and evaluates each one, compares them
and records a **best baseline model**. Hyperparameter search is deliberately *not*
part of this phase (that is Phase 6).

### Model Selection Agent

The candidate list comes from a **controlled registry** in `tools/model_tools.py`.
There is no dynamic class creation and no LLM: a registry entry is a
`ModelDefinition` holding the display name, family, scaling requirement and
baseline parameters, plus a private `_factory` attribute that builds the
estimator. The factory is never serialised.

| Model | Family | Scaling | Baseline parameters |
|---|---|---|---|
| Logistic Regression | `linear` | required | `max_iter=1000, random_state=42` |
| Decision Tree | `tree_based` | not needed | `random_state=42` |
| Random Forest | `ensemble` | not needed | `n_estimators=100, random_state=42` |
| K-Nearest Neighbors | `instance_based` | required | `n_neighbors=5` |
| Support Vector Machine | `kernel_based` | required | `probability=False, random_state=42` |

The selection rule is transparent and deterministic: every registered model whose
`supported_problem_types` contains the detected problem type is selected, always
in registry order. Nothing is added, scored or invented. Each selected model gets a
`selection_reason`, the excluded ones (if any) get their own reason, and the whole
decision is logged so it can be shown to the user.

The agent is a pure decision step: it never instantiates or fits a model, and it
verifies its prerequisites (Phase 2 done, Phase 3 completed with a usable
configuration, a resolved target and a classification problem type) before doing
anything. If the problem type is regression or uncertain, it **skips** with a
reason instead of guessing.

### Training Agent

**Data flow**

```
CSV (read only)
  -> remove exact duplicates (in memory, Phase 3 strategy)
  -> X / y separation          (target can never enter X)
  -> train_test_split(test_size=0.2, random_state=42, stratify=y when safe)
  -> Pipeline(preprocessor, estimator)  fitted on the TRAINING SPLIT ONLY
       -> predict the test split
       -> accuracy / precision / recall / F1  (average="weighted", zero_division=0)
       -> confusion matrix
       -> StratifiedKFold CV on the training split, on the same Pipeline
```

**Leakage safety — the core rule**

1. Each experiment builds its **own** `ColumnTransformer` and Pipeline. Sharing one
   transformer between models would leak fitted state, because scikit-learn fits
   steps in place.
2. Scaling variants come from `config_for_scaling(base, scale_numeric)`, which
   returns a **new validated** `PreprocessingConfig`. The Phase 3 configuration in
   the state is never mutated, so tree models run unscaled without changing what
   preprocessing recorded.
3. The pipeline is fitted **only** on the training split, and the test split is
   transformed through that fitted pipeline.
4. Cross-validation runs on the **training split only**, using the *unfitted*
   pipeline. `cross_validate` clones and refits it per fold, so preprocessing never
   sees the validation rows, and the caller's pipeline stays unfitted afterwards.
   `n_jobs=1` and `error_score="raise"` keep the run reproducible and loud.

**Splitting, with safe fallbacks**

| Setting | Value |
|---|---|
| `test_size` | `0.2` |
| `random_state` | `42` |
| `stratify` | used when the smallest class allows it, otherwise skipped with a recorded reason |

Same policy for CV: 5 folds, `shuffle=True`, `random_state=42`, reduced safely to
the smallest class in the training split (never below 2). When not even 2-fold
stratified CV is possible, cross-validation is **disabled with a recorded reason**
and the single train/test evaluation is still reported — the run never crashes for
a small dataset.

**Metrics and ranking**

All four metrics use `average="weighted"` with `zero_division=0`, so multiclass
problems and never-predicted classes stay well defined. Confusion matrices are
JSON-safe records with de-duplicated labels (observed ∪ predicted, native Python
types) — this matters because a naive implementation produces a `['0','0','1','1']`
label list with a 4x4 matrix.

The **best baseline model** is ranked by:

1. `f1_score` descending (primary)
2. `accuracy` descending (secondary tie-break)
3. model name, for full determinism

Failed experiments are excluded from the ranking but never hidden.

**Failure handling**

One failing model does not stop the run. It is recorded as
`ExperimentStatus.FAILED` with its error message; if training succeeded but CV
failed, the experiment is `PARTIAL` and its hold-out metrics are still valid.

**Structured output**

```json
{
  "model_selection_status": "completed",
  "selected_models": [
    {"name": "Logistic Regression", "model_type": "linear", "scaling_required": true,
     "default_parameters": {"max_iter": 1000, "random_state": 42},
     "selection_reason": "Supported baseline for binary_classification; uses feature scaling."}
  ],
  "model_selection_summary": {
    "status": "completed", "selected_count": 5,
    "selected_models": ["Logistic Regression", "Decision Tree", "Random Forest",
                        "K-Nearest Neighbors", "Support Vector Machine"],
    "excluded_models": {}, "status_reason": ""
  },
  "training_status": "completed",
  "data_split": {
    "test_size": 0.2, "random_state": 42, "stratified": true, "stratification_reason": "",
    "total_samples": 120, "train_samples": 96, "test_samples": 24,
    "duplicate_rows_detected": 3, "duplicate_rows_removed": 3,
    "duplicate_handling_strategy": "remove_exact_duplicates",
    "input_feature_count": 7, "excluded_feature_count": 1,
    "class_distribution_before_split": {"0": 47, "1": 73},
    "class_distribution_train": {"0": 37, "1": 59},
    "class_distribution_test": {"0": 10, "1": 14}
  },
  "cross_validation_config": {
    "enabled": true, "n_splits": 5, "shuffle": true, "random_state": 42,
    "scoring": {"accuracy": "accuracy", "f1": "f1_weighted"}, "adjusted_reason": ""
  },
  "experiment_results": [
    {"experiment_id": "exp_52da88f2", "model_name": "Logistic Regression",
     "accuracy": 0.875, "precision": 0.8804, "recall": 0.875, "f1_score": 0.8761,
     "cv_accuracy_mean": 0.8953, "cv_f1_mean": 0.8941,
     "transformed_feature_count": 13, "preprocessing_scaling": true,
     "training_time": 0.005, "prediction_time": 0.001, "cv_time": 0.199,
     "status": "success", "error_message": "",
     "confusion_matrix": {"labels": ["0", "1"], "matrix": [[8, 1], [2, 13]], "total": 24}}
  ],
  "model_comparison": {
    "primary_metric": "f1_score", "secondary_metric": "accuracy",
    "successful_count": 5, "failed_count": 0, "best_model_name": "Logistic Regression",
    "ranking_rule": "primary: f1_score (descending); secondary: accuracy (descending); ..."
  },
  "best_baseline_model": {"model_name": "Logistic Regression", "f1_score": 0.8761},
  "runtime_objects": {
    "best_pipeline": {"type": "Pipeline", "fitted": true, "serialized": false}
  }
}
```

**Result on the sample dataset** (`datasets/sample_classification.csv`, 120 rows
after de-duplication, 96 train / 24 test, 5-fold stratified CV):

| rank | model | accuracy | f1 | precision | recall | cv acc | cv f1 | scaling |
|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.8750 | 0.8761 | 0.8804 | 0.8750 | 0.8953 | 0.8941 | on |
| 2 | Support Vector Machine | 0.8333 | 0.8357 | 0.8497 | 0.8333 | 0.8321 | 0.8278 | on |
| 3 | K-Nearest Neighbors | 0.7917 | 0.7935 | 0.8661 | 0.7917 | 0.8005 | 0.7960 | on |
| 4 | Decision Tree | 0.7500 | 0.7536 | 0.7675 | 0.7500 | 0.7700 | 0.7702 | off |
| 5 | Random Forest | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.8532 | 0.8519 | off |

Best **baseline** model: **Logistic Regression** (F1 0.8761). Note that Random
Forest has the best cross-validated F1 (0.8519 vs 0.7702 for the tree) but a worse
hold-out score — exactly the kind of variance Phase 5 will analyse. Hyperparameter
optimization (Phase 6) may change the winner.

## Phase 5: Evaluation Agent

Phase 5 **interprets** the baseline models; it never trains anything. The agent
reads `experiment_results`, `cross_validation_results`, `model_comparison` and
`best_baseline_model` from the shared state and produces one JSON-safe
`EvaluationReport`. Re-running it on the same state is free and deterministic: no
estimator is fitted, refitted or re-scored.

**Data flow**

```
ExperimentResult + CrossValidationResult + ModelComparison   (read only)
  -> compare_models()      : one ModelEvaluation per experiment, gaps included
  -> rank_models()         : Phase 4 rule, unchanged (F1, then accuracy, then name)
  -> analyze_performance_gaps()        : close / moderate / substantial
  -> analyze_confusion_matrix()        : counts, per-class metrics, error pattern
  -> analyze_execution_time()          : total, average, fastest, slowest
  -> count_experiments()               : models evaluated vs hyperparameter trials
  -> generate_model_recommendation()   : best baseline model + why + what it is not
  -> EvaluationReport                  : observations, limitations, recommendations
```

### Hold-out vs cross-validation gap

For every evaluated model the agent computes the **signed** difference
`gap = holdout - cross_validation_mean` for F1 and for accuracy, then classifies
its magnitude with the thresholds in `utils/config.py`:

| Category | Rule (default) |
|---|---|
| `close` | `abs(gap) < 0.03` |
| `moderate` | `0.03 <= abs(gap) < 0.10` |
| `substantial` | `abs(gap) >= 0.10` |
| `not_available` | no cross-validation result or missing metric |

Both thresholds are configurable, and a model without a cross-validation result
gets `not_available` plus a recorded reason — the gap is never estimated. The
report always attaches the rule and this disclaimer verbatim:

> Reporting heuristic, not a statistical test: the hold-out/cross-validation gap is
> measured on one split of a small dataset, so its size alone neither proves nor
> disproves overfitting or instability.

A `substantial` gap is worded as *"may indicate sensitivity to the selected split
and warrants further investigation"*. The report never says a model *is*
overfitting, *is definitely* robust, or *will work perfectly* on real-world data —
and it explicitly states that Phase 4 recorded no training-set metrics, so no
train-versus-test gap can be reported from this run.

### Confusion matrix analysis

`analyze_confusion_matrix()` derives per-class precision, recall and F1 **from the
stored matrix**, which is mathematically exact (`precision = TP/(TP+FP)`,
`recall = TP/(TP+FN)`) and needs no retraining:

```python
for row_index, label in enumerate(labels):          # each label once as positive
    tp = matrix[row_index][row_index]
    support         = sum(matrix[row_index])          # true label
    predicted_total = sum(r[row_index] for r in matrix)
    fp = predicted_total - tp
    fn = support - tp
    tn = total - support - fp
```

No label is assumed to be the "interesting" one — the project holds no domain
metadata, so each label is read once as the positive class, and the report says so.
The derived values are cross-checked by the test suite: the support-weighted average
of the per-class F1 values reproduces the weighted F1 scikit-learn computed in
Phase 4 (0.8761 on the sample dataset).

* **Binary**: total / correct / incorrect counts, TN-FP-FN-TP per label, and the
  dominant direction — *"Reading the label '0' as the positive class, false
  positives (2) exceed false negatives (1)"* — with an explicit note that no
  real-world consequence is attached to that direction.
* **Multiclass**: the classes involved in the most errors (false negatives and
  false positives reported separately, because one mistake is counted as both) and
  the dominant confusion pairs. Classes with zero mistakes are never listed.

A missing matrix produces `available=False` and the recorded note
`"Per-class precision/recall not available from Phase-4 records."` instead of
invented numbers; a malformed matrix raises `EvaluationDataError`.

### Ranking, timing and counts

The evaluation table has the columns `rank, model, acc, prec, rec, f1, cv acc,
cv f1, f1 gap, acc gap, train s, total s, status`. The ranking rule is the Phase 4
rule and is asserted to produce the **same winner** as the training report, so the
two can never contradict each other.

`ExecutionTimeAnalysis` reports total and average experiment time, the fastest and
slowest successful model (ties broken by name), the training / prediction / CV
split and the number of timed experiments. Only experiments that produced metrics
are timed, and the report states that these are single-run measurements, not a
benchmark.

`ExperimentCount` separates `baseline_models_attempted`, `successful_experiments`,
`partial_experiments`, `failed_experiments` and `total_experiments` from
`hyperparameter_experiments`, which is **0** in Phase 5. The two are different
quantities — Phase 4 trains each model once — and Phase 11 needs both.

### Structured output

```json
{
  "evaluation_status": "completed",
  "evaluation_report": {
    "overall_status": "completed",
    "ranking_rule": "primary: f1_score (descending); secondary: accuracy (descending); ...",
    "best_model": {
      "best_baseline_model": "Logistic Regression",
      "selection_reason": "highest hold-out f1_score (0.8761) among 5 successful baseline experiment(s); ties are broken on accuracy",
      "f1_score": 0.8761, "accuracy": 0.875, "cv_f1_mean": 0.8941, "f1_gap": -0.018,
      "parameters": {"max_iter": 1000, "random_state": 42},
      "confusion_matrix_error_pattern": "Reading the label '0' as the positive class, ..."
    },
    "model_evaluations": [
      {"rank": 1, "model_name": "Logistic Regression", "f1_score": 0.8761,
       "cv_f1_mean": 0.8941, "f1_gap": -0.018012, "f1_gap_classification": "close",
       "evaluation_status": "completed", "total_time": 0.2159}
    ],
    "gap_analysis": {"models_with_gap": 5, "close_agreement_models": ["Logistic Regression", "..."],
                     "substantial_difference_models": ["Random Forest"],
                     "largest_absolute_gap": -0.101857, "metric": "f1", "disclaimer": "..."},
    "confusion_matrix_analysis": [
      {"model_name": "Logistic Regression", "available": true, "total_samples": 24,
       "correct_predictions": 21, "incorrect_predictions": 3,
       "class_metrics": [{"label": "0", "support": 9, "true_positives": 8,
                          "false_positives": 2, "false_negatives": 1, "true_negatives": 13,
                          "precision": 0.8, "recall": 0.888889, "f1_score": 0.842105}]}
    ],
    "execution_time_analysis": {"total_experiment_time": 2.238589, "average_experiment_time": 0.447718,
                                "fastest_model": "Decision Tree", "slowest_model": "Random Forest"},
    "experiment_count": {"baseline_models_attempted": 5, "successful_experiments": 5,
                         "failed_experiments": 0, "total_experiments": 5,
                         "hyperparameter_experiments": 0},
    "observations": ["..."], "limitations": ["..."], "recommendations": ["..."]
  }
}
```

### Result on the sample dataset

`datasets/sample_classification.csv` (5 baseline models, 24 hold-out samples):

| model | f1 gap | accuracy gap | category | hold-out correct / total |
|---|---|---|---|---|
| Logistic Regression | -0.0180 | -0.0203 | close | 21 / 24 |
| Support Vector Machine | +0.0079 | +0.0012 | close | 20 / 24 |
| K-Nearest Neighbors | -0.0025 | -0.0089 | close | 19 / 24 |
| Decision Tree | -0.0167 | -0.0200 | close | 18 / 24 |
| Random Forest | -0.1019 | -0.1032 | substantial | 18 / 24 |

Four of five models are in close agreement between the hold-out split and the
cross-validation mean. Random Forest is the outlier: its cross-validated F1 is
0.8519 while its hold-out F1 is 0.7500, which the report flags as a `substantial`
gap and as a model to re-check with a different `random_state` — stated as a
hypothesis to investigate, not as a diagnosis. On the multiclass sample dataset all
five models show close gaps, and Decision Tree wins the ranking through the
documented name tie-break at equal F1.

## Phase 6: Optimization Agent (Optuna)

Optuna is the **first new runtime dependency** of the project (`optuna>=4.0,<5.0`).
It is used only by the Optimization Agent, so the Phase 1-5 pipeline stays fully
usable with the search disabled — `python app.py --no-optimize`, or
`OptimizationAgent().run(state, cross_validation=False)` for a recorded skip.

The agent tunes the models that already won the Phase 5 ranking, compares the
tuned result against the untuned baseline for the same model, and keeps **both**
winners. It never replaces `evaluation_report.best_model`.

### Which models get tuned

The candidate set is the Phase 5 ranking, unchanged: only successful (or
partial) experiments with metrics are considered, ordered by `f1_score`
(descending), then `accuracy` (descending), then model name (ascending). The
first `OPTUNA_TOP_MODELS` (default **2**) names are tuned. Ties therefore resolve
to the same model on every run and on every machine, and a model with no search
space is skipped rather than silently attempted.

### Search spaces

One explicit, bounded space per model. Nothing is generated from a string, and
every suggested value was checked against the installed scikit-learn:

| Model | Search space |
|---|---|
| Logistic Regression | `C` log-uniform in [1e-4, 1e2]; `solver` categorical; `max_iter=2000` fixed (convergence is not tuned, which removes a search dimension) |
| Decision Tree | `max_depth` int in [1, 20]; `min_samples_split` int in [2, 20]; `min_samples_leaf` int in [1, 10]; `criterion` in gini / entropy / log_loss |
| Random Forest | `n_estimators` int in [50, 200] step 10; `max_depth` in [None, 5, 10, 20, 30]; `min_samples_split` int in [2, 20]; `min_samples_leaf` int in [1, 10]; `max_features` in sqrt / log2 / None / float in (0.0, 1.0) |
| K-Nearest Neighbors | `n_neighbors` int in [1, 30]; `weights` in uniform / distance; `metric` in minkowski / euclidean / manhattan / chebyshev |
| Support Vector Machine | `C` log-uniform in [1e-3, 1e3]; `kernel` in rbf / linear / poly / sigmoid; `gamma` log-uniform in [1e-4, 1e0] **only for non-linear kernels** (a linear kernel ignores it, so searching it would waste a dimension) |

The Logistic Regression solver list is **problem-type aware**. On scikit-learn
1.8.0 `liblinear` fails for a multiclass target, so binary runs may choose it
(`lbfgs`, `liblinear`) while multiclass runs choose from `lbfgs`, `saga`,
`newton-cg` instead. A search that kept `liblinear` would have burned its whole
trial budget on identically failing configurations.

`random_state` is deliberately **not** a search dimension. It is set to
`OPTUNA_RANDOM_SEED` and injected into every configuration that accepts one, so a
recorded `best_params` dict always rebuilds the model that produced its score.

### Objective, cross-validation and leakage prevention

* **Objective:** the mean `f1_weighted` of a `StratifiedKFold` cross-validation on
  the **training split only**, maximised. This is the same weighted-F1 metric the
  whole project ranks on, so no phase silently switches convention.
* **Same fold conventions as Phase 4:** `shuffle=True`, `random_state=42`, and
  the same small-class safeguard — a class with too few members reduces the fold
  count and the reason is recorded on the result rather than hidden. A
  cross-validation that cannot run safely is **refused before the first trial**,
  and a target with fewer than two classes is refused too, because every fold
  would score 0.0 or 1.0 by accident and the search would maximise the accident.
* **No leakage, structurally rather than by convention:** every trial builds a
  complete `Pipeline(preprocessor -> estimator)` and hands the *unfitted* pipeline
  to the cross-validator with `error_score="raise"`. Each fold therefore refits
  the imputer, encoder and scaler on its own training rows only. Fitting the
  preprocessor once before cutting the folds would make every score slightly
  optimistic, and the search would optimise that optimism.
* **The hold-out split is never used to choose hyperparameters.** It is read once
  per model, *after* the winning configuration is fixed, to produce the
  independent baseline-vs-optimized comparison.

### Configuration

| Setting | Value | Meaning |
|---|---|---|
| `OPTUNA_N_TRIALS` | `20` | Trials per model. No count is hard-coded anywhere else. |
| `OPTUNA_RANDOM_SEED` | `42` | Seeds `TPESampler` and every estimator. |
| `OPTUNA_TOP_MODELS` | `2` | How many of the baseline ranking to tune. |
| `OPTUNA_TIMEOUT` | `None` | Optional wall-clock limit per model, in seconds. |
| `OPTUNA_SAMPLER_NAME` | `tpe` | Sampler family. |
| `OPTUNA_DIRECTION` | `maximize` | The objective is a score. |
| `OPTUNA_OBJECTIVE_METRIC` | `f1_weighted` | The metric being maximised. |

Reproducibility is a fixed property, not a lucky outcome: with the same seed and
the same data, two runs produce the same suggestions, the same best
configuration, the same scores, the same trial counts and the same winner. Only
the wall-clock timings differ between runs, and the report says so.

### Baseline versus optimized

Both sides are always reported separately and labelled with the split they came
from. Improvement is `optimized_cv_f1 - baseline_cv_f1`; a percentage is added
only when the baseline denominator is safe (strictly greater than
`OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE`), otherwise the absolute difference is
reported with a note explaining why no ratio is given. The direction is labelled
`improved`, `degraded`, `unchanged` or `not_available` — a degraded result is
recorded, never hidden.

### Failure handling

| Situation | Behaviour |
|---|---|
| One trial raises | Recorded as a `FAIL` trial with its captured message; the study continues. The failed trial is excluded from the best configuration. |
| Every trial of one model fails | That model is recorded as `failed` with the reason; the next selected model is still optimized and the phase status becomes `partial`. |
| Every selected model fails | Phase status `failed`, and the Phase 5 baseline stays visible as the recommendation, because no optimized model can be recommended. |
| Cross-validation cannot run safely | Search refused before the first trial, with the recorded reason. |
| Model has no search space | Skipped with a stated reason and filled from further down the ranking. |
| Prerequisite phase did not finish | Phase aborts, logs the error, records it and re-raises. |

Optuna's verbosity is set to `ERROR` so a failed trial is not printed twice; the
reason is recorded on the `TrialRecord` and in the project log instead.

### Structured output

```json
{
  "optimization_status": "completed",
  "optimization_report": {
    "overall_status": "completed",
    "sampler": "tpe", "sampler_seed": 42,
    "objective_metric": "f1_weighted",
    "n_trials_per_model": 20, "top_models": 2, "cv_n_splits": 5,
    "models_selected_for_optimization": ["Logistic Regression", "Support Vector Machine"],
    "models_not_optimized": ["Decision Tree", "Random Forest", "K-Nearest Neighbors"],
    "optimization_results": [
      {"model_name": "Support Vector Machine", "status": "completed",
       "baseline_cv_f1_mean": 0.8278, "optimized_cv_f1_mean": 0.9046,
       "baseline_holdout_f1_score": 0.8357, "holdout_f1_score": 0.9578,
       "improvement": 0.0767, "improvement_percentage": 9.2704,
       "improvement_direction": "improved",
       "best_params": {"kernel": "linear", "C": 14.566096837677359},
       "best_trial_number": 18,
       "total_trials": 20, "completed_trials": 20, "failed_trials": 0,
       "trials": [{"trial_number": 0, "params": {...}, "objective_value": 0.86,
                   "state": "COMPLETE", "duration": 0.19, "error_message": ""}]}
    ],
    "best_optimized_model": {"best_optimized_model": "Support Vector Machine", "...": "..."},
    "best_baseline_model":  {"best_baseline_model": "Logistic Regression", "...": "..."},
    "counts": {"baseline_models_attempted": 5, "optimized_models_attempted": 2,
               "completed_trials": 40, "failed_trials": 0, "total_trials": 40,
               "note": "... a model experiment and an Optuna trial are different
                        quantities and are not added together."},
    "observations": ["..."], "limitations": ["..."], "recommendations": ["..."]
  }
}
```

`state.optimization_results` is a read-only view over
`optimization_report.optimization_results`, and is skipped during serialisation so
the results are never written twice. No Optuna or scikit-learn object is stored
in the state; `json.dumps(state.to_dict())` succeeds without a `default=` hook.

**Experiment counts and trial counts are kept apart.** A model experiment is one
fit of one model with the registry defaults; an Optuna trial is one
cross-validated evaluation of one candidate configuration. The report prints both
and states explicitly that they are not added together.

### Ranking of the tuned models

The tuned models are ranked with the project's existing rule — hold-out
`f1_score`, then hold-out `accuracy`, then model name — so the phase introduces
no new convention. This is recorded as a limitation: the ranking reads the
hold-out scores, so the winner's hold-out numbers are no longer an untouched
estimate of generalisation. **The hyperparameters themselves were chosen on the
training split only**; it is the comparison between models, not the choice of
parameters, that touches the hold-out split.

### Result on the sample dataset

`datasets/sample_classification.csv` (5 baseline models, 24 hold-out samples,
20 trials per model, seed 42):

| model | baseline cv f1 | optimized cv f1 | improvement | baseline holdout f1 | optimized holdout f1 | trials |
|---|---|---|---|---|---|---|
| Support Vector Machine | 0.8278 | 0.9046 | +0.0767 (+9.27%) | 0.8357 | 0.9578 | 20 |
| Logistic Regression | 0.8941 | 0.9048 | +0.0106 (+1.19%) | 0.8761 | 0.9167 | 20 |

Logistic Regression remains the best **baseline** model (Phase 5, unchanged);
Support Vector Machine is the best **optimized** model. Both are kept, because the
optimized score is the value the search maximized and the baseline score was not,
so neither number is directly comparable to the other.

On `datasets/sample_multiclass.csv` the candidates are Decision Tree and Random
Forest (the Phase 5 winners). Both reach a cross-validated F1 of 1.0000 with their
default parameters already, so every trial scores the same and the improvement is
exactly `0.0000` with the direction `unchanged`. That is a property of this small
synthetic sample fixture, not evidence that the models generalise; the report says
so in its limitations ("The datasets are small synthetic samples with no domain
meaning attached to the labels, so nothing in this run shows how the models would
behave on real data"). A no-improvement outcome is reported as measured rather
than suppressed.

### What this phase does not claim

A higher cross-validation score after tuning means a better configuration was
found *for this search budget*. It does not guarantee better performance on
unseen data, and it does not remove the risk of overfitting the cross-validation
folds. The reported limitations are generated from what actually happened in the
run — trial budget, single seed, CV selection bias, hold-out ranking, untuned
models, single split, single-run timings — not from a fixed disclaimer block.

## Phase 7: LangGraph Orchestration

### Why an orchestration layer

Phases 2-6 produced six agents that work. What they did not have was a single
place that decides **what runs next**, **what happens when something fails**, and
**whether the expensive step is worth running**. Until now that logic lived in
`app.py` as a straight sequence of `try` blocks. That is fine for one path and
bad the moment you want to ask questions like:

* if the Preprocessing Agent fails, is it safe to let the Training Agent run?
* how many times should a node be retried, and which errors are worth retrying?
* should the Optuna search run at all for this dataset?
* what is the status of the run as a whole, not just per agent?

Phase 7 adds that missing layer as a [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph`. LangGraph was chosen because it makes control flow *data*: routing
is a function of the state and returns the next node, so the run order is
inspectable and testable rather than implied by indentation.

**What was deliberately not done:** no agent logic was moved, rewritten or
duplicated. Preprocessing still lives in `agents/preprocessing_agent.py`, the
Optuna search still lives in `agents/optimization_agent.py`, and each agent is
still usable on its own exactly as before. The graph calls the same six objects
the sequential path calls, in the same order, with the same arguments.

### The six nodes and the agents they invoke

| Graph node | Agent invoked | Node responsibility |
|---|---|---|
| `data_analysis` | `DataAnalysisAgent` | Load the CSV and summarise it |
| `preprocessing` | `PreprocessingAgent` | Build the preprocessing pipeline |
| `model_selection` | `ModelSelectionAgent` | Choose the candidate models |
| `training` | `TrainingAgent` | Fit each candidate, optionally with CV |
| `evaluation` | `EvaluationAgent` | Score the baselines and rank them |
| `optimization` | `OptimizationAgent` | Run the Optuna search |

Each node in `graph/nodes.py` does exactly three things: call one agent, record
whether it worked, and return a partial state update. Every node delegates to
the same shared runner, `_execute_node`, so the retry policy and the logging
format cannot drift between nodes.

### Graph structure

```
START
  |
  v
Data Analysis --+
  |             |
  v             |
Preprocessing --+
  |             |
  v             |
Model Selection --+        every required node ends the workflow when it
  |             |         fails: the router returns END instead of the
  v             |         next node, so no downstream agent is ever
Training -------+         entered with incomplete state
  |             |
  v             |
Evaluation -----+---> [ Should Optimize? ] --no--> END
  |                        |
  |                       yes
  v                        |
Optimization <-------------+
  |
  v
 END
```

The compiled graph has 12 edges:

* `START -> data_analysis` (direct);
* `data_analysis -> preprocessing -> model_selection -> training -> evaluation
  -> optimization` (conditional, so a failure can divert to `END`);
* one conditional edge to `END` from each of the six nodes.

The extra `END` edges are what make failure handling visible in the structure
rather than hidden inside a node. `describe_structure()` prints the node and
edge lists as plain text for debugging and for the tests, so no drawing library
is needed.

### Conditional routing after Evaluation

The `Should Optimize?` decision is a pure function of the graph state in
`graph/routing.py`. It is checked in this fixed order, and the first match wins:

| Order | Condition | Decision | Route |
|---|---|---|---|
| 1 | Optimization disabled for this run | `skip_disabled` | `END` |
| 2 | Evaluation did not complete | `skip_evaluation_incomplete` | `END` |
| 3 | An optimization run already happened | `skip_already_done` | `END` |
| 4 | No successful experiments to tune | `skip_no_candidates` | `END` |
| 5 | Otherwise | `run` | `optimization` |

Two details are worth stating explicitly:

* **The decision is recorded, not recomputed.** The evaluation node writes the
  decision into the state *before* the routing edge is taken. Recomputing it
  from the final state would be wrong: once the optimization node has run, rule 3
  matches, and the report would claim the search was skipped when it actually
  ran.
* **Candidates are counted from the experiments, not from a status flag.**
  `training_status` is `completed` even when every single experiment failed, so
  rule 4 counts `AgentState.successful_experiments` instead.

### Why Optuna never runs twice

Rules 3 and 4 together make a second search impossible, and the edge structure
closes the loop: no edge points backwards, and the optimization node's only
outgoing edge is `END`. `LANGGRAPH_MAX_STEPS` is additionally passed to
LangGraph as a hard step ceiling, so even a routing bug could not produce an
unbounded run. There is no cycle in the graph to return to.

### Failure handling

Every required node failure ends the workflow. Two kinds of failure are
recognised, because the agents report them differently:

1. **The agent raised.** `DatasetNotFoundError`, `AnalysisNotReadyError`,
   `PreprocessingError`, `ModelToolError`, `TrainingToolError`,
   `EvaluationToolError` and friends are raised when a prerequisite is unmet.
   The node records a `GraphError`, requests a stop, and the router sends the
   run to `END`.
2. **The agent recorded a failure and returned.** The Evaluation Agent does
   this: it sets `evaluation_status = "failed"` and returns normally. The node
   itself completed, so the workflow status is derived from the agent status as
   well as the node status, and the run is reported as `failed` rather than
   being quietly treated as a success.

In both cases the machine learning state computed so far is preserved and
returned to the caller. Nothing already computed is discarded, and nothing is
silently swallowed: every failure becomes a `GraphError` record with the node,
the exception type, the message and the attempt number.

**The optimization node is the exception.** If optimization fails, the Phase 5
evaluation result is a valid, complete answer, so the workflow does **not** stop.
The run finishes with status `partial`: the baseline model and its metrics are
still reported, and the failed search is reported as an error.

### Retry policy

`LANGGRAPH_MAX_RETRIES = 1`: a node may be retried **once**, and never more.
There is no unbounded retry loop, and the retry ceiling is enforced per node
rather than per workflow.

Retryable errors are classified by *cause*, not by type name alone:

| Error source | Retryable | Reason |
|---|---|---|
| The project's own taxonomy (`tools/*.py`) | No | Deterministic: a missing file or an unmet prerequisite will fail again identically |
| Anything unexpected (`RuntimeError`, `OSError`, ...) | Yes, once | Possibly transient |

A node that fails and then succeeds on its retry is reported as **completed**,
with `retries = 1` and the failed attempt kept in the error list. A node that
keeps failing stops the workflow after the retry, with `attempts = 2`.

LangGraph's built-in retry policy was rejected for this: it cannot classify an
error, it does not record the retry count in the state, and it would retry a
validation error that can never succeed. The policy is implemented explicitly in
`_execute_node` so it is visible and testable.

### Graph versus the sequential path

Both paths call the same six agents and produce the same machine learning
results. They differ only in what the run reports and how it reacts to failure.

| | Sequential (`app.py`) | Graph (`app.py --graph`) |
|---|---|---|
| Agent invocation | Direct calls in `main()` | Invoked by graph nodes |
| Node order | Fixed by the code | Defined by graph edges, inspectable via `describe_structure()` |
| On failure | Caught per agent, run continues or stops by hand | Recorded as a `GraphError`, router sends the run to `END` |
| Retry | None | At most one, and only for unexpected errors |
| Optimization | `--no-optimize` flag | `--no-optimize` **and** a state-driven `Should Optimize?` decision |
| Result | Per-phase printed reports | Per-phase reports **plus** a `WorkflowResult`: status, nodes executed, failed node, retry count, decision, final error |

Both paths remain available. The sequential path is the default and is
unchanged, so nothing that worked in Phase 6 stops working.

### `WorkflowResult`

`run_workflow()` returns a frozen dataclass, not a nested dict:

```python
from graph.workflow import run_workflow

result = run_workflow(dataset_path="datasets/sample_classification.csv")

result.workflow_status        # WorkflowStatus.COMPLETED | PARTIAL | FAILED
result.executed_nodes         # ['data_analysis', ..., 'optimization']
result.failed_node            # the node that failed, or None
result.total_retries          # retries actually used
result.optimization_decision  # OptimizationDecision.RUN | SKIP_DISABLED | ...
result.optimization_outcome   # OptimizationOutcome.COMPLETED | NOT_RUN | ...
result.errors                 # [GraphError(node, error_type, message, attempt, retryable)]
result.final_agent_state      # the same AgentState the sequential path would return
result.to_dict()              # JSON-safe, no custom encoder needed
```

Derived views (`record_for`, `completed_nodes`, `failed_nodes`, `last_error`) are
provided so a caller never has to filter the records by hand.

### Command line usage

```
python app.py --dataset datasets/sample_classification.csv --graph
python app.py --dataset datasets/sample_multiclass.csv --graph --no-optimize
python app.py --dataset path/to/your.csv --target your_target --graph
python app.py --graph --json
```

Without `--graph` the sequential path runs exactly as before. `--no-optimize`
works on both paths; on the graph path it is the first rule checked by
`Should Optimize?`.

The log stays one line per event:

```
[GRAPH] workflow -> STARTED (optimization=enabled, retries=1)
[GRAPH] Data Analysis -> STARTED
[GRAPH] Data Analysis -> COMPLETED (0.0169 s)
...
[GRAPH] Optimization -> STARTED
[GRAPH] Optimization -> COMPLETED (41.2261 s)
[GRAPH] workflow -> FINISHED status=completed
```

A retry is reported in place, on one line:

```
[GRAPH] Training -> FAILED (RuntimeError on attempt 1)
[GRAPH] Training -> RETRY 1/1
[GRAPH] Training -> FAILED (giving up after 2 attempt(s): RuntimeError)
```

### No language model is involved

This phase uses LangGraph as an orchestration library only. There is no prompt,
no model call, no token budget and no network request anywhere in `graph/`. The
routing functions are ordinary Python that read the state and return a node name,
which is why they are directly unit-testable.

The dependency list is deliberately narrow: `langgraph` only. It pulls in
`langchain-core` transitively, because `langgraph-prebuilt` requires it; the
`langchain` agent framework itself is not installed or used. `utils/config.py`
publishes the forbidden list in `LANGGRAPH_FORBIDDEN_LLM_DEPENDENCIES`, and the
test suite asserts that none of those modules is imported and that no file in
`graph/` references one. The core pipeline has always worked without an LLM, and
Phase 7 does not change that.

## Phase 8: Optional Local LLM Layer (Ollama)

**The core ML pipeline does not require an LLM.** Phase 8 adds an *optional*
explanation and planning layer that runs entirely on your own machine through
[Ollama](https://ollama.com/download). Every measured result in this project is
still produced by the deterministic pipeline. The language model only reads a
summarised context and produces text; it is never allowed to change a metric, a
model, a ranking, a dataset, a search space or a routing decision.

```bash
# The Phase 1-7 behaviour, unchanged
python app.py --graph

# The same run plus generated explanations, when a local model is available
python app.py --graph --llm
```

The second command exits 0 whether or not the explanations were produced. If
Ollama is not running, the run still completes and a short notice says so, per
step. Silent degradation is the failure mode this phase is designed to avoid.

### What it can and cannot do

| It can | It cannot |
|---|---|
| Summarise what a step measured | Invent a metric that was not measured |
| Separate observations from interpretations from hypotheses | Claim to have run an experiment |
| Propose what to investigate next | Retrain, reconfigure or replace a model |
| Suggest which candidates are worth tuning | Change an Optuna search space or a decision |
| Be completely absent | Fail a run, set `stop_requested` or route the graph |

The rules the model is given are explicit: the recorded metrics are
authoritative, no value may be invented, no experiment may be claimed, nothing is
retrained, and the dataset is never modified. Observations, interpretations and
hypotheses are returned in separate fields so one cannot be presented as another
by accident. Phrasing that would overstate certainty (`"proves"`,
`"guarantees"`, `"production ready"`) is **detected and reported**, never
silently rewritten, so an overclaim cannot pass as a validated claim.

### Configuration

Everything is in `utils/config.py`, and every value is off by default.

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_ENABLED` | `False` | The layer is inert unless asked for |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Must be a loopback address; override with the `OLLAMA_BASE_URL` env var |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Override with the `OLLAMA_MODEL` env var |
| `OLLAMA_TIMEOUT` | `30` | Seconds per request, override with `OLLAMA_TIMEOUT` |
| `OLLAMA_MAX_RETRIES` | `1` | No retry storm from an optional layer |
| `OLLAMA_TEMPERATURE` | `0` | Reproducible text, override with `OLLAMA_TEMPERATURE` |

There is no API key setting, because there is nothing to authenticate against a
local server. `llm/client.py` rejects a non-loopback base URL, so the layer
cannot be repointed at a hosted endpoint even by accident.

To use it:

```bash
# 1. Install Ollama separately, then start it
ollama serve

# 2. Pull a small model yourself (the project never downloads anything)
ollama pull qwen2.5:3b

# 3. Run the workflow with explanations
python app.py --graph --llm
```

A GPU is not required, and a small model is the default so it runs on modest
hardware. If the model is missing, the project says so and names the `ollama
pull` command instead of fetching anything itself.

### Architecture

```
graph/llm_nodes.py  (optional nodes, only with --llm)
        |
        v
llm/service.py      decides whether to call, and owns every failure
   |          |
   v          v
llm/prompts.py  llm/client.py  -->  Ollama on localhost
                    |
                    v
              llm/parser.py     --> validated llm/state.py records
```

- `llm/client.py` is the **only** module that speaks HTTP, using the standard
  library's `urllib.request`. Phase 8 therefore adds **no dependency** to
  `requirements.txt`, and no Ollama SDK is imported anywhere.
- `llm/prompts.py` holds every instruction and builds the context through an
  allowlist. Only summarised values are sent: never raw rows, file contents,
  credentials, environment variables or logs.
- `llm/parser.py` extracts the JSON object, validates it against the expected
  fields, and makes **one** controlled repair attempt before giving up. An
  invalid answer becomes a recorded status, never an exception.
- `llm/service.py` runs a health check before assembling a prompt, and **no
  method in it raises**. Every failure becomes a typed `LLMResult`.
- `llm/state.py` defines the typed records (`LLMStatus`, `LLMResult`,
  `LLMInterpretation`, `LLMRecommendation`). The `llm/` package never imports
  `graph/`, so the adapter is independent of the orchestration layer.
- `graph/llm_nodes.py` adds three optional nodes *after* the step they explain.
  They never mutate `AgentState`, never set `stop_requested` and never route.

### Five outcomes, always distinguished

`LLMStatus` reports which of these happened, rather than collapsing them into a
boolean:

| Status | Meaning |
|---|---|
| `DISABLED` | Not requested; the default |
| `AVAILABLE` | Health check passed: server up, model present |
| `UNAVAILABLE` | Ollama could not be reached |
| `MODEL_UNAVAILABLE` | Server up, the configured model is not installed |
| `REQUEST_FAILED` | Timeout, HTTP error or connection failure |
| `INVALID_RESPONSE` | The answer was not usable JSON in the expected shape |
| `COMPLETED` | A request was made and a valid response was parsed |

### Determinism is preserved

`python app.py --graph` compiles exactly the Phase 7 graph: the same twelve edges
and the same eight required nodes. With `--llm` three optional nodes are added,
which adds six edges and interleaves them into the run, but:

- the **required** node sequence and every machine learning number are identical;
- `route_after_llm_evaluation` reads the decision the Evaluation node already
  recorded, never the generated text;
- `--llm` on its own prints a note and runs the sequential path unchanged.

The test suite asserts this by running the same workflow both ways and comparing
the results, and by stripping every LLM key from a state and re-deciding.

### What this phase does not claim

- The generated text is not verified. Checking it against the measurements would
  require re-running the pipeline, which a language model cannot do.
- A small local model reasons well below a hosted model. Read the output as a
  draft for review; when it disagrees with a measured value, the measurement is
  correct and the text is wrong.
- An optimisation plan is a priority hint. Optuna searched its own pre-defined
  spaces, and a suggestion that was not followed changed no result.
- Only the configured base URL is reachable, and only a summarised context is
  sent. Nothing is transmitted to any external service.

## Phase 9: Model Persistence and Streamlit Dashboard

### What this phase adds

Two things, and neither is required to train a model:

1. a **deployable artifact** written to disk at the end of a workflow run, and
2. a **read-only dashboard** that shows what that run recorded.

The core ML pipeline is unchanged. Phases 2 to 8 behave exactly as before.

### Model persistence

The fitted pipeline is saved with `joblib` after the model has been selected. It
is the whole `sklearn.pipeline.Pipeline` - preprocessor and estimator together -
so a consumer never has to re-impute, re-encode or re-scale anything.

```python
from pathlib import Path
from tools.persistence_tools import load_persisted_model

model = load_persisted_model(Path("models"))       # None when nothing is saved
model.pipeline.predict(row)                        # one call, preprocessing included
model.metadata["model_name"]                       # what was saved, and from where
```

### Where the artifacts go

Three files, all in `models/` by default (`config.MODELS_DIR`):

| File | Contents |
|---|---|
| `models/best_model.joblib` | The fitted `Pipeline`, via `joblib` |
| `models/best_model_metadata.json` | Model name and type, selection source, target column, feature columns, task type, metrics, cross-validation metrics, best parameters, `final_fit` record, library versions, limitations |
| `models/run_results.json` | The full run record the dashboard reads: dataset analysis, evaluation, optimization, baseline and optimized model tables, the final selection, and any LLM text |

`joblib.load` is called from **one** place in the whole project,
`tools/persistence_tools.py`, and only for a path derived from configuration. A
test asserts this by parsing every source file, so it cannot be bypassed by
accident.

Writes are atomic: a temporary file in the destination directory, `fsync`, then
`os.replace`. The model is written **before** the metadata, so a metadata file
can never describe a model that is not there yet.

### The selection policy

Persistence does not invent a new ranking. It reuses the project's existing one -
hold-out `f1_score` descending, then hold-out `accuracy` descending, then model
name ascending, which is the key `rank_experiments` and `rank_optimized_results`
already use:

- the best tuned model wins if it exists **and** carries a hold-out `f1_score`;
- otherwise the best valid baseline is persisted;
- if neither is usable, the run records a failure and writes nothing. It does not
  invent a model.

Metric provenance never mixes. A baseline selection reports Phase 4's hold-out
metrics (`metrics_source: "evaluation_holdout"`) together with its stored
confusion matrix. An optimized selection reports Phase 6's hold-out score
(`metrics_source: "optimization_holdout"`) with precision, recall and the
confusion matrix left as `null` plus an explanatory note, because Phase 6 does
not record them and re-scoring a tuned model would be a second evaluation.

### The final fit happens after selection

Fitting the deployable artifact is a separate step that runs **only after** the
model has already been chosen. It changes no selection, no optimization result
and no metric:

- **baseline selection** - the pipeline the Training Agent already fitted is
  saved unchanged (`reused_fitted_pipeline`). Nothing is refitted.
- **optimized selection** - the estimator is rebuilt from the recorded best
  parameters and the Phase 3 preprocessing definition, then refitted on the
  **training split only** (`rebuilt_and_refitted_pipeline`). The hold-out rows
  are never used to fit.

### Running the workflow before the dashboard

```bash
# Write the artifacts, then open the dashboard
python app.py --graph --dataset datasets/sample_classification.csv
streamlit run dashboard.py

# Or skip the hyperparameter search: the best baseline is still persisted
python app.py --graph --no-optimize --dataset datasets/sample_classification.csv

# Or skip persistence entirely
python app.py --graph --no-save-model
```

Persistence is **on by default** for `--graph`. `--no-save-model` turns it off.

> The sequential path (no `--graph`) still stops after Evaluation, as it always
> has, and therefore writes nothing. This is pre-existing behaviour and is not
> changed by Phase 9; use `--graph` to persist a model.

### The dashboard

```bash
streamlit run dashboard.py
```

Ten sections: header, system status, dataset overview, model comparison,
evaluation, confusion matrix, optimization, LLM interpretation, prediction and
model information.

### Why the dashboard does not train

- The numbers come from `run_results.json`, which the workflow already wrote.
- The prediction calls the **saved** pipeline. There is no second copy of the
  imputation, encoding or scaling logic anywhere in `dashboard.py`.
- Streamlit re-runs the script on every interaction. If the dashboard fitted
  anything, every widget click would retrain a model.
- A rerun of the page can therefore never change a result.

`tools/dashboard_tools.py` holds all of the logic and imports neither Streamlit
nor an agent; `dashboard.py` is presentation only. That split is what makes the
logic testable without a browser and keeps the dashboard from being able to start
a workflow.

### When there is no artifact

The dashboard still starts. It shows *"No persisted model is available yet."*
plus the command to produce one, and every other section says what it has instead
of failing or showing a blank panel.

### When optimization fails

The search failing is not a reason to lose the run. The Phase 4 evaluation result
is preserved, the best valid baseline is persisted, and the metadata records
`selection_source: "baseline"` with `optimization_used: false`. In the graph, a
failed or skipped search routes to the persistence node rather than ending the
workflow; if persistence itself fails, the workflow status is `PARTIAL` rather
than `FAILED`, because the machine learning result is still valid.

### Why uploading a `.joblib` is not supported

Loading a pickle **executes arbitrary code** inside the Python process. A local
dashboard cannot make that safe, so there is no file uploader anywhere in the
project, and a test parses every source file to confirm it. The dashboard loads
exactly one thing: the artifact this project wrote, from a path it derives from
its own configuration.

## Phase 10: Local MLflow Experiment Tracking

**MLflow records what the pipeline already decided; it never decides anything.**
Phase 10 adds a local observability layer: every `--graph` run is recorded in a
local MLflow tracking store, and the Phase 9 dashboard gains an *Experiment
Tracking* section that reads those runs. No model, metric, ranking, dataset,
search space, or routing decision is changed by its presence, and nothing about
a run's result depends on whether it was recorded.

The store is a SQLite database at `experiments/tracking/mlflow.db`, created by a
run, never read to make a decision. `mlflow` is in `requirements.txt` pinned to
`mlflow>=3.0.0,<4.0.0`.

## Phase 11: Conventional ML vs Autonomous Workflow Benchmark

Phase 11 answers, with evidence instead of opinion, whether the autonomous
multi-agent workflow beats a careful conventional (manual) ML pass on the same
data. It runs both sides on the same committed datasets, the same split, the
same preprocessing defaults, the same five model families and the same weighted
metrics, then writes the comparison to JSON, CSV and three charts and shows it
in a read-only dashboard section.

- The **conventional** side is a fixed manual **21-config grid** over the shared
  model registry, executed with tools only — no agents, no LangGraph, no LLM,
  no Optuna (`benchmarks/conventional_workflow.py`).
- The **autonomous** side is the *existing* `run_workflow` graph (LLM off,
  Optuna on, persistence to a scratch directory, tracking off) — no new agent
  workflow file (`benchmarks/benchmark_runner.py`).
- Timing is `time.perf_counter()` around the machine-learning work of each
  workflow; one experiment = one unique model configuration.
- Benchmark MLflow recording goes to the dedicated store
  `experiments/tracking_benchmark/mlflow.db` (`--no-mlflow` disables it) — never
  the canonical `experiments/tracking/` store.

```bash
python app.py --benchmark                     # both committed datasets
python app.py --benchmark --dataset classification   # one dataset
python app.py --benchmark --all --no-mlflow   # both, without recording
```

The dashboard's section 12, *Conventional vs Autonomous*, reads only
`reports/phase11_comparison.json` and never triggers a run; when the file is
absent it shows the "No Phase 11 benchmark results are available yet." message.

### The verdict is honest by construction

The comparison reports `delta_f1 = autonomous − conventional` and its verdict
states whichever side scored higher. On the committed fixtures both datasets
were matched exactly (F1 delta +0.0000 both times): the same model family won on
each side, while the manual protocol needed fewer experiments and, on
multiclass, about a third of the autonomous runtime. Those numbers are data,
reported as measured.

### What is recorded

- **Dataset metadata, not the data.** The dataset name, task type, target
  column (and how it was detected), row/column counts, missing-value and
  duplicate counts, used-feature counts, and the SHA-256 digest of the file.
  The raw rows are never stored in MLflow.
- **What each node did.** Every node's status, the final `WorkflowStatus`, the
  list of executed nodes, per-node timings, and the Optuna summary (best
  trial, objective value, sampler, total trials).
- **How the model was chosen.** The `FinalModelSelection` (candidate name,
  metrics, selection source, ranking rule) exactly as the persistence stage
  decided it, plus a reference to the `PersistenceReport`.
- **Evaluation.** The hold-out `f1_score`, `accuracy`, `precision`, `recall`
  and cross-validation mean, and the confusion matrix logged as a JSON file.
- **Run status mirrors workflow status.** A `completed`, `partial`, or
  `failed` workflow is recorded as MLflow `FINISHED` or `FAILED` accordingly;
  the exact `workflow_status` value is kept as a tag.

`models/best_model.joblib` remains the primary artifact. MLflow stores a run
reference and the recorded decisions, never a competing model copy.

### The governing invariant

MLflow never influences the workflow. The tracking wrapper opens a run *before*
the graph executes, records the finished state *after* it, and hands back the
same `WorkflowResult` it received. If MLflow raises for any reason, the error is
isolated: a warning is logged, the run is left unrecorded, and the workflow
completes exactly as it would have. The test suite drives a genuinely failing
store and asserts the pipeline still finishes with identical results.

### When a run is recorded

```bash
pip install "mlflow>=3.0.0,<4.0.0"    # add requirements.txt, or already there

python app.py --graph                 # run and record it
python app.py --graph --no-mlflow     # run, record nothing
python app.py --graph --no-optimize   # a baseline-only run, still recorded
```

Recording is the **caller's request *and* the configuration**: `--no-mlflow`
and `MLFLOW_ENABLED=false` are independent switches, and either one alone turns
tracking off for that run. `MLFLOW_ENABLED` is read from the environment or
`.env` (see `.env.example`). The sequential (non-graph) path predates the graph
and does not record runs.

### The dashboard section

The dashboard never retrains a model. Section 11, *Experiment Tracking*, is
built by `tools.dashboard_tools.build_experiment_section`, which reads the store
and renders the latest run (model, F1, accuracy, dataset, status, timings) plus
a compact history table. "Latest" is the newest recorded run, so a user who runs
`python app.py --graph` again immediately sees that run in the dashboard,
without a retrain. The store is opened read-only, the dashboard accepts no
MLflow URI input, and the store path comes only from `utils/config.py`.

### Reproducibility guard

A run records the MLflow version that wrote it and the mode used, so a future
MLflow release cannot silently change what a recorded run means. The test suite
also isolates itself from the real store: tests that are not about tracking pass
`enable_tracking=False`, and a session-wide artifact guard hash-checks
`experiments/tracking/` exactly as it does `models/`, failing the session if a
test ever writes to the user's store.

## Technology Stack

| Category | Technology | Purpose | Status |
|---|---|---|---|
| Programming Language | Python 3.11+ | Core implementation | In use (3.12) |
| Data Processing | pandas, numpy | Data manipulation and numerical operations | In use |
| Machine Learning | scikit-learn | Preprocessing pipeline (Phase 3), models, training and cross-validation (Phase 4) | In use |
| Evaluation | scikit-learn metrics + pure functions | Baseline evaluation and interpretation (Phase 5), no extra dependency | In use |
| Hyperparameter Optimization | Optuna | Model tuning (Phase 6) | In use (4.9.0) |
| Orchestration | LangGraph | Agent coordination (Phase 7) | In use (1.2.12) |
| Local LLM | Ollama via `urllib.request` (optional) | Explanations and planning hints (Phase 8) | In use, **no dependency added**, never required for the core |
| UI | Streamlit | Dashboard (Phase 9) | In use (1.65.0), never required for the core |
| Visualization | None beyond Streamlit's built-in charts | Bar chart, confusion matrix chart | **No dependency added**: `st.bar_chart` covers both |
| Experiment Tracking | MLflow | Track every `--graph` run locally (SQLite), pure observability, never influences a decision (Phase 10) | In use (3.16.1), never required for the core |
| Benchmark charts | matplotlib (Agg, imported only by `benchmarks/benchmark_runner.py`) | Phase 11 comparison charts (F1 / runtime / experiments) | In use, optional (charts are best-effort) |
| Model Persistence | joblib | Save/load the fitted pipeline (Phase 9) | In use (1.5.3) |
| Version Control | Git, GitHub | Development and collaboration | In use |
| Testing | unittest (standard library), run through pytest | Test suite | In use |

**Note:** The pipeline is deterministic and does **not** depend on an LLM being available.

## Development Phases

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Project initialization & architecture | **Done** |
| **Phase 2** | Dataset Analysis Agent | **Done** |
| **Phase 3** | Preprocessing Agent | **Done** |
| **Phase 4** | Model Selection & Training Agents | **Done** |
| **Phase 5** | Evaluation Agent | **Done** |
| **Phase 6** | Hyperparameter Optimization (Optuna) | **Done** |
| **Phase 7** | LangGraph-based Multi-Agent Orchestration | **Done** |
| **Phase 8** | Optional Local LLM Integration (Ollama) | **Done** (optional layer, no new dependency) |
| **Phase 9** | Model Persistence and Streamlit Dashboard | **Done** |
| **Phase 10** | MLflow Experiment Tracking | **Done** |
| **Phase 11** | Conventional ML vs Autonomous Multi-Agent Comparison | **Done** |
| **Phase 12** | Final Research Documentation and Project Packaging | **Done** |

## Project Structure

```
autonomous_multi_agent_ml_system/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py              # shared agent behaviour (Phase 1)
│   ├── data_analysis_agent.py     # Phase 2
│   ├── preprocessing_agent.py     # Phase 3
│   ├── model_selection_agent.py   # Phase 4
│   ├── training_agent.py          # Phase 4
│   ├── evaluation_agent.py        # Phase 5
│   └── optimization_agent.py      # Phase 6 Optuna search orchestration
├── graph/                            # LangGraph orchestration (Phase 7)
│   ├── __init__.py
│   ├── state.py                      # GraphState, records, WorkflowResult
│   ├── nodes.py                      # one node per agent, retry, logging
│   ├── llm_nodes.py                  # Phase 8 optional explanation nodes
│   ├── routing.py                    # routing + optimization decision
│   └── workflow.py                   # graph build, run, report
├── benchmarks/                       # Phase 11 controlled conventional-vs-autonomous benchmark
│   ├── __init__.py
│   ├── benchmark_config.py           # pinned dataset hashes, 21-config grid, benchmark MLflow store
│   ├── conventional_workflow.py      # tools-only manual grid side (no agents/LLM/Optuna)
│   └── benchmark_runner.py           # side-by-side runner, deltas, MLflow recording, JSON/CSV/PNG writers
├── llm/                              # optional local LLM layer (Phase 8)
│   ├── __init__.py
│   ├── state.py                      # typed LLMStatus/LLMResult records
│   ├── prompts.py                    # instruction text + allowlisted context
│   ├── client.py                     # the only module that speaks HTTP
│   ├── parser.py                     # JSON extraction, schema validation
│   └── service.py                    # health check, one repair, never raises
├── tools/
│   ├── __init__.py
│   ├── data_tools.py              # Phase 2 dataset analysis functions
│   ├── preprocessing_tools.py     # Phase 3 preprocessing functions
│   ├── model_tools.py             # Phase 4 controlled model registry
│   ├── training_tools.py          # Phase 4 split, pipeline, CV, metrics
│   ├── evaluation_tools.py        # Phase 5 gap, confusion matrix, timing, counts
│   └── optimization_tools.py      # Phase 6 search spaces, studies, trials, report
├── utils/
│   ├── __init__.py
│   ├── config.py                  # paths + tunable thresholds
│   ├── logger.py                  # logging setup
│   └── state.py                   # typed shared state and data contracts
├── tests/
│   ├── __init__.py
│   ├── helpers.py                 # fixtures and expected values
│   ├── test_data_tools.py         # tool level tests
│   ├── test_data_analysis_agent.py# agent level tests
│   ├── test_preprocessing_tools.py   # Phase 3 tool level tests
│   ├── test_preprocessing_agent.py   # Phase 3 agent level tests
│   ├── test_model_tools.py           # Phase 4 registry tests
│   ├── test_training_tools.py        # Phase 4 split/pipeline/CV/metrics tests
│   ├── test_model_selection_agent.py # Phase 4 model selection agent tests
│   ├── test_training_agent.py        # Phase 4 training agent tests
│   ├── test_evaluation_tools.py      # Phase 5 gap/confusion-matrix/timing tools
│   ├── test_evaluation_agent.py      # Phase 5 evaluation agent + integration
│   ├── test_optimization_tools.py    # Phase 6 search spaces, trials, comparison
│   └── test_optimization_agent.py    # Phase 6 optimization agent + integration
│   ├── test_graph_state.py          # Phase 7 graph state and records
│   ├── test_graph_nodes.py          # Phase 7 nodes, failure, retry
│   ├── test_graph_routing.py        # Phase 7 conditional routing
│   ├── test_workflow.py             # Phase 7 compiled graph end to end
│   ├── test_llm_client.py           # Phase 8 health checks, transport, timeout
│   ├── test_llm_prompts.py          # Phase 8 instructions, allowlisted context
│   ├── test_llm_parser.py           # Phase 8 JSON extraction and validation
│   ├── test_llm_service.py          # Phase 8 orchestration and fallback
│   ├── test_llm_integration.py      # Phase 8 end-to-end, no server required
│   ├── test_conventional_workflow.py # Phase 11 manual grid side: hashes, 21-config grid, canonical numbers
│   └── test_benchmark_comparison.py  # Phase 11 runner: deltas, MLflow, artifacts, dashboard, CLI
├── datasets/
│   ├── sample_classification.csv  # synthetic binary target (development only)
│   └── sample_multiclass.csv      # synthetic 3-class target
├── models/
├── reports/
├── experiments/
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.11 or newer (developed and tested on 3.12)
- Git

### Installation
```bash
git clone <repository-url>
cd autonomous_multi_agent_ml_system

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Running the pipeline
```bash
# Full run on the bundled binary sample dataset (analysis -> preprocessing ->
# model selection -> training -> evaluation)
python app.py

# Analyse another CSV dataset
python app.py --dataset datasets/my_data.csv

# Provide the target column explicitly
python app.py --dataset datasets/my_data.csv --target outcome

# Also print the structured summary as JSON
python app.py --dataset datasets/sample_multiclass.csv --json

# Run Phase 2 only, without the Preprocessing Agent
python app.py --no-preprocess

# Stop after the Model Selection Agent (no training)
python app.py --no-train

# Skip baseline cross-validation (hold-out metrics only)
python app.py --no-cross-validation

# Stop after the Training Agent (no evaluation report)
python app.py --no-evaluate

# Stop after the Evaluation Agent (skip the Optuna search)

# Run the same six agents through the LangGraph orchestration layer
# (explicit routing, failure handling, one bounded retry and the
#  Should Optimize? decision) instead of the direct sequential calls
python app.py --graph

# Orchestration without the Optuna search: the decision is recorded
# as skip_disabled and the workflow ends after the Evaluation Agent
python app.py --graph --no-optimize
python app.py --no-optimize

# Phase 8: same run, plus the optional local LLM explanations (Ollama).
# The run completes and exits 0 whether or not Ollama is available.
python app.py --graph --llm
```

The command prints six reports: the data analysis report (columns and types,
feature groups, missing values, duplicates, target detection with its confidence
and reasoning, problem type, class distribution, statistics, data quality issues
and the agent messages), the preprocessing report (status, selected features,
excluded features and reasons, strategies, duplicate strategy, pipeline
architecture), the model selection report (selected models, scaling plan,
per-model reasons, baseline parameters), the training report (split, CV
configuration, per-model metrics and timings, confusion matrices, the ranked
comparison table, the ranking rule and the best baseline model), the evaluation
report (status, best baseline recommendation with its selection reason and error
pattern, the ranked comparison table with both gaps, the hold-out vs
cross-validation analysis with its thresholds and disclaimer, the confusion matrix
analysis with derived per-class metrics, execution time, experiment counts,
observations, limitations and recommendations) and the optimization report (status,
the models selected and not selected, per-model search details with the baseline
and optimized scores on separate lines, the best parameters, the improvement with
its direction, the ranked baseline-vs-optimized table, the best optimized
recommendation, the unchanged Phase 5 baseline winner, the separated experiment
and trial counts, the objective and ranking rules, observations, limitations and
recommendations).

### Running the tests
```bash
python -m unittest discover -s tests -t .

# Just the optional LLM layer
python -m unittest tests.test_llm_client tests.test_llm_prompts \
                   tests.test_llm_parser tests.test_llm_service \
                   tests.test_llm_integration
```

The Phase 8 suite **never requires a running Ollama server**. Every request goes
through a fake transport that returns scripted responses, so the tests are fast
and deterministic. The one place the real configured URL is touched is a test
that asserts the *unavailable* path is handled; it is skipped automatically when
a server happens to be running, because then the fallback would not be the code
under test. Nothing in the suite downloads a model.

Phase 8 adds tests for the default configuration being off, the three
distinguishable unavailability outcomes (server down, model missing, both fine),
timeouts and HTTP errors, JSON extraction from a response wrapped in prose or a
fenced code block, wrong-shape answers that are still valid JSON, missing and
unknown fields, scalar-to-list coercion, forbidden-claim detection,
optimisation-plan validation against the real model registry, code-shaped text
rejection, prompt determinism, the allowlist that keeps raw rows and secrets
out of the context, JSON-serialisability of the workflow result, the graph shape
with and without the layer, routing independence, every failure mode recovering,
both sample datasets end to end, the `--graph` and `--graph --llm` command lines,
repeat-run determinism, and that the bundled CSV files come out byte-identical.

The suite covers CSV loading, shape and feature-type detection, missing values,
duplicates, target detection, binary and multiclass classification detection,
class distribution, data quality checks, state updates and invalid input
handling. Phase 3 adds imputation and encoding configuration, one-hot encoding,
unknown-category handling, scaling, identifier / constant / datetime /
high-cardinality exclusion, duplicate handling, ColumnTransformer creation,
pipeline fitting and transforming, JSON-serialisability of the state, and an
explicit data-leakage test that proves the pipeline only learns from the training
split. Phase 4 adds registry and candidate-selection tests, X/y separation, the
stratified split and its documented fallbacks, pipeline structure and fitting,
missing values, categorical and unknown categories, identifier exclusion, metric
and confusion-matrix correctness, cross-validation (including small-class
handling), the baseline ranking rule, per-model failure recording, end-to-end
runs on both sample datasets, reproducibility, and a second explicit leakage suite
that hand-builds a split whose two halves differ by three orders of magnitude and
proves the imputer, scaler, encoder and standardisation learned from the training
half only. Phase 5 adds 187 tests for the gap arithmetic and its
close/moderate/substantial boundaries (including negative gaps and configurable
thresholds), binary and multiclass confusion-matrix analysis with derived per-class
metrics, the ranking rule and its agreement with Phase 4, execution-time totals,
experiment counts that keep models and hyperparameter trials apart, handling of
failed / partial experiments, missing cross-validation results and missing
confusion matrices, JSON-serialisability of the report, the best-model
recommendation, unsupported and empty evaluation states, the prerequisite check, a
wording audit that rejects unsupported overfitting claims, and the full
analysis → evaluation pipeline on both sample datasets.

### Using the agents in Python
```python
from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from utils.state import AgentState

state = AgentState()
state = DataAnalysisAgent().run(state, dataset_path="datasets/sample_classification.csv")

print(state.problem_type)              # binary_classification
print(state.target_column)             # Outcome
print(state.class_distribution)        # {"1": ClassDistributionEntry(...), ...}
print(state.analysis_summary.summary_text)

state = PreprocessingAgent().run(state)
print(state.preprocessing_status)              # completed
print(state.preprocessing_numerical_features)  # ['age', 'bmi', 'glucose', 'blood_pressure']
print(state.preprocessing_categorical_features)# ['gender', 'region', 'smoking']
print(state.excluded_features_by_reason)       # {'identifier_like': ['record_id']}
print(state.preprocessing_summary.summary_text)

state = ModelSelectionAgent().run(state)
print(state.selected_model_names)  # 5 baseline models, in registry order

state = TrainingAgent().run(state)
print(state.training_status)             # completed
print(state.experiment_count)            # 5
print(state.best_baseline_model.model_name)  # Logistic Regression
print(state.model_comparison.table_text)      # ranked comparison table

for experiment in state.experiment_results:
    print(experiment.model_name, experiment.accuracy, experiment.f1_score,
          experiment.cv_f1_mean, experiment.status)

# The fitted best pipeline is kept at runtime only (Phase 9 saves it):
state.best_pipeline.predict(X_test)      # Pipeline(preprocessor, model)

state = EvaluationAgent().run(state)      # reads the records above, trains nothing
report = state.evaluation_report
print(state.evaluation_status)                    # completed
print(report.best_model.best_baseline_model)      # Logistic Regression
print(report.selection_reason)                    # why it won
print(report.gap_analysis.summary)                # how hold-out and CV agree
print(report.gap_analysis.disclaimer)              # what the gap does NOT prove
print(report.table_text)                          # rank model acc prec rec f1 cv ... gaps
print(report.execution_time_analysis.total_experiment_time)
print(report.experiment_count.total_experiments, report.experiment_count.hyperparameter_experiments)
for analysis in report.confusion_matrix_analysis: # per-class metrics derived from the matrix
    print(analysis.model_name, analysis.correct_predictions, analysis.error_pattern)
print(report.limitations)                         # what this run cannot claim
```

Two useful knobs for experiments (both default to the reproducible values):

```python
TrainingAgent().run(state, cross_validation=False)              # hold-out only
TrainingAgent().run(state, model_parameters={"Random Forest": {"n_estimators": 300}})
```

An invalid override fails **only that experiment**; the other four still train and
the failure is recorded with its error message.

## Sample Datasets
`datasets/sample_classification.csv` (123 rows) and `datasets/sample_multiclass.csv`
(92 rows) are small synthetic datasets created for development and testing. They
contain numerical and categorical columns, a few missing values and a few
duplicated rows so the analysis can be verified end to end. They are generated
with a fixed seed, so results are reproducible.

## Design Principles
- **Local-first & Open-source:** no paid APIs or cloud dependencies
- **Modular:** agents, tools and utilities have separate responsibilities
- **Explainable:** every decision (target, problem type, quality issues,
  preprocessing, exclusions, model choice) comes with a reason that can be shown
  to the user
- **Incremental:** built phase by phase, never over-engineered at once
- **Reproducible:** deterministic pipeline, fixed random seed, committed samples
- **Safe:** agents report problems instead of silently changing the data
- **Leakage-free:** every transformer is fitted on training data only, never up
  front; cross-validation refits the whole Pipeline inside each fold
- **Honest reporting (Phase 5):** a measured difference is always stored next to
  its threshold, its interpretation and its limitation; the gap heuristic is
  labelled a heuristic, and training-set metrics are reported as unavailable
  instead of being faked
- **No silent degradation:** a model that fails, a split that cannot be
  stratified, a dataset too small for cross-validation — each is recorded with a
  reason instead of being hidden

## Results

Final measured results (details and charts: `reports/final_results.md`,
`reports/final_project_report.md`, `reports/phase11_final_report.md`,
`reports/phase11_comparison.json`).

### Conventional vs Autonomous benchmark

| Dataset | Metric | Conventional | Autonomous | Difference |
|---|---|---:|---:|---:|
| Binary (`sample_classification.csv`) | F1 | 0.957780 | 0.957780 | 0 |
| | Accuracy | 0.958333 | 0.958333 | 0 |
| | CV F1 | 0.893303 | 0.904579 | +0.011276 |
| | Runtime (s) / Experiments | 13.135 / 21 | 14.307 / 45 | +1.172 / +24 |
| Multiclass (`sample_multiclass.csv`) | F1 | 1.0000 | 1.0000 | 0 |
| | Accuracy | 1.0000 | 1.0000 | 0 |
| | CV F1 | 1.0000 | 1.0000 | 0 |
| | Runtime (s) / Experiments | 13.397 / 21 | 41.682 / 45 | +28.285 / +24 |

On both datasets the autonomous workflow **matched** the conventional hold-out
F1 (`delta_f1 = +0.0000`) while automating a broader search; it did not claim a
performance win. The conventional grid was cheaper in runtime and experiments.

### Test suite

`pytest -q` from the project root: **1887 passed, 0 failed, 7 warnings, 1369
subtests passed, exit code 0** (the final-release audit added three config
regression tests; details in `reports/final_evaluation_report.md`).

## Limitations

- Phase 6 optimized winners carry limited precision/recall provenance (not
  recorded for the autonomous benchmark rows).
- The benchmark datasets are small synthetic samples; results do not estimate
  real-world generalization.
- The benchmark measures machine runtime, not human time.
- Autonomous optimization costs more runtime and more experiments on the tested
  fixtures.
- The local-LLM live success path depends on a separately installed Ollama and
  model being available; the pipeline itself does not.
- The benchmark scope is limited to the two tested datasets and to
  classification.
- `models/` currently holds a baseline Logistic Regression artifact from an
  earlier run; it is replaced on each `--graph` run and never used in benchmark
  measurement.

## Future Scope

Larger real-world datasets, regression support, more model families, an
untouched external test set, richer multi-run experiment analysis, distributed
search, advanced agent planning, production serving, an MLflow model registry,
monitoring and drift detection are all clearly future work.

## License

No license has been defined for this project. The repository contains no
`LICENSE` file and no license is claimed; treat the code as academic/
presentation material unless a license is added later.