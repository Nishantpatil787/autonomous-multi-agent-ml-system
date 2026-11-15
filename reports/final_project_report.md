# Autonomous Multi-Agent System for Machine Learning Model Development and Optimization

Master project report. Every architecture statement, command, metric and test
count in this document refers to the actual implementation in this repository
and to measured results recorded in earlier phase reports
(`reports/phase9_final_report.md`, `reports/phase10_final_report.md`,
`reports/phase11_final_report.md`) and in `reports/phase11_comparison.json`.

---

## 1. Abstract

Machine learning model development still requires a large amount of repeated
manual work: inspecting a new dataset, choosing how to clean and transform it,
selecting candidate models, tuning hyperparameters, evaluating results and
deciding which model to keep. This project proposes an autonomous multi-agent
framework that automates the major stages of a conventional machine learning
workflow so that each stage is delegated to a specialised agent, coordinated by
an orchestrator, with minimal human intervention.

The system is open-source, local-first and uses no paid APIs. It accepts a
tabular CSV dataset and runs a deterministic pipeline — data analysis,
preprocessing, model selection, training, evaluation and hyperparameter
optimisation — orchestrated with LangGraph. An optional local LLM layer adds
human-readable explanations of completed stages but is never allowed to change
a metric, a model or a routing decision. Final models are persisted as fitted
scikit-learn pipelines with metadata, every graph run is recorded in a local
MLflow store, and a read-only Streamlit dashboard presents the results.

A controlled Phase 11 benchmark compared the autonomous workflow against a
careful conventional (manual, tools-only) workflow on two committed datasets.
On the binary dataset both sides selected a linear SVM with hold-out F1
0.957780; on the multiclass dataset both sides selected a Decision Tree with
hold-out F1 1.0000. The autonomous workflow matched the conventional hold-out
performance while automating search and orchestration; it consumed more
experiments and runtime and did not outperform the manual grid.

## 2. Introduction

Developing a machine learning model for a tabular dataset normally repeats the
same series of tasks. The developer must first understand the data — column
types, missing values, duplicates, class balance — then decide how to preprocess
it (impute, encode, scale), choose candidate model families, split the data
without leaking information, train and cross-validate each candidate, evaluate
the results honestly, tune hyperparameters, and finally keep exactly one model
with enough evidence to justify the choice. Each task is well understood, yet
each is time-consuming, error-prone and easy to get slightly wrong — in
particular the evaluation step, where a false sense of accuracy comes from
evaluating on data the model was tuned on.

This project automates that workflow as a chain of specialised agents. A user
provides a CSV file (and optionally the target column), and the system performs
data analysis, preprocessing, model selection, training, evaluation,
hyperparameter optimisation, final model persistence and experiment tracking,
and presents the outcome in a dashboard. The automation is deliberately
bounded: every algorithmic decision uses controlled registries, fixed seeds and
leakage-free splits, so "autonomous" here means *orchestrated, reproducible and
auditable* rather than unconstrained.

## 3. Problem Statement

Machine learning model development requires repeated manual work such as data
analysis, model selection and hyperparameter tuning. This project addresses the
problem of reducing that manual effort: it proposes an autonomous multi-agent
framework that automates these tasks with minimal human intervention, while
keeping every automated decision deterministic, transparent and verifiable
against a conventional manual baseline.

## 4. Objectives

The project implements a working, local-first autonomous multi-agent machine
learning system that can:

1. Automate data analysis of a tabular dataset (types, missing values,
   duplicates, statistics, class distribution, target resolution and problem
   type detection).
2. Automate preprocessing (feature exclusions with reasons, imputation,
   encoding, scaling) built as a reproducible pipeline without leakage.
3. Automate model selection from a controlled registry of five candidate
   families (Logistic Regression, Decision Tree, Random Forest, KNN, SVM).
4. Automate training with a stratified hold-out split, cross-validation and a
   deterministic ranking rule.
5. Automate evaluation of recorded results only — hold-out metrics, CV metrics,
   confusion matrices and a generalization-gap analysis, without retraining.
6. Automate hyperparameter optimisation with Optuna (TPE sampler, bounded search
   spaces, seeded, evaluated on the training part of the split only).
7. Orchestrate the agents with LangGraph (explicit routing, bounded retries,
   deterministic conditional decisions).
8. Provide an optional local LLM layer for human-readable explanations that can
   never influence any decision.
9. Persist the final model as a complete fitted preprocessing + estimator
   pipeline with metadata and safe loading.
10. Provide a Streamlit dashboard that is strictly read-only and never retrains.
11. Track every graph run in a local MLflow store (SQLite, no server).
12. Compare the autonomous workflow against a conventional tools-only ML workflow
    on the same data, same split and same metrics (Phase 11 benchmark).

## 5. Existing/Traditional Workflow

The conventional workflow used as the comparison baseline in Phase 11 is the
manual reference implementation. On each dataset it performs, with tools and
the shared model registry only:

1. a deterministic split (`test_size = 0.2`, `random_state = 42`, stratified);
2. the same preprocessing defaults used by the autonomous side (no leakage);
3. a **manual 21-configuration grid** over the five candidate families (each
   family with its curated hyperparameter grid);
4. per-configuration training and evaluation with the same weighted metrics;
5. selection of the best configuration by the same ranking rule (primary
   weighted F1, secondary accuracy, then model name);
6. timing of the whole pass with `time.perf_counter()`.

It intentionally uses no agents, no LangGraph, no LLM and no Optuna: it is the
"human writes a grid search script by hand" baseline. Its results are recorded
in `reports/phase11_comparison.json` (marked `workflow: conventional`).

## 6. Proposed System

The proposed system is a multi-agent architecture in which each agent owns one
stage of the workflow and communicates through a single structured shared state
object (`AgentState`). The stages are data analysis, preprocessing, model
selection, training, evaluation and optimization, coordinated by LangGraph with
a deterministic `Should Optimize?` decision, completed by model persistence and
human-readable explanation by an optional local LLM.

Design invariants implemented in code and asserted by the test suite:

- every trainable decision is seeded (`random_state = 42`), including Optuna;
- every metric is computed with `weighted` averaging on the same hold-out split;
- preprocessing statistics are learned on the training part only (no leakage);
- the evaluation agent never retrains; it interprets recorded results;
- the LLM layer is inert unless asked (`--llm`), returns strictly validated
  JSON, and cannot influence routing;
- persistence writes a single fitted pipeline plus metadata and results JSON
  atomically, and `load_persisted_model` is the only joblib load site.

## 7. System Architecture

The architecture is a linear pipeline coordinated by a state machine. The main
flow is:

```
Dataset
  ↓
Data Analysis Agent
  ↓
Preprocessing Agent
  ↓
Model Selection Agent
  ↓
Training Agent
  ↓
Evaluation Agent
  ↓
Should Optimize? ── No ──► Final Model Selection
  │ Yes                                │
  ▼                                    ▼
Optuna Optimization                Model Persistence
  ↓                                    ↓
  └──────────► Final Model Selection   MLflow (local tracking)
                                        ↓
                                   Streamlit Dashboard
```

The optional local LLM layer (`llm/`) is connected only to explanation of
completed stages; deterministic ML and routing remain authoritative. The graph
compiles seven executable nodes (`data_analysis`, `preprocessing`,
`model_selection`, `training`, `evaluation`, `optimization`,
`model_persistence`) plus START/END and the conditional edge after evaluation.

## 8. Agent Architecture

Every agent inherits `agents/base_agent.py`, shares the `AgentState` and
delegates reusable operations to `tools/`. The following agents exist:

### OrchestratorAgent (`agents/orchestrator.py`)
- Responsibility: Phase 1 placeholder for overall workflow coordination.
- Input / output: `AgentState` (unchanged).
- Important logic: logs that coordination is ready. The actual coordination
  role is fully realised in Phase 7 by the compiled LangGraph in `graph/`.
- Failure behavior: applies the base `BaseAgent` message logging contract only.

### DataAnalysisAgent (`agents/data_analysis_agent.py`)
- Responsibility: read-only exploration of the dataset.
- Input: raw CSV path; Output: features, types, missing values, duplicates,
  statistics, class distribution, resolved target column, problem type, data
  quality issues, agent messages.
- Important logic: target-column heuristic resolution with a documented
  confidence and reasoning; problem-type detection (binary / multiclass /
  regression).
- Failure behavior: a missing or malformed dataset is reported as a failed
  agent message; the graph records the failure (see §14) instead of masking it.

### PreprocessingAgent (`agents/preprocessing_agent.py`)
- Responsibility: decide exclusions, imputation, encoding and scaling and build
  a reproducible, *unfitted* `ColumnTransformer`.
- Input: analysed feature groups and types; Output: selected features, excluded
  features with reasons, strategy summary, pipeline configuration.
- Important logic: exclusions are decided from the analysis alone; the
  transformer parameters are fitted later only on the training split.
- Failure behavior: recorded in state; the graph retries once, then routes to
  the error path.

### ModelSelectionAgent (`agents/model_selection_agent.py`)
- Responsibility: select candidate models for the run.
- Input: problem type; Output: candidate `ModelDefinition` records.
- Important logic: candidates come exclusively from the controlled registry in
  `tools/model_tools.py` (LR, DT, RF, KNN, SVM); no arbitrary model class is
  ever instantiated.
- Failure behavior: no candidate for a given problem type is detected during
  vectorisation into the compiled graph workers.

### TrainingAgent (`agents/training_agent.py`)
- Responsibility: train every candidate, evaluate cross-validation and hold-out
  metrics, rank the baseline candidates.
- Input: pipeline configuration and candidate definitions; Output: fitted
  models, metrics, timings, confusion matrices, ranked best baseline.
- Important logic: `StratifiedKFold(5, shuffle=True, random_state=42)` on the
  training split for CV; hold-out metrics on the untouched test split; a
  documented deterministic ranking (weighted F1 primary, accuracy secondary,
  model name tie-break).
- Failure behavior: a family that fails to fit (e.g. too few samples for CV) is
  recorded with the reason instead of being silently dropped.

### EvaluationAgent (`agents/evaluation_agent.py`)
- Responsibility: interpret baseline results only — never trains.
- Input: recorded baseline results; Output: gap analysis (hold-out vs CV),
  confusion-matrix analysis, timings and counts, recommendation.
- Important logic: the "generalization gap" is the difference between hold-out
  and cross-validated metrics, computed from the recorded numbers.
- Failure behavior: not applicable to decisions; the agent produces a report.

### OptimizationAgent (`agents/optimization_agent.py`)
- Responsibility: run the Optuna search on the top baselines and compare tuned
  versus untuned models.
- Input: best baseline models; Output: best optimized model, per-model trial
  summaries and a comparison report.
- Important logic: TPE sampler, bounded per-family search spaces,
  `OPTUNA_RANDOM_SEED = 42`, objective = mean weighted F1 over the split's CV,
  evaluated *only* on the training part; per-model trial budget (20 trials).
- Failure behavior: a failed trial is recorded; if no usable optimized winner
  exists, the baseline candidate is kept (documented by the persistence agent).

### PersistenceAgent (`agents/persistence_agent.py`)
- Responsibility: choose the final model and write the artifacts.
- Input: best baseline and best optimized models; Output: `models/`
  `best_model.joblib`, `best_model_metadata.json`, `run_results.json`.
- Important logic: final selection compares optimized vs baseline candidates
  with the existing ranking rule on the same hold-out split they were scored on;
  the final fit is a refit on training data; writes are atomic.
- Failure behavior: an invalid/missing candidate falls back to the recorded
  valid baseline; reasons are written into the metadata (`selection_reason`).

## 9. Data Analysis

Phase 2 (`tools/data_tools.py`, `agents/data_analysis_agent.py`). The analysis
stage is read-only: it reports column names and types, per-group feature
breakdown, missing values, duplicates, descriptive statistics, class
distribution, target resolution with reasoning, problem type and data quality
issues, and it resolves the target column heuristically when not provided.

Measured on the committed datasets:

| Dataset | Rows | Columns | Target | Problem type |
|---|---:|---:|---|---|
| `sample_classification.csv` | 123 | 9 | `Outcome` (binary) | binary classification |
| `sample_multiclass.csv` | 92 | 7 | `class_label` (3 classes) | multiclass classification |

The class distributions, missing-value findings and quality issues for each run
are printed in the CLI data-analysis report and stored in the run's dashboard
section 3.

## 10. Preprocessing

Phase 3 (`tools/preprocessing_tools.py`, `agents/preprocessing_agent.py`). The
preprocessing agent:

- excludes features for documented reasons (high missingness, all-constant,
  drop-information columns such as an ID, or near-duplicate content) and
  records **why** each exclusion happened;
- **numerical imputation**: missing values filled with the median of the feature
  (learned on training data);
- **categorical imputation**: missing values filled with the mode (learned on
  training data);
- **encoding**: one-hot encoding for categorical features;
- **scaling**: standardisation (StandardScaler) for numerical features;
- builds a `ColumnTransformer` pipeline so preprocessing and model form one
  estimator.

Leakage prevention: no statistic is computed on the whole dataset before the
split; the split happens first, and every imputation/scaling parameter is fitted
on the training split and applied to the test split.

## 11. Model Selection and Training

Phase 4 (`tools/model_tools.py`, `tools/training_tools.py`). The system trains
the five candidate families from the controlled registry:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbours (KNN)
- Support Vector Machine (SVM)

Procedure:

1. **Split**: `train_test_split(test_size=0.2, random_state=42, stratify=y)`
   when the class sizes allow it.
2. **Cross-validation**: `StratifiedKFold(5, shuffle=True, random_state=42)`
   on the training split, with a documented safe reduction when a fold would
   lose a class.
3. **Training**: each candidate is fitted inside the preprocessing pipeline on
   the training split; hold-out predictions are made on the untouched test
   split.
4. **Ranking**: a deterministic rule — primary weighted F1 (descending),
   secondary accuracy (descending), then model name (ascending) as the final
   tie-break — selects the best baseline.

The deterministic seed means repeated runs on the same dataset produce the same
training/evaluation results.

## 12. Evaluation

Phase 5 (`tools/evaluation_tools.py`, `agents/evaluation_agent.py`). The
evaluation agent interprets recorded results only. Metrics are computed with
`weighted` averaging over classes:

- **Accuracy**: fraction of correct predictions.
- **Precision**: weighted mean of per-class precision.
- **Recall**: weighted mean of per-class recall.
- **F1**: weighted harmonic mean of precision and recall.
- **CV metrics**: mean of the per-fold metrics on the training split.
- **Confusion matrix**: recorded per run for the best model.
- **Generalization gap**: hold-out metric minus CV metric, reported with
  interpretation (a large drop signals overfitting/variance).

A limitation recorded in Phase 5/6: the optimization stage records the weighted
F1 and accuracy of the optimized winners but not their precision/recall, which
is why the autonomous columns of the Phase 11 benchmark show no precision/recall
for optimized winners.

## 13. Hyperparameter Optimization

Phase 6 (`tools/optimization_tools.py`, `agents/optimization_agent.py`,
dependency: `optuna>=4.0,<5.0`).

- **Sampler**: Tree-structured Parzen Estimator (TPE).
- **Search spaces**: bounded per-family ranges defined in
  `tools/optimization_tools.py` (for example SVM `C` and kernel, RF
  `n_estimators`/`max_depth`/`min_samples_split`, tree depths and split
  constraints, KNN `n_neighbors` and distance metric).
- **Trial budget**: 20 trials per tuned model.
- **Objective**: maximise the mean weighted-F1 from the split's CV on the
  **training part only**; the hold-out split is never used during search.
- **CV strategy**: the same `StratifiedKFold` and seed as Phase 4.
- **Leakage prevention**: preprocessing parameters are fitted inside the CV
  folds' training partitions; Optuna itself is seeded (`OPTUNA_RANDOM_SEED =
  RANDOM_STATE = 42`).
- **Comparing fairly**: tuned vs untuned metrics are compared on the same
  hold-out split.

The optimization runs only after the deterministic `Should Optimize?` decision
(§14) and only on the top baseline models.

## 14. LangGraph Orchestration

Phase 7 (`graph/`; dependency: `langgraph`). The compiled workflow is:

```
START
  ↓
Data Analysis
  ↓
Preprocessing
  ↓
Model Selection
  ↓
Training
  ↓
Evaluation
  ↓
Should Optimize? ── No ──► Model Persistence ──► END
  │ Yes
  ▼
Optuna Optimization
  ↓
Model Persistence ──► END
```

- **Conditional optimization**: after evaluation a deterministic routing
  function decides whether the Optuna search is worthwhile; a
  small-gain/disabled decision is recorded (`skip_disabled`) instead of running
  the search.
- **Retries**: each node has a bounded retry policy (one retry per node);
  repeated failure records the error and routes to the error path — failures are
  never silently masked.
- **Failures**: a failed stage writes a structured `GraphError` and the run
  still produces a report of what succeeded and what failed.
- **Deterministic routing**: all routing decisions are pure functions of the
  shared state; the optional LLM nodes (added only with `--llm`) never route.

## 15. Local LLM Integration

Phase 8 (`llm/`; no additional Python dependency). The layer is optional and off
by default (`OLLAMA_ENABLED=false`):

- it talks only to a **local Ollama** server over loopback
  (`OLLAMA_BASE_URL=http://localhost:11434`; the client rejects non-loopback
  hosts, so it cannot be repointed at a hosted endpoint);
- default model `qwen2.5:3b`, temperature 0, 30-second timeout, at most one
  retry — so an unavailable model never delays the pipeline meaningfully;
- the model receives allowlisted, summarised context and must return **strictly
  structured JSON**; `llm/parser.py` extracts and validates it, with one
  controlled repair attempt;
- any health-check or parse failure falls back to a documented status value and
  **never raises**; the run completes and exits 0 with or without Ollama;
- the LLM output is explanatory only — it is attached to explanation nodes, it
  does not change metrics, models or routing, so the system works identically
  without an LLM.

## 16. Model Persistence

Phase 9 (`tools/persistence_tools.py`, `agents/persistence_agent.py`).

- **Artifact**: `models/best_model.joblib` stores the *whole fitted pipeline*
  (`preprocessor` → `estimator`), so prediction needs no manual transformation.
- **Metadata**: `models/best_model_metadata.json` records the model name,
  type, selection source (`baseline` / `optimized`), selection reason, pipeline
  steps, and the decision rule.
- **Results**: `models/run_results.json` records the node-walk timings and
  evaluation results consumed by the dashboard.
- **Final model selection**: compares the best baseline and the best optimized
  candidate with the existing ranking rule on the hold-out split they were
  already scored on; when optimization was disabled or produced no usable
  winner, the baseline candidate becomes the final model.
- **Safe loading**: `load_persisted_model()` is the only `joblib.load` site in
  the codebase; writes are atomic, so a partially written artifact is not
  readable as a model.

## 17. Streamlit Dashboard

The dashboard (`dashboard.py`, `tools/dashboard_tools.py`) is read-only: it
renders `models/run_results.json`, the saved model and the MLflow store, and it
never trains, tunes or evaluates. It implements **12 sections**:

1. Header (project and pipeline, plus the section index)
2. System status
3. Dataset overview
4. Model comparison
5. Evaluation
6. Confusion matrix
7. Optimization
8. LLM interpretation
9. Prediction (live predictions through the saved pipeline)
10. Model information
11. Experiment tracking (latest recorded MLflow run + history)
12. Conventional vs Autonomous (Phase 11 benchmark verdicts and deltas)

The dashboard renders from the actual artifacts: section 3 from the analysis
record, sections 4–7 from the evaluation/optimization records, section 9
through `load_persisted_model`, section 11 from the MLflow store and section 12
from `reports/phase11_comparison.json`.

## 18. MLflow Experiment Tracking

Phase 10 (`tools/mlflow_tools.py`; dependency: `mlflow>=3.0,<4.0`).

- **Local tracking only**: the canonical store is a local SQLite file,
  `experiments/tracking/mlflow.db`; no server is started and none is required.
- **Experiments and runs**: graph runs record into the experiment
  `autonomous-multi-agent-ml`; each run logs parameters (dataset name, model,
  split/CV and Optuna settings), metrics (hold-out and CV F1, accuracy,
  timings), and a `langgraph` tag, with per-key `log_param`/`log_metric` calls
  for the client API used.
- **Artifacts and history**: the dashboard section 11 lists the latest run and a
  bounded history; `MLFLOW_HISTORY_LIMIT` bounds the dashboard table.
- **Reproducibility**: recorded runs replay the same seed/settings and the store
  itself is byte-validated by tests (dataset hashes and artifact integrity).
- **Failure isolation**: MLflow recording is lazy and never influences a
  decision; a tracking failure cannot fail a workflow run. `--no-mlflow` (or
  `MLFLOW_ENABLED=false`) disables recording per run, and the benchmark records
  only into its own dedicated store (never the canonical one).

## 19. Conventional vs Autonomous Benchmark

Phase 11 (`benchmarks/`). Both sides run the same committed datasets, the same
split (`random_state = 42`), the same preprocessing defaults, the same five
candidate families and the same weighted metrics. The conventional side is a
tools-only 21-configuration grid; the autonomous side is the existing
`run_workflow` graph (LLM off, Optuna on with 20 trials per tuned model,
persistence to scratch, tracking off). Full details and charts:
`reports/phase11_final_report.md`, `reports/phase11_comparison.json` / `.csv`.

### Binary Dataset (`sample_classification.csv`; target `Outcome`)

| Metric | Conventional | Autonomous | Difference |
|---|---:|---:|---:|
| F1 | 0.957780 | 0.957780 | 0 |
| Accuracy | 0.958333 | 0.958333 | 0 |
| CV F1 (mean) | 0.893303 | 0.904579 | +0.011276 |
| Runtime (s) | 13.135 | 14.307 | +1.172 |
| Experiments | 21 | 45 | +24 |

Both sides won with a linear SVM (conventional `C=10`; autonomous
`C≈14.566`, the Optuna plateau twin). Verdict: `delta_f1 = +0.0000` (match).

### Multiclass Dataset (`sample_multiclass.csv`; target `class_label`)

| Metric | Conventional | Autonomous | Difference |
|---|---:|---:|---:|
| F1 | 1.0000 | 1.0000 | 0 |
| Accuracy | 1.0000 | 1.0000 | 0 |
| CV F1 (mean) | 1.0000 | 1.0000 | 0 |
| Runtime (s) | 13.397 | 41.682 | +28.285 |
| Experiments | 21 | 45 | +24 |

Both sides won with a Decision Tree. Verdict: `delta_f1 = +0.0000` (match).
Precision/recall for the autonomous side were not recorded (Phase 6
limitation), so those cells are blank rather than fabricated.

## 20. Results and Discussion

Measured result, stated without overclaiming:

- **Performance**: on both committed datasets the autonomous workflow matched
  the conventional hold-out performance exactly (`delta_f1 = +0.0000`). The
  same winning family emerged on each side, and on the binary dataset the
  autonomous CV-F1 (0.904579) slightly exceeded the conventional (0.893303).
- **Runtime**: autonomous took longer (binary +1.172 s, multiclass +28.285 s);
  these are single-run wall-clock timings around the ML work, not quality
  measures.
- **Experiment count**: conventional 21 grid configurations versus autonomous 45
  work items (5 baseline fits + 40 trials). The autonomous side explores a
  broader search space by design.
- **Automation**: the autonomous run replaced the hand-written grid script with
  the multi-agent graph and an Optuna search, automated the experimentation it
  performed, recorded it in MLflow, and produced the same hold-out scores.
- **Trade-offs**: on these two small synthetic fixtures the careful manual grid
  was cheaper and scored identically; the autonomous workflow's value here is
  the automation and breadth of search, not a performance win. Generalization to
  real datasets is not estimated by these fixtures.

## 21. Reproducibility

Summary — full detail in `reports/reproducibility.md` and `.env.example`:

- **Python 3.12.3**; dependencies pinned by range in `requirements.txt`
  (pandas 3.0.1, numpy 2.4.2, scikit-learn 1.8.0, optuna 4.9.0, langgraph
  1.2.12, joblib 1.5.3, streamlit 1.65.0, mlflow 3.16.1, matplotlib 3.11.2,
  pytest 9.1.1 in the verified environment).
- **Random seed**: `random_state = 42` for split, CV, model estimators and
  Optuna (`OPTUNA_RANDOM_SEED = 42`).
- **Datasets** (committed, hash-asserted): `sample_classification.csv`
  (SHA-256 `49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c`),
  `sample_multiclass.csv` (SHA-256
  `0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8`).
- **MLflow**: canonical store `experiments/tracking/mlflow.db`; benchmark store
  `experiments/tracking_benchmark/mlflow.db`; experiment names
  `autonomous-multi-agent-ml` and `benchmark-phase11`.
- **Artifact structure**: `models/best_model.joblib`,
  `models/best_model_metadata.json`, `models/run_results.json`; benchmark
  outputs `reports/phase11_comparison.{json,csv}` plus three PNG charts; final
  diagrams regenerated with `python make_final_diagrams.py`.
- **Commands**: `pytest -q`; `python app.py --graph [--no-optimize] [--llm]
  [--no-save-model] [--no-mlflow]`; `streamlit run dashboard.py`;
  `python app.py --benchmark [--all] [--dataset {classification,multiclass}]
  [--no-mlflow]`.

## 22. Testing and Validation

The full test suite runs with `pytest -q` from the project root. Final measured
result:

- **1887 passed**, 0 failed, 7 warnings, **1369 subtests passed**, exit code 0
  (≈ 7 min; includes three config regression tests added by the final release
  audit - see `reports/final_evaluation_report.md`).

Coverage relevant to validation:

- **Dataset integrity**: tests assert the SHA-256 of both committed datasets and
  that no test writes into them.
- **Binary / multiclass validation**: workflow and benchmark tests run both
  datasets end to end and assert the recorded verdicts and comparison files.
- **Dashboard validation**: `test_dashboard.py` renders every section from
  synthetic/real records and asserts nothing trains.
- **Artifact integrity**: a session guard (`ProjectArtifactIsolationGuard`)
  byte-checks `models/` and `experiments/tracking/` before the suite and after
  every test, so tests and benchmarks are proven incapable of mutating canonical
  artifacts; benchmark MLflow writes are confined to the dedicated store.
- **Benchmark fairness**: tests pin the split, preprocessing, candidate registry,
  metrics and `--no-mlflow`/`--benchmark` interactions.

## 23. Security and Reliability

- **No paid APIs** and no credentials: there are no API keys, accounts or secret
  settings anywhere in the repository (`.env.example` documents only optional
  overrides).
- **Local-first**: the LLM layer talks only to a loopback Ollama host and
  rejects any non-loopback URL; MLflow is a local SQLite file.
- **No arbitrary joblib uploads**: `load_persisted_model` is the single load
  site and loads from a fixed models directory, never from user-supplied paths.
- **Controlled Ollama host**: base URL must be loopback; temperature 0, one
  retry, strict JSON schema validation.
- **Deterministic routing**: all routing is a pure function of state; the LLM
  never controls flow.
- **Failure isolation**: MLflow and LLM failures can never fail or affect a
  workflow run; per-node bounded retries with recorded error routing.
- **Artifact protection**: the test guard byte-validates `models/` and
  `experiments/tracking/`, and benchmark recording is confined to its own store.

## 24. Limitations

Only real limitations are listed:

- Phase 6 optimized winners carry limited precision/recall provenance: the
  optimization stage records weighted F1 and accuracy for tuned winners but not
  precision/recall, so the autonomous benchmark rows omit those cells.
- The benchmark datasets are small synthetic samples (123 and 92 rows); results
  do not estimate real-world generalization.
- The benchmark measures machine runtime, not human time; a real developer's
  grid-search effort is not captured.
- Autonomous optimization requires more runtime and more experiments than the
  hand-picked grid on the tested fixtures.
- The local-LLM live success path depends on a separately installed Ollama and
  model being available; the pipeline itself does not require them.
- The benchmark scope is limited to the two tested datasets and to
  classification tasks.
- Models/currently contains a baseline Logistic Regression artifact from an
  earlier deployment run; it is honest build output, is replaced on each
  `--graph` run, and never participates in benchmark measurement.

## 25. Future Scope

Clearly future work, not claims about the current system:

- larger real-world datasets and more classification tasks;
- regression support and additional problem types;
- additional model families and richer search spaces;
- an untouched external test set held out from development;
- richer experiment analysis (multi-run statistics, learning curves);
- distributed execution of the search;
- advanced agent planning;
- production deployment concerns (serving, monitoring);
- an MLflow model registry;
- drift detection for deployed models.

## 26. Conclusion

This project demonstrates an autonomous, local-first multi-agent machine
learning system that automates data analysis, preprocessing, model selection,
training, evaluation, hyperparameter optimization, persistence and experiment
tracking behind a LangGraph orchestrator, with an optional local LLM layer and a
read-only Streamlit dashboard.

The Phase 11 controlled benchmark shows that the autonomous workflow **matched**
the conventional tools-only workflow's hold-out performance on both committed
datasets (binary F1 0.957780 = 0.957780; multiclass F1 1.0000 = 1.0000) while
automating a broader model search, at the cost of additional experiments and
runtime. The measured evidence supports the conclusion that the system
automates and documents the major stages of a conventional ML workflow and
achieves parity with a careful manual baseline on the tested datasets — it does
**not** support a claim that autonomous ML is universally better. The project is
complete for its stated scope: reproducible, comprehensively tested, and ready
for evaluation as a college research/engineering project.