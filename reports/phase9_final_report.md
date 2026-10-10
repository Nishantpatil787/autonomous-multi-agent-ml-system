# Phase 9 Final Report: Model Persistence and Streamlit Dashboard

**Project:** Autonomous Multi-Agent System for Machine Learning Model Development and Optimization
**Phase:** 9 - Model Persistence and Streamlit Dashboard
**Interpreter used for all results below:** Python 3.12.3
**Seed:** `random_state=42` everywhere, unchanged from earlier phases

---

## 1. Objective

Turn a completed workflow run into two things it was not before:

1. a **deployable artifact** on disk - the fitted pipeline plus a metadata document - written automatically at the end of a run; and
2. a **read-only dashboard** that presents what that run recorded.

Phase 9 adds neither a metric, a model, nor a ranking. Its acceptance conditions were:

- the core pipeline must still run with **no LLM and no Ollama installed**;
- the dashboard must **never train**, never tune and never re-evaluate;
- persistence must not be able to influence any machine learning decision;
- no uploaded `.joblib`/`.pkl` may ever be loaded;
- the two committed datasets must be byte-identical afterwards;
- every pre-existing test must stay green.

---

## 2. Existing Architecture Reviewed

Before writing anything, the following were read and reused rather than replaced:

| Existing piece | What Phase 9 reused from it |
|---|---|
| `utils/state.py` contracts | `AgentState`, `EvaluationReport`, `OptimizationReport`, `ExperimentResult`, `OptimizationResult`, `OptimizedModelRecommendation` |
| `tools/training_tools.rank_experiments` | The ranking key, verbatim, as the final selection rule |
| `tools/optimization_tools.rank_optimized_results` | The same key again, confirming Phase 6 already agreed with Phase 5 |
| `graph/state.py` | `GraphState`, and the pattern of a node enum plus a `REQUIRED_NODES` subset |
| `graph/routing.py` | The conditional-destination-map pattern, `DEGRADABLE_NODE_VALUES`, `non_retryable_error_types()` |
| `graph/workflow.py` | `WORKFLOW_ORDER`, the conditional-edge builder, `expected_structure()` |
| `utils/config.py` | The constant-and-docstring convention (`PERSISTENCE_*`, `DASHBOARD_*`) |
| `app.py` | The `argparse` design and the per-phase report section format |
| `tests/helpers.py` | `BINARY_DATASET`, `MULTICLASS_DATASET`, `file_hash` |

Three deliberate decisions came out of that inspection:

- **No new state field was invented.** Every metadata value maps to a field that already existed. The only state additions are `persistence_status`, `persistence_report`, `persistence_output_dir` and `enable_persistence`.
- **`GraphNode.MODEL_PERSISTENCE` was added to the existing `GraphNode` enum**, not as a separate enum. A separate `PersistenceGraphNode` enum would have made persistence invisible to the existing node helpers.
- **The sequential (non-graph) path was left alone.** It still stops after Evaluation. Persistence is a graph node, so `--graph` is what persists a model. This is pre-existing behaviour, documented in §16 rather than changed.

---

## 3. Model Persistence Architecture

### 3.1 Two files, one direction

```
tools/persistence_tools.py     pure functions, no agent imports, no I/O policy
agents/persistence_agent.py    orchestration: prerequisites, ordering, failure recording
```

`tools/persistence_tools.py` exports 32 names (`__all__`), all of them functions or small dataclasses:

| Group | Names |
|---|---|
| Selection | `select_final_model`, `baseline_candidate`, `optimized_candidate`, `experiment_for`, `optimization_result_for`, `Candidate` |
| Final fit | `build_final_pipeline`, `build_tuned_pipeline`, `rebuild_training_split`, `resolve_model_type`, `FinalPipeline` |
| Metadata | `build_metadata`, `build_run_results`, `library_versions`, `json_safe` |
| Write | `write_artifacts`, `save_pipeline`, `write_json`, `artifact_paths`, `artifacts_exist`, `resolve_output_dir`, `ArtifactPaths` |
| Read | `load_persisted_model`, `load_pipeline`, `load_run_results`, `read_json`, `validate_schema`, `PersistedModel` |
| Errors | `PersistenceError`, `NoModelToPersistError`, `ArtifactWriteError`, `ArtifactReadError` |

### 3.2 The agent's ordering contract

`PersistenceAgent.run()` performs three steps in a fixed order, and the docstring says why:

1. **select** - `select_final_model(state)`, then log `selection_reason`;
2. **final fit** - `build_final_pipeline(state, selection)`, then log the strategy;
3. **write** - `_verify_paths(...)` then `write_artifacts(...)`.

The selection is recorded on the state *before* anything is written, so a failure in step 3 can never change what step 1 decided.

### 3.3 Failure handling

Four typed exceptions under one base:

```
PersistenceError
├── NoModelToPersistError    neither candidate carries the primary metric
├── ArtifactWriteError       the write itself failed
└── ArtifactReadError        an existing artifact is unreadable or malformed
```

`_record_failure(state, reason)` sets `persistence_status = "failed"`, logs the reason and appends to `state.errors`, and then **re-raises**. Recording without swallowing was the point: the graph needs the reason in order to report *why* an artifact is missing, and the caller needs the exception in order not to believe it succeeded.

`PersistenceAgent` deliberately **does not** change `AgentState.status`. A missing artifact is not a failed evaluation.

### 3.4 Atomic writes

Every file is written the same way: `tempfile.mkstemp` in the destination directory → write → `os.fsync` → `os.replace`. `os.replace` is atomic on Windows and POSIX, so a reader never sees a half-written artifact.

The write order is **model, then metadata, then results**. If metadata were written first and the process died between the two, the dashboard would find metadata describing a model that does not exist - and `load_dashboard_state` would have to decide which of the two it trusts.

---

## 4. Final Model Selection Policy

### 4.1 The rule, in full

Persistence introduces **no new metric, threshold or tie-break.** It reuses the key `rank_experiments` and `rank_optimized_results` already use:

> primary hold-out `f1_score` descending → secondary hold-out `accuracy` descending → model name ascending.

`select_final_model` then applies exactly two cases:

1. **A usable tuned candidate exists** (Phase 6 recorded a `best_optimized_model` carrying a hold-out `f1_score`) → compare the two candidates with the rule above and persist the winner.
2. **No usable tuned candidate** → persist the best valid baseline.

If neither candidate carries the primary metric, `NoModelToPersistError` is raised. The run does not invent a model.

Note that the gate is *"is there a usable tuned candidate?"*, **not** *"what does `optimization_status` say?"*. A search that was disabled, one that was skipped, one that failed, and one that finished but recorded nothing usable all arrive at the same place, which is the honest reading of all four. The status is still recorded in the metadata.

### 4.2 Metric provenance never mixes

This was the subtlest correctness issue in the phase.

| | Baseline selection | Optimized selection |
|---|---|---|
| `metrics_source` | `evaluation_holdout` | `optimization_holdout` |
| accuracy, `f1_score` | Phase 4 | Phase 6 |
| precision, recall | Phase 4 | **`null`** |
| `confusion_matrix` | Phase 5 stored matrix | **`null`** + `confusion_matrix_note` |

Phase 6 records hold-out `f1_score` and `accuracy` for a tuned model, but no precision, recall or confusion matrix. Filling those in from the *untuned* baseline would have put numbers next to each other that were never measured on the same model. Instead they are `null` and the dashboard prints `-`, with the note explaining why.

The symmetric mistake - re-scoring the tuned model to obtain the missing values - was rejected: that is a second evaluation pipeline, which the requirements exclude and which would make the artifact's metrics unreproducible from the run that produced them.

### 4.3 The final fit happens after selection

`build_final_pipeline` returns a `FinalPipeline(pipeline, strategy, note, train_samples)` and one of two strategies:

| Strategy | When | What happens |
|---|---|---|
| `reused_fitted_pipeline` | Baseline selection, `state.best_pipeline` present | The already-fitted pipeline is saved **unchanged**. `train_samples=0`. |
| `rebuilt_and_refitted_pipeline` | Optimized selection | The estimator is rebuilt from the recorded best parameters and the Phase 3 preprocessing definition, then refitted. `train_samples` is recorded. |

The note text for the reuse strategy is asserted not to contain the words "hold-out", because for that strategy nothing was fitted at all and saying otherwise would overstate what the artifact does.

Both strategies fit on the **training split only**. `rebuild_training_split` recreates the split with the same configuration Phases 4-6 used (`test_size=0.2`, `random_state=42`, stratified), which is how `train_samples=96` on the binary dataset coexists with `test_samples=24` on 123 rows: 3 duplicate rows were removed before splitting.

The refit changes **no** selection, **no** optimization result and **no** metric. The metadata metrics remain the ones measured during the run.

---

## 5. Artifact Structure

Three files in `models/` (`config.MODELS_DIR`):

| File | Size (binary run) | Contents |
|---|---|---|
| `models/best_model.joblib` | 3 268 bytes | The fitted `sklearn.pipeline.Pipeline`, via `joblib` |
| `models/best_model_metadata.json` | 4 874 bytes | 39 top-level keys, described in §6 |
| `models/run_results.json` | 27 169 bytes | The full read model the dashboard consumes |

The saved pipeline has exactly two named steps:

```
preprocessor  ->  ColumnTransformer(SimpleImputer + StandardScaler | SimpleImputer + OneHotEncoder)
model         ->  the selected estimator
```

Everything a consumer needs is inside `preprocessor`: the imputation strategies, the scaler and the fitted category vocabulary. This is why the dashboard needs no second copy of any of it.

`ArtifactPaths` is a frozen dataclass of `directory, model, metadata, results`, derived from **configuration only**. `resolve_output_dir(output_dir)` falls back to `config.PERSISTENCE_OUTPUT_DIR` then `config.MODELS_DIR`.

`artifacts_exist(output_dir)` returns True when the model **and** the metadata are both present. The report bundle is deliberately not part of that test, because a partial `run_results.json` still leaves a usable artifact - the dashboard degrades to model-only rather than claiming nothing was saved.

---

## 6. Metadata Structure

`best_model_metadata.json`, 39 top-level keys, verified against the real binary artifact:

**Identity** - `artifact`, `schema_version` (1), `produced_by` (`agents/persistence_agent.py`), `artifact_size_bytes`, `dataset_name`, `target_column`, `task_type`, `problem_type_reason`

**Selection** - `model_name`, `model_type`, `selection_source`, `selection_reason`, `selection_rule`, `optimization_used`, `optimization_status`, `baseline_model_name`, `baseline_f1_score`, `optimized_model_name`, `optimized_f1_score`

**Metrics** - `metrics` (`accuracy`, `precision`, `recall`, `f1_score`), `cv_metrics` (`cv_f1_mean`, `cv_accuracy_mean`, `cv_f1_std`, `cv_n_splits`, `source`), `metrics_source`, `confusion_matrix`, `confusion_matrix_note`

**Schema** - `feature_columns`, `feature_count`, `numerical_features`, `categorical_features`, `excluded_features`, `class_labels`, `class_count`, `pipeline_steps`, `preprocessing`, `split`, `best_params`

**Provenance** - `final_fit` (`strategy`, `note`, `train_samples`), `library_versions`, `notes`, `limitations`

Three details worth naming:

- **The key is `f1_score`, not `f1`.** The requirements asked for `metrics{f1}`; the project's existing `EvaluationMetrics` field is `f1_score`. The existing name won, so that the metadata is directly comparable with `evaluation_report.model_evaluations`. This is a deliberate, flagged deviation.
- **`metrics_source`** is not in the requested list and was added because without it a reader cannot tell whether `precision: null` is a Phase 6 limitation or a Phase 4 bug.
- **`preprocessing` is stored as a summary** (imputation strategy, scaling, encoding, excluded columns), not a serialized transformer. The transformer itself is inside the joblib file; duplicating it as JSON would be a second source of truth that could disagree with the one actually used.

`run_results.json` has 11 top-level keys: `baseline_models`, `dataset`, `evaluation`, `final_model`, `llm`, `notes`, `optimization`, `optimized_models`, `preprocessing`, `produced_by`, `schema_version`.

---

## 7. LangGraph Integration

### 7.1 Node placement

`GraphNode.MODEL_PERSISTENCE` was appended to `WORKFLOW_ORDER`, directly after the conditional optimization branch:

```
data_analysis -> preprocessing -> model_selection -> training -> evaluation
                                                          |            |
                                                          |            +--> optimization --> model_persistence --> END
                                                          |                     (skip / no candidates)
                                                          +--> model_persistence --> END
                                                                (optimization disabled, skipped or failed)
                                                          +--> END
                                                                (stop_requested)
```

Both exits of the optimization decision reach persistence. A run that skipped or failed the search still persists the best valid baseline, which is the whole point of the fallback in §4.1.

### 7.2 Structure, verified

| Configuration | Nodes | Edges |
|---|---|---|
| `--graph` | 9 | 15 |
| `--graph --llm` | 10 | 21 |

Both match `expected_structure()` exactly. The `enable_llm=True` build adds three nodes and six edges, unchanged from Phase 8 in count.

### 7.3 Persistence does not control the ML path

- **`routing.decide_optimization` is untouched.** No persistence field is read by it.
- **`MODEL_PERSISTENCE` is not in `REQUIRED_NODES`**, which stays at 5. A run without persistence is still a complete run.
- **`RoutingIndependenceTestCase` was extended, not relaxed.** `route_after_llm_evaluation` now returns `model_persistence` for a skip, and each test additionally asserts the route is *identical with and without* LLM results - a stronger statement of "the model cannot influence control flow" than the old `== END`.
- **`routing.DEGRADABLE_NODE_VALUES = ("optimization", "model_persistence")`.** Persistence failure yields `WorkflowStatus.PARTIAL`, not `FAILED`, because the machine learning result is still valid.
- **`non_retryable_error_types()` includes `PersistenceError`**, so a persistence failure is not retried - retrying a deterministic write failure would just repeat it.

### 7.4 One graph-shape consequence

Persistence is now the pipeline's last node, so the LLM layer rewires two more edges (`evaluation -> model_persistence`, `optimization -> model_persistence`) on top of Phase 7's two. `test_the_enabled_graph_keeps_every_phase_7_edge_that_still_exists` was updated to the four-edge set, and a new companion test asserts the complement - that no unconditional pipeline edge was removed.

---

## 8. Streamlit Dashboard

### 8.1 The split

```
dashboard.py                 Streamlit presentation only (481 lines)
tools/dashboard_tools.py     all logic, no Streamlit, no agent, no graph (1 194 lines)
        |
        v
tools/persistence_tools.py   the trusted loader
        |
        v
models/                      the one artifact this project wrote
```

`tools/dashboard_tools.py` imports neither `streamlit` nor any agent or graph module, which is asserted by AST inspection of the import list. That is what makes the logic testable without a browser and what makes it impossible for the dashboard to start a workflow.

`dashboard.py` imports neither an agent nor `graph`, is asserted by the same technique, and contains **zero** top-level call statements: `main()` is guarded by `if __name__ == "__main__":`, so `import dashboard` renders nothing. `st.set_page_config` was moved inside `main()` for the same reason - it is the first command of a *run*, not of an import.

### 8.2 The ten sections

Rendered and verified in a real browser against the multiclass artifact:

| # | Section | Source |
|---|---|---|
| 1 | Header - title, 7-stage subtitle, section index | `config` |
| 2 | System status | `results` + `metadata` |
| 3 | Dataset overview | `results["dataset"]` (Phase 2 record) |
| 4 | Model comparison | `results["baseline_models"]` + `["optimized_models"]` |
| 5 | Evaluation | `results["evaluation"]` + metadata metrics |
| 6 | Confusion matrix | `metadata["confusion_matrix"]`, or the note |
| 7 | Optimization | `results["optimization"]` |
| 8 | LLM interpretation | `results["llm"]` |
| 9 | Prediction | the saved pipeline |
| 10 | Model information | `metadata` |

The title is section 1 and the nine panels are numbered 2-10, so the numbering matches the specification exactly; the header prints the index so a reader arriving mid-page knows where they are.

### 8.3 Three defects found by rendering it

Unit tests could not have found these, because they only fail when Streamlit actually executes the script. Opening the page in a browser and rendering it headlessly with `streamlit.testing.v1.AppTest` both did:

1. **`st.bar_chart` does not accept `key`.** The page died with `TypeError: VegaChartsMixin.bar_chart() got an unexpected keyword argument 'key'`. `key` was removed. Verified against the installed signature: in Streamlit 1.65.0 `bar_chart`, `metric` and `table` have no `key` parameter, while `dataframe`, `selectbox`, `number_input`, `text_input` and `button` do.
2. **`render_optimization` assumed a `note` key** that the unavailable path did not return, so the page died with `KeyError: 'note'` when no artifact existed. The builder now always returns a note; the view uses `.get()` with a fallback in all three sections that share the pattern.
3. **The data cache leaked across renders.** `st.cache_data` with no arguments keys on the function's own source, so the first rendered test decided the state of every later one. `_render_from()` now clears the cache before each render.

Defect 3 was in the test I had just written, which is worth recording: the first version of this test class was order-dependent and would have been flaky.

---

## 9. Dashboard Features

- **Never trains.** Streamlit re-runs the script on every interaction; a dashboard that fitted anything would retrain a model on every widget click. Asserted three ways: `Pipeline.fit` is patched to raise during both loading and prediction; no training entry point is called; and the estimator arrives already fitted.
- **Never duplicates preprocessing.** The frame the form builds is passed straight to `state.model.pipeline.predict`. A test asserts the dashboard's answer equals a direct pipeline call for the same row - if the view had re-implemented imputation, scaling or encoding, that equality would fail for at least some inputs.
- **Differentiates BASELINE from OPTIMIZED** in both the table (`Kind` column) and the chart (`color="Kind"`). Support Vector Machine appears in both on the binary run, so the `Persisted` marker is keyed on (name, **kind**) and exactly one row is ever flagged.
- **Reports both trial counts.** `total_trials` (40, summed over completed models) and `trials_per_model` (20) are separate fields, because "40 trials" and "20 trials" are both true and one field could only be one of them.
- **Shows `-` for an unmeasured metric.** `format_metric(None)` returns `NOT_RECORDED`, never `0.0000`, because a zero reads as a measurement.
- **No `0.0` for an empty numeric box.** `build_prediction_frame` maps a missing numeric input to `NaN`, not `0.0`. For a column like `glucose` (training mean 132), zero is a fabricated measurement; `NaN` reaches the imputer the model was actually trained with.
- **Refuses an unseen category.** The encoder would silently encode an unknown level as all-zeros, so `run_prediction` checks every text field against the recorded options and returns a refusal naming the value and the allowed set.
- **Works with no Ollama.** Section 8 states that the layer is off by default, that the dashboard works without it, and that a language model cannot measure, train, validate, decide or control the workflow.
- **No absolute paths.** The artifact directory is rendered relative to the project root (`models`), falling back to the directory name for a directory outside the project.

---

## 10. Prediction Workflow

1. `build_prediction_fields(state)` reads `feature_columns`, `numerical_features`, `categorical_features`, `class_labels` and `dataset.numerical_statistics` from the saved documents.
2. Each field gets `kind` (`"number"` or `"text"`) from the **recorded** feature type, not from inspecting values. A `text` field carries the category options the dataset contained.
3. Each numeric default is the **recorded training mean**. A default of zero for a standardised column is a valid but meaningless row.
4. `build_prediction_frame(fields, values)` builds a one-row `DataFrame` with exactly the pipeline's input columns. The target column is deliberately **not** added - including it would make the pipeline predict a value already present in the input.
5. `run_prediction(state, fields, values)` calls `state.model.pipeline.predict(frame)` and, when `PersistedModel.supports_probability` is true, `predict_proba` as well.
6. `PredictionOutcome` returns `ok`, `predicted_class`, `probabilities` (a list of `{class, probability}` pairs, labelled because a bare number is ambiguous in a three-class task), `confidence`, `model_name`, `message` and `detail`.

Verified against the multiclass artifact: `Predicted class: gamma`, three labelled probabilities summing to 1.0, highest probability 1.0000 - and the dashboard's answer identical to a direct `pipeline.predict` call.

`supports_probability` is read from `pipeline.named_steps["model"]`. On the binary run the selected estimator is `SVC(C=14.566, kernel="linear", random_state=42)`, which has **no** `predict_proba`, so `supports_probability` is correctly `False`, `probabilities` is `None`, and the outcome carries a message saying why rather than showing a blank area. The predicted class comes from the same saved pipeline either way.

---

## 11. Security Considerations

**No uploaded model, anywhere.** Loading a pickle executes arbitrary code inside the Python process; a local dashboard cannot make that safe. Therefore:

- there is no `st.file_uploader` in the project - asserted on the **rendered** app (`list(app.file_uploader) == []`), not only on the source;
- `joblib.load` is called from **one function in one file**, `tools/persistence_tools.load_pipeline`. A test parses every `.py` file in the project with `ast` and fails if any other module calls it;
- `load_persisted_model(output_dir)` derives every path from `config`. Its only parameter names a **directory**, never a model file, so there is no argument through which a caller could point the project at an arbitrary file;
- `validate_schema(payload, *, name)` checks `schema_version` and raises `ArtifactReadError`, so a file that is a valid pickle of something else is rejected with a reason.

**Nothing sensitive is persisted.** No dataset rows, no credentials, no environment variables, no API keys. The metadata holds names, numbers and library versions. The dashboard shows no secret-bearing field, asserted over the rendered model-information rows.

**Local only.** `OLLAMA_ENABLED = False` by default. `LLM_ALLOWED_HOSTS` is unchanged and `require_local_url` still rejects anything non-loopback. No project data leaves the machine at any point.

---

## 12. Testing

### 12.1 Counts

| Module | Test methods |
|---|---|
| `tests/test_persistence_tools.py` (new) | 107 |
| `tests/test_dashboard.py` (new) | 96 |
| **Phase 9 total** | **203** |
| 21 pre-existing modules | 1 438 |
| 2 added to `tests/test_llm_integration.py` (see §13.2) | 2 |
| **Project total** | **1 643** across 23 modules |

Collected result: **1 677 passed** (test methods plus the subtests pytest counts separately), 0 failed, 6 warnings.

### 12.2 Persistence coverage

Selection: optimized winner chosen; baseline chosen when the tuned winner is removed; baseline chosen when the tuned winner carries no hold-out F1; both candidates recorded so the choice is auditable.

Final fit: reuse strategy returns the identical object; rebuild strategy records the training-split size and asserts `train_samples + test_samples < rows`, so train+test can never be fitted together; strategy choice follows the selection source.

Metadata: all required keys present; `JSON`-serialisable; no `NaN` or `Infinity`; absent values are `null`, not `0`; `metrics_source` correct per selection source; `confusion_matrix` present for a baseline and `null` + note for a tuned model; `best_params` are the tuned parameters for an optimized selection.

Write/reload: save → load → predict round trip; the reloaded pipeline is the tuned one; metadata survives the round trip; `artifacts_exist` true for model + metadata; a failure on the report bundle still leaves a usable model and an absent `run_results.json`; atomic replace exercised.

Security: `joblib.load` appears in exactly one module (AST); the loader returns `None` for a missing artifact; invalid metadata raises `ArtifactReadError`.

Invariants: both dataset SHA-256 hashes unchanged after the whole suite.

Test isolation: a guard covers the project's `models/` directory for the whole session, detects a created, deleted or overwritten file, treats a missing directory as valid, is proven to be installed, and a static check rejects any `run_workflow` call in the suite that would write there. See §13.1.

### 12.3 Dashboard coverage

Module boundaries (AST): the helper imports no `streamlit`, no agent, no graph; the view imports no agent, no graph; the view never calls `joblib.load`; the view has no `file_uploader`; the view has no top-level calls and does have a `__main__` guard.

Loading: with an artifact, without one, and with a damaged one - three distinct outcomes. Every section reports `available=False` when there is nothing. The prediction form has no fields and a prediction is refused with a reason.

Sections, for **both** selection sources and **both** datasets: status labels, dataset record, comparison table kinds and the single persisted row, chart grouping, evaluation metrics and their provenance, the generalisation gap text, confusion matrix presence or its explanatory note, optimization performed or not, the LLM section without Ollama, model information with no path and no secret.

Prediction: one field per feature; numeric and text told apart; category options recorded; defaults equal the recorded means; frame columns equal the pipeline's; empty input becomes `NaN`; the dashboard's answer equals a direct pipeline call; probabilities labelled and summing to 1; confidence is the highest probability; an unseen category is refused; an empty form is refused.

**Rendering (`streamlit.testing.v1.AppTest`)** - the eight tests that unit tests could not replace: the page renders with a persisted artifact and with none; the title and all nine numbered subheaders; the subtitle names all seven stages; no file uploader on the rendered app; one input per persisted feature; clicking Predict produces a prediction with `Pipeline.fit` patched to raise; the no-upload explanation is visible on the page.

### 12.4 A note on skips

Nine conditional skips exist across the two new modules, plus the two that were already in the suite:

| Module | Count | Why |
|---|---|---|
| `tests/test_persistence_tools.py` | 5 | Skip when the run selected the branch the test is *not* about: baseline won so no pipeline was rebuilt; the estimator exposes `predict_proba` so a no-probability expectation is inapplicable; an optimized model was selected so no confusion matrix exists |
| `tests/test_dashboard.py` | 3 | The same three conditions, at the dashboard layer |
| both | 1 | `streamlit.testing.v1` is not importable |

**None of them exist to turn a failure green.** A skip is emitted only when the artifact the test loaded made the premise false - e.g. asserting that probabilities are absent when the chosen estimator has `predict_proba`. Every one of these conditions is covered on **both** branches by a separate test, so no branch of the code is left unverified. The eight `AppTest` tests have no skip on the happy path; only the import guard is guarded.

The same audit over the 21 pre-existing modules finds 2 skips, both from before Phase 9.

---

## 13. Regression Test Results

```
$ pytest -q
1677 passed, 6 warnings, 1364 subtests passed in 944.99s (0:15:44)
```

- **No test was deleted, no skip was added to force green, no assertion was weakened.**
- 23 test modules; 1 440 pre-existing test methods, of which three had their assertion updated and two were added (§13.2).
- `models/` was byte-identical before and after the run, enforced by the session guard described in §13.1. Both dataset SHA-256 hashes are unchanged.
- Warnings are the six pre-existing ones: five `UserWarning` from `sklearn.model_selection` about a class with only 2 members against `n_splits=3`, and one `DeprecationWarning` from a `subTest` returning a non-`None` value. None are new.
- Runtime is dominated by the real Optuna searches and real dataset runs in the fixture setups, not by the assertions.

### 13.1 Pre-existing tests that had to change, and why

Three failures, all of the same kind: a Phase 8 assertion that encoded the Phase 7 graph shape.

### 13.1 A test-isolation defect Phase 9 introduced, and the guard against it

Turning persistence on by default had a consequence nobody checked for: **the test suite was overwriting the project's own artifacts.**

Eighteen `run_workflow` calls in `tests/test_llm_integration.py` drive the *real* agents on the *real* datasets. Those tests exist to prove that the optional LLM layer changes nothing, so they were written before any artifact existed to damage. With `PERSISTENCE_ENABLED = True`, each one wrote to `models/`, replacing the optimized Support Vector Machine a real `--graph` run produces with a baseline Logistic Regression from a degraded `enable_optimization=False` run.

The suite still reported success. Nothing failed, and the substitution was invisible: same three filenames, plausible sizes, and a metadata document describing a different model than the one that had been there.

**The fix, in two parts.**

1. **Each affected call now says what it wants.** All eighteen calls take `enable_persistence=False`. They are tests of the LLM layer; persistence is not their subject, and scoping them correctly also makes them faster.
2. **A session guard makes the class of mistake loud.** `tests/conftest.py` hashes every file in `models/` before the first test and again after the last one. Any file created, deleted or overwritten fails the session with a non-zero exit status and a message naming the files, even when every individual test passed.

The guard is proved to work rather than assumed to: with the keyword removed from one real-agent call, that test still passed, and the session failed with

```
ERROR: models/ was modified while the tests ran:
    - best_model.joblib: overwritten by a test
    - best_model_metadata.json: overwritten by a test
    - run_results.json: overwritten by a test
A test ran the real persistence stage without being given its own output directory.
Pass output_dir= or enable_persistence=False to run_workflow / PersistenceAgent.run in that test.
```

`tests/test_persistence_tools.py::TestProjectArtifactIsolationGuard` covers the guard itself (six tests) and adds a static check that fails if any future `run_workflow` call in the suite omits both `enable_persistence` and `persistence_output_dir` while running unstubbed agents. The static check is what points at the offending line; the session guard is what catches the case where nobody thought to write such a check.

Both dataset SHA-256 hashes are unchanged after the whole suite.

### 13.2 Pre-existing tests that had to change, and why

Three failures, all of the same kind: a Phase 8 assertion that encoded the Phase 7 graph shape.

| Test | Old assertion | New assertion |
|---|---|---|
| `test_the_enabled_graph_keeps_every_phase_7_edge_that_still_exists` | exactly 2 edges removed by `--llm` | 4 edges - the 2 Phase 7 rewires plus the 2 that now lead into the pipeline's new last node |
| `test_a_skip_decision_is_unchanged_by_llm_results` | skip routes to `END` | skip routes to `model_persistence`, **and** the route equals the route of the same state without LLM results |
| `test_a_disabled_run_cannot_be_reopened_by_an_llm_result` | disabled routes to `END` | disabled routes to `model_persistence`, **and** again equals the no-LLM route |

The behaviour under test - that a language model cannot change control flow - is unchanged and now asserted more strongly. No metric, threshold or ranking moved.

---

## 14. Binary Dataset Results

`datasets/sample_classification.csv` - 123 rows, 9 columns, target `Outcome`, 5 numerical / 3 categorical candidate features, 17 missing values, 3 duplicate rows, class 0 = 47 (38.21%) / class 1 = 76 (61.79%).

### 14.1 Baseline ranking (Phase 4/5)

| Rank | Model | Hold-out F1 | CV F1 mean | Hold-out accuracy |
|---|---|---|---|---|
| 1 | Logistic Regression | 0.8761 | 0.8941 | 0.8750 |
| 2 | Support Vector Machine | 0.8357 | 0.8278 | 0.8333 |
| 3 | K-Nearest Neighbors | 0.7935 | 0.7960 | 0.7917 |
| 4 | Decision Tree | 0.7536 | 0.7702 | 0.7500 |
| 5 | Random Forest | 0.7500 | 0.8519 | 0.7500 |

### 14.2 Optimization (Phase 6)

Candidates: Logistic Regression, Support Vector Machine. Objective `f1_weighted`, TPE sampler, `sampler_seed=42`, 20 trials per model, 5-fold CV on the training split.

| Model | Baseline F1 | Optimized F1 | Improvement | Best trial | Best parameters |
|---|---|---|---|---|---|
| Logistic Regression | 0.8761 | 0.9167 | +0.0106 | 15/20 | `C=20.8756, solver=lbfgs` |
| Support Vector Machine | 0.8357 | 0.9578 | +0.0767 | 18/20 | `kernel=linear, C=14.5661` |

### 14.3 Persistence

| Field | Value |
|---|---|
| Workflow status | `completed` |
| Persistence status | `completed` |
| Model | Support Vector Machine (`kernel_based`) |
| `selection_source` | `optimized` |
| `metrics_source` | `optimization_holdout` |
| Hold-out metrics | accuracy 0.958333, precision `null`, recall `null`, `f1_score` 0.95778 |
| CV metrics | `cv_f1_mean` 0.904579, `cv_accuracy_mean` 0.905789, 5 folds |
| `best_params` | `{"kernel": "linear", "C": 14.566096837677359}` |
| `final_fit.strategy` | `rebuilt_and_refitted_pipeline` |
| `final_fit.train_samples` | 96 (of 96 train / 24 test) |
| Features | 7 (`record_id` excluded, reason `identifier_like`) |
| Class labels | `["0", "1"]` |
| Artifact size | 3 268 bytes |
| `library_versions` | scikit-learn 1.8.0, joblib 1.5.3 |
| Pipeline steps | `["preprocessor", "model"]` |

Artifacts written: `best_model.joblib`, `best_model_metadata.json`, `run_results.json`.

Identical to the Phase 8 result, as expected: persistence is deterministic and reads only state that Phase 6 already produced.

---

## 15. Multiclass Dataset Results

`datasets/sample_multiclass.csv` - 92 rows, 7 columns, target `class_label`, 3 numerical / 3 categorical candidate features, classes alpha 30 (32.61%) / beta 31 (33.70%) / gamma 31 (33.70%), `sample_id` excluded as `identifier_like`.

### 15.1 Baseline ranking (Phase 4/5)

| Rank | Model | Hold-out F1 | Hold-out accuracy | CV F1 mean |
|---|---|---|---|---|
| 1 | Decision Tree | 1.0000 | 1.0000 | 1.0000 |
| 2 | Random Forest | 1.0000 | 1.0000 | 1.0000 |
| 3 | Logistic Regression | 0.8857 | 0.8889 | 0.8566 |
| 4 | Support Vector Machine | 0.8857 | 0.8889 | 0.8543 |
| 5 | K-Nearest Neighbors | 0.7641 | 0.7778 | 0.7116 |

Logistic Regression and Support Vector Machine tie on both hold-out F1 and hold-out accuracy; the name-ascending final key of the existing rule separates them. That tie-break is Phase 5's, not Phase 9's.

### 15.2 Optimization (Phase 6)

| Model | Baseline F1 | Optimized F1 | Improvement | Best trial | Best parameters |
|---|---|---|---|---|---|
| Decision Tree | 1.0000 | 1.0000 | 0.0000 | 0/20 | `max_depth=8, min_samples_split=20, min_samples_leaf=8, criterion=gini` |
| Random Forest | 1.0000 | 1.0000 | 0.0000 | 0/20 | `n_estimators=80, max_depth=None, min_samples_split=18, min_samples_leaf=7, max_features=None` |

The improvement is exactly `0.0000`, not approximately: the hold-out split is trivially separable, so every configuration reaches F1 1.0 and the best trial is trial 0 - the sampler's first suggestion. This is the honest result and it is reported as such rather than presented as a tuning success.

### 15.3 Persistence

| Field | Value |
|---|---|
| Model | Decision Tree (`tree_based`) |
| `selection_source` | `optimized` |
| `metrics_source` | `optimization_holdout` |
| Hold-out metrics | accuracy 1.0, precision `null`, recall `null`, `f1_score` 1.0 |
| CV metrics | `cv_f1_mean` 1.0, `cv_accuracy_mean` 1.0, 5 folds |
| `final_fit.strategy` | `rebuilt_and_refitted_pipeline` |
| `final_fit.train_samples` | 72 (of 72 train / 18 test) |
| Features | 5 |
| Class labels | `["alpha", "beta", "gamma"]` |
| Artifact size | 1 938 bytes |
| `supports_probability` | `True` |

The baseline and the tuned candidate are the **same model** (Decision Tree). Because a usable tuned candidate exists, the rule selects the optimized source and the artifact is rebuilt from the tuned parameters - which happen to equal the baseline defaults, since the search could not improve on a perfect score.

### 15.4 End-to-end verification

The reloaded artifact predicted **92 / 92** rows of the dataset correctly, and the dashboard path agreed with a direct `pipeline.predict` call on rows 0, 17 and 41.

---

## 16. Limitations

1. **Precision, recall and a confusion matrix are `null` for a tuned model.** Phase 6 does not record them, and persistence does not re-score. A user wanting those numbers for a tuned model gets them only for the baseline, and the dashboard says so. Filling them in would require a second evaluation.
2. **The dashboard depends on a completed run.** It reads artifacts; it cannot produce them. The no-artifact page is informative but has nothing to show, by design.
3. **The sequential path does not persist.** `--graph` is required. This is pre-existing Phase 8 behaviour and was left unchanged rather than quietly altered.
4. **One artifact at a time.** `models/` holds the last run. There is no registry, no versioning and no run history; `run_results.json` is overwritten. Experiment tracking (MLflow or similar) is Phase 10 work.
5. **The artifact is not portable across library versions.** `library_versions` records scikit-learn and joblib, but nothing enforces compatibility. A pickle written by a different scikit-learn may fail to load, and the error is reported rather than handled.
6. **`predict_proba` depends on the estimator.** `SVC` does not expose it. The binary run therefore shows a predicted class and no probabilities, which is correct but less informative than the multiclass run.
7. **The metrics describe the split that produced them.** One `random_state=42` split of a 123-row dataset. The generalisation gap is reported with a disclaimer because its size on one small split is not evidence of overfitting.
8. **No performance data on the dashboard's own responsiveness.** Artifact size is recorded (3 268 bytes) and is small, but no load-time benchmark was taken.
9. **Ollama was not available on this machine**, so the LLM section was validated along the *unavailable* path in both the workflow and the dashboard. The success path is covered by existing Phase 8 tests with a stubbed client, not by a live model.
10. **The render tests depend on Streamlit's `AppTest`.** If that API changes, `test_the_page_renders_*` fail for environmental reasons. That is the correct trade: a view bug should be a red test.
11. **The isolation guard protects `models/` only.** It hashes the artifact directory and nothing else, and it detects the change after the fact rather than preventing it. A test that wrote to `datasets/` or to a previous phase's report would not be caught by it - only by the dataset hash assertions that already exist.
12. **A test can still be slow for the wrong reason.** Several tests build a real Optuna search in `setUp` because that is the only honest way to test the optimized path. The suite's runtime is therefore dominated by real training, which is deliberate but means a red test can take minutes to appear.

---

## 17. Phase 10 Recommendation

**Experiment tracking, and making the artifact a first-class object rather than the newest file in a folder.**

Concretely, in priority order:

1. **A run registry.** `run_results.json` is currently overwritten by the next run. Persist one immutable document per run with a timestamp, dataset hash, git-free content digest and configuration snapshot, and let the dashboard select between runs. This is the single highest-value addition, because it makes a result *reproducible and comparable* rather than merely *present*.
2. **Metrics across runs.** With a registry, comparing hold-out F1 for the same dataset across configurations becomes a stored, queryable fact instead of a copy-paste.
3. **Close the metric gap honestly.** Reuse Phase 4's evaluation machinery to score the *tuned* pipeline on the hold-out split **once**, recording it in a clearly separate namespace from Phase 6's search metrics. This would remove limitation 1. It must not feed back into selection, tuning or any other decision.
4. **Artifact versioning and compatibility.** A recorded `schema_version` already exists; a loader that checks the scikit-learn version and refuses politely would remove limitation 5.
5. **Dataset coverage.** Both fixtures are synthetic and small, and the multiclass one is perfectly separable. A noisier, more realistic dataset would exercise the evaluation, optimization and persistence paths far harder than any of them currently are.

**Not recommended for Phase 10:** a real LLM dependency, any cloud service, or any authentication layer. All three would break the property the project has maintained since Phase 1 - that the whole thing runs locally, offline, for free, with no LLM installed.