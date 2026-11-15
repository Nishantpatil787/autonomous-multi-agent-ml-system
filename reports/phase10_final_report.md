# Phase 10 Final Report: Local MLflow Experiment Tracking

## 1. Objective

Make every `--graph` workflow run **recorded, comparable and inspectable** on the
user's own machine, without changing a single machine-learning decision.

Phase 10 adds a pure observability layer over the Phase 1-9 pipeline:

- every `--graph` run is recorded into a **local MLflow tracking store**
  (SQLite, `experiments/tracking/mlflow.db`), created and written only by a run,
- the Phase 9 dashboard gains an **Experiment Tracking section** that reads that
  store, and
- the pipeline's results, artifact and dataset are provably unchanged by the
  tracking layer's presence or absence.

The phase's governing rule, enforced in code and asserted by the test suite:
**MLflow records what the agents already decided; it never decides anything.**

## 2. Scope and Non-Goals

In scope:

1. A dedicated `tools/mlflow_tools.py` module (25 exported names) containing the
   whole tracking layer: `TrackingError` (the single exception type),
   `TrackingStatus` (`DISABLED`, `UNAVAILABLE`, `RECORDED`, `FAILED`),
   `TrackedRun`, `track_workflow_run`, and the read side
   (`experiment_summary`, `recent_runs`, `tracking_configuration`).
2. A wrapper around `run_workflow` in `graph/workflow.py` that opens one parent
   run per workflow and records the finished state after the graph executes.
3. The `--no-mlflow` CLI switch and the `MLFLOW_ENABLED` environment
   switch (`.env.example` documents it).
4. The dashboard's section 11, *Experiment Tracking*.
5. Test isolation that keeps the suite out of the user's real store, a session
   guard over `experiments/tracking/`, and 174 new tests in
   `tests/test_mlflow_tools.py`.

Explicitly out of scope:

- **No graph node was added.** Tracking cannot be a node: a node runs after the
  other stages and could not open a run that spans the whole workflow. The seven
  nodes and `WORKFLOW_ORDER` from Phases 7-9 are untouched
  (`test_no_graph_node_was_added_for_tracking` pins this).
- **No cloud, no server, no UI input.** The store is a local SQLite file whose
  path comes only from `utils/config.py`. The dashboard accepts no MLflow URI.
- **No second model artifact.** `models/best_model.joblib` remains the primary
  artifact; MLflow stores a run reference and recorded decisions.
- **No retraining from the dashboard.** Section 11 only reads recorded runs.

## 3. Architecture

```
app.py --graph  ──►  run_workflow(...)  graph/workflow.py
                       │  wrapper:
                       │    TrackedRun.start()     (one parent run)
                       │    build_workflow_result()
                       │    track_workflow_run()   (logs finished state)
                       ▼
                    WorkflowResult (unchanged)
                 ┌──┴─────────────────────────────┐
                 ▼                                 ▼
            tools/mlflow_tools.py            experiments/tracking/mlflow.db
            (no imports from agents/ or            (SQLite, one experiment:
             graph/ — tested)                      autonomous-multi-agent-ml)
```

The wrapper opens the run before the graph executes and records *after* it
completes, and returns the exact `WorkflowResult` the graph produced. When
tracking is requested it is the **caller's request AND the configuration**:
`run_workflow`'s call site passes `enable_tracking = bool(enable_tracking) and
bool(config.MLFLOW_ENABLED)`, so `--no-mlflow` and `MLFLOW_ENABLED=false` are
independent switches and either one alone turns tracking off.

The module boundary is enforced by tests: `tools/mlflow_tools.py` imports
nothing from `agents/` or `graph/`, and `tools/dashboard_tools.py` remains
MLflow-free (asserted by parsing its source).

## 4. The Tracking Store and Configuration

| Setting | Value |
|---|---|
| Store URI (default) | `sqlite:///experiments/tracking/mlflow.db` |
| Experiment name | `autonomous-multi-agent-ml` |
| MLflow version | `3.16.1` in this environment; `requirements.txt` pins `mlflow>=3.0.0,<4.0.0` |
| Env override | `MLFLOW_TRACKING_URI` (used for the failure-isolation validation only) |
| Enable switch | `MLFLOW_ENABLED` (env / `.env`), default `true` |
| Agent hint | `MLFLOW_DISABLE_AGENT_HINT=1` set via `os.environ.setdefault` before any MLflow import |
| Model artifact logging | `MLFLOW_LOG_MODEL_ARTIFACT=false` — `models/best_model.joblib` is the single artifact |

`utils/config.py` publishes the `MLFLOW_*` block. Config-structure tests pin the
local-only property (`sqlite`, never a remote URI), that the store lives under
`experiments/tracking/` (never under `models/`), and that the URI keeps the
Windows drive letter.

Value-length guards match both MLflow's constraints and the project's
"explicit errors" convention: params are truncated at 6000 characters, tag
values at 8000, and entity keys at 250. Empty-string parameters are dropped
rather than recorded as noise. `optimization.baseline_model_name` is recorded
instead of a `repr` of a candidate object, keeping the store JSON-stable.

## 5. The Recorded Schema

A run records the workflow's *decisions as data*. Everything below was read back
from the two real canonical validation runs (section 16/17), not invented.

**Params** (namespaced by stage):

- `dataset.*` — name, SHA-256 digest, rows, columns, task type, target column
  and how it was detected, missing values, duplicate rows, used feature count.
  The rows themselves are never stored.
- `preprocessing.*` — encoding, imputations, scaling, excluded features.
- `evaluation.*` — ranking rule, selection reason, selected model, split sizes.
- `optimization.*` — sampler (`tpe`), `sampler_seed=42`, per-model trial count,
  `best_model`, `best_params`, `best_trial_number`, searched/skipped lists.
- `final_model.*` — name, type, `selection_source` (`optimized`/`baseline`),
  `optimization_used`, best parameters, baseline model name.
- `persistence.*` — status, artifact names (`best_model.joblib`,
  `best_model_metadata.json`, `run_results.json`), `schema_version`,
  `final_fit_strategy` (`rebuilt_and_refitted_pipeline`).
- Capability params (`comparison.*`, `*_enabled`, `execution_mode`, ...).

**Metrics** (raw float values, MLflow-native):

- `evaluation.holdout_f1_score`, `evaluation.holdout_accuracy`,
  `evaluation.holdout_precision`, `evaluation.holdout_recall`,
  `evaluation.cv_f1_mean`, `evaluation.cv_accuracy_mean`, the generalisation gap
  family.
- `final_model.holdout_f1_score`, `final_model.holdout_accuracy`,
  `final_model.improvement`, `final_model.optimized_f1_score`.
- `optimization.holdout_f1_score`, `optimization.improvement`,
  `optimization.best_trial_number`, `optimization.time_seconds`.
- `timing.node.<each node>.seconds` — computed from `NodeRecord.duration`,
  `timing.node_count=7`, `timing.node_sum_seconds`, `timing.workflow.total_seconds`.

**Tags** — `workflow_status` (the exact workflow enum value, e.g. `completed`),
`phase=10`, `project`, `task_type`, `dataset`, `execution_mode=graph`,
`llm_enabled`, `optimization_enabled`, `persistence_enabled`, `stopped_early`,
`total_retries`, `workflow=langgraph`.

**Artifacts** (JSON, per run): `dataset_summary.json`,
`preprocessing_config.json`, `model_comparison.json`,
`optimization_trials.json`, `confusion_matrix.json`, `reproducibility.json`.
These reuse the existing `FinalModelSelection`/`PersistenceReport`/evaluation
records rather than recomputing anything.

## 6. The Governing Invariant: MLflow Never Decides

The wrapper hands the graph's `WorkflowResult` back unchanged, and the read side
of the whole pipeline is untouched: no agent, no routing decision, no metric and
no artifact reads the store. Per-node timings are read from `NodeRecord.duration`
(the workflow's own record), not re-measured.

The proof is in the tests and in two independent real runs:

1. `tests/test_workflow.py` and `tests/test_llm_integration.py` fingerprint
   workflow results with and without the layer; the digests of the `run_results`
   structure are equal.
2. A real CLI run with tracking on and the same command with `--no-mlflow`
   produced byte-identical `run_results.json` digests (after normalising the
   wall-clock timings that appear inside observation strings — those are of
   course timing noise, not workflow state).
3. The recorded runs show `phase=10`, `workflow=langgraph`, `execution_mode=graph`
   and the exact Phase 9 metrics (section 16/17) — the store observed the run; it
   did not create it.

## 7. Run Status Mirroring

The run's MLflow status is a **mapping of the workflow status, recorded
explicitly**:

| Workflow status | MLflow status | `workflow_status` tag |
|---|---|---|
| `completed` | `FINISHED` | `completed` |
| `partial` | `FINISHED` | `partial` |
| `failed` | `FAILED` | `failed` |

The mapping is an internal function, logged through `TrackedRun.end()`, and the
*exact* workflow enum value is always kept as the tag so no information is lost
in the mapping. Validated on a real failing run: `--graph --dataset
does_not_exist.csv` recorded a run with `workflow_status=failed` and
`mlflow_status=FAILED` (section 18).

## 8. Failure Isolation

**MLflow failing never breaks the pipeline.** The wrapper catches every tracking
exception, logs `type(error).__name__` and the message under
`config.MLFLOW_LOG_PREFIX`, and returns the workflow result untouched. The
workflow is never inside the tracking try/except.

Validated end-to-end with a *genuinely* failing store: `sqlite:///NUL/...`
(the Windows `NUL` device cannot be a directory). The run emitted:

```
WARNING [MLFLOW] the tracking store could not be read: FileNotFoundError:
[WinError 3] The system cannot find the path specified: 'NUL\denied'
Experiment tracking is enabled but this run was not recorded; see the log for the reason.
```

The workflow exited `0`, reported `Workflow status: completed`, `Failed node: none`,
and produced the identical SVM result. A first attempt with a merely-non-existent
path (`C:/definitely/not/a/real/dir/mlflow.db`) was *not* a failure — MLflow
created the directory and recorded the run — which is itself reported here as a
finding: MLflow creates its parent directories, so a "won't create" assumption
must never be relied on. The scratch directory was removed afterwards.

`TestWorkflowStillSucceeds` in the test suite covers the same property with the
tracker itself raising mid-run.

## 9. Reproducibility Guard

- **Version whitelist.** `config.MLFLOW_VERSION_WHITELIST` is recorded with each
  run; the requirements pin is bounded (`mlflow>=3.0.0,<4.0.0`), and a test
  asserts the installed version is inside the permitted range. A future major
  release cannot silently change how a recorded run reads back.
- **Seed provenance.** `optimization.sampler_seed=42` and the Optuna trial list
  are recorded, so the recorded "best trial" can be checked against the pipeline's
  own `random_state=42` determinism.
- **Recorded inputs.** The dataset SHA-256 is stored with the run; the two
  canonical dataset hashes are themselves re-verified after every validation run
  (section 18).

## 10. CLI Integration

`app.py` gained `--no-mlflow`:

```bash
python app.py --graph                 # run and record
python app.py --graph --no-mlflow     # run, record nothing
```

The footer notice branches so the user always knows *why*:

- recorded → `Recorded in MLflow experiment 'autonomous-multi-agent-ml' (run <id>, status FINISHED).`
- `--no-mlflow` → `Experiment tracking skipped (--no-mlflow). Every result above was computed and reported exactly as it would have been.`
- `MLFLOW_ENABLED=false` → `Experiment tracking disabled by configuration (MLFLOW_ENABLED). ...`
- tracking requested but not recorded → `Experiment tracking is enabled but this run was not recorded; see the log for the reason.`
- package not installed → `MLflow experiment tracking is unavailable (the package is not installed). Every result above is unaffected.`

The notice reads the store with `mlflow_tools.experiment_summary()` inside a
`try/except`, so a notice can never break the report. The banner names the
current phase: `Project Status: Phase 10 - local MLflow experiment tracking`.

## 11. Dashboard Section 11

`dashboard.py` renders `st.subheader("11. Experiment tracking")` via
`render_experiment_tracking()`. The section:

- reads the store through `experiment_summary()` wrapped in `@st.cache_data`
  with a `try/except` so a missing/corrupt store renders *"No experiments have
  been recorded…"* instead of crashing the page,
- shows the latest run (experiment, short run id, MLflow status, workflow
  status, start time, store URI, selected model, F1, accuracy, dataset,
  optimization status),
- renders a history table of recent runs and the six recorded JSON artifacts,
- uses `tools/dashboard_tools.build_experiment_section`, which stays MLflow-free
  (AST-verified) so dashboard logic remains testable without the dependency,
- accepts **no MLflow URI input** — the store path comes only from
  `utils/config.py`.

The dashboard never trains: section 11 is a pure read of the store, and cached
so a rerun of the dashboard never retrains anything.

## 12. Test Isolation for the Real Store

Phase 9's lesson — *a test that writes into the project's artifacts is wrong even
when it passes* — applied again, to the store:

1. **Every call site opts out explicitly.** The 63 non-tracking `run_workflow`
   calls in `tests/test_workflow.py`, `tests/test_llm_integration.py` and
   `tests/test_dashboard.py` now pass `enable_tracking=False` (mirroring the
   existing `enable_persistence=False` idiom), and the CLI tests' `base()`
   helper adds `--no-mlflow`. Only the two tracking-integration tests in
   `tests/test_mlflow_tools.py` leave tracking on, and each points at its own
   temporary store.
2. **A session guard makes the mistake loud.** `tests/conftest.py` now builds a
   second `ProjectArtifactIsolationGuard` over `experiments/tracking/` alongside
   the Phase 9 guard over `models/`. Both directories are hashed before the first
   test and again after the last; any change fails the session with a non-zero
   exit and a message naming the files, even if every test passed.

Both guards are covered by `TestProjectArtifactIsolationGuard` in
`tests/test_persistence_tools.py` (which also carries the static check that no
unstubbed `run_workflow` call omits its isolation arguments).

## 13. Defect Found — the Suite Was Polluting the Real Tracking Store

**Symptom.** The store grew from 7 legitimate runs to 96, then 596, while
developing this phase. The dashboard's "recent runs" table was filling with stub
runs over the non-existent `stub.csv` fixtures — test data, not experiments.

**Root cause.** Phase 10 turned tracking on in the default workflow. The Phase 8
integration tests — written before any store existed — each recorded a run into
the *user's* `experiments/tracking/mlflow.db` on every call. Nothing failed, and
the substitution was invisible: MLflow happily recorded whatever it was given.

**Fix.** Section 12: explicit `enable_tracking=False` at every non-tracking call
site (works under every runner, including `python -m unittest discover`, which
ignores `conftest.py`) plus the session guard. With the keyword removed from one
real-agent call, the guard's message names the files exactly as the Phase 9
guard does.

**Cleaning the real store.** After the fix landed, `experiments/tracking/`
was reset and rebuilt with only the two canonical validation runs (section 16/17).
The delivered store contains exactly those two runs.

## 14. Defect Found — a Pre-existing Patcher Leak in test_workflow.py

**Symptom.** Running `tests/test_workflow.py` *before* `tests/test_llm_integration.py`
failed the five real-dataset LLM tests with `TrainingToolError: no models`, even
though both files were green in the default (alphabetical) order and in
isolation.

**Root cause.** A pre-existing, latent defect in
`test_report_shows_the_error_for_a_failed_run` (Phase 9 code, untouched by this
phase's feature work):

```python
self._patcher.stop()                      # retire the setUp patcher
self._patcher = patch_agents(training={"error": TrainingToolError("no models")})
result = run_workflow(...)
```

`setUp` registered the *first* patcher for cleanup with `addCleanup`, but the
*reassigned* second patcher was never registered. When the test ended, cleanup
stopped the already-stopped first patcher, and the second patcher — a stubbed
`TrainingAgent` that fails with `no models` on every call — survived for the
rest of the session. Alphabetical order put `test_workflow.py` last, so the leak
was never bitten under `pytest -q`; any other order exposed it, and it corrupted
unrelated tests into meaningless "training failed" runs.

**Fix (one line, in the test itself).** After the reassignment:

```python
self.addCleanup(self._patcher.stop)   # the replacement must retire too
```

**Validation.** With the fix, `pytest tests/test_workflow.py
tests/test_llm_integration.py` passes (143 tests, 376 subtests — the previously
failing order), and the default-order suite is unaffected. This is the first
pre-existing test-hygiene defect Phase 10 found and repaired rather than routed
around.

## 15. Testing

All numbers below come from the final full suite run, executed from the project
root in alphabetical order after every change in this phase landed.

```
1884 passed, 7 warnings, 1369 subtests passed in 396.96s (0:06:36)
```

- **No test was deleted; no skip was added to force green; no assertion was weakened.**
- 174 tests were added in `tests/test_mlflow_tools.py` (steps 1-30 of the phase
  specification, including structure, configuration, the recorded schema,
  failure isolation, the read side, the dashboard source lint, and the
  no-store-write/no-models-touch properties).
- Two pre-existing assertions were updated to name the current phase (section 15.1).
- `models/` and `experiments/tracking/` were byte-identical before and after the
  run, enforced by the two session guards (§12).
- The store held exactly the same two recorded canonical runs before and after
  the suite (`RUN_COUNT` 2 → 2): the suite wrote nothing to the user's store.

### 15.1 Pre-existing tests that had to change, and why

| Test | Change |
|---|---|
| `test_llm_integration.py::test_the_sequential_path_is_unchanged_by_the_new_flag` | banner assertion `Phase 9` → `Phase 10` (the test's own docstring says each phase updates it) |
| `test_dashboard.py::test_the_page_shows_the_title_and_all_ten_sections` | renamed `..._all_eleven_sections`; expected subheader list gains `"11. Experiment tracking"` |

Both are documentation of the current phase, not behavioural changes: no ML
result, metric, threshold or ranking moved.

## 16. Binary Dataset Validation

Canonical run `0e99fb29be9e4e2f9eeacb6d7cff663a` —
`python app.py --graph` on `datasets/sample_classification.csv`:

| Field | Recorded value |
|---|---|
| Dataset | `sample_classification.csv`, 123 rows, 9 columns, `dataset.sha256` `49ea0e2964cf…fd23c` |
| Workflow / MLflow status | `completed` / `FINISHED` |
| Best baseline | Logistic Regression, F1 0.8761, accuracy 0.8750 |
| Best optimized | **Support Vector Machine 0.9578** (trial 18/20, `kernel=linear, C=14.566096837677359)` |
| Final model | Support Vector Machine, `selection_source=optimized`, `final_fit_strategy=rebuilt_and_refitted_pipeline` |
| Recorded metrics | `final_model.holdout_f1_score` 0.95778, `final_model.holdout_accuracy` 0.958333 |
| Artifacts | `best_model.joblib`, `best_model_metadata.json`, `run_results.json` written to `models/` |

Identical to the Phase 9 binary result (§18).

## 17. Multiclass Dataset Validation

Canonical run `7a79111cecfd41b39ceb2f19ed047fa2` —
`python app.py --graph --dataset datasets\sample_multiclass.csv`:

| Field | Recorded value |
|---|---|
| Dataset | `sample_multiclass.csv`, 92 rows, 7 columns, `dataset.sha256` `0b70768d76af…539a8` |
| Workflow / MLflow status | `completed` / `FINISHED` |
| Best baseline | Decision Tree, F1 1.0000, accuracy 1.0000 |
| Best optimized | Decision Tree, F1 1.0000 (best trial 0/20 — the split is trivially separable; reported honestly) |
| Final model | Decision Tree, `selection_source=optimized`, artifact rebuilt from the tuned params (which equal the defaults) |
| Recorded metrics | `final_model.holdout_f1_score` 1.0, `final_model.holdout_accuracy` 1.0 |

Identical to the Phase 9 multiclass result (§18).

## 18. Regression: Phase 9 Results Unchanged

| Check | Result |
|---|---|
| Binary optimized model | SVM, `selection_source=optimized`, F1 0.95778, accuracy 0.958333 — Phase 9 identical |
| Multiclass final model | Decision Tree, F1 1.0000 / 1.0000 — Phase 9 identical |
| `models/best_model.joblib` | still the binary optimized SVM written by a real `--graph` run (metadata reads `Support Vector Machine`) |
| `run_results.json` digest | equal between tracking-on and `--no-mlflow` runs, and equal across repeat tracking-on runs after normalising wall-clock timings inside observation strings |
| Dataset SHA-256 | `49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c` and `0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8` — unchanged before and after every run in this phase |
| README line 968 | `**The core ML pipeline does not require an LLM.**` — unchanged |
| Workflow shapes | 7 nodes, `WORKFLOW_ORDER` untouched; `graph_version` 0.8.0 untouched (Phase 10 adds no node) |

**CLI variant matrix** (every variant exit 0 with the Phase 9 baseline numbers):

| Variant | Persistence | Tracking | Result |
|---|---|---|---|
| `--graph` | writes | records | SVM 0.9578/0.9583, FINISHED |
| `--graph --no-optimize` | writes baseline | records | LogReg 0.8761/0.8750 |
| `--graph --llm` | writes | records | SVM; LLM layer present, no effect on metrics |
| `--graph --llm --no-optimize` | writes baseline | records | LogReg baseline |
| `--graph --no-save-model` | skipped | records | `Persistence status: skipped` |
| `--graph --no-mlflow` | writes | none | identical results, skip notice |
| `--graph` + `MLFLOW_ENABLED=false` | writes | none | identical results, disabled notice |
| `--graph` + failing store (`NUL`) | writes | isolated | exit 0, `Failed node: none`, not-recorded notice |
| `--graph --dataset does_not_exist.csv` | — | scratch store | exit 1, `Workflow status: failed`, recorded `mlflow_status=FAILED` in the scratch store (never the real one) |
| `--graph --save-model` | — | — | exit 2 — argparse correctly rejects the non-existent flag (only `--no-save-model` exists) |

## 19. Limitations

1. **MLflow is a new dependency, even though it is never required by the core.**
   It is bounded (`>=3.0.0,<4.0.0`) and recorded in `requirements.txt`, but it
   is the first non-python dependency beyond pandas/scikit-learn/optuna.
2. **The store is single-process-ish.** SQLite is fine for local, sequential
   CLI runs; two `--graph` processes at once are not supported and this is not
   validated.
3. **Status mapping collapses `partial` and `completed` to `FINISHED`.** The
   lossless `workflow_status` tag keeps the exact value; MLflow itself only knows
   the three run-statuses.
4. **"Latest run" is heap-by-start-time, not by dataset.** A user switching
   datasets sees the most recent run in section 11; the history table shows the
   rest. There is deliberately no per-dataset filtering logic yet.
5. **Tracking is only wired to the graph path.** The Phase 6 sequential path
   predates the graph and does not record runs; this is pre-existing behaviour
   left unchanged.
6. **The recorded timings are wall-clock durations** (`NodeRecord.duration`).
   They are workflow facts, not benchmarks; noise between machines is expected
   and stated in the dashboard caption.
7. **Ollama was not available on this machine**, so the `--graph --llm`
   variant validated the unavailable/fallback path; the success path is covered
   by the existing stubbed Phase 8 tests.
8. **The isolation guard detects after the fact.** It hashes before and after;
   it cannot prevent a first write. The explicit `enable_tracking=False` call
   sites are the prevention; the guard is the alarm.

## 20. Conclusion and Phase 11 Recommendation

Phase 10 is complete: a local, dependency-bounded MLflow tracking store records
every `--graph` run's decisions — dataset metadata (with SHA-256, never the raw
rows), per-node statuses and timings, evaluation and optimization metrics, the
`FinalModelSelection`/`PersistenceReport` records, the confusion matrix and the
reproducibility data — while the workflow's results, artifact and datasets are
byte-for-byte unchanged from Phase 9. Run status mirrors workflow status,
including `FAILED` for a failed workflow; a failing store never breaks a run;
the dashboard reads the store without ever training; and the test suite is
fully isolated from the user's real store, guarded by a second session guard.

The phase also found and fixed two real defects: a pre-existing patcher leak in
`test_workflow.py` that silently corrupted later tests under any non-alphabetical
ordering, and the suite-wide pollution of the tracking store, which was cleaned
and rebuilt with exactly the two canonical runs.

For **Phase 11** (Conventional ML vs Autonomous Multi-Agent comparison), the
store now provides the missing ingredient Phase 9 asked for: a per-run record,
keyed by run id, of the dataset, configuration and metrics that produced each
artifact. Recommended, in priority order:

1. **Compare runs across datasets/configurations** by querying the store's
   recorded metrics instead of copy-pasting `run_results.json` values — the
   schema and run ids are already in place.
2. **Persist the digest of the persisted artifact** (`models/best_model.joblib`)
   as a run tag on recording, so a stored metric can always be checked against
   the exact artifact it belongs to. `MLFLOW_LOG_MODEL_ARTIFACT` stays off;
   this is a reference, not a duplicate.
3. **Close Phase 9's metric gap** — score the *tuned* pipeline on the hold-out
   split once and store it in a clearly separate namespace (it must not feed
   selection). A run tag that proves the stored `final_model` metrics are
   matching-reads of the pipeline's own records would keep observability honest.

**Not recommended for Phase 11:** a remote MLflow server, MLflow autologging,
any cloud service, or any LLM dependency in the comparison path. All would break
the property maintained since Phase 1 — local, offline, free, deterministic.