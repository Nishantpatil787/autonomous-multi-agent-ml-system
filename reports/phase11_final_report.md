# Phase 11 Final Report: Controlled Benchmark — Conventional vs Autonomous Workflow

## 1. Objective

Answer one question with evidence instead of opinion: **does the autonomous
multi-agent workflow beat a careful conventional (manual) machine-learning
pass on the same data?**

Phase 11 adds a controlled, reproducible benchmark that runs both sides on the
same committed datasets, the same split, the same preprocessing defaults, the
same five candidate model families and the same weighted metrics — and writes
the comparison to JSON, CSV and three charts, plus a read-only dashboard
section.

The governing rule, enforced in code and asserted by the test suite:

- the **conventional** side is a manual grid over the shared model registry,
  executed with tools only (no agents, no LangGraph, no LLM, no Optuna);
- the **autonomous** side is the *existing* `run_workflow` graph (LLM off,
  Optuna on, persistence to a scratch directory, tracking off) — no new agent
  workflow file;
- both sides are timed with `time.perf_counter()` around the machine-learning
  work only.

## 2. Scope and Non-Goals

In scope:

1. `benchmarks/` — a new package with three modules: `benchmark_config.py`
   (pinned datasets, hashes, grid, MLflow store constants),
   `conventional_workflow.py` (the manual side) and `benchmark_runner.py` (the
   side-by-side runner, the comparison records, MLflow recording and the
   JSON/CSV/PNG writers).
2. `--benchmark` (+ `--dataset classification|multiclass`, `--all`, honored
   with `--no-mlflow`) in `app.py`, dispatching before any agent is built.
3. Dashboard **section 12, "Conventional vs Autonomous"** — read-only, showing
   `DASHBOARD_NO_BENCHMARK` when `reports/phase11_comparison.json` is absent.
4. `reports/phase11_comparison.json` / `.csv` and three PNG charts
   (`phase11_f1_comparison.png`, `phase11_runtime_comparison.png`,
   `phase11_experiment_comparison.png`).
5. Benchmark MLflow recording in the **dedicated** store
   `experiments/tracking_benchmark/mlflow.db` (never the canonical store).
6. Two new hermetic test files (`tests/test_conventional_workflow.py`,
   `tests/test_benchmark_comparison.py` — 32 new tests) and the updated
   pre-existing tests that name the phase.

Explicitly out of scope:

- **No new agent, node or workflow file.** The autonomous side is a wrapper
  over the existing `run_workflow` (Phase 7-9 machinery untouched:
  `WORKFLOW_ORDER`, 7 nodes, `graph_version` 0.8.0).
- **No training from the dashboard.** Section 12 only ever reads the JSON
  report.
- **No change to the canonical artifacts.** `experiments/tracking/mlflow.db`
  keeps exactly the two Phase 10 runs; the benchmark writes only to
  `experiments/tracking_benchmark/` and temporary scratch directories.
- **No hyperparameter tuning for the conventional side.** It is a fixed 21-
  configuration grid, by design the "careful manual" baseline.
- **No random seeds added to the data path**: both sides already use the
  project's fixed `random_state=42` split.

## 3. Architecture

```
python app.py --benchmark [--dataset classification|multiclass] [--all]
        │  (dispatch before any agent is built)
        ▼
benchmarks/benchmark_runner.run_benchmark(datasets, n_trials=20, ...)
        │  for each dataset:
        ├─► run_conventional_workflow(dataset)        benchmarks/conventional_workflow.py
        │        tools only: data_tools, preprocessing_tools, training_tools
        │        21-config grid from benchmark_config.build_conventional_grid()
        │        selection = project ranking rule (rank_experiments)
        └─► _run_autonomous_side(dataset)             benchmarks/benchmark_runner.py
                 run_workflow(enable_llm=False, enable_optimization=True,
                              n_trials=OPTUNA_N_TRIALS, enable_persistence=True,
                              persistence_output_dir=<scratch>, enable_tracking=False)
                 reads FinalModelSelection + evaluation/optimization counts
        ▼
   _build_comparison(dataset, sha, conventional, autonomous)
        ▼
   record_benchmark_mLflow(...)  ──►  experiments/tracking_benchmark/mlflow.db (best-effort)
   _write_json / _write_csv / _write_charts  ──►  reports/phase11_*  (JSON written last,
                                                  so metadata.files_written is complete)
```

Two measurements are taken around the same work, side by side in the same
process: the analyst's grid pass and the autonomous graph pass. `run_benchmark`
returns a `BenchmarkSuiteResult` with one `DatasetComparison` per dataset and
the CLI prints a formatted summary (section 8).

## 4. Benchmark Configuration and Pinned Datasets

| Setting | Value |
|---|---|
| Binary fixture | `datasets/sample_classification.csv` — SHA-256 `49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c` |
| Multiclass fixture | `datasets/sample_multiclass.csv` — SHA-256 `0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8` |
| Split | `test_size=0.2`, `random_state=42`, stratify when possible (project default, unchanged) |
| Candidate families | the five registry families: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine |
| Conventional grid | exactly **21 configurations**, every one inside the Phase 6 search-space bounds (decade `C` values `{0.1, 1.0, 10.0}`, tree depths `(None, 5, 8)`, estimators `{50, 100}`, `n_neighbors {3, 5, 7}`, SVM kernels `{linear, rbf}`) — see `build_conventional_grid()` |
| Autonomous trials | `OPTUNA_N_TRIALS = 20` per top-2 model (Optuna TPE, `sampler_seed=42`) |
| Experiment definition | one experiment = one unique model configuration trained and evaluated on the hold-out split; CV folds inside an experiment are not counted as experiments |
| Timing | `time.perf_counter()` around the ML work of each workflow |
| Benchmark MLflow store | `sqlite:///...experiments/tracking_benchmark/mlflow.db`, experiment `benchmark-phase11` (separate file from the canonical `experiments/tracking/mlflow.db`, asserted by test) |

The `EXPECTED_DATASET_SHA256` map is pinned in `benchmark_config.py` and is
asserted against the real fixture bytes byte-for-byte by both new test files;
the runner refuses nothing it has not validated and records the hash on every
comparison.

## 5. The Fairness Contract

Both sides receive the identical inputs, in one process, with no special
knowledge passed to either:

1. **Same dataset bytes** — same committed fixture, verified by SHA-256 on
   every run.
2. **Same split** — both sides call the project's `training_tools.split_dataset`
   with the same defaults (`test_size 0.2, random_state 42, stratified`); the
   smoke result is that both report 96/24 (binary) and 72/18 (multiclass).
3. **Same preprocessing defaults** — median / most-frequent imputation, one-hot
   encoding, standard scaling for linear models, `record_id` excluded as
   identifier-like; both sides go through `training_tools.build_model_pipeline`.
4. **Same candidate families** — the five registry definitions
   (`tools/model_tools.py`), the same constructors the graph uses.
5. **Same metrics** — weighted average F1 (primary), accuracy (secondary),
   CV-F1 mean; a conventional grid experiment runs the exact same
   `evaluate_classifier` + `run_cross_validation` the agents run.
6. **Same selection rule** — f1 descending, accuracy descending, model name
   ascending. The autonomous baseline stage uses `select_best_experiment`; the
   conventional grid uses `rank_experiments` (the same documented rule at
   configuration granularity — see section 13 for why this matters).

Timing is the only intrinsically asymmetric input: the manual side cannot race
the graph side, and the report states this as a note rather than hiding it.

## 6. The Conventional Workflow (tools only)

`benchmarks/conventional_workflow.py` exposes `ConventionalWorkflowResult` and
`run_conventional_workflow(dataset_path, ...)`:

1. `data_tools` load + `_sha256` of the exact bytes.
2. `training_tools.split_dataset` (same call the agents make).
3. Family detection for the target and problem type (same helper the agents
   use), input feature count and excluded features.
4. The deterministic 21-config grid, executed with
   `training_tools.build_model_pipeline` → `evaluate_classifier` →
   `run_cross_validation` — the same training tools, in the same order.
5. A failing configuration is recorded as an `ExperimentResult` with
   `status=failed` and its error message; it never aborts the pass.
6. `_select_grid_best` applies the project ranking rule and returns a
   `BestModelInfo` + the winning `ExperimentResult`.

A structural test parses the module source and asserts it imports nothing from
`agents`, `graph`, `llm` or `optuna` and no `tools.optimization_tools` usage:
the manual side genuinely cannot smuggle in the autonomous machinery.

## 7. The Autonomous Workflow (existing run_workflow)

`_run_autonomous_side` wraps the existing graph exactly as specified:

```
run_workflow(dataset_path, enable_optimization=True, cross_validation=True,
             n_trials=N, enable_llm=False, enable_persistence=True,
             persistence_output_dir=<scratch>, enable_tracking=False)
```

- the run is timed with `time.perf_counter()` around the call;
- persistence goes to a scratch directory (a temporary directory by default —
  the run never writes into `models/`);
- tracking is off, so nothing touches the canonical MLflow store;
- the extraction reads `state.persistence_report.selection`
  (`FinalModelSelection`: `model_name`, `selection_source`,
  baseline/optimized hold-out F1 + accuracy, CV F1 means, `best_params`),
  `evaluation_report.experiment_count` (baseline experiments) and
  `optimization_report.counts` (completed/failed trials), the `data_split`
  facts, and the per-node durations from `WorkflowResult.node_records`.

One dataset's experiment count on the autonomous side is therefore `baseline
experiments + actual completed Optuna trials` — the same "one experiment = one
unique configuration" definition the conventional side uses. No new agent
workflow file was added; this is Phase 7-10 code executed as-is.

## 8. The Comparison: Deltas and Verdicts

`DatasetComparison` carries one `WorkflowBenchmarkResult` per side plus the
arithmetic the dashboard and the report need:

| Field | Rule |
|---|---|
| `delta_f1` / `delta_accuracy` / `delta_cv_f1` | `autonomous − conventional` |
| `delta_runtime` | `autonomous.runtime − conventional.runtime` |
| `delta_experiments` | `autonomous.experiments − conventional.experiments` |
| `f1_relative_percent` | `100 × delta_f1 / conventional.f1` (0 when conventional F1 is 0) |
| `verdict` | `autonomous matched or exceeded conventional F1 (delta_f1=...)` when `delta_f1 ≥ 0`, else `conventional scored higher F1 (delta_f1=...)` |

The verdict is honest by construction: it **never assumes the autonomous side
wins**. It reports the sign of a measured delta. If the conventional grid wins,
the verdict says so.

## 9. MLflow Recording in the Dedicated Benchmark Store

`record_benchmark_mLflow` records each dataset comparison into the **benchmark**
store (`experiments/tracking_benchmark/mlflow.db`), never the canonical one
(the two URIs are asserted different, and the canonical path is asserted absent
from the benchmark URI).

- Per side (conventional / autonomous) one run is created, tagged
  `workflow`, `dataset`, `phase`, `kind=benchmark`.
- Params: `dataset`, `workflow`, `selected_model`, `selection_source`,
  `split_test_size`, `split_random_state`, `dataset_sha256`.
- Metrics: `f1_score`, `accuracy`, `precision`, `recall`, `cv_f1_mean`,
  `runtime_seconds`, `experiments`.
- Recording is **best-effort**: any failure (blocked dependency, unreachable
  store, malformed URI) returns `{"recorded": False, "error": ...}` and can
  never fail the benchmark. The two failure tests assert exactly this
  (a mocked `MlflowClient` that raises, and a nonexistent URI path).

The real `--benchmark` run recorded exactly **4 runs** (2 per dataset, one per
workflow) into `benchmark-phase11`, all `FINISHED` (section 16/17).

## 10. CLI Integration

`app.py` gains:

```
python app.py --benchmark                 # both committed datasets
python app.py --benchmark --all           # explicit form of the above
python app.py --benchmark --dataset classification   # binary only
python app.py --benchmark --dataset multiclass      # multiclass only
python app.py --benchmark --no-mlflow     # skip the MLflow recording
```

- `--benchmark` dispatches **before** `AgentState()` is built — the benchmark
  is not a graph run, it is the controlled experiment.
- `--no-mlflow` propagates as `benchmark_mLflow=False` (same switch as the
  graph path). The CLI tests assert the flag reaches the runner.
- A complete benchmark exits 0; a hard top-level failure exits 1; a dataset
  whose autonomous status is not `success` is reported and also returns 1.
- Every existing flag keeps its behaviour; adding the new flags broke nothing
  (the full suite, section 15).

## 11. Dashboard Section 12 (read-only)

`dashboard.py` renders **section 12, "Conventional vs Autonomous"** after
section 11 (Experiment tracking). It calls `helpers.build_benchmark_section()`
(`tools/dashboard_tools.py`), which reads only
`reports/phase11_comparison.json`:

- when the file is missing or unreadable it returns `available=False` and the
  section shows `config.DASHBOARD_NO_BENCHMARK` ("No Phase 11 benchmark results
  are available yet.");
- when present it renders one dataframe per dataset (workflow, best model,
  source, F1, accuracy, CV F1, runtime, experiments), the four deltas, the
  verdict and the notes.

The section is **read-only by construction**, and a test proves it on source:
`render_benchmark` contains no `run_benchmark(`, `run_workflow(` or
`run_dataset_benchmark(` call. The dashboard tests now assert 12 sections and
that `render_benchmark` is wired into `main()`.

## 12. Test Isolation: Canonical Store and `models/` Untouched

- The canonical store `experiments/tracking/mlflow.db` held **exactly 2 runs**
  (`0e99fb29…` SVM 0.95778, `7a79111c…` Decision Tree 1.0) before and after
  every Phase 11 execution, including the full suite (run count 2 → 2).
- Every pytest session is protected by the existing conftest session guards,
  which hash `models/` and `experiments/tracking/` before and after the session
  and fail when either changed — all sessions green.
- The benchmark's autonomous persistence targets a scratch directory (a
  temporary directory by default; the real run logged
  `[PERSIST] saved … to %TEMP%\phase11_artifacts_hgvhdf3g`), never `models/`.
- Benchmark MLflow goes only to `experiments/tracking_benchmark/mlflow.db`.
- For completeness: `models/` currently holds `best_model.joblib` (2027 bytes)
  whose metadata records a baseline Logistic Regression from an
  optimization-disabled run. No Phase 11 code path writes to `models/`; the
  artifact loads through `load_persisted_model` and predicts (verified with a
  one-row prediction). The Phase 11 report chooses evidence over the rumoured
  state: the artifact's own metadata is quoted, not assumed.

## 13. Defect Found — `select_best_experiment` Collapses the Grid by Model Name

The project's `select_best_experiment` (Phase 4/5) is correct for the baseline
stage, where each candidate family appears exactly once: it ranks all
experiments, takes the best model *name*, and returns the **first experiment
with that name** as the "winner".

For the Phase 11 grid that is wrong: one family contributes several
configurations, so the first experiment named e.g. "Support Vector Machine" is
the first grid row (`C=0.1, linear`), not the top-scoring configuration. A
test driven by a broken grid config exposed this immediately: a failed
`C=-1.0` Logistic Regression was chosen as "best" because it was the first
record with the winning model name.

Fix: `conventional_workflow._select_grid_best` uses `rank_experiments` — the
same documented rule (f1 desc, accuracy desc, model name asc) — at
*configuration* granularity, and matches the top-ranked entry back to its
`ExperimentResult`. The selection reason now names the winning configuration's
metrics, and the smoke grid selects the true winner: SVM `C=10, linear`
(`f1 0.9578`), not `C=0.1, linear`.

## 14. Defect Found — MLflow 3.x Dropped the Batch `log_params`/`log_metrics`

The first MLflow happy-path test failed with
`AttributeError: 'MlflowClient' object has no attribute 'log_params'`. MLflow
3.0 removed the batch client methods `log_params`/`log_metrics` in favour of
the per-key `log_param`/`log_metric` (and `log_batch`). The recording was
silently caught by the try/except — which is exactly what the failure-isolation
design wants — but the store never got the runs.

Fix: `record_benchmark_mLflow` logs params and metrics in per-key loops. The
happy-path test now reads the 4 runs (2 per dataset) back from a scratch sqlite
store and checks the recorded `f1_score` and the `workflow` tags.

## 15. Testing

All numbers below come from the final full suite run, executed from the project
root in alphabetical order after every change in this phase landed.

```
1884 passed, 7 warnings, 1369 subtests passed in 396.96s (0:06:36)
```

Exit code **0**, guard re-armed: the session artifact guard (section 12)
verified after the run that `models/` and `experiments/tracking/` were
byte-identical to their pre-run state. The count includes the regression test
added in §15.2 below.

- **No test was deleted; no skip was added to force green; no assertion was weakened.**
- 32 new tests: 12 in `tests/test_conventional_workflow.py` (pinned hashes,
  grid size and search-space bounds, tools-only import lint, split identity,
  canonical-number reproduction for both datasets, determinism, ranking rule,
  failure recording, typed records) and 20 in
  `tests/test_benchmark_comparison.py` (data integrity, benchmark-store/canonical
  separation, autonomous-side extraction, delta/verdict arithmetic, MLflow
  happy path + failure isolation, JSON/CSV/PNG artifact writing, one honest
  end-to-end run on the binary fixture with `n_trials=1` into scratch dirs,
  the read-only dashboard section, and the `--benchmark` CLI dispatch with the
  workflow runners stubbed).
- Updated pre-existing tests that document the phase:

| Test | Change |
|---|---|
| `test_llm_integration.py::test_the_sequential_path_is_unchanged_by_the_new_flag` | banner assertion `Phase 10` → `Phase 11` (docstring says each phase updates it) |
| `test_dashboard.py::test_the_page_shows_the_title_and_all_eleven_sections` | renamed `…all_twelve_sections`; expected subheader gains `"12. Conventional vs Autonomous"` |
| `test_mlflow_tools.py::test_the_section_number_is_eleven` / `test_every_section_is_wired_into_main` | section numbers 11+12 asserted; `render_benchmark` added to the `main()` wiring list |

Both pre-existing updates are documentation of the current phase: no ML
result, metric, threshold or ranking moved.

### 15.1 The full-suite gate exposed a latent persistence leak in a pre-existing test

The first full-suite run reported **`1883 passed` but exit code 1**: the
conftest session guard (section 12) flagged
`models/run_results.json: overwritten by a test`. The offender was a
pre-existing test, `tests/test_workflow.py::TestDatasetsAreUnchanged::test_a_workflow_run_does_not_modify_the_dataset`,
which drives the *real* compiled graph on the real binary dataset with
`enable_tracking=False` but **no persistence isolation**. Phase 9 had made
persistence part of the workflow, so that call's real PersistenceAgent node
silently rewrote the project's `models/` artifacts on every suite run:
`best_model.joblib` and `best_model_metadata.json` happened to be byte-identical
to the pre-existing baseline artifacts (deterministic pipeline), while
`run_results.json` differed — exactly the single file the guard named. This is
the same defect class the conftest guard was written for (Phase 9 §12 header),
and it predates Phase 11; no Phase 11 code path writes to `models/` (section
12), and the new Phase 11 tests are themselves guarded (they run the real
workflow into scratch directories only).

Fix: the test now passes `enable_persistence=False` — the exact remedy
conftest's message prescribes — with a comment explaining why. The test's
purpose (the dataset is byte-unchanged by a run) is unaffected; no assertion
was added, removed or weakened. `TestDatasetsAreUnchanged` passes in isolation
with exit code 0 and the guard re-armed.

### 15.2 A second, deeper leak: the MLflow test helper wrote into `models/`

Two more full-suite runs still exited 1 with the guard naming
`models/run_results.json`. A one-off instrumentation plugin (a temporary
pytest hook that hashed `models/` at every test boundary, deleted after the
diagnosis) pinned the writer to `tests/test_mlflow_tools.py`, and narrowing
runs isolated it to the module's `persisted_state` helper.

Root cause: the helper set a dynamic `persistence_output_dir` attribute on the
state and then called `PersistenceAgent().run(state)` **without an explicit
`output_dir` argument**. `AgentState` — unlike the graph's `GraphState` — has
no `persistence_output_dir` field, so the attribute the helper set was
invisible to the agent. Every E2E MLflow test class built its state through
this helper, so each one re-persisted the baseline model into
`config.MODELS_DIR`. `best_model.joblib` and `best_model_metadata.json`
survived because the baseline pipeline is deterministic (byte-identical
rewrites change no hash); `run_results.json` carries node-walk timings, so it
differed on every run — which is precisely why the guard kept naming that one
file. The same defect also explains why the checked-in `models/` content (a
baseline Logistic Regression) no longer matched the Phase 10 report's
described SVM: earlier suite runs had already replaced it.

Fix: the helper now passes `output_dir=output_dir`, the same explicit
convention the graph node already uses (`graph/nodes.py`). Regression
protection is two-layered:

1. A behavioral regression test,
   `tests/test_persistence_tools.py::TestProjectArtifactIsolationGuard::test_the_mlflow_state_helper_never_touches_the_configured_models`,
   drives the real helper the way the MLflow test classes do and asserts the
   artifacts land in its private directory while `models/` stays byte-for-byte
   identical.
2. The session guard's source scan was extended to flag any bare
   `PersistenceAgent().run(...)` call in a test file (it previously covered
   only `run_workflow(...)` — the gap this exact call slipped through).

The suite count rose from 1883 to 1884 — the regression test is the only
addition. No existing assertion was weakened and no skip was introduced.

## 16. Binary Dataset Results

`datasets/sample_classification.csv` (123 rows, 9 columns, 7 features used;
split 96 train / 24 test stratified; SHA-256
`49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c`).

| Metric | Conventional (grid) | Autonomous (optimized) | Δ |
|---|---|---|---|
| Best model | **Support Vector Machine** | **Support Vector Machine** | same family |
| Configuration | `C=10.0, kernel=linear` | `C=14.5660968…, kernel=linear` (Optuna) | — |
| Weighted F1 | **0.957780** | **0.957780** | **+0.000000** |
| Accuracy | 0.958333 | 0.958333 | +0.000000 |
| Precision | 0.960938 | not recorded (Phase 6 limitation) | — |
| Recall | 0.958333 | not recorded (Phase 6 limitation) | — |
| CV F1 (mean) | 0.893303 | 0.904579 | +0.011276 |
| Runtime (s) | 13.135 | 14.307 | +1.172 |
| Experiments | 21 | 45 (5 baseline + 40 trials) | +24 |

Verdict: **autonomous matched or exceeded conventional F1 (delta_f1=+0.0000)**.

Both sides converged on the same model family and the same hold-out score to
four decimals; Optuna's chosen `C≈14.57` sits on the same flat plateau as the
grid's `C=10`, so the F1 delta is a measured zero, not an assumption. The
autonomous side needed 45 experiments and 14.3 s of ML work; the manual grid
needed 21 configurations and 13.1 s.

## 17. Multiclass Dataset Results

`datasets/sample_multiclass.csv` (92 rows, 7 columns, 5 features used; split
72 train / 18 test stratified; SHA-256
`0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8`).

| Metric | Conventional (grid) | Autonomous (optimized) | Δ |
|---|---|---|---|
| Best model | **Decision Tree** | **Decision Tree** | same family |
| Configuration | `max_depth=2` | `max_depth=8, min_samples_split=20, min_samples_leaf=8, criterion=gini` | — |
| Weighted F1 | **1.000000** | **1.000000** | **+0.000000** |
| Accuracy | 1.000000 | 1.000000 | +0.000000 |
| Precision / Recall | 1.0 / 1.0 | not recorded (Phase 6 limitation) | — |
| CV F1 (mean) | 1.000000 | 1.000000 | +0.000000 |
| Runtime (s) | 13.397 | 41.682 | +28.285 |
| Experiments | 21 | 45 (5 baseline + 40 trials) | +24 |

Verdict: **autonomous matched or exceeded conventional F1 (delta_f1=+0.0000)**.

This dataset is trivially separable: six of the 21 manual configurations
already reach F1 = 1.0, the grid winner being the simplest of them
(`max_depth=2`). The autonomous search paid 41.7 s (mostly Random Forest, 32.2 s
of its 20 trials) to also reach 1.0. Quality ties; the manual side was cheaper.

## 18. Limitations

1. **Precision/recall are not recorded for optimized models** on the autonomous
   side (Phase 6 limitation: per-class metrics exist only for baseline
   evaluations). The comparison therefore uses F1/accuracy/CV-F1 — weighted
   metrics both sides always have.
2. **Runtime is workload cost, not quality.** Two different code paths are
   timed end-to-end in the same process; the numbers are single-shot wall-clock
   values from one machine and one run, as the runner's own notes say.
3. **The datasets are small synthetic fixtures.** F1 deltas of exactly ±0.0000
   on a 24-row and an 18-row hold-out split are coarse; the multiclass dataset
   is trivially separable. Nothing here extrapolates to real data.
4. **One split, one seed.** Both sides share it (that is the point), but every
   number is a single-estimate, not a repeated-measures estimate.
5. **The grid is fixed at 21 configurations.** A different manual grid could
   score higher on binary (e.g. a finer C sweep); the benchmark compares *this*
   specified manual protocol against *this* specified autonomous protocol.
6. **Optuna uses set-of-experiment counts.** 45 autonomous experiments = 5
   baselines + 40 trials; trials that get pruned or fail are still journeys
   through the pipeline and are counted as experiments.
7. **Dashboard section 12 is a snapshot.** It renders the last
   `phase11_comparison.json` written; there is deliberately no live store
   query and no re-run trigger.
8. **MLflow adds the benchmark store as a second tracking database.** It is a
   separate, removable file under `experiments/tracking_benchmark/`; the
   canonical `experiments/tracking/mlflow.db` (2 runs) is never opened by the
   benchmark.
9. **The `NUL`-store style failure exercise is not repeated here.** MLflow
   failure isolation is covered by the two dedicated tests (blocked client,
   non-existent path) rather than a manual run.
10. **`models/` was not rebuilt by this phase and was not rebuilt at all.** The
    current artifact is a baseline Logistic Regression left by an earlier
    optimization-disabled run; Phase 11 wrote nothing to `models/` (section
    12), and the artifact loads and predicts.

## 19. Conclusion and Phase 12 Recommendation

Phase 11 is complete: a controlled, reproducible benchmark now runs the
conventional (tools-only, 21-config manual grid) and the autonomous (existing
`run_workflow` with 20 Optuna trials per top-2 model) workflows side by side on
both committed datasets, and its results are written to
`reports/phase11_comparison.json`, `.csv` and three PNG charts, recorded into a
dedicated MLflow store, and shown in a strictly read-only dashboard section 12.

Evidence from the real run (`python app.py --benchmark`, both datasets):

- **Binary:** conventional SVM `C=10, linear` F1 **0.957780** vs autonomous
  SVM (Optuna `C=14.57, linear`) F1 **0.957780** — Δ F1 = **0.0000**.
- **Multiclass:** conventional Decision Tree (`max_depth=2`) F1 **1.0000** vs
  autonomous Decision Tree (tuned) F1 **1.0000** — Δ F1 = **0.0000**.
- Costs: conventional 13.1 s / 13.4 s and 21 experiments per dataset;
  autonomous 14.3 s / 41.7 s and 45 experiments per dataset.

**The evidence does not show the autonomous workflow beating the conventional
one on these fixtures.** On both datasets the winner is the same model family
and the F1 delta is a measured zero, while the manual protocol needed far
fewer experiments and, on multiclass, a third of the time. The honest verdict
is "matched, not exceeded".

**Recommendation for Phase 12:** the benchmark machinery is now in place and
should be used where it can discriminate. The next comparison worth making is
on a dataset that is not trivially separable and has enough hold-out rows for a
delta to be meaningful — for example a larger, real dataset (or a
multi-seed / repeated-split protocol) — and/or feature engineering, where the
autonomous search over a bigger parameter and preprocessing space has room to
demonstrate value or to be shown not to. The benchmark's verdict function
already supports that honestly: it reports whichever side scores higher.