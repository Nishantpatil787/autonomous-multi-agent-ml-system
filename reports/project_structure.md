# Final Project Structure

The project root is `autonomous_multi_agent_ml_system/`. The layout separates
**orchestration** (`agents/`), **graph control flow** (`graph/`), **reusable
data operations** (`tools/`), **shared typed contracts** (`utils/`), **optional
LLM layer** (`llm/`) and the **benchmark package** (`benchmarks/`), with output
folders for data, artifacts, reports and experiments. Only the important
directories and their responsibilities are listed; generated caches
(`__pycache__/`, `.pytest_cache/`) and per-run build output are omitted.

## Root files

| Path | Responsibility |
|---|---|
| `app.py` | Command line entry point. Runs the sequential or the `--graph` workflow, the optional LLM layer, persistence, and the Phase 11 `--benchmark` CLI. |
| `dashboard.py` | Streamlit presentation file, sections 1-12. Presentation only: it never trains, tunes or evaluates; all logic lives in `tools/dashboard_tools.py`. |
| `requirements.txt` | Pinned dependency ranges for pandas, numpy, scikit-learn, optuna, langgraph, joblib, streamlit, mlflow, matplotlib and pytest. |
| `README.md` | Full project documentation: overview, phases 2-11, technology stack, project structure, commands. |
| `.gitignore` | Ignored build output: caches, `models/*.joblib`, `experiments/`, `mlruns/`, `logs/`, generated `reports/*.csv|json|html`, non-sample datasets. |
| `.env.example` | Documented, optional environment overrides. No credentials anywhere; everything is commented out. |

## `agents/` — agent orchestration

One agent per major workflow stage. Each agent reads and updates the shared
`AgentState` and never bypasses `tools/`.

| Path | Responsibility |
|---|---|
| `base_agent.py` | Shared agent behaviour: structured `agent_messages` logging. |
| `orchestrator.py` | Phase 1 placeholder for workflow coordination; the role is realized by LangGraph (Phase 7), so this agent only logs. |
| `data_analysis_agent.py` | Loads the dataset, detects feature types, missing values, duplicates, statistics, class distribution, resolves the target column and detects the problem type (read-only). |
| `preprocessing_agent.py` | Decides feature exclusions, imputation, encoding, scaling and builds a reproducible (unfitted) `ColumnTransformer`. |
| `model_selection_agent.py` | Selects candidate models from the controlled registry (LR, DT, RF, KNN, SVM). |
| `training_agent.py` | Trains each candidate with the hold-out split and cross-validation; records metrics, timings, confusion matrices and the ranked best baseline. |
| `evaluation_agent.py` | Interprets the recorded baseline results only (never trains): gap analysis, confusion-matrix analysis, timing and counts, recommendation. |
| `optimization_agent.py` | Runs the Optuna search on the top baselines and compares tuned vs untuned for the same models. |
| `persistence_agent.py` | Selects the final model with the existing ranking, performs the final fit, and writes the three artifacts (`models/`). |

## `tools/` — reusable, testable data operations

| Path | Responsibility |
|---|---|
| `data_tools.py` | Dataset loading, feature-type detection, missing values, duplicates, statistics, target resolution heuristic, problem-type detection, data quality checks. |
| `preprocessing_tools.py` | Feature exclusion rules, imputation/encoding/scaling configuration, ColumnTransformer builder, duplicate removal. |
| `model_tools.py` | The controlled model registry (`ModelDefinition` per family) and scaling-config helpers. |
| `training_tools.py` | Stratified split, per-model pipeline fitting, cross-validation, weighted metrics, confusion matrices, deterministic ranking. |
| `evaluation_tools.py` | Hold-out vs CV gap arithmetic, confusion-matrix analysis, execution-time analysis, experiment counts, recommendation. |
| `optimization_tools.py` | Bounded search spaces, Optuna study/trial handling and the baseline-vs-optimized comparison report. |
| `persistence_tools.py` | Artifact paths, final-model selection, build metadata / run results, atomic writes, safe `joblib.load` (the only load site). |
| `mlflow_tools.py` | MLflow recording of a completed workflow (lazy import; never influences a decision). |
| `dashboard_tools.py` | Read-only dashboard section builders; imports neither Streamlit nor the agents. |

## `graph/` — LangGraph orchestration

| Path | Responsibility |
|---|---|
| `state.py` | `GraphState`, `WorkflowResult`, `GraphError`, node enumerations and records. |
| `nodes.py` | One node per agent invokes its agent, records success/failure, applies the bounded retry policy. |
| `routing.py` | Deterministic routing functions: failure routing and the `Should Optimize?` decision. |
| `llm_nodes.py` | Optional explanation nodes added only with `--llm`; they never mutate state or route. |
| `workflow.py` | Graph build, compile, `run_workflow()`, structure report. |

## `llm/` — optional local LLM layer (Ollama)

| Path | Responsibility |
|---|---|
| `state.py` | Typed records: `LLMStatus`, `LLMResult`, `LLMInterpretation`, `LLMRecommendation`. |
| `prompts.py` | Instruction text and the allowlisted summarised context sent to the model. |
| `client.py` | The only module that speaks HTTP, to Ollama over loopback (`urllib.request`); rejects non-loopback hosts. |
| `parser.py` | JSON extraction, schema validation, one controlled repair attempt. |
| `service.py` | Health check, prompt assembly, fallback handling; never raises. |

## `benchmarks/` — Phase 11 controlled benchmark

| Path | Responsibility |
|---|---|
| `benchmark_config.py` | Pinned dataset hashes, the 21-config conventional grid, benchmark MLflow store settings. |
| `conventional_workflow.py` | The tools-only manual grid side (no agents, no LangGraph, no LLM, no Optuna), with grid-best `_select_grid_best`. |
| `benchmark_runner.py` | Side-by-side runner, deltas and verdicts, benchmark MLflow recording, JSON/CSV/PNG report writers. |

## Output folders

| Path | Responsibility |
|---|---|
| `datasets/` | Committed small synthetic samples (`sample_classification.csv`, `sample_multiclass.csv`); real datasets are not versioned. |
| `models/` | Build output: `best_model.joblib` (fitted pipeline), `best_model_metadata.json`, `run_results.json`. |
| `reports/` | Per-phase Markdown deliverables, benchmark JSON/CSV/PNG, and the Phase 12 final documentation set. |
| `experiments/` | Machine-local MLflow state: canonical store `experiments/tracking/mlflow.db` and benchmark store `experiments/tracking_benchmark/mlflow.db`. |
| `logs/` | Runtime logs (created automatically). |

## `utils/` — shared contracts

| Path | Responsibility |
|---|---|
| `config.py` | All paths and tunable thresholds: split/CV settings, Optuna settings, OLLAMA settings, persistence filenames, MLflow URI/experiment, gap thresholds. |
| `state.py` | The typed shared `AgentState` (data contracts) and its JSON-safe serialisation. |
| `logger.py` | Logging setup (one line per event). |

## `tests/` — test suite

- `tests/helpers.py`: shared fixtures, expected hashes, datasets paths,
  `ProjectArtifactIsolationGuard`.
- One test module per tool and per agent (e.g. `test_data_tools.py`,
  `test_training_agent.py`), plus `test_workflow.py` (compiled graph end to end),
  `test_mlflow_tools.py`, `test_dashboard.py`, `test_llm_*.py`,
  `test_conventional_workflow.py` and `test_benchmark_comparison.py`.
- The suite is run with `pytest -q` from the project root.