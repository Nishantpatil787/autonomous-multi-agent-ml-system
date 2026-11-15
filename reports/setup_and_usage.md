# Setup and Usage Guide

Every command below is the project's real command line interface (verified
against `app.py` and the documentation in `README.md`). Run commands from the
project root. The project was developed and tested on **Python 3.12.3**;
`requirements.txt` targets **Python 3.11+**.

> None of the steps requires an account, an API key, a paid service or a model
> download. The only optional component is the local LLM layer, which needs a
> separately installed [Ollama](https://ollama.com/download) server.

---

## 1. Create the environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the ranges for pandas, numpy, scikit-learn, optuna,
langgraph, joblib, streamlit, mlflow, matplotlib and pytest. Verified with
Python 3.12.3 (pandas 3.0.1, numpy 2.4.2, scikit-learn 1.8.0, optuna 4.9.0,
langgraph 1.2.12, joblib 1.5.3, streamlit 1.65.0, mlflow 3.16.1,
matplotlib 3.11.2, pytest 9.1.1). The optional LLM layer adds **no** Python
dependency.

## 3. Run the tests

```bash
# Full suite (the project's acceptance command)
pytest -q

# Or standard-library discovery
python -m unittest discover -s tests -t .
```

Expected final result: `1887 passed, 7 warnings, 1369 subtests passed`,
exit code 0 (three config regression tests were added by the final release
audit).

## 4. Run the graph workflow (autonomous, full pipeline)

```bash
python app.py --graph --dataset datasets/sample_classification.csv
```

This runs Data Analysis → Preprocessing → Model Selection → Training →
Evaluation → (conditional) Optuna Optimization → Model Persistence, and records
the run in the local MLflow store. Persistence is on by default, so it also
writes `models/best_model.joblib`, `models/best_model_metadata.json` and
`models/run_results.json`.

## 5. Run the no-optimization workflow

```bash
python app.py --graph --no-optimize --dataset datasets/sample_classification.csv
```

The `Should Optimize?` decision is recorded as `skip_disabled` and the workflow
ends after Evaluation (the best baseline is still persisted). This works on the
sequential path too:

```bash
python app.py --no-optimize
```

## 6. Run the LLM-enabled workflow

Requires a local Ollama server and a pulled model (the project never downloads
one):

```bash
ollama serve          # start Ollama (installed separately)
ollama pull qwen2.5:3b   # the default OLLAMA_MODEL
python app.py --graph --llm --dataset datasets/sample_classification.csv
```

`--llm` requires `--graph`. The run completes and exits 0 whether or not Ollama
is available; the LLM is explanatory only and can never change a metric, a
model or a routing decision.

## 7. Save (or skip) the model

Persistence is on by default for `--graph`:

```bash
python app.py --graph --dataset datasets/sample_classification.csv   # saves
python app.py --graph --no-save-model                                # skips
```

The saved artifact is the whole fitted pipeline
(`preprocessor → estimator`) in `models/best_model.joblib`, plus the metadata
and run-results JSON. Load it in Python:

```python
from pathlib import Path
from tools.persistence_tools import load_persisted_model

model = load_persisted_model(Path("models"))   # None when nothing is saved
prediction = model.pipeline.predict(row)        # preprocessing included
print(model.metadata["model_name"])
```

## 8. Start the Streamlit dashboard

```bash
streamlit run dashboard.py
```

The dashboard is read-only: it renders `models/run_results.json`, the saved
model and the MLflow store, and it never retrains anything. It shows all 12
sections (header, system status, dataset overview, model comparison, evaluation,
confusion matrix, optimization, LLM interpretation, prediction, model
information, experiment tracking, conventional vs autonomous). Run the workflow
first so there is an artifact to show; otherwise it starts with a "No persisted
model is available yet." notice.

## 9. Run the conventional-vs-autonomous benchmark

```bash
# Both committed datasets
python app.py --benchmark

# A single dataset
python app.py --benchmark --dataset classification
python app.py --benchmark --dataset multiclass

# Both datasets, explicit; or without MLflow recording
python app.py --benchmark --all
python app.py --benchmark --all --no-mlflow
```

Outputs: `reports/phase11_comparison.json`, `reports/phase11_comparison.csv`,
and three charts (`phase11_f1_comparison.png`,
`phase11_runtime_comparison.png`, `phase11_experiment_comparison.png`).
Benchmark MLflow recording goes to the dedicated store
`experiments/tracking_benchmark/mlflow.db` and never touches the canonical
`experiments/tracking/` store.

## 10. Access MLflow local tracking

No MLflow server is started and none is required. The canonical store is a local
SQLite file, `experiments/tracking/mlflow.db`, created by the first `--graph`
run (here relative to the project root). Three ways to read it:

1. **Dashboard**: open `dashboard.py` section 11 (Experiment tracking), which
   shows the latest run and a bounded history table.
2. **Python API** (read-only in a script):

   ```python
   from mlflow.client import MlflowClient
   from utils import config

   client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
   experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
   runs = client.search_runs([experiment.experiment_id])
   for run in sorted(runs, key=lambda r: r.info.start_time or 0):
       print(run.info.run_id, run.data.params.get("dataset_name"))
   ```

3. **MLflow UI (optional, standard tool, not project code)**:

   ```bash
   mlflow ui --backend-store-uri sqlite:///experiments/tracking/mlflow.db
   ```

To stop recording for a run, pass `--no-mlflow` (or set `MLFLOW_ENABLED=false`).
See `.env.example` for every optional override (`MLFLOW_TRACKING_URI`,
`MLFLOW_EXPERIMENT_NAME`, `MLFLOW_HISTORY_LIMIT`, `OLLAMA_*`, `LOG_LEVEL`).