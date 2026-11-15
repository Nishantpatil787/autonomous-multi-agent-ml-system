# Demonstration Guide (College Project Viva)

A practical walk-through of the project for a live demonstration. Each step names
the exact command to run and the exact screen to show. There are **no fictional
screenshots** in this document: where a screenshot is useful, the caption below
the step tells you what to capture.

> Readiness checklist before you start: Python 3.12.3 installed,
> `pip install -r requirements.txt` completed, one workflow run already done
> (`python app.py --graph`) so `models/` contains an artifact, and Streamlit
> available. Everything except step 2 runs with the default dataset and works
> offline.

---

## 1. Start Project

Show the project layout and open the README:

```
dir                 # Windows
ls                  # Linux / macOS
```

Then `README.md` and the folder list `agents/ tools/ graph/ llm/ benchmarks/
tests/`. Point at the layered design: agents call tools, the graph calls
agents, the LLM is optional. **Screenshot:** the project root listing.

Optionally show the quick verification that the environment is healthy:

```bash
python -c "from utils import config; print(config.PROJECT_ROOT)"
```

## 2. Run Autonomous Workflow

```bash
python app.py --graph --dataset datasets/sample_classification.csv
```

Explain that `--graph` runs the compiled LangGraph: Data Analysis →
Preprocessing → Model Selection → Training → Evaluation → Optuna (conditional)
→ Persistence, and that every step records into the shared `AgentState`.
**Screenshot:** the first CLI report (data analysis) and the final line that
prints the overall verdict and exit status.

## 3. Show Data Analysis

From the run above, scroll to the **Data Analysis** report. Show:

- detected columns and types, missing values, duplicates;
- target resolution (`Outcome` detected with its confidence and reasoning);
- problem type (binary classification), class distribution, statistics;
- the agent message log showing `DataAnalysisAgent` completed.

The folder it explores is read-only, so the run never changes the dataset.
**Screenshot:** the data analysis report block.

## 4. Show Preprocessing

Scroll to the **Preprocessing** report. Show:

- selected features and excluded features **with reasons**;
- imputation / encoding / scaling strategies;
- pipeline architecture (`ColumnTransformer`);
- that the transformed data is still a pandas frame passed between stages.

Explain leakage prevention: imputation and scaling parameters are learned on
the training part of the split only. **Screenshot:** the preprocessing report.

## 5. Show Model Selection

Scroll to the **Model Selection** report. Show the controlled registry of five
candidate families — Logistic Regression, Decision Tree, Random Forest, KNN,
SVM — each with its curated hyperparameter grid / search spaces. Explain that
no arbitrary model class is instantiated; every candidate comes from
`tools/model_tools.py`. **Screenshot:** the model comparison table.

## 6. Show Optimization

Scroll to the **Optimization** report. Show:

- that Optuna ran with the TPE sampler, a bounded search space per family, and
  a fixed `random_state`;
- the number of trials and which models were tuned;
- best optimized vs best baseline comparison (metrics on the **same** hold-out
  split — no tuning on the hold-out).

Keep it short: the point is the *process* (autonomous search, bounded spaces,
leakage-free evaluation), not the numbers. **Screenshot:** the optimization
table or the Optuna progress line.

## 7. Show Evaluation

Scroll to the **Evaluation** report. Show:

- hold-out metrics: weighted F1, accuracy, (precision/recall where recorded);
- cross-validation mean metrics and the **generalization gap** (hold-out vs CV);
- confusion matrix;
- the recommendation the agent produced.

This is the stage the project treats as the source of truth: the evaluation
agent interprets recorded results and never retrains. **Screenshot:** the
evaluation block, especially the confusion matrix.

## 8. Show Saved Model

Show the persisted artifacts:

```
dir models            # best_model.joblib, best_model_metadata.json, run_results.json
```

Open `models/best_model_metadata.json` and point at `pipeline_steps`
(`preprocessor`, `model`), `model_name`, and the selection reason. Explain that
`best_model.joblib` stores the *whole fitted pipeline* (preprocessing +
estimator), so `predict()` needs no manual transformation. **Screenshot:** the
models folder listing and the metadata JSON.

## 9. Open Dashboard

```bash
streamlit run dashboard.py
```

Walk the 12 numbered sections from the sidebar / header index:

1. header (project & pipeline) — 2. system status — 3. dataset overview —
4. model comparison — 5. evaluation — 6. confusion matrix — 7. optimization —
8. LLM interpretation — 9. prediction — 10. model information — 11. experiment
tracking — 12. conventional vs autonomous.

Stress that the dashboard is **read-only**: it renders
`models/run_results.json`, the saved model and the MLflow store; it never
retrains. **Screenshot:** the header/section index and one data section.

## 10. Make Prediction

In the dashboard, use section 9 (Prediction): type feature values and press
predict; the answer comes from the saved pipeline. Then show the same thing in
code:

```python
from pathlib import Path
from tools.persistence_tools import load_persisted_model

model = load_persisted_model(Path("models"))           # None when no artifact
if model is not None:
    row = {"Pregnancies": 1, "Glucose": 120, "BloodPressure": 70,
           "SkinThickness": 25, "Insulin": 50, "BMI": 26.5,
           "DiabetesPedigreeFunction": 0.35, "Age": 33}
    import pandas as pd
    print(model.pipeline.predict(pd.DataFrame([row])))
```

**Screenshot:** a prediction result, either in section 9 or the console.

## 11. Show MLflow

Open dashboard section 11 (Experiment tracking): it shows the latest recorded
run (model, F1, accuracy, dataset, status, timings) and a history table. Then
show the local store location (`experiments/tracking/mlflow.db` — a SQLite
file, no server) and the optional UI browser:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/tracking/mlflow.db
```

Explain that recording is off by default on the sequential path, on by default
for `--graph`, and never influences any decision (all imports are lazy).
**Screenshot:** section 11 and/or the MLflow UI run list.

## 12. Show Conventional vs Autonomous Comparison

Two places to show:

- Dashboard **section 12** (Conventional vs Autonomous): the per-dataset
  verdict (match/beat/worse) and the delta table.
- The benchmark artifacts:

```
reports/phase11_comparison.json
reports/phase11_comparison.csv
reports/phase11_f1_comparison.png
reports/phase11_runtime_comparison.png
reports/phase11_experiment_comparison.png
```

Run it live only if you have a few minutes; otherwise show the recorded
results:

```bash
python app.py --benchmark --dataset multiclass
python app.py --benchmark --dataset classification
```

**Screenshot:** dashboard section 12 and the three chart PNGs.

## 13. Explain Results

Use `reports/final_results.md` as the script. State the measured outcome
neutrally:

- Binary: both sides won with SVM; hold-out F1 0.957780 on both sides
  (delta +0.0000); autonomous CV-F1 0.904579 vs conventional 0.893303;
  conventional cheapest (21 experiments, 13.135 s vs 45, 14.307 s).
- Multiclass: both sides won with Decision Tree; F1 1.0000 on both sides
  (delta +0.0000); conventional fastest (13.397 s vs 41.682 s).

The honest conclusion: *on these two fixtures the autonomous workflow matched
the conventional hold-out scores while automating search and orchestration —
it did not beat the careful manual grid, and no claim of universal superiority
is made.* **Screenshot:** the binary/multiclass result tables.

## 14. Explain Limitations

Cover the documented limitations without being prompted (they are part of the
evaluation):

- Phase 6 winners have limited precision/recall provenance (the benchmark's
  precision/recall column is null for the autonomous side);
- the benchmark datasets are small, synthetic samples — results do not estimate
  real-world generalization;
- the benchmark measures machine runtime, not human time;
- autonomous optimization costs more runtime and more experiments;
- the local-LLM success path depends on a separately installed Ollama + model
  being available;
- benchmark scope is limited to the two tested datasets;
- the repository keeps no recorded UI screenshots; capture them as you demo,
  following the captions in this guide.

Close by pointing to `reports/final_project_report.md` as the single document
that gathers all of the above.