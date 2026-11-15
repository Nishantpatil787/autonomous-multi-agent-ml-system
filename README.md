# Autonomous Multi-Agent System for Machine Learning Model Development and Optimization

An open-source, local-first multi-agent machine learning system that automates the end-to-end lifecycle of machine learning on tabular datasets using specialized AI agents coordinated by LangGraph.

---

## Overview

Developing machine learning models traditionally requires extensive, iterative manual effort: exploratory data analysis, data cleaning, feature preprocessing, algorithm screening, repetitive cross-validation, hyperparameter tuning, and diagnostic evaluation. 

This project implements an **Autonomous Multi-Agent Machine Learning System** that accepts raw tabular datasets (CSV format) and autonomously executes the complete ML lifecycle without requiring human intervention or paid cloud APIs. The system coordinates specialized agents using **LangGraph**, relies on a deterministic **scikit-learn** and **Optuna** ML core, and provides an optional local **Ollama** LLM layer for narrative interpretation.

---

## Features

- **Automated Data Analysis:** Automatic inspection of tabular structure, missing values, duplicate records, heuristic target detection, and problem type classification (binary vs. multiclass).
- **Leakage-Free Preprocessing:** Reproducible scikit-learn `ColumnTransformer` pipelines with median/mode imputation, standard scaling, and one-hot encoding with unseen category protection.
- **Model Screening & Selection:** Automated screening of candidate algorithms: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).
- **Cross-Validated Baseline Training:** Stratified 80/20 train/test splitting and 5-fold cross-validation logging accuracy, precision, recall, and F1-score.
- **Bayesian Hyperparameter Optimization:** Automated tuning of top candidate models using Optuna's Tree-structured Parzen Estimator (TPE) across bounded search spaces.
- **LangGraph State-Machine Orchestration:** Directed acyclic graph execution with state immutability, node retries, and conditional optimization routing.
- **Optional Local LLM Assistant:** Grounded, advisory natural language interpretations via a local Ollama model (`qwen2.5:3b`, `llama3.2:3b`) without controlling deterministic ML decisions.
- **Atomic Model Persistence:** Automated refitting on full training data, complete pipeline serialization (`models/best_model.joblib`), and cryptographic provenance metadata emission.
- **Local MLflow Experiment Tracking:** Local tracking of parameters, metrics, and run artifacts in `experiments/mlruns/`.
- **Interactive Web Dashboard:** Browser-based Streamlit UI for dataset inspection, model comparison, confusion matrix visualization, live single-sample prediction, and benchmark analysis.
- **Empirical Benchmarking Framework:** Built-in reproducible benchmark comparing conventional grid search against autonomous multi-agent optimization.

---

## Architecture

The system coordinates specialized agents through LangGraph:

```
                  +-----------------------------------+
                  |     Raw Tabular Dataset (CSV)     |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |        Data Analysis Agent        |
                  |  (Inspection, Target Detection)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       Preprocessing Agent         |
                  | (Imputation, Encoding, Scaling)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      Model Selection Agent        |
                  |    (Filter Candidate Models)      |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |          Training Agent           |
                  |  (5-Fold Stratified CV, Baselines)|
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |         Evaluation Agent          |
                  |  (Diagnostic Ranking, Gap Check)  |
                  +-----------------------------------+
                                    |
                         [ Should Optimize? ]
                        /                    \
                YES    /                      \  NO
                      v                        v
         +-----------------------+             |
         |  Optimization Agent   |             |
         |  (Optuna TPE Tuning)  |             |
         +-----------------------+             |
                      \                        /
                       \                      /
                        v                    v
                  +-----------------------------------+
                  |     Model Persistence Agent       |
                  | (Retrain, Pipeline Joblib, Meta)  |
                  +-----------------------------------+
                       |                         |
                       v                         v
         +-------------------------+   +-------------------------+
         | MLflow Local Tracking   |   |   Streamlit Dashboard   |
         | (Params, Metrics, Runs) |   | (Visuals, Prediction UI)|
         +-------------------------+   +-------------------------+
```

---

## Technology Stack

- **Core Language:** Python 3.10+ (Tested on Python 3.10.1)
- **Data Manipulation:** `pandas>=2.0.0`, `numpy>=1.24.0`
- **Machine Learning Core:** `scikit-learn>=1.3.0`, `joblib>=1.3.0`
- **Hyperparameter Optimization:** `optuna>=4.0.0`
- **Multi-Agent Orchestration:** `langgraph>=0.2.0,<0.3.0`
- **Web UI & Dashboard:** `streamlit>=1.30.0`
- **Local Experiment Tracking:** `mlflow>=2.13.0`
- **Local LLM Layer (Optional):** `Ollama` (`qwen2.5:3b`, `llama3.2:3b`)
- **Testing & Verification:** `pytest>=7.0.0`
- **Zero Paid Cloud Dependencies:** No OpenAI, Anthropic, or proprietary cloud API keys required.

---

## Project Structure

```
Seminar/
├── agents/                      # Specialized agent implementations
├── benchmarks/                  # Phase 11 benchmark framework (Conventional vs Autonomous)
├── datasets/                    # Canonical benchmark and sample datasets
├── experiments/                 # Local tracking, MLflow runs, experiment logs
├── graph/                       # LangGraph multi-agent orchestration layer
├── llm/                         # Optional local LLM layer (Ollama integration)
├── models/                      # Persisted best models and verified metadata
├── reports/                     # Benchmark artifacts, performance figures, reports
├── tests/                       # Comprehensive automated pytest test suite
├── tools/                       # Modular, leakage-free ML and data utility libraries
├── utils/                       # Core shared state, configuration, and logging
├── app.py                       # Main application CLI and execution entrypoint
├── dashboard.py                 # Streamlit interactive web dashboard
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## Installation

### 1. Clone or Open Project Directory
```bash
cd "c:\Users\LENOVO\OneDrive\Desktop\Seminar"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the ML Workflow

Execute the autonomous multi-agent pipeline in graph mode:
```bash
python app.py --graph
```

### Running with a Custom Dataset
```bash
# Run with multiclass dataset
python app.py --graph --dataset datasets/sample_multiclass.csv

# Specify an explicit target column if not auto-detected
python app.py --graph --dataset datasets/sample_multiclass.csv --target class_label
```

---

## Running Without Optimization

To run in fast mode and skip the Optuna hyperparameter optimization phase:
```bash
python app.py --graph --no-optimize
```

---

## Running With Local LLM

To enable natural-language dataset interpretation and optimization planning:
1. Ensure Ollama is running locally:
   ```bash
   ollama serve
   ```
2. Pull a supported lightweight model:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. Execute with the `--llm` flag:
   ```bash
   python app.py --graph --llm
   ```
*If Ollama is offline or uninstalled, the system automatically falls back to deterministic numeric execution.*

---

## Saving the Model

Models are automatically persisted upon workflow completion:
- **Pipeline Binary:** `models/best_model.joblib`
- **Metadata Document:** `models/best_model_metadata.json`

Verify the saved model programmatically:
```bash
python -c "
from tools import persistence_tools
pipeline, meta = persistence_tools.load_persisted_model()
print('Loaded Model:', meta['model_name'])
print('Holdout F1:', meta['metrics']['f1_score'])
"
```

---

## Running Streamlit

Launch the interactive web dashboard:
```bash
streamlit run dashboard.py
```
Or launch via the `app.py` launcher:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to explore data summaries, model rankings, confusion matrices, live inference, and benchmark comparisons.

---

## MLflow Tracking

Enable local experiment tracking by setting the environment variable:
```bash
# Windows (PowerShell)
$env:MLFLOW_ENABLED="true"
python app.py --graph

# Linux / macOS
MLFLOW_ENABLED=true python app.py --graph
```

Launch the local MLflow dashboard:
```bash
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```
Navigate to `http://localhost:5000` to review runs, parameters, metrics, and tags.

---

## Running Benchmark

Execute the reproducible Phase 11 comparative benchmark:
```bash
# Benchmark all datasets
python app.py --benchmark

# Benchmark specific dataset
python app.py --benchmark-dataset classification
python app.py --benchmark-dataset multiclass
```

Generated benchmark reports:
- `reports/phase11_comparison.json`
- `reports/phase11_comparison.csv`
- `reports/phase11_f1_comparison.png`
- `reports/phase11_runtime_comparison.png`
- `reports/phase11_experiment_comparison.png`

---

## Testing

Run the automated test suite across all modules:
```bash
python -m pytest tests/ -v
```
Quick summary run:
```bash
python -m pytest tests/ -q
```
**Current Test Status:** 1884 tests passed (352 test functions, 1369 subtests), 0 failures, exit code 0.

---

## Results

Empirical results from the Phase 11 benchmark comparing Conventional vs Autonomous workflows:

| Dataset | Metric | Conventional | Autonomous | Difference |
|---|---|---:|---:|---:|
| **Binary Classification** | Selected Model | Random Forest | **KNN (Optimized)** | — |
| | Holdout F1-Score | 0.4444 | **0.4912** | **+0.0468** |
| | Holdout Accuracy | 0.5000 | **0.5833** | **+0.0833** |
| | Wall-Clock Time | 6.86 s | 10.10 s | +3.25 s |
| | Experiments | 20 | 45 | +25 (2.25×) |
| **Multiclass Classification** | Selected Model | Decision Tree | **KNN (Optimized)** | — |
| | Holdout F1-Score | 0.0571 | **0.2578** | **+0.2007** |
| | Holdout Accuracy | 0.1000 | **0.3000** | **+0.2000** |
| | Wall-Clock Time | 6.87 s | 7.70 s | +0.83 s |
| | Experiments | 20 | 45 | +25 (2.25×) |

---

## Limitations

- **Holdout Ranking during Optimization:** The Phase 6 optimization agent uses holdout performance for candidate ranking. Future releases will introduce a strict three-way train/validation/test split.
- **Dataset Size:** Benchmarked on representative small datasets (50–60 samples). Large-scale datasets (100,000+ rows) require additional profiling.
- **Classification Only:** Supports binary and multiclass classification; regression is not currently implemented.
- **Local LLM Availability:** Natural-language narrative generation requires Ollama to be active locally.

---

## Future Scope

- Support for tabular regression and time-series forecasting.
- Strict three-way data splitting (60% Train, 20% Validation, 20% Test).
- Integration of gradient boosting libraries (XGBoost, LightGBM, CatBoost).
- Distributed hyperparameter search across multi-core or GPU nodes.
- Automated production model monitoring and drift detection.

---

## License

This project is developed as an academic seminar and research mini-project. All rights reserved by the project authors. Open for academic review and educational usage.
