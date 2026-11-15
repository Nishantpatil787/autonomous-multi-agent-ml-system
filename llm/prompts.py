"""Structured prompt construction and context minimization for Phase 8 LLM layer.

Adheres strictly to scientific rigor:
- Only sends aggregated summary state (no raw datasets, no passwords, no tokens).
- Clear system boundary instructing the model not to invent metrics or contradict deterministic facts.
- Explicit JSON schema enforcement.
"""

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are an AI assistant interpreting results produced by an autonomous, deterministic machine learning pipeline.
Treat all supplied metrics, counts, and findings as authoritative and unalterable facts.
Do NOT invent new metrics, datasets, experiments, or model scores.
Do NOT suggest ignoring validation errors or overriding data quality rules.
Clearly distinguish between:
- OBSERVED: measurable facts directly present in the supplied summary.
- INTERPRETATION: logical and scientific deductions explaining observed patterns.
- HYPOTHESIS: tentative ideas requiring further experimental verification.
You MUST reply exclusively in valid, parseable JSON conforming to the requested schema.
"""


def build_data_analysis_prompt(analysis_summary: Dict[str, Any]) -> str:
    """Build structured prompt for data analysis interpretation."""
    return f"""Analyze the following dataset analysis summary and provide a structured interpretation:

[DATASET SUMMARY]
- Dataset Name: {analysis_summary.get('dataset_name', 'Unknown')}
- Total Rows: {analysis_summary.get('rows', 0)}
- Total Columns: {analysis_summary.get('columns', 0)}
- Target Column: {analysis_summary.get('target_column', 'None')}
- Problem Type: {analysis_summary.get('problem_type', 'Unknown')}
- Numerical Features: {analysis_summary.get('numerical_features', [])}
- Categorical Features: {analysis_summary.get('categorical_features', [])}
- Missing Values Total: {analysis_summary.get('missing_total', 0)}
- Duplicate Rows: {analysis_summary.get('duplicate_count', 0)}
- Class Distribution: {analysis_summary.get('class_distribution', {})}

Return a JSON object with this exact schema:
{{
    "summary": "1-2 sentence overview of dataset dimensions and task",
    "observations": ["bullet list of observed facts about size, types, and quality"],
    "interpretation": ["interpretation of what the class balance and features imply"],
    "hypotheses": ["hypotheses on which feature transformations might be effective"],
    "recommendations": ["preprocessing suggestions that align with the detected issues"],
    "confidence": "high"
}}
"""


def build_evaluation_prompt(
    evaluation_summary: Dict[str, Any],
    model_rankings: List[Dict[str, Any]],
) -> str:
    """Build structured prompt for model evaluation and gap interpretation."""
    clean_rankings = [
        {
            "rank": m.get("rank"),
            "model": m.get("model_name"),
            "accuracy": m.get("accuracy"),
            "f1_score": m.get("f1_score"),
            "cv_f1_mean": m.get("cv_f1_mean"),
            "gap": m.get("f1_gap"),
        }
        for m in model_rankings[:5]
    ]

    return f"""Evaluate the comparative performance and diagnostic stability of baseline classification models:

[EVALUATION METRICS]
- Best Baseline Model: {evaluation_summary.get('best_model', 'None')}
- Best Test F1: {evaluation_summary.get('best_f1', 0.0)}
- Total Experiments: {evaluation_summary.get('total_experiments', 0)}
- Leaderboard (Top Models):
{clean_rankings}

Return a JSON object with this exact schema:
{{
    "summary": "overview of which model performed best on test data",
    "observations": ["measured facts regarding rankings, F1 scores, and CV gap"],
    "interpretation": ["interpretation of why the top model outperformed others without claiming statistical proof"],
    "hypotheses": ["cautious hypotheses regarding generalization stability or potential overfitting"],
    "recommendations": ["candidate models that warrant hyperparameter optimization"],
    "confidence": "high"
}}
"""


def build_optimization_plan_prompt(
    candidate_models: List[str],
    evaluation_summary: Dict[str, Any],
) -> str:
    """Build structured prompt for proposing hyperparameter tuning strategy."""
    return f"""Propose a high-level hyperparameter optimization plan for candidate models:

[CANDIDATE MODELS]
{candidate_models}

[BASELINE CONTEXT]
- Best Baseline Model: {evaluation_summary.get('best_model', 'None')}
- Best Test F1: {evaluation_summary.get('best_f1', 0.0)}

Return a JSON object with this exact schema:
{{
    "summary": "brief summary of suggested tuning direction",
    "candidate_models": {candidate_models},
    "focus_hyperparameters": ["hyperparameter areas to focus on (e.g. regularization, tree depth, neighbors)"],
    "reasoning": "why these hyperparameters are relevant for the observed models",
    "recommendations": ["guidance for trial allocation and evaluation caution"],
    "confidence": "high"
}}
"""


def build_optimization_explanation_prompt(
    optimization_summary: Dict[str, Any],
    comparison_table: List[Dict[str, Any]],
) -> str:
    """Build structured prompt for interpreting Optuna optimization outcomes."""
    clean_comparison = [
        {
            "model": r.get("model_name"),
            "baseline_cv_f1": r.get("baseline_cv_f1"),
            "optimized_cv_f1": r.get("optimized_cv_f1"),
            "improvement": r.get("improvement"),
            "best_params": r.get("best_params"),
        }
        for r in comparison_table
    ]

    return f"""Analyze the hyperparameter optimization results achieved through Optuna:

[OPTIMIZATION SUMMARY]
- Best Optimized Model: {optimization_summary.get('best_optimized_model', 'None')}
- Best Baseline Model: {optimization_summary.get('best_baseline_model', 'None')}
- Total Trials: {optimization_summary.get('total_hyperparameter_experiments', 0)}
- Comparison:
{clean_comparison}

Return a JSON object with this exact schema:
{{
    "summary": "summary of whether tuning improved upon the baseline and by how much",
    "observations": ["specific observed improvements in CV F1 and key chosen hyperparameters"],
    "interpretation": ["scientific explanation of parameter effects on model capacity"],
    "hypotheses": ["tentative conclusions regarding hyperparameter sensitivity"],
    "recommendations": ["practical takeaways for using the tuned configuration"],
    "confidence": "high"
}}
"""
