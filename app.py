"""Application entrypoint for the Autonomous Multi-Agent Machine Learning System.

Phase 7 (LangGraph orchestration):
- `python app.py --graph`              Run via LangGraph multi-agent graph.
- `python app.py --graph --no-optimize`  Graph mode without optimization.
- `python app.py --graph --dataset <path>` Specify a custom dataset.

Phase 6 (sequential, default):
- `python app.py`                      Sequential pipeline through Phase 6.
- `python app.py --no-optimize`        Sequential pipeline through Phase 5.
- `streamlit run app.py`               Interactive Streamlit UI.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from utils.config import Config
from utils.logger import get_logger
from utils.state import AgentState, WorkflowStatus

logger = get_logger("App")


def display_cli_status(run_optimization: bool = True) -> None:
    """Display project status and run demonstration workflow through Phase 6 in terminal mode.

    Args:
        run_optimization: If True (default) runs the Hyperparameter Optimization Agent.
                          Pass False (via --no-optimize) to stop after Evaluation.
    """
    banner = "=" * 80
    print(banner)
    print("        Autonomous Multi-Agent Machine Learning System")
    print("         PHASE 6: HYPERPARAMETER OPTIMIZATION (Optuna)")
    print(banner)

    sample_csv = Config.DATASETS_DIR / "sample_classification.csv"
    if sample_csv.exists():
        print(f"\n[1] Running Data Analysis Agent on: {sample_csv.name}...")
        state = AgentState(dataset_path=str(sample_csv))
        state = DataAnalysisAgent().run(state)
        print(f"    - Target Column : {state.target_column} ({state.target_detection_method})")
        print(f"    - Problem Type  : {state.problem_type}")
        print(f"    - Dimensions    : {state.dataset_shape[0]} rows, {state.dataset_shape[1]} columns")

        print("\n[2] Running Preprocessing Agent...")
        state = PreprocessingAgent().run(state)
        print(f"    - Selected Features: {len(state.selected_numerical_features)} numerical, {len(state.selected_categorical_features)} categorical")
        print(f"    - Excluded Features: {state.excluded_features}")

        print("\n[3] Running Model Selection Agent...")
        state = ModelSelectionAgent().run(state)
        print(f"    - Candidate Models Selected: {state.selected_models}")

        print("\n[4] Running Training Agent (with 5-fold CV & Stratified Train/Test Split)...")
        state = TrainingAgent().run(state)

        print("\n[5] Running Evaluation Agent (Gap Diagnostics & Performance Synthesis)...")
        state = EvaluationAgent().run(state)

        report = state.evaluation_report
        print("\n" + "-" * 80)
        print(f"{'Rank':<6}{'Model':<24}{'Accuracy':<10}{'F1-Score':<10}{'CV F1':<10}{'F1 Gap':<10}{'Gap Class':<10}")
        print("-" * 80)
        for row in state.model_evaluations:
            cv_f1 = f"{row['cv_f1_mean']:.4f}" if row.get("cv_f1_mean") is not None else "N/A"
            f1_gap = f"{row['f1_gap']:+.4f}" if row.get("f1_gap") is not None else "N/A"
            rank_str = str(row.get("rank", "-"))
            gap_cls = row.get("f1_gap_classification", "N/A")
            print(f"{rank_str:<6}{row['model_name']:<24}{row['accuracy']:<10.4f}{row['f1_score']:<10.4f}{cv_f1:<10}{f1_gap:<10}{gap_cls:<10}")
        print("-" * 80)

        print(f"\n[+] BEST BASELINE MODEL : {report.get('best_model')}")
        print(f"[+] SELECTION REASON    : {report.get('selection_reason')}")

        time_info = report.get("execution_time_analysis", {})
        print(f"\n[+] EXECUTION TELEMETRY :")
        print(f"    - Total Experiment Time : {time_info.get('total_experiment_time', 0.0):.4f}s")
        print(f"    - Average Experiment Time: {time_info.get('average_experiment_time', 0.0):.4f}s")
        print(f"    - Fastest Model          : {time_info.get('fastest_model')} ({time_info.get('fastest_time', 0.0):.4f}s)")
        print(f"    - Slowest Model          : {time_info.get('slowest_model')} ({time_info.get('slowest_time', 0.0):.4f}s)")
        print(f"    - Total Experiments      : {time_info.get('total_experiments', 0)} ({time_info.get('successful_experiments', 0)} succeeded)")

        print(f"\n[+] EVALUATION OBSERVATIONS :")
        for obs in report.get("observations", []):
            print(f"    * {obs}")

        # ----------------------------------------------------------------
        # Phase 6: Hyperparameter Optimization
        # ----------------------------------------------------------------
        if run_optimization:
            print("\n" + "=" * 80)
            print(f"[6] Running Hyperparameter Optimization Agent (Optuna, {Config.OPTUNA_N_TRIALS} trials × top-{Config.OPTUNA_TOP_MODELS} models)...")
            state = OptimizationAgent().run(state)

            print("\n" + "-" * 80)
            print(f"{'Model':<28}{'Baseline CV F1':<18}{'Optimized CV F1':<18}{'Improvement':<14}{'Trials':<10}{'Status':<12}")
            print("-" * 80)
            for row in state.optimization_comparison:
                b_cv = f"{row.get('baseline_cv_f1', 0.0):.4f}"
                o_cv = f"{row.get('optimized_cv_f1', 0.0):.4f}"
                imp = row.get("improvement", 0.0)
                imp_pct = row.get("improvement_pct")
                imp_str = f"{imp:+.4f}" + (f" ({imp_pct:+.1f}%)" if imp_pct is not None else "")
                trials = f"{row.get('successful_trials', 0)}/{row.get('n_trials', 0)}"
                status = row.get("status", "?")
                print(f"{row.get('model_name', ''):<28}{b_cv:<18}{o_cv:<18}{imp_str:<14}{trials:<10}{status:<12}")
            print("-" * 80)

            print(f"\n[+] BEST OPTIMIZED MODEL  : {state.best_optimized_model}")
            print(f"[+] BEST BASELINE MODEL   : {state.best_baseline_model}")
            if state.best_optimized_model:
                for r in state.optimization_comparison:
                    if r.get("model_name") == state.best_optimized_model:
                        hp = r.get("best_params", {})
                        print(f"[+] BEST HYPERPARAMETERS  : {json.dumps(hp, indent=2)}")
                        hm = r.get("optimized_f1")
                        if hm is not None:
                            print(f"[+] HOLDOUT F1 (optimized): {hm:.4f}")
                        break
            print(f"\n[+] TOTAL HYPERPARAMETER TRIALS: {state.hyperparameter_experiments}")
            print(f"    - Successful : {state.successful_trials}")
            print(f"    - Failed     : {state.failed_trials}")
    else:
        print(f"\n[-] Sample dataset not found at: {sample_csv}")

    print("\n" + banner)
    if run_optimization:
        print("Phase 6 complete. Hyperparameter optimization finished with Optuna.")
    else:
        print("Phase 5 complete. Evaluation report synthesized without retraining models.")
        print("Tip: Remove --no-optimize to run Phase 6 hyperparameter optimization.")
    print(banner)


def render_streamlit_ui() -> None:
    """Render interactive Streamlit dashboard interface."""
    import dashboard
    dashboard.render_dashboard()






def display_graph_status(
    dataset_path: str = None,
    target_column: str = None,
    run_optimization: bool = True,
    run_llm: bool = False,
) -> None:
    """Run the workflow via LangGraph orchestration and display results.

    Args:
        dataset_path: Optional CSV path; defaults to sample_classification.csv.
        target_column: Optional target column name to predict.
        run_optimization: If True, the optimization node is included.
        run_llm: If True, optional Phase-8 Ollama LLM nodes are activated.
    """
    from graph.workflow import run_graph

    banner = "=" * 80
    print(banner)
    print("        Autonomous Multi-Agent Machine Learning System")
    print("         PHASE 8: LOCAL LLM INTERPRETATION & ORCHESTRATION")
    print(banner)

    csv_path = dataset_path or str(Config.DATASETS_DIR / "sample_classification.csv")
    print(f"\n[GRAPH] Running LangGraph workflow on: {csv_path}")
    if target_column:
        print(f"[GRAPH] Target Column    : {target_column}")
    print(f"[GRAPH] Optimization enabled: {run_optimization}")
    print(f"[GRAPH] Local LLM enabled  : {run_llm}")

    result = run_graph(
        dataset_path=csv_path,
        target_column=target_column,
        enable_optimization=run_optimization,
        max_retries=Config.LANGGRAPH_MAX_RETRIES,
        enable_llm=run_llm,
    )

    print(f"\n[GRAPH] Workflow Status  : {result.workflow_status}")
    print(f"[GRAPH] Nodes Completed  : {result.completed_nodes}")
    if result.failed_node:
        print(f"[GRAPH] Failed Node      : {result.failed_node}")
        print(f"[GRAPH] Error            : {result.error_message}")
    print(f"[GRAPH] Optimization     : {'executed' if result.optimization_executed else 'skipped'}")
    print(f"[GRAPH] Best Baseline    : {result.best_baseline_model}")
    if result.optimization_executed:
        print(f"[GRAPH] Best Optimized   : {result.best_optimized_model}")
    if result.retry_counts:
        print(f"[GRAPH] Retry Counts     : {result.retry_counts}")
    if result.mlflow_run_id:
        print(f"[GRAPH] MLflow Run ID    : {result.mlflow_run_id}")

    if run_llm and result.final_agent_state:
        state = result.final_agent_state
        print("\n" + "-" * 80)
        print("          PHASE 8 LOCAL LLM INTERPRETATIONS")
        print("-" * 80)
        print(f"[LLM] Global Status: {state.llm_status}")
        if state.data_analysis_explanation:
            print(f"\n[LLM - Data Analysis Summary]:\n{state.data_analysis_explanation.get('summary', 'N/A')}")
        if state.evaluation_explanation:
            print(f"\n[LLM - Evaluation Interpretation]:\n{state.evaluation_explanation.get('summary', 'N/A')}")
        if state.optimization_explanation:
            print(f"\n[LLM - Optimization Interpretation]:\n{state.optimization_explanation.get('summary', 'N/A')}")

    print("\n" + banner)
    print("Phase 8 graph execution complete.")
    print(banner)


def display_benchmark_status(
    dataset_name: Optional[str] = None,
    all_datasets: bool = False,
) -> None:
    """Run Phase 11 comparative benchmark (Conventional vs Autonomous) and display results."""
    from benchmarks.benchmark_config import BENCHMARK_DATASETS, BenchmarkDatasetSpec
    from benchmarks.comparison import BenchmarkRunner

    banner = "=" * 80
    print(banner)
    print("        Autonomous Multi-Agent Machine Learning System")
    print("   PHASE 11: CONVENTIONAL VS AUTONOMOUS ML WORKFLOW COMPARISON")
    print(banner)

    runner = BenchmarkRunner()

    if dataset_name and not all_datasets:
        # Determine dataset spec
        key = dataset_name.lower().strip()
        if key in BENCHMARK_DATASETS:
            spec = BENCHMARK_DATASETS[key]
        elif "multi" in key:
            spec = BENCHMARK_DATASETS["multiclass_classification"]
        elif "class" in key or "binary" in key:
            spec = BENCHMARK_DATASETS["binary_classification"]
        else:
            # Custom path
            custom_path = Path(dataset_name)
            spec = BenchmarkDatasetSpec(
                name=custom_path.name,
                file_path=custom_path,
                target_column=None,
                problem_type="unknown",
                expected_sha256="",
            )

        print(f"\n[BENCHMARK] Running comparative benchmark on: {spec.name}...")
        comp = runner.run_benchmark_for_dataset(spec)
        results = {spec.name: comp}
        runner.save_json_report(results)
        runner.save_csv_report(results)
        runner.generate_plots(results)
    else:
        print("\n[BENCHMARK] Running comparative benchmark across all standard datasets...")
        results = runner.run_all()

    # Print clean comparison tables
    for key, comp in results.items():
        print("\n" + "-" * 80)
        print(f" DATASET: {comp.dataset_name} ({comp.task_type})")
        print(f" SHA-256: {comp.dataset_hash}")
        print("-" * 80)
        print(f"{'Metric':<20} | {'Conventional':<15} | {'Autonomous':<15} | {'Difference':<15}")
        print("-" * 75)
        print(f"{'Selected Model':<20} | {comp.conventional.selected_display_name:<15} | {comp.autonomous.selected_display_name:<15} | {'-':<15}")
        print(f"{'Holdout Accuracy':<20} | {comp.conventional.holdout_accuracy:<15.4f} | {comp.autonomous.holdout_accuracy:<15.4f} | {comp.accuracy_difference:<+15.4f}")
        print(f"{'Holdout Precision':<20} | {comp.conventional.holdout_precision:<15.4f} | {comp.autonomous.holdout_precision:<15.4f} | {comp.precision_difference:<+15.4f}")
        print(f"{'Holdout Recall':<20} | {comp.conventional.holdout_recall:<15.4f} | {comp.autonomous.holdout_recall:<15.4f} | {comp.recall_difference:<+15.4f}")
        print(f"{'Holdout F1-Score':<20} | {comp.conventional.holdout_f1:<15.4f} | {comp.autonomous.holdout_f1:<15.4f} | {comp.f1_difference:<+15.4f}")
        print(f"{'CV F1-Score':<20} | {comp.conventional.cv_f1_mean:<15.4f} | {comp.autonomous.cv_f1_mean:<15.4f} | {comp.cv_f1_difference:<+15.4f}")
        print(f"{'Runtime (seconds)':<20} | {comp.conventional.total_runtime_seconds:<15.2f} | {comp.autonomous.total_runtime_seconds:<15.2f} | {comp.runtime_difference:<+15.2f}")
        print(f"{'Experiment Count':<20} | {comp.conventional.experiment_count:<15} | {comp.autonomous.experiment_count:<15} | {comp.experiment_difference:<+15}")

    print("\n" + banner)
    print("Benchmark artifacts created:")
    print("  - JSON Report : reports/phase11_comparison.json")
    print("  - CSV Report  : reports/phase11_comparison.csv")
    print("  - F1 Plot     : reports/phase11_f1_comparison.png")
    print("  - Runtime Plot: reports/phase11_runtime_comparison.png")
    print("  - Exp Plot    : reports/phase11_experiment_comparison.png")
    print(banner)


def main() -> None:
    """Main execution router."""
    is_streamlit = "streamlit" in sys.modules and any(
        "streamlit" in arg for arg in sys.argv
    )

    if is_streamlit:
        render_streamlit_ui()
    else:
        parser = argparse.ArgumentParser(
            description="Autonomous Multi-Agent ML System — Phase 11"
        )
        parser.add_argument(
            "--no-optimize",
            action="store_true",
            default=False,
            help="Skip hyperparameter optimization.",
        )
        parser.add_argument(
            "--graph",
            action="store_true",
            default=False,
            help="Use LangGraph orchestration instead of the sequential pipeline.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default=None,
            help="Path to a custom CSV dataset (used with --graph).",
        )
        parser.add_argument(
            "--target",
            type=str,
            default=None,
            help="Name of the target column to predict (used with --graph).",
        )
        parser.add_argument(
            "--llm",
            action="store_true",
            default=False,
            help="Activate optional Phase-8 local Ollama LLM interpretation layer.",
        )
        parser.add_argument(
            "--benchmark",
            action="store_true",
            default=False,
            help="Run Phase 11 comparative benchmark between Conventional and Autonomous ML workflows.",
        )
        parser.add_argument(
            "--benchmark-dataset",
            type=str,
            default=None,
            help="Run benchmark on a specific dataset ('classification', 'multiclass', or path).",
        )
        parser.add_argument(
            "--benchmark-all",
            action="store_true",
            default=False,
            help="Run benchmark across all standard datasets.",
        )
        args, _ = parser.parse_known_args()

        if args.benchmark or args.benchmark_all or args.benchmark_dataset:
            display_benchmark_status(
                dataset_name=args.benchmark_dataset,
                all_datasets=args.benchmark_all or (args.benchmark and not args.benchmark_dataset),
            )
        elif args.graph:
            display_graph_status(
                dataset_path=args.dataset,
                target_column=args.target,
                run_optimization=not args.no_optimize,
                run_llm=args.llm,
            )
        else:
            display_cli_status(run_optimization=not args.no_optimize)


if __name__ == "__main__":
    main()
