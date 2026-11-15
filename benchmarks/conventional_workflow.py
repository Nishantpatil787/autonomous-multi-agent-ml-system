"""Conventional Machine Learning Workflow for Phase 11 Benchmarking.

Implements a deterministic, realistic traditional ML pipeline:
1. Load dataset & inspect features/target.
2. Apply duplicate handling and standard train/test split.
3. Configure preprocessing fitted strictly on training data.
4. Evaluate candidate model configurations using 5-fold cross-validation.
5. Select the best configuration based on CV F1-score.
6. Fit the selected model on training data and evaluate once on the held-out test set.
7. Record all wall-clock timings, experiment counts, and metrics.

Does NOT use Optuna, LangGraph, Ollama, or MLflow for decision-making.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from benchmarks.benchmark_config import CONVENTIONAL_SEARCH_CONFIGS
from tools import data_tools, model_tools, preprocessing_tools, training_tools
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("ConventionalWorkflow")


@dataclass
class ConventionalExperimentRecord:
    """Record of a single model configuration evaluated during the conventional search."""

    model_key: str
    display_name: str
    params: Dict[str, Any]
    cv_f1_mean: float
    cv_accuracy_mean: float
    cv_folds: int
    training_time_seconds: float
    is_baseline_default: bool = False


@dataclass
class ConventionalResult:
    """Complete structured output from the conventional ML workflow."""

    dataset_path: str
    target_column: str
    problem_type: str
    train_samples: int
    test_samples: int
    feature_count: int
    selected_model: str
    selected_display_name: str
    selected_params: Dict[str, Any]
    holdout_accuracy: float
    holdout_precision: float
    holdout_recall: float
    holdout_f1: float
    cv_f1_mean: float
    cv_accuracy_mean: float
    total_runtime_seconds: float
    search_runtime_seconds: float
    experiment_count: int
    evaluated_configurations: List[ConventionalExperimentRecord] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    confusion_matrix_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "feature_count": self.feature_count,
            "selected_model": self.selected_model,
            "selected_display_name": self.selected_display_name,
            "selected_params": self.selected_params,
            "holdout_accuracy": self.holdout_accuracy,
            "holdout_precision": self.holdout_precision,
            "holdout_recall": self.holdout_recall,
            "holdout_f1": self.holdout_f1,
            "cv_f1_mean": self.cv_f1_mean,
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "total_runtime_seconds": round(self.total_runtime_seconds, 4),
            "search_runtime_seconds": round(self.search_runtime_seconds, 4),
            "experiment_count": self.experiment_count,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": self.confusion_matrix_labels,
        }


class ConventionalMLWorkflow:
    """Deterministic traditional machine learning workflow benchmark."""

    def __init__(
        self,
        random_state: int = Config.RANDOM_STATE,
        test_size: float = Config.TEST_SIZE,
        cv_folds: int = Config.CV_FOLDS,
    ) -> None:
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

    def run(
        self,
        dataset_path: Union[str, Path],
        target_column: Optional[str] = None,
        custom_search_configs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> ConventionalResult:
        """Execute the conventional ML workflow on the specified dataset.

        Args:
            dataset_path: Path to dataset CSV file.
            target_column: Name of target column. If None, uses heuristic detection.
            custom_search_configs: Optional override for manual parameter configurations.

        Returns:
            ConventionalResult containing metrics, timing, and experiment count.
        """
        start_wall_time = time.perf_counter()
        csv_path = Path(dataset_path)

        # 1. Load data & duplicate handling
        df, duplicates_removed = training_tools.load_training_dataframe(
            csv_path, duplicate_strategy=Config.DEFAULT_DUPLICATE_STRATEGY
        )

        # 2. Target identification
        if not target_column:
            target_info = data_tools.analyze_target(df)
            target_column = target_info["target_column"]

        problem_type, _ = data_tools.detect_problem_type(df[target_column])

        # 3. Features & Target separation
        X, y = training_tools.prepare_features_and_target(df, target_column)

        # 4. Same train/test split convention as autonomous workflow
        X_train, X_test, y_train, y_test, is_stratified, split_reason = training_tools.split_dataset(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 5. Preprocessing setup (equivalent to Phase 3)
        num_cols = list(X_train.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X_train.select_dtypes(exclude=[np.number]).columns)

        filter_res = preprocessing_tools.filter_usable_features(
            feature_names=list(X_train.columns),
            numerical_features=num_cols,
            categorical_features=cat_cols,
        )
        selected_num = filter_res["selected_numerical_features"]
        selected_cat = filter_res["selected_categorical_features"]
        excluded = filter_res["excluded_features"]

        if excluded:
            X_train = X_train.drop(columns=[c for c in excluded if c in X_train.columns])
            X_test = X_test.drop(columns=[c for c in excluded if c in X_test.columns])

        base_prep_config = preprocessing_tools.build_pipeline_config(
            selected_numerical_features=selected_num,
            selected_categorical_features=selected_cat,
            excluded_features=filter_res["excluded_features"],
            exclusion_reasons=filter_res["exclusion_reasons"],
            numerical_imputation_strategy="median",
            categorical_imputation_strategy="most_frequent",
            encoding_strategy="one_hot",
            scaling_strategy="standard_scaler",
            scale_numeric=True,
        )

        # 6. Evaluate Candidate Model Configurations
        search_configs = custom_search_configs or CONVENTIONAL_SEARCH_CONFIGS
        evaluated_records: List[ConventionalExperimentRecord] = []
        search_start_time = time.perf_counter()

        candidate_model_keys = list(search_configs.keys())

        for model_key in candidate_model_keys:
            model_def = model_tools.get_model_definition(model_key)
            scaling_required = bool(model_def["scaling_required"])
            display_name = model_def["display_name"]

            # Adapt scaling flag strictly per model definition
            model_prep = dict(base_prep_config)
            model_prep["scale_numeric"] = scaling_required

            configs_to_test = search_configs[model_key]

            for config_params in configs_to_test:
                t0 = time.perf_counter()
                preprocessor = preprocessing_tools.recreate_column_transformer_from_config(model_prep)
                estimator = model_tools.build_estimator(model_key, param_overrides=config_params)

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", estimator),
                    ]
                )

                # 5-fold cross-validation on X_train / y_train
                cv_res = training_tools.run_cross_validation(
                    pipeline,
                    X_train,
                    y_train,
                    n_splits=self.cv_folds,
                    random_state=self.random_state,
                )
                t_elapsed = time.perf_counter() - t0

                cv_f1 = cv_res.get("cv_f1_mean") or 0.0
                cv_acc = cv_res.get("cv_accuracy_mean") or 0.0

                record = ConventionalExperimentRecord(
                    model_key=model_key,
                    display_name=display_name,
                    params=config_params,
                    cv_f1_mean=cv_f1,
                    cv_accuracy_mean=cv_acc,
                    cv_folds=cv_res.get("cv_folds", 0),
                    training_time_seconds=t_elapsed,
                )
                evaluated_records.append(record)

        search_runtime = time.perf_counter() - search_start_time

        # 7. Model Selection
        # Select best configuration by cv_f1_mean (descending), then cv_accuracy (descending), then lower time
        evaluated_records.sort(
            key=lambda r: (r.cv_f1_mean, r.cv_accuracy_mean, -r.training_time_seconds),
            reverse=True,
        )
        best_record = evaluated_records[0]

        # 8. Final Holdout Evaluation
        # Fit the best configuration on full training set, evaluate on test set
        best_model_def = model_tools.get_model_definition(best_record.model_key)
        final_prep = dict(base_prep_config)
        final_prep["scale_numeric"] = bool(best_model_def["scaling_required"])

        final_preprocessor = preprocessing_tools.recreate_column_transformer_from_config(final_prep)
        final_estimator = model_tools.build_estimator(
            best_record.model_key, param_overrides=best_record.params
        )

        final_pipeline = Pipeline(
            steps=[
                ("preprocessor", final_preprocessor),
                ("model", final_estimator),
            ]
        )
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)

        holdout_metrics = training_tools.evaluate_classifier(y_test, y_pred)
        cm_data = training_tools.calculate_confusion_matrix(y_test, y_pred)

        total_runtime = time.perf_counter() - start_wall_time

        return ConventionalResult(
            dataset_path=str(csv_path),
            target_column=target_column,
            problem_type=problem_type,
            train_samples=len(X_train),
            test_samples=len(X_test),
            feature_count=len(X_train.columns),
            selected_model=best_record.model_key,
            selected_display_name=best_record.display_name,
            selected_params=best_record.params,
            holdout_accuracy=holdout_metrics["accuracy"],
            holdout_precision=holdout_metrics["precision"],
            holdout_recall=holdout_metrics["recall"],
            holdout_f1=holdout_metrics["f1_score"],
            cv_f1_mean=best_record.cv_f1_mean,
            cv_accuracy_mean=best_record.cv_accuracy_mean,
            total_runtime_seconds=total_runtime,
            search_runtime_seconds=search_runtime,
            experiment_count=len(evaluated_records),
            evaluated_configurations=evaluated_records,
            confusion_matrix=cm_data["matrix"],
            confusion_matrix_labels=cm_data["labels"],
        )
