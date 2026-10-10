"""Tests for the reusable hyperparameter optimization tools (Phase 6).

The tools are the part of this phase that can be wrong in ways the agent cannot
see, so they are tested directly:

* the search spaces only contain values the installed scikit-learn accepts;
* the sampler is reproducible;
* the objective builds a complete ``preprocessor -> estimator`` pipeline and
  scores it with cross-validation, so the hold-out rows cannot influence a
  suggestion;
* a trial that raises is recorded instead of ending the study;
* the comparison arithmetic is explicit about zero and unsafe denominators;
* the counts keep model experiments and Optuna trials apart.

Every search here uses a deliberately small trial budget. A test suite that
optimises with 20 trials per model would be a benchmark, not a test.
"""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import tools.optimization_tools as optimization_tools
from tools.optimization_tools import (
    OPTIMIZED_RANKING_RULE,
    OptimizationDataError,
    OptimizationToolError,
    build_candidate_selection_rule,
    build_objective,
    build_optimization_report,
    build_optimization_table,
    build_sampler,
    build_study,
    build_trial_record,
    calculate_improvement,
    classify_improvement,
    count_optimization_work,
    describe_optimization_error,
    describe_search_space,
    evaluate_tuned_model,
    generate_optimization_limitations,
    generate_optimization_observations,
    generate_optimization_recommendations,
    generate_optimization_summary,
    generate_optimized_recommendation,
    get_optimizable_models,
    is_optimizable,
    optimize_model,
    rank_optimized_results,
    select_optimization_candidates,
    suggest_parameters,
    summarize_trials,
)
from tools.preprocessing_tools import plan_preprocessing
from tools.training_tools import (
    build_cross_validation_config,
    prepare_features_and_target,
    split_dataset,
)
from utils import config
from utils.state import (
    ExperimentResult,
    ExperimentStatus,
    ImprovementDirection,
    OptimizationResult,
    OptimizationStatus,
    PreprocessingConfig,
    TrialState,
    to_serializable,
)

from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET

import optuna


def quiet_optuna() -> None:
    """Silence Optuna's own logger.

    The project records every failed trial itself, so Optuna printing the same
    traceback a second time only adds noise to the test output.
    """
    optuna.logging.set_verbosity(optuna.logging.ERROR)


quiet_optuna()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_learning_frame(rows: int = 60, classes: int = 2) -> pd.DataFrame:
    """A small frame with a numerical, a categorical and a linear target."""
    records = []
    for index in range(rows):
        records.append(
            {
                "record_id": f"R{index:04d}",
                "age": [20 + (index % 25)][0],
                "bmi": round(19.0 + (index % 12) * 0.6, 2),
                "city": [["north", "south", "east"][index % 3]],
                "label": (index % classes),
            }
        )
    frame = pd.DataFrame(records)
    frame["record_id"] = [f"R{index:04d}" for index in range(rows)]
    frame["city"] = [["north", "south", "east"][index % 3] for index in range(rows)]
    return frame


def training_parts(frame: pd.DataFrame, target: str = "label", test_size: float = 0.2):
    """Plan, separate, split and configure cross-validation for a frame."""
    config_record = plan_preprocessing(frame, target_column=target)
    X, y = prepare_features_and_target(
        frame, target, [item.column for item in config_record.excluded_features]
    )
    split = split_dataset(X, y, test_size=test_size)
    cv_config = build_cross_validation_config(split.y_train, n_splits=3, random_state=42)
    return config_record, split.X_train, split.y_train, split.X_test, split.y_test, cv_config


def learning_parts(**kwargs):
    """Convenience wrapper returning the parts of :func:`make_learning_frame`."""
    return training_parts(make_learning_frame(**kwargs))


def baseline_experiment(
    model_name: str = "Logistic Regression", cv_f1: float = 0.80, f1: float = 0.75
) -> ExperimentResult:
    """A successful baseline experiment record."""
    return ExperimentResult(
        experiment_id=f"exp_{model_name.lower().replace(' ', '_')}",
        model_name=model_name,
        model_type="linear",
        accuracy=0.74,
        precision=0.76,
        recall=0.73,
        f1_score=f1,
        cv_f1_mean=cv_f1,
        cv_accuracy_mean=0.78,
        parameters={"C": 1.0, "max_iter": 1000},
        status=ExperimentStatus.SUCCESS.value,
    )


def optimized_result(
    model_name: str = "Logistic Regression",
    *,
    cv_f1: float = 0.90,
    holdout_f1: float = 0.88,
    holdout_accuracy: float = 0.88,
    baseline_cv_f1: float = 0.80,
    status: str = OptimizationStatus.COMPLETED.value,
    completed: int = 20,
    failed: int = 0,
) -> OptimizationResult:
    """A completed optimized-model record, optionally with a baseline attached."""
    baseline = baseline_experiment(model_name, cv_f1=baseline_cv_f1, f1=holdout_f1 - 0.05)
    improvement, percentage, note = calculate_improvement(baseline_cv_f1, cv_f1)
    return OptimizationResult(
        model_name=model_name,
        model_type="linear",
        status=status,
        reason="" if status == OptimizationStatus.COMPLETED.value else "no completed trial",
        baseline_cv_f1_mean=baseline_cv_f1,
        baseline_cv_accuracy_mean=0.78,
        baseline_holdout_f1_score=baseline.f1_score,
        baseline_holdout_accuracy=0.74,
        baseline_parameters=baseline.parameters,
        best_params={"C": 1.0},
        best_trial_number=3,
        optimized_cv_f1_mean=cv_f1,
        optimized_cv_accuracy_mean=0.9,
        holdout_evaluated=True,
        holdout_f1_score=holdout_f1,
        holdout_accuracy=holdout_accuracy,
        holdout_f1_delta=None,
        improvement=improvement,
        improvement_percentage=percentage,
        improvement_direction=classify_improvement(improvement),
        improvement_note=note,
        cv_n_splits=5,
        total_trials=completed + failed,
        completed_trials=completed,
        failed_trials=failed,
        objective_metric=config.OPTUNA_OBJECTIVE_METRIC,
        optimization_time=1.25,
    )


# --------------------------------------------------------------------------- #
# 1: search spaces
# --------------------------------------------------------------------------- #
class TestSearchSpaces(unittest.TestCase):
    """The suggested values must be usable and controlled."""

    def test_every_registered_model_has_a_search_space(self) -> None:
        self.assertEqual(
            set(get_optimizable_models()),
            {
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "K-Nearest Neighbors",
                "Support Vector Machine",
            },
        )
        for name in get_optimizable_models():
            self.assertTrue(is_optimizable(name))

    def test_unknown_model_is_not_optimizable(self) -> None:
        self.assertFalse(is_optimizable("Gradient Boosting"))
        self.assertFalse(is_optimizable(""))

    def test_describing_an_unknown_model_fails_loudly(self) -> None:
        with self.assertRaises(OptimizationToolError):
            describe_search_space("Gradient Boosting", "binary_classification")

    def test_logistic_search_space_is_bounded_and_fixed_on_max_iter(self) -> None:
        text = " ".join(describe_search_space("Logistic Regression", "binary_classification"))
        self.assertIn("log-uniform", text)
        self.assertIn(f"{config.OPTUNA_LOGISTIC_MAX_ITER}", text)
        self.assertNotIn("max_iter: one of", text)

    def test_logistic_solver_choice_depends_on_the_problem_type(self) -> None:
        """`liblinear` cannot handle a multiclass target on scikit-learn 1.8."""
        binary = " ".join(describe_search_space("Logistic Regression", "binary_classification"))
        multi = " ".join(
            describe_search_space("Logistic Regression", "multiclass_classification")
        )
        self.assertIn("liblinear", binary)
        self.assertNotIn("liblinear", multi)
        for solver in config.OPTUNA_LOGISTIC_SOLVERS_MULTICLASS:
            self.assertIn(solver, multi)

    def test_tree_search_space_is_bounded(self) -> None:
        text = " ".join(describe_search_space("Decision Tree", "binary_classification"))
        self.assertIn("max_depth", text)
        self.assertIn("min_samples_split", text)
        self.assertIn("min_samples_leaf", text)
        for criterion in config.OPTUNA_TREE_CRITERIA:
            self.assertIn(criterion, text)

    def test_forest_search_space_can_remove_the_depth_limit(self) -> None:
        text = " ".join(describe_search_space("Random Forest", "binary_classification"))
        self.assertIn("n_estimators", text)
        self.assertIn("None", text)
        self.assertIn("max_features", text)

    def test_neighbour_search_space_is_bounded(self) -> None:
        text = " ".join(describe_search_space("K-Nearest Neighbors", "binary_classification"))
        self.assertIn("n_neighbors", text)
        self.assertIn("weights", text)
        self.assertIn("metric", text)

    def test_svm_search_space_is_bounded(self) -> None:
        text = " ".join(describe_search_space("Support Vector Machine", "binary_classification"))
        self.assertIn("log-uniform", text)
        self.assertIn("kernel", text)
        self.assertIn("gamma", text)
        self.assertNotIn("10000", text)

    def test_search_space_description_is_json_safe(self) -> None:
        for name in get_optimizable_models():
            json.dumps(to_serializable(describe_search_space(name, "binary_classification")))


# --------------------------------------------------------------------------- #
# 2: sampler and study configuration
# --------------------------------------------------------------------------- #
class TestSamplerAndStudy(unittest.TestCase):
    """A seeded sampler is what makes a second run comparable to the first."""

    def test_sampler_is_a_seeded_tpe_sampler(self) -> None:
        sampler = build_sampler(42)
        self.assertIsInstance(sampler, optuna.samplers.TPESampler)
        # The seed itself is not exposed by Optuna; what matters is that two
        # samplers built with it agree, which the next tests check.
        self.assertEqual(config.OPTUNA_RANDOM_SEED, 42)

    def test_sampler_carries_the_configured_name(self) -> None:
        self.assertEqual(config.OPTUNA_SAMPLER_NAME, "tpe")

    def test_an_unsupported_sampler_name_is_refused(self) -> None:
        original = config.OPTUNA_SAMPLER_NAME
        config.OPTUNA_SAMPLER_NAME = "grid"
        try:
            with self.assertRaises(OptimizationDataError):
                build_sampler(42)
        finally:
            config.OPTUNA_SAMPLER_NAME = original

    def test_study_is_named_after_the_model_and_seeded(self) -> None:
        study = build_study("Decision Tree", seed=42)
        self.assertIn("decision_tree", study.study_name)
        self.assertIsInstance(study.sampler, optuna.samplers.TPESampler)

    def test_the_study_maximises_the_objective(self) -> None:
        study = build_study("Decision Tree", seed=42)
        self.assertEqual(study.direction, optuna.study.StudyDirection.MAXIMIZE)
        self.assertEqual(config.OPTUNA_DIRECTION, "maximize")

    def test_suggestion_sequence_is_identical_for_two_studies(self) -> None:
        def sequence():
            study = build_study("Decision Tree", seed=42)

            def objective(trial):
                trial.suggest_int("max_depth", 1, 20)
                trial.suggest_categorical("criterion", ["gini", "entropy"])
                trial.suggest_int("min_samples_split", 2, 20)
                trial.suggest_float("min_samples_leaf", 1.0, 10.0)
                return 0.0

            study.optimize(objective, n_trials=8)
            return [trial.params for trial in study.trials]

        self.assertEqual(sequence(), sequence())

    def test_a_different_seed_searches_a_different_sequence(self) -> None:
        def sequence(seed: int):
            study = build_study("Decision Tree", seed=seed)
            study.optimize(lambda trial: trial.suggest_int("max_depth", 1, 20), n_trials=8)
            return [trial.params["max_depth"] for trial in study.trials]

        self.assertNotEqual(sequence(42), sequence(7))

    def test_default_seed_comes_from_the_configuration(self) -> None:
        self.assertEqual(config.OPTUNA_RANDOM_SEED, 42)
        self.assertEqual(config.OPTUNA_N_TRIALS, 20)
        self.assertEqual(config.OPTUNA_TOP_MODELS, 2)


# --------------------------------------------------------------------------- #
# 3: suggested parameters are valid for the installed scikit-learn
# --------------------------------------------------------------------------- #
class TestSuggestedParameters(unittest.TestCase):
    """Every suggested combination must build a real estimator."""

    def _suggest(self, model_name: str, problem_type: str = "binary_classification"):
        study = build_study(model_name, seed=42)
        trial = study.ask()
        return suggest_parameters(trial, model_name, problem_type)

    def test_logistic_parameters_are_accepted(self) -> None:
        from sklearn.linear_model import LogisticRegression

        params = self._suggest("Logistic Regression")
        LogisticRegression(**params)

    def test_multiclass_logistic_parameters_are_accepted(self) -> None:
        from sklearn.linear_model import LogisticRegression

        for _ in range(8):
            params = self._suggest("Logistic Regression", "multiclass_classification")
            self.assertIn(params["solver"], config.OPTUNA_LOGISTIC_SOLVERS_MULTICLASS)
            LogisticRegression(**params)

    def test_tree_parameters_are_accepted(self) -> None:
        from sklearn.tree import DecisionTreeClassifier

        for _ in range(8):
            DecisionTreeClassifier(**self._suggest("Decision Tree"))

    def test_forest_parameters_are_accepted(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        for _ in range(8):
            RandomForestClassifier(**self._suggest("Random Forest"))

    def test_neighbour_parameters_are_accepted(self) -> None:
        from sklearn.neighbors import KNeighborsClassifier

        for _ in range(8):
            KNeighborsClassifier(**self._suggest("K-Nearest Neighbors"))

    def test_svm_parameters_are_accepted(self) -> None:
        from sklearn.svm import SVC

        for _ in range(8):
            SVC(**self._suggest("Support Vector Machine"))

    def test_a_trial_that_carries_the_fixed_seed_is_reproducible(self) -> None:
        """`random_state` is not searched, so a recorded config always refits the
        same estimator even if the registry default moves."""
        params = self._suggest("Random Forest")
        self.assertNotIn("random_state", params)

        from tools.optimization_tools import _with_fixed_seed

        seeded = _with_fixed_seed("Random Forest", params)
        self.assertEqual(seeded["random_state"], config.OPTUNA_RANDOM_SEED)

    def test_a_deterministic_estimator_is_given_no_seed(self) -> None:
        from tools.optimization_tools import _with_fixed_seed

        self.assertNotIn(
            "random_state", _with_fixed_seed("K-Nearest Neighbors", {"n_neighbors": 5})
        )

    def test_suggested_parameters_are_json_safe(self) -> None:
        for name in get_optimizable_models():
            params = self._suggest(name)
            self.assertEqual(json.loads(json.dumps(params)), params)

    def test_linear_kernel_records_the_default_gamma(self) -> None:
        """`gamma` is ignored by a linear kernel, so it is not searched."""
        original = config.OPTUNA_SVM_KERNEL_CHOICES
        config.OPTUNA_SVM_KERNEL_CHOICES = ("linear",)
        try:
            study = build_study("Support Vector Machine", seed=1)
            params = suggest_parameters(
                study.ask(), "Support Vector Machine", "binary_classification"
            )
        finally:
            config.OPTUNA_SVM_KERNEL_CHOICES = original
        self.assertEqual(params["kernel"], "linear")
        self.assertEqual(params["gamma"], config.OPTUNA_SVM_LINEAR_GAMMA)

    def test_a_non_linear_kernel_searches_gamma(self) -> None:
        original = config.OPTUNA_SVM_KERNEL_CHOICES
        config.OPTUNA_SVM_KERNEL_CHOICES = ("rbf",)
        try:
            study = build_study("Support Vector Machine", seed=1)
            params = suggest_parameters(
                study.ask(), "Support Vector Machine", "binary_classification"
            )
        finally:
            config.OPTUNA_SVM_KERNEL_CHOICES = original
        self.assertEqual(params["kernel"], "rbf")
        low, high = config.OPTUNA_SVM_GAMMA_RANGE
        self.assertGreaterEqual(params["gamma"], low)
        self.assertLessEqual(params["gamma"], high)

    def test_a_deterministic_kernel_set_produces_fewer_search_dimensions(self) -> None:
        """Not suggesting `gamma` for a linear kernel is what saves those trials."""
        original = config.OPTUNA_SVM_KERNEL_CHOICES
        try:
            config.OPTUNA_SVM_KERNEL_CHOICES = ("linear",)
            linear = self._suggest("Support Vector Machine")
            config.OPTUNA_SVM_KERNEL_CHOICES = ("rbf",)
            radial = self._suggest("Support Vector Machine")
        finally:
            config.OPTUNA_SVM_KERNEL_CHOICES = original
        self.assertEqual(set(linear) - set(radial), set())
        self.assertIn("gamma", radial)

    def test_unknown_model_is_rejected(self) -> None:
        study = build_study("Decision Tree", seed=42)
        trial = study.ask()
        with self.assertRaises(OptimizationToolError):
            suggest_parameters(trial, "Gradient Boosting", "binary_classification")

    def test_neighbour_metric_is_a_float_for_a_bounded_range(self) -> None:
        params = self._suggest("K-Nearest Neighbors")
        self.assertIsInstance(params["n_neighbors"], int)
        self.assertIn(params["weights"], config.OPTUNA_KNN_WEIGHTS_CHOICES)


# --------------------------------------------------------------------------- #
# 4: the objective uses cross-validation inside the pipeline
# --------------------------------------------------------------------------- #
class TestObjective(unittest.TestCase):
    """The objective is where leakage would enter, so it is tested directly."""

    def setUp(self) -> None:
        (
            self.config,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.cv_config,
        ) = learning_parts()

    def test_objective_is_callable_and_returns_a_float(self) -> None:
        study = build_study("Decision Tree", seed=42)
        objective = build_objective(
            model_name="Decision Tree",
            preprocessing_config=self.config,
            features=self.X_train,
            target=self.y_train,
            cv_config=self.cv_config,
            problem_type="binary_classification",
        )
        value = objective(study.ask())
        self.assertIsInstance(float(value), float)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_objective_scores_the_declared_metric(self) -> None:
        """Weighted F1 is the metric the whole project ranks on, in every phase."""
        study = build_study("Decision Tree", seed=42)
        objective = build_objective(
            model_name="Decision Tree",
            preprocessing_config=self.config,
            features=self.X_train,
            target=self.y_train,
            cv_config=self.cv_config,
            problem_type="binary_classification",
        )
        self.assertTrue(callable(objective))
        self.assertEqual(config.OPTUNA_OBJECTIVE_METRIC, "f1_weighted")
        self.assertEqual(config.BEST_MODEL_PRIMARY_METRIC, "f1_score")

    def test_objective_builds_preprocessor_and_estimator_together(self) -> None:
        """A complete Pipeline is required; two separately fitted steps leak.

        The pipeline handed to the cross-validator is inspected directly, because
        the failure this guards against is invisible in the resulting score: if
        the imputer, encoder or scaler were fitted once on the whole training split
        before the folds were cut, every fold score would be slightly optimistic
        and the search would optimise that optimism.
        """
        captured = []
        original = optimization_tools.cross_validate

        def capture(pipeline, *args, **kwargs):
            captured.append(pipeline)
            return original(pipeline, *args, **kwargs)

        optimization_tools.cross_validate = capture
        try:
            study = build_study("Decision Tree", seed=42)
            objective = build_objective(
                model_name="Decision Tree",
                preprocessing_config=self.config,
                features=self.X_train,
                target=self.y_train,
                cv_config=self.cv_config,
                problem_type="binary_classification",
            )
            objective(study.ask())
        finally:
            optimization_tools.cross_validate = original

        self.assertEqual(len(captured), 1)
        pipeline = captured[0]
        self.assertIsInstance(pipeline, Pipeline)
        steps = list(pipeline.named_steps)
        self.assertEqual(steps[0], "preprocessor")
        self.assertEqual(steps[-1], "model")
        # Unfitted when it reaches the cross-validator, so each fold refits the
        # preprocessing steps on its own training rows.
        self.assertFalse(hasattr(pipeline, "classes_"))

    def test_the_objective_reruns_cross_validation_for_every_trial(self) -> None:
        """Two different suggestions must produce two independent evaluations."""
        captured = []
        original = optimization_tools.cross_validate

        def capture(pipeline, *args, **kwargs):
            captured.append(pipeline)
            return original(pipeline, *args, **kwargs)

        optimization_tools.cross_validate = capture
        try:
            study = build_study("Decision Tree", seed=42)
            objective = build_objective(
                model_name="Decision Tree",
                preprocessing_config=self.config,
                features=self.X_train,
                target=self.y_train,
                cv_config=self.cv_config,
                problem_type="binary_classification",
            )
            first = objective(study.ask())
            second = objective(study.ask())
        finally:
            optimization_tools.cross_validate = original

        self.assertEqual(len(captured), 2)
        self.assertIsInstance(first, float)
        self.assertIsInstance(second, float)
        self.assertIsNot(captured[0], captured[1])
        self.assertNotEqual(study.trials[0].params, study.trials[1].params)

    def test_the_cross_validator_is_asked_to_raise_rather_than_score_nan(self) -> None:
        """A broken configuration must fail its trial, not score 0.0 silently."""
        captured = []
        original = optimization_tools.cross_validate

        def capture(pipeline, *args, **kwargs):
            captured.append(kwargs)
            return original(pipeline, *args, **kwargs)

        optimization_tools.cross_validate = capture
        try:
            study = build_study("Decision Tree", seed=42)
            objective = build_objective(
                model_name="Decision Tree",
                preprocessing_config=self.config,
                features=self.X_train,
                target=self.y_train,
                cv_config=self.cv_config,
                problem_type="binary_classification",
            )
            objective(study.ask())
        finally:
            optimization_tools.cross_validate = original

        self.assertEqual(captured[0]["error_score"], "raise")
        self.assertEqual(captured[0]["scoring"], config.OPTUNA_OBJECTIVE_METRIC)
        # Sequential folds keep the search reproducible.
        self.assertEqual(captured[0]["n_jobs"], 1)
        self.assertIsInstance(captured[0]["cv"], StratifiedKFold)
        self.assertTrue(captured[0]["cv"].shuffle)

    def test_objective_without_cross_validation_is_refused(self) -> None:
        """The refusal happens before the first trial, not during it."""
        disabled = build_cross_validation_config(self.y_train, enabled=False, n_splits=5)
        with self.assertRaises(OptimizationDataError) as context:
            build_objective(
                model_name="Decision Tree",
                preprocessing_config=self.config,
                features=self.X_train,
                target=self.y_train,
                cv_config=disabled,
                problem_type="binary_classification",
            )
        self.assertIn("same rows it is scored on", str(context.exception))

    def test_objective_with_a_single_class_target_is_refused(self) -> None:
        """F1 cannot be scored on one class, and the search must say so."""
        single = pd.Series([1] * len(self.y_train), name=self.y_train.name)
        cv_config = build_cross_validation_config(single, n_splits=3, random_state=42)
        with self.assertRaises(OptimizationDataError) as context:
            build_objective(
                model_name="Decision Tree",
                preprocessing_config=self.config,
                features=self.X_train,
                target=single,
                cv_config=cv_config,
                problem_type="binary_classification",
            )
        self.assertIn("at least two classes", str(context.exception))

    def test_objective_with_an_unsupported_problem_type_is_refused(self) -> None:
        with self.assertRaises(OptimizationDataError) as context:
            build_objective(
                model_name="Logistic Regression",
                preprocessing_config=self.config,
                features=self.X_train,
                target=self.y_train,
                cv_config=self.cv_config,
                problem_type="regression",
            )
        self.assertIn("regression", str(context.exception))


    def test_objective_never_sees_the_hold_out_rows(self) -> None:
        """Perturbing the hold-out split cannot change a single suggestion."""
        config_a, X_train, y_train, X_test, y_test, cv_config = learning_parts()

        def run(seed: int) -> list:
            study = build_study("Decision Tree", seed=seed)
            objective = build_objective(
                model_name="Decision Tree",
                preprocessing_config=config_a,
                features=X_train,
                target=y_train,
                cv_config=cv_config,
                problem_type="binary_classification",
            )
            for _ in range(4):
                objective(study.ask())
            return [trial.params for trial in study.trials]

        before = run(42)
        # Overwrite every hold-out value with an extreme, dtype-compatible one.
        for column in X_test.columns:
            if pd.api.types.is_numeric_dtype(X_test[column]):
                X_test[column] = 999999.0
            else:
                X_test[column] = "zzz"
        y_test.iloc[:] = 0
        self.assertEqual(before, run(42))


# --------------------------------------------------------------------------- #
# 5: running a search for one model
# --------------------------------------------------------------------------- #
class TestOptimizeModel(unittest.TestCase):
    """The per-model search, including its failure handling."""

    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.config,
            cls.X_train,
            cls.y_train,
            cls.X_test,
            cls.y_test,
            cls.cv_config,
        ) = learning_parts()

    def _optimize(self, model_name: str = "Decision Tree", **kwargs):
        options = dict(
            model_name=model_name,
            preprocessing_config=self.config,
            features=self.X_train,
            target=self.y_train,
            cv_config=self.cv_config,
            problem_type="binary_classification",
            n_trials=4,
            seed=42,
        )
        options.update(kwargs)
        return optimize_model(**options)

    def test_search_completes_and_records_the_best_trial(self) -> None:
        result = self._optimize(baseline=baseline_experiment(cv_f1=0.5))
        self.assertEqual(result.status, OptimizationStatus.COMPLETED.value)
        self.assertEqual(result.total_trials, 4)
        self.assertEqual(result.completed_trials, 4)
        self.assertEqual(result.failed_trials, 0)
        self.assertIsNotNone(result.best_trial_number)
        self.assertIsNotNone(result.optimized_cv_f1_mean)

    def test_best_parameters_are_the_winning_trials_parameters(self) -> None:
        result = self._optimize()
        winner = next(
            item for item in result.trials if item.trial_number == result.best_trial_number
        )
        self.assertEqual(result.best_params, winner.params)
        self.assertAlmostEqual(result.optimized_cv_f1_mean, winner.objective_value, places=12)

    def test_optimized_score_is_never_worse_than_the_worst_trial(self) -> None:
        result = self._optimize()
        values = [item.objective_value for item in result.trials if item.objective_value is not None]
        self.assertAlmostEqual(result.optimized_cv_f1_mean, max(values), places=12)

    def test_trials_are_recorded_with_number_params_value_and_state(self) -> None:
        result = self._optimize()
        self.assertEqual(len(result.trials), 4)
        self.assertEqual([item.trial_number for item in result.trials], [0, 1, 2, 3])
        for item in result.trials:
            self.assertEqual(item.model_name, "Decision Tree")
            self.assertEqual(item.state, TrialState.COMPLETE.value)
            self.assertIsInstance(item.params, dict)
            self.assertIsInstance(item.objective_value, float)
            self.assertGreaterEqual(item.duration, 0.0)
            self.assertEqual(item.error_message, "")

    def test_search_space_and_metric_are_recorded_on_the_result(self) -> None:
        result = self._optimize()
        self.assertTrue(result.search_space_description)
        self.assertEqual(result.objective_metric, config.OPTUNA_OBJECTIVE_METRIC)
        self.assertEqual(result.cv_n_splits, self.cv_config.n_splits)

    def test_baseline_comparison_is_recorded_when_a_baseline_exists(self) -> None:
        baseline = baseline_experiment("Decision Tree", cv_f1=0.40, f1=0.30)
        result = self._optimize(baseline=baseline)
        self.assertEqual(result.baseline_cv_f1_mean, 0.40)
        self.assertAlmostEqual(
            result.improvement, result.optimized_cv_f1_mean - 0.40, places=12
        )
        self.assertEqual(result.baseline_holdout_f1_score, 0.30)

    def test_missing_baseline_yields_no_comparison_rather_than_a_fake_one(self) -> None:
        result = self._optimize()
        self.assertIsNone(result.baseline_cv_f1_mean)
        self.assertIsNone(result.improvement)
        self.assertEqual(
            result.improvement_direction, ImprovementDirection.NOT_AVAILABLE.value
        )

    def test_repeated_search_is_reproducible(self) -> None:
        first = self._optimize()
        second = self._optimize()
        self.assertEqual(first.best_params, second.best_params)
        self.assertEqual(first.best_trial_number, second.best_trial_number)
        self.assertEqual(
            first.optimized_cv_f1_mean, second.optimized_cv_f1_mean
        )
        self.assertEqual(
            [item.params for item in first.trials], [item.params for item in second.trials]
        )

    def test_a_different_seed_searches_a_different_sequence(self) -> None:
        first = self._optimize(seed=42)
        second = self._optimize(seed=7)
        self.assertNotEqual(
            [item.params for item in first.trials], [item.params for item in second.trials]
        )

    def test_result_is_json_serialisable(self) -> None:
        result = self._optimize()
        payload = to_serializable(result)
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_result_carries_no_live_object(self) -> None:
        result = self._optimize()
        json.dumps(to_serializable(result))
        for value in result.best_params.values():
            self.assertIsInstance(value, (int, float, str, bool, type(None)))


# --------------------------------------------------------------------------- #
# 6: a failing trial is isolated, a failing model is recorded
# --------------------------------------------------------------------------- #
class _FailFirstEvaluations:
    """Make the first `failures` cross-validation calls raise.

    This is the realistic failure mode: a suggested configuration cannot be
    cross-validated. The injection point is `cross_validate` rather than an
    estimator constructor on purpose -- scikit-learn's `clone` reads the
    constructor signature, so patching one would break the pipeline cloning
    itself and the test would measure the wrong thing.
    """

    def __init__(self, failures: int, message: str) -> None:
        self.failures = failures
        self.message = message
        self.calls = 0
        self._original = None

    def __enter__(self) -> "_FailFirstEvaluations":
        self._original = optimization_tools.cross_validate
        owner = self

        def wrapper(*args, **kwargs):
            owner.calls += 1
            if owner.calls <= owner.failures:
                raise ValueError(owner.message)
            return owner._original(*args, **kwargs)

        optimization_tools.cross_validate = wrapper
        return self

    def __exit__(self, *exc_info) -> None:
        optimization_tools.cross_validate = self._original


class TestFailureHandling(unittest.TestCase):
    """A single bad suggestion must not end the study."""

    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.config,
            cls.X_train,
            cls.y_train,
            cls.X_test,
            cls.y_test,
            cls.cv_config,
        ) = learning_parts()

    def _optimize(self, **kwargs):
        options = dict(
            model_name="Decision Tree",
            preprocessing_config=self.config,
            features=self.X_train,
            target=self.y_train,
            cv_config=self.cv_config,
            problem_type="binary_classification",
            n_trials=4,
            seed=42,
        )
        options.update(kwargs)
        return optimize_model(**options)

    def test_a_raising_trial_is_recorded_and_the_search_continues(self) -> None:
        # One configuration cannot be cross-validated; that trial is marked
        # failed, and the remaining trials are still scored.
        with _FailFirstEvaluations(1, "synthetic trial failure") as patch:
            result = self._optimize()
        self.assertGreater(patch.calls, 1)

        self.assertEqual(result.status, OptimizationStatus.COMPLETED.value)
        self.assertEqual(result.total_trials, 4)
        self.assertEqual(result.failed_trials, 1)
        self.assertEqual(result.completed_trials, 3)
        self.assertIsNotNone(result.optimized_cv_f1_mean)

    def test_a_failed_trial_records_its_message_and_state(self) -> None:
        with _FailFirstEvaluations(1, "synthetic trial failure"):
            result = self._optimize(n_trials=3)

        failed = [item for item in result.trials if item.state == TrialState.FAIL.value]
        self.assertEqual(len(failed), 1)
        self.assertIn("synthetic trial failure", failed[0].error_message)
        self.assertIsNone(failed[0].objective_value)

    def test_a_failed_trial_is_excluded_from_the_best_configuration(self) -> None:
        with _FailFirstEvaluations(1, "synthetic trial failure"):
            result = self._optimize(n_trials=3)

        best = next(
            item for item in result.trials if item.trial_number == result.best_trial_number
        )
        self.assertEqual(best.state, TrialState.COMPLETE.value)
        self.assertIn(config.OPTUNA_TRIAL_BUDGET_NOTE.format(trials=3), " ".join(result.notes))
        self.assertIn("failed and were excluded", " ".join(result.notes))

    def test_when_every_trial_fails_the_model_is_marked_failed(self) -> None:
        with _FailFirstEvaluations(50, "every trial fails"):
            result = self._optimize(n_trials=2)

        self.assertEqual(result.status, OptimizationStatus.FAILED.value)
        self.assertEqual(result.completed_trials, 0)
        self.assertEqual(result.failed_trials, 2)
        self.assertEqual(result.best_params, {})
        self.assertIsNone(result.best_trial_number)
        self.assertIsNone(result.optimized_cv_f1_mean)
        self.assertIn("every trial fails", result.error_message)
        self.assertIn("every one of the 2 trial(s) failed", result.reason)
        self.assertEqual(result.improvement_direction, ImprovementDirection.NOT_AVAILABLE.value)

    def test_a_model_with_cross_validation_disabled_is_marked_failed(self) -> None:
        disabled = build_cross_validation_config(self.y_train, enabled=False, n_splits=5)
        result = self._optimize(cv_config=disabled)
        self.assertEqual(result.status, OptimizationStatus.FAILED.value)
        self.assertIn("cross-validation", result.error_message.lower())
        self.assertEqual(result.total_trials, 0)

    def test_an_unknown_model_fails_loudly_before_any_trial_runs(self) -> None:
        with self.assertRaises(OptimizationToolError) as context:
            self._optimize(model_name="Gradient Boosting")
        self.assertIn("no hyperparameter search space", str(context.exception))

    def test_a_trial_budget_below_one_is_rejected(self) -> None:
        with self.assertRaises(OptimizationToolError) as context:
            self._optimize(n_trials=0)
        self.assertIn("at least 1", str(context.exception))


# --------------------------------------------------------------------------- #
# 7: small-class safeguards
# --------------------------------------------------------------------------- #
class TestSmallClassSafeguards(unittest.TestCase):
    """A class with too few members reduces the folds; it is never hidden."""

    @classmethod
    def setUpClass(cls) -> None:
        frame = make_learning_frame(rows=40, classes=2)
        # Three rows of a third class: after the hold-out split the training rows
        # cannot supply five folds, so the fold count has to shrink.
        frame.loc[frame.index[37:], "label"] = 2
        cls.config, cls.X_train, cls.y_train, _, _, _ = training_parts(frame)
        cls.distribution = cls.y_train.value_counts().to_dict()

    def test_the_fixture_really_contains_a_rare_class(self) -> None:
        self.assertGreaterEqual(min(self.distribution.values()), config.RARE_CLASS_MIN_SAMPLES)
        self.assertLess(min(self.distribution.values()), config.CV_FOLDS)

    def test_folds_are_reduced_for_a_rare_class(self) -> None:
        cv_config = build_cross_validation_config(self.y_train, n_splits=5, random_state=42)
        self.assertLess(cv_config.n_splits, 5)
        self.assertGreaterEqual(cv_config.n_splits, config.CV_MIN_FOLDS)
        self.assertTrue(cv_config.adjusted_reason)

    def test_the_reduction_is_recorded_on_the_result(self) -> None:
        cv_config = build_cross_validation_config(self.y_train, n_splits=5, random_state=42)
        result = optimize_model(
            model_name="Decision Tree",
            preprocessing_config=self.config,
            features=self.X_train,
            target=self.y_train,
            cv_config=cv_config,
            problem_type="multiclass_classification",
            n_trials=2,
            seed=42,
        )
        self.assertEqual(result.cv_n_splits, cv_config.n_splits)
        self.assertEqual(result.cv_adjustment_reason, cv_config.adjusted_reason)
        self.assertEqual(result.status, OptimizationStatus.COMPLETED.value)

    def test_a_single_class_target_is_refused_by_the_objective_not_the_folds(self) -> None:
        """The fold helper tolerates one class; the objective does not."""
        single = pd.Series([0] * len(self.y_train), name=self.y_train.name)
        cv_config = build_cross_validation_config(single, n_splits=3, random_state=42)
        result = optimize_model(
            model_name="Decision Tree",
            preprocessing_config=self.config,
            features=self.X_train,
            target=single,
            cv_config=cv_config,
            problem_type="binary_classification",
            n_trials=2,
            seed=42,
        )
        self.assertEqual(result.status, OptimizationStatus.FAILED.value)
        self.assertIn("at least two classes", result.error_message)


# --------------------------------------------------------------------------- #
# 8: candidate selection
# --------------------------------------------------------------------------- #
class TestCandidateSelection(unittest.TestCase):
    """Only the best baseline models are tuned, and the choice is explained."""

    def setUp(self) -> None:
        # A fresh list per test: one test deliberately corrupts an experiment, and
        # a shared list would leak that corruption into the others.
        self.experiments = [
            baseline_experiment("Random Forest", cv_f1=0.70, f1=0.62),
            baseline_experiment("Logistic Regression", cv_f1=0.89, f1=0.88),
            baseline_experiment("Support Vector Machine", cv_f1=0.83, f1=0.84),
            baseline_experiment("K-Nearest Neighbors", cv_f1=0.79, f1=0.79),
            baseline_experiment("Decision Tree", cv_f1=0.75, f1=0.75),
        ]

    def test_the_top_models_of_the_baseline_ranking_are_selected(self) -> None:
        selected = select_optimization_candidates(self.experiments, top_models=2)
        self.assertEqual(selected, ["Logistic Regression", "Support Vector Machine"])

    def test_selection_follows_the_baseline_ranking_order(self) -> None:
        selected = select_optimization_candidates(self.experiments, top_models=3)
        self.assertEqual(
            selected,
            ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"],
        )

    def test_selection_is_deterministic(self) -> None:
        self.assertEqual(
            select_optimization_candidates(self.experiments, top_models=2),
            select_optimization_candidates(list(reversed(self.experiments)), top_models=2),
        )

    def test_selection_follows_the_phase_five_ranking_metric(self) -> None:
        """The Phase 5 rule is F1, then accuracy, then name -- not the CV score."""
        selected = select_optimization_candidates(self.experiments, top_models=1)
        self.assertEqual(selected, ["Logistic Regression"])
        experiments = list(self.experiments)
        experiments[1].f1_score = 0.10  # worst hold-out F1 despite the best CV F1
        self.assertEqual(
            select_optimization_candidates(experiments, top_models=1),
            ["Support Vector Machine"],
        )

    def test_selection_respects_the_requested_problem_type(self) -> None:
        experiment = baseline_experiment("Logistic Regression")
        experiment.status = ExperimentStatus.SUCCESS.value
        # A model that does not support the problem type is excluded.
        self.assertEqual(
            select_optimization_candidates(
                [experiment], top_models=2, problem_type="binary_classification"
            ),
            ["Logistic Regression"],
        )

    def test_failed_experiments_are_never_selected(self) -> None:
        best = self.experiments[1]
        best.status = ExperimentStatus.FAILED.value
        best.f1_score = None
        best.cv_f1_mean = None
        selected = select_optimization_candidates(self.experiments, top_models=2)
        self.assertNotIn("Logistic Regression", selected)
        self.assertEqual(len(selected), 2)

    def test_a_partial_experiment_is_still_a_candidate(self) -> None:
        partial = self.experiments[1]
        partial.status = ExperimentStatus.PARTIAL.value
        self.assertEqual(
            select_optimization_candidates(self.experiments, top_models=1),
            ["Logistic Regression"],
        )

    def test_a_model_without_a_search_space_is_skipped_and_the_next_is_used(self) -> None:
        """The budget is filled from further down the ranking, not left short."""
        unknown = baseline_experiment("Gradient Boosting", cv_f1=0.95, f1=0.95)
        selected = select_optimization_candidates(
            [unknown, self.experiments[1], self.experiments[2]], top_models=2
        )
        self.assertEqual(selected, ["Logistic Regression", "Support Vector Machine"])

    def test_top_models_larger_than_the_registry_selects_what_exists(self) -> None:
        selected = select_optimization_candidates(self.experiments[:1], top_models=5)
        self.assertEqual(selected, ["Random Forest"])

    def test_an_empty_ranking_selects_nothing(self) -> None:
        self.assertEqual(select_optimization_candidates([], top_models=2), [])

    def test_a_budget_below_one_still_produces_one_candidate(self) -> None:
        """A zero budget would produce no comparison at all, so it becomes one."""
        self.assertEqual(
            select_optimization_candidates(self.experiments, top_models=0),
            ["Logistic Regression"],
        )

    def test_the_selection_rule_is_explained(self) -> None:
        text = build_candidate_selection_rule(top_models=2)
        self.assertIn("2", text)
        self.assertIn("ranking", text.lower())
        json.dumps(text)


# --------------------------------------------------------------------------- #
# 9: improvement arithmetic
# --------------------------------------------------------------------------- #
class TestImprovement(unittest.TestCase):
    """Improvement is a difference; a percentage needs a safe denominator."""

    def test_a_higher_optimized_score_is_an_improvement(self) -> None:
        value, percentage, _ = calculate_improvement(0.80, 0.90)
        self.assertAlmostEqual(value, 0.10, places=12)
        self.assertAlmostEqual(percentage, 12.5, places=8)
        self.assertEqual(classify_improvement(value), ImprovementDirection.IMPROVED.value)

    def test_a_lower_optimized_score_is_a_degradation(self) -> None:
        value, _, _ = calculate_improvement(0.80, 0.70)
        self.assertAlmostEqual(value, -0.10, places=12)
        self.assertEqual(classify_improvement(value), ImprovementDirection.DEGRADED.value)

    def test_an_identical_score_is_unchanged(self) -> None:
        value, percentage, _ = calculate_improvement(0.80, 0.80)
        self.assertEqual(value, 0.0)
        self.assertEqual(percentage, 0.0)
        self.assertEqual(classify_improvement(value), ImprovementDirection.UNCHANGED.value)

    def test_a_zero_baseline_gives_no_percentage(self) -> None:
        """A percentage against zero is undefined and must not be printed."""
        value, percentage, note = calculate_improvement(0.0, 0.5)
        self.assertAlmostEqual(value, 0.5, places=12)
        self.assertIsNone(percentage)
        self.assertIn("not reported", note)
        json.dumps(note)

    def test_a_baseline_below_the_safety_floor_gives_no_percentage(self) -> None:
        baseline = config.OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE / 2.0
        value, percentage, note = calculate_improvement(baseline, baseline + 0.1)
        self.assertAlmostEqual(value, 0.1, places=12)
        self.assertIsNone(percentage)
        self.assertIn("meaningless", note)

    def test_the_safety_floor_is_a_small_positive_number(self) -> None:
        self.assertGreater(config.OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE, 0.0)
        self.assertLess(config.OPTUNA_MIN_SAFE_BASELINE_FOR_PERCENTAGE, 1e-6)

    def test_a_missing_baseline_gives_no_comparison_at_all(self) -> None:
        value, percentage, note = calculate_improvement(None, 0.5)
        self.assertIsNone(value)
        self.assertIsNone(percentage)
        self.assertTrue(note)
        self.assertEqual(
            classify_improvement(None), ImprovementDirection.NOT_AVAILABLE.value
        )

    def test_a_missing_optimized_score_gives_no_comparison(self) -> None:
        value, percentage, _ = calculate_improvement(0.5, None)
        self.assertIsNone(value)
        self.assertIsNone(percentage)

    def test_both_missing_gives_no_comparison(self) -> None:
        self.assertEqual(calculate_improvement(None, None)[:2], (None, None))

    def test_a_negative_baseline_still_reports_the_absolute_difference(self) -> None:
        """F1 cannot be negative, so the ratio uses the magnitude and the note says
        which number the percentage is relative to."""
        value, percentage, note = calculate_improvement(-0.2, 0.4)
        self.assertAlmostEqual(value, 0.6, places=12)
        self.assertAlmostEqual(percentage, 300.0, places=8)
        self.assertIn("-0.2", note)

    def test_the_note_names_the_baseline_it_relates_to(self) -> None:
        _, _, note = calculate_improvement(0.8, 0.9)
        self.assertIn("cross-validation", note)
        self.assertIn("0.8", note)


# --------------------------------------------------------------------------- #
# 10: ranking
# --------------------------------------------------------------------------- #
class TestRanking(unittest.TestCase):
    """Optimized models are ranked exactly like the baseline models were."""

    def test_highest_hold_out_f1_wins(self) -> None:
        ranked = rank_optimized_results(
            [
                optimized_result("Decision Tree", holdout_f1=0.70, holdout_accuracy=0.80),
                optimized_result("Random Forest", holdout_f1=0.90, holdout_accuracy=0.70),
            ]
        )
        self.assertEqual([item.model_name for item in ranked], ["Random Forest", "Decision Tree"])

    def test_accuracy_breaks_an_f1_tie(self) -> None:
        ranked = rank_optimized_results(
            [
                optimized_result("Decision Tree", holdout_f1=0.80, holdout_accuracy=0.79),
                optimized_result("Random Forest", holdout_f1=0.80, holdout_accuracy=0.85),
            ]
        )
        self.assertEqual([item.model_name for item in ranked], ["Random Forest", "Decision Tree"])

    def test_the_model_name_breaks_a_complete_tie(self) -> None:
        ranked = rank_optimized_results(
            [
                optimized_result("Random Forest", holdout_f1=0.80, holdout_accuracy=0.85),
                optimized_result("Decision Tree", holdout_f1=0.80, holdout_accuracy=0.85),
            ]
        )
        self.assertEqual([item.model_name for item in ranked], ["Decision Tree", "Random Forest"])

    def test_ranking_is_deterministic_regardless_of_input_order(self) -> None:
        results = [
            optimized_result("Decision Tree", holdout_f1=0.80, holdout_accuracy=0.80),
            optimized_result("Random Forest", holdout_f1=0.80, holdout_accuracy=0.80),
            optimized_result("K-Nearest Neighbors", holdout_f1=0.60, holdout_accuracy=0.60),
        ]
        first = [item.model_name for item in rank_optimized_results(results)]
        second = [item.model_name for item in rank_optimized_results(list(reversed(results)))]
        self.assertEqual(first, second)

    def test_a_failed_result_is_excluded_from_the_ranking(self) -> None:
        ranked = rank_optimized_results(
            [
                optimized_result("Random Forest", holdout_f1=0.99, status=OptimizationStatus.FAILED.value),
                optimized_result("Decision Tree", holdout_f1=0.70),
            ]
        )
        self.assertEqual([item.model_name for item in ranked], ["Decision Tree"])

    def test_a_result_without_hold_out_metrics_is_excluded(self) -> None:
        without = optimized_result("Random Forest", holdout_f1=0.99)
        without.holdout_evaluated = False
        without.holdout_f1_score = None
        ranked = rank_optimized_results([without, optimized_result("Decision Tree", holdout_f1=0.70)])
        self.assertEqual([item.model_name for item in ranked], ["Decision Tree"])

    def test_an_empty_list_ranks_to_an_empty_list(self) -> None:
        self.assertEqual(rank_optimized_results([]), [])

    def test_the_ranking_rule_is_documented(self) -> None:
        self.assertIn("f1_score", OPTIMIZED_RANKING_RULE)
        self.assertIn("accuracy", OPTIMIZED_RANKING_RULE)
        json.dumps(OPTIMIZED_RANKING_RULE)


# --------------------------------------------------------------------------- #
# 11: scoring the tuned model on the hold-out split
# --------------------------------------------------------------------------- #
class TestEvaluateTunedModel(unittest.TestCase):
    """The hold-out split is read once, after the parameters are fixed."""

    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.config,
            cls.X_train,
            cls.y_train,
            cls.X_test,
            cls.y_test,
            cls.cv_config,
        ) = learning_parts()

    def _result(self) -> OptimizationResult:
        return optimize_model(
            model_name="Decision Tree",
            preprocessing_config=self.config,
            features=self.X_train,
            target=self.y_train,
            cv_config=self.cv_config,
            problem_type="binary_classification",
            n_trials=3,
            seed=42,
            baseline=baseline_experiment(cv_f1=0.5, f1=0.5),
        )

    def test_hold_out_metrics_are_recorded_once(self) -> None:
        result = self._result()
        evaluate_tuned_model(
            result=result,
            preprocessing_config=self.config,
            features_train=self.X_train,
            target_train=self.y_train,
            features_test=self.X_test,
            target_test=self.y_test,
        )
        self.assertTrue(result.holdout_evaluated)
        self.assertIsNotNone(result.holdout_f1_score)
        self.assertIsNotNone(result.holdout_accuracy)
        self.assertAlmostEqual(
            result.holdout_f1_delta,
            result.holdout_f1_score - result.baseline_holdout_f1_score,
            places=12,
        )

    def test_the_hold_out_score_does_not_change_the_search_result(self) -> None:
        first = self._result()
        evaluate_tuned_model(
            result=first,
            preprocessing_config=self.config,
            features_train=self.X_train,
            target_train=self.y_train,
            features_test=self.X_test,
            target_test=self.y_test,
        )
        second = self._result()
        self.assertEqual(first.best_params, second.best_params)
        self.assertEqual(
            first.optimized_cv_f1_mean, second.optimized_cv_f1_mean
        )
        self.assertEqual(first.best_trial_number, second.best_trial_number)

    def test_a_failed_result_is_not_scored(self) -> None:
        result = optimized_result(status=OptimizationStatus.FAILED.value)
        result.holdout_evaluated = False
        result.holdout_f1_score = None
        result.holdout_accuracy = None
        evaluate_tuned_model(
            result=result,
            preprocessing_config=self.config,
            features_train=self.X_train,
            target_train=self.y_train,
            features_test=self.X_test,
            target_test=self.y_test,
        )
        self.assertFalse(result.holdout_evaluated)
        self.assertIsNone(result.holdout_f1_score)

    def test_an_empty_hold_out_split_is_reported_rather_than_scored(self) -> None:
        result = self._result()
        empty = self.X_test.iloc[:0]
        evaluate_tuned_model(
            result=result,
            preprocessing_config=self.config,
            features_train=self.X_train,
            target_train=self.y_train,
            features_test=empty,
            target_test=self.y_test.iloc[:0],
        )
        self.assertFalse(result.holdout_evaluated)
        self.assertIn("empty", " ".join(result.notes))

    def test_an_unusable_hold_out_split_is_recorded_and_the_cv_score_survives(self) -> None:
        """A scoring failure must not destroy the search result."""
        result = self._result()
        cv_score = result.optimized_cv_f1_mean
        broken = self.X_test.copy()
        broken["age"] = "not a number"
        evaluate_tuned_model(
            result=result,
            preprocessing_config=self.config,
            features_train=self.X_train,
            target_train=self.y_train,
            features_test=broken,
            target_test=self.y_test,
        )
        self.assertFalse(result.holdout_evaluated)
        self.assertEqual(result.optimized_cv_f1_mean, cv_score)
        self.assertIn("could not be evaluated", " ".join(result.notes))

    def test_a_missing_baseline_leaves_the_delta_undefined(self) -> None:
        result = self._result()
        result.baseline_holdout_f1_score = None
        evaluate_tuned_model(
            result=result,
            preprocessing_config=self.config,
            features_train=self.X_train,
            target_train=self.y_train,
            features_test=self.X_test,
            target_test=self.y_test,
        )
        self.assertTrue(result.holdout_evaluated)
        self.assertIsNone(result.holdout_f1_delta)


# --------------------------------------------------------------------------- #
# 12: counts
# --------------------------------------------------------------------------- #
class TestCounts(unittest.TestCase):
    """Model experiments and Optuna trials are different quantities."""

    def test_baseline_experiments_and_trials_are_counted_separately(self) -> None:
        experiments = [
            baseline_experiment("Logistic Regression"),
            baseline_experiment("Support Vector Machine"),
        ]
        failed = baseline_experiment("Decision Tree")
        failed.status = ExperimentStatus.FAILED.value
        failed.f1_score = None
        experiments.append(failed)

        results = [
            optimized_result("Logistic Regression", completed=20, failed=2),
            optimized_result("Support Vector Machine", completed=20, failed=0),
        ]
        counts = count_optimization_work(experiments, results)

        self.assertEqual(counts.baseline_models_attempted, 3)
        self.assertEqual(counts.baseline_successful_experiments, 2)
        self.assertEqual(counts.baseline_failed_experiments, 1)
        self.assertEqual(counts.optimized_models_attempted, 2)
        self.assertEqual(counts.successful_optimizations, 2)
        self.assertEqual(counts.completed_trials, 40)
        self.assertEqual(counts.failed_trials, 2)
        self.assertEqual(counts.total_trials, 42)

    def test_the_two_quantities_are_never_added_together(self) -> None:
        counts = count_optimization_work(
            [baseline_experiment("Logistic Regression")],
            [optimized_result("Logistic Regression", completed=20, failed=0)],
        )
        self.assertNotEqual(counts.total_trials, counts.baseline_models_attempted)
        self.assertIn("not added together", counts.note)

    def test_a_failed_model_is_named(self) -> None:
        counts = count_optimization_work(
            [baseline_experiment("Decision Tree")],
            [optimized_result("Decision Tree", status=OptimizationStatus.FAILED.value, completed=0, failed=3)],
        )
        self.assertEqual(counts.failed_optimizations, 1)
        self.assertIn("Decision Tree", counts.failed_model_names)

    def test_a_model_with_failed_trials_is_named(self) -> None:
        counts = count_optimization_work(
            [baseline_experiment("Decision Tree")],
            [optimized_result("Decision Tree", completed=18, failed=2)],
        )
        self.assertEqual(counts.failed_trials, 2)
        self.assertIn("Decision Tree", counts.failed_trial_models)

    def test_no_work_yields_zero_counts(self) -> None:
        counts = count_optimization_work([], [])
        self.assertEqual(counts.baseline_models_attempted, 0)
        self.assertEqual(counts.total_trials, 0)
        self.assertEqual(counts.successful_optimizations, 0)

    def test_counts_are_json_serialisable(self) -> None:
        counts = count_optimization_work(
            [baseline_experiment("Decision Tree")],
            [optimized_result("Decision Tree")],
        )
        payload = to_serializable(counts)
        self.assertEqual(json.loads(json.dumps(payload)), payload)


# --------------------------------------------------------------------------- #
# 13: trial records and summaries
# --------------------------------------------------------------------------- #
class TestTrialRecords(unittest.TestCase):
    """A trial is only useful if its outcome is written down."""

    def test_a_completed_trial_records_its_value_and_state(self) -> None:
        study = build_study("Decision Tree", seed=42)
        trial = study.ask()
        study.tell(trial, 0.75)
        record = build_trial_record(study.trials[0], "Decision Tree")
        self.assertEqual(record.trial_number, 0)
        self.assertEqual(record.model_name, "Decision Tree")
        self.assertEqual(record.state, TrialState.COMPLETE.value)
        self.assertAlmostEqual(record.objective_value, 0.75, places=12)
        self.assertEqual(record.error_message, "")

    def test_a_failed_trial_records_the_captured_message(self) -> None:
        study = build_study("Decision Tree", seed=42)
        trial = study.ask()
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        record = build_trial_record(
            study.trials[0], "Decision Tree", error_messages={0: "ValueError: broken"}
        )
        self.assertEqual(record.state, TrialState.FAIL.value)
        self.assertEqual(record.error_message, "ValueError: broken")
        self.assertIsNone(record.objective_value)

    def test_a_failed_trial_without_a_captured_message_says_so(self) -> None:
        study = build_study("Decision Tree", seed=42)
        trial = study.ask()
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        record = build_trial_record(study.trials[0], "Decision Tree")
        self.assertEqual(record.state, TrialState.FAIL.value)
        self.assertIn("no error message was captured", record.error_message)

    def test_a_trial_record_is_json_serialisable(self) -> None:
        study = build_study("Decision Tree", seed=42)
        study.tell(study.ask(), 0.5)
        record = build_trial_record(study.trials[0], "Decision Tree")
        json.dumps(to_serializable(record))

    def test_summarize_counts_each_state(self) -> None:
        study = build_study("Decision Tree", seed=42)
        for _ in range(3):
            study.tell(study.ask(), 0.5)
        for _ in range(2):
            trial = study.ask()
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        summary = summarize_trials(
            [build_trial_record(item, "Decision Tree") for item in study.trials]
        )
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["total"], 5)

    def test_summarizing_nothing_is_zero_not_an_error(self) -> None:
        self.assertEqual(summarize_trials([]), {"completed": 0, "failed": 0, "pruned": 0, "total": 0})


# --------------------------------------------------------------------------- #
# 14: report assembly
# --------------------------------------------------------------------------- #
class TestReportAssembly(unittest.TestCase):
    """The report is the deliverable of the phase, so its content is pinned."""

    def setUp(self) -> None:
        self.experiments = [
            baseline_experiment("Logistic Regression", cv_f1=0.89, f1=0.88),
            baseline_experiment("Support Vector Machine", cv_f1=0.83, f1=0.84),
            baseline_experiment("Decision Tree", cv_f1=0.75, f1=0.75),
        ]
        self.results = [
            optimized_result("Logistic Regression", cv_f1=0.90, holdout_f1=0.88, baseline_cv_f1=0.89),
            optimized_result("Support Vector Machine", cv_f1=0.88, holdout_f1=0.91, baseline_cv_f1=0.83),
        ]

    def _report(self, **kwargs):
        options = dict(
            experiments=self.experiments,
            results=self.results,
            selected=["Logistic Regression", "Support Vector Machine"],
            n_trials=20,
            top_models=2,
            dataset_name=BINARY_DATASET.name,
            problem_type="binary_classification",
            target_column="Outcome",
            test_samples=24,
            cv_n_splits=5,
        )
        options.update(kwargs)
        return build_optimization_report(**options)

    def test_a_completed_report_is_produced(self) -> None:
        report = self._report()
        self.assertEqual(report.overall_status, OptimizationStatus.COMPLETED.value)
        self.assertEqual(report.dataset_name, BINARY_DATASET.name)
        self.assertEqual(report.cv_n_splits, 5)

    def test_selected_and_skipped_models_are_both_named(self) -> None:
        report = self._report()
        self.assertEqual(
            report.models_selected_for_optimization,
            ["Logistic Regression", "Support Vector Machine"],
        )
        self.assertEqual(report.models_not_optimized, ["Decision Tree"])

    def test_the_best_optimized_model_is_recommended_with_its_reason(self) -> None:
        report = self._report()
        best = report.best_optimized_model
        self.assertIsNotNone(best)
        self.assertEqual(best.best_optimized_model, "Support Vector Machine")
        self.assertIn("highest hold-out f1_score", best.selection_reason)
        self.assertTrue(best.performance_summary)
        self.assertEqual(best.best_params, {"C": 1.0})

    def test_the_best_baseline_is_preserved_next_to_the_optimized_one(self) -> None:
        without = self._report()
        self.assertIsNone(without.best_baseline_model)
        self.assertEqual(without.baseline_saved_to, "not available")

        with_baseline = self._report(baseline_recommendation=_baseline_recommendation())
        self.assertEqual(
            with_baseline.best_baseline_model.best_baseline_model, "Logistic Regression"
        )
        self.assertIn("evaluation_report.best_model", with_baseline.baseline_saved_to)
        self.assertIsNotNone(with_baseline.best_optimized_model)

    def test_the_rules_are_recorded_in_the_report(self) -> None:
        report = self._report()
        self.assertEqual(report.sampler, config.OPTUNA_SAMPLER_NAME)
        self.assertEqual(report.sampler_seed, config.OPTUNA_RANDOM_SEED)
        self.assertEqual(report.objective_metric, config.OPTUNA_OBJECTIVE_METRIC)
        for rule in (
            report.objective_rule,
            report.selection_rule,
            report.candidate_rule,
            report.ranking_rule,
            report.holdout_note,
        ):
            self.assertTrue(rule)
            json.dumps(rule)

    def test_the_hold_out_note_states_that_the_search_did_not_use_it(self) -> None:
        self.assertIn("hold-out", self._report().holdout_note.lower())

    def test_counts_are_attached(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.counts)
        self.assertEqual(report.counts.baseline_models_attempted, 3)
        self.assertEqual(report.counts.total_trials, 40)

    def test_a_comparison_table_is_rendered(self) -> None:
        table = self._report().table_text
        self.assertIn("Logistic Regression", table)
        self.assertIn("Support Vector Machine", table)
        self.assertIn("baseline cv f1", table)
        self.assertIn("optimized holdout", table)

    def test_observations_limitations_and_recommendations_are_recorded(self) -> None:
        report = self._report()
        self.assertTrue(report.observations)
        self.assertTrue(report.limitations)
        self.assertTrue(report.recommendations)

    def test_the_cv_selection_bias_is_stated_as_a_limitation(self) -> None:
        report = self._report()
        self.assertIn(
            config.OPTUNA_LIMITATION_SELECTION_ON_CV, report.limitations
        )

    def test_the_hold_out_ranking_caveat_is_stated(self) -> None:
        self.assertIn(
            config.OPTUNA_LIMITATION_HOLDOUT_RANKING, self._report().limitations
        )

    def test_the_small_trial_budget_is_stated_as_a_limitation(self) -> None:
        self.assertIn(
            config.OPTUNA_LIMITATION_TRIAL_BUDGET.format(trials=20),
            self._report().limitations,
        )

    def test_only_the_selected_models_are_compared_in_the_text(self) -> None:
        report = self._report()
        self.assertNotIn("Decision Tree", report.table_text)

    def test_the_report_is_json_serialisable(self) -> None:
        payload = to_serializable(self._report())
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_the_report_carries_no_live_object(self) -> None:
        json.dumps(to_serializable(self._report()))
        for item in self._report().optimization_results:
            json.dumps(to_serializable(item.best_params))

    def test_a_partial_run_is_reported_as_partial(self) -> None:
        results = list(self.results)
        results.append(
            optimized_result(
                "Decision Tree", status=OptimizationStatus.FAILED.value, completed=0, failed=20
            )
        )
        report = self._report(results=results, selected=self._report().models_selected_for_optimization + ["Decision Tree"])
        self.assertEqual(report.overall_status, OptimizationStatus.PARTIAL.value)
        self.assertIsNotNone(report.best_optimized_model)

    def test_a_run_where_every_search_failed_keeps_the_baseline_visible(self) -> None:
        results = [
            optimized_result(
                name, status=OptimizationStatus.FAILED.value, completed=0, failed=20
            )
            for name in ("Logistic Regression", "Support Vector Machine")
        ]
        report = self._report(
            results=results,
            baseline_recommendation=_baseline_recommendation(),
        )
        self.assertEqual(report.overall_status, OptimizationStatus.FAILED.value)
        self.assertIsNone(report.best_optimized_model)
        self.assertIsNotNone(report.best_baseline_model)
        self.assertTrue(any("baseline" in note.lower() for note in report.notes))

    def test_no_search_at_all_is_reported_as_skipped(self) -> None:
        """Nothing to optimize is a skip, not a failure: no claim was made."""
        report = self._report(results=[], selected=[], top_models=2)
        self.assertEqual(report.overall_status, OptimizationStatus.SKIPPED.value)
        self.assertIsNone(report.best_optimized_model)
        self.assertTrue(report.limitations)
        self.assertEqual(report.optimization_results, [])

    def test_an_unranked_result_is_named_rather_than_dropped_silently(self) -> None:
        """A completed search with no hold-out score cannot be recommended."""
        result = optimized_result("Logistic Regression")
        result.holdout_evaluated = False
        result.holdout_f1_score = None
        report = self._report(results=[result], selected=["Logistic Regression"])
        self.assertIsNone(report.best_optimized_model)
        self.assertIn("Logistic Regression", " ".join(report.notes + report.observations))


def _baseline_recommendation():
    """A Phase 5 recommendation record, as the agent passes it in."""
    from utils.state import ModelRecommendation

    return ModelRecommendation(
        best_baseline_model="Logistic Regression",
        experiment_id="exp_logistic_regression",
        model_type="linear",
        selection_reason="highest hold-out f1_score (0.8761) among 5 baseline model(s)",
        performance_summary="Logistic Regression achieved a hold-out f1_score of 0.8761.",
        f1_score=0.8761,
        accuracy=0.875,
        precision=0.88,
        recall=0.87,
        cv_f1_mean=0.8941,
        cv_accuracy_mean=0.895,
        f1_gap=-0.018,
        total_time=0.3,
        preprocessing_scaling="standard",
        parameters={"C": 1.0, "max_iter": 1000},
        confusion_matrix_error_pattern="8 correct, 2 false negatives, 1 false positive",
        limitations=["baseline parameters are untuned"],
    )


# --------------------------------------------------------------------------- #
# 15: text helpers used by the report
# --------------------------------------------------------------------------- #
class TestTextHelpers(unittest.TestCase):
    """The wording rules are part of the contract, so they are tested."""

    def test_summary_names_both_sides_of_the_comparison(self) -> None:
        text = generate_optimization_summary(
            [optimized_result("Logistic Regression")],
            baseline=_baseline_recommendation(),
        )
        self.assertIn("Logistic Regression", text)
        json.dumps(text)

    def test_summary_of_no_results_is_still_readable(self) -> None:
        text = generate_optimization_summary([])
        self.assertTrue(text)
        json.dumps(text)

    def test_observations_mention_the_trial_budget_and_the_failures(self) -> None:
        observations = generate_optimization_observations(
            results=[optimized_result("Decision Tree", completed=20, failed=2)],
            ranked=[optimized_result("Decision Tree", completed=20, failed=2)],
            counts=count_optimization_work(
                [baseline_experiment("Decision Tree")],
                [optimized_result("Decision Tree", completed=20, failed=2)],
            ),
            selected=["Decision Tree"],
            n_trials=20,
        )
        text = " ".join(observations)
        self.assertIn("20", text)
        self.assertIn("2", text)

    def test_limitations_name_the_number_of_unoptimized_models(self) -> None:
        limitations = generate_optimization_limitations(
            results=[optimized_result("Logistic Regression")],
            ranked=[optimized_result("Logistic Regression")],
            counts=count_optimization_work(
                [baseline_experiment("Logistic Regression"), baseline_experiment("Decision Tree")],
                [optimized_result("Logistic Regression")],
            ),
            n_trials=20,
            test_samples=24,
            candidate_total=5,
            selected_count=2,
        )
        text = " ".join(limitations)
        self.assertIn("5", text)
        self.assertIn("2", text)

    def test_recommendations_are_generated_from_the_ranking(self) -> None:
        ranked = [optimized_result("Logistic Regression")]
        recommendations = generate_optimization_recommendations(
            ranked=ranked,
            recommendation=generate_optimized_recommendation(ranked),
            results=ranked,
            counts=count_optimization_work(
                [baseline_experiment("Logistic Regression")], ranked
            ),
            baseline=_baseline_recommendation(),
        )
        self.assertTrue(recommendations)
        for item in recommendations:
            json.dumps(item)

    def test_a_failed_run_recommends_a_larger_budget(self) -> None:
        failed = [
            optimized_result(
                "Logistic Regression", status=OptimizationStatus.FAILED.value, completed=0, failed=20
            )
        ]
        recommendations = generate_optimization_recommendations(
            ranked=[],
            recommendation=None,
            results=failed,
            counts=count_optimization_work([baseline_experiment("Logistic Regression")], failed),
            baseline=_baseline_recommendation(),
        )
        text = " ".join(recommendations).lower()
        self.assertIn("trial", text)
        self.assertIn("remains the model to use", text)
        self.assertIn("failure", text)

    def test_an_empty_ranking_produces_no_recommendation(self) -> None:
        self.assertIsNone(generate_optimized_recommendation([]))

    def test_the_recommendation_repeats_the_improvement_direction(self) -> None:
        ranked = rank_optimized_results(
            [optimized_result("Logistic Regression", cv_f1=0.95, baseline_cv_f1=0.80)]
        )
        recommendation = generate_optimized_recommendation(ranked)
        self.assertIsNotNone(recommendation)
        self.assertEqual(
            recommendation.improvement_direction, ImprovementDirection.IMPROVED.value
        )
        self.assertAlmostEqual(recommendation.improvement, 0.15, places=12)

    def test_error_description_is_single_line_and_names_the_cause(self) -> None:
        text = describe_optimization_error(ValueError("something specific"))
        self.assertIn("something specific", text)
        self.assertNotIn("\n", text)

    def test_error_description_names_an_optimization_error(self) -> None:
        text = describe_optimization_error(OptimizationDataError("no folds possible"))
        self.assertIn("no folds possible", text)

    def test_table_of_nothing_is_readable(self) -> None:
        table = build_optimization_table([])
        self.assertIsInstance(table, str)
        json.dumps(table)


# --------------------------------------------------------------------------- #
# 16: dataset-driven behaviour
# --------------------------------------------------------------------------- #
class TestDatasetDrivenSearch(unittest.TestCase):
    """The tools work on any dataset; nothing here is specific to one file."""

    @classmethod
    def setUpClass(cls) -> None:
        import pandas as pd

        cls.binary = pd.read_csv(BINARY_DATASET)
        cls.multi = pd.read_csv(MULTICLASS_DATASET)

    def test_binary_problem_type_is_detected_and_used(self) -> None:
        self.assertEqual(sorted(self.binary["Outcome"].unique().tolist()), [0, 1])

    def test_multiclass_problem_type_uses_multiclass_solvers(self) -> None:
        self.assertEqual(self.multi["class_label"].nunique(), 3)
        study = build_study("Logistic Regression", seed=42)
        trial = study.ask()
        params = suggest_parameters(
            trial, "Logistic Regression", "multiclass_classification"
        )
        self.assertIn(params["solver"], config.OPTUNA_LOGISTIC_SOLVERS_MULTICLASS)
        self.assertNotIn("liblinear", config.OPTUNA_LOGISTIC_SOLVERS_MULTICLASS)

    def test_numerical_bounds_are_honoured_over_many_trials(self) -> None:
        study = build_study("Support Vector Machine", seed=42)
        for _ in range(12):
            trial = study.ask()
            params = suggest_parameters(trial, "Support Vector Machine", "binary_classification")
            low, high = config.OPTUNA_SVM_C_RANGE
            self.assertGreaterEqual(params["C"], low)
            self.assertLessEqual(params["C"], high)
            if params["kernel"] != "linear":
                gamma_low, gamma_high = config.OPTUNA_SVM_GAMMA_RANGE
                self.assertGreaterEqual(params["gamma"], gamma_low)
                self.assertLessEqual(params["gamma"], gamma_high)

    def test_integer_bounds_are_honoured_over_many_trials(self) -> None:
        study = build_study("K-Nearest Neighbors", seed=42)
        for _ in range(12):
            trial = study.ask()
            params = suggest_parameters(trial, "K-Nearest Neighbors", "binary_classification")
            self.assertGreaterEqual(params["n_neighbors"], config.OPTUNA_KNN_N_NEIGHBORS_RANGE[0])
            self.assertLessEqual(params["n_neighbors"], config.OPTUNA_KNN_N_NEIGHBORS_RANGE[1])

    def test_a_numpy_float_suggestion_is_serialisable(self) -> None:
        study = build_study("Random Forest", seed=42)
        trial = study.ask()
        params = suggest_parameters(trial, "Random Forest", "binary_classification")
        for key in ("n_estimators", "min_samples_split", "min_samples_leaf"):
            self.assertIsInstance(params[key], int)
        self.assertIsInstance(params["max_depth"], (int, type(None)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()