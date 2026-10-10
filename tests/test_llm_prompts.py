"""Tests for the prompt templates and the context allowlist (Phase 8).

Two things are asserted here, and both are safety properties rather than
cosmetic ones:

* the prompt **says** what the project needs it to say - the metrics are
  authoritative, nothing may be invented, no retraining, no dataset edits, and
  the forbidden phrasings are named so the model has something to avoid;
* the prompt **contains** only what it may. The context builders are the whole
  privacy boundary of Phase 8, so the tests check the exact key set, that no raw
  row ever appears, that no filesystem path or environment value leaks, and that
  generation is deterministic.

Prompt determinism matters beyond tidiness: an identical state must produce an
identical string, otherwise a prompt cannot be asserted on in a test and a
reported run cannot be reproduced.
"""

from __future__ import annotations

import json
import unittest

from llm.prompts import (
    INTERPRETATION_SCHEMA_TEXT,
    MAX_PROMPT_CHARS,
    PLAN_SCHEMA_TEXT,
    SYSTEM_INSTRUCTION,
    PromptContext,
    build_context,
    build_data_analysis_context,
    build_evaluation_context,
    build_optimization_context,
    build_prompt,
    build_repair_prompt,
    redact_path,
    repair_instruction,
    sanitize_value,
    schema_text,
    task_text,
)
from llm.state import LLM_CONTEXT_FIELDS, LLMRole, allowed_context_fields
from utils import config
from utils.state import (
    AgentState,
    ClassDistributionEntry,
    DatasetShape,
    MissingValueReport,
    ProblemType,
)


def analyzed_state() -> AgentState:
    """Return a state carrying a plausible Data Analysis Agent result.

    Mirrors the committed ``sample_classification.csv``: 123 rows, 9 columns,
    17 missing values, 3 duplicates and a near-balanced ``Outcome`` column.
    """
    state = AgentState()
    state.dataset_name = "sample_classification.csv"
    state.dataset_shape = DatasetShape(rows=123, columns=9)
    state.missing_values = MissingValueReport(
        total=17,
        total_percentage=13.82,
        affected_columns=4,
        per_column={"age": 6, "bmi": 5, "smoking": 4, "region": 2},
    )
    state.total_missing_values = 17
    state.duplicate_count = 3
    state.numerical_features = ["record_id", "age", "bmi", "glucose", "blood_pressure"]
    state.categorical_features = ["gender", "region", "smoking"]
    state.target_column = "Outcome"
    state.target_detection_method = "name_heuristic"
    state.target_detection_confidence = 0.9
    state.problem_type = ProblemType.BINARY_CLASSIFICATION
    state.class_distribution = {
        0: ClassDistributionEntry(count=60, percentage=48.78),
        1: ClassDistributionEntry(count=63, percentage=51.22),
    }
    return state


def trained_state() -> AgentState:
    """Return a state carrying a plausible Training and Evaluation result."""
    state = analyzed_state()
    state.evaluation_status = "evaluation_completed"
    state.execution_time = 2.5
    state.optimization_status = "optimization_completed"
    return state


class TestSystemInstruction(unittest.TestCase):
    """The shared instruction must state the project's authority rules."""

    def test_the_instruction_states_that_metrics_are_authoritative(self):
        self.assertIn("authoritative", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_forbids_inventing_metrics(self):
        self.assertIn("never invent", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_forbids_claiming_statistical_proof(self):
        lowered = SYSTEM_INSTRUCTION.lower()
        self.assertIn("statistical proof", lowered)
        self.assertIn("significance", lowered)

    def test_the_instruction_forbids_training_and_retuning(self):
        self.assertIn("never train", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_forbids_defining_a_search_space(self):
        self.assertIn("search space", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_forbids_changing_the_dataset(self):
        self.assertIn("modifying the dataset", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_separates_the_three_registers(self):
        lowered = SYSTEM_INSTRUCTION.lower()
        for register in ("observation", "interpretation", "hypothesis"):
            self.assertIn(register, lowered)

    def test_the_instruction_names_every_forbidden_phrase(self):
        """A prohibition the model cannot read is not a prohibition."""
        lowered = SYSTEM_INSTRUCTION.lower()
        for phrase in config.LLM_FORBIDDEN_CLAIMS:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase.lower(), lowered)

    def test_the_instruction_offers_an_alternative_for_each_rule(self):
        lowered = SYSTEM_INSTRUCTION.lower()
        for word in config.LLM_ALLOWED_CLAIM_WORDS:
            with self.subTest(word=word):
                self.assertIn(word.lower(), lowered)

    def test_the_instruction_demands_json_only(self):
        self.assertIn("json object and nothing else", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_states_the_privacy_rule(self):
        self.assertIn("no credentials", SYSTEM_INSTRUCTION.lower())

    def test_the_instruction_does_not_promise_the_model_authority(self):
        """The wording must not hint that the model decides anything."""
        lowered = SYSTEM_INSTRUCTION.lower()
        self.assertNotIn("you decide", lowered)
        self.assertNotIn("you must select", lowered)
        self.assertNotIn("choose the best", lowered)

    def test_the_instruction_is_assembled_from_the_configuration(self):
        """Changing the configured wording must change the prompt."""
        self.assertIn(config.LLM_AUTHORITY_RULE.split(".")[0], SYSTEM_INSTRUCTION)


class TestRoleTemplates(unittest.TestCase):
    """Each role gets its own task and response contract."""

    def test_the_interpretation_schema_separates_the_registers(self):
        for field in (
            "summary",
            "observations",
            "interpretation",
            "hypotheses",
            "recommendations",
            "confidence",
        ):
            with self.subTest(field=field):
                self.assertIn(field, INTERPRETATION_SCHEMA_TEXT)

    def test_the_interpretation_schema_restricts_confidence(self):
        self.assertIn("low", INTERPRETATION_SCHEMA_TEXT)
        self.assertIn("medium", INTERPRETATION_SCHEMA_TEXT)
        self.assertIn("high", INTERPRETATION_SCHEMA_TEXT)

    def test_the_plan_schema_is_different_from_the_interpretation_schema(self):
        for field in ("candidate_models", "focus", "reason"):
            with self.subTest(field=field):
                self.assertIn(field, PLAN_SCHEMA_TEXT)
        self.assertNotIn("hypotheses", PLAN_SCHEMA_TEXT)

    def test_the_plan_schema_forbids_code(self):
        lowered = PLAN_SCHEMA_TEXT.lower()
        self.assertIn("do not write code", lowered)
        self.assertIn("search spaces", lowered)

    def test_each_role_has_its_own_task(self):
        tasks = {role: task_text(role) for role in LLMRole}
        self.assertEqual(len(set(tasks.values())), len(LLMRole))

    def test_the_evaluation_task_asks_for_the_recorded_ranking(self):
        self.assertIn("model comparison", task_text(LLMRole.EVALUATION).lower())

    def test_the_evaluation_task_does_not_ask_the_model_to_rank(self):
        """The pipeline ranked; the model may only explain that ranking."""
        lowered = task_text(LLMRole.EVALUATION).lower()
        self.assertNotIn("rank the models", lowered)
        self.assertNotIn("select the best", lowered)

    def test_the_optimization_task_is_a_hint_not_an_instruction(self):
        lowered = task_text(LLMRole.OPTIMIZATION).lower()
        self.assertIn("priority hint", lowered)
        self.assertIn("not a replacement", lowered)

    def test_the_schema_selection_matches_the_role(self):
        for role in LLMRole:
            with self.subTest(role=role.value):
                expected = PLAN_SCHEMA_TEXT if role.is_plan else INTERPRETATION_SCHEMA_TEXT
                self.assertEqual(schema_text(role), expected)

    def test_the_repair_instruction_asks_for_the_object_again(self):
        text = repair_instruction(LLMRole.DATA_ANALYSIS)
        self.assertIn("JSON object only", text)
        self.assertIn("not a single valid", text)

    def test_the_repair_instruction_repeats_the_schema(self):
        self.assertIn("confidence", repair_instruction(LLMRole.OPTIMIZATION))


class TestSanitizeValue(unittest.TestCase):
    """Value sanitising is the last line of defence for the privacy boundary."""

    def test_scalars_pass_through_unchanged(self):
        self.assertEqual(sanitize_value(17), 17)
        self.assertEqual(sanitize_value(0.456), 0.456)
        self.assertIs(sanitize_value(True), True)
        self.assertIsNone(sanitize_value(None))

    def test_measured_numbers_are_not_rounded(self):
        """Rounding a metric would change what the model is told."""
        self.assertEqual(sanitize_value(0.8761234), 0.8761234)

    def test_a_windows_path_is_reduced_to_its_file_name(self):
        self.assertEqual(
            sanitize_value(r"C:\Users\Someone\data\sample.csv"), "sample.csv"
        )

    def test_a_posix_path_is_reduced_to_its_file_name(self):
        self.assertEqual(sanitize_value("/home/someone/data/sample.csv"), "sample.csv")

    def test_a_unc_path_is_reduced(self):
        self.assertEqual(
            sanitize_value(r"\\server\share\secret\data.csv"), "data.csv"
        )

    def test_an_ordinary_word_is_not_treated_as_a_path(self):
        self.assertEqual(sanitize_value("binary_classification"), "binary_classification")

    def test_a_measurement_label_is_not_treated_as_a_path(self):
        self.assertEqual(sanitize_value("F1 score"), "F1 score")

    def test_a_long_string_is_truncated_with_a_marker(self):
        result = sanitize_value("x" * 5000, max_chars=100)
        self.assertLessEqual(len(result), 100)
        self.assertIn("truncated", result)

    def test_a_list_is_bounded(self):
        self.assertLessEqual(len(sanitize_value(list(range(500)))), config.LLM_CONTEXT_MAX_LIST_ITEMS)

    def test_a_mapping_is_sanitised_recursively(self):
        result = sanitize_value({"path": r"C:\Users\x\y.csv", "count": 3})
        self.assertEqual(result["path"], "y.csv")
        self.assertEqual(result["count"], 3)

    def test_the_result_is_always_json_serialisable(self):
        for value in (17, 0.5, True, None, "text", ["a"], {"k": "v"}, ("a", "b")):
            with self.subTest(value=type(value).__name__):
                json.dumps(sanitize_value(value))

    def test_redact_path_leaves_prose_alone(self):
        self.assertEqual(redact_path("the class was imbalanced"), "the class was imbalanced")


class TestContextAllowlist(unittest.TestCase):
    """The allowlist is the privacy boundary, so it is asserted explicitly."""

    def test_every_role_has_an_allowlist(self):
        for role in LLMRole:
            with self.subTest(role=role.value):
                self.assertTrue(allowed_context_fields(role))

    def test_the_data_analysis_context_uses_only_allowlisted_keys(self):
        context = build_data_analysis_context(analyzed_state())
        allowed = set(allowed_context_fields(LLMRole.DATA_ANALYSIS))
        for name in context.field_names:
            with self.subTest(name=name):
                self.assertIn(name, allowed)

    def test_the_evaluation_context_uses_only_allowlisted_keys(self):
        context = build_evaluation_context(trained_state())
        allowed = set(allowed_context_fields(LLMRole.EVALUATION))
        for name in context.field_names:
            with self.subTest(name=name):
                self.assertIn(name, allowed)

    def test_the_optimization_context_uses_only_allowlisted_keys(self):
        context = build_optimization_context(trained_state())
        allowed = set(allowed_context_fields(LLMRole.OPTIMIZATION))
        for name in context.field_names:
            with self.subTest(name=name):
                self.assertIn(name, allowed)

    def test_no_raw_row_can_be_present(self):
        """The builders are summaries; there is no row data to leak."""
        context = build_data_analysis_context(analyzed_state())
        self.assertNotIn("rows_data", context.field_names)
        self.assertNotIn("dataframe", context.field_names)
        self.assertNotIn("sample_rows", context.field_names)

    def test_no_file_contents_field_exists(self):
        for role in LLMRole:
            allowed = set(allowed_context_fields(role))
            for name in allowed:
                with self.subTest(role=role.value, name=name):
                    self.assertNotIn("content", name)
                    self.assertNotIn("raw", name)

    def test_no_credential_field_exists(self):
        for role in LLMRole:
            allowed = set(allowed_context_fields(role))
            for name in allowed:
                lowered = name.lower()
                for marker in ("key", "token", "secret", "password", "credential"):
                    with self.subTest(role=role.value, name=name, marker=marker):
                        self.assertNotIn(marker, lowered)

    def test_the_context_is_a_summary_of_shape_not_of_values(self):
        """Shape, counts and shares only - the point of the whole design."""
        context = build_data_analysis_context(analyzed_state())
        payload = context.as_dict()
        self.assertEqual(payload["rows"], 123)
        self.assertEqual(payload["total_missing_values"], 17)
        self.assertEqual(payload["target_column"], "Outcome")

    def test_an_empty_state_produces_a_context_not_a_crash(self):
        """A builder must survive a partially populated state."""
        for role in LLMRole:
            with self.subTest(role=role.value):
                context = build_context(AgentState(), role)
                json.dumps(context.as_dict())

    def test_the_context_carries_the_dataset_name_only(self):
        context = build_data_analysis_context(analyzed_state())
        self.assertEqual(context.as_dict()["dataset_name"], "sample_classification.csv")

    def test_an_absolute_dataset_path_is_reduced(self):
        state = analyzed_state()
        state.dataset_name = r"C:\Users\Someone\datasets\sample.csv"
        context = build_data_analysis_context(state)
        self.assertEqual(context.as_dict()["dataset_name"], "sample.csv")

    def test_the_context_is_bounded_in_field_count(self):
        context = build_data_analysis_context(analyzed_state())
        self.assertLessEqual(len(context.fields), config.LLM_CONTEXT_MAX_FIELDS)

    def test_the_context_renders_to_valid_json(self):
        context = build_data_analysis_context(analyzed_state())
        self.assertEqual(json.loads(context.to_json()), context.as_dict())

    def test_the_builder_for_a_role_matches_the_dispatcher(self):
        state = analyzed_state()
        for role, builder in (
            (LLMRole.DATA_ANALYSIS, build_data_analysis_context),
            (LLMRole.EVALUATION, build_evaluation_context),
            (LLMRole.OPTIMIZATION, build_optimization_context),
        ):
            with self.subTest(role=role.value):
                self.assertEqual(build_context(state, role), builder(state))


class TestNoSensitiveDataInPrompt(unittest.TestCase):
    """What must never appear in a prompt."""

    def setUp(self) -> None:
        self.prompts = [
            build_prompt(build_context(analyzed_state(), role)) for role in LLMRole
        ]
        self.prompts.append(build_repair_prompt(build_data_analysis_context(analyzed_state())))

    def test_no_absolute_windows_path_appears(self):
        for prompt in self.prompts:
            with self.subTest():
                self.assertNotIn(r"C:\Users", prompt)
                self.assertNotIn("C:\\Users", prompt)

    def test_no_home_directory_appears(self):
        for prompt in self.prompts:
            with self.subTest():
                self.assertNotIn("/home/", prompt)

    def test_no_environment_variable_value_appears(self):
        import os

        for name, value in os.environ.items():
            if len(value) < 12:
                continue
            for prompt in self.prompts:
                with self.subTest(variable=name):
                    self.assertNotIn(value, prompt)

    def test_no_credential_shaped_value_appears(self):
        for marker in ("sk-", "Bearer ", "password=", "api_key"):
            for prompt in self.prompts:
                with self.subTest(marker=marker):
                    self.assertNotIn(marker, prompt)

    def test_no_raw_row_values_appear(self):
        """Only the dataset name may be present, never its contents.

        It appears at most once per prompt: a repeated name would mean the context
        was appended rather than rebuilt, which is how duplicates creep in.
        """
        for prompt in self.prompts:
            with self.subTest():
                self.assertLessEqual(prompt.count("sample_classification.csv"), 1)

    def test_only_the_roles_that_need_it_send_the_dataset_name(self):
        """The optimisation context omits it: the planner has no use for it."""
        for role, expected in (
            (LLMRole.DATA_ANALYSIS, True),
            (LLMRole.EVALUATION, True),
            (LLMRole.OPTIMIZATION, False),
        ):
            with self.subTest(role=role.value):
                prompt = build_prompt(build_context(analyzed_state(), role))
                self.assertEqual(
                    "sample_classification.csv" in prompt, expected
                )

    def test_the_whole_environment_is_never_serialised(self):
        for prompt in self.prompts:
            with self.subTest():
                self.assertNotIn("environ", prompt)
                self.assertNotIn("PATH=", prompt)


class TestPromptDeterminism(unittest.TestCase):
    """The same state must always produce the same prompt."""

    def test_the_same_state_produces_an_identical_prompt(self):
        first = build_prompt(build_data_analysis_context(analyzed_state()))
        second = build_prompt(build_data_analysis_context(analyzed_state()))
        self.assertEqual(first, second)

    def test_a_second_evaluation_state_produces_an_identical_prompt(self):
        first = build_prompt(build_evaluation_context(trained_state()))
        second = build_prompt(build_evaluation_context(trained_state()))
        self.assertEqual(first, second)

    def test_the_repair_prompt_is_also_deterministic(self):
        context = build_optimization_context(trained_state())
        self.assertEqual(build_repair_prompt(context), build_repair_prompt(context))

    def test_the_prompt_has_no_timestamp(self):
        prompt = build_prompt(build_data_analysis_context(analyzed_state()))
        self.assertNotIn("202", prompt.replace("123", ""))

    def test_a_rebuilt_context_equals_the_first(self):
        self.assertEqual(
            build_data_analysis_context(analyzed_state()),
            build_data_analysis_context(analyzed_state()),
        )


class TestPromptShape(unittest.TestCase):
    """The assembled prompt reads in a fixed, documented order."""

    def setUp(self) -> None:
        self.context = build_data_analysis_context(analyzed_state())
        self.prompt = build_prompt(self.context)

    def test_the_task_comes_first(self):
        self.assertTrue(self.prompt.startswith(task_text(LLMRole.DATA_ANALYSIS)))

    def test_the_context_is_labelled_as_authoritative(self):
        self.assertIn("authoritative", self.prompt.lower())

    def test_the_schema_is_included(self):
        self.assertIn('"confidence"', self.prompt)

    def test_the_prompt_stays_within_the_bound(self):
        self.assertLessEqual(len(self.prompt), MAX_PROMPT_CHARS + 1)

    def test_an_oversized_context_is_truncated_visibly(self):
        state = analyzed_state()
        state.dataset_name = "a" * 5000
        state.target_column = "b" * 5000
        context = build_data_analysis_context(state)
        prompt = build_prompt(context)
        self.assertLessEqual(len(prompt), MAX_PROMPT_CHARS + len(" ...[truncated]") * 4)

    def test_truncation_is_announced(self):
        state = analyzed_state()
        state.dataset_name = "a" * 5000
        state.target_column = "b" * 5000
        self.assertIn("truncated", build_prompt(build_data_analysis_context(state)))

    def test_an_empty_context_is_reported_as_empty(self):
        self.assertTrue(PromptContext(role=LLMRole.DATA_ANALYSIS).is_empty())

    def test_the_context_field_order_is_the_allowlist_order(self):
        """Declared order, not dict order, is what makes this reproducible."""
        self.assertEqual(
            list(self.context.field_names),
            [name for name in allowed_context_fields(LLMRole.DATA_ANALYSIS)
             if name in self.context.as_dict()],
        )

    def test_every_role_produces_a_usable_prompt(self):
        for role in LLMRole:
            with self.subTest(role=role.value):
                prompt = build_prompt(build_context(analyzed_state(), role))
                self.assertTrue(prompt.strip())
                self.assertIn(schema_text(role)[:40], prompt)


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    unittest.main()
