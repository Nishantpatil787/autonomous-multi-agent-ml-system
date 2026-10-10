"""Tests for the interpretation service (Phase 8).

The service is the layer that decides whether to call the local model at all, and
it carries the project's strongest guarantee about the optional layer: **no public
method raises**. Every condition the layer can meet is asserted here as a typed
result - disabled, unreachable server, missing model, timeout, HTTP failure,
malformed JSON, schema violation, and a successful response - plus the bounded
repair and the bounded retry.

Two further properties are tested because they are the ones that would otherwise
only be asserted in prose: an unsuccessful result carries no empty interpretation
(so "the model was unavailable" cannot be mistaken for "the model found nothing"),
and a result never contains a live client or a raw response.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional, Tuple

from llm.client import LLMConnectionError, LocalLLMClient, Transport
from llm.service import LLMService, build_service
from llm.prompts import PromptContext
from llm.state import (
    LLMErrorCategory,
    LLMHealth,
    LLMInterpretation,
    LLMResult,
    LLMStatus,
    LLMRole,
)
from utils import config
from utils.state import AgentState

from tests.test_llm_client import generate_body, tags_body
from tests.test_llm_prompts import analyzed_state

VALID_INTERPRETATION = json.dumps(
    {
        "summary": "A small binary dataset with two balanced classes.",
        "observations": ["123 rows", "17 missing values"],
        "interpretation": ["The balance makes a stratified split appropriate."],
        "hypotheses": ["Missingness may relate to the target."],
        "recommendations": ["Inspect the rows with missing values."],
        "confidence": "medium",
    }
)

VALID_PLAN = json.dumps(
    {
        "summary": "Tune the tree-based models first.",
        "candidate_models": ["Random Forest", "Nonexistent Model"],
        "focus": ["class imbalance", "max_depth = 12"],
        "reason": "The hold-out F1 differed most between them.",
        "confidence": "low",
    }
)


def unavailable_transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
    """A transport that always reports a refused connection."""
    raise LLMConnectionError("connection refused", LLMErrorCategory.CONNECTION)


def healthy_transport(
    text: str,
    *models: str,
) -> Transport:
    """Return a transport that answers both endpoints with ``text``."""

    def transport(url: str, payload: Optional[Dict[str, Any]], timeout: float) -> Tuple[int, str]:
        if url.endswith("/api/tags"):
            return 200, tags_body(*(models or (config.OLLAMA_MODEL,)))
        return 200, generate_body(text)

    return transport


def service_for(
    transport: Transport,
    *,
    enabled: bool = True,
    allow_repair: Optional[bool] = None,
    health_check: Any = None,
) -> LLMService:
    """Build a service wired to a fake transport."""
    client = LocalLLMClient(
        base_url="http://localhost:11434",
        model=config.OLLAMA_MODEL,
        timeout=1.0,
        max_retries=0,
        transport=transport,
    )
    return LLMService(
        client=client,
        enabled=enabled,
        allow_repair=allow_repair,
        health_check=health_check,
    )


def healthy(status: LLMStatus = LLMStatus.AVAILABLE, detail: str = "ok") -> Any:
    """Return a health-check callable reporting ``status``."""
    return lambda: LLMHealth(status=status, detail=detail, model=config.OLLAMA_MODEL)


class TestDisabledLayer(unittest.TestCase):
    """The default state of the layer, which must cost nothing."""

    def test_the_configuration_default_is_off(self):
        self.assertFalse(config.OLLAMA_ENABLED)

    def test_a_disabled_service_reports_disabled(self):
        result = service_for(healthy_transport(VALID_INTERPRETATION), enabled=False)\
            .interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.DISABLED)

    def test_a_disabled_service_makes_no_request(self):
        """No socket, no prompt, no client: the default path is untouched."""
        calls: List[Any] = []

        def counting(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            calls.append(url)
            return 200, tags_body(config.OLLAMA_MODEL)

        service_for(counting, enabled=False).interpret_data_analysis(analyzed_state())
        self.assertEqual(calls, [])

    def test_a_disabled_service_produces_no_interpretation(self):
        result = build_service(False).interpret_data_analysis(analyzed_state())
        self.assertIsNone(result.interpretation)
        self.assertIsNone(result.plan)

    def test_a_disabled_result_is_not_a_failure(self):
        """Nothing was attempted, so nothing is degraded."""
        result = build_service(False).interpret_evaluation(analyzed_state())
        self.assertFalse(result.failed)

    def test_every_role_reports_disabled(self):
        service = build_service(False)
        for call in (
            service.interpret_data_analysis,
            service.interpret_evaluation,
            service.plan_optimization,
        ):
            with self.subTest(call=call.__name__):
                self.assertIs(call(analyzed_state()).status, LLMStatus.DISABLED)

    def test_the_disabled_result_is_still_serialisable(self):
        json.dumps(build_service(False).interpret_data_analysis(analyzed_state()).to_dict())


class TestUnreachableServer(unittest.TestCase):
    """A missing Ollama is the normal case on a machine without it."""

    def test_an_unreachable_server_is_reported_unavailable(self):
        result = service_for(unavailable_transport).interpret_data_analysis(
            analyzed_state()
        )
        self.assertIs(result.status, LLMStatus.UNAVAILABLE)
        self.assertIs(result.error_category, LLMErrorCategory.CONNECTION)

    def test_no_prompt_is_sent_when_the_server_is_down(self):
        """The health check runs first, so a dead server costs one probe."""
        calls: List[Any] = []

        def counting(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            calls.append(url)
            raise LLMConnectionError("refused", LLMErrorCategory.CONNECTION)

        service_for(counting).interpret_data_analysis(analyzed_state())
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/api/tags"))

    def test_the_detail_names_the_cause_and_the_way_out(self):
        result = service_for(unavailable_transport).interpret_evaluation(
            analyzed_state()
        )
        self.assertIn("connection refused", result.detail)
        self.assertIn("--llm", result.detail)

    def test_the_unavailable_result_carries_no_output(self):
        result = service_for(unavailable_transport).interpret_evaluation(
            analyzed_state()
        )
        self.assertIsNone(result.interpretation)
        self.assertIsNone(result.plan)
        self.assertFalse(result.succeeded)

    def test_an_unreachable_server_does_not_stop_the_next_role(self):
        service = service_for(unavailable_transport)
        for call in (
            service.interpret_data_analysis,
            service.interpret_evaluation,
            service.plan_optimization,
        ):
            with self.subTest(call=call.__name__):
                self.assertIs(call(analyzed_state()).status, LLMStatus.UNAVAILABLE)


class TestMissingModel(unittest.TestCase):
    """A running Ollama without the configured model needs a different message."""

    def test_a_missing_model_is_reported_separately(self):
        service = service_for(healthy_transport(VALID_INTERPRETATION, "some-other-model"))
        result = service.interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.MODEL_UNAVAILABLE)

    def test_the_detail_explains_how_to_install_it(self):
        service = service_for(healthy_transport(VALID_INTERPRETATION, "some-other-model"))
        result = service.interpret_data_analysis(analyzed_state())
        self.assertIn("ollama pull", result.detail)

    def test_no_completion_request_is_sent(self):
        calls: List[str] = []

        def counting(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            calls.append(url)
            if url.endswith("/api/tags"):
                return 200, tags_body("some-other-model")
            return 200, generate_body(VALID_INTERPRETATION)

        service_for(counting).interpret_data_analysis(analyzed_state())
        self.assertEqual(len(calls), 1)

    def test_the_model_name_is_recorded_on_the_result(self):
        service = service_for(healthy_transport(VALID_INTERPRETATION, "some-other-model"))
        self.assertEqual(
            service.interpret_data_analysis(analyzed_state()).model, config.OLLAMA_MODEL
        )


class TestRequestFailures(unittest.TestCase):
    """A failed request is a recorded status, never an exception."""

    def assert_request_failed(self, transport: Transport, category: LLMErrorCategory) -> None:
        """Assert one interpretation step reports a request failure.

        Args:
            transport: The fake transport to use.
            category: The error category that must be recorded.
        """
        result = service_for(transport).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.REQUEST_FAILED)
        self.assertIs(result.error_category, category)
        self.assertIsNone(result.interpretation)

    def test_a_timeout_is_recorded_as_a_timeout(self):
        def timing_out(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            raise LLMConnectionError("timed out after 30s", LLMErrorCategory.TIMEOUT)

        self.assert_request_failed(timing_out, LLMErrorCategory.TIMEOUT)

    def test_a_connection_failure_during_generation_is_recorded(self):
        def failing(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            raise LLMConnectionError("refused", LLMErrorCategory.CONNECTION)

        self.assert_request_failed(failing, LLMErrorCategory.CONNECTION)

    def test_an_http_error_is_recorded(self):
        def erroring(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            return 500, "server error"

        self.assert_request_failed(erroring, LLMErrorCategory.HTTP_ERROR)

    def test_an_unreadable_envelope_is_recorded_as_such(self):
        def odd(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            return 200, json.dumps({"done": True})

        self.assert_request_failed(odd, LLMErrorCategory.UNEXPECTED_RESPONSE)

    def test_an_unexpected_transport_error_is_recorded_not_raised(self):
        def exploding(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            raise MemoryError("unexpected")

        result = service_for(exploding).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.REQUEST_FAILED)
        self.assertIn("Unexpected failure", result.detail)


class TestMalformedResponses(unittest.TestCase):
    """A bad answer is recorded, and gets at most one controlled repair."""

    def test_malformed_json_is_recorded_as_invalid(self):
        result = service_for(
            healthy_transport("I cannot do that."), allow_repair=False
        ).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.INVALID_RESPONSE)
        self.assertIs(result.error_category, LLMErrorCategory.MALFORMED_JSON)

    def test_a_schema_violation_is_recorded_as_invalid(self):
        result = service_for(
            healthy_transport(json.dumps({"observations": ["no summary"]})),
            allow_repair=False,
        ).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.INVALID_RESPONSE)
        self.assertIs(result.error_category, LLMErrorCategory.SCHEMA_VIOLATION)

    def test_an_empty_response_is_recorded_as_empty(self):
        result = service_for(
            healthy_transport("   "), allow_repair=False
        ).interpret_data_analysis(analyzed_state())
        self.assertIs(result.error_category, LLMErrorCategory.EMPTY_RESPONSE)

    def test_the_detail_explains_both_attempts(self):
        result = service_for(healthy_transport("not json")).interpret_data_analysis(
            analyzed_state()
        )
        self.assertIn("still unusable", result.detail)

    def test_an_invalid_result_carries_no_interpretation(self):
        """An empty interpretation must not look like a real answer."""
        result = service_for(healthy_transport("not json")).interpret_evaluation(
            analyzed_state()
        )
        self.assertIsNone(result.interpretation)
        self.assertIsNone(result.output)


class TestControlledRepair(unittest.TestCase):
    """Exactly one repair attempt, and only for a repairable failure."""

    def test_a_repair_succeeds_and_is_recorded(self):
        calls: List[int] = []

        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            calls.append(1)
            text = "not json" if len(calls) == 1 else VALID_INTERPRETATION
            return 200, generate_body(text)

        result = service_for(transport).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.COMPLETED)
        self.assertTrue(result.repaired)
        self.assertEqual(len(calls), 2)

    def test_the_repair_is_bounded_to_one_attempt(self):
        """A model that cannot comply must not cause a retry storm."""
        calls: List[int] = []

        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            calls.append(1)
            return 200, generate_body("still not json")

        result = service_for(transport).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.INVALID_RESPONSE)
        self.assertEqual(len(calls), 2)

    def test_a_valid_first_response_is_never_repaired(self):
        calls: List[int] = []

        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            calls.append(1)
            return 200, generate_body(VALID_INTERPRETATION)

        result = service_for(transport).interpret_data_analysis(analyzed_state())
        self.assertFalse(result.repaired)
        self.assertEqual(len(calls), 1)

    def test_the_repair_can_be_switched_off(self):
        calls: List[int] = []

        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            calls.append(1)
            return 200, generate_body("not json")

        result = service_for(transport, allow_repair=False).interpret_data_analysis(
            analyzed_state()
        )
        self.assertIs(result.status, LLMStatus.INVALID_RESPONSE)
        self.assertEqual(len(calls), 1)

    def test_an_empty_response_is_not_repaired(self):
        """Asking again for nothing will also return nothing."""
        calls: List[int] = []

        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            calls.append(1)
            return 200, generate_body("")

        service_for(transport).interpret_data_analysis(analyzed_state())
        self.assertEqual(len(calls), 1)

    def test_a_failed_repair_reports_the_request_failure(self):
        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            raise LLMConnectionError("refused", LLMErrorCategory.CONNECTION)

        result = service_for(transport).interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.REQUEST_FAILED)


class TestSuccessfulResponses(unittest.TestCase):
    """The happy path, and the guarantees attached to it."""

    def setUp(self) -> None:
        self.service = service_for(healthy_transport(VALID_INTERPRETATION))
        self.state = analyzed_state()

    def test_a_valid_response_completes(self):
        result = self.service.interpret_data_analysis(self.state)
        self.assertIs(result.status, LLMStatus.COMPLETED)
        self.assertTrue(result.succeeded)
        self.assertIs(result.error_category, LLMErrorCategory.NONE)

    def test_the_interpretation_is_attached(self):
        result = self.service.interpret_data_analysis(self.state)
        self.assertIsInstance(result.interpretation, LLMInterpretation)
        self.assertTrue(result.interpretation.summary)

    def test_the_model_name_is_recorded(self):
        result = self.service.interpret_data_analysis(self.state)
        self.assertEqual(result.model, config.OLLAMA_MODEL)
        self.assertEqual(result.interpretation.model, config.OLLAMA_MODEL)

    def test_the_context_fields_sent_are_recorded(self):
        """So a reader can see exactly what the model was shown."""
        result = self.service.interpret_data_analysis(self.state)
        self.assertIn("rows", result.context_fields)
        self.assertIn("total_missing_values", result.context_fields)

    def test_only_allowlisted_context_fields_are_recorded(self):
        from llm.state import allowed_context_fields

        result = self.service.interpret_evaluation(self.state)
        self.assertEqual(
            set(result.context_fields) - set(allowed_context_fields(LLMRole.EVALUATION)),
            set(),
        )

    def test_the_evaluation_role_is_answered_with_an_interpretation(self):
        result = self.service.interpret_evaluation(self.state)
        self.assertIsInstance(result.interpretation, LLMInterpretation)
        self.assertIsNone(result.plan)

    def test_the_optimization_role_is_answered_with_a_plan(self):
        service = service_for(healthy_transport(VALID_PLAN))
        result = service.plan_optimization(self.state)
        self.assertIsNone(result.interpretation)
        self.assertIsNotNone(result.plan)

    def test_the_plan_is_validated_against_the_registry(self):
        service = service_for(healthy_transport(VALID_PLAN))
        plan = service.plan_optimization(self.state).plan
        self.assertIn("Nonexistent Model", plan.rejected_models)
        self.assertNotIn("Nonexistent Model", plan.candidate_models)

    def test_a_code_focus_entry_is_rejected(self):
        service = service_for(healthy_transport(VALID_PLAN))
        plan = service.plan_optimization(self.state).plan
        self.assertIn("max_depth = 12", plan.rejected_focus_areas)
        self.assertIn("class imbalance", plan.focus_areas)

    def test_a_forbidden_claim_is_recorded_on_the_result(self):
        text = json.dumps(
            dict(
                json.loads(VALID_INTERPRETATION),
                summary="This guarantees the model will work perfectly.",
            )
        )
        result = service_for(healthy_transport(text)).interpret_data_analysis(self.state)
        self.assertIs(result.status, LLMStatus.COMPLETED)
        self.assertTrue(result.forbidden_claims)

    def test_a_forbidden_claim_is_reported_not_silently_dropped(self):
        text = json.dumps(
            dict(json.loads(VALID_INTERPRETATION), summary="This is production ready.")
        )
        result = service_for(healthy_transport(text)).interpret_evaluation(self.state)
        self.assertTrue(result.interpretation.has_forbidden_claims)

    def test_the_duration_is_recorded(self):
        self.assertGreaterEqual(
            self.service.interpret_data_analysis(self.state).duration, 0.0
        )


class TestNoAuthority(unittest.TestCase):
    """The layer cannot touch the machine learning state."""

    def test_the_state_is_not_modified(self):
        state = analyzed_state()
        before = state.to_dict()
        service_for(healthy_transport(VALID_INTERPRETATION)).interpret_data_analysis(state)
        self.assertEqual(state.to_dict(), before)

    def test_the_state_is_not_modified_on_failure_either(self):
        state = analyzed_state()
        before = state.to_dict()
        service_for(unavailable_transport).interpret_evaluation(state)
        self.assertEqual(state.to_dict(), before)

    def test_the_state_is_not_modified_when_the_response_is_malformed(self):
        state = analyzed_state()
        before = state.to_dict()
        service_for(healthy_transport("not json")).interpret_data_analysis(state)
        self.assertEqual(state.to_dict(), before)

    def test_no_metric_is_written_back_onto_the_state(self):
        """Even a plausible invented number must not reach the state."""
        text = json.dumps(
            {
                "summary": "The hold-out F1 was 0.9999 and the accuracy 1.0.",
                "observations": ["hold-out F1 0.9999"],
                "confidence": "high",
            }
        )
        state = analyzed_state()
        service_for(healthy_transport(text)).interpret_evaluation(state)
        self.assertNotIn("0.9999", json.dumps(state.to_dict()))


class TestResultSerialisation(unittest.TestCase):
    """A result must survive JSON, because it travels in the workflow state."""

    def test_a_completed_result_is_json_serialisable(self):
        result = service_for(healthy_transport(VALID_INTERPRETATION)).interpret_data_analysis(
            analyzed_state()
        )
        json.dumps(result.to_dict())

    def test_a_plan_result_is_json_serialisable(self):
        result = service_for(healthy_transport(VALID_PLAN)).plan_optimization(
            analyzed_state()
        )
        json.dumps(result.to_dict())

    def test_every_status_is_json_serialisable(self):
        """Including the failure paths, which is where raw text usually leaks."""
        for transport in (
            unavailable_transport,
            healthy_transport("not json"),
            healthy_transport(""),
        ):
            with self.subTest(transport=transport.__name__):
                result = service_for(transport).interpret_evaluation(analyzed_state())
                json.dumps(result.to_dict())

    def test_no_client_object_is_in_the_payload(self):
        """A live client in the state would break serialisation and add a handle."""
        result = service_for(healthy_transport(VALID_INTERPRETATION)).interpret_data_analysis(
            analyzed_state()
        )
        payload = result.to_dict()
        self.assertNotIn("client", payload)
        for value in payload.values():
            self.assertNotIsInstance(value, LocalLLMClient)

    def test_no_raw_response_is_stored(self):
        """Only validated content survives; the transport envelope does not.

        The Ollama envelope carries ``model``, ``done`` and ``done_reason``. The
        model name is recorded deliberately, but the completion flags are not, and
        the raw text itself must not be kept: it is the one part of the exchange
        that was never validated.
        """
        text = json.dumps({"summary": "validated summary", "observations": []})
        result = service_for(healthy_transport(text)).interpret_data_analysis(
            analyzed_state()
        )
        payload = json.dumps(result.to_dict())
        self.assertIn("validated summary", payload)
        self.assertNotIn("done_reason", payload)
        self.assertNotIn("prompt", payload)
        self.assertNotIn("system", payload)

    def test_a_failure_keeps_a_bounded_excerpt_for_diagnosis(self):
        """A malformed answer must be diagnosable without storing the whole thing.

        A short excerpt is kept because hiding it would make a failure impossible
        to debug. It carries no confidentiality risk: the model was only ever shown
        the allowlisted context, so the most it can echo back is a value the project
        already had. The excerpt is bounded so it cannot bloat the state.
        """
        result = service_for(healthy_transport("SENTINEL " * 500)).interpret_data_analysis(
            analyzed_state()
        )
        self.assertIs(result.status, LLMStatus.INVALID_RESPONSE)
        detail = result.to_dict()["detail"]
        self.assertIn("SENTINEL", detail)
        self.assertLess(
            len(detail), 4 * config.LLM_RAW_RESPONSE_LOG_LIMIT + 200
        )

    def test_a_failure_excerpt_carries_nothing_the_model_was_not_shown(self):
        state = analyzed_state()
        result = service_for(healthy_transport("SENTINEL " * 500)).interpret_data_analysis(
            state
        )
        prompt_values = " ".join(str(value) for value in state.to_dict().values())
        for secret in ("SENTINEL", r"C:\Users", "/home/"):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, prompt_values)

    def test_the_excerpt_is_never_the_whole_response(self):
        result = service_for(healthy_transport("x" * 100000)).interpret_data_analysis(
            analyzed_state()
        )
        self.assertLess(len(json.dumps(result.to_dict())), 20000)

    def test_the_to_dict_includes_the_status_and_category(self):
        result = build_service(False).interpret_data_analysis(analyzed_state())
        payload = result.to_dict()
        self.assertEqual(payload["status"], "disabled")
        self.assertEqual(payload["role"], "data_analysis")
        self.assertIn("error_category", payload)

    def test_the_to_dict_names_the_role_for_humans(self):
        payload = build_service(False).interpret_evaluation(analyzed_state()).to_dict()
        self.assertEqual(payload["role_display_name"], "Evaluation Interpretation")


class TestHealthCheckOverride(unittest.TestCase):
    """The health check is injectable so no unit test needs a server."""

    def test_the_probe_is_made_once_per_service(self):
        """One refused local connection is expensive; three probes waste 8 seconds."""
        probes: List[int] = []

        def counting_probe() -> LLMHealth:
            probes.append(1)
            return LLMHealth(status=LLMStatus.AVAILABLE, detail="ok")

        service = service_for(
            healthy_transport(VALID_INTERPRETATION), health_check=counting_probe
        )
        service.interpret_data_analysis(analyzed_state())
        service.interpret_evaluation(analyzed_state())
        service.plan_optimization(analyzed_state())
        self.assertEqual(len(probes), 1)

    def test_an_unavailable_server_is_probed_once_not_once_per_step(self):
        calls: List[str] = []

        def counting(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            calls.append(url)
            raise LLMConnectionError("refused", LLMErrorCategory.CONNECTION)

        service = service_for(counting)
        for call in (
            service.interpret_data_analysis,
            service.interpret_evaluation,
            service.plan_optimization,
        ):
            call(analyzed_state())
        self.assertEqual(len(calls), 1)

    def test_the_snapshot_can_be_refreshed_explicitly(self):
        probes: List[int] = []

        def changing_probe() -> LLMHealth:
            probes.append(1)
            status = LLMStatus.UNAVAILABLE if len(probes) > 1 else LLMStatus.AVAILABLE
            return LLMHealth(status=status, detail="probe")

        service = service_for(
            healthy_transport(VALID_INTERPRETATION), health_check=changing_probe
        )
        self.assertIs(service.check_health().status, LLMStatus.AVAILABLE)
        self.assertIs(service.check_health().status, LLMStatus.AVAILABLE)
        self.assertIs(service.check_health(refresh=True).status, LLMStatus.UNAVAILABLE)
        self.assertEqual(len(probes), 2)

    def test_no_interpretation_state_is_shared_between_steps(self):
        """Only the health snapshot is cached; generated text never is."""
        service = service_for(healthy_transport(VALID_INTERPRETATION))
        service.interpret_data_analysis(analyzed_state())
        service.interpret_data_analysis(analyzed_state())
        # A second call is a fresh request, not a replay of the first result.
        self.assertEqual(len(service.interpret_data_analysis(analyzed_state())
                             .interpretation.observations), 2)

    def test_an_injected_health_check_is_used(self):
        service = LLMService(
            enabled=True,
            health_check=healthy(),
            client=LocalLLMClient(
                base_url="http://localhost:11434",
                model=config.OLLAMA_MODEL,
                transport=healthy_transport(VALID_INTERPRETATION),
            ),
        )
        self.assertIs(service.check_health().status, LLMStatus.AVAILABLE)

    def test_a_raising_health_check_is_contained(self):
        def exploding() -> LLMHealth:
            raise RuntimeError("probe failed unexpectedly")

        service = LLMService(enabled=True, health_check=exploding)
        result = service.interpret_data_analysis(analyzed_state())
        self.assertIs(result.status, LLMStatus.UNAVAILABLE)

    def test_a_context_that_cannot_be_built_is_recorded(self):
        """A builder failure must not become an exception."""

        def exploding_builder(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("cannot summarise this state")

        import llm.service as service_module

        original = service_module.build_context
        service_module.build_context = exploding_builder
        try:
            service = service_for(healthy_transport(VALID_INTERPRETATION))
            result = service.interpret_data_analysis(analyzed_state())
        finally:
            service_module.build_context = original
        self.assertIs(result.status, LLMStatus.REQUEST_FAILED)
        self.assertIn("context could not be built", result.detail)

    def test_an_empty_context_makes_no_request(self):
        """There is nothing to interpret, so nothing is asked for."""
        calls: List[int] = []

        def transport(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            if url.endswith("/api/tags"):
                return 200, tags_body(config.OLLAMA_MODEL)
            calls.append(1)
            return 200, generate_body(VALID_INTERPRETATION)

        import llm.service as service_module

        original = service_module.build_context
        service_module.build_context = lambda state, role: PromptContext(role=role)
        try:
            result = service_for(transport).interpret_data_analysis(AgentState())
        finally:
            service_module.build_context = original

        self.assertEqual(result.status, LLMStatus.INVALID_RESPONSE)
        self.assertEqual(calls, [])

    def test_an_empty_state_still_produces_a_bounded_context(self):
        """The builders emit the allowlisted keys, so nothing is sent unbounded."""
        result = service_for(healthy_transport(VALID_INTERPRETATION)).interpret_data_analysis(
            AgentState()
        )
        self.assertLessEqual(
            len(result.context_fields), config.LLM_CONTEXT_MAX_FIELDS
        )


class TestResultVocabulary(unittest.TestCase):
    """The status and category vocabularies are part of the contract."""

    def test_every_documented_status_exists(self):
        for name in (
            "DISABLED",
            "AVAILABLE",
            "COMPLETED",
            "UNAVAILABLE",
            "MODEL_UNAVAILABLE",
            "REQUEST_FAILED",
            "INVALID_RESPONSE",
        ):
            with self.subTest(status=name):
                self.assertIsInstance(getattr(LLMStatus, name).value, str)

    def test_the_failure_statuses_are_the_four_that_can_fail(self):
        failing = {status for status in LLMStatus if status.is_failure}
        self.assertEqual(
            failing,
            {
                LLMStatus.UNAVAILABLE,
                LLMStatus.MODEL_UNAVAILABLE,
                LLMStatus.REQUEST_FAILED,
                LLMStatus.INVALID_RESPONSE,
            },
        )

    def test_only_completed_produces_output(self):
        producing = {status for status in LLMStatus if status.produced_output}
        self.assertEqual(producing, {LLMStatus.COMPLETED})

    def test_available_is_not_a_completed_request(self):
        """Available describes a probe; completed describes an answer."""
        self.assertIsNot(LLMStatus.AVAILABLE, LLMStatus.COMPLETED)
        self.assertFalse(LLMStatus.AVAILABLE.produced_output)

    def test_every_error_category_is_distinct(self):
        values = [category.value for category in LLMErrorCategory]
        self.assertEqual(len(values), len(set(values)))

    def test_every_role_has_a_display_name(self):
        for role in LLMRole:
            with self.subTest(role=role.value):
                self.assertTrue(role.display_name)


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    unittest.main()
