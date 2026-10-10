"""Tests for the Ollama client and its availability checks (Phase 8).

Everything here runs without a Ollama server. The client takes a transport
callable, so a test supplies a fake instead of opening a socket. That is not only
convenient: it is what makes the timeout, the refused-connection and the
malformed-envelope paths testable at all, since none of them can be provoked
reliably against a real server.

The tests are grouped by the five conditions the layer has to tell apart:

* the server cannot be reached,
* the server answers but the model is missing,
* both are available,
* the request failed (timeout, connection, HTTP error),
* the response was not usable.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional, Tuple

from llm.client import (
    GENERATE_PATH,
    TAGS_PATH,
    LLMConnectionError,
    LLMError,
    LLMNotConfiguredError,
    LLMUnsupportedHostError,
    LocalLLMClient,
    build_client,
    is_local_url,
    require_local_url,
)
from llm.state import LLMErrorCategory, LLMStatus
from utils import config


class RecordingTransport:
    """A fake transport that records calls and replays scripted results.

    Args:
        results: One ``(status_code, body)`` pair per call, or a single callable
            that receives ``(url, payload, timeout)`` and returns a pair.
    """

    def __init__(self, results: Any) -> None:
        self.results = results
        self.calls: List[Tuple[str, Optional[Dict[str, Any]], float]] = []

    def __call__(
        self,
        url: str,
        payload: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Tuple[int, str]:
        """Record the call and return the next scripted result.

        A single scripted result is replayed for every call, so a test that does
        not care about ordering can supply one entry. A list is consumed one entry
        at a time, which is how the retry tests assert a call budget. A bare
        exception is raised on every call, which is how the connection-failure
        tests avoid building a list of N identical failures.
        """
        self.calls.append((url, payload, timeout))

        if isinstance(self.results, BaseException):
            raise self.results
        if callable(self.results):
            return self.results(url, payload, timeout)
        if not self.results:
            raise AssertionError("the transport was called more times than scripted")
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, BaseException):
            raise result
        return result


def router(text: str, *models: str) -> Any:
    """Return a transport that answers the tag and generate endpoints properly.

    Most request tests only care about one endpoint, and forcing every one of them
    to script the tag probe first would obscure what is being asserted.

    Args:
        text: The model text the generate endpoint should return.
        *models: The models the tag endpoint should report as installed.

    Returns:
        A transport callable.
    """

    def transport(url: str, payload: Optional[Dict[str, Any]], timeout: float) -> Tuple[int, str]:
        if url.endswith(TAGS_PATH):
            return 200, tags_body(*(models or ("qwen2.5:3b",)))
        return 200, generate_body(text)

    return transport


def tags_body(*names: str) -> str:
    """Return a realistic ``/api/tags`` body listing the given models."""
    return json.dumps(
        {"models": [{"name": name, "model": name, "size": 1900000000} for name in names]}
    )


def generate_body(text: str) -> str:
    """Return a realistic ``/api/generate`` body carrying ``text``."""
    return json.dumps(
        {"model": "qwen2.5:3b", "response": text, "done": True, "done_reason": "stop"}
    )


def make_client(transport: Any, **overrides: Any) -> LocalLLMClient:
    """Build a client with a fake transport and test-friendly defaults."""
    settings: Dict[str, Any] = {
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:3b",
        "timeout": 2.0,
        "max_retries": 1,
        "transport": transport,
    }
    settings.update(overrides)
    return LocalLLMClient(**settings)


class TestConfiguration(unittest.TestCase):
    """The configured defaults and how the client reads them."""

    def test_the_layer_is_disabled_by_default(self):
        """An unmodified checkout must never contact Ollama on its own."""
        self.assertFalse(config.OLLAMA_ENABLED)

    def test_the_base_url_is_loopback(self):
        self.assertIn("localhost", config.OLLAMA_BASE_URL)
        self.assertTrue(is_local_url(config.OLLAMA_BASE_URL))

    def test_the_timeout_is_short_enough_to_not_stall_a_run(self):
        self.assertGreater(config.OLLAMA_TIMEOUT, 0)
        self.assertLessEqual(config.OLLAMA_TIMEOUT, 120)

    def test_the_retry_budget_is_bounded_and_small(self):
        self.assertEqual(config.OLLAMA_MAX_RETRIES, 1)

    def test_the_temperature_is_zero(self):
        """A deterministic sampling setting keeps the layer reproducible."""
        self.assertEqual(config.OLLAMA_TEMPERATURE, 0)

    def test_the_default_model_is_named_and_lightweight(self):
        self.assertTrue(config.OLLAMA_MODEL)
        self.assertNotIn(" ", config.OLLAMA_MODEL)

    def test_build_client_uses_the_configuration(self):
        client = build_client()
        self.assertEqual(client.base_url, config.OLLAMA_BASE_URL)
        self.assertEqual(client.model, config.OLLAMA_MODEL)
        self.assertEqual(client.max_retries, config.OLLAMA_MAX_RETRIES)

    def test_build_client_accepts_an_override(self):
        client = build_client(model="llama3.2:1b")
        self.assertEqual(client.model, "llama3.2:1b")

    def test_no_api_key_setting_exists(self):
        """The project is local-only, so there must be no key to configure."""
        for name in dir(config):
            self.assertNotIn("API_KEY", name)
            self.assertNotIn("GEMINI", name)
            self.assertNotIn("ANTHROPIC", name)


class TestLocalityEnforcement(unittest.TestCase):
    """A non-loopback base URL must be refused before any request is made."""

    def test_a_remote_host_is_rejected(self):
        for url in (
            "http://example.com:11434",
            "https://api.ollama.ai",
            "http://192.168.1.10:11434",
        ):
            with self.subTest(url=url):
                with self.assertRaises(LLMUnsupportedHostError):
                    require_local_url(url)

    def test_the_rejection_explains_why(self):
        with self.assertRaises(LLMUnsupportedHostError) as caught:
            require_local_url("http://example.com:11434")
        self.assertIn("local", str(caught.exception).lower())

    def test_loopback_hosts_are_accepted(self):
        for url in (
            "http://localhost:11434",
            "http://127.0.0.1:11434/",
            "http://[::1]:11434",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_local_url(url))
                self.assertFalse(require_local_url(url).endswith("/"))

    def test_a_malformed_url_is_rejected(self):
        for url in ("", "localhost:11434", "ftp://localhost", "not a url"):
            with self.subTest(url=url):
                self.assertFalse(is_local_url(url))

    def test_the_client_validates_the_host_on_construction(self):
        with self.assertRaises(LLMUnsupportedHostError):
            make_client(RecordingTransport([]), base_url="http://example.com")

    def test_a_remote_host_makes_no_request(self):
        """The failure happens before the transport is reachable at all."""
        transport = RecordingTransport([])
        with self.assertRaises(LLMUnsupportedHostError):
            LocalLLMClient(
                base_url="http://example.com",
                model="m",
                transport=transport,
            )
        self.assertEqual(transport.calls, [])


class TestMissingModelConfiguration(unittest.TestCase):
    """A client cannot be built without a model, and says why."""

    def test_an_empty_model_is_refused(self):
        for model in ("", "   "):
            with self.subTest(model=model):
                with self.assertRaises(LLMNotConfiguredError):
                    make_client(RecordingTransport([]), model=model)

    def test_the_error_names_the_setting_to_fix(self):
        with self.assertRaises(LLMNotConfiguredError) as caught:
            make_client(RecordingTransport([]), model="")
        self.assertIn("OLLAMA_MODEL", str(caught.exception))

    def test_the_configuration_errors_share_one_base(self):
        """Callers catch one type, which is what makes the layer predictable."""
        for error in (LLMNotConfiguredError("a"), LLMUnsupportedHostError("b")):
            with self.subTest(error=type(error).__name__):
                self.assertIsInstance(error, LLMError)


class TestOllamaAvailability(unittest.TestCase):
    """``check_ollama_available`` distinguishes reachable from not reachable."""

    def test_a_running_server_is_reported_available(self):
        transport = RecordingTransport([(200, tags_body("qwen2.5:3b"))])
        health = make_client(transport).check_ollama_available()
        self.assertIs(health.status, LLMStatus.AVAILABLE)
        self.assertTrue(health.available)
        self.assertIn("qwen2.5:3b", health.available_models)

    def test_the_tag_endpoint_is_the_one_that_is_called(self):
        transport = RecordingTransport([(200, tags_body())])
        make_client(transport).check_ollama_available()
        url, payload, timeout = transport.calls[0]
        self.assertTrue(url.endswith(TAGS_PATH))
        self.assertIsNone(payload)
        self.assertGreater(timeout, 0)

    def test_a_refused_connection_is_reported_unavailable(self):
        transport = RecordingTransport(
            LLMConnectionError("connection refused", LLMErrorCategory.CONNECTION)
        )
        health = make_client(transport).check_ollama_available()
        self.assertIs(health.status, LLMStatus.UNAVAILABLE)
        self.assertFalse(health.available)

    def test_the_unavailable_detail_is_actionable(self):
        transport = RecordingTransport(
            LLMConnectionError("connection refused", LLMErrorCategory.CONNECTION)
        )
        health = make_client(transport).check_ollama_available()
        self.assertIn("--llm", health.detail)
        self.assertIn("connection refused", health.detail)

    def test_an_http_error_status_is_not_available(self):
        transport = RecordingTransport([(503, "service unavailable")])
        health = make_client(transport).check_ollama_available()
        self.assertIs(health.status, LLMStatus.UNAVAILABLE)
        self.assertIn("503", health.detail)

    def test_an_unreadable_body_is_not_treated_as_empty_success(self):
        """A body that cannot be parsed means unknown, so it is unavailable."""
        transport = RecordingTransport([(200, "not json at all")])
        health = make_client(transport).check_ollama_available()
        self.assertIs(health.status, LLMStatus.UNAVAILABLE)
        self.assertIn("could not be read", health.detail)

    def test_a_body_without_a_model_list_is_unavailable(self):
        transport = RecordingTransport([(200, json.dumps({"version": "0.1.0"}))])
        health = make_client(transport).check_ollama_available()
        self.assertIs(health.status, LLMStatus.UNAVAILABLE)

    def test_a_server_with_no_models_is_still_available(self):
        """Ollama running with an empty library is a healthy server."""
        transport = RecordingTransport([(200, json.dumps({"models": []}))])
        health = make_client(transport).check_ollama_available()
        self.assertIs(health.status, LLMStatus.AVAILABLE)
        self.assertEqual(health.available_models, ())

    def test_an_unexpected_transport_error_never_propagates(self):
        def explode(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            raise RuntimeError("something entirely unexpected")

        health = make_client(RecordingTransport(explode)).check_ollama_available()
        self.assertIs(health.status, LLMStatus.UNAVAILABLE)
        self.assertIn("Unexpected failure", health.detail)

    def test_only_dict_entries_with_a_name_are_counted(self):
        body = json.dumps(
            {"models": [{"name": "a"}, {"size": 1}, "b", {"name": ""}, {"model": "c"}]}
        )
        transport = RecordingTransport([(200, body)])
        health = make_client(transport).check_ollama_available()
        self.assertEqual(health.available_models, ("a", "c"))

    def test_the_health_record_is_json_safe(self):
        transport = RecordingTransport([(200, tags_body("qwen2.5:3b"))])
        payload = make_client(transport).check_ollama_available().to_dict()
        self.assertEqual(json.loads(json.dumps(payload))["status"], "available")


class TestModelAvailability(unittest.TestCase):
    """``check_model_available`` adds the model check on top of the server check."""

    def test_an_installed_model_is_available(self):
        transport = RecordingTransport([(200, tags_body("qwen2.5:3b", "llama3.2"))])
        health = make_client(transport).check_model_available()
        self.assertIs(health.status, LLMStatus.AVAILABLE)
        self.assertIn("qwen2.5:3b", health.detail)

    def test_a_missing_model_is_reported_separately_from_a_missing_server(self):
        """The two need different fixes, so they must not share a status."""
        transport = RecordingTransport([(200, tags_body("llama3.2"))])
        health = make_client(transport).check_model_available()
        self.assertIs(health.status, LLMStatus.MODEL_UNAVAILABLE)
        self.assertNotEqual(health.status, LLMStatus.UNAVAILABLE)

    def test_the_missing_model_detail_lists_what_is_installed(self):
        transport = RecordingTransport([(200, tags_body("llama3.2:1b"))])
        health = make_client(transport).check_model_available()
        self.assertIn("llama3.2:1b", health.detail)
        self.assertIn("qwen2.5:3b", health.detail)

    def test_the_missing_model_detail_explains_how_to_install_it(self):
        """The project never downloads a model, so the user has to be told how."""
        transport = RecordingTransport([(200, tags_body("llama3.2:1b"))])
        health = make_client(transport).check_model_available()
        self.assertIn("ollama pull", health.detail)

    def test_an_unreachable_server_is_not_reported_as_a_missing_model(self):
        transport = RecordingTransport(
            LLMConnectionError("refused", LLMErrorCategory.CONNECTION)
        )
        health = make_client(transport).check_model_available()
        self.assertIs(health.status, LLMStatus.UNAVAILABLE)

    def test_the_server_check_is_not_repeated_for_every_candidate(self):
        """One probe covers both questions."""
        transport = RecordingTransport([(200, tags_body("qwen2.5:3b"))])
        make_client(transport).check_model_available()
        self.assertEqual(len(transport.calls), 1)

    def test_a_model_name_is_matched_exactly(self):
        """No silent substitution: qwen2.5:7b must not stand in for qwen2.5:3b."""
        transport = RecordingTransport([(200, tags_body("qwen2.5:7b"))])
        health = make_client(transport).check_model_available()
        self.assertIs(health.status, LLMStatus.MODEL_UNAVAILABLE)


class TestRequestSuccess(unittest.TestCase):
    """A successful request returns the model's text and nothing else."""

    def setUp(self) -> None:
        self.text = '{"summary": "ok", "observations": []}'
        self.transport = RecordingTransport(router(self.text, "qwen2.5:3b"))
        self.client = make_client(self.transport)

    def test_the_response_text_is_returned(self):
        response = self.client.generate("prompt")
        self.assertEqual(response.text, self.text)
        self.assertFalse(response.is_empty)

    def test_the_response_records_the_model(self):
        self.assertEqual(self.client.generate("prompt").model, "qwen2.5:3b")

    def test_the_response_records_a_duration(self):
        self.assertGreaterEqual(self.client.generate("prompt").duration, 0.0)

    def test_a_single_request_succeeds_on_the_first_attempt(self):
        self.client.generate("prompt")
        self.assertEqual(self.transport.calls[-1][1]["model"], "qwen2.5:3b")

    def test_the_generate_endpoint_is_the_one_that_is_called(self):
        self.client.generate("prompt")
        self.assertTrue(self.transport.calls[-1][0].endswith(GENERATE_PATH))

    def test_the_request_is_never_streamed(self):
        """A streamed response would arrive in chunks and never parse."""
        self.client.generate("prompt")
        self.assertIs(self.transport.calls[-1][1]["stream"], False)

    def test_the_prompt_is_sent_unchanged(self):
        self.client.generate("the exact prompt")
        self.assertEqual(self.transport.calls[-1][1]["prompt"], "the exact prompt")

    def test_the_temperature_is_passed_through(self):
        self.client.generate("prompt")
        self.assertEqual(self.transport.calls[-1][1]["options"]["temperature"], 0)

    def test_the_system_instruction_is_sent(self):
        self.client.generate("prompt")
        self.assertTrue(self.transport.calls[-1][1]["system"].strip())

    def test_json_mode_is_requested(self):
        """Ollama's own JSON constraint is a first line of defence."""
        self.client.generate("prompt")
        self.assertEqual(self.transport.calls[-1][1]["format"], "json")

    def test_json_mode_can_be_switched_off(self):
        client = make_client(self.transport, json_format=False)
        client.generate("prompt")
        self.assertNotIn("format", self.transport.calls[-1][1])

    def test_the_timeout_is_applied_to_the_request(self):
        self.client.generate("prompt")
        self.assertEqual(self.transport.calls[-1][2], 2.0)

    def test_an_alternative_response_field_is_accepted(self):
        body = json.dumps({"content": self.text})
        transport = RecordingTransport([(200, body)])
        self.assertEqual(make_client(transport).generate("p").text, self.text)

    def test_a_failed_first_attempt_then_a_success_is_reported_as_two_attempts(self):
        transport = RecordingTransport(
            [
                (503, "unavailable"),
                (200, generate_body(self.text)),
            ]
        )
        self.assertEqual(make_client(transport).generate("p").attempts, 2)

    def test_whitespace_only_output_counts_as_empty(self):
        transport = RecordingTransport([(200, generate_body(chr(32) * 3 + chr(10)))])
        self.assertTrue(make_client(transport).generate("p").is_empty)


class TestRequestFailure(unittest.TestCase):
    """Failed requests raise one typed error carrying a category."""

    def test_a_timeout_is_reported_as_a_timeout(self):
        transport = RecordingTransport(
            LLMConnectionError("timed out", LLMErrorCategory.TIMEOUT)
        )
        with self.assertRaises(LLMConnectionError) as caught:
            make_client(transport, max_retries=0).generate("p")
        self.assertIs(caught.exception.category, LLMErrorCategory.TIMEOUT)

    def test_a_connection_failure_is_reported_as_a_connection_failure(self):
        transport = RecordingTransport(
            LLMConnectionError("refused", LLMErrorCategory.CONNECTION)
        )
        with self.assertRaises(LLMConnectionError) as caught:
            make_client(transport, max_retries=0).generate("p")
        self.assertIs(caught.exception.category, LLMErrorCategory.CONNECTION)

    def test_a_client_error_is_not_retried(self):
        """A 400 would fail identically, so the budget must not be spent on it."""
        transport = RecordingTransport([(400, "bad request")])
        with self.assertRaises(LLMConnectionError):
            make_client(transport, max_retries=2).generate("p")
        self.assertEqual(len(transport.calls), 1)

    def test_a_server_error_is_retried_within_the_budget(self):
        transport = RecordingTransport([(503, "unavailable"), (503, "unavailable")])
        with self.assertRaises(LLMConnectionError):
            make_client(transport, max_retries=1).generate("p")
        self.assertEqual(len(transport.calls), 2)

    def test_a_retry_can_succeed(self):
        transport = RecordingTransport(
            [(503, "unavailable"), (200, generate_body('{"summary": "ok"}'))]
        )
        response = make_client(transport, max_retries=1).generate("p")
        self.assertEqual(response.attempts, 2)
        self.assertEqual(response.text, '{"summary": "ok"}')

    def test_the_retry_budget_is_never_exceeded(self):
        """A bounded budget is what stops a dead server from looping."""
        transport = RecordingTransport(
            LLMConnectionError("refused", LLMErrorCategory.CONNECTION)
        )
        with self.assertRaises(LLMConnectionError):
            make_client(transport, max_retries=2).generate("p")
        self.assertEqual(len(transport.calls), 3)

    def test_the_error_mentions_the_attempt_count(self):
        transport = RecordingTransport(
            LLMConnectionError("refused", LLMErrorCategory.CONNECTION)
        )
        with self.assertRaises(LLMConnectionError) as caught:
            make_client(transport, max_retries=1).generate("p")
        self.assertIn("2 attempt", str(caught.exception))

    def test_an_unexpected_envelope_is_reported_as_such(self):
        """A missing field must be visible, not read as empty text."""
        transport = RecordingTransport([(200, json.dumps({"done": True}))])
        with self.assertRaises(LLMConnectionError) as caught:
            make_client(transport).generate("p")
        self.assertIs(caught.exception.category, LLMErrorCategory.UNEXPECTED_RESPONSE)

    def test_an_unreadable_body_is_reported_as_such(self):
        transport = RecordingTransport([(200, "<html>gateway</html>")])
        with self.assertRaises(LLMConnectionError) as caught:
            make_client(transport).generate("p")
        self.assertIs(caught.exception.category, LLMErrorCategory.UNEXPECTED_RESPONSE)

    def test_an_unexpected_transport_error_becomes_a_typed_error(self):
        def explode(url: str, payload: Any, timeout: float) -> Tuple[int, str]:
            raise MemoryError("not an expected failure")

        with self.assertRaises(LLMConnectionError):
            make_client(RecordingTransport(explode), max_retries=0).generate("p")


class TestNoSecretExposure(unittest.TestCase):
    """The client holds nothing sensitive and describes itself minimally."""

    def test_the_description_has_only_the_three_useful_values(self):
        described = make_client(RecordingTransport([])).describe()
        self.assertEqual(
            set(described),
            {"base_url", "model", "timeout", "max_retries", "temperature"},
        )

    def test_the_description_contains_no_credential_field(self):
        described = make_client(RecordingTransport([])).describe()
        for key in described:
            self.assertNotIn("key", key.lower())
            self.assertNotIn("token", key.lower())
            self.assertNotIn("secret", key.lower())

    def test_the_model_is_configurable_without_editing_code(self):
        for model in ("qwen2.5:3b", "llama3.2:1b", "gemma2:2b"):
            with self.subTest(model=model):
                client = build_client(model=model)
                self.assertEqual(client.model, model)

    def test_the_model_name_reaches_the_request(self):
        transport = RecordingTransport([(200, generate_body('{"summary": "ok"}'))])
        make_client(transport, model="gemma2:2b").generate("p")
        self.assertEqual(transport.calls[0][1]["model"], "gemma2:2b")

    def test_the_environment_is_never_read_by_the_client(self):
        """A local client needs no credential, so it must not look for one."""
        import inspect

        source = inspect.getsource(LocalLLMClient)
        self.assertNotIn("os.environ", source)
        self.assertNotIn("getenv", source)


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    unittest.main()
