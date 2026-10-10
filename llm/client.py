"""HTTP transport and local availability checks for Ollama (Phase 8).

This is the only module in the project that knows how to talk to Ollama. The rest
of the code depends on :class:`LocalLLMClient` and on the typed records in
:mod:`llm.state`, never on an HTTP endpoint, a URL or an Ollama request shape.

Design decisions worth stating explicitly:

* **No third-party client.** Ollama speaks plain HTTP and JSON over a loopback
  socket, so the standard library's :mod:`urllib.request` is enough. The project
  therefore adds no dependency in Phase 8, keeps working when the official
  ``ollama`` package is not installed, and stays testable without a server. The
  official client is deliberately not used: it would add a dependency for two
  endpoints and would make the tests depend on a mock of that library instead of
  on a real HTTP contract.
* **Local only.** :func:`require_local_url` rejects any host that is not
  loopback, so a mistyped base URL fails immediately instead of sending project
  data to a remote service.
* **Nothing raises out.** Every method returns a typed record carrying a status
  and an error category. A client is an optional decoration on a deterministic
  pipeline, so it must never be able to stop it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from llm.prompts import SYSTEM_INSTRUCTION
from llm.state import (
    LLMErrorCategory,
    LLMHealth,
    LLMResponse,
    LLMRole,
    LLMStatus,
)
from utils import config
from utils.logger import setup_logger

logger = setup_logger("llm.client")

#: Ollama's tag endpoint: reports which models are installed locally.
TAGS_PATH = "/api/tags"

#: Ollama's completion endpoint.
GENERATE_PATH = "/api/generate"

#: HTTP status codes worth one retry. A 4xx is a client-side problem that would
#: fail identically on a second attempt, so only server-side and rate-limit
#: responses are retried.
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class LLMError(RuntimeError):
    """Base class for every error this layer raises on purpose.

    It exists so the service layer can catch exactly the failures this layer
    knows about. Anything else that escapes is treated as unexpected, which is
    still recorded rather than propagated.
    """


class LLMNotConfiguredError(LLMError):
    """The optional local LLM layer is disabled or has no model configured."""


class LLMUnsupportedHostError(LLMError):
    """The configured base URL does not point at this machine."""


class LLMConnectionError(LLMError):
    """The Ollama server could not be reached."""

    def __init__(self, message: str, category: LLMErrorCategory) -> None:
        """Store the failure category alongside the message.

        Args:
            message: The one-line reason the request failed.
            category: Whether this was a timeout, a refusal or an HTTP error.
        """
        super().__init__(message)
        self.category = category


# --------------------------------------------------------------------------- #
# Locality
# --------------------------------------------------------------------------- #


def require_local_url(base_url: str) -> str:
    """Validate that ``base_url`` addresses the local machine.

    Phase 8 is local-first by construction, so this is an allowlist rather than a
    warning: a base URL that points anywhere else is rejected before a single byte
    of project context is sent.

    Args:
        base_url: The configured Ollama base URL.

    Returns:
        The URL with any trailing slash removed.

    Raises:
        LLMUnsupportedHostError: If the scheme is not HTTP(S) or the host is not
            loopback.
    """
    parsed = urlparse(str(base_url or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise LLMUnsupportedHostError(
            f"{base_url!r} is not a usable Ollama base URL: it needs an http(s) "
            "scheme and a host."
        )
    if host not in config.LLM_ALLOWED_HOSTS:
        allowed = ", ".join(config.LLM_ALLOWED_HOSTS)
        raise LLMUnsupportedHostError(
            f"The Ollama base URL host {host!r} is not local. Phase 8 only talks to "
            f"this machine ({allowed}), so project data can never be sent off-host."
        )
    return str(base_url).rstrip("/")


def is_local_url(base_url: str) -> bool:
    """Return True when ``base_url`` addresses the local machine.

    Args:
        base_url: The configured Ollama base URL.

    Returns:
        True when :func:`require_local_url` would accept the URL.
    """
    try:
        require_local_url(base_url)
    except LLMUnsupportedHostError:
        return False
    return True


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #

#: Signature of a request callable, injected for tests. It receives the full URL,
#: the JSON payload and the timeout, and returns ``(status_code, body_text)``.
Transport = Callable[[str, Dict[str, Any], float], Tuple[int, str]]


def urllib_transport(url: str, payload: Optional[Mapping[str, Any]], timeout: float) -> Tuple[int, str]:
    """Perform one HTTP request with the standard library.

    Args:
        url: Absolute URL to call.
        payload: JSON body, or None for a GET.
        timeout: Socket timeout in seconds.

    Returns:
        ``(status_code, body_text)``.

    Raises:
        LLMConnectionError: On a timeout, a refused connection or any other
            transport failure. The error category distinguishes the three.
    """
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), body
    except urllib.error.HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a body is a nicety, not a requirement
            body = ""
        return int(error.code), body
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise LLMConnectionError(
                f"The request to the local model timed out after {timeout:g}s.",
                LLMErrorCategory.TIMEOUT,
            ) from error
        raise LLMConnectionError(
            f"The Ollama server could not be reached: {_one_line(reason)}",
            LLMErrorCategory.CONNECTION,
        ) from error
    except TimeoutError as error:
        raise LLMConnectionError(
            f"The request to the local model timed out after {timeout:g}s.",
            LLMErrorCategory.TIMEOUT,
        ) from error
    except OSError as error:
        raise LLMConnectionError(
            f"The Ollama server could not be reached: {_one_line(error)}",
            LLMErrorCategory.CONNECTION,
        ) from error


def _one_line(value: Any, limit: int = 200) -> str:
    """Flatten a value to one bounded line for a log or an error message.

    Args:
        value: Any value, converted with ``str``.
        limit: Maximum length.

    Returns:
        A single-line, length-bounded string.
    """
    flattened = " ".join(str(value).split())
    return flattened if len(flattened) <= limit else flattened[: limit - 3] + "..."


@dataclass
class LocalLLMClient:
    """Talks to a local Ollama server and never raises at its caller.

    The client owns the transport, the timeout, the bounded retry and the parsing
    of Ollama's own envelope. It does **not** validate the model's content: that is
    :mod:`llm.parser`'s job, and keeping the two apart is what lets the parser be
    tested without any HTTP at all.

    Args:
        base_url: Ollama base URL. Must be loopback; checked immediately so a
            wrong value is reported before any request is attempted.
        model: Local model name. Never downloaded by this project.
        timeout: Per-request wall-clock budget in seconds.
        max_retries: Extra attempts allowed for a transient failure.
        temperature: Sampling temperature. 0 keeps a local model as close to
            reproducible as it gets.
        transport: Request callable. Replaced in tests, so no test needs a server.
        json_format: Whether to ask Ollama to constrain generation to JSON.
    """

    base_url: str
    model: str
    timeout: float = 30.0
    max_retries: int = 1
    temperature: float = 0.0
    transport: Transport = urllib_transport
    json_format: bool = True

    def __post_init__(self) -> None:
        """Normalise and validate the configuration.

        Raises:
            LLMNotConfiguredError: If no model was configured.
            LLMUnsupportedHostError: If the base URL is not loopback.
        """
        self.base_url = require_local_url(self.base_url)
        self.model = str(self.model or "").strip()
        if not self.model:
            raise LLMNotConfiguredError(
                "No Ollama model is configured. Set OLLAMA_MODEL or "
                "config.OLLAMA_MODEL to a model that is already installed locally."
            )
        self.timeout = max(0.1, float(self.timeout))
        self.max_retries = max(0, int(self.max_retries))
        self.temperature = float(self.temperature)

    # -- introspection --------------------------------------------------- #

    def describe(self) -> Dict[str, Any]:
        """Return a log-safe description of the configuration.

        No secret is ever stored in this client, so there is nothing to redact;
        the method exists to make that explicit and to keep the log line to the
        three values an operator actually needs.

        Returns:
            A mapping with the base URL, the model and the timeout.
        """
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
        }

    # -- availability ---------------------------------------------------- #

    def check_ollama_available(self) -> LLMHealth:
        """Check that the Ollama server answers.

        Args:
            None.

        Returns:
            An :class:`~llm.state.LLMHealth` reporting ``AVAILABLE`` or
            ``UNAVAILABLE`` with a one-line reason. Never raises.
        """
        started = time.perf_counter()
        try:
            status_code, body = self.transport(self._url(TAGS_PATH), None, self.timeout)
        except LLMConnectionError as error:
            return self._unavailable(
                f"{_one_line(error)}. Start the Ollama application and try again, or "
                "run without --llm: the pipeline does not need it.",
                started,
            )
        except Exception as error:  # noqa: BLE001 - an optional layer cannot raise
            return self._unavailable(
                f"Unexpected failure while checking Ollama: {_one_line(error)}",
                started,
            )

        if status_code != 200:
            return self._unavailable(
                f"Ollama answered with HTTP {status_code} on {TAGS_PATH}.",
                started,
            )

        models = self._installed_models(body)
        if models is None:
            return self._unavailable(
                "Ollama answered but the tag list could not be read, so the local "
                "installation is not usable as configured.",
                started,
            )

        return LLMHealth(
            status=LLMStatus.AVAILABLE,
            detail=f"Ollama is running at {self.base_url} with "
            f"{len(models)} model(s) installed.",
            model=self.model,
            available_models=models,
            duration=time.perf_counter() - started,
        )

    def check_model_available(self) -> LLMHealth:
        """Check that the configured model is installed locally.

        Args:
            None.

        Returns:
            An :class:`~llm.state.LLMHealth` reporting ``AVAILABLE`` when the model
            is installed, ``MODEL_UNAVAILABLE`` when Ollama answers but the model is
            missing, or ``UNAVAILABLE`` when Ollama itself cannot be reached.
            Never raises.
        """
        server = self.check_ollama_available()
        if not server.available:
            return LLMHealth(
                status=LLMStatus.UNAVAILABLE,
                detail=server.detail,
                model=self.model,
                available_models=server.available_models,
                duration=server.duration,
            )

        if self.model in server.available_models:
            return LLMHealth(
                status=LLMStatus.AVAILABLE,
                detail=f"Model '{self.model}' is installed locally.",
                model=self.model,
                available_models=server.available_models,
                duration=server.duration,
            )

        installed = ", ".join(server.available_models) or "none"
        return LLMHealth(
            status=LLMStatus.MODEL_UNAVAILABLE,
            detail=(
                f"Ollama is running, but the configured model '{self.model}' is not "
                f"installed locally (installed: {installed}). "
                f"{config.LLM_MODEL_UNAVAILABLE_NOTE}"
            ),
            model=self.model,
            available_models=server.available_models,
            duration=server.duration,
        )

    # -- requests -------------------------------------------------------- #

    def generate(self, prompt: str, *, system: str = "", role: Optional[LLMRole] = None) -> LLMResponse:
        """Send one prompt to the local model and return its raw answer.

        A malformed Ollama envelope is reported as
        :attr:`~LLMErrorCategory.UNEXPECTED_RESPONSE` rather than guessed at, so a
        wrong response shape is visible instead of silently read as empty text.

        Args:
            prompt: The user prompt. Already bounded by
                :func:`llm.prompts.build_prompt`.
            system: The system instruction. Empty falls back to the shared one.
            role: The calling role, used for the log line only.

        Returns:
            The :class:`~llm.state.LLMResponse` with the model's text.

        Raises:
            LLMConnectionError: When every attempt failed.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system or SYSTEM_INSTRUCTION,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.json_format:
            # Ollama's own JSON mode. This is a first line of defence only: the
            # parser validates the text itself regardless of what came back.
            payload["format"] = "json"

        started = time.perf_counter()
        last_category = LLMErrorCategory.CONNECTION
        attempts = 0

        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                status_code, body = self.transport(
                    self._url(GENERATE_PATH), payload, self.timeout
                )
            except LLMConnectionError as error:
                last_category = error.category
                self._log_retry(role, attempt, _one_line(error))
                continue
            except Exception as error:  # noqa: BLE001 - recorded, never propagated
                raise LLMConnectionError(
                    f"Unexpected failure while sending the request: {_one_line(error)}",
                    LLMErrorCategory.CONNECTION,
                ) from error

            if status_code != 200:
                message = f"Ollama answered with HTTP {status_code}."
                if status_code in _RETRYABLE_HTTP_STATUS and attempt <= self.max_retries:
                    self._log_retry(role, attempt, message)
                    last_category = LLMErrorCategory.HTTP_ERROR
                    continue
                raise LLMConnectionError(message, LLMErrorCategory.HTTP_ERROR)

            text = self._extract_response_text(body)
            if text is None:
                raise LLMConnectionError(
                    "Ollama returned a response without a readable 'response' field.",
                    LLMErrorCategory.UNEXPECTED_RESPONSE,
                )
            return LLMResponse(
                text=text,
                model=self.model,
                duration=time.perf_counter() - started,
                attempts=attempts,
            )

        raise LLMConnectionError(
            f"The local model did not answer after {attempts} attempt(s).",
            last_category,
        )

    # -- internals ------------------------------------------------------- #

    def _url(self, path: str) -> str:
        """Return the absolute URL for an Ollama endpoint."""
        return f"{self.base_url}{path}"

    def _unavailable(self, detail: str, started: float) -> LLMHealth:
        """Build and log an ``UNAVAILABLE`` health record.

        Args:
            detail: The one-line reason, already actionable.
            started: The value returned by ``time.perf_counter()`` at the start.

        Returns:
            The health record.
        """
        logger.warning("%s Ollama -> UNAVAILABLE (%s)", config.LLM_LOG_PREFIX, detail)
        return LLMHealth(
            status=LLMStatus.UNAVAILABLE,
            detail=detail,
            model=self.model,
            duration=time.perf_counter() - started,
        )

    @staticmethod
    def _installed_models(body: str) -> Optional[Tuple[str, ...]]:
        """Return the locally installed model names from a tag-list body.

        Args:
            body: The response body of ``/api/tags``.

        Returns:
            The model names, or None when the body could not be read.
        """
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        entries = payload.get("models")
        if not isinstance(entries, list):
            return None
        names: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model")
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(names)

    @staticmethod
    def _extract_response_text(body: str) -> Optional[str]:
        """Return the model's text from an Ollama generate body.

        Args:
            body: The response body of ``/api/generate``.

        Returns:
            The text, or None when the envelope was not readable.
        """
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        for key in ("response", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return None

    def _log_retry(self, role: Optional[LLMRole], attempt: int, detail: str) -> None:
        """Log one bounded retry without logging the prompt or the response."""
        label = role.display_name if role is not None else "Request"
        logger.warning(
            "%s %s -> RETRY %d/%d (%s)",
            config.LLM_LOG_PREFIX,
            label,
            attempt,
            self.max_retries,
            detail,
        )


def build_client(**overrides: Any) -> LocalLLMClient:
    """Build a client from the configuration, with optional overrides.

    Args:
        **overrides: Any field of :class:`LocalLLMClient`, for example ``model``.

    Returns:
        The configured client.

    Raises:
        LLMNotConfiguredError: If no model is configured.
        LLMUnsupportedHostError: If the configured base URL is not loopback.
    """
    settings: Dict[str, Any] = {
        "base_url": config.OLLAMA_BASE_URL,
        "model": config.OLLAMA_MODEL,
        "timeout": config.OLLAMA_TIMEOUT,
        "max_retries": config.OLLAMA_MAX_RETRIES,
        "temperature": config.OLLAMA_TEMPERATURE,
        "json_format": config.OLLAMA_JSON_FORMAT,
    }
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return LocalLLMClient(**settings)


__all__ = [
    "GENERATE_PATH",
    "TAGS_PATH",
    "LLMConnectionError",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMUnsupportedHostError",
    "LocalLLMClient",
    "Transport",
    "build_client",
    "is_local_url",
    "require_local_url",
    "urllib_transport",
]