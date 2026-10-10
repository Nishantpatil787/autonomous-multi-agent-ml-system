"""The optional explanation and planning layer (Phase 8).

This is the only place that decides *whether* to call the local model, and it is
built so that the answer is almost always "no, or not usefully":

* the layer is off unless ``OLLAMA_ENABLED`` is set or the caller passes
  ``enabled=True``, so the default run never builds a client;
* a health check runs first, so an unreachable server or a missing model is
  detected before a prompt is even assembled;
* the whole request path is wrapped, so **no method here raises**. Every failure
  becomes an :class:`~llm.state.LLMResult` carrying a status and an error
  category, because a language model must never be able to stop a deterministic
  pipeline.

The architectural boundary is enforced by construction rather than by convention:
this module returns explanation records and nothing else. It cannot write to
``AgentState``, it cannot produce a metric, and it cannot return a routing
decision. The graph reads the results *after* it has already routed.

Each interpretation step follows the same shape, which is why the three public
methods differ only in their role:

    health check -> build context -> request -> parse -> (one repair) -> result
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from llm.client import (
    LLMConnectionError,
    LLMError,
    LLMNotConfiguredError,
    LLMUnsupportedHostError,
    LocalLLMClient,
    build_client,
)
from llm.parser import LLMParseError, ParsedLLMResponse, excerpt, parse_response
from llm.prompts import (
    PromptContext,
    build_context,
    build_prompt,
    build_repair_prompt,
)
from llm.state import (
    LLMErrorCategory,
    LLMHealth,
    LLMResponse,
    LLMResult,
    LLMRole,
    LLMStatus,
)
from utils import config
from utils.logger import setup_logger
from utils.state import AgentState

logger = setup_logger("llm.service")


class LLMService:
    """Optional interpretation service over a local model.

    Args:
        client: The client to use. Built from the configuration when None. The
            client is only constructed when the layer is enabled, so a disabled run
            needs no reachable model and no configured base URL.
        enabled: Whether the layer is active for this instance. None reads
            ``config.OLLAMA_ENABLED``.
        allow_repair: Whether one controlled repair request is allowed after an
            unusable response. Defaults to ``config.LLM_ALLOW_ONE_REPAIR``.
        health_check: Optional callable returning an :class:`~llm.state.LLMHealth`.
            Replaced in tests so no unit test needs a server.

    The instance caches exactly one thing between calls: the health snapshot. A
    service is built once per workflow run, so the probe is made once instead of
    once per interpretation step - which matters because a refused local connection
    costs several seconds on Windows and three identical probes would add ten
    seconds to a run on precisely the machines most likely to lack Ollama. Nothing
    else is shared: no prompt, no response and no parsed record survives a call,
    so one step cannot leak anything into the next.
    """

    def __init__(
        self,
        client: Optional[LocalLLMClient] = None,
        *,
        enabled: Optional[bool] = None,
        allow_repair: Optional[bool] = None,
        health_check: Optional[Callable[[], LLMHealth]] = None,
    ) -> None:
        self.enabled = bool(config.OLLAMA_ENABLED if enabled is None else enabled)
        self.allow_repair = bool(
            config.LLM_ALLOW_ONE_REPAIR if allow_repair is None else allow_repair
        )
        self._client = client
        self._health_check = health_check
        # Set only when a client could not be built; keeps the reason for the
        # failure instead of returning an unexplained None.
        self._configuration_error: str = ""
        # The single health snapshot for this run. See the class docstring.
        self._health: Optional[LLMHealth] = None

    # -- configuration --------------------------------------------------- #

    @property
    def model_name(self) -> str:
        """Return the configured model name, or an empty string when unset."""
        if self._client is not None:
            return self._client.model
        return str(config.OLLAMA_MODEL or "")

    def client(self) -> Optional[LocalLLMClient]:
        """Return the client, building it on first use.

        Returns:
            The client, or None when no client could be built.

        The client is built lazily so a disabled service, or a run that never
        reaches an interpretation step, does not validate a base URL it will not
        use. A configuration problem is returned to the caller as a status rather
        than raised.
        """
        if self._client is not None:
            return self._client
        if self._configuration_error:
            # Already tried and failed; retrying per call would repeat the warning.
            return None
        try:
            self._client = build_client()
        except LLMError as error:
            self._configuration_error = _flat(error)
            logger.warning(
                "%s client -> NOT CONFIGURED (%s)",
                config.LLM_LOG_PREFIX,
                self._configuration_error,
            )
        return self._client

    # -- health ---------------------------------------------------------- #

    def check_health(self, refresh: bool = False) -> LLMHealth:
        """Check that the local model can be used.

        The result is remembered for the lifetime of the service, which is one
        workflow run. A server that is down now is still down three nodes later, and
        a refused local connection is not cheap on Windows, so re-probing would only
        add delay without adding information.

        Args:
            refresh: Force a new probe instead of using the remembered snapshot.

        Returns:
            The health record. Never raises.
        """
        if not refresh and self._health is not None:
            return self._health

        self._health = self._probe()
        return self._health

    def _probe(self) -> LLMHealth:
        """Make one health probe and return its result.

        Returns:
            The health record. Never raises.
        """
        if self._health_check is not None:
            try:
                return self._health_check()
            except Exception as error:  # noqa: BLE001 - an optional layer cannot raise
                return LLMHealth(
                    status=LLMStatus.UNAVAILABLE,
                    detail=f"Unexpected failure during the health check: {_flat(error)}",
                    model=self.model_name,
                )

        client = self.client()
        if client is None:
            return LLMHealth(
                status=LLMStatus.UNAVAILABLE,
                detail=self._configuration_error or "No client could be built.",
                model=self.model_name,
            )
        try:
            return client.check_model_available()
        except Exception as error:  # noqa: BLE001 - an optional layer cannot raise
            return LLMHealth(
                status=LLMStatus.UNAVAILABLE,
                detail=f"Unexpected failure during the health check: {_flat(error)}",
                model=self.model_name,
            )

    # -- the three interpretation steps ----------------------------------- #

    def interpret_data_analysis(self, state: AgentState) -> LLMResult:
        """Interpret the Data Analysis Agent result.

        Args:
            state: The authoritative machine learning state.

        Returns:
            The :class:`~llm.state.LLMResult` for the data-analysis role.
        """
        return self._run(LLMRole.DATA_ANALYSIS, state)

    def interpret_evaluation(self, state: AgentState) -> LLMResult:
        """Interpret the evaluation result, including the model ranking.

        Args:
            state: The authoritative machine learning state.

        Returns:
            The :class:`~llm.state.LLMResult` for the evaluation role.
        """
        return self._run(LLMRole.EVALUATION, state)

    def plan_optimization(self, state: AgentState) -> LLMResult:
        """Propose a high-level optimisation plan.

        The plan is a suggestion: it is validated against the project's own model
        registry and cannot define a search space or choose which models are tuned.

        Args:
            state: The authoritative machine learning state.

        Returns:
            The :class:`~llm.state.LLMResult` for the optimisation-planner role.
        """
        return self._run(LLMRole.OPTIMIZATION, state)

    # -- the shared step -------------------------------------------------- #

    def _run(self, role: LLMRole, state: AgentState) -> LLMResult:
        """Run one interpretation step and return a result, never raising.

        Args:
            role: The interpretation step.
            state: The authoritative machine learning state.

        Returns:
            The :class:`~llm.state.LLMResult`.
        """
        started = time.perf_counter()

        if not self.enabled:
            return self._record(
                role,
                LLMStatus.DISABLED,
                detail=(
                    "The optional local LLM layer is disabled for this run; no request "
                    "was made and no interpretation was produced."
                ),
                error_category=LLMErrorCategory.NOT_CONFIGURED,
                started=started,
            )

        health = self.check_health()
        if health.status is LLMStatus.MODEL_UNAVAILABLE:
            return self._record(
                role,
                LLMStatus.MODEL_UNAVAILABLE,
                detail=health.detail,
                error_category=LLMErrorCategory.NOT_CONFIGURED,
                started=started,
                model=health.model,
            )
        if not health.available:
            return self._record(
                role,
                LLMStatus.UNAVAILABLE,
                detail=health.detail,
                error_category=LLMErrorCategory.CONNECTION,
                started=started,
                model=health.model,
            )

        try:
            context = build_context(state, role)
        except Exception as error:  # noqa: BLE001 - recorded, never propagated
            return self._record(
                role,
                LLMStatus.REQUEST_FAILED,
                detail=f"The prompt context could not be built: {_flat(error)}",
                error_category=LLMErrorCategory.SCHEMA_VIOLATION,
                started=started,
                model=health.model,
            )

        if context.is_empty():
            return self._record(
                role,
                LLMStatus.INVALID_RESPONSE,
                detail=(
                    "No summarised context was available for this step, so there was "
                    "nothing to interpret and no request was made."
                ),
                error_category=LLMErrorCategory.SCHEMA_VIOLATION,
                started=started,
                model=health.model,
                context_fields=context.field_names,
            )

        client = self.client()
        if client is None:
            return self._record(
                role,
                LLMStatus.UNAVAILABLE,
                detail=self._configuration_error or "No client could be built.",
                error_category=LLMErrorCategory.NOT_CONFIGURED,
                started=started,
                context_fields=context.field_names,
            )

        try:
            response = client.generate(build_prompt(context), role=role)
        except LLMConnectionError as error:
            return self._record(
                role,
                LLMStatus.REQUEST_FAILED,
                detail=_flat(error),
                error_category=error.category,
                started=started,
                model=health.model,
                context_fields=context.field_names,
            )
        except LLMUnsupportedHostError as error:
            return self._record(
                role,
                LLMStatus.UNAVAILABLE,
                detail=_flat(error),
                error_category=LLMErrorCategory.UNSUPPORTED_HOST,
                started=started,
                model=health.model,
                context_fields=context.field_names,
            )
        except LLMNotConfiguredError as error:
            return self._record(
                role,
                LLMStatus.MODEL_UNAVAILABLE,
                detail=_flat(error),
                error_category=LLMErrorCategory.NOT_CONFIGURED,
                started=started,
                model=health.model,
                context_fields=context.field_names,
            )
        except LLMError as error:
            return self._record(
                role,
                LLMStatus.REQUEST_FAILED,
                detail=_flat(error),
                error_category=LLMErrorCategory.CONNECTION,
                started=started,
                model=health.model,
                context_fields=context.field_names,
            )
        except Exception as error:  # noqa: BLE001 - recorded, never propagated
            return self._record(
                role,
                LLMStatus.REQUEST_FAILED,
                detail=f"Unexpected failure while requesting: {_flat(error)}",
                error_category=LLMErrorCategory.CONNECTION,
                started=started,
                model=health.model,
                context_fields=context.field_names,
            )

        return self._finish(role, context, response, health.model, started)

    def _finish(
        self,
        role: LLMRole,
        context: PromptContext,
        response: LLMResponse,
        model: str,
        started: float,
    ) -> LLMResult:
        """Parse a response and, once, try the controlled repair.

        Args:
            role: The interpretation step.
            context: The context that produced the prompt.
            response: The raw model response.
            model: The model name.
            started: The value ``time.perf_counter()`` returned at the start.

        Returns:
            The final :class:`~llm.state.LLMResult` for the step.
        """
        if response.is_empty:
            return self._record(
                role,
                LLMStatus.INVALID_RESPONSE,
                detail="The local model returned an empty response.",
                error_category=LLMErrorCategory.EMPTY_RESPONSE,
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts,
            )

        parsed: ParsedLLMResponse = parse_response(response.text, role, model=model)
        if parsed.ok:
            return self._record(
                role,
                LLMStatus.COMPLETED,
                detail="A valid structured response was parsed.",
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts,
                repaired=response.repaired,
                parsed=parsed,
            )

        if not (self.allow_repair and parsed.error is not None and parsed.error.repairable):
            return self._record(
                role,
                LLMStatus.INVALID_RESPONSE,
                detail=self._parse_detail(parsed.error),
                error_category=(
                    parsed.error.category
                    if parsed.error is not None
                    else LLMErrorCategory.SCHEMA_VIOLATION
                ),
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts,
            )

        logger.info(
            "%s %s -> REPAIR 1/1 (%s)",
            config.LLM_LOG_PREFIX,
            role.display_name,
            self._parse_detail(parsed.error),
        )
        client = self.client()
        if client is None:
            return self._record(
                role,
                LLMStatus.INVALID_RESPONSE,
                detail=self._parse_detail(parsed.error),
                error_category=parsed.error.category,
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts,
            )

        try:
            repaired = client.generate(build_repair_prompt(context), role=role)
        except LLMConnectionError as error:
            return self._record(
                role,
                LLMStatus.REQUEST_FAILED,
                detail=(
                    f"The repair request failed: {_flat(error)} "
                    f"(first response: {self._parse_detail(parsed.error)})"
                ),
                error_category=error.category,
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts,
            )
        except Exception as error:  # noqa: BLE001 - recorded, never propagated
            return self._record(
                role,
                LLMStatus.REQUEST_FAILED,
                detail=(
                    f"Unexpected failure during the repair request: {_flat(error)} "
                    f"(first response: {self._parse_detail(parsed.error)})"
                ),
                error_category=LLMErrorCategory.CONNECTION,
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts,
            )

        retry = parse_response(repaired.text, role, model=model)
        if retry.ok:
            return self._record(
                role,
                LLMStatus.COMPLETED,
                detail="A valid structured response was parsed after one repair request.",
                started=started,
                model=model,
                context_fields=context.field_names,
                attempts=response.attempts + repaired.attempts,
                repaired=True,
                parsed=retry,
            )

        return self._record(
            role,
            LLMStatus.INVALID_RESPONSE,
            detail=(
                "The response was still unusable after one repair request. "
                f"First: {self._parse_detail(parsed.error)} "
                f"Repair: {self._parse_detail(retry.error)}"
            ),
            error_category=(
                retry.error.category
                if retry.error is not None
                else LLMErrorCategory.SCHEMA_VIOLATION
            ),
            started=started,
            model=model,
            context_fields=context.field_names,
            attempts=response.attempts + repaired.attempts,
            repaired=True,
        )

    # -- reporting -------------------------------------------------------- #

    def _record(
        self,
        role: LLMRole,
        status: LLMStatus,
        *,
        started: float,
        detail: str = "",
        error_category: LLMErrorCategory = LLMErrorCategory.NONE,
        model: str = "",
        context_fields: Any = (),
        attempts: int = 0,
        repaired: bool = False,
        parsed: Optional[ParsedLLMResponse] = None,
    ) -> LLMResult:
        """Build a result record and emit the matching one-line log entry.

        Args:
            role: The interpretation step.
            status: The outcome.
            started: The value ``time.perf_counter()`` returned at the start.
            detail: A one-line explanation, already log-safe.
            error_category: Which kind of failure occurred.
            model: The model name.
            context_fields: The context field names that were sent.
            attempts: How many requests were made.
            repaired: Whether a repair request was used.
            parsed: The validated record, when there is one.

        Returns:
            The :class:`~llm.state.LLMResult`.
        """
        duration = time.perf_counter() - started
        result = LLMResult(
            role=role,
            status=status,
            model=model or self.model_name,
            detail=detail,
            error_category=error_category,
            duration=duration,
            attempts=attempts,
            repaired=repaired,
            context_fields=tuple(context_fields),
            interpretation=parsed.interpretation if parsed is not None else None,
            plan=parsed.plan if parsed is not None else None,
        )
        self._log(result)
        return result

    def _log(self, result: LLMResult) -> None:
        """Emit one single-line log entry for a result.

        The line never contains the prompt, the context or the generated text, so a
        log file can never leak project data. Only the status, the model, the
        duration and a bounded reason are recorded.
        """
        label = result.role.display_name
        if result.status is LLMStatus.COMPLETED:
            detail = f"model={result.model or 'unknown'}, {result.duration:.2f} s"
            if result.repaired:
                detail += ", after one repair request"
            claims = result.forbidden_claims
            if claims:
                detail += f", {len(claims)} phrasing warning(s)"
            logger.info("%s %s -> COMPLETED (%s)", config.LLM_LOG_PREFIX, label, detail)
        elif result.status is LLMStatus.DISABLED:
            logger.info("%s %s -> DISABLED", config.LLM_LOG_PREFIX, label)
        else:
            logger.warning(
                "%s %s -> %s (%s: %s)",
                config.LLM_LOG_PREFIX,
                label,
                result.status.value.upper(),
                result.error_category.value,
                _flat(result.detail, 160),
            )

    @staticmethod
    def _parse_detail(error: Optional[LLMParseError]) -> str:
        """Render a parse error as a one-line reason."""
        if error is None:
            return "the response could not be parsed."
        text = f"{error.category.value}: {error.detail}"
        if error.raw_excerpt:
            text = f"{text} (response began: {excerpt(error.raw_excerpt, 120)})"
        return _flat(text)


def _flat(value: Any, limit: int = 240) -> str:
    """Flatten a value to one bounded line."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_service(enabled: bool) -> LLMService:
    """Build a service for one run.

    Args:
        enabled: Whether the layer should be active for this run. False builds a
            service that will return a ``DISABLED`` result without ever creating a
            client.

    Returns:
        The service.
    """
    return LLMService(enabled=enabled)


__all__ = ["LLMService", "build_service"]
