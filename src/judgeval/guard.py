from __future__ import annotations

import asyncio
import functools
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, TypeVar, cast

from opentelemetry.trace import Span, Status, StatusCode

from judgeval.background_queue import enqueue as bg_enqueue
from judgeval.internal.api.models import GuardResponse
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.logger import judgeval_logger
from judgeval.trace.base_tracer import BaseTracer, _format_inputs
from judgeval.utils.serialize import serialize_attribute

C = TypeVar("C", bound=Callable[..., Any])

Position = Literal["input", "tool_input", "tool_output", "output"]
Mode = Literal["shadow", "enforce"]
FailMode = Literal["open", "closed"]
Outcome = Literal["pass", "fail", "unsure", "error"]

# ponytail: the API client has one 30 s httpx timeout, so an enforce check that
# times out keeps its worker until then; pass a per-call timeout if this pool backs up.
_enforce_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="judgeval-guard")


@dataclass(frozen=True)
class GuardDecision:
    """What a guard concluded about one check.

    Attributes:
        outcome: `pass`, `fail` (confident violation), `unsure` (violation
            below the threshold) or `error` (the check failed or timed out).
        would_block: Whether the check would block in enforce mode.
        blocked: Whether it did block (`would_block` and mode is enforce).
        confidence: The judge's confidence, or None on `error`.
        reason: The server's reason, or the error message.
    """

    outcome: Outcome
    would_block: bool
    blocked: bool
    confidence: Optional[float]
    reason: str


class GuardTripped(Exception):
    """Raised by `Guard.check()` when an enforce-mode guard blocks."""

    def __init__(self, decision: GuardDecision):
        super().__init__(decision.reason)
        self.decision = decision


_CHECK_FAILED = "Guard check failed"


def _refusal(decision: GuardDecision) -> str:
    return f"Blocked by guardrail: {decision.reason}"


class Guard:
    """A live guardrail: a binary Judgment judge checked inline on agent input,
    tool calls or output.

    The judge is written as a violation detector (`true` = violation). Only a
    confident violation (`confidence >= threshold`) fails; low confidence and
    errors follow `fail_mode`. In `shadow` mode (the default) the check runs in
    the background and never blocks; in `enforce` mode it waits up to
    `timeout_s` and blocks. Each check is recorded as a `judgment.guard` child
    span of the current span.

    Args:
        judge_id: ID of a binary, prompt-backed judge in the tracer's project.
        mode: `"shadow"` (record only) or `"enforce"` (block).
        fail_mode: `"open"` lets unsure or failed checks through,
            `"closed"` blocks them.
        threshold: Minimum confidence for a violation to block.
        timeout_s: How long an enforce check may take before it counts as an error.

    Examples:
        ```python
        no_external_email = Guard(judge_id="...", mode="enforce", fail_mode="closed")
        off_topic = Guard(judge_id="...")

        @Tracer.observe(span_type="tool")
        @no_external_email.tool(evidence=lambda to, subject, body: {"to": to})
        def send_email(to: str, subject: str, body: str) -> str: ...

        @Tracer.observe(span_type="agent")
        def agent(message: str) -> str:
            off_topic.check("input", {"message": message})
            ...
        ```
    """

    __slots__ = ("judge_id", "mode", "fail_mode", "threshold", "timeout_s")

    def __init__(
        self,
        judge_id: str,
        *,
        mode: Mode = "shadow",
        fail_mode: FailMode = "open",
        threshold: float = 0.8,
        timeout_s: float = 3.0,
    ):
        self.judge_id = judge_id
        self.mode = mode
        self.fail_mode = fail_mode
        self.threshold = threshold
        self.timeout_s = timeout_s

    def check(self, position: Position, evidence: Any) -> Optional[GuardDecision]:
        """Check `evidence` at `position` (usually `"input"` or `"output"`).

        Returns:
            The decision in enforce mode; None in shadow mode, where the check
            finishes in the background.

        Raises:
            GuardTripped: In enforce mode, when the check blocks.
        """
        decision = self._run(position, evidence)
        if decision is not None and decision.blocked:
            raise GuardTripped(decision)
        return decision

    async def acheck(
        self, position: Position, evidence: Any
    ) -> Optional[GuardDecision]:
        """Async form of `check()`."""
        return await asyncio.to_thread(self.check, position, evidence)

    def tool(
        self,
        position: Literal["tool_input", "tool_output"] = "tool_input",
        evidence: Optional[Callable[..., Any]] = None,
        on_block: Optional[Callable[[GuardDecision], Any]] = None,
    ) -> Callable[[C], C]:
        """Decorator that guards a tool function (sync or async).

        Place it under `@Tracer.observe(span_type="tool")` so the check is a
        child of the tool span. `tool_input` checks the arguments before the
        call and skips the call on block; `tool_output` checks the result.
        On block the tool returns `on_block(decision)` instead, by default
        `"Blocked by guardrail: <reason>"`, so the agent can recover.

        Args:
            position: `"tool_input"` or `"tool_output"`.
            evidence: Builds the evidence sent to the judge. Called with the
                tool's arguments, plus the result as the first argument for
                `tool_output`. Defaults to
                `{"tool": <name>, "input": <arguments>}` (+ `"output"`).
            on_block: Returns the value the tool returns when blocked.
        """
        refuse = on_block or _refusal

        def decorator(f: C) -> C:
            def build_evidence(args: tuple, kwargs: dict, *result: Any) -> Any:
                if evidence is not None:
                    return evidence(*result, *args, **kwargs)
                built = {"tool": f.__name__, "input": _format_inputs(f, args, kwargs)}
                if result:
                    built["output"] = result[0]
                return built

            if inspect.iscoroutinefunction(f):

                @functools.wraps(f)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if position == "tool_input" and (
                        blocked := await asyncio.to_thread(
                            self._blocking_decision,
                            position,
                            build_evidence(args, kwargs),
                        )
                    ):
                        return refuse(blocked)
                    result = await f(*args, **kwargs)
                    if position == "tool_output" and (
                        blocked := await asyncio.to_thread(
                            self._blocking_decision,
                            position,
                            build_evidence(args, kwargs, result),
                        )
                    ):
                        return refuse(blocked)
                    return result

                return cast(C, async_wrapper)

            @functools.wraps(f)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if position == "tool_input" and (
                    blocked := self._blocking_decision(
                        position, build_evidence(args, kwargs)
                    )
                ):
                    return refuse(blocked)
                result = f(*args, **kwargs)
                if position == "tool_output" and (
                    blocked := self._blocking_decision(
                        position, build_evidence(args, kwargs, result)
                    )
                ):
                    return refuse(blocked)
                return result

            return cast(C, sync_wrapper)

        return decorator

    def _blocking_decision(
        self, position: Position, evidence: Any
    ) -> Optional[GuardDecision]:
        """Runs the check and returns its decision only when it blocks."""
        decision = self._run(position, evidence)
        return decision if decision is not None and decision.blocked else None

    def _run(self, position: Position, evidence: Any) -> Optional[GuardDecision]:
        """Starts the check's span under the current span, then evaluates
        inline (enforce) or in the background queue (shadow)."""
        span = BaseTracer.getOTELTracer().start_span(
            "judgment.guard",
            attributes={
                AttributeKeys.JUDGMENT_SPAN_KIND: "guard",
                AttributeKeys.JUDGMENT_GUARD_JUDGE_ID: self.judge_id,
                AttributeKeys.JUDGMENT_GUARD_POSITION: position,
                AttributeKeys.JUDGMENT_GUARD_MODE: self.mode,
                AttributeKeys.JUDGMENT_INPUT: serialize_attribute(
                    {"position": position, "evidence": evidence},
                    BaseTracer._get_serializer(),
                ),
            },
        )
        request = self._request(position, evidence)

        if self.mode == "enforce":
            return self._evaluate(
                span,
                lambda: _enforce_pool.submit(request).result(timeout=self.timeout_s),
            )
        if not bg_enqueue(lambda: self._evaluate(span, request)):
            self._evaluate(span, _raise_queue_full)
        return None

    def _request(
        self, position: Position, evidence: Any
    ) -> Callable[[], GuardResponse]:
        """Binds the API call to the active tracer's project and client, the
        same ones `Tracer.tag` uses."""
        tracer = BaseTracer._get_proxy_provider().get_active_tracer()
        client = tracer._client if tracer else None
        project_id = tracer.project_id if tracer else None

        def request() -> GuardResponse:
            if client is None or project_id is None:
                raise RuntimeError("No active tracer with a project to run the guard")
            return client.post_projects_guard(
                project_id,
                {"judge_id": self.judge_id, "position": position, "evidence": evidence},
            )

        return request

    def _evaluate(
        self, span: Span, request: Callable[[], GuardResponse]
    ) -> GuardDecision:
        """Runs the request, decides, and records the decision on the span."""
        started = time.perf_counter()
        response: Optional[GuardResponse]
        try:
            response = request()
            reason = response["reason"]
            span.set_attribute(
                AttributeKeys.JUDGMENT_OUTPUT,
                serialize_attribute(response, BaseTracer._get_serializer()),
            )
        except Exception as e:
            judgeval_logger.warning(f"[Guard] Check on {self.judge_id} failed: {e!r}")
            response = None
            reason = _CHECK_FAILED
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, repr(e)))

        decision = self._decide(response, reason)
        span.set_attribute(AttributeKeys.JUDGMENT_GUARD_OUTCOME, decision.outcome)
        span.set_attribute(
            AttributeKeys.JUDGMENT_GUARD_WOULD_BLOCK, decision.would_block
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_GUARD_LATENCY_MS,
            (time.perf_counter() - started) * 1000,
        )
        if decision.confidence is not None:
            span.set_attribute(
                AttributeKeys.JUDGMENT_GUARD_CONFIDENCE, decision.confidence
            )
        span.end()
        return decision

    def _decide(self, response: Optional[GuardResponse], reason: str) -> GuardDecision:
        outcome: Outcome
        if response is None:
            outcome = "error"
        elif not response["result"]:
            outcome = "pass"
        elif response["confidence"] >= self.threshold:
            outcome = "fail"
        else:
            outcome = "unsure"
        would_block = outcome == "fail" or (
            outcome != "pass" and self.fail_mode == "closed"
        )
        return GuardDecision(
            outcome=outcome,
            would_block=would_block,
            blocked=self.mode == "enforce" and would_block,
            confidence=response["confidence"] if response else None,
            reason=reason,
        )


def _raise_queue_full() -> GuardResponse:
    raise RuntimeError("Background queue full, guard check dropped")


__all__ = ["Guard", "GuardDecision", "GuardTripped"]
