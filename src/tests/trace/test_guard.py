from __future__ import annotations

import threading
from unittest.mock import MagicMock

from judgeval.background_queue import flush
from judgeval.guard import Guard
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.trace.base_tracer import BaseTracer

VIOLATION = {"result": True, "confidence": 0.95, "reason": "external address"}


def test_enforce_tool_guard_refuses_confident_fail(tracer, collecting_exporter):
    tracer._client = MagicMock(**{"post_projects_guard.return_value": VIOLATION})
    send = MagicMock()

    @BaseTracer.observe(span_type="tool")
    @Guard("judge-1", mode="enforce").tool()
    def send_email(to: str) -> str:
        send(to)
        return "queued"

    assert send_email("x@evil.test") == "Blocked by guardrail: external address"
    send.assert_not_called()
    tracer._client.post_projects_guard.assert_called_once_with(
        "proj-123",
        {
            "judge_id": "judge-1",
            "position": "tool_input",
            "evidence": {"tool": "send_email", "input": {"to": "x@evil.test"}},
        },
    )
    guard_span = next(
        s for s in collecting_exporter.spans if s.name == "judgment.guard"
    )
    tool_span = next(s for s in collecting_exporter.spans if s.name == "send_email")
    assert guard_span.parent.span_id == tool_span.context.span_id
    assert guard_span.attributes[AttributeKeys.JUDGMENT_GUARD_OUTCOME] == "fail"
    assert guard_span.attributes[AttributeKeys.JUDGMENT_GUARD_WOULD_BLOCK] is True


def test_shadow_check_returns_immediately_and_never_raises(tracer, collecting_exporter):
    release = threading.Event()

    def slow_violation(*_):
        release.wait(5)
        return VIOLATION

    tracer._client = MagicMock(**{"post_projects_guard.side_effect": slow_violation})

    assert (
        Guard("judge-1", fail_mode="closed").check("input", {"message": "hi"}) is None
    )
    assert not any(s.name == "judgment.guard" for s in collecting_exporter.spans)

    release.set()
    assert flush(5000)
    guard_span = next(
        s for s in collecting_exporter.spans if s.name == "judgment.guard"
    )
    assert guard_span.attributes[AttributeKeys.JUDGMENT_GUARD_MODE] == "shadow"
    assert guard_span.attributes[AttributeKeys.JUDGMENT_GUARD_WOULD_BLOCK] is True
