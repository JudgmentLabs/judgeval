"""Shared classification and transport facts; adapter smoke tests cover wiring."""

import asyncio
from types import SimpleNamespace

import pytest

from judgeval.integrations.mcp._observer import (
    McpObserver,
    McpOptions,
    _without_uri_secrets,
    request_headers,
    transport_session_id,
    _CallRecord,
    RequestFacts,
)
from judgeval.integrations.mcp._v1 import ModelResults
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider


@pytest.mark.asyncio
async def test_invalid_session_and_unicode_do_not_hide_native_cancellation(
    tracer, native_requests
):
    observer = McpObserver(
        McpOptions(inject_agent_arguments=False, session_id=lambda _: 123), "test", None
    )
    facts = RequestFacts(
        "tools/call",
        {"name": "tool\ud800", "arguments": {"key\ud800": 1}},
        None,
        None,
        {},
        {},
        None,
        None,
        None,
    )

    async def cancel(_):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await observer.observe(facts, cancel, ModelResults(), None)
    JudgmentTracerProvider.get_instance().force_flush()
    calls = [call for batch in native_requests for call in batch["calls"]]
    assert [
        (
            call["outcome"],
            call["tool_name"],
            call["argument_keys"],
            call.get("session_id"),
            call["metadata"].get("judgment.mcp.conversation_id_source"),
            call.get("error_type"),
        )
        for call in calls
    ] == [("cancelled", "tool�", ["key�"], None, None, None)]


class McpError(Exception):
    def __init__(self, code):
        self.error = SimpleNamespace(code=code)


@pytest.mark.parametrize(
    "error, message, expected",
    [
        (None, "Unknown tool: nope", "unknown_tool"),
        (None, "Tool nope disabled", "tool_disabled"),
        (McpError(-32602), "bad params", "invalid_input"),
        (None, "Input validation error: x", "invalid_input"),
        (None, '{"error":"rate_limited"}', "rate_limited"),
        (None, '{"error":"tok_1234"}', "tool_error"),
        (None, "Order not found", "tool_error"),
    ],
)
def test_failure_classification(error, message, expected):
    observer = McpObserver(
        McpOptions(capture_error_messages=True), server_name="test", server_version="1"
    )
    assert observer._error_type(error, True, message) == expected


def test_disabled_error_messages_do_not_record_message_derived_codes():
    observer = McpObserver(McpOptions(), server_name="test", server_version="1")
    assert observer._error_type(None, True, '{"error":"rate_limited"}') == "tool_error"


def failing_redactor(text):
    raise ValueError("redaction failed")


@pytest.mark.parametrize(
    "options, expected",
    [
        (McpOptions(), None),
        (McpOptions(capture_error_messages=True), "secret for a@b.co"),
        (
            McpOptions(capture_error_messages=True, redact=lambda _: "[redacted]"),
            "[redacted]",
        ),
        (McpOptions(capture_error_messages=True, redact=failing_redactor), None),
    ],
)
def test_error_text_is_opt_in_and_redacted_only_by_customer_callback(options, expected):
    observer = McpObserver(options, server_name="test", server_version="1")
    record = _CallRecord(error=RuntimeError("secret for a@b.co"))
    assert observer._failure(record, None, SimpleNamespace(), SimpleNamespace()) == (
        "RuntimeError",
        expected,
    )


def test_transport_facts_omit_credentials_and_normalize_headers():
    request = SimpleNamespace(headers={"Mcp-Session-Id": "transport-1"})
    headers = request_headers(request)
    assert (
        headers,
        transport_session_id(headers, request),
        _without_uri_secrets("https://user:password@host/tool?token=secret#private"),
    ) == (
        {"mcp-session-id": "transport-1"},
        "transport-1",
        "https://host/tool",
    )
