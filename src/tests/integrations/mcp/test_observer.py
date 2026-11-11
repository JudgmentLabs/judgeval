"""How a request's facts become span attributes, independent of `mcp` itself."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from judgeval.integrations.mcp import mark_mcp_call_error, set_mcp_caller
from judgeval.integrations.mcp._observer import (
    _INVALID_INPUT,
    _UNKNOWN_TOOL,
    McpObserver,
    McpOptions,
    _recorded_keys,
    _without_uri_secrets,
    attempt,
    request_headers,
    transport_session_id,
)


def _observer(**options: Any) -> McpObserver:
    return McpObserver(
        McpOptions(record_spans=True, **options),
        server_name="orders",
        server_version="1",
    )


class OrderNotFound(Exception):
    pass


class ToolError(Exception):
    pass


class ValidationError(Exception):
    pass


class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.error = SimpleNamespace(code=code, message=message)


@pytest.mark.parametrize(
    "message, disabled",
    [
        ("Unknown tool: nope", False),
        ("Unknown tool: 'nope'", False),
        ("Tool nope not found", False),
        ("Tool 'nope' not found", False),
        ("MCP error -32602: Tool nope not found", False),
        ("Tool nope disabled", True),
        ("MCP error -32602: Tool nope disabled", True),
    ],
)
def test_unknown_tool_wordings(message, disabled):
    match = _UNKNOWN_TOOL.search(message)
    assert match is not None
    assert (match.group("state") == "disabled") is disabled


@pytest.mark.parametrize(
    "message",
    ["Order not found", "Unknown error", "The tool nope was not found in stock"],
)
def test_other_failures_are_not_unknown_tools(message):
    assert _UNKNOWN_TOOL.search(message) is None


@pytest.mark.parametrize(
    "message",
    [
        "Input validation error: 'x' is not of type 'integer'",
        "MCP error -32602: Input validation error: Invalid arguments for tool add",
        "Error executing tool add: 1 validation error for addArguments\na\n  bad",
        "Error executing tool add: 2 validation errors for addArguments",
        "1 validation error for call[add]\na\n  Input should be a valid integer",
    ],
)
def test_invalid_input_wordings(message):
    assert _INVALID_INPUT.search(message) is not None


def test_a_tools_own_validation_failure_is_not_invalid_input():
    message = "Error executing tool lookup: 1 validation error for Order\nid\n  bad"
    assert _INVALID_INPUT.search(message) is None


@pytest.mark.parametrize(
    "error, has_result, message, expected",
    [
        (None, True, "Unknown tool: nope", "unknown_tool"),
        (None, True, "Tool nope disabled", "tool_disabled"),
        (
            McpError(-32602, "Unknown tool: nope"),
            False,
            "Unknown tool: nope",
            "unknown_tool",
        ),
        (McpError(-32602, "bad params"), False, "bad params", "invalid_input"),
        (None, True, "Input validation error: x", "invalid_input"),
        (ValidationError("bad"), False, "bad", "invalid_input"),
        (OrderNotFound("o-1"), True, "order o-1 not found", "OrderNotFound"),
        (OrderNotFound("o-1"), False, "order o-1 not found", "OrderNotFound"),
        (Exception("boom"), True, "boom", "tool_error"),
        (ToolError("x"), True, '{"error":"rate_limited"}', "rate_limited"),
        (None, True, '{"error":"payment.declined","detail":"x"}', "payment.declined"),
        # Codes with digits could be tokens, so they're not recorded.
        (None, True, '{"error":"tok_1234"}', "tool_error"),
        (None, True, '{"error":5}', "tool_error"),
        (None, True, '{"error":"rate_limited"', "tool_error"),
        (None, True, "[1, 2]", "tool_error"),
        (None, True, None, "tool_error"),
        (McpError(-32603, "internal"), False, "internal", "-32603"),
        (None, False, None, "_OTHER"),
    ],
)
def test_error_type(error, has_result, message, expected):
    assert _observer()._error_type(error, has_result, message) == expected


def test_error_type_without_messages_ignores_returned_codes():
    observer = _observer(capture_error_messages=False)
    assert observer._error_type(None, True, '{"error":"rate_limited"}') == "tool_error"


def test_error_type_redacts_returned_codes():
    observer = _observer(redact=lambda text: text.replace("acme", "x"))
    assert observer._error_type(None, True, '{"error":"acme_down"}') == "x_down"
    blank = _observer(redact=lambda text: "")
    assert blank._error_type(None, True, '{"error":"acme_down"}') == "tool_error"


@pytest.mark.parametrize(
    "uri, expected",
    [
        (
            "https://user:pw@api.example.com/v1/orders?token=abc#top",
            "https://api.example.com/v1/orders",
        ),
        ("postgres://admin:secret@db:5432/orders", "postgres://db:5432/orders"),
        ("orders://o-1?signature=xyz", "orders://o-1"),
        ("file:///var/data/report.csv", "file:///var/data/report.csv"),
        ("s3://bucket/key.json", "s3://bucket/key.json"),
        ("note#private", "note"),
    ],
)
def test_resource_uris_lose_credentials(uri, expected):
    assert _without_uri_secrets(uri) == expected


class TestRecordedKeys:
    def test_leaves_out_injected_names_the_tool_does_not_declare(self):
        arguments = {"a": 1, "context": "c", "llm_model": "m", "conversation_id": "x"}
        assert _recorded_keys(arguments, frozenset({"a"})) == "a"
        assert _recorded_keys(arguments, frozenset({"a", "context"})) == "a,context"

    def test_records_every_name_when_nothing_was_injected(self):
        arguments = {"a": 1, "context": "c"}
        assert _recorded_keys(arguments, None) == "a,context"

    def test_is_capped(self):
        arguments = {f"{'k' * 100}{index}": index for index in range(40)}
        keys = _recorded_keys(arguments, None)
        assert keys is not None
        recorded = keys.split(",")
        assert len(recorded) == 30
        assert all(len(key) == 64 for key in recorded)

    def test_empty(self):
        assert _recorded_keys({}, None) is None


class TestTransportFacts:
    def test_headers_are_lowercased(self):
        request = SimpleNamespace(headers={"Mcp-Session-Id": "s-1", "X-User": "ann"})
        assert request_headers(request) == {"mcp-session-id": "s-1", "x-user": "ann"}

    def test_unreadable_headers_are_empty(self):
        class Broken:
            def items(self) -> Any:
                raise RuntimeError("closed")

        assert request_headers(SimpleNamespace(headers=Broken())) == {}
        assert request_headers(None) == {}

    def test_session_from_streamable_http_header(self):
        assert transport_session_id({"mcp-session-id": "s-1"}, None) == "s-1"

    def test_session_from_sse_query(self):
        request = SimpleNamespace(query_params={"session_id": "abc"})
        assert transport_session_id({}, request) == "abc"

    @pytest.mark.parametrize(
        "request_object", [None, SimpleNamespace(query_params={}), SimpleNamespace()]
    )
    def test_no_session(self, request_object: Optional[Any]):
        assert transport_session_id({}, request_object) is None


def test_helpers_outside_a_call_do_nothing():
    mark_mcp_call_error(OrderNotFound("o-1"))
    set_mcp_caller(user_id="u-1", customer_id="c-1")


def test_attempt_swallows_failures():
    assert attempt(lambda: 1 / 0) is None
    assert attempt(lambda: 7) == 7
