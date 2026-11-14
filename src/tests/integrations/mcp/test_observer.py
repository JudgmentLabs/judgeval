"""Shared classification and transport facts; adapter smoke tests cover wiring."""

import asyncio
import json
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
from judgeval.integrations.mcp._v2 import WireResults
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider


@pytest.mark.asyncio
async def test_native_lookup_avoids_full_listing_and_keeps_structured_handoff(tracer):
    from mcp.types import CallToolResult

    lookups = []
    listings = []
    received = []

    async def lookup(name):
        lookups.append(name)
        return SimpleNamespace(
            parameters={"type": "object", "additionalProperties": False},
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )

    async def listing(cursor):
        listings.append(cursor)
        raise AssertionError("Call path must not list tools")

    async def invoke(arguments):
        received.append(arguments)
        return CallToolResult(content=[], structuredContent={"ok": True})

    observer = McpObserver(McpOptions(), "test", None, tool_lookup=lookup)
    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": {"intent": "Searching"}},
        None,
        None,
        {},
        {},
        None,
        None,
        None,
    )
    adapter_type = (
        WireResults if "is_error" in CallToolResult.model_fields else ModelResults
    )
    results = [
        await observer.observe(facts, invoke, adapter_type(), listing) for _ in range(2)
    ]
    assert (
        lookups,
        listings,
        received,
        [
            sorted(result.model_dump(by_alias=True)["structuredContent"])
            for result in results
        ],
    ) == (
        ["lookup", "lookup"],
        [],
        [{}, {}],
        [["_judgment", "ok"], ["_judgment", "ok"]],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("inject", [True, False])
async def test_explicit_arguments_record_business_values_without_changing_calls(
    inject, tracer
):
    from dataclasses import replace

    observer = McpObserver(
        McpOptions(
            inject_agent_arguments=inject,
            intent_argument="context",
            model_argument="llm_model",
            conversation_id_argument="chat_id",
            redact=lambda text: text.upper(),
        ),
        "test",
        None,
    )
    args = {
        "context": "find tickets",
        "llm_model": "agent-model",
        "chat_id": "customer-chat",
    }
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in args},
        "additionalProperties": False,
    }
    observer.remember_listing([("lookup", schema, None)])
    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": args},
        None,
        None,
        {},
        {},
        None,
        None,
        None,
    )
    prepared = await observer._prepare(facts, None)
    missing = await observer._prepare(
        replace(facts, params={"name": "lookup", "arguments": {}}), None
    )
    keys = (
        "judgment.mcp.intent",
        "judgment.llm.model",
        "judgment.session_id",
        "judgment.mcp.conversation_id_source",
    )
    assert {
        "telemetry": [prepared.attributes.get(key) for key in keys],
        "forwarded": prepared.arguments if prepared.arguments is not None else args,
        "minted": prepared.minted_conversation_id,
        "missing": [missing.attributes.get(key) for key in keys],
        "missing_minted": missing.minted_conversation_id,
    } == {
        "telemetry": ["FIND TICKETS", "agent-model", "customer-chat", "provided"],
        "forwarded": args,
        "minted": None,
        "missing": [None] * 4,
        "missing_minted": None,
    }


@pytest.mark.parametrize("style", ["model-v1", "wire-v2", "model-v2"])
def test_structured_only_handoff_preserves_output_without_content(style):
    from pydantic import create_model
    from judgeval.integrations.mcp._arguments import (
        ToolContract,
        conversation_id_prompt,
    )

    payload = {"ok": True}
    if style == "model-v1":
        adapter = ModelResults()
        model = create_model("StructuredV1Result", structuredContent=(dict, ...))
        original = model(structuredContent=payload)
    elif style == "wire-v2":
        adapter = WireResults()
        original = {"structuredContent": payload}
    else:
        adapter = WireResults()
        model = create_model("StructuredResult", structured_content=(dict, ...))
        original = model(structured_content=payload)
    prompt = conversation_id_prompt("id")
    result = adapter.with_text(original, prompt, ToolContract(frozenset(), False, True))
    if style == "wire-v2":
        structured, content = result["structuredContent"], result["content"]
    else:
        structured = getattr(
            result, "structuredContent" if style == "model-v1" else "structured_content"
        )
        content = [item.model_dump(exclude_none=True) for item in result.content]
    assert (structured, content, payload) == (
        {"ok": True, "_judgment": json.loads(prompt)},
        [{"type": "text", "text": prompt}],
        {"ok": True},
    )


@pytest.mark.asyncio
async def test_call_contracts_follow_current_caller_and_fail_open(tracer):
    from mcp.types import CallToolResult

    async def unavailable_tool(_):
        raise RuntimeError("lookup unavailable")

    observer = McpObserver(McpOptions(), "test", None, tool_lookup=unavailable_tool)
    forwarded = []
    adapter = (
        WireResults if "is_error" in CallToolResult.model_fields else ModelResults
    )()
    schema = {"type": "object", "additionalProperties": False}
    output = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": {"intent": "business"}},
        None,
        None,
        {},
        {},
        None,
        None,
        None,
    )

    async def invoke(arguments):
        forwarded.append(arguments)
        return CallToolResult.model_validate(
            {"content": [], "structuredContent": {"ok": True}}
        )

    async def first_caller(_):
        return [("lookup", schema, output)], None

    async def second_caller(_):
        return [
            (
                "lookup",
                {**schema, "properties": {"intent": {"type": "string"}}},
                {"type": "object", "properties": {"_judgment": {}}},
            )
        ], None

    async def unavailable(_):
        raise RuntimeError("listing unavailable")

    results = [
        await observer.observe(facts, invoke, adapter, listing)
        for listing in (first_caller, second_caller, unavailable)
    ]
    handoffs = [
        "_judgment" in result.model_dump(by_alias=True)["structuredContent"]
        for result in results
    ]
    assert (forwarded, handoffs) == (
        [{}, {"intent": "business"}, None],
        [True, False, False],
    )


@pytest.mark.asyncio
async def test_stateful_calls_offer_handoff_on_success_and_error_until_echo(
    tracer, collecting_exporter
):
    from dataclasses import replace
    from mcp.types import CallToolResult

    adapter_type = (
        WireResults if "is_error" in CallToolResult.model_fields else ModelResults
    )

    observer = McpObserver(McpOptions(), "test", None)
    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": {}},
        None,
        "transport",
        {},
        {},
        None,
        None,
        None,
    )

    async def listing(_):
        return [
            ("lookup", {"type": "object", "additionalProperties": False}, None)
        ], None

    async def succeed(_):
        return CallToolResult(content=[])

    error = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "declined"}], "isError": True}
    )

    async def fail(_):
        return error

    first = await observer.observe(facts, succeed, adapter_type(), listing)
    handle = (
        json.loads(first.content[-1].text)["conversation_id"] if first.content else None
    )
    second = await observer.observe(
        replace(
            facts, params={"name": "lookup", "arguments": {"conversation_id": handle}}
        ),
        succeed,
        adapter_type(),
        listing,
    )
    failed = await observer.observe(facts, fail, adapter_type(), listing)
    failed_handle = json.loads(failed.content[-1].text)["conversation_id"]
    await observer.observe(
        replace(
            facts,
            params={"name": "lookup", "arguments": {"conversation_id": failed_handle}},
        ),
        succeed,
        adapter_type(),
        listing,
    )
    tracer.force_flush()
    assert {
        "handoff": isinstance(handle, str),
        "echo_handoff": second.content,
        "error_preserved": failed.model_dump(by_alias=True)["isError"]
        and failed.content[:-1] == error.content,
        "original_content": len(error.content),
        "sessions": [
            (
                s.attributes.get("judgment.session_id"),
                s.attributes.get("judgment.mcp.conversation_id_source"),
            )
            for s in collecting_exporter.spans
        ],
    } == {
        "handoff": True,
        "echo_handoff": [],
        "error_preserved": True,
        "original_content": 1,
        "sessions": [
            (handle, "minted"),
            (handle, "echoed"),
            (failed_handle, "minted"),
            (failed_handle, "echoed"),
        ],
    }


@pytest.mark.asyncio
async def test_paginated_tools_remain_callable_after_twenty_pages(tracer):
    from mcp.types import CallToolResult

    observer = McpObserver(McpOptions(), "test", None)
    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": {"intent": "find tickets"}},
        None,
        None,
        {},
        {},
        None,
        None,
        None,
    )
    forwarded = []

    async def listing(cursor):
        page = int(cursor or 0)
        return [
            (
                "lookup" if page == 21 else str(page),
                {"type": "object", "additionalProperties": False},
                None,
            )
        ], (None if page == 21 else str(page + 1))

    async def invoke(arguments):
        forwarded.append(arguments)
        return CallToolResult(content=[])

    await observer.observe(facts, invoke, ModelResults(), listing)
    assert forwarded == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cached", "discovered", "pages", "timeout"])
async def test_lowlevel_ownership_reuse_and_bounded_discovery(mode, monkeypatch):
    from judgeval.integrations.mcp import _observer

    monkeypatch.setattr(_observer, "_DISCOVERY_TIMEOUT_SECONDS", 0.02)
    observer = McpObserver(McpOptions(), "test", None)
    schema = {
        "type": "object",
        "properties": {"intent": {"type": "string"}},
        "additionalProperties": False,
    }
    if mode == "cached":
        observer.shape_listing(
            {"tools": [{"name": "lookup", "inputSchema": schema}]}, WireResults()
        )
    pages = []
    received = []

    async def listing(cursor):
        pages.append(cursor)
        if mode == "cached":
            raise RuntimeError("listing unavailable")
        if mode == "timeout":
            await asyncio.Event().wait()
        if mode == "discovered":
            return [("lookup", schema, None)], None
        return [], str(len(pages))

    async def invoke(arguments):
        received.append(
            arguments if arguments is not None else facts.params["arguments"]
        )
        return {"isError": True}

    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": {"intent": "business", "llm_model": "model"}},
        None,
        None,
        {},
        {},
        None,
        None,
        None,
    )
    for _ in range(2):
        await observer.observe(facts, invoke, WireResults(), listing)
    assert (received, len(pages)) == (
        [
            {"intent": "business"}
            if mode in ("cached", "discovered")
            else facts.params["arguments"]
        ]
        * 2,
        {"cached": 0, "discovered": 1, "pages": 100, "timeout": 1}[mode],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field_style", ["wire", "model"])
async def test_resolver_tool_models_preserve_output_eligibility(field_style):
    from mcp.types import Tool

    descriptor = {
        "name": "lookup",
        "inputSchema": {"type": "object", "additionalProperties": False},
        "outputSchema": {"type": "object", "maxProperties": 1},
    }
    observer = McpObserver(
        McpOptions(
            resolve_original_tool=lambda _: Tool(**descriptor)
            if field_style == "model"
            else descriptor
        ),
        "test",
        None,
    )
    contract = await observer.contract_for("lookup", None)
    assert (contract.declared, contract.structured_output) == (frozenset(), False)


@pytest.mark.asyncio
async def test_discovery_cache_coalesces_bounds_and_expires_misses(monkeypatch):
    from judgeval.integrations.mcp import _observer

    monkeypatch.setattr(_observer, "_MAX_CACHED_TOOLS", 2)
    clock = [0.0]
    monkeypatch.setattr(_observer.time, "monotonic", lambda: clock[0])
    cache = _observer._ToolDiscoveryCache()
    lookups = []

    async def missing():
        lookups.append("lookup")
        await asyncio.sleep(0)
        return None

    await asyncio.gather(*(cache.resolve("missing", missing) for _ in range(10)))
    await cache.resolve("missing", missing)
    coalesced = len(lookups)
    clock[0] = 2.0
    await cache.resolve("missing", missing)
    expired = len(lookups)
    cache.remember("a", None)
    cache.remember("b", None)
    await cache.resolve("missing", missing)
    assert (coalesced, expired, len(lookups), len(cache._entries)) == (1, 2, 3, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["notify", "send_notification"])
async def test_notification_invalidates_inflight_and_advertised_contracts(method):
    from judgeval.integrations.mcp import _observer

    observer = McpObserver(McpOptions(), "test", None)
    cache = observer._listed_contracts
    started, finish = asyncio.Event(), asyncio.Event()
    contract = _observer.tool_contract(
        {"type": "object", "additionalProperties": False}, observer.argument_names
    )
    sent = []

    async def notify(**kwargs):
        sent.append(next(iter(kwargs.values())))

    async def lookup():
        started.set()
        await finish.wait()
        return contract

    target = SimpleNamespace(**{method: notify})
    observer.watch_tool_changes(target)
    observer.watch_tool_changes(target)
    cache.remember("listed", contract)
    pending = asyncio.create_task(cache.resolve("cold", lookup))
    await started.wait()
    notification = "notifications/tools/list_changed"
    value = (
        SimpleNamespace(root=SimpleNamespace(method=notification))
        if method == "send_notification"
        else notification
    )
    await getattr(target, method)(
        **{"notification" if method == "send_notification" else "method": value}
    )
    finish.set()
    assert (await pending, list(cache._entries), sent) == (
        None,
        [],
        [value],
    )


@pytest.mark.asyncio
async def test_invalid_session_does_not_hide_cancellation(tracer, collecting_exporter):
    observer = McpObserver(
        McpOptions(inject_agent_arguments=False, session_id=lambda _: 123), "test", None
    )
    facts = RequestFacts(
        "tools/call",
        {"name": "tool", "arguments": {"key": 1}},
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
    calls = [span.attributes for span in collecting_exporter.spans]
    assert [
        (
            call["judgment.mcp.outcome"],
            call["gen_ai.tool.name"],
            call["judgment.mcp.argument_keys"],
            call.get("judgment.session_id"),
            call.get("judgment.mcp.conversation_id_source"),
            call.get("error.type"),
        )
        for call in calls
    ] == [("cancelled", "tool", '["key"]', None, None, None)]


class McpError(Exception):
    def __init__(self, code):
        self.error = SimpleNamespace(code=code)


@pytest.mark.asyncio
async def test_transport_session_survives_failed_outcome_detection(
    tracer, collecting_exporter
):
    class BrokenResults(ModelResults):
        def is_error(self, result):
            raise RuntimeError("classification unavailable")

    from mcp.types import CallToolResult

    observer = McpObserver(McpOptions(inject_conversation_id=False), "test", None)
    facts = RequestFacts(
        "tools/call",
        {"name": "lookup", "arguments": {}},
        None,
        "transport-session",
        {},
        {},
        None,
        None,
        None,
    )

    async def listing(_):
        return [
            ("lookup", {"type": "object", "additionalProperties": False}, None)
        ], None

    original = CallToolResult(content=[], isError=True)

    async def invoke(_):
        return original

    result = await observer.observe(facts, invoke, BrokenResults(), listing)
    JudgmentTracerProvider.get_instance().force_flush()
    call = collecting_exporter.spans[-1]
    assert (
        result is original,
        call.attributes["judgment.session_id"],
        call.attributes["judgment.mcp.conversation_id_source"],
        call.attributes["judgment.mcp.outcome"],
        call.status.status_code.name,
    ) == (True, "transport-session", "transport", "error", "ERROR")


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
