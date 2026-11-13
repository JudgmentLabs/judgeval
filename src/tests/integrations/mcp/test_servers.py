import json

import pytest

from judgeval.integrations.mcp import instrument_mcp_server
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
from tests.integrations.mcp.harness import MCP_MAJOR, build_server, connect, styles


@pytest.mark.asyncio
@pytest.mark.skipif(MCP_MAJOR != 1, reason="MCP 1.x adapter")
async def test_lowlevel_discovery_preserves_metadata_without_progress_token(tracer):
    from mcp import types

    orders = build_server("mcp1-lowlevel")
    seen = []
    original = orders.server.request_handlers[types.ListToolsRequest]

    async def listing(request):
        if request is not None:
            seen.append(
                request.params.meta.model_dump(by_alias=True, exclude_none=True)
            )
        return await original(request)

    orders.server.request_handlers[types.ListToolsRequest] = listing
    instrument_mcp_server(orders.server)
    async with connect(orders) as client:
        await client.send(
            "tools/call",
            {
                "name": "add",
                "arguments": {"a": 1, "b": 2},
                "_meta": {"tenant": "caller", "progressToken": "progress"},
            },
        )
    assert (seen, orders.calls) == ([{"tenant": "caller"}], [{"a": 1, "b": 2}])


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_original_tool_resolver_on_lowlevel_and_native_lookup_priority(
    style, tracer
):
    from tests.integrations.mcp.harness import LOW_LEVEL

    orders = build_server(style)
    resolved = []

    async def resolve(name):
        resolved.append(name)
        return {
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            }
        }

    instrument_mcp_server(orders.server, resolve_original_tool=resolve)
    async with connect(orders) as client:
        result = await client.call("add", {"a": 1, "b": 2, "intent": "Adding"})
    assert (orders.calls, resolved, result.texts[0], bool(result.conversation_id)) == (
        [{"a": 1, "b": 2}],
        ["add"] if style in LOW_LEVEL else [],
        "3",
        True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_reinstrumentation_warns_on_changed_options_and_keeps_original(
    style, tracer, collecting_exporter, monkeypatch
):
    from judgeval.logger import judgeval_logger

    orders = build_server(style)
    warnings = []
    monkeypatch.setattr(judgeval_logger, "warning", warnings.append)
    results = [
        instrument_mcp_server(orders.server),
        instrument_mcp_server(orders.server),
        instrument_mcp_server(orders.server, capture_input=True),
    ]
    async with connect(orders) as client:
        await client.call("add", {"a": 1, "b": 2})
    tracer.force_flush()
    assert {
        "results": results,
        "warnings": warnings,
        "captured": [
            "judgment.input" in span.attributes
            for span in collecting_exporter.spans
            if span.name == "tools/call add"
        ],
    } == {
        "results": [True, True, True],
        "warnings": [
            "MCP server is already instrumented with different options; keeping the original configuration."
        ],
        "captured": [False],
    }


@pytest.mark.asyncio
async def test_fastmcp_dictionary_output_preserves_client_data(tracer):
    fastmcp = pytest.importorskip("fastmcp")
    server = fastmcp.FastMCP("dictionary")

    @server.tool()
    def counts() -> dict[str, int]:
        return {"open": 3, "closed": 2}

    instrument_mcp_server(server)
    async with fastmcp.Client(server) as client:
        tools = await client.list_tools()
        result = await client.call_tool("counts", {"intent": "Counting"})
    assert {
        "data": result.data,
        "structured": result.structured_content,
        "output_schema": tools[0].outputSchema,
        "handoff": any(
            '"conversation_id"' in item.text
            for item in result.content
            if item.type == "text"
        ),
    } == {
        "data": {"open": 3, "closed": 2},
        "structured": {"open": 3, "closed": 2},
        "output_schema": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
        "handoff": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_custom_injected_names_are_stripped_and_conversation_handoff_uses_alias(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(
        orders.server,
        intent_argument="goal",
        model_argument="agent_model",
        conversation_id_argument="chat_id",
    )
    async with connect(orders) as client:
        listing = (await client.all_tools())["add"]
        first = await client.call(
            "add", {"a": 1, "b": 2, "goal": "Adding", "agent_model": "agent-model"}
        )
        handle = first.conversation_id
        second = await client.call("add", {"a": 2, "b": 3, "chat_id": handle})
    tracer.force_flush()
    calls = [
        span.attributes
        for span in collecting_exporter.spans
        if span.attributes.get("mcp.method.name") == "tools/call"
    ]
    assert {
        "fields": sorted(listing["properties"]),
        "received": orders.calls,
        "intent": calls[0].get("judgment.mcp.intent"),
        "model": calls[0].get("judgment.llm.model"),
        "continuity": bool(handle)
        and calls[0].get("judgment.session_id") == calls[1].get("judgment.session_id"),
        "note": any("Pass this chat_id" in text for text in first.texts),
        "second_id": second.conversation_id,
    } == {
        "fields": ["a", "agent_model", "b", "chat_id", "goal"],
        "received": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
        "intent": "Adding",
        "model": "agent-model",
        "continuity": True,
        "note": True,
        "second_id": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_session_handoff_records_calls_as_separate_traces(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)
    async with connect(orders) as client:
        await client.list_tools()
        first = await client.call(
            "add", {"a": 1, "b": 2, "intent": "Adding", "llm_model": "gpt-6"}
        )
        session = (
            first.structured["_judgment"]["conversation_id"]
            if first.structured
            else first.conversation_id
        )
        second = await client.call("add", {"a": 2, "b": 3, "conversation_id": session})
    JudgmentTracerProvider.get_instance().force_flush()
    calls = [
        span
        for span in collecting_exporter.spans
        if span.attributes.get("mcp.method.name") == "tools/call"
    ]
    assert {
        "protocol_spans": [
            span.name
            for span in collecting_exporter.spans
            if span.attributes.get("mcp.method.name") not in (None, "tools/call")
        ],
        "results": (first.texts[0], second.texts),
        "received": orders.calls,
        "session": bool(session),
        "handoff": json.loads(first.texts[-1])["conversation_id"],
        "sessions": [
            (
                call.attributes["judgment.session_id"],
                call.attributes["judgment.mcp.conversation_id_source"],
            )
            for call in calls
        ],
        "roots": [span.parent is None for span in calls],
        "distinct_traces": calls[0].context.trace_id != calls[1].context.trace_id,
        "children": [
            (
                child.context.trace_id == call.context.trace_id,
                child.parent.span_id == call.context.span_id,
            )
            for child, call in zip(
                [span for span in collecting_exporter.spans if span.name == "database"],
                calls,
            )
        ],
        "reported": (
            calls[0].attributes["judgment.mcp.intent"],
            calls[0].attributes["judgment.llm.model"],
        ),
    } == {
        "protocol_spans": [],
        "results": ("3", ["5"]),
        "received": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
        "session": True,
        "handoff": session,
        "sessions": [(session, "minted"), (session, "echoed")],
        "roots": [True, True],
        "distinct_traces": True,
        "children": [(True, True), (True, True)],
        "reported": ("Adding", "gpt-6"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_callbacks_fail_open_and_instrumentation_is_idempotent(style, tracer):
    orders = build_server(style)

    def fail(_):
        raise RuntimeError("callback unavailable")

    first = instrument_mcp_server(orders.server, identify=fail, redact=fail)
    second = instrument_mcp_server(orders.server)
    async with connect(orders) as client:
        result = await client.call("add", {"a": 1, "b": 2, "intent": "Adding"})
    assert (first, second, result.texts[0], orders.calls) == (
        True,
        True,
        "3",
        [{"a": 1, "b": 2}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_explicit_parent_and_application_session_with_conversation_injection_off(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(
        orders.server,
        inject_conversation_id=False,
        session_id=lambda _: "application-session",
        capture_input=True,
        capture_output=True,
        redact_payload=lambda value, field: {"captured": field},
    )
    provider = JudgmentTracerProvider.get_instance()
    with provider.get_tracer("server").start_as_current_span("ambient"):
        async with connect(orders) as client:
            listing = (await client.all_tools())["add"]
            first = await client.call("add", {"a": 1, "b": 2, "intent": "Adding"})
            await client.send(
                "tools/call",
                {
                    "name": "add",
                    "arguments": {"a": 2, "b": 3},
                    "_meta": {"traceparent": f"00-{'a' * 32}-{'b' * 16}-01"},
                },
            )
    tracer.force_flush()
    calls = [
        span
        for span in collecting_exporter.spans
        if span.attributes.get("mcp.method.name") == "tools/call"
    ]
    assert {
        "arguments": sorted(listing["properties"]),
        "handoff": first.conversation_id,
        "received": orders.calls,
        "root": calls[0].parent,
        "remote": (calls[1].context.trace_id, calls[1].parent.span_id),
        "sessions": [span.attributes["judgment.session_id"] for span in calls],
        "payloads": [
            (
                json.loads(span.attributes["judgment.input"]),
                json.loads(span.attributes["judgment.output"]),
            )
            for span in calls
        ],
    } == {
        "arguments": ["a", "b", "intent", "llm_model"],
        "handoff": None,
        "received": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
        "root": None,
        "remote": (int("a" * 32, 16), int("b" * 16, 16)),
        "sessions": ["application-session", "application-session"],
        "payloads": [({"captured": "input"}, {"captured": "output"})] * 2,
    }
