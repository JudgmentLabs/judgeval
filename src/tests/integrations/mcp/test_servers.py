"""End-to-end tests against real servers of each supported style, in memory."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import anyio
import pytest

pytest.importorskip("mcp")

from judgeval.integrations.mcp import (  # noqa: E402
    McpCaller,
    McpRequestInfo,
    instrument_mcp_server,
)
from judgeval.integrations.mcp import _observer  # noqa: E402
from judgeval.version import get_version  # noqa: E402
from tests.integrations.mcp.harness import (  # noqa: E402
    AGENT_ARGUMENTS,
    HIGH_LEVEL,
    LOW_LEVEL,
    LOWLEVEL_VERSION,
    MCP_MAJOR,
    MCP_RELEASE,
    SERVER_NAME,
    attributes,
    build_server,
    connect,
    expected_client_name,
    lowlevel_validates_input,
    only_span,
    spans,
    styles,
)

UUID_V7 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


def _schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    return schema.get("properties") or {}


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_listing_adds_optional_agent_arguments(style, tracer):
    orders = build_server(style)
    original = None
    if style in LOW_LEVEL:
        async with connect(build_server(style)) as client:
            original = await client.all_tools()
    assert instrument_mcp_server(orders.server) is True

    async with connect(orders) as client:
        tools = await client.all_tools()

    add = tools["add"]
    for name in AGENT_ARGUMENTS:
        assert _schema_properties(add)[name]["type"] == "string"
        assert "analytics" in _schema_properties(add)[name]["description"]
        assert name not in (add.get("required") or [])
    assert {"a", "b"} <= set(_schema_properties(add))
    # The tool's own argument keeps its definition.
    own = _schema_properties(tools["own_context"])["context"]
    assert "analytics" not in own.get("description", "")
    if original is not None:
        # Schemas that accept other keys, or that compose subschemas, are left alone.
        assert tools["open"] == original["open"]
        assert tools["any_of"] == original["any_of"]
        assert add.get("additionalProperties") is False
        assert add["required"] == original["add"]["required"]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_call_records_reported_arguments_and_hides_them_from_the_tool(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server, server_build="build-42")

    async with connect(orders) as client:
        await client.list_tools()
        outcome = await client.call(
            "add",
            {
                "a": 1,
                "b": 2,
                "context": "Checking the order total before refunding the customer",
                "llm_model": "claude-sonnet-5",
            },
        )

    assert outcome.is_error is False
    assert outcome.texts[0] == "3"
    assert orders.received("add") == [{"a": 1, "b": 2}]

    span = only_span(collecting_exporter, "tools/call add")
    assert span.kind.name == "SERVER"
    assert span.status.status_code.name == "UNSET"
    found = attributes(span)
    assert found["mcp.method.name"] == "tools/call"
    assert found["gen_ai.operation.name"] == "execute_tool"
    assert found["gen_ai.tool.name"] == "add"
    assert found["judgment.mcp.argument_keys"] == "a,b"
    assert (
        found["judgment.mcp.intent"]
        == "Checking the order total before refunding the customer"
    )
    assert found["judgment.mcp.model"] == "claude-sonnet-5"
    assert found["judgment.mcp.server.name"] == SERVER_NAME
    assert found["judgment.mcp.server.build"] == "build-42"
    assert found["judgment.mcp.client.name"] == expected_client_name()
    assert found["judgment.mcp.sdk_version"] == get_version()
    assert found["jsonrpc.request.id"]
    assert found["mcp.protocol.version"]
    assert "mcp.session.id" not in found
    assert "error.type" not in found
    if style in LOW_LEVEL:
        assert found["judgment.mcp.server.version"] == LOWLEVEL_VERSION
    # Values the call carried never reach the span.
    assert "1" not in found["judgment.mcp.argument_keys"]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_conversation_id_is_minted_then_echoed(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        first = await client.call("add", {"a": 1, "b": 2})
        conversation_id = first.conversation_id
        assert conversation_id is not None and UUID_V7.match(conversation_id)
        # The prompt is appended after the tool's own content.
        assert first.texts[0] == "3"
        assert first.texts[-1].startswith('{"conversation_id"')
        second = await client.call(
            "add", {"a": 2, "b": 3, "conversation_id": conversation_id}
        )

    assert second.texts == ["5"]
    assert second.conversation_id is None
    if first.structured is not None:
        # Structured output stays exactly what the tool's output schema says.
        assert first.structured == {"result": 3}
    assert orders.received("add") == [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
    minted, echoed = [
        attributes(span) for span in spans(collecting_exporter, "tools/call add")
    ]
    assert minted["judgment.mcp.conversation_id"] == conversation_id
    assert minted["judgment.mcp.conversation_id_source"] == "minted"
    assert echoed["judgment.mcp.conversation_id"] == conversation_id
    assert echoed["judgment.mcp.conversation_id_source"] == "echoed"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
@pytest.mark.parametrize(
    "reported",
    [
        "not-a-conversation",
        # A UUID, but version 4: only IDs this integration minted are accepted.
        "4c7b1c4e-3f4c-4b7e-9f0a-2d5e8c1a7b3d",
        "",
    ],
)
async def test_unrecognized_conversation_id_is_replaced(
    style, reported, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        outcome = await client.call(
            "add", {"a": 1, "b": 2, "conversation_id": reported}
        )

    assert outcome.is_error is False
    assert orders.received("add") == [{"a": 1, "b": 2}]
    found = attributes(only_span(collecting_exporter, "tools/call add"))
    assert found["judgment.mcp.conversation_id"] == outcome.conversation_id
    assert found["judgment.mcp.conversation_id"] != reported
    assert found["judgment.mcp.conversation_id_source"] == "minted"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_unknown_model_and_personal_details_are_not_recorded(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        await client.call(
            "add",
            {
                "a": 1,
                "b": 2,
                "llm_model": "Unknown",
                "context": "Refund for jane.doe@example.com, card 4242 4242 4242 4242, "
                "SSN 123-45-6789, from 10.0.0.12, phone +1 (415) 555-0100",
            },
        )

    found = attributes(only_span(collecting_exporter, "tools/call add"))
    assert "judgment.mcp.model" not in found
    assert found["judgment.mcp.intent"] == (
        "Refund for [email], card [card], SSN [ssn], from [ip], phone [phone]"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_tool_argument_named_like_an_agent_argument_is_the_tools(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        outcome = await client.call(
            "own_context", {"context": "warehouse 7", "llm_model": "gpt-6"}
        )

    assert outcome.texts[0] == "context=warehouse 7"
    assert orders.received("own_context") == [{"context": "warehouse 7"}]
    found = attributes(only_span(collecting_exporter, "tools/call own_context"))
    assert "judgment.mcp.intent" not in found
    assert found["judgment.mcp.model"] == "gpt-6"
    assert found["judgment.mcp.argument_keys"] == "context"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_tool_owned_conversation_id_is_neither_minted_nor_read(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        tools = await client.all_tools()
        outcome = await client.call(
            "own_conversation",
            {"conversation_id": "thread-7", "context": "Reading an earlier thread"},
        )

    listed = _schema_properties(tools["own_conversation"])
    assert "analytics" not in listed["conversation_id"].get("description", "")
    assert "analytics" in listed["context"]["description"]
    # No conversation ID is appended for the agent to put in the tool's argument.
    assert outcome.texts == ["thread=thread-7"]
    assert orders.received("own_conversation") == [{"conversation_id": "thread-7"}]
    found = attributes(only_span(collecting_exporter, "tools/call own_conversation"))
    assert found["judgment.mcp.intent"] == "Reading an earlier thread"
    assert "judgment.mcp.conversation_id" not in found
    assert found["judgment.mcp.argument_keys"] == "conversation_id"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles(*sorted(LOW_LEVEL)))
async def test_names_a_schema_requires_or_matches_are_the_tools(
    style, tracer, collecting_exporter
):
    names = ("required_context", "pattern_context")
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        tools = await client.all_tools()
        outcomes = [
            await client.call(name, {"context": "warehouse 7", "llm_model": "gpt-6"})
            for name in names
        ]

    for name, outcome in zip(names, outcomes):
        listed = _schema_properties(tools[name])
        assert "context" not in listed
        assert {"llm_model", "conversation_id"} <= set(listed)
        assert outcome.texts[0] == "context=warehouse 7"
        assert orders.received(name) == [{"context": "warehouse 7"}]
        found = attributes(only_span(collecting_exporter, f"tools/call {name}"))
        assert "judgment.mcp.intent" not in found
        assert found["judgment.mcp.model"] == "gpt-6"
        assert found["judgment.mcp.argument_keys"] == "context"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_call_before_listing_still_hides_agent_arguments(
    style, tracer, collecting_exporter
):
    """A call can reach a server that never listed its tools, like a replica."""
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        outcome = await client.call(
            "add", {"a": 4, "b": 5, "context": "Adding before any listing"}
        )

    assert outcome.is_error is False, outcome
    assert outcome.texts[0] == "9"
    assert orders.received("add") == [{"a": 4, "b": 5}]
    found = attributes(only_span(collecting_exporter, "tools/call add"))
    assert found["judgment.mcp.intent"] == "Adding before any listing"
    # The listing used to learn the tool's schema isn't a request of its own.
    assert spans(collecting_exporter, "tools/list") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles(*sorted(HIGH_LEVEL)))
async def test_tools_added_after_instrumenting_are_traced(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    @orders.server.tool()
    def ship(order_id: str) -> str:
        orders.calls.append(("ship", {"order_id": order_id}))
        return f"shipped {order_id}"

    async with connect(orders) as client:
        tools = await client.all_tools()
        outcome = await client.call(
            "ship", {"order_id": "o-1", "context": "Shipping a paid order"}
        )

    assert "context" in _schema_properties(tools["ship"])
    assert outcome.texts[0] == "shipped o-1"
    assert orders.received("ship") == [{"order_id": "o-1"}]
    found = attributes(only_span(collecting_exporter, "tools/call ship"))
    assert found["judgment.mcp.intent"] == "Shipping a paid order"


def _expected_error_type(style: str, case: str) -> str:
    if case == "lookup":
        # Only a 2.x low-level server lets the exception itself through; the
        # others turn it into an error result before tracing sees it.
        return "OrderNotFound" if style == "mcp2-lowlevel" else "tool_error"
    return {
        "lookup_marked": "OrderNotFound",
        "unknown": "unknown_tool",
        "invalid": "invalid_input",
        "coded": "rate_limited",
    }[case]


ERROR_CASES = {
    # The order ID is the call's own argument, so it is left out.
    "lookup": ("lookup", {"order_id": "o-404"}, "order [argument] not found"),
    "lookup_marked": (
        "lookup_marked",
        {"order_id": "o-404"},
        "order [argument] not found",
    ),
    "unknown": ("nope", {}, "nope"),
    "invalid": ("add", {"a": "one", "b": 2}, None),
    "coded": ("coded", {}, "rate_limited"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
@pytest.mark.parametrize("case", sorted(ERROR_CASES))
async def test_failed_calls_record_why(style, case, tracer, collecting_exporter):
    if case == "coded" and style in HIGH_LEVEL:
        pytest.skip("only a low-level tool returns its own error result")
    if (
        case == "invalid"
        and style == "mcp1-lowlevel"
        and not lowlevel_validates_input()
    ):
        pytest.skip("this `mcp` doesn't check arguments against the schema")
    tool, arguments, message = ERROR_CASES[case]
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        outcome = await client.call(tool, arguments)
        # The server keeps working after a failure.
        after = await client.call("add", {"a": 1, "b": 1})

    assert outcome.is_error is not False
    assert after.texts[0] == "2"
    span = spans(collecting_exporter, f"tools/call {tool}")[0]
    found = attributes(span)
    assert found["error.type"] == _expected_error_type(style, case), (
        found,
        span.status.description,
    )
    assert span.status.status_code.name == "ERROR"
    if style == "mcp2-mcpserver" and case.startswith("lookup"):
        # MCPServer doesn't put an unexpected exception's text in the result.
        message = f"Error executing tool {tool}"
    if message is not None:
        assert message in (span.status.description or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_error_messages_leave_out_argument_values_and_identifiers(
    style, tracer, collecting_exporter
):
    redacted: List[str] = []

    def redact(text: str) -> str:
        redacted.append(text)
        return text

    orders = build_server(style)
    instrument_mcp_server(orders.server, redact=redact)

    async with connect(orders) as client:
        await client.list_tools()
        await client.call(
            "add", {"a": "sk-live-4f9a8b7c", "b": {"note": "it's\nnested-secret"}}
        )
        await client.call("lookup", {"order_id": "o-77 for ana@example.com"})
        await client.call("owner", {"order_id": "o-1"})

    failed = [
        span
        for span in spans(collecting_exporter)
        if span.status.status_code.name == "ERROR"
    ]
    assert len(failed) == 3
    recorded = [span.status.description or "" for span in failed]
    for text in recorded + redacted:
        for secret in ("sk-live", "nested-secret", "o-77", "ana@", "10.0.0.7"):
            assert secret not in text, (secret, text)
    # The built-in scrubbing runs before the builder's own redaction.
    assert len(redacted) == 3
    lookup = only_span(collecting_exporter, "tools/call lookup")
    owner = only_span(collecting_exporter, "tools/call owner")
    assert attributes(lookup)["error.type"] == _expected_error_type(style, "lookup")
    assert attributes(owner)["error.type"] == _expected_error_type(style, "lookup")
    if style != "mcp2-mcpserver":
        assert "order [argument] not found" in (lookup.status.description or "")
        # Dates and times stay; short arguments such as "o-1" are kept.
        assert (
            "order o-1 belongs to [email], last seen from [ip] at 2026-11-08 01:16:19"
            in (owner.status.description or "")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_error_messages_can_be_left_out(style, tracer, collecting_exporter):
    orders = build_server(style)
    instrument_mcp_server(orders.server, capture_error_messages=False)

    async with connect(orders) as client:
        await client.list_tools()
        await client.call("lookup_marked", {"order_id": "o-404"})
        if style in LOW_LEVEL:
            await client.call("coded", {})

    span = only_span(collecting_exporter, "tools/call lookup_marked")
    assert attributes(span)["error.type"] == "OrderNotFound"
    assert span.status.status_code.name == "ERROR"
    assert not span.status.description
    if style in LOW_LEVEL:
        # A code read from the message is part of the message.
        coded = only_span(collecting_exporter, "tools/call coded")
        assert attributes(coded)["error.type"] == "tool_error"
        assert not coded.status.description


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_redact_rewrites_messages_and_intent(style, tracer, collecting_exporter):
    orders = build_server(style)
    instrument_mcp_server(
        orders.server, redact=lambda text: text.replace("o-404", "[order]")
    )

    async with connect(orders) as client:
        await client.list_tools()
        await client.call("refund-o-404", {})
        await client.call(
            "add", {"a": 1, "b": 2, "context": "Totals for order o-404 today"}
        )

    failed = only_span(collecting_exporter, "tools/call refund-o-404")
    assert attributes(failed)["error.type"] == "unknown_tool"
    assert "o-404" not in (failed.status.description or "")
    assert "refund-[order]" in (failed.status.description or "")
    added = attributes(only_span(collecting_exporter, "tools/call add"))
    assert added["judgment.mcp.intent"] == "Totals for order [order] today"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_failing_redact_drops_the_text_but_not_the_call(
    style, tracer, collecting_exporter
):
    def redact(text: str) -> str:
        raise RuntimeError("redaction service down")

    orders = build_server(style)
    instrument_mcp_server(orders.server, redact=redact)

    async with connect(orders) as client:
        await client.list_tools()
        failed = await client.call("lookup_marked", {"order_id": "o-404"})
        added = await client.call("add", {"a": 1, "b": 2, "context": "Adding"})

    assert failed.is_error is not False
    assert added.texts[0] == "3"
    failed_span = only_span(collecting_exporter, "tools/call lookup_marked")
    assert attributes(failed_span)["error.type"] == "OrderNotFound"
    assert not failed_span.status.description
    assert "judgment.mcp.intent" not in attributes(
        only_span(collecting_exporter, "tools/call add")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_identify_and_set_mcp_caller(style, tracer, collecting_exporter):
    seen = []

    def identify(info: McpRequestInfo) -> McpCaller:
        seen.append(info)
        return McpCaller(user_id="user-from-identify", customer_id="acme")

    orders = build_server(style)
    instrument_mcp_server(orders.server, identify=identify)

    async with connect(orders) as client:
        await client.list_tools()
        await client.call("add", {"a": 1, "b": 2})
        await client.call("whoami", {})

    added = attributes(only_span(collecting_exporter, "tools/call add"))
    assert added["enduser.id"] == "user-from-identify"
    assert added["judgment.customer_id"] == "acme"
    # What the tool itself says wins over identify.
    whoami = attributes(only_span(collecting_exporter, "tools/call whoami"))
    assert whoami["enduser.id"] == "user-from-tool"
    assert whoami["judgment.customer_id"] == "customer-from-tool"
    assert seen and all(isinstance(info.headers, dict) for info in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
@pytest.mark.parametrize(
    "identify",
    [
        pytest.param(lambda info: 1 / 0, id="raises"),
        pytest.param(lambda info: {"user_id": "not-a-caller"}, id="wrong-type"),
        pytest.param(lambda info: McpCaller(user_id=""), id="empty"),
    ],
)
async def test_identify_problems_never_reach_the_call(
    style, identify, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server, identify=identify)

    async with connect(orders) as client:
        await client.list_tools()
        outcome = await client.call("add", {"a": 1, "b": 2})

    assert outcome.texts[0] == "3"
    found = attributes(only_span(collecting_exporter, "tools/call add"))
    assert "enduser.id" not in found and "judgment.customer_id" not in found


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_concurrent_calls_keep_their_own_caller(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)
    tags = [f"user-{index}" for index in range(6)]

    async with connect(orders) as client:
        await client.list_tools()
        async with anyio.create_task_group() as group:
            for tag in tags:
                group.start_soon(client.call, "tagged", {"tag": tag})

    by_request = spans(collecting_exporter, "tools/call tagged")
    assert sorted(attributes(span)["enduser.id"] for span in by_request) == tags


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_injection_can_be_turned_off(style, tracer, collecting_exporter):
    orders = build_server(style)
    original = None
    async with connect(build_server(style)) as client:
        original = await client.all_tools()
    instrument_mcp_server(orders.server, inject_agent_arguments=False)

    async with connect(orders) as client:
        tools = await client.all_tools()
        outcome = await client.call("add", {"a": 1, "b": 2})

    assert tools == original
    assert outcome.texts == ["3"]
    found = attributes(only_span(collecting_exporter, "tools/call add"))
    assert found["judgment.mcp.argument_keys"] == "a,b"
    for name in (
        "judgment.mcp.conversation_id",
        "judgment.mcp.intent",
        "judgment.mcp.model",
    ):
        assert name not in found


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_trace_context_in_meta_parents_the_span(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        await client.call("add", {"a": 1, "b": 2}, meta={"traceparent": TRACEPARENT})
        await client.call("add", {"a": 1, "b": 2})

    joined, alone = spans(collecting_exporter, "tools/call add")
    assert format(joined.context.trace_id, "032x") == TRACEPARENT.split("-")[1]
    assert joined.parent is not None
    assert format(joined.parent.span_id, "016x") == TRACEPARENT.split("-")[2]
    assert format(alone.context.trace_id, "032x") != TRACEPARENT.split("-")[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_a_span_around_the_server_is_not_the_parent(
    style, tracer, collecting_exporter
):
    from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

    orders = build_server(style)
    instrument_mcp_server(orders.server)
    app_tracer = JudgmentTracerProvider.get_instance().get_tracer("app")

    with app_tracer.start_as_current_span("server main"):
        async with connect(orders) as client:
            await client.list_tools()
            await client.call("add", {"a": 1, "b": 2})
            await client.call(
                "add", {"a": 1, "b": 2}, meta={"traceparent": TRACEPARENT}
            )
            await client.call("add", {"a": 1, "b": 2})

    first, joined, second = spans(collecting_exporter, "tools/call add")
    assert first.parent is None and second.parent is None
    assert first.context.trace_id != second.context.trace_id
    assert format(joined.parent.span_id, "016x") == TRACEPARENT.split("-")[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_spans_started_by_a_tool_are_its_children(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        await client.call("traced", {})

    call = only_span(collecting_exporter, "tools/call traced")
    inner = only_span(collecting_exporter, "inner work")
    assert orders.events["child_parent"] == call.context.span_id
    assert inner.context.trace_id == call.context.trace_id


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_cancelled_call_is_marked_cancelled(style, tracer, collecting_exporter):
    orders = build_server(style)
    instrument_mcp_server(orders.server)
    orders.events["started"] = anyio.Event()

    async with connect(orders) as client:
        await client.list_tools()
        if MCP_MAJOR == 1:
            from mcp import types

            request_id = client.session._request_id

            async def call() -> None:
                await client.call("slow", {"seconds": 30})

            async with anyio.create_task_group() as group:
                group.start_soon(call)
                await orders.events["started"].wait()
                await client.session.send_notification(
                    types.ClientNotification(
                        types.CancelledNotification(
                            method="notifications/cancelled",
                            params=types.CancelledNotificationParams(
                                requestId=request_id, reason="user stopped"
                            ),
                        )
                    )
                )
                with anyio.fail_after(10):
                    while not spans(collecting_exporter, "tools/call slow"):
                        await anyio.sleep(0.02)
                group.cancel_scope.cancel()
        else:
            async with anyio.create_task_group() as group:
                group.start_soon(client.call, "slow", {"seconds": 30})
                await orders.events["started"].wait()
                group.cancel_scope.cancel()
            with anyio.fail_after(10):
                while not spans(collecting_exporter, "tools/call slow"):
                    await anyio.sleep(0.02)
        after = await client.call("add", {"a": 1, "b": 1})

    slow = attributes(only_span(collecting_exporter, "tools/call slow"))
    assert slow["judgment.mcp.cancelled"] == "true"
    assert "error.type" not in slow
    if MCP_MAJOR > 1 or MCP_RELEASE >= (1, 20):
        # Older `mcp` releases can't serve a call after a cancellation, traced or not.
        assert after.texts[0] == "2"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_resources_and_prompts_are_traced(style, tracer, collecting_exporter):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.read_resource("orders://o-1?token=secret-token#details")
        await client.get_prompt("summarize", {"topic": "refunds"})

    read = attributes(only_span(collecting_exporter, "resources/read"))
    assert read["mcp.resource.uri"] == "orders://o-1"
    assert "gen_ai.tool.name" not in read
    prompt = attributes(only_span(collecting_exporter, "prompts/get summarize"))
    assert prompt["gen_ai.prompt.name"] == "summarize"
    assert "judgment.mcp.argument_keys" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_instrumenting_twice_traces_once(style, tracer, collecting_exporter):
    orders = build_server(style)
    assert instrument_mcp_server(orders.server) is True
    assert instrument_mcp_server(orders.server, server_name="other") is True

    async with connect(orders) as client:
        tools = await client.all_tools()
        await client.call("add", {"a": 1, "b": 2, "context": "Adding twice"})

    assert list(_schema_properties(tools["add"])).count("context") == 1
    found = attributes(only_span(collecting_exporter, "tools/call add"))
    assert found["judgment.mcp.server.name"] == SERVER_NAME
    assert orders.received("add") == [{"a": 1, "b": 2}]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_every_request_gets_a_span(style, tracer, collecting_exporter):
    orders = build_server(style)
    instrument_mcp_server(orders.server, server_name="orders-prod")

    async with connect(orders) as client:
        await client.list_tools()

    listed = attributes(only_span(collecting_exporter, "tools/list"))
    assert listed["mcp.method.name"] == "tools/list"
    assert listed["judgment.mcp.server.name"] == "orders-prod"
    assert listed["judgment.mcp.client.name"] == expected_client_name()
    assert "gen_ai.tool.name" not in listed
    for span in collecting_exporter.spans:
        assert span.kind.name == "SERVER"
        assert attributes(span)["judgment.mcp.sdk_version"] == get_version()


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
@pytest.mark.parametrize(
    "broken", ["_start_span", "_prepare", "_record_failure", "_shape_result"]
)
async def test_tracing_failures_never_reach_the_call(
    style, broken, tracer, collecting_exporter, monkeypatch
):
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("tracing broke")

    orders = build_server(style)
    instrument_mcp_server(orders.server)
    monkeypatch.setattr(_observer.McpObserver, broken, fail)

    async with connect(orders) as client:
        await client.list_tools()
        added = await client.call("add", {"a": 1, "b": 2, "context": "Adding"})
        failed = await client.call("lookup_marked", {"order_id": "o-404"})

    assert added.is_error is False
    assert added.texts[0] == "3"
    # Agent arguments are still removed, so a strict tool accepts the call.
    assert orders.received("add") == [{"a": 1, "b": 2}]
    assert failed.is_error is not False


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_without_a_tracer_calls_still_work(style):
    orders = build_server(style)
    assert instrument_mcp_server(orders.server) is True

    async with connect(orders) as client:
        tools = await client.all_tools()
        outcome = await client.call("add", {"a": 1, "b": 2, "context": "No tracer"})

    assert "context" in _schema_properties(tools["add"])
    assert outcome.texts[0] == "3"
    assert orders.received("add") == [{"a": 1, "b": 2}]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles(*sorted(LOW_LEVEL)))
async def test_open_and_composed_schemas_pass_arguments_through(
    style, tracer, collecting_exporter
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)

    async with connect(orders) as client:
        await client.list_tools()
        opened = await client.call("open", {"x": 1, "context": "kept"})
        composed = await client.call("any_of", {"a": "s", "context": "kept"})

    assert json.loads(opened.texts[0]) == {"x": 1, "context": "kept"}
    assert json.loads(composed.texts[0]) == {"a": "s", "context": "kept"}
    assert opened.conversation_id is None and composed.conversation_id is None
    for name, keys in (("open", "x,context"), ("any_of", "a,context")):
        found = attributes(only_span(collecting_exporter, f"tools/call {name}"))
        assert found["judgment.mcp.argument_keys"] == keys
        assert "judgment.mcp.intent" not in found
        assert "judgment.mcp.conversation_id" not in found


def test_unsupported_objects_are_left_untraced(tracer, caplog):
    class NotAServer:
        middleware = "not a list"

    assert instrument_mcp_server(NotAServer()) is False
    assert instrument_mcp_server(object()) is False
    assert instrument_mcp_server(None) is False
