import json

import pytest

from judgeval.integrations.mcp import instrument_mcp_server
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
from tests.integrations.mcp.harness import build_server, connect, styles


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_session_handoff_records_native_calls_without_spans(
    style, tracer, collecting_exporter, native_requests
):
    orders = build_server(style)
    instrument_mcp_server(orders.server)
    async with connect(orders) as client:
        await client.list_tools()
        first = await client.call(
            "add", {"a": 1, "b": 2, "context": "Adding", "llm_model": "gpt-6"}
        )
        session = (
            first.structured["_judgment"]["conversation_id"]
            if first.structured
            else first.conversation_id
        )
        second = await client.call("add", {"a": 2, "b": 3, "conversation_id": session})
    JudgmentTracerProvider.get_instance().force_flush()
    calls = [call for batch in native_requests for call in batch["calls"]]
    assert {
        "results": (first.texts[0], second.texts),
        "received": orders.calls,
        "session": bool(session),
        "handoff": json.loads(first.texts[-1])["conversation_id"],
        "native_sessions": [
            (
                call["session_id"],
                call["metadata"]["judgment.mcp.conversation_id_source"],
            )
            for call in calls
        ],
        "spans": collecting_exporter.spans,
        "reported": (calls[0]["intent"], calls[0]["model"]),
    } == {
        "results": ("3", ["5"]),
        "received": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
        "session": True,
        "handoff": session,
        "native_sessions": [(session, "minted"), (session, "echoed")],
        "spans": [],
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
        result = await client.call("add", {"a": 1, "b": 2, "context": "Adding"})
    assert (first, second, result.texts[0], orders.calls) == (
        True,
        True,
        "3",
        [{"a": 1, "b": 2}],
    )
