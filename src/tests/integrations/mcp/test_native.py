import json

import httpx
import pytest

from judgeval.integrations.mcp import McpCaller, capture_mcp_call, instrument_mcp_server
from judgeval.integrations.mcp._exporter import (
    OMITTED,
    McpCallExporter,
    capture_payload,
)
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
from tests.integrations.mcp.harness import build_server, connect, styles


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_explicit_session_and_default_capture(style, tracer, native_requests):
    orders = build_server(style)
    instrument_mcp_server(orders.server, session_id=lambda _: "application-session")
    async with connect(orders) as client:
        await client.call("add", {"a": 1, "b": 2})
    capture_mcp_call(
        {
            "timestamp": "2026-11-10T00:00:00Z",
            "duration_ms": 1,
            "outcome": "success",
            "server_name": "custom",
            "tool_name": "manual",
            "argument_keys": [],
            "metadata": {},
            "input": None,
        }
    )
    JudgmentTracerProvider.get_instance().force_flush()
    assert [
        {
            "tool": call["tool_name"],
            "session": call.get("session_id"),
            "input_present": "input" in call,
            "input": call.get("input"),
            "output_present": "output" in call,
        }
        for batch in native_requests
        for call in batch["calls"]
    ] == [
        {
            "tool": "add",
            "session": "application-session",
            "input_present": False,
            "input": None,
            "output_present": False,
        },
        {
            "tool": "manual",
            "session": None,
            "input_present": True,
            "input": None,
            "output_present": False,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_native_calls_without_spans(
    style, tracer, collecting_exporter, native_requests
):
    orders = build_server(style)
    tracer.environment = "test"
    instrument_mcp_server(
        orders.server,
        capture_input=True,
        capture_output=True,
        identify=lambda _: McpCaller(user_id="user-1"),
        redact_payload=lambda value, field: {**value, "a": "[redacted]"}
        if field == "input"
        else value,
    )
    async with connect(orders) as client:
        await client.list_tools()
        await client.call("add", {"a": 1, "b": 2})
        await client.call("add", {"a": "invalid", "b": 2})
    JudgmentTracerProvider.get_instance().force_flush()
    calls = [call for batch in native_requests for call in batch["calls"]]
    assert {
        "distinct_ids": calls[0]["call_id"] != calls[1]["call_id"],
        "events": [
            {
                "tool": call["tool_name"],
                "outcome": call["outcome"],
                "input": call["input"],
                "keys": call["argument_keys"],
                "user": call["user_id"],
                "environment": call["environment_name"],
                "trace": call.get("trace_id"),
                "output": "output" in call,
                "duration": call["duration_ms"] >= 0,
            }
            for call in calls
        ],
        "spans": len(collecting_exporter.spans),
    } == {
        "distinct_ids": True,
        "events": [
            {
                "tool": "add",
                "outcome": outcome,
                "input": {"a": "[redacted]", "b": 2},
                "keys": ["a", "b"],
                "user": "user-1",
                "environment": "test",
                "trace": None,
                "output": outcome == "success" or style != "mcp2-lowlevel",
                "duration": True,
            }
            for outcome in ("success", "error")
        ],
        "spans": 0,
    }


def test_delivery_retries_identical_batches_and_stops_on_permanent_errors(monkeypatch):
    bodies = []
    statuses = iter([503, 200, 400])

    def post(client, url, **kwargs):
        bodies.append(kwargs["content"])
        return httpx.Response(next(statuses))

    monkeypatch.setattr(httpx.Client, "post", post)
    monkeypatch.setattr(
        "judgeval.integrations.mcp._exporter.time.sleep", lambda _: None
    )
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    event = {"call_id": "stable", "input": {"key": "original"}}
    exporter.enqueue(event)
    event["input"]["key"] = "changed"
    exporter.force_flush()
    exporter.enqueue({"call_id": "second"})
    exporter.shutdown()
    assert {
        "bodies": [json.loads(body) for body in bodies],
        "identical": bodies[0] == bodies[1],
    } == {
        "bodies": [{"calls": [{"call_id": "stable", "input": {"key": "original"}}]}] * 2
        + [{"calls": [{"call_id": "second"}]}],
        "identical": True,
    }


def test_capture_detaches_and_omits_unsafe_payloads():
    original = {"token": "secret"}

    def redact(value, field):
        value["token"] = "safe"
        return value

    def fail(value, field):
        raise ValueError("secret")

    assert {
        "captured": capture_payload(original, "input", redact),
        "original": original,
        "failed": capture_payload(original, "output", fail),
        "oversized": capture_payload("x" * (128 * 1024), "output", None),
        "null": capture_payload(None, "output", None),
    } == {
        "captured": {"token": "safe"},
        "original": {"token": "secret"},
        "failed": OMITTED,
        "oversized": OMITTED,
        "null": None,
    }


@pytest.mark.parametrize("count", [101, 1001])
def test_delivery_bounds_batches_and_queue(count, native_requests):
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    for index in range(count):
        exporter.enqueue({"call_id": str(index)})
    exporter.shutdown()
    retained = min(count, 1000)
    assert [len(batch["calls"]) for batch in native_requests] == [
        min(100, retained - offset) for offset in range(0, retained, 100)
    ]
