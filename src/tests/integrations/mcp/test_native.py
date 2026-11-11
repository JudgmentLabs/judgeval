import json
import threading
import gc
import weakref

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

COMPLETED_CALL = {
    "timestamp": "2026-11-10T00:00:00Z",
    "duration_ms": 1,
    "outcome": "success",
    "server_name": "custom",
    "tool_name": "lookup",
}


def test_exporter_flushes_and_closes_when_its_owner_is_collected(native_requests):
    class Owner:
        pass

    owner = Owner()
    reference = weakref.ref(owner)
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    exporter.enqueue({**COMPLETED_CALL, "call_id": "collected-owner"})
    provider = JudgmentTracerProvider.get_instance()
    provider.add_background_exporter(exporter, owner)
    del owner
    gc.collect()
    assert (reference(), exporter.closed, native_requests) == (
        None,
        True,
        [{"calls": [{**COMPLETED_CALL, "call_id": "collected-owner"}]}],
    )


def test_background_delivery_recovers_after_flush_contention(
    monkeypatch, native_requests
):
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback

        def start(self):
            timers.append(self.callback)

        def cancel(self):
            pass

    monkeypatch.setattr(threading, "Timer", Timer)
    flush = McpCallExporter.force_flush
    outcomes = iter([False, True])

    def contended_flush(exporter):
        return flush(exporter) if next(outcomes) else False

    monkeypatch.setattr(McpCallExporter, "force_flush", contended_flush)
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    exporter.enqueue({**COMPLETED_CALL, "call_id": "event"})
    timers.pop(0)()
    if timers:
        timers.pop(0)()
    monkeypatch.setattr(McpCallExporter, "force_flush", flush)
    exporter.shutdown()
    assert native_requests == [{"calls": [{**COMPLETED_CALL, "call_id": "event"}]}]


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
                "error_message": call.get("error_message"),
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
                "error_message": None,
            }
            for outcome in ("success", "error")
        ],
        "spans": 0,
    }


def test_delivery_retries_identical_batches_and_stops_on_permanent_errors(monkeypatch):
    bodies = []
    statuses = iter([503, 200, 400])

    def post(client, method, url, **kwargs):
        bodies.append(kwargs["content"])
        return httpx.Response(next(statuses), json={})

    monkeypatch.setattr(httpx.Client, "request", post)
    monkeypatch.setattr(
        "judgeval.integrations.mcp._exporter.time.sleep", lambda _: None
    )
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    event = {**COMPLETED_CALL, "call_id": "stable", "input": {"key": "original"}}
    exporter.enqueue({**event, "error_type": ""})
    exporter.enqueue(event)
    event["input"]["key"] = "changed"
    exporter.force_flush()
    exporter.enqueue({**COMPLETED_CALL, "call_id": "second"})
    exporter.shutdown()
    assert {
        "bodies": [json.loads(body) for body in bodies],
        "identical": bodies[0] == bodies[1],
    } == {
        "bodies": [
            {
                "calls": [
                    {
                        **COMPLETED_CALL,
                        "call_id": "stable",
                        "input": {"key": "original"},
                    }
                ]
            }
        ]
        * 2
        + [{"calls": [{**COMPLETED_CALL, "call_id": "second"}]}],
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


@pytest.mark.parametrize("mode", ["deadline", "cooldown"])
def test_delivery_retains_unattempted_calls_and_respects_retry_after(mode, monkeypatch):
    clock = {"monotonic": 0, "wall": 1000}
    bodies = []
    monkeypatch.setattr(
        "judgeval.integrations.mcp._exporter.time.monotonic", lambda: clock["monotonic"]
    )
    monkeypatch.setattr(
        "judgeval.integrations.mcp._exporter.time.time", lambda: clock["wall"]
    )

    def request(client, method, url, **kwargs):
        bodies.append(kwargs["content"])
        if len(bodies) == 1 and mode == "deadline":
            clock["monotonic"] = 31
        return httpx.Response(
            429 if len(bodies) == 1 and mode == "cooldown" else 200,
            json={},
            headers={"Retry-After": "120"},
        )

    monkeypatch.setattr(httpx.Client, "request", request)
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    for n in range(101):
        exporter.enqueue({**COMPLETED_CALL, "call_id": str(n)})
    first_flush = exporter.force_flush()
    first_attempts = len(bodies)
    clock.update(monotonic=0, wall=1121)
    exporter.shutdown()
    assert {
        "first_flush": first_flush,
        "first_attempts": first_attempts,
        "sizes": [len(json.loads(body)["calls"]) for body in bodies],
        "identical": mode == "deadline" or bodies[0] == bodies[1],
    } == {
        "first_flush": False,
        "first_attempts": 1,
        "sizes": [100, 1] if mode == "deadline" else [100, 100, 1],
        "identical": True,
    }


def test_v2_capture_preserves_existing_middleware():
    from types import SimpleNamespace
    from judgeval.integrations.mcp._observer import McpObserver, McpOptions
    from judgeval.integrations.mcp._v2 import instrument, JudgmentMcpMiddleware

    upstream = object()
    server = SimpleNamespace(middleware=[upstream], get_request_handler=lambda _: None)
    observer = McpObserver(McpOptions(), server_name="test", server_version="1")
    instrument(observer, server)
    instrument(observer, server)
    assert {
        "native": isinstance(server.middleware[0], JudgmentMcpMiddleware),
        "remaining": server.middleware[1:],
    } == {"native": True, "remaining": [upstream]}


@pytest.mark.parametrize("count", [101, 1010])
def test_delivery_bounds_batches_and_queue(count, native_requests, caplog, monkeypatch):
    original = httpx.Client.request

    def rejected(client, method, url, **kwargs):
        original(client, method, url, **kwargs)
        return httpx.Response(403, json={})

    monkeypatch.setattr(httpx.Client, "request", rejected)
    exporter = McpCallExporter("http://localhost", "key", "org", "project")
    for index in range(count):
        exporter.enqueue({**COMPLETED_CALL, "call_id": str(index)})
    exporter.shutdown()
    exporter.enqueue({**COMPLETED_CALL, "call_id": "after-shutdown"})
    retained = min(count, 1000)
    assert {
        "sizes": [len(batch["calls"]) for batch in native_requests],
        "warnings": len(
            [record for record in caplog.records if "MCP export" in record.message]
        ),
    } == {
        "sizes": [min(100, retained - offset) for offset in range(0, retained, 100)],
        "warnings": 1,
    }
