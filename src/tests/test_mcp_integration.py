"""Runs the helper on a real `mcp` 2.x server and client."""

import asyncio

import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp import Client  # noqa: E402
from mcp.server import MCPServer  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from judgeval.integrations.mcp import McpCaller, instrument_mcp_server  # noqa: E402

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)


def call_tool(identify) -> dict:
    exporter.clear()
    server = MCPServer("orders")

    @server.tool()
    def search(query: str) -> str:
        return query

    instrument_mcp_server(server, identify=identify)

    async def run() -> None:
        async with Client(server) as client:
            await client.call_tool("search", {"query": "secret-value"})

    asyncio.run(run())
    span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.tool.name") == "search"
    )
    return dict(span.attributes)


def test_adds_server_and_caller_without_recording_arguments():
    attributes = call_tool(lambda _ctx: McpCaller(user_id="ana", customer_id="acme"))
    assert {
        "server": attributes.get("judgment.mcp.server.name"),
        "user": attributes.get("enduser.id"),
        "customer": attributes.get("judgment.customer_id"),
        "has_argument_value": "secret-value" in str(attributes),
    } == {
        "server": "orders",
        "user": "ana",
        "customer": "acme",
        "has_argument_value": False,
    }


def test_a_failing_identify_never_affects_the_call():
    def identify(_ctx):
        raise RuntimeError("identify failed")

    assert call_tool(identify).get("mcp.method.name") == "tools/call"


def test_warns_instead_of_failing_on_a_server_without_middleware():
    assert instrument_mcp_server(object()) is None
