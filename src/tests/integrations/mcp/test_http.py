"""One deployed HTTP smoke test; shared behavior is covered in memory."""

import importlib.util

import anyio
import pytest

from judgeval.integrations.mcp import instrument_mcp_server
from tests.integrations.mcp.harness import MCP_MAJOR, McpClient, build_server, serve


@pytest.mark.asyncio
async def test_http_calls_reach_the_active_tracer(tracer, native_requests):
    style = "mcp1-fastmcp" if MCP_MAJOR == 1 else "mcp2-mcpserver"
    orders = build_server(style)
    instrument_mcp_server(orders.server, session_id=lambda _: "http-session")
    streamable = importlib.util.find_spec("mcp.client.streamable_http") is not None
    app = orders.server.streamable_http_app() if streamable else orders.server.sse_app()
    with serve(app) as url, anyio.fail_after(30):
        if MCP_MAJOR == 1:
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            if streamable:
                from mcp.client.streamable_http import streamablehttp_client

                transport = streamablehttp_client(f"{url}/mcp")
            else:
                transport = sse_client(f"{url}/sse")
            async with transport as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    result = await McpClient(session).call("add", {"a": 1, "b": 2})
        else:
            from mcp.client.client import Client

            async with Client(f"{url}/mcp") as client:
                result = await McpClient(client.session).call("add", {"a": 1, "b": 2})
    tracer.force_flush()
    assert (
        result.texts,
        [
            (call["tool_name"], call["session_id"])
            for batch in native_requests
            for call in batch["calls"]
        ],
    ) == (
        ["3"],
        [("add", "http-session")],
    )
