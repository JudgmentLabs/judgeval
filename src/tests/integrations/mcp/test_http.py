"""One deployed HTTP smoke test; shared behavior is covered in memory."""

import importlib.util

import anyio
import pytest
import httpx

from judgeval.integrations.mcp import instrument_mcp_server
from tests.integrations.mcp.harness import MCP_MAJOR, McpClient, build_server, serve


@pytest.mark.asyncio
async def test_http_calls_reach_the_active_tracer(tracer, collecting_exporter):
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
            (
                span.attributes["gen_ai.tool.name"],
                span.attributes["judgment.session_id"],
            )
            for span in collecting_exporter.spans
            if span.attributes.get("mcp.method.name") == "tools/call"
        ],
    ) == (
        ["3"],
        [("add", "http-session")],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("standalone", [False, True])
async def test_legacy_stateless_http_retains_each_clients_identity(
    tracer, collecting_exporter, standalone
):
    if standalone and importlib.util.find_spec("fastmcp") is None:
        pytest.skip("standalone FastMCP is not installed")
    style = (
        "fastmcp"
        if standalone
        else ("mcp1-fastmcp" if MCP_MAJOR == 1 else "mcp2-mcpserver")
    )
    orders = build_server(style)
    if not standalone and not hasattr(orders.server, "streamable_http_app"):
        pytest.skip("Streamable HTTP was introduced after mcp 1.6")
    instrument_mcp_server(orders.server)
    if standalone:
        app = orders.server.http_app(stateless_http=True, json_response=True)
    elif MCP_MAJOR == 1:
        orders.server.settings.stateless_http = True
        orders.server.settings.json_response = True
        app = orders.server.streamable_http_app()
    else:
        app = orders.server.streamable_http_app(stateless_http=True, json_response=True)

    stream_statuses = []
    neutral_tokens = []

    async def call(client, name):
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }
        response = await client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": name, "version": "client-1"},
                },
            },
        )
        token = response.headers["mcp-session-id"]
        neutral_tokens.append(token.startswith("mcp-session-v1."))
        headers["Mcp-Session-Id"] = (
            token.replace("mcp-session-v1.", "judgment-mcp-v1.")
            if name == "client-b"
            else token
        )
        for request_id in (2, 3):
            result = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": "add", "arguments": {"a": 1, "b": 2}},
                },
            )
            result.raise_for_status()
        for method in ("GET", "DELETE"):
            async with client.stream(method, "/mcp", headers=headers) as response:
                stream_statuses.append(
                    (response.status_code, response.headers.get("allow"))
                )

    with serve(app) as url, anyio.fail_after(30):
        async with (
            httpx.AsyncClient(base_url=url) as client,
            anyio.create_task_group() as group,
        ):
            group.start_soon(call, client, "client-a")
            group.start_soon(call, client, "client-b")
    assert (
        sorted(
            [
                (
                    span.attributes.get("judgment.mcp.client_name"),
                    span.attributes.get("judgment.mcp.client.version"),
                    span.attributes.get("mcp.protocol.version"),
                )
                for span in collecting_exporter.spans
                if span.attributes.get("mcp.method.name") == "tools/call"
            ]
        ),
        sorted(stream_statuses),
        neutral_tokens,
    ) == (
        [("client-a", "client-1", "2025-06-18")] * 2
        + [("client-b", "client-1", "2025-06-18")] * 2,
        [(405, "POST")] * 4,
        [True, True],
    )
