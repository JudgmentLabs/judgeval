"""Servers deployed over HTTP, served from another thread as in production.

The server thread never sees the tracer that ``Tracer.init()`` made active in
the test, so these also check that spans still reach it.
"""

from __future__ import annotations

import contextlib
import importlib.util
from typing import Any, AsyncIterator, Dict, List, Optional

import anyio
import pytest

pytest.importorskip("mcp")

from judgeval.integrations.mcp import McpCaller, instrument_mcp_server  # noqa: E402
from tests.integrations.mcp.harness import (  # noqa: E402
    CLIENT_NAME,
    CLIENT_VERSION,
    MCP_MAJOR,
    McpClient,
    attributes,
    build_server,
    client_info_supported,
    request_exposed,
    serve,
    spans,
    styles,
)

HAS_STREAMABLE_HTTP = importlib.util.find_spec("mcp.client.streamable_http") is not None


def _identify_from_headers(info: Any) -> McpCaller:
    return McpCaller(
        user_id=info.headers.get("x-user") or info.headers.get("user-agent"),
        customer_id=info.headers.get("x-customer"),
    )


async def _converse(client: McpClient) -> None:
    """Two calls, the second echoing a conversation ID when the first had one."""
    await client.list_tools()
    first = await client.call(
        "add", {"a": 1, "b": 2, "context": "First call over HTTP"}
    )
    assert first.is_error is False, first
    second_arguments: Dict[str, Any] = {"a": 2, "b": 3}
    if first.conversation_id is not None:
        second_arguments["conversation_id"] = first.conversation_id
    second = await client.call("add", second_arguments)
    assert second.is_error is False, second
    assert second.conversation_id is None


def _conversation_key(found: Dict[str, Any]) -> Optional[str]:
    """How MCP analytics groups calls: the conversation, else the session."""
    return found.get("judgment.mcp.conversation_id") or found.get("mcp.session.id")


def _calls(exporter: Any) -> List[Dict[str, Any]]:
    found = [attributes(span) for span in spans(exporter, "tools/call add")]
    assert len(found) == 2, [span.name for span in exporter.spans]
    return found


@contextlib.asynccontextmanager
async def _mcp1_client(url: str, transport: str) -> AsyncIterator[McpClient]:
    from mcp import ClientSession, types

    headers = {"x-user": "alice", "x-customer": "acme"}
    if transport == "sse":
        from mcp.client.sse import sse_client

        streams_context: Any = sse_client(f"{url}/sse", headers=headers)
    else:
        from mcp.client.streamable_http import streamablehttp_client

        streams_context = streamablehttp_client(f"{url}/mcp", headers=headers)
    options: Dict[str, Any] = {}
    if client_info_supported():
        options["client_info"] = types.Implementation(
            name=CLIENT_NAME, version=CLIENT_VERSION
        )
    with anyio.fail_after(30):
        async with streams_context as streams:
            async with ClientSession(streams[0], streams[1], **options) as session:
                await session.initialize()
                yield McpClient(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stateful", "stateless", "sse"])
@pytest.mark.parametrize("style", styles("mcp1-fastmcp"))
async def test_mcp1_fastmcp_over_http(style, transport, tracer, collecting_exporter):
    if transport != "sse" and not HAS_STREAMABLE_HTTP:
        pytest.skip("this `mcp` has no streamable HTTP transport")
    orders = build_server(style)
    if transport == "stateless":
        orders.server.settings.stateless_http = True
    instrument_mcp_server(
        orders.server, record_spans=True, identify=_identify_from_headers
    )
    app = (
        orders.server.sse_app()
        if transport == "sse"
        else orders.server.streamable_http_app()
    )

    with serve(app) as url:
        async with _mcp1_client(url, transport) as client:
            await _converse(client)

    first, second = _calls(collecting_exporter)
    assert orders.received("add") == [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
    assert first["judgment.mcp.intent"] == "First call over HTTP"
    assert "judgment.mcp.intent" not in second
    if request_exposed():
        for found in (first, second):
            assert found["enduser.id"] == "alice"
            assert found["judgment.customer_id"] == "acme"
            assert found["user_agent.original"]
    if transport == "stateless" or not request_exposed():
        # No session: the agent carries the conversation instead.
        assert "mcp.session.id" not in first
        assert first["judgment.mcp.conversation_id_source"] == "minted"
        assert second["judgment.mcp.conversation_id_source"] == "echoed"
    else:
        assert first["mcp.session.id"]
        assert first["mcp.session.id"] == second["mcp.session.id"]
        assert first["judgment.mcp.conversation_id"] != first["mcp.session.id"]
    assert _conversation_key(first) == _conversation_key(second)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport",
    ["stateful-legacy", "stateful-modern", "stateless-legacy", "stateless-modern"],
)
@pytest.mark.parametrize("style", styles("mcp2-mcpserver"))
async def test_mcp2_mcpserver_over_streamable_http(
    style, transport, tracer, collecting_exporter
):
    from mcp import types
    from mcp.client.client import Client

    orders = build_server(style)
    instrument_mcp_server(
        orders.server, record_spans=True, identify=_identify_from_headers
    )
    app = orders.server.streamable_http_app(
        stateless_http=transport.startswith("stateless")
    )

    with serve(app) as url:
        with anyio.fail_after(30):
            async with Client(
                f"{url}/mcp",
                mode="legacy" if transport.endswith("legacy") else "auto",
                client_info=types.Implementation(
                    name=CLIENT_NAME, version=CLIENT_VERSION
                ),
            ) as client:
                await _converse(McpClient(client.session))

    first, second = _calls(collecting_exporter)
    assert orders.received("add") == [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
    assert first["enduser.id"]
    if transport == "stateful-legacy":
        assert first["mcp.session.id"]
        assert first["mcp.session.id"] == second["mcp.session.id"]
        assert first["judgment.mcp.conversation_id"] != first["mcp.session.id"]
        assert first["judgment.mcp.client.name"] == CLIENT_NAME
    else:
        # The 2026-07-28 protocol and stateless servers have no session.
        assert "mcp.session.id" not in first
        assert first["judgment.mcp.conversation_id_source"] == "minted"
        assert second["judgment.mcp.conversation_id_source"] == "echoed"
    if transport.endswith("modern"):
        assert first["mcp.protocol.version"] >= "2026"
        # The client sends who it is with every request.
        assert first["judgment.mcp.client.name"] == CLIENT_NAME
    assert _conversation_key(first) == _conversation_key(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles("mcp2-mcpserver"))
async def test_mcp2_mcpserver_over_sse(style, tracer, collecting_exporter):
    from mcp import types
    from mcp.client.client import Client
    from mcp.client.sse import sse_client

    orders = build_server(style)
    instrument_mcp_server(orders.server, record_spans=True)

    with serve(orders.server.sse_app()) as url:
        with anyio.fail_after(30):
            async with Client(
                sse_client(f"{url}/sse"),
                mode="legacy",
                client_info=types.Implementation(
                    name=CLIENT_NAME, version=CLIENT_VERSION
                ),
            ) as client:
                await _converse(McpClient(client.session))

    first, second = _calls(collecting_exporter)
    assert first["mcp.session.id"]
    assert first["mcp.session.id"] == second["mcp.session.id"]
    assert first["judgment.mcp.client.name"] == CLIENT_NAME
    assert first["judgment.mcp.conversation_id"] != first["mcp.session.id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles("fastmcp"))
async def test_fastmcp_over_http(style, tracer, collecting_exporter):
    from fastmcp import Client as FastMCPClient

    orders = build_server(style)
    instrument_mcp_server(orders.server, record_spans=True)

    with serve(orders.server.http_app()) as url:
        with anyio.fail_after(30):
            async with FastMCPClient(f"{url}/mcp") as client:
                await _converse(McpClient(client.session))

    first, second = _calls(collecting_exporter)
    assert orders.received("add") == [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
    assert first["judgment.mcp.intent"] == "First call over HTTP"
    assert _conversation_key(first)
    assert _conversation_key(first) == _conversation_key(second)
    if MCP_MAJOR == 1:
        assert first["mcp.session.id"]
