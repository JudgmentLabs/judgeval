"""Minimal real-server fixtures shared by the MCP adapter smoke tests."""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import inspect
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest

pytest.importorskip("mcp")

MCP_MAJOR = int(importlib.metadata.version("mcp").split(".")[0])
CLIENT_NAME = "orders-agent"
CLIENT_VERSION = "2.0.0"
CONVERSATION_PROMPT_PREFIX = '{"conversation_id"'
LOW_LEVEL = {"mcp1-lowlevel", "mcp2-lowlevel"}


def styles():
    available = {
        "mcp1-fastmcp": MCP_MAJOR == 1,
        "mcp1-lowlevel": MCP_MAJOR == 1,
        "mcp2-mcpserver": MCP_MAJOR == 2,
        "mcp2-lowlevel": MCP_MAJOR == 2,
        "fastmcp": importlib.util.find_spec("fastmcp") is not None,
    }
    return [
        pytest.param(
            name, marks=pytest.mark.skipif(not installed, reason="not installed")
        )
        for name, installed in available.items()
    ]


@dataclass
class Orders:
    server: Any
    style: str
    calls: list = field(default_factory=list)

    @property
    def lowlevel(self):
        return getattr(
            self.server,
            "_mcp_server",
            getattr(self.server, "_lowlevel_server", self.server),
        )


def build_server(style):
    def add(a: int, b: int) -> int:
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("a and b must be integers")
        orders.calls.append({"a": a, "b": b})
        from judgeval import Tracer

        with Tracer.start_as_current_span("database"):
            return a + b

    if style not in LOW_LEVEL:
        if style == "mcp1-fastmcp":
            from mcp.server.fastmcp import FastMCP
        elif style == "mcp2-mcpserver":
            from mcp.server.mcpserver import MCPServer as FastMCP
        else:
            from fastmcp import FastMCP
        orders = Orders(FastMCP("orders"), style)
        orders.server.tool()(add)
        return orders

    from mcp import types
    from mcp.server.lowlevel import Server

    tool = {
        "name": "add",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    }
    if MCP_MAJOR == 1:
        server = Server("orders")

        @server.list_tools()
        async def list_tools():
            return [types.Tool.model_validate(tool)]

        @server.call_tool()
        async def call_tool(name, arguments):
            return [types.TextContent(type="text", text=str(add(**arguments)))]
    else:

        async def list_tools(ctx, params):
            return types.ListToolsResult.model_validate({"tools": [tool]})

        async def call_tool(ctx, params):
            from mcp.shared.exceptions import MCPError

            try:
                value = add(**params.arguments)
            except (TypeError, ValueError) as error:
                raise MCPError(code=-32602, message="bad params") from error
            return types.CallToolResult.model_validate(
                {"content": [{"type": "text", "text": str(value)}]}
            )

        server = Server("orders", on_list_tools=list_tools, on_call_tool=call_tool)
    orders = Orders(server, style)
    return orders


@dataclass
class Outcome:
    is_error: bool | None
    texts: list[str]
    structured: Any = None

    @property
    def conversation_id(self):
        return next(
            (
                json.loads(text)["conversation_id"]
                for text in self.texts
                if text.startswith(CONVERSATION_PROMPT_PREFIX)
            ),
            None,
        )


class McpClient:
    def __init__(self, session):
        self.session = session

    async def send(self, method, params):
        from mcp import types

        prefix = "ListTools" if method == "tools/list" else "CallTool"
        request = getattr(types, prefix + "Request").model_validate(
            {"method": method, **({"params": params} if params is not None else {})}
        )
        if MCP_MAJOR == 1:
            request = types.ClientRequest(request)
        return await self.session.send_request(
            request, getattr(types, prefix + "Result")
        )

    async def list_tools(self):
        return await self.send("tools/list", None)

    async def all_tools(self):
        result = await self.list_tools()
        return {
            tool.name: tool.model_dump(by_alias=True)["inputSchema"]
            for tool in result.tools
        }

    async def call(self, name, arguments):
        try:
            result = await self.send(
                "tools/call", {"name": name, "arguments": arguments}
            )
        except Exception:
            return Outcome(None, [])
        wire = result.model_dump(by_alias=True, exclude_none=True)
        return Outcome(
            wire.get("isError", False),
            [
                item["text"]
                for item in wire.get("content", [])
                if item["type"] == "text"
            ],
            wire.get("structuredContent"),
        )


@contextlib.asynccontextmanager
async def connect(orders):
    from mcp import types

    info = types.Implementation(name=CLIENT_NAME, version=CLIENT_VERSION)
    with anyio.fail_after(30):
        if orders.style == "fastmcp":
            from fastmcp import Client

            async with Client(orders.server, client_info=info) as client:
                yield McpClient(client.session)
        elif MCP_MAJOR == 1:
            from mcp.shared.memory import create_connected_server_and_client_session

            options = {}
            if (
                "client_info"
                in inspect.signature(
                    create_connected_server_and_client_session
                ).parameters
            ):
                options["client_info"] = info
            async with create_connected_server_and_client_session(
                orders.lowlevel, **options
            ) as session:
                yield McpClient(session)
        else:
            from mcp.client.client import Client

            async with Client(orders.server, client_info=info) as client:
                yield McpClient(client.session)


@contextlib.contextmanager
def serve(app):
    """Runs an ASGI app on a free port in another thread, as a deployment would."""
    import uvicorn

    # Bound here and handed over, so parallel test workers can't race for it.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="warning",
            lifespan="on",
            # An SSE stream stays open, so don't wait long for it on shutdown.
            timeout_graceful_shutdown=1,
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError("The test HTTP server didn't start")
            time.sleep(0.02)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
