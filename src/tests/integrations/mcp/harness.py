"""Real MCP servers in each supported style, and a client that talks to them.

A style is skipped when the installed packages don't provide it, so the same
tests run against `mcp` 1.x, `mcp` 2.x and each `fastmcp` major. The client
sends requests in their wire shape, so a test controls exactly what reaches the
server, including ``_meta``.
"""

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
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

import anyio
import pytest
from opentelemetry.sdk.trace import ReadableSpan

from judgeval.integrations.mcp import mark_mcp_call_error, set_mcp_caller
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

MCP_VERSION = importlib.metadata.version("mcp")
MCP_MAJOR = int(MCP_VERSION.split(".")[0])
MCP_RELEASE = tuple(int(part) for part in MCP_VERSION.split(".")[:2])
FASTMCP_VERSION = (
    importlib.metadata.version("fastmcp")
    if importlib.util.find_spec("fastmcp") is not None
    else None
)

CLIENT_NAME = "orders-agent"
CLIENT_VERSION = "2.0.0"
SERVER_NAME = "orders"
LOWLEVEL_VERSION = "9.9.9"

STYLES: Dict[str, bool] = {
    "mcp1-fastmcp": MCP_MAJOR == 1,
    "mcp1-lowlevel": MCP_MAJOR == 1,
    "mcp2-mcpserver": MCP_MAJOR == 2,
    "mcp2-lowlevel": MCP_MAJOR == 2,
    "fastmcp": FASTMCP_VERSION is not None,
}
HIGH_LEVEL = frozenset({"mcp1-fastmcp", "mcp2-mcpserver", "fastmcp"})
LOW_LEVEL = frozenset({"mcp1-lowlevel", "mcp2-lowlevel"})
AGENT_ARGUMENTS = ("context", "llm_model", "conversation_id")
CONVERSATION_PROMPT_PREFIX = '{"conversation_id"'


def styles(*names: str) -> List[Any]:
    """Parameters for each style, skipped when it isn't installed."""
    return [
        pytest.param(
            name,
            id=name,
            marks=pytest.mark.skipif(
                not STYLES[name], reason=f"{name} isn't installed"
            ),
        )
        for name in (names or tuple(STYLES))
    ]


def lowlevel_validates_input() -> bool:
    """`mcp` 1.x checks arguments against the listed schema since 1.10."""
    if MCP_MAJOR != 1:
        return True
    from mcp.server.lowlevel import Server

    return "validate_input" in inspect.signature(Server.call_tool).parameters


def request_exposed() -> bool:
    """Whether handlers can see the HTTP request. Early 1.x releases can't."""
    if MCP_MAJOR != 1:
        return True
    import dataclasses

    from mcp.shared.context import RequestContext

    return "request" in {field.name for field in dataclasses.fields(RequestContext)}


def client_info_supported() -> bool:
    if MCP_MAJOR != 1:
        return True
    from mcp.shared.memory import create_connected_server_and_client_session

    return (
        "client_info"
        in inspect.signature(create_connected_server_and_client_session).parameters
    )


class OrderNotFound(Exception):
    pass


class ToolFailure(Exception):
    """Fails a low-level `mcp` 1.x tool with exactly this message."""


class InvalidArguments(Exception):
    """Arguments a low-level server that doesn't check them can't use."""


@dataclass
class Orders:
    """A server, and the arguments its tools received."""

    server: Any
    style: str
    calls: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    events: Dict[str, Any] = field(default_factory=dict)

    def received(self, name: str) -> List[Dict[str, Any]]:
        return [arguments for tool, arguments in self.calls if tool == name]

    @property
    def lowlevel(self) -> Any:
        for attribute in ("_lowlevel_server", "_mcp_server"):
            inner = getattr(self.server, attribute, None)
            if inner is not None:
                return inner
        return self.server


def _closed(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_STRING = {"type": "string"}

# The tools a low-level server lists, in their wire shape.
LOWLEVEL_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "add",
        "description": "Adds two numbers.",
        "inputSchema": _closed(
            {"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a", "b"]
        ),
    },
    {
        "name": "own_context",
        "description": "Declares its own context argument.",
        "inputSchema": _closed(
            {"context": {"type": "string", "description": "The tool's own context."}},
            ["context"],
        ),
    },
    {
        "name": "lookup",
        "inputSchema": _closed({"order_id": _STRING}, ["order_id"]),
    },
    {
        "name": "lookup_marked",
        "inputSchema": _closed({"order_id": _STRING}, ["order_id"]),
    },
    {"name": "coded", "inputSchema": _closed({}, [])},
    {"name": "whoami", "inputSchema": _closed({}, [])},
    {"name": "tagged", "inputSchema": _closed({"tag": _STRING}, ["tag"])},
    {"name": "traced", "inputSchema": _closed({}, [])},
    {
        "name": "slow",
        "inputSchema": _closed({"seconds": {"type": "number"}}, ["seconds"]),
    },
    {
        "name": "open",
        "inputSchema": {"type": "object", "additionalProperties": True},
    },
    {
        "name": "any_of",
        "inputSchema": {
            "type": "object",
            "anyOf": [{"properties": {"a": _STRING}, "required": ["a"]}],
        },
    },
]


def child_span_parent() -> Optional[int]:
    """Starts a span inside a tool and returns its parent's span ID."""
    tracer = JudgmentTracerProvider.get_instance().get_tracer("orders-tool")
    with tracer.start_as_current_span("inner work") as span:
        parent = getattr(span, "parent", None)
        return None if parent is None else parent.span_id


async def run_lowlevel_tool(
    orders: Orders, name: str, arguments: Dict[str, Any]
) -> Tuple[str, bool]:
    """The tools of the low-level servers. Returns the text and whether it failed."""
    if name == "add":
        if not all(isinstance(arguments.get(key), int) for key in ("a", "b")):
            raise InvalidArguments("a and b must be integers")
        orders.calls.append((name, dict(arguments)))
        return str(arguments["a"] + arguments["b"]), False
    if name == "own_context":
        orders.calls.append((name, dict(arguments)))
        return f"context={arguments['context']}", False
    if name == "lookup":
        raise OrderNotFound(f"order {arguments['order_id']} not found")
    if name == "lookup_marked":
        error = OrderNotFound(f"order {arguments['order_id']} not found")
        mark_mcp_call_error(error)
        return str(error), True
    if name == "coded":
        return json.dumps({"error": "rate_limited", "retry_after_seconds": 3}), True
    if name == "whoami":
        set_mcp_caller(user_id="user-from-tool", customer_id="customer-from-tool")
        return "ok", False
    if name == "tagged":
        set_mcp_caller(user_id=arguments["tag"])
        await anyio.sleep(0.05)
        return arguments["tag"], False
    if name == "traced":
        orders.events["child_parent"] = child_span_parent()
        return "ok", False
    if name == "slow":
        orders.events["started"].set()
        await anyio.sleep(arguments["seconds"])
        return "done", False
    if name in ("open", "any_of"):
        orders.calls.append((name, dict(arguments)))
        return json.dumps(arguments, sort_keys=True), False
    raise LookupError(f"Unknown tool: {name}")


def _register_high_level(server: Any, orders: Orders) -> None:
    @server.tool()
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        orders.calls.append(("add", {"a": a, "b": b}))
        return a + b

    @server.tool()
    def own_context(context: str) -> str:
        """Declares its own context argument."""
        orders.calls.append(("own_context", {"context": context}))
        return f"context={context}"

    @server.tool()
    def lookup(order_id: str) -> str:
        raise OrderNotFound(f"order {order_id} not found")

    @server.tool()
    def lookup_marked(order_id: str) -> str:
        error = OrderNotFound(f"order {order_id} not found")
        mark_mcp_call_error(error)
        raise error

    @server.tool()
    def whoami() -> str:
        set_mcp_caller(user_id="user-from-tool", customer_id="customer-from-tool")
        return "ok"

    @server.tool()
    async def tagged(tag: str) -> str:
        set_mcp_caller(user_id=tag)
        await anyio.sleep(0.05)
        return tag

    @server.tool()
    def traced() -> str:
        orders.events["child_parent"] = child_span_parent()
        return "ok"

    @server.tool()
    async def slow(seconds: float) -> str:
        orders.events["started"].set()
        await anyio.sleep(seconds)
        return "done"

    @server.resource("orders://{order_id}")
    def order(order_id: str) -> str:
        return json.dumps({"order_id": order_id})

    @server.prompt()
    def summarize(topic: str) -> str:
        return f"Summarize {topic}."


def build_server(style: str) -> Orders:
    """An uninstrumented server named ``orders`` in the given style."""
    if style == "mcp1-fastmcp":
        from mcp.server.fastmcp import FastMCP

        orders = Orders(FastMCP(SERVER_NAME), style)
        _register_high_level(orders.server, orders)
        return orders
    if style == "mcp2-mcpserver":
        from mcp.server.mcpserver import MCPServer

        orders = Orders(MCPServer(SERVER_NAME), style)
        _register_high_level(orders.server, orders)
        return orders
    if style == "fastmcp":
        from fastmcp import FastMCP as StandaloneFastMCP

        orders = Orders(StandaloneFastMCP(SERVER_NAME), style)
        _register_high_level(orders.server, orders)
        return orders
    if style == "mcp1-lowlevel":
        return _mcp1_lowlevel()
    if style == "mcp2-lowlevel":
        return _mcp2_lowlevel()
    raise ValueError(style)


def _mcp1_lowlevel() -> Orders:
    from mcp.server.lowlevel import Server

    orders = Orders(Server(SERVER_NAME, version=LOWLEVEL_VERSION), "mcp1-lowlevel")
    register_mcp1_lowlevel_handlers(orders)
    return orders


def register_mcp1_lowlevel_handlers(orders: Orders) -> None:
    from mcp import types

    server = orders.server

    @server.list_tools()
    async def list_tools() -> List[Any]:
        return [types.Tool.model_validate(tool) for tool in LOWLEVEL_TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Any]:
        text, failed = await run_lowlevel_tool(orders, name, arguments or {})
        if failed:
            # 1.x reports an exception raised here as an error result.
            raise ToolFailure(text)
        return [types.TextContent(type="text", text=text)]

    @server.read_resource()
    async def read_resource(uri: Any) -> str:
        return json.dumps({"uri": str(uri)})

    @server.get_prompt()
    async def get_prompt(name: str, arguments: Optional[Dict[str, str]]) -> Any:
        return types.GetPromptResult.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": f"Summarize {arguments}"},
                    }
                ]
            }
        )


def _mcp2_lowlevel() -> Orders:
    from mcp import types
    from mcp.server.lowlevel import Server

    holder: Dict[str, Orders] = {}

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult.model_validate({"tools": LOWLEVEL_TOOLS})

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        try:
            text, failed = await run_lowlevel_tool(
                holder["orders"], params.name, dict(params.arguments or {})
            )
        except (LookupError, InvalidArguments) as error:
            # How a 2.x low-level server refuses a call: it checks no schema.
            from mcp.shared.exceptions import MCPError

            raise MCPError(code=-32602, message=str(error)) from error
        return types.CallToolResult.model_validate(
            {"content": [{"type": "text", "text": text}], "isError": failed}
        )

    async def on_read_resource(ctx: Any, params: Any) -> Any:
        return types.ReadResourceResult.model_validate(
            {"contents": [{"uri": str(params.uri), "text": "{}"}]}
        )

    async def on_get_prompt(ctx: Any, params: Any) -> Any:
        return types.GetPromptResult.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": f"Summarize {params.name}"},
                    }
                ]
            }
        )

    server = Server(
        SERVER_NAME,
        version=LOWLEVEL_VERSION,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_read_resource=on_read_resource,
        on_get_prompt=on_get_prompt,
    )
    orders = Orders(server, "mcp2-lowlevel")
    holder["orders"] = orders
    return orders


@dataclass
class Outcome:
    """A response, normalized across `mcp` versions."""

    is_error: Optional[bool]
    texts: List[str]
    structured: Any = None
    # The JSON-RPC error message, when the request failed outright.
    error: Optional[str] = None

    @property
    def conversation_id(self) -> Optional[str]:
        for text in self.texts:
            if text.startswith(CONVERSATION_PROMPT_PREFIX):
                return json.loads(text)["conversation_id"]
        return None


_REQUESTS = {
    "tools/list": ("ListToolsRequest", "ListToolsResult"),
    "tools/call": ("CallToolRequest", "CallToolResult"),
    "resources/read": ("ReadResourceRequest", "ReadResourceResult"),
    "prompts/get": ("GetPromptRequest", "GetPromptResult"),
}


class McpClient:
    """Sends wire-shaped requests over a connected `mcp` ClientSession."""

    def __init__(self, session: Any) -> None:
        self.session = session

    async def send(self, method: str, params: Optional[Dict[str, Any]]) -> Any:
        from mcp import types

        request_name, result_name = _REQUESTS[method]
        wire: Dict[str, Any] = {"method": method}
        if params is not None:
            wire["params"] = params
        request = getattr(types, request_name).model_validate(wire)
        if MCP_MAJOR == 1:
            request = types.ClientRequest(request)
        return await self.session.send_request(request, getattr(types, result_name))

    async def list_tools(
        self, cursor: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
        result = await self.send(
            "tools/list", None if cursor is None else {"cursor": cursor}
        )
        wire = result.model_dump(by_alias=True, exclude_none=True)
        return (
            {tool["name"]: tool.get("inputSchema", {}) for tool in wire["tools"]},
            wire.get("nextCursor"),
        )

    async def all_tools(self) -> Dict[str, Dict[str, Any]]:
        tools, cursor = await self.list_tools()
        while cursor is not None:
            page, cursor = await self.list_tools(cursor)
            tools.update(page)
        return tools

    async def call(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Outcome:
        params: Dict[str, Any] = {"name": name, "arguments": arguments or {}}
        if meta is not None:
            params["_meta"] = meta
        return await self._outcome("tools/call", params)

    async def read_resource(self, uri: str) -> Outcome:
        return await self._outcome("resources/read", {"uri": uri})

    async def get_prompt(self, name: str, arguments: Dict[str, str]) -> Outcome:
        return await self._outcome(
            "prompts/get", {"name": name, "arguments": arguments}
        )

    async def _outcome(self, method: str, params: Dict[str, Any]) -> Outcome:
        try:
            result = await self.send(method, params)
        except Exception as error:
            inner = getattr(error, "error", None)
            message = getattr(inner, "message", None) or str(error)
            return Outcome(None, [], error=message)
        wire = result.model_dump(by_alias=True, exclude_none=True)
        texts = [
            item["text"]
            for item in wire.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return Outcome(
            wire.get("isError") is True, texts, wire.get("structuredContent")
        )


@contextlib.asynccontextmanager
async def connect(orders: Orders) -> AsyncIterator[McpClient]:
    """Connects a client to the server in memory, as the style's own SDK does."""
    with anyio.fail_after(30):
        if orders.style == "fastmcp":
            from fastmcp import Client as FastMCPClient
            from mcp import types

            async with FastMCPClient(
                orders.server,
                client_info=types.Implementation(
                    name=CLIENT_NAME, version=CLIENT_VERSION
                ),
            ) as client:
                yield McpClient(client.session)
        elif MCP_MAJOR == 1:
            from mcp import types
            from mcp.shared.memory import create_connected_server_and_client_session

            options: Dict[str, Any] = {}
            if client_info_supported():
                options["client_info"] = types.Implementation(
                    name=CLIENT_NAME, version=CLIENT_VERSION
                )
            async with create_connected_server_and_client_session(
                orders.lowlevel, **options
            ) as session:
                yield McpClient(session)
        else:
            from mcp import types
            from mcp.client.client import Client

            async with Client(
                orders.server,
                client_info=types.Implementation(
                    name=CLIENT_NAME, version=CLIENT_VERSION
                ),
            ) as client:
                yield McpClient(client.session)


def expected_client_name() -> str:
    """Clients of `mcp` before 1.10 always call themselves ``mcp``."""
    return CLIENT_NAME if client_info_supported() else "mcp"


def spans(exporter: Any, name: Optional[str] = None) -> List[ReadableSpan]:
    return [span for span in exporter.spans if name is None or span.name == name]


def only_span(exporter: Any, name: str) -> ReadableSpan:
    found = spans(exporter, name)
    assert len(found) == 1, [span.name for span in exporter.spans]
    return found[0]


def attributes(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


@contextlib.contextmanager
def serve(app: Any) -> Iterator[str]:
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
