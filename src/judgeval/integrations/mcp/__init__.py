"""Optional detail for the spans `mcp` 2.x servers already emit.

`mcp` 2.x traces every request on its own, so a Python server only needs an
OTLP exporter to appear in Judgment. Those spans don't say which client called,
which server handled it, or who the end user was. This adds that, without
changing your tools or their results.

    from judgeval.integrations.mcp import instrument_mcp_server

    server = MCPServer("orders")
    instrument_mcp_server(server, identify=lambda ctx: McpCaller(user_id=...))

Argument values and results are never recorded, and a failure here never affects
a call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from opentelemetry import trace

from judgeval.logger import judgeval_logger

_CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"


@dataclass(frozen=True)
class McpCaller:
    """Who a call acts for. Hash these if your identifiers are sensitive."""

    user_id: Optional[str] = None
    customer_id: Optional[str] = None


Identify = Callable[[Any], Optional[McpCaller]]


class JudgmentMcpMiddleware:
    """Adds client, server, and end-user attributes to the current MCP span."""

    def __init__(self, server_name: Optional[str], identify: Optional[Identify]):
        self._server_name = server_name
        self._identify = identify

    async def __call__(
        self, ctx: Any, call_next: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        try:
            self._annotate(ctx)
        except Exception as error:
            judgeval_logger.debug(f"MCP tracing failed: {error}")
        return await call_next(ctx)

    def _annotate(self, ctx: Any) -> None:
        span = trace.get_current_span()
        client = _client_info(ctx)
        headers = getattr(getattr(ctx, "request", None), "headers", None)
        caller = self._caller(ctx)
        attributes = {
            "judgment.mcp.server.name": self._server_name,
            "judgment.mcp.client.name": _read(client, "name"),
            "judgment.mcp.client.version": _read(client, "version"),
            "mcp.session.id": _session_id(ctx, headers),
            "user_agent.original": headers.get("user-agent") if headers else None,
            "enduser.id": caller.user_id if caller else None,
            "judgment.customer_id": caller.customer_id if caller else None,
        }
        for key, value in attributes.items():
            if isinstance(value, str) and value:
                span.set_attribute(key, value)

    def _caller(self, ctx: Any) -> Optional[McpCaller]:
        if self._identify is None:
            return None
        try:
            return self._identify(ctx)
        except Exception as error:
            judgeval_logger.debug(f"MCP identify failed: {error}")
            return None


def instrument_mcp_server(
    server: Any,
    *,
    server_name: Optional[str] = None,
    identify: Optional[Identify] = None,
) -> None:
    """Adds optional detail to an `mcp` 2.x server's spans.

    Args:
        server: An `MCPServer` or low-level `Server`.
        server_name: Defaults to the name the server advertises.
        identify: Returns the caller for a request context. Exceptions are ignored.
    """
    lowlevel = getattr(server, "_lowlevel_server", server)
    middleware = getattr(lowlevel, "middleware", None)
    if not isinstance(middleware, list):
        judgeval_logger.warning(
            "instrument_mcp_server needs mcp 2.x. Calls still run without this detail."
        )
        return
    name = server_name or getattr(lowlevel, "name", None)
    middleware.append(JudgmentMcpMiddleware(name, identify))


def _client_info(ctx: Any) -> Any:
    meta = getattr(ctx, "meta", None)
    from_meta = meta.get(_CLIENT_INFO_META_KEY) if isinstance(meta, dict) else None
    if from_meta is not None:
        return from_meta
    params = getattr(getattr(ctx, "session", None), "client_params", None)
    return getattr(params, "client_info", None)


def _session_id(ctx: Any, headers: Any) -> Optional[str]:
    connection = getattr(ctx, "connection", None)
    session_id = getattr(connection, "session_id", None)
    return session_id or (headers.get("mcp-session-id") if headers else None)


def _read(info: Any, key: str) -> Optional[str]:
    if isinstance(info, dict):
        value = info.get(key)
    else:
        value = getattr(info, key, None)
    return value if isinstance(value, str) else None


__all__ = ["JudgmentMcpMiddleware", "McpCaller", "instrument_mcp_server"]
