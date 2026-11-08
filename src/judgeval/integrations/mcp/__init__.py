"""Traces the tool calls your MCP server handles, for MCP analytics in Judgment.

    from judgeval import Tracer
    from judgeval.integrations.mcp import instrument_mcp_server

    Tracer.init(project_name="orders-mcp")
    server = FastMCP("orders")
    instrument_mcp_server(server)

Each request becomes an OpenTelemetry SERVER span that follows the MCP
semantic conventions and adds ``judgment.mcp.*`` attributes: client, server,
argument names, and the intent, model and conversation the agent reports.
A call joins the agent's trace when ``params._meta`` carries one.

Supported servers: ``FastMCP`` and the low-level ``Server`` from `mcp` 1.x,
``MCPServer`` and the low-level ``Server`` from `mcp` 2.x, and the standalone
``fastmcp`` package. Argument values and results are never recorded, and a
failure in tracing never affects a call.
"""

from __future__ import annotations

import weakref
from typing import Any, Optional

from judgeval.logger import judgeval_logger

from ._observer import (
    Identify,
    McpCaller,
    McpObserver,
    McpOptions,
    McpRequestInfo,
    Redact,
    mark_mcp_call_error,
    set_mcp_caller,
)

_instrumented: "weakref.WeakSet[Any]" = weakref.WeakSet()


def instrument_mcp_server(
    server: Any,
    *,
    server_name: Optional[str] = None,
    server_build: Optional[str] = None,
    inject_agent_arguments: bool = True,
    identify: Optional[Identify] = None,
    capture_error_messages: bool = True,
    redact: Optional[Redact] = None,
) -> bool:
    """Traces every request an MCP server handles, through the active tracer.

    Call it once per server, after ``Tracer.init()`` and before the server
    starts. It changes your server in one way by default: each tool's published
    schema gets optional ``context``, ``llm_model`` and ``conversation_id``
    arguments, which are removed before your tool runs. When a call has no
    session, its result also gets a short text item with a conversation ID for
    the agent to send back.

    Args:
        server: A ``FastMCP``, ``MCPServer`` or low-level ``Server`` from `mcp`,
            or a ``FastMCP`` from the ``fastmcp`` package.
        server_name: Defaults to the name the server advertises.
        server_build: The deployed build, used to compare releases.
        inject_agent_arguments: Set to False to leave your tools and their
            results exactly as they are, at the cost of agent-reported intent
            and model, and of sessions for clients that send no session ID.
        identify: Returns the end user or customer of a request. Called once
            the request finishes. Exceptions are ignored.
        capture_error_messages: Records the text of a failure. False keeps only
            its type.
        redact: Rewrites failure messages, reported intent and resource URIs
            before they leave the process.

    Returns:
        True if the server is traced. On an unsupported server it logs a
        warning, returns False, and the server runs untraced.
    """
    try:
        return _instrument(
            server,
            McpOptions(
                server_name=server_name,
                server_build=server_build,
                inject_agent_arguments=inject_agent_arguments,
                identify=identify,
                capture_error_messages=capture_error_messages,
                redact=redact,
            ),
        )
    except Exception as error:
        judgeval_logger.warning(f"Could not instrument MCP server: {error!r}")
        return False


def _lowlevel_server(server: Any) -> Any:
    """Finds the low-level server that every server style is built on."""
    for attribute in ("_lowlevel_server", "_mcp_server"):
        inner = getattr(server, attribute, None)
        if inner is not None:
            return inner
    return server


def _instrument(server: Any, options: McpOptions) -> bool:
    lowlevel = _lowlevel_server(server)
    if lowlevel in _instrumented:
        return True
    observer = McpObserver(
        options,
        server_name=_text(getattr(lowlevel, "name", None)),
        server_version=_text(getattr(lowlevel, "version", None)),
    )
    from . import _v1, _v2

    if _v2.instrument(observer, lowlevel) or _v1.instrument(observer, lowlevel):
        # Both adapters already skip handlers they wrapped, so a server that
        # can't be remembered here is still safe to instrument twice.
        try:
            _instrumented.add(lowlevel)
        except TypeError:
            pass
        return True
    judgeval_logger.warning(
        "Could not instrument MCP server: it isn't an mcp 1.x or 2.x server. "
        "Calls still run, but aren't traced."
    )
    return False


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


__all__ = [
    "McpCaller",
    "McpRequestInfo",
    "instrument_mcp_server",
    "mark_mcp_call_error",
    "set_mcp_caller",
]
