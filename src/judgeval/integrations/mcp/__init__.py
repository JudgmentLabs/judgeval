"""Traces the tool calls your MCP server handles, for MCP analytics in Judgment.

    from judgeval import Tracer
    from judgeval.integrations.mcp import instrument_mcp_server

    Tracer.init(project_name="orders-mcp")
    server = FastMCP("orders")
    instrument_mcp_server(server)

Completed tool calls go directly to the native MCP endpoint without creating
OTel spans, independently of OTel sampling.

Supported servers: ``FastMCP`` and the low-level ``Server`` from `mcp` 1.x,
``MCPServer`` and the low-level ``Server`` from `mcp` 2.x, and the standalone
``fastmcp`` package. Input/output capture is opt-in. Telemetry failures never
affect tool execution. Call ``Tracer.force_flush()`` before serverless return
or ``Tracer.shutdown()`` when stopping. Bounded best-effort queues retry
transient errors three times with stable IDs/data. Hung or terminated calls
may never emit a completed event.
"""

from __future__ import annotations

import weakref
from typing import Any, Optional, Callable

from judgeval.logger import judgeval_logger
from ._exporter import McpCall, RedactPayload, OMITTED

from ._observer import (
    Identify,
    McpCaller,
    McpObserver,
    McpOptions,
    McpRequestInfo,
    Redact,
    mark_mcp_call_error,
    set_mcp_caller,
    capture_mcp_call,
)

_instrumented: "weakref.WeakSet[Any]" = weakref.WeakSet()


def instrument_mcp_server(
    server: Any,
    *,
    server_name: Optional[str] = None,
    server_build: Optional[str] = None,
    inject_agent_arguments: bool = True,
    identify: Optional[Identify] = None,
    capture_error_messages: bool = False,
    redact: Optional[Redact] = None,
    capture_input: bool = False,
    capture_output: bool = False,
    redact_payload: Optional[RedactPayload] = None,
    session_id: Optional[Callable[[McpRequestInfo], Optional[str]]] = None,
) -> bool:
    """Exports completed native MCP calls using the active tracer's settings.

    Call it once per server, after ``Tracer.init()`` and before the server
    starts. It changes your server in one way by default: each tool's published
    schema gets optional ``context``, ``llm_model`` and ``conversation_id``
    arguments, which are removed before your tool runs. When a call has no
    logical session, its result gets an ID in supplemental text and compatible
    existing ``structuredContent._judgment``. Compatible output schemas gain
    the matching optional property. Text-only/unsupported/colliding outputs
    retain text handoff. Transport sessions never define logical sessions.

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
        capture_error_messages: Opt in to failure text; defaults to False.
            Enabled text is sent as-is unless transformed by redact.
        redact: Rewrites failure messages, reported intent and resource URIs
            before they leave the process. No automatic PII scrubbing is applied.
            Agent-reported intent is enabled by default; use this callback to
            remove sensitive values or disable inject_agent_arguments.
        capture_input: Capture business arguments as-is; defaults to False.
        capture_output: Capture the original MCP result as-is; defaults to False.
        redact_payload: Transform a detached JSON input/output snapshot.
            Raise or return OMITTED to omit it; None captures JSON null.
            Non-JSON/oversized payloads (128 KiB) are omitted.
        session_id: Returns an authoritative application session ID without
            relying on agent echo. Receives McpRequestInfo.

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
                capture_input=capture_input,
                capture_output=capture_output,
                redact_payload=redact_payload,
                session_id=session_id,
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
        tool_manager=getattr(server, "_tool_manager", None),
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
    "OMITTED",
    "McpCall",
    "capture_mcp_call",
    "McpCaller",
    "McpRequestInfo",
    "instrument_mcp_server",
    "mark_mcp_call_error",
    "set_mcp_caller",
]
