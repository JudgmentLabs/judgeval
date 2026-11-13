"""Traces the tool calls your MCP server handles, for MCP analytics in Judgment.

    from judgeval import Tracer
    from judgeval.integrations.mcp import instrument_mcp_server

    Tracer.init(project_name="orders-mcp")
    server = FastMCP("orders")
    instrument_mcp_server(server)

Each tool call creates an OTel span using the active Judgment tracer. Explicit
incoming trace context is honored; otherwise calls start new traces. Instrumented
work inside the tool becomes child spans.

Supported servers: MCP 1.x/2.x and standalone FastMCP. Input/output capture is
opt-in. Telemetry failures never affect tool execution. Call Tracer.force_flush()
before serverless return or Tracer.shutdown() when stopping.
"""

from __future__ import annotations

import weakref
from typing import Any, Optional, Callable

from judgeval.logger import judgeval_logger
from ._payload import RedactPayload, OMITTED

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

_instrumented: "weakref.WeakKeyDictionary[Any, weakref.ReferenceType[McpOptions]]" = (
    weakref.WeakKeyDictionary()
)


def instrument_mcp_server(
    server: Any,
    *,
    server_name: Optional[str] = None,
    server_build: Optional[str] = None,
    inject_agent_arguments: bool = True,
    inject_conversation_id: bool = True,
    intent_argument: Optional[str] = None,
    model_argument: Optional[str] = None,
    conversation_id_argument: Optional[str] = None,
    identify: Optional[Identify] = None,
    capture_error_messages: bool = False,
    redact: Optional[Redact] = None,
    capture_input: bool = False,
    capture_output: bool = False,
    redact_payload: Optional[RedactPayload] = None,
    session_id: Optional[Callable[[McpRequestInfo], Optional[str]]] = None,
) -> bool:
    """Traces MCP tool calls using the active tracer's settings.

    Call it once per server, after ``Tracer.init()`` and before the server
    starts. It changes your server in one way by default: each tool's published
    schema gets optional ``intent``, ``llm_model`` and ``conversation_id``
    arguments, which are removed before your tool runs. When a call has no
    application or echoed conversation session, its successful result gets an ID in
    supplemental text and compatible existing ``structuredContent._judgment``.
    Compatible output schemas gain
    the matching optional property. Text-only/unsupported/colliding outputs
    retain text handoff. Transport session IDs are used only when no application
    or conversation session is available, including while waiting for the agent
    to echo a newly offered ID.

    Args:
        server: A ``FastMCP``, ``MCPServer`` or low-level ``Server`` from `mcp`,
            or a ``FastMCP`` from the ``fastmcp`` package.
        server_name: Defaults to the name the server advertises.
        server_build: The deployed build, used to compare releases.
        inject_agent_arguments: Set to False to leave your tools and their
            results exactly as they are, at the cost of agent-reported intent
            and model, and of sessions for clients that send no session ID.
        inject_conversation_id: Set to False to disable conversation-ID schema
            injection and handoff independently of intent/model injection.
        intent_argument: Name to use instead of intent. Explicitly opts a tool's
            existing argument into telemetry; otherwise injects an optional field.
            Existing arguments are never stripped or changed. Missing/non-string
            values are omitted. Explicit names can be read with injection off.
        model_argument: Same behavior for the calling model; defaults to llm_model.
        conversation_id_argument: Same behavior for conversation_id. Existing
            fields supply application session IDs; injected fields use generated
            UUIDv7 handles. Callback/baggage sessions still take priority.
        identify: Returns the end user or customer of a request. Called once
            the request finishes. Exceptions are ignored.
        capture_error_messages: Opt in to failure text; defaults to False.
            Enabled text is sent as-is unless transformed by redact.
        redact: Rewrites failure messages and reported intent
            before they leave the process. No automatic PII scrubbing is applied.
            Agent-reported intent is enabled by default; use this callback to
            remove sensitive values or disable inject_agent_arguments.
        capture_input: Capture business arguments as-is; defaults to False.
        capture_output: Capture the original MCP result as-is; defaults to False.
        redact_payload: Transform a detached JSON input/output snapshot.
            Raise or return OMITTED to omit it; None captures JSON null.
            Non-JSON/oversized payloads (128 KiB) are omitted. Oversized source
            payloads are omitted before this callback runs.
        session_id: Returns an authoritative application session ID without
            relying on agent echo. Receives McpRequestInfo. Existing
            set_session_id baggage is also authoritative; scope it to each
            request rather than the lifetime of the server.

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
                inject_conversation_id=inject_conversation_id,
                intent_argument=intent_argument,
                model_argument=model_argument,
                conversation_id_argument=conversation_id_argument,
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
    previous = _instrumented.get(lowlevel)
    if previous is not None:
        if previous() != options:
            judgeval_logger.warning(
                "MCP server is already instrumented with different options; "
                "keeping the original configuration."
            )
        return True
    observer = McpObserver(
        options,
        server_name=_text(getattr(lowlevel, "name", None)),
        server_version=_text(getattr(lowlevel, "version", None)),
        tool_manager=getattr(server, "_tool_manager", None),
        tool_lookup=getattr(server, "get_tool", None)
        or getattr(getattr(server, "_tool_manager", None), "get_tool", None),
    )
    from . import _v1, _v2

    if _v2.instrument(observer, lowlevel) or _v1.instrument(observer, lowlevel):
        # Both adapters already skip handlers they wrapped, so a server that
        # can't be remembered here is still safe to instrument twice.
        try:
            _instrumented[lowlevel] = weakref.ref(options)
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
    "McpCaller",
    "McpRequestInfo",
    "instrument_mcp_server",
    "mark_mcp_call_error",
    "set_mcp_caller",
]
