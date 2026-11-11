"""Traces MCP requests, independent of which `mcp` version delivers them.

The version adapters turn a request into :class:`RequestFacts` and hand
:meth:`McpObserver.observe` a way to run it and to read and change its result.
Everything a span records, and every change made to a request or its result,
is decided here, so both `mcp` versions behave the same.
"""

from __future__ import annotations

import contextvars
import json
import re
import time
import uuid
import threading
from datetime import datetime, timezone
from weakref import WeakKeyDictionary
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from opentelemetry.trace import (
    INVALID_SPAN,
    Span,
    SpanKind,
    Status,
    StatusCode,
    set_span_in_context,
)
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

from judgeval.logger import judgeval_logger
from judgeval.version import get_version

from ._arguments import (
    AGENT_ARGUMENTS,
    CONVERSATION_ARGUMENT,
    ReportedArguments,
    ToolContract,
    conversation_id_prompt,
    mint_conversation_id,
    read_reported_arguments,
    schema_with_agent_arguments,
    tool_contract,
    without_agent_arguments,
)
from ._scrub import argument_values, scrub_error_message
from ._exporter import McpCall, McpCallExporter, OMITTED, RedactPayload, capture_payload

T = TypeVar("T")

TRACER_NAME = "judgeval-mcp"
CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
_MAX_ERROR_MESSAGE_LENGTH = 1_000
_MAX_RECORDED_KEYS = 30
_MAX_RECORDED_KEY_LENGTH = 64
_MAX_LISTING_PAGES = 20
_INVALID_PARAMS_CODE = -32602
# How each SDK words a call to a tool it doesn't have, or has turned off.
_UNKNOWN_TOOL = re.compile(
    r"^(?:MCP error -?\d+: )?"
    r"(?:Tool \S+ (?P<state>not found|disabled)|Unknown tool\b|Tool '[^']*' not found)"
)
# How each SDK words arguments that failed the tool's input schema: `mcp`'s
# JSON Schema check, and pydantic's errors for an `mcp` or `fastmcp` tool.
_INVALID_INPUT = re.compile(
    r"^(?:MCP error -?\d+: )?Input validation error"
    r"|\bvalidation errors? for (?:\S+Arguments\b|call\[)"
)
# Words only: anything with digits could be a token rather than an error code.
_PUBLIC_ERROR_CODE = re.compile(r"^[A-Za-z][A-Za-z_.:-]{0,39}$")
# Exception classes that say nothing about why a call failed.
_GENERIC_ERROR_NAMES = frozenset(
    {"Exception", "BaseException", "ToolError", "McpError", "MCPError"}
)

_propagator = TraceContextTextMapPropagator()


@dataclass(frozen=True)
class McpCaller:
    """Who a call acts for. Hash these if your identifiers are sensitive."""

    user_id: Optional[str] = None
    customer_id: Optional[str] = None


@dataclass(frozen=True)
class McpRequestInfo:
    """What ``identify`` can read about a request."""

    session_id: Optional[str]
    headers: Mapping[str, str]
    # The access token the `mcp` auth middleware validated, when there is one.
    auth_info: Any
    # The transport's own request object (a Starlette ``Request`` over HTTP).
    request: Any


Identify = Callable[[McpRequestInfo], Optional[McpCaller]]
Redact = Callable[[str], str]


@dataclass(frozen=True)
class McpOptions:
    server_name: Optional[str] = None
    server_build: Optional[str] = None
    inject_agent_arguments: bool = True
    identify: Optional[Identify] = None
    capture_error_messages: bool = True
    redact: Optional[Redact] = None
    capture_input: bool = False
    capture_output: bool = False
    redact_payload: Optional[RedactPayload] = None
    record_spans: bool = False
    session_id: Optional[Callable[[McpRequestInfo], Optional[str]]] = None


@dataclass(frozen=True)
class RequestFacts:
    """One inbound request, as both `mcp` versions can describe it."""

    method: str
    # Wire names: ``name``, ``arguments``, ``uri``, ``_meta``.
    params: Mapping[str, Any]
    request_id: Optional[str]
    session_id: Optional[str]
    headers: Mapping[str, str]
    client: Mapping[str, Any]
    protocol_version: Optional[str]
    auth_info: Any
    request: Any


ListedTools = Sequence[Tuple[str, Any]]


class ResultAdapter(Protocol):
    """Reads and changes one `mcp` version's result objects."""

    def is_error(self, result: Any) -> bool: ...

    def payload(self, result: Any) -> Any: ...

    def first_text(self, result: Any) -> Optional[str]: ...

    def with_text(self, result: Any, text: str, tool_name: Any) -> Any: ...

    def listed_tools(self, result: Any) -> Optional[ListedTools]: ...

    def with_tool_schemas(
        self, result: Any, schema_for: Callable[[str, Any], Optional[Any]]
    ) -> Any: ...


ListTools = Callable[[Optional[str]], Awaitable[Tuple[ListedTools, Optional[str]]]]


@dataclass
class _CallRecord:
    error: Optional[BaseException] = None
    caller: Dict[str, Optional[str]] = field(default_factory=dict)
    identified: bool = False
    failure: Optional[Tuple[str, Optional[str]]] = None
    argument_keys: List[str] = field(default_factory=list)
    environment: Optional[str] = None


@dataclass(frozen=True)
class _PreparedCall:
    attributes: Dict[str, str]
    # Replaces the call's arguments when injected ones were removed.
    arguments: Optional[Dict[str, Any]]
    minted_conversation_id: Optional[str]
    span_name: str


_records: contextvars.ContextVar[Optional[_CallRecord]] = contextvars.ContextVar(
    "judgeval_mcp_call", default=None
)
_exporters: WeakKeyDictionary[Any, McpCallExporter] = WeakKeyDictionary()
_exporters_lock = threading.Lock()


def mark_mcp_call_error(error: BaseException) -> None:
    """Records the original error before you turn it into a tool error.

    Its class name becomes the call's ``error.type``, so named exception
    classes tell you why calls fail.
    """
    record = _records.get()
    if record is not None:
        record.error = error


def set_mcp_caller(
    user_id: Optional[str] = None, customer_id: Optional[str] = None
) -> None:
    """Attaches the end user or customer of the current MCP call."""
    record = _records.get()
    if record is None:
        return
    if user_id is not None:
        record.caller["user_id"] = user_id
    if customer_id is not None:
        record.caller["customer_id"] = customer_id


def capture_mcp_call(
    call: McpCall, *, redact_payload: Optional[RedactPayload] = None
) -> None:
    """Capture a completed custom-server call. Supplied payloads opt in to capture."""

    def capture() -> None:
        exporter = _active_exporter()
        if exporter is None:
            return
        event = {**call, "call_id": call.get("call_id", str(uuid.uuid4()))}
        tracer = _active_tracer()
        if (
            "environment_name" not in event
            and tracer is not None
            and tracer.environment
        ):
            event["environment_name"] = _bounded(tracer.environment, 256)
        for column in ("input", "output"):
            if column in call:
                payload = capture_payload(call[column], column, redact_payload)
                event.pop(column, None)
                if payload is not OMITTED:
                    event[column] = payload
        exporter.enqueue(event)

    attempt(capture)


def attempt(work: Callable[[], T]) -> Optional[T]:
    """Runs tracing work so that a failure in it never reaches the MCP call."""
    try:
        return work()
    except Exception:
        judgeval_logger.debug("MCP telemetry operation failed")
        return None


async def attempt_async(work: Callable[[], Awaitable[T]]) -> Optional[T]:
    try:
        return await work()
    except Exception:
        judgeval_logger.debug("MCP telemetry operation failed")
        return None


def request_headers(request: Any) -> Dict[str, str]:
    """The HTTP headers of a request, with lowercase names. Empty over stdio."""
    headers = getattr(request, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(name).lower(): str(value) for name, value in headers.items()}
    except Exception:
        return {}


def transport_session_id(headers: Mapping[str, str], request: Any) -> Optional[str]:
    """The session a transport assigned: streamable HTTP's header, else SSE's.

    The SSE transport posts each message to a URL with a ``session_id`` query
    parameter.
    """
    session_id = headers.get("mcp-session-id")
    if not session_id:
        query = getattr(request, "query_params", None)
        session_id = attempt(lambda: query.get("session_id")) if query else None
    return session_id if isinstance(session_id, str) and session_id else None


def access_token() -> Any:
    """The access token the `mcp` auth middleware validated for this request."""
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token

        return get_access_token()
    except Exception:
        return None


def is_cancellation(error: BaseException) -> bool:
    try:
        import anyio

        if isinstance(error, anyio.get_cancelled_exc_class()):
            return True
    except Exception:
        pass
    import asyncio

    return isinstance(error, asyncio.CancelledError)


class McpObserver:
    """Traces the requests of one server and applies its injection settings."""

    def __init__(
        self,
        options: McpOptions,
        server_name: Optional[str],
        server_version: Optional[str],
    ) -> None:
        self.options = options
        self._server_name = options.server_name or server_name
        self._server_version = server_version
        self._contracts: Dict[str, Optional[ToolContract]] = {}
        # Requests can run where the tracer set by `Tracer.init()` isn't
        # active, such as a server running in another thread.
        self._tracer = attempt(_active_tracer)

    @property
    def inject(self) -> bool:
        return self.options.inject_agent_arguments

    def remember_listing(self, tools: Optional[ListedTools]) -> None:
        """Keeps what each listed tool declares, before injection changes it."""
        for name, schema in tools or ():
            self._contracts[name] = tool_contract(schema)

    async def contract_for(
        self, name: str, list_tools: Optional[ListTools]
    ) -> Optional[ToolContract]:
        """The tool's contract, listing the server's tools once if it's unknown.

        A call can reach a server that never listed its tools, such as another
        replica of a stateless server.
        """
        if name not in self._contracts and list_tools is not None:
            cursor: Optional[str] = None
            seen: Set[str] = set()
            for _ in range(_MAX_LISTING_PAGES):
                tools, cursor = await list_tools(cursor)
                self.remember_listing(tools)
                # A server that repeats a cursor would otherwise be listed again.
                if cursor is None or cursor in seen or name in self._contracts:
                    break
                seen.add(cursor)
        return self._contracts.get(name)

    def strip_known(self, method: str, params: Mapping[str, Any]) -> Optional[Dict]:
        """Removes injected arguments using only what is already known."""
        if method != "tools/call" or not self.inject:
            return None
        name = params.get("name")
        contract = self._contracts.get(name) if isinstance(name, str) else None
        arguments = params.get("arguments")
        if contract is None or contract.open or not isinstance(arguments, Mapping):
            return None
        return without_agent_arguments(arguments, contract.declared)

    async def observe(
        self,
        facts: RequestFacts,
        invoke: Callable[[Optional[Dict[str, Any]]], Awaitable[Any]],
        adapter: ResultAdapter,
        list_tools: Optional[ListTools],
    ) -> Any:
        """Runs one request inside a span. Tracing failures never reach the call."""
        token = attempt(self._activate_tracer)
        try:
            return await self._observe(facts, invoke, adapter, list_tools)
        finally:
            if token is not None:
                attempt(lambda: _active_tracer_var().reset(token))

    def _activate_tracer(self) -> Optional[contextvars.Token[Any]]:
        """Uses the tracer active at instrumentation if the request has none."""
        variable = _active_tracer_var()
        if self._tracer is None or variable.get() is not None:
            return None
        return variable.set(self._tracer)

    async def _observe(
        self,
        facts: RequestFacts,
        invoke: Callable[[Optional[Dict[str, Any]]], Awaitable[Any]],
        adapter: ResultAdapter,
        list_tools: Optional[ListTools],
    ) -> Any:
        prepared = await attempt_async(lambda: self._prepare(facts, list_tools))
        if prepared is None:
            arguments = attempt(lambda: self.strip_known(facts.method, facts.params))
            result = await invoke(arguments)
            return self._shaped(facts, result, adapter, None)

        record = _CallRecord()
        span = (
            attempt(lambda: self._start_span(facts, prepared))
            if self.options.record_spans
            else None
        )
        exporter = attempt(_active_exporter) if facts.method == "tools/call" else None
        arguments = (
            prepared.arguments
            if prepared.arguments is not None
            else facts.params.get("arguments", {})
        )
        captured = (
            capture_payload(arguments, "input", self.options.redact_payload)
            if self.options.capture_input and exporter is not None
            else OMITTED
        )
        record.argument_keys = (
            [_bounded(str(key), 256) for key in arguments if key][:100]
            if isinstance(arguments, Mapping)
            else []
        )
        tracer = _active_tracer()
        record.environment = tracer.environment if tracer is not None else None
        timestamp = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()

        def complete(result: Any, cancelled: bool) -> None:
            if facts.method == "tools/call" and exporter is not None:
                attempt(
                    lambda: self._export_call(
                        exporter,
                        facts,
                        prepared,
                        record,
                        result,
                        adapter,
                        span,
                        timestamp,
                        started,
                        captured,
                        cancelled,
                    )
                )

        token = _records.set(record)
        try:
            result = await self._run(span, lambda: invoke(prepared.arguments))
        except BaseException as error:
            cancelled = is_cancellation(error)
            if record.error is None and not cancelled:
                record.error = error
            complete(None, cancelled)
            if span is not None:
                self._finish(span, record, None, facts, adapter, cancelled)
            raise
        finally:
            _records.reset(token)
        complete(result, False)
        if span is not None:
            self._finish(span, record, result, facts, adapter, False)
        return self._shaped(facts, result, adapter, prepared.minted_conversation_id)

    def _export_call(
        self,
        exporter: McpCallExporter,
        facts: RequestFacts,
        prepared: _PreparedCall,
        record: _CallRecord,
        result: Any,
        adapter: ResultAdapter,
        span: Optional[Span],
        timestamp: str,
        started: float,
        captured: Any,
        cancelled: bool,
    ) -> None:
        duration = max(0, (time.monotonic() - started) * 1000)
        attributes = prepared.attributes
        caller = self._caller(record, facts)
        failed = result is None or adapter.is_error(result)
        call: Dict[str, Any] = {
            "call_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "duration_ms": duration,
            "outcome": "cancelled" if cancelled else "error" if failed else "success",
            "server_name": _bounded(self._server_name or "mcp", 256),
            "tool_name": _bounded(str(facts.params.get("name") or "unknown"), 256),
            "argument_keys": record.argument_keys,
            "metadata": {
                key: _bounded(value, 512)
                for key, value in attributes.items()
                if key
                in {
                    "mcp.session.id",
                    "mcp.protocol.version",
                    "jsonrpc.request.id",
                    "judgment.mcp.server.version",
                    "judgment.mcp.server.build",
                    "judgment.mcp.client.version",
                    "judgment.mcp.sdk_version",
                    "judgment.mcp.conversation_id_source",
                }
            },
        }
        for column, key, limit in (
            ("session_id", "judgment.mcp.conversation_id", 512),
            ("client_name", "judgment.mcp.client.name", 256),
            ("intent", "judgment.mcp.intent", 2048),
            ("model", "judgment.mcp.model", 256),
        ):
            if key in attributes:
                call[column] = _bounded(attributes[key], limit)
        user_id = caller.get("user_id")
        if user_id:
            call["user_id"] = _bounded(user_id, 512)
        customer_id = caller.get("customer_id")
        if customer_id:
            call["metadata"]["customer_id"] = _bounded(customer_id, 512)
        if record.environment:
            call["environment_name"] = _bounded(record.environment, 256)
        if failed and not cancelled:
            error_type, recorded = self._failure(record, result, facts, adapter)
            call["error_type"] = _bounded(error_type, 256)
            if recorded is not None:
                call["error_message"] = recorded
        output = (
            capture_payload(
                adapter.payload(result), "output", self.options.redact_payload
            )
            if result is not None and self.options.capture_output
            else OMITTED
        )
        for column, payload in (("input", captured), ("output", output)):
            if payload is not OMITTED:
                call[column] = payload
        if self.options.capture_input and captured is OMITTED:
            call["metadata"]["input_omitted"] = True
        if self.options.capture_output and result is not None and output is OMITTED:
            call["metadata"]["output_omitted"] = True
        if span is not None and span.get_span_context().is_valid:
            context = span.get_span_context()
            call.update(
                trace_id=f"{context.trace_id:032x}", span_id=f"{context.span_id:016x}"
            )
        exporter.enqueue(call)

    async def _run(
        self, span: Optional[Span], call: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Runs the request with its span current, so the tool's spans nest."""
        scope = None if span is None else attempt(lambda: _span_scope(span))
        if scope is None:
            return await call()
        with scope:
            return await call()

    def _shaped(
        self,
        facts: RequestFacts,
        result: Any,
        adapter: ResultAdapter,
        minted_conversation_id: Optional[str],
    ) -> Any:
        shaped = attempt(
            lambda: self._shape_result(facts, result, adapter, minted_conversation_id)
        )
        return result if shaped is None else shaped

    def _shape_result(
        self,
        facts: RequestFacts,
        result: Any,
        adapter: ResultAdapter,
        minted_conversation_id: Optional[str],
    ) -> Any:
        if facts.method == "tools/list":
            return self.shape_listing(result, adapter)
        if minted_conversation_id is not None:
            prompt = conversation_id_prompt(minted_conversation_id)
            shaped = attempt(
                lambda: adapter.with_text(result, prompt, facts.params.get("name"))
            )
            return result if shaped is None else shaped
        return result

    def shape_listing(self, result: Any, adapter: ResultAdapter) -> Any:
        self.remember_listing(attempt(lambda: adapter.listed_tools(result)))
        if not self.inject:
            return result
        shaped = attempt(lambda: adapter.with_tool_schemas(result, self._listed_schema))
        return result if shaped is None else shaped

    def _listed_schema(self, name: str, schema: Any) -> Optional[Any]:
        contract = self._contracts.get(name)
        if contract is None or contract.open or not isinstance(schema, Mapping):
            return None
        return schema_with_agent_arguments(schema, contract.declared)

    async def _prepare(
        self, facts: RequestFacts, list_tools: Optional[ListTools]
    ) -> _PreparedCall:
        method = facts.method
        params = facts.params
        meta = _mapping(params.get("_meta"))
        raw_arguments = params.get("arguments")
        args: Mapping[str, Any] = (
            raw_arguments if isinstance(raw_arguments, Mapping) else {}
        )
        name = params.get("name") if isinstance(params.get("name"), str) else None
        tool_name = name if method == "tools/call" else None

        contract: Optional[ToolContract] = None
        if tool_name is not None and self.inject:
            contract = await attempt_async(
                lambda: self.contract_for(tool_name, list_tools)
            )
        injected = contract is not None and not contract.open
        declared: FrozenSet[str] = contract.declared if contract else frozenset()
        reported = (
            read_reported_arguments(args, declared)
            if injected
            else ReportedArguments(None, None, None)
        )
        session_id = self.options.session_id
        logical_session = (
            attempt(
                lambda: session_id(
                    McpRequestInfo(
                        facts.session_id, facts.headers, facts.auth_info, facts.request
                    )
                )
            )
            if session_id is not None
            else None
        ) or reported.conversation_id
        minted = (
            mint_conversation_id()
            if injected
            # A tool's own conversation_id argument isn't one the agent can echo.
            and CONVERSATION_ARGUMENT not in declared
            and logical_session is None
            else None
        )
        client = (
            _mapping(meta.get(CLIENT_INFO_META_KEY))
            or facts.client
            # The session learns who the client is only once this succeeds.
            or (_mapping(params.get("clientInfo")) if method == "initialize" else {})
        )
        intent = None if reported.intent is None else self._redacted(reported.intent)
        uri = params.get("uri")
        attributes = _defined(
            {
                "mcp.method.name": method,
                "mcp.protocol.version": _text(meta.get(PROTOCOL_VERSION_META_KEY))
                or facts.headers.get("mcp-protocol-version")
                or facts.protocol_version,
                "mcp.session.id": facts.session_id,
                "jsonrpc.request.id": facts.request_id,
                "gen_ai.operation.name": "execute_tool" if tool_name else None,
                "gen_ai.tool.name": tool_name,
                "gen_ai.prompt.name": name if method == "prompts/get" else None,
                "mcp.resource.uri": self._redacted(_without_uri_secrets(uri))
                if method == "resources/read" and isinstance(uri, str)
                else None,
                "user_agent.original": facts.headers.get("user-agent"),
                "judgment.mcp.server.name": self._server_name,
                "judgment.mcp.server.version": self._server_version,
                "judgment.mcp.server.build": self.options.server_build,
                "judgment.mcp.client.name": _text(client.get("name")),
                "judgment.mcp.client.version": _text(client.get("version")),
                "judgment.mcp.argument_keys": _recorded_keys(
                    args, declared if injected else None
                )
                if tool_name
                else None,
                "judgment.mcp.conversation_id": logical_session or minted,
                "judgment.mcp.conversation_id_source": "provided"
                if logical_session and logical_session != reported.conversation_id
                else "echoed"
                if reported.conversation_id
                else "minted"
                if minted
                else None,
                "judgment.mcp.intent": intent,
                "judgment.mcp.model": reported.model,
                "judgment.mcp.sdk_version": get_version(),
            }
        )
        target = name if method in ("tools/call", "prompts/get") else None
        return _PreparedCall(
            attributes=attributes,
            arguments=without_agent_arguments(args, declared)
            if injected and any(key in AGENT_ARGUMENTS for key in args)
            else None,
            minted_conversation_id=minted,
            span_name=f"{method} {target}" if target else method,
        )

    def _start_span(self, facts: RequestFacts, prepared: _PreparedCall) -> Span:
        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

        provider = JudgmentTracerProvider.get_instance()
        meta = _mapping(facts.params.get("_meta"))
        # Only the caller's ``_meta`` trace context can be the parent. A span
        # that happens to be current, such as one around the whole server or
        # the HTTP request that opened the session, would otherwise put every
        # call into one trace. Baggage is kept.
        current = set_span_in_context(INVALID_SPAN, provider.get_current_context())
        parent = (
            _propagator.extract(
                {key: value for key, value in meta.items() if isinstance(value, str)},
                context=current,
            )
            if isinstance(meta.get("traceparent"), str)
            else current
        )
        return provider.get_tracer(TRACER_NAME).start_span(
            prepared.span_name,
            context=parent,
            kind=SpanKind.SERVER,
            attributes=prepared.attributes,
        )

    def _finish(
        self,
        span: Span,
        record: _CallRecord,
        result: Any,
        facts: RequestFacts,
        adapter: ResultAdapter,
        cancelled: bool,
    ) -> None:
        def annotate() -> None:
            caller = self._caller(record, facts)
            span.set_attributes(
                _defined(
                    {
                        "enduser.id": caller.get("user_id"),
                        "judgment.customer_id": caller.get("customer_id"),
                    }
                )
            )
            if cancelled:
                span.set_attribute("judgment.mcp.cancelled", "true")
            elif result is None or adapter.is_error(result):
                self._record_failure(span, record, result, facts, adapter)

        attempt(annotate)
        attempt(span.end)

    def _caller(
        self, record: _CallRecord, facts: RequestFacts
    ) -> Dict[str, Optional[str]]:
        if not record.identified:
            record.caller = {**self._identify(facts), **record.caller}
            record.identified = True
        return record.caller

    def _identify(self, facts: RequestFacts) -> Dict[str, Optional[str]]:
        identify = self.options.identify
        if identify is None:
            return {}
        info = McpRequestInfo(
            session_id=facts.session_id,
            headers=facts.headers,
            auth_info=facts.auth_info,
            request=facts.request,
        )
        caller = attempt(lambda: identify(info))
        if not isinstance(caller, McpCaller):
            return {}
        return {
            key: value
            for key, value in (
                ("user_id", caller.user_id),
                ("customer_id", caller.customer_id),
            )
            if isinstance(value, str) and value
        }

    def _record_failure(
        self,
        span: Span,
        record: _CallRecord,
        result: Any,
        facts: RequestFacts,
        adapter: ResultAdapter,
    ) -> None:
        error_type, recorded = self._failure(record, result, facts, adapter)
        span.set_attribute("error.type", error_type)
        span.set_status(Status(StatusCode.ERROR, recorded))

    def _failure(
        self,
        record: _CallRecord,
        result: Any,
        facts: RequestFacts,
        adapter: ResultAdapter,
    ) -> Tuple[str, Optional[str]]:
        if record.failure is not None:
            return record.failure
        error = record.error
        message = (
            attempt(lambda: adapter.first_text(result)) if result is not None else None
        )
        if message is None and error is not None:
            message = _error_message(error)
        recorded = (
            self._recorded_message(message, facts.params)
            if self.options.capture_error_messages and message is not None
            else None
        )
        record.failure = (
            self._error_type(error, result is not None, message),
            None if recorded is None else recorded[:_MAX_ERROR_MESSAGE_LENGTH],
        )
        return record.failure

    def _recorded_message(
        self, message: str, params: Mapping[str, Any]
    ) -> Optional[str]:
        """The failure text a span keeps. None if it can't be made safe to keep."""
        scrubbed = attempt(
            lambda: scrub_error_message(message, argument_values(params))
        )
        return None if scrubbed is None else self._redacted(scrubbed)

    def _error_type(
        self, error: Optional[BaseException], has_result: bool, message: Optional[str]
    ) -> str:
        """A short, low-cardinality category for why a call failed.

        A refusal before the handler ran, else the original error's class, else
        a structured error code the tool returned, else the generic kind.
        """
        code = _error_code(error)
        unknown = _UNKNOWN_TOOL.search(message) if message is not None else None
        if unknown is not None:
            return (
                "tool_disabled"
                if unknown.group("state") == "disabled"
                else "unknown_tool"
            )
        if (
            code == _INVALID_PARAMS_CODE
            or (message is not None and _INVALID_INPUT.search(message) is not None)
            or (error is not None and type(error).__name__ == "ValidationError")
        ):
            return "invalid_input"
        name = type(error).__name__ if error is not None else None
        if name is not None and name not in _GENERIC_ERROR_NAMES:
            return name
        if self.options.capture_error_messages:
            public = _public_error_code(message)
            returned = None if public is None else self._redacted(public)
            if returned:
                return returned
        if has_result:
            return "tool_error"
        if isinstance(code, int):
            return str(code)
        return name or "_OTHER"

    def _redacted(self, text: str) -> Optional[str]:
        """Applies the builder's redaction. If it fails, the text isn't sent."""
        redact = self.options.redact
        if redact is None:
            return text
        value = attempt(lambda: redact(text))
        return value if isinstance(value, str) else None


def _span_scope(span: Span) -> Any:
    from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

    return JudgmentTracerProvider.get_instance().use_span(
        span,
        end_on_exit=False,
        record_exception=False,
        set_status_on_exception=False,
    )


def _active_tracer_var() -> contextvars.ContextVar[Any]:
    from judgeval.trace.judgment_tracer_provider import _active_tracer_var

    return _active_tracer_var


def _active_tracer() -> Any:
    return _active_tracer_var().get()


def _bounded(value: str, limit: int) -> str:
    """Match the native contract's JavaScript (UTF-16) string limits."""
    return value.encode("utf-16-le")[: limit * 2].decode("utf-16-le", errors="ignore")


def _active_exporter() -> Optional[McpCallExporter]:
    from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

    tracer = _active_tracer()
    if (
        tracer is None
        or not tracer._enable_monitoring
        or not all(
            (tracer.api_url, tracer.api_key, tracer.organization_id, tracer.project_id)
        )
    ):
        return None
    with _exporters_lock:
        exporter = _exporters.get(tracer)
        if exporter is None:
            exporter = McpCallExporter(
                tracer.api_url,
                tracer.api_key,
                tracer.organization_id,
                tracer.project_id,
            )
            _exporters[tracer] = exporter
            JudgmentTracerProvider.get_instance().add_background_exporter(exporter)
        return exporter


def _error_code(error: Optional[BaseException]) -> Any:
    if error is None:
        return None
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(getattr(error, "error", None), "code", None)
    return code


def _error_message(error: BaseException) -> Optional[str]:
    inner = getattr(error, "error", None)
    message = getattr(inner, "message", None)
    if isinstance(message, str) and message:
        return message
    text = str(error)
    return text or None


def _public_error_code(message: Optional[str]) -> Optional[str]:
    """Reads the ``error`` field of a structured error a tool returned as JSON."""
    if message is None or not message.startswith("{"):
        return None
    try:
        parsed = json.loads(message)
    except ValueError:
        return None
    code = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(code, str) and _PUBLIC_ERROR_CODE.match(code):
        return code
    return None


def _without_uri_secrets(uri: str) -> str:
    """Drops URI parts that commonly carry credentials: user info, query, fragment."""
    uri = re.sub(r"^([a-z][a-z\d+.-]*://)[^/@?#]*@", r"\1", uri, flags=re.IGNORECASE)
    return re.sub(r"[?#].*$", "", uri)


def _recorded_keys(
    args: Mapping[str, Any], declared: Optional[FrozenSet[str]]
) -> Optional[str]:
    """Lists a call's argument names, leaving out injected ones."""
    keys: List[str] = [
        key[:_MAX_RECORDED_KEY_LENGTH]
        for key in args
        if isinstance(key, str)
        and (declared is None or key not in AGENT_ARGUMENTS or key in declared)
    ][:_MAX_RECORDED_KEYS]
    return ",".join(keys) if keys else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _defined(attributes: Mapping[str, Optional[str]]) -> Dict[str, str]:
    return {
        key: value
        for key, value in attributes.items()
        if isinstance(value, str) and value != ""
    }


__all__ = [
    "McpCaller",
    "McpObserver",
    "McpOptions",
    "McpRequestInfo",
    "RequestFacts",
    "ResultAdapter",
    "attempt",
    "mark_mcp_call_error",
    "set_mcp_caller",
]
