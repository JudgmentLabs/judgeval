"""Traces MCP calls, independent of which `mcp` version delivers them.

The version adapters turn a request into :class:`RequestFacts` and hand
:meth:`McpObserver.observe` a way to run it and to read and change its result.
Everything a call records, and every change made to a request or its result,
is decided here, so both `mcp` versions behave the same.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
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
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.logger import judgeval_logger
from judgeval.version import get_version

from ._arguments import (
    AGENT_ARGUMENTS,
    ArgumentNames,
    ToolContract,
    conversation_output_schema,
    conversation_id_prompt,
    mint_conversation_id,
    read_reported_arguments,
    schema_with_agent_arguments,
    tool_contract,
    without_agent_arguments,
)
from ._payload import OMITTED, RedactPayload, capture_payload

_propagator = TraceContextTextMapPropagator()
TRACER_NAME = "judgeval-mcp"

T = TypeVar("T")

CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
_MAX_ERROR_MESSAGE_LENGTH = 1_000
_MAX_RECORDED_KEYS = 30
_MAX_RECORDED_KEY_LENGTH = 64
_MAX_DISCOVERY_PAGES = 100
_DISCOVERY_TIMEOUT_SECONDS = 1.0
_MISSING_TOOL_CACHE_SECONDS = 1.0
_MAX_CACHED_TOOLS = 1_000
_MAX_DISCOVERY_ATTEMPTS = 2
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


class OriginalTool(Protocol):
    @property
    def inputSchema(self) -> Mapping[str, Any]: ...


class OriginalToolV2(Protocol):
    @property
    def input_schema(self) -> Mapping[str, Any]: ...


ResolveOriginalTool = Callable[
    [str],
    Optional[Mapping[str, Any] | OriginalTool | OriginalToolV2]
    | Awaitable[Optional[Mapping[str, Any] | OriginalTool | OriginalToolV2]],
]


@dataclass(frozen=True)
class McpOptions:
    server_name: Optional[str] = None
    server_build: Optional[str] = None
    inject_agent_arguments: bool = True
    inject_conversation_id: bool = True
    intent_argument: Optional[str] = None
    model_argument: Optional[str] = None
    conversation_id_argument: Optional[str] = None
    identify: Optional[Identify] = None
    capture_error_messages: bool = False
    redact: Optional[Redact] = None
    capture_input: bool = False
    capture_output: bool = False
    redact_payload: Optional[RedactPayload] = None
    session_id: Optional[Callable[[McpRequestInfo], Optional[str]]] = None
    resolve_original_tool: Optional[ResolveOriginalTool] = None


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


ListedTools = Sequence[Tuple[str, Any, Any]]


class ResultAdapter(Protocol):
    """Reads and changes one `mcp` version's result objects."""

    def is_error(self, result: Any) -> bool: ...

    def payload(self, result: Any) -> Any: ...

    def first_text(self, result: Any) -> Optional[str]: ...

    def with_text(
        self,
        result: Any,
        text: str,
        contract: Optional[ToolContract],
    ) -> Any: ...

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


@dataclass(frozen=True)
class _PreparedCall:
    attributes: Dict[str, str]
    # Replaces the call's arguments when injected ones were removed.
    arguments: Optional[Dict[str, Any]]
    minted_conversation_id: Optional[str]


_records: contextvars.ContextVar[Optional[_CallRecord]] = contextvars.ContextVar(
    "judgeval_mcp_call", default=None
)


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


@dataclass
class _CachedContract:
    value: Optional[ToolContract] = None
    pending: Optional[asyncio.Task[Optional[ToolContract]]] = None
    expires_at: float = float("inf")


class _ToolDiscoveryCache:
    """Bounds server-local contracts and coalesces cold discovery attempts."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, _CachedContract] = OrderedDict()
        self._generation = 0

    def clear(self) -> None:
        self._generation += 1
        self._entries.clear()

    def remember(self, name: str, contract: Optional[ToolContract]) -> None:
        self._store(name, _CachedContract(value=contract))

    def _store(self, name: str, entry: _CachedContract) -> None:
        self._entries.pop(name, None)
        self._entries[name] = entry
        if len(self._entries) > _MAX_CACHED_TOOLS:
            self._entries.popitem(last=False)

    async def resolve(
        self,
        name: str,
        lookup: Callable[[], Coroutine[Any, Any, Optional[ToolContract]]],
    ) -> Optional[ToolContract]:
        for _ in range(_MAX_DISCOVERY_ATTEMPTS):
            generation = self._generation
            entry = self._entries.get(name)
            if entry is None or entry.expires_at <= time.monotonic():
                entry = _CachedContract(pending=asyncio.create_task(lookup()))
                self._store(name, entry)
            if entry.pending is not None:
                # One cancelled request must not cancel discovery for other waiters.
                entry.value = await asyncio.shield(entry.pending)
                entry.pending = None
                if entry.value is None:
                    entry.expires_at = time.monotonic() + _MISSING_TOOL_CACHE_SECONDS
            current = self._entries.get(name)
            # Eviction affects retention, not ownership for the waiting call.
            if generation == self._generation and (current is None or current is entry):
                return entry.value
            # Prefer a newer listing, or retry once after a schema invalidation.
        return None


class McpObserver:
    """Traces the requests of one server and applies its injection settings."""

    def __init__(
        self,
        options: McpOptions,
        server_name: Optional[str],
        server_version: Optional[str],
        tool_manager: Any = None,
        tool_lookup: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.options = options
        self.argument_names = ArgumentNames(
            options.intent_argument,
            options.model_argument,
            options.conversation_id_argument,
        )
        self._server_name = options.server_name or server_name
        self._server_version = server_version
        self._request_contracts: contextvars.ContextVar[
            Optional[Dict[str, Optional[ToolContract]]]
        ] = contextvars.ContextVar("mcp_tool_contracts", default=None)
        self._listed_contracts = _ToolDiscoveryCache()
        self._tool_manager = tool_manager
        self._tool_lookup = tool_lookup
        # Requests can run where the tracer set by `Tracer.init()` isn't
        # active, such as a server running in another thread.
        self._tracer = attempt(_active_tracer)

    @property
    def inject(self) -> bool:
        return self.options.inject_agent_arguments

    def invalidate_tools(self) -> None:
        self._listed_contracts.clear()

    def watch_tool_changes(self, target: Any) -> None:
        """Invalidates contracts when this session advertises changed tools."""
        method = (
            "send_notification" if hasattr(target, "send_notification") else "notify"
        )
        original = getattr(target, method, None)
        if not callable(original) or getattr(original, "__judgment_tools__", False):
            return

        async def notify(*args: Any, **kwargs: Any) -> Any:
            notification = (
                args[0] if args else kwargs.get("notification", kwargs.get("method"))
            )
            root = getattr(notification, "root", notification)
            name = root if isinstance(root, str) else getattr(root, "method", None)
            if name == "notifications/tools/list_changed":
                self.invalidate_tools()
            return await original(*args, **kwargs)

        setattr(notify, "__judgment_tools__", True)
        setattr(target, method, notify)

    @property
    def _contracts(self) -> Dict[str, Optional[ToolContract]]:
        contracts = self._request_contracts.get()
        if contracts is None:
            contracts = {}
            self._request_contracts.set(contracts)
        return contracts

    def remember_listing(self, tools: Optional[ListedTools]) -> None:
        """Keeps what each listed tool declares, before injection changes it."""
        for name, schema, output in tools or ():
            tool = getattr(self._tool_manager, "_tools", {}).get(name)
            self._remember_tool(name, schema, tool, output)

    def _remember_tool(self, name: str, schema: Any, tool: Any, output: Any) -> None:
        model = getattr(getattr(tool, "fn_metadata", None), "arg_model", None)
        function = getattr(tool, "fn", None)
        closed = (
            model.model_config.get("extra") != "allow"
            if model is not None
            else callable(function)
            and all(
                parameter.kind != inspect.Parameter.VAR_KEYWORD
                for parameter in inspect.signature(function).parameters.values()
            )
        )
        # Python function tools ignore extras by default; raw JSON-schema
        # tools must explicitly declare closed inputs.
        if closed and isinstance(schema, Mapping):
            schema = {**schema, "additionalProperties": False}
        contract = tool_contract(schema, self.argument_names)
        if contract is not None:
            contract = replace(
                contract,
                structured_output=output is None
                or conversation_output_schema(output) is not None,
            )
        self._contracts[name] = contract

    async def _remember_native_tool(self, name: str) -> None:
        if self._tool_lookup is not None:
            tool = self._tool_lookup(name)
            if inspect.isawaitable(tool):
                tool = await tool
            schema = getattr(tool, "parameters", getattr(tool, "input_schema", None))
            if schema is not None:
                self._remember_tool(
                    name, schema, tool, getattr(tool, "output_schema", None)
                )

    async def contract_for(
        self, name: str, list_tools: Optional[ListTools]
    ) -> Optional[ToolContract]:
        """Uses advertised contracts first, with bounded lookup on fresh replicas."""
        if name not in self._contracts:
            await attempt_async(lambda: self._remember_native_tool(name))
        if name not in self._contracts:

            async def lookup() -> Optional[ToolContract]:
                return await self._lookup_contract(name, list_tools)

            self._contracts[name] = (
                await lookup()
                if self.options.resolve_original_tool is not None
                or self._tool_lookup is not None
                else await self._listed_contracts.resolve(name, lookup)
            )
        return self._contracts.get(name)

    async def _lookup_contract(
        self, name: str, list_tools: Optional[ListTools]
    ) -> Optional[ToolContract]:
        token = self._request_contracts.set({})
        try:
            await attempt_async(
                lambda: asyncio.wait_for(
                    self._discover_contract(name, list_tools),
                    timeout=_DISCOVERY_TIMEOUT_SECONDS,
                )
            )
            return self._contracts.get(name)
        finally:
            self._request_contracts.reset(token)

    async def _discover_contract(
        self, name: str, list_tools: Optional[ListTools]
    ) -> None:
        resolver = self.options.resolve_original_tool
        if name not in self._contracts and resolver is not None:
            tool = resolver(name)
            if inspect.isawaitable(tool):
                tool = await tool
            schema = (
                tool.get("inputSchema")
                if isinstance(tool, Mapping)
                else getattr(tool, "input_schema", getattr(tool, "inputSchema", None))
            )
            contract = tool_contract(schema, self.argument_names)
            if contract is not None:
                output = (
                    tool.get("outputSchema")
                    if isinstance(tool, Mapping)
                    else getattr(
                        tool, "output_schema", getattr(tool, "outputSchema", None)
                    )
                )
                self._contracts[name] = replace(
                    contract,
                    structured_output=output is None
                    or conversation_output_schema(output) is not None,
                )
            return
        if name not in self._contracts and list_tools is not None:
            cursor: Optional[str] = None
            seen: Set[str] = set()
            for _ in range(_MAX_DISCOVERY_PAGES):
                await asyncio.sleep(0)
                tools, cursor = await list_tools(cursor)
                self.remember_listing(tools)
                # A server that repeats a cursor would otherwise be listed again.
                if cursor is None or cursor in seen or name in self._contracts:
                    break
                seen.add(cursor)

    def strip_known(self, method: str, params: Mapping[str, Any]) -> Optional[Dict]:
        """Removes injected arguments using only what is already known."""
        if method != "tools/call" or not self.inject:
            return None
        name = params.get("name")
        contract = self._contracts.get(name) if isinstance(name, str) else None
        arguments = params.get("arguments")
        if contract is None or contract.open or not isinstance(arguments, Mapping):
            return None
        return without_agent_arguments(
            arguments,
            contract.declared,
            self.argument_names,
            self.options.inject_conversation_id,
        )

    async def observe(
        self,
        facts: RequestFacts,
        invoke: Callable[[Optional[Dict[str, Any]]], Awaitable[Any]],
        adapter: ResultAdapter,
        list_tools: Optional[ListTools],
    ) -> Any:
        """Records a request without letting telemetry failures affect the call."""
        if facts.method != "tools/call":
            result = await invoke(None)
            return self._shaped(facts, result, adapter, None)
        token = attempt(self._activate_tracer)
        contracts_token = self._request_contracts.set({})
        try:
            return await self._observe(facts, invoke, adapter, list_tools)
        finally:
            self._request_contracts.reset(contracts_token)
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
        span = attempt(lambda: self._start_span(facts, prepared))
        arguments = (
            prepared.arguments
            if prepared.arguments is not None
            else facts.params.get("arguments", {})
        )
        if (
            span is not None
            and self.options.capture_input
            and facts.method == "tools/call"
        ):
            attempt(lambda: self._payload(span, arguments, "input"))
        token = _records.set(record)
        try:
            result = await self._run(
                span,
                lambda: invoke(prepared.arguments),
                prepared.attributes.get("judgment.session_id"),
            )
        except BaseException as error:
            cancelled = is_cancellation(error)
            if record.error is None and not cancelled:
                record.error = error
            if span is not None:
                self._finish(span, record, None, facts, adapter, cancelled)
            raise
        finally:
            _records.reset(token)
        if span is not None:
            self._finish(span, record, result, facts, adapter, False)
        return self._shaped(facts, result, adapter, prepared.minted_conversation_id)

    async def _run(
        self,
        span: Optional[Span],
        call: Callable[[], Awaitable[Any]],
        session_id: Optional[str],
    ) -> Any:
        """Runs the request with its span current, so the tool's spans nest."""
        scope = None if span is None else attempt(lambda: _span_scope(span, session_id))
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
            prompt = conversation_id_prompt(
                minted_conversation_id, self.argument_names.names[2]
            )
            name = facts.params.get("name")
            contract = self._contracts.get(name) if isinstance(name, str) else None
            shaped = attempt(
                lambda: adapter.with_text(
                    result,
                    prompt,
                    contract,
                )
            )
            return result if shaped is None else shaped
        return result

    def shape_listing(self, result: Any, adapter: ResultAdapter) -> Any:
        tools = attempt(lambda: adapter.listed_tools(result))
        self.remember_listing(tools)
        for name, _, _ in tools or ():
            self._listed_contracts.remember(name, self._contracts.get(name))
        if not self.inject:
            return result
        shaped = attempt(lambda: adapter.with_tool_schemas(result, self._listed_schema))
        return result if shaped is None else shaped

    def _listed_schema(self, name: str, schema: Any) -> Optional[Any]:
        contract = self._contracts.get(name)
        if contract is None or contract.open or not isinstance(schema, Mapping):
            return None
        return schema_with_agent_arguments(
            schema,
            contract.declared,
            self.options.inject_conversation_id,
            self.argument_names,
        )

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
        if tool_name is not None and (
            self.inject
            or any(
                (
                    self.options.intent_argument,
                    self.options.model_argument,
                    self.options.conversation_id_argument,
                )
            )
        ):
            contract = await attempt_async(
                lambda: self.contract_for(tool_name, list_tools)
            )
        injected = self.inject and contract is not None and not contract.open
        declared: FrozenSet[str] = contract.declared if contract else frozenset()
        reported = read_reported_arguments(
            args, declared, self.argument_names, injected=injected
        )
        session_id = self.options.session_id
        provided_session = (
            attempt(
                lambda: session_id(
                    McpRequestInfo(
                        facts.session_id, facts.headers, facts.auth_info, facts.request
                    )
                )
            )
            if session_id is not None
            else None
        )
        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
        from judgeval.trace.baggage import get_baggage

        configured_session = (
            _text(provided_session)
            or _text(
                get_baggage(
                    AttributeKeys.JUDGMENT_SESSION_ID.value,
                    context=JudgmentTracerProvider.get_instance().get_current_context(),
                )
            )
            or (
                reported.conversation_id
                if self.options.conversation_id_argument is not None
                and (self.argument_names.names[2] in declared or not injected)
                else None
            )
        )
        conversation_id = (
            reported.conversation_id if self.options.inject_conversation_id else None
        )
        logical_session = configured_session or conversation_id
        minted = (
            mint_conversation_id()
            if injected
            and self.options.inject_conversation_id
            # A tool's own conversation_id argument isn't one the agent can echo.
            and self.argument_names.names[2] not in declared
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
                "judgment.mcp.server_name": self._server_name or "mcp",
                "judgment.mcp.tool_name": tool_name,
                "judgment.session_id": logical_session or minted or facts.session_id,
                "judgment.mcp.server.version": self._server_version,
                "judgment.mcp.server.build": self.options.server_build,
                "judgment.mcp.client_name": _text(client.get("name")),
                "judgment.mcp.client.version": _text(client.get("version")),
                "judgment.mcp.argument_keys": _recorded_keys(
                    args,
                    declared if injected else None,
                    self.argument_names.names
                    if self.options.inject_conversation_id
                    else self.argument_names.names[:2],
                )
                if tool_name
                else None,
                "judgment.mcp.conversation_id": logical_session or minted,
                "judgment.mcp.conversation_id_source": "provided"
                if configured_session
                else "echoed"
                if conversation_id
                else "minted"
                if minted
                else "transport"
                if facts.session_id
                else None,
                "judgment.mcp.intent": intent,
                "judgment.llm.model": reported.model,
                "judgment.mcp.sdk_version": get_version(),
            }
        )
        return _PreparedCall(
            attributes=attributes,
            arguments=without_agent_arguments(
                args, declared, self.argument_names, self.options.inject_conversation_id
            )
            if injected and any(key in self.argument_names.names for key in args)
            else None,
            minted_conversation_id=minted,
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
        from judgeval.trace.baggage import set_baggage

        session = prepared.attributes.get("judgment.session_id")
        if session:
            parent = set_baggage("judgment.session_id", session, context=parent)
        return provider.get_tracer(TRACER_NAME).start_span(
            f"{facts.method} {facts.params['name']}"
            if isinstance(facts.params.get("name"), str)
            else facts.method,
            context=parent,
            kind=SpanKind.SERVER,
            attributes=prepared.attributes,
        )

    def _payload(self, span: Span, value: Any, field: Any) -> None:
        payload = capture_payload(value, field, self.options.redact_payload)
        if payload is OMITTED:
            span.set_attribute(f"judgment.mcp.{field}_omitted", "true")
        else:
            span.set_attribute(
                f"judgment.{field}",
                json.dumps(payload, ensure_ascii=False, allow_nan=False),
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
        outcome = (
            "cancelled"
            if cancelled
            else "error"
            if record.error is not None or result is None
            else "error"
            if attempt(lambda: adapter.is_error(result)) is not False
            else "success"
        )
        attempt(lambda: span.set_attribute("judgment.mcp.outcome", outcome))
        if outcome == "error":
            attempt(lambda: span.set_status(Status(StatusCode.ERROR)))

        def annotate() -> None:
            caller = self._caller(record, facts)
            span.set_attributes(
                _defined(
                    {
                        "judgment.customer_user_id": caller.get("user_id"),
                        "judgment.customer_id": caller.get("customer_id"),
                    }
                )
            )
            if outcome == "error":
                error_type, message = self._failure(record, result, facts, adapter)
                span.set_attribute("error.type", error_type)
                span.set_attribute("judgment.mcp.error_type", error_type)
                if message is not None:
                    span.set_attribute("judgment.mcp.error_message", message)
                span.set_status(Status(StatusCode.ERROR, message))
            if (
                self.options.capture_output
                and result is not None
                and facts.method == "tools/call"
            ):
                self._payload(span, adapter.payload(result), "output")

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
            self._redacted(message)
            if self.options.capture_error_messages and message is not None
            else None
        )
        record.failure = (
            self._error_type(error, result is not None, message),
            None if recorded is None else recorded[:_MAX_ERROR_MESSAGE_LENGTH],
        )
        return record.failure

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


def _span_scope(span: Span, session_id: Optional[str]) -> Any:
    from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

    from contextlib import contextmanager
    from judgeval.trace.baggage import set_baggage

    def attach():
        provider = JudgmentTracerProvider.get_instance()
        current = provider.get_current_context()
        if session_id:
            current = set_baggage("judgment.session_id", session_id, context=current)
        token = provider.attach_context(set_span_in_context(span, current))
        return provider, token

    @contextmanager
    def scope():
        attached = attempt(attach)
        try:
            yield
        finally:
            if attached is not None:
                provider, token = attached
                attempt(lambda: provider.detach_context(token))

    return scope()


def _active_tracer_var() -> contextvars.ContextVar[Any]:
    from judgeval.trace.judgment_tracer_provider import _active_tracer_var

    return _active_tracer_var


def _active_tracer() -> Any:
    return _active_tracer_var().get()


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
    args: Mapping[str, Any],
    declared: Optional[FrozenSet[str]],
    names: Sequence[str] = tuple(AGENT_ARGUMENTS),
) -> Optional[str]:
    """Lists a call's argument names, leaving out injected ones."""
    keys: List[str] = [
        key[:_MAX_RECORDED_KEY_LENGTH]
        for key in args
        if isinstance(key, str)
        and (declared is None or key not in names or key in declared)
    ][:_MAX_RECORDED_KEYS]
    return json.dumps(keys) if keys else None


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
