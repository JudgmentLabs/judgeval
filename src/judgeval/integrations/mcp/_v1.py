"""`mcp` 1.x: wraps the low-level server's ``request_handlers``.

Every 1.x server, whether written with ``FastMCP``, the low-level ``Server`` or
the standalone ``fastmcp`` package (2.x and 3.x), registers its request handlers
in that one dictionary, keyed by request class, and the session dispatches
every request through it. Wrapping its entries traces every request, before
the server checks a tool's arguments, with the result still in hand.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from ._arguments import (
    CONVERSATION_ARGUMENT,
    ToolContract,
    conversation_output_schema,
    structured_with_prompt,
)
from ._observer import (
    ListedTools,
    McpObserver,
    RequestFacts,
    access_token,
    attempt,
    request_headers,
    transport_session_id,
)
from ._payload import result_payload

_WRAPPED_MARKER = "__judgeval_mcp_wrapped__"


class ModelResults:
    """Results as 1.x handlers return them: ``ServerResult`` wrapping a model."""

    def __init__(
        self,
        inject_conversation_id: bool = True,
        conversation_argument: str = CONVERSATION_ARGUMENT,
    ) -> None:
        self._inject_conversation_id = inject_conversation_id
        self._conversation_argument = conversation_argument
        self._structured_tools: set[str] = set()

    def is_error(self, result: Any) -> bool:
        return getattr(_root(result), "isError", None) is True

    def payload(self, result: Any) -> Any:
        return result_payload(_root(result))

    def first_text(self, result: Any) -> Optional[str]:
        content = getattr(_root(result), "content", None)
        if not isinstance(content, list) or not content:
            return None
        text = getattr(content[0], "text", None)
        return text if isinstance(text, str) else None

    def with_text(
        self,
        result: Any,
        text: str,
        tool_name: Any,
        contract: Optional[ToolContract] = None,
    ) -> Any:
        annotate = (
            contract.structured_output
            if contract is not None and contract.structured_output is not None
            else tool_name in self._structured_tools
        )
        root = _root(result)
        content = getattr(root, "content", None)
        if content is None:
            content = []
        if not isinstance(content, list):
            return result
        from mcp import types

        updates: Dict[str, Any] = {
            "content": [*content, types.TextContent(type="text", text=text)]
        }
        if annotate:
            structured = structured_with_prompt(
                getattr(root, "structuredContent", None), text
            )
            if structured is not None:
                updates["structuredContent"] = structured
        updated = root.model_copy(update=updates)
        return _rewrap(result, updated)

    def listed_tools(self, result: Any) -> Optional[ListedTools]:
        tools = getattr(_root(result), "tools", None)
        if not isinstance(tools, list):
            return None
        return [
            (
                tool.name,
                getattr(tool, "inputSchema", None),
                getattr(tool, "outputSchema", None),
            )
            for tool in tools
            if isinstance(getattr(tool, "name", None), str)
        ]

    def with_tool_schemas(
        self, result: Any, schema_for: Callable[[str, Any], Optional[Any]]
    ) -> Any:
        root = _root(result)
        tools = getattr(root, "tools", None)
        if not isinstance(tools, list):
            return result
        updated = []
        for tool in tools:
            schema = schema_for(tool.name, getattr(tool, "inputSchema", None))
            updates = {"inputSchema": schema}
            output = getattr(tool, "outputSchema", None)
            extended = (
                conversation_output_schema(output)
                if self._inject_conversation_id
                and schema is not None
                and self._conversation_argument in schema.get("properties", {})
                and self._conversation_argument
                not in (getattr(tool, "inputSchema", {}).get("properties", {}))
                else None
            )
            self._structured_tools.discard(tool.name)
            if schema is not None and (output is None or extended is not None):
                self._structured_tools.add(tool.name)
            if extended is not None:
                updates["outputSchema"] = extended
            # A copy, so a tool object the server reuses keeps its own schema.
            updated.append(tool if schema is None else tool.model_copy(update=updates))
        return _rewrap(result, root.model_copy(update={"tools": updated}))


def _root(result: Any) -> Any:
    return getattr(result, "root", result)


def _rewrap(result: Any, root: Any) -> Any:
    if getattr(result, "root", None) is not None and hasattr(result, "model_copy"):
        return result.model_copy(update={"root": root})
    return root


class _TracedHandlers(dict):
    """Wraps handlers the server registers after it was instrumented."""

    def __init__(self, handlers: Dict[Any, Any], wrap: Callable[[Any, Any], Any]):
        self._wrap = wrap
        super().__init__({key: wrap(key, value) for key, value in handlers.items()})

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, self._wrap(key, value))

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]


def instrument(observer: McpObserver, lowlevel: Any) -> bool:
    handlers = getattr(lowlevel, "request_handlers", None)
    if not isinstance(handlers, dict):
        return False
    try:
        from mcp import types
        from mcp.server.lowlevel import server as server_module

        request_ctx = getattr(server_module, "request_ctx", None)
        if request_ctx is None:
            return False
    except Exception:
        return False
    results = ModelResults(
        observer.options.inject_conversation_id,
        observer.argument_names.names[2],
    )
    list_request = getattr(types, "ListToolsRequest", None)

    async def list_tools(cursor: Optional[str]) -> Tuple[ListedTools, Optional[str]]:
        """Asks the server's own tools/list handler, without tracing it."""
        handler = lowlevel.request_handlers.get(list_request)
        if handler is None:
            return [], None
        original = getattr(handler, _WRAPPED_MARKER, handler)
        request: Dict[str, Any] = {"method": "tools/list"}
        context = request_ctx.get(None)
        incoming = getattr(context, "meta", None)
        meta = (
            incoming.model_dump(by_alias=True, exclude_none=True)
            if incoming is not None
            else {}
        )
        params: Dict[str, Any] = {
            "_meta": {
                key: value for key, value in meta.items() if key != "progressToken"
            }
        }
        if cursor is not None:
            params["cursor"] = cursor
        request["params"] = params
        result = await original(types.ListToolsRequest.model_validate(request))
        observer.shape_listing(result, results)
        next_cursor = getattr(_root(result), "nextCursor", None)
        return results.listed_tools(result) or [], (
            next_cursor if isinstance(next_cursor, str) else None
        )

    def wrap(request_type: Any, handler: Any) -> Any:
        if not callable(handler) or getattr(handler, _WRAPPED_MARKER, None) is not None:
            return handler

        async def traced(req: Any) -> Any:
            # The server lists its own tools with no request to check a call.
            if req is None or not isinstance(getattr(req, "method", None), str):
                result = await handler(req)
                if request_type is list_request:
                    observer.remember_listing(
                        attempt(lambda: results.listed_tools(result))
                    )
                return result
            context = request_ctx.get(None)
            facts = attempt(lambda: _facts(req, context))
            if facts is None:
                return await handler(req)

            async def invoke(arguments: Optional[Dict[str, Any]]) -> Any:
                stripped = (
                    None
                    if arguments is None or req.params is None
                    else attempt(
                        lambda: req.model_copy(
                            update={
                                "params": req.params.model_copy(
                                    update={"arguments": arguments}
                                )
                            }
                        )
                    )
                )
                return await handler(req if stripped is None else stripped)

            return await observer.observe(facts, invoke, results, list_tools)

        setattr(traced, _WRAPPED_MARKER, handler)
        return traced

    lowlevel.request_handlers = _TracedHandlers(handlers, wrap)
    return True


def _facts(req: Any, context: Any) -> RequestFacts:
    params_model = getattr(req, "params", None)
    params: Dict[str, Any] = {}
    if params_model is not None:
        params = params_model.model_dump(by_alias=True, exclude_none=True)
        # A resource URI is a URL object on 1.x models.
        if params.get("uri") is not None:
            params["uri"] = str(params["uri"])
        # ``_meta`` is aliased as ``meta`` on some 1.x models.
        meta = getattr(params_model, "meta", None)
        if meta is not None and "_meta" not in params:
            params["_meta"] = meta.model_dump(by_alias=True, exclude_none=True)
    request = getattr(context, "request", None)
    headers = request_headers(request)
    session = getattr(context, "session", None)
    client_params = getattr(session, "client_params", None)
    client_info = getattr(client_params, "clientInfo", None)
    client: Dict[str, Any] = {}
    if client_info is not None:
        client = {
            "name": getattr(client_info, "name", None),
            "version": getattr(client_info, "version", None),
        }
    request_id = getattr(context, "request_id", None)
    return RequestFacts(
        method=req.method,
        params=params,
        request_id=None if request_id is None else str(request_id),
        session_id=transport_session_id(headers, request),
        headers=headers,
        client=client,
        protocol_version=getattr(client_params, "protocolVersion", None),
        auth_info=access_token(),
        request=request,
    )
