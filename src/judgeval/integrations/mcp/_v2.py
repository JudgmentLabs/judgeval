"""`mcp` 2.x: one middleware on the low-level server sees every request.

``Server.middleware`` is public in `mcp` 2.x and runs before parameter
validation, so a middleware can remove injected arguments before any tool
schema checks them, and it receives the result in its wire shape. The
high-level ``MCPServer`` and the standalone ``fastmcp`` package (4.x) are built
on the same low-level server, so this covers all of them.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from ._arguments import (
    CONVERSATION_ARGUMENT,
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


class WireResults:
    """Results in the shape the 2.x runner sends: dicts with wire names."""

    def __init__(
        self,
        inject_conversation_id: bool = True,
        conversation_argument: str = CONVERSATION_ARGUMENT,
    ) -> None:
        self._inject_conversation_id = inject_conversation_id
        self._conversation_argument = conversation_argument
        self._structured_tools: set[str] = set()

    def payload(self, result: Any) -> Any:
        return (
            dict(result)
            if isinstance(result, Mapping)
            else result.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    def is_error(self, result: Any) -> bool:
        if isinstance(result, Mapping):
            return result.get("isError") is True
        return getattr(result, "is_error", None) is True

    def first_text(self, result: Any) -> Optional[str]:
        content = (
            result.get("content")
            if isinstance(result, Mapping)
            else getattr(result, "content", None)
        )
        if not isinstance(content, list) or not content:
            return None
        first = content[0]
        text = (
            first.get("text")
            if isinstance(first, Mapping)
            else getattr(first, "text", None)
        )
        return text if isinstance(text, str) else None

    def with_text(self, result: Any, text: str, tool_name: Any) -> Any:
        if isinstance(result, Mapping):
            content = result.get("content")
            if content is None:
                content = []
            if not isinstance(content, list):
                return result
            updates = {}
            if tool_name in self._structured_tools:
                structured = structured_with_prompt(
                    result.get("structuredContent"), text
                )
                if structured is not None:
                    updates["structuredContent"] = structured
            return {
                **result,
                **updates,
                "content": [*content, {"type": "text", "text": text}],
            }
        content = getattr(result, "content", None)
        if content is None:
            content = []
        if isinstance(content, list) and hasattr(result, "model_copy"):
            item = _text_item(text)
            updates = {"content": [*content, item]}
            if tool_name in self._structured_tools:
                structured = structured_with_prompt(
                    getattr(result, "structured_content", None), text
                )
                if structured is not None:
                    updates["structured_content"] = structured
            return result.model_copy(update=updates)
        return result

    def listed_tools(self, result: Any) -> Optional[ListedTools]:
        tools = (
            result.get("tools")
            if isinstance(result, Mapping)
            else getattr(result, "tools", None)
        )
        if not isinstance(tools, list):
            return None
        listed = []
        for tool in tools:
            if isinstance(tool, Mapping):
                name, schema = tool.get("name"), tool.get("inputSchema")
            else:
                name, schema = (
                    getattr(tool, "name", None),
                    getattr(tool, "input_schema", None),
                )
            if isinstance(name, str):
                listed.append((name, schema))
        return listed

    def with_tool_schemas(
        self, result: Any, schema_for: Callable[[str, Any], Optional[Any]]
    ) -> Any:
        if isinstance(result, Mapping) and isinstance(result.get("tools"), list):
            tools: List[Any] = []
            for tool in result["tools"]:
                name = tool.get("name") if isinstance(tool, Mapping) else None
                schema = (
                    schema_for(name, tool.get("inputSchema"))
                    if isinstance(name, str)
                    else None
                )
                tools.append(
                    tool
                    if schema is None
                    or not isinstance(tool, Mapping)
                    or not isinstance(name, str)
                    else {
                        **tool,
                        **self._schema_updates(
                            name,
                            schema,
                            tool.get("outputSchema"),
                            tool.get("inputSchema"),
                        ),
                    }
                )
            return {**result, "tools": tools}
        tools_attribute = getattr(result, "tools", None)
        if isinstance(tools_attribute, list) and hasattr(result, "model_copy"):
            updated = []
            for tool in tools_attribute:
                schema = schema_for(tool.name, getattr(tool, "input_schema", None))
                updated.append(
                    tool
                    if schema is None
                    else tool.model_copy(
                        update={
                            key.replace("Schema", "_schema"): value
                            for key, value in self._schema_updates(
                                tool.name,
                                schema,
                                getattr(tool, "output_schema", None),
                                getattr(tool, "input_schema", None),
                            ).items()
                        }
                    )
                )
            return result.model_copy(update={"tools": updated})
        return result

    def _schema_updates(
        self, name: str, schema: Any, output: Any, original: Any
    ) -> Dict[str, Any]:
        extended = (
            conversation_output_schema(output)
            if self._inject_conversation_id
            and self._conversation_argument in schema.get("properties", {})
            and self._conversation_argument
            not in ((original or {}).get("properties", {}))
            else None
        )
        self._structured_tools.discard(name)
        if output is None or extended is not None:
            self._structured_tools.add(name)
        return {
            "inputSchema": schema,
            **({"outputSchema": extended} if extended is not None else {}),
        }


def _text_item(text: str) -> Any:
    from mcp.types import TextContent

    return TextContent(type="text", text=text)


class JudgmentMcpMiddleware:
    """Traces each request the server handles. Notifications pass through."""

    def __init__(self, observer: McpObserver) -> None:
        self._observer = observer
        self._results = WireResults(
            observer.options.inject_conversation_id,
            observer.argument_names.names[2],
        )

    async def __call__(
        self, ctx: Any, call_next: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        if getattr(ctx, "request_id", None) is None:
            return await call_next(ctx)
        facts = attempt(lambda: _facts(ctx))
        if facts is None:
            return await call_next(ctx)

        async def invoke(arguments: Optional[Dict[str, Any]]) -> Any:
            stripped = (
                None
                if arguments is None
                else attempt(
                    lambda: dataclasses.replace(
                        ctx, params={**(ctx.params or {}), "arguments": arguments}
                    )
                )
            )
            return await call_next(ctx if stripped is None else stripped)

        async def list_tools(
            cursor: Optional[str],
        ) -> Tuple[ListedTools, Optional[str]]:
            return await self._list_tools(ctx, call_next, cursor)

        return await self._observer.observe(facts, invoke, self._results, list_tools)

    async def _list_tools(
        self,
        ctx: Any,
        call_next: Callable[[Any], Awaitable[Any]],
        cursor: Optional[str],
    ) -> Tuple[ListedTools, Optional[str]]:
        """Lists the tools this caller would see, through the rest of the chain.

        Like `mcp`'s own lookup of a tool's schema, this runs the remaining
        middleware, so a listing scoped to the caller stays scoped. It isn't
        traced, because this middleware isn't part of the rest of the chain.
        """
        params: Dict[str, Any] = {}
        meta = ctx.params.get("_meta") if isinstance(ctx.params, Mapping) else None
        if isinstance(meta, Mapping):
            # The protocol's own entries, without a progress token for the call.
            params["_meta"] = {
                key: value for key, value in meta.items() if key != "progressToken"
            }
        if cursor is not None:
            params["cursor"] = cursor
        result = await call_next(
            dataclasses.replace(ctx, method="tools/list", params=params)
        )
        tools = self._results.listed_tools(result) or []
        self._observer.shape_listing(result, self._results)
        next_cursor = (
            result.get("nextCursor")
            if isinstance(result, Mapping)
            else getattr(result, "next_cursor", None)
        )
        return tools, next_cursor if isinstance(next_cursor, str) else None


def _facts(ctx: Any) -> RequestFacts:
    params = ctx.params if isinstance(ctx.params, Mapping) else {}
    request = getattr(ctx, "request", None)
    headers = request_headers(request)
    session = getattr(ctx, "session", None)
    connection = getattr(session, "_connection", None)
    client_params = getattr(connection, "client_params", None)
    client_info = getattr(client_params, "client_info", None)
    client: Dict[str, Any] = {}
    if client_info is not None:
        client = {
            "name": getattr(client_info, "name", None),
            "version": getattr(client_info, "version", None),
        }
    session_id = getattr(connection, "session_id", None)
    request_id = getattr(ctx, "request_id", None)
    return RequestFacts(
        method=ctx.method,
        params=params,
        request_id=None if request_id is None else str(request_id),
        session_id=session_id
        if isinstance(session_id, str) and session_id
        else transport_session_id(headers, request),
        headers=headers,
        client=client,
        protocol_version=getattr(ctx, "protocol_version", None),
        auth_info=access_token(),
        request=request,
    )


def instrument(observer: McpObserver, lowlevel: Any) -> bool:
    """Makes the execution span current for the server's existing telemetry."""
    middleware = getattr(lowlevel, "middleware", None)
    if not isinstance(middleware, list) or not callable(
        getattr(lowlevel, "get_request_handler", None)
    ):
        return False
    kept = [
        entry for entry in middleware if not isinstance(entry, JudgmentMcpMiddleware)
    ]
    middleware[:] = [JudgmentMcpMiddleware(observer), *kept]
    return True
