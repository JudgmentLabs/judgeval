"""Wrapper that patches the Claude Agent SDK for automatic tracing."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.trace.tracer import Tracer
from judgeval.utils.serialize import safe_serialize, serialize_attribute
from judgeval.utils.wrappers import immutable_wrap_async, immutable_wrap_async_iterator


Ctx = Dict[str, Any]


@dataclasses.dataclass(slots=True)
class TracingState:
    """Shared mutable state carrying the parent span context across turns."""

    parent_context: Any = None


@dataclasses.dataclass(slots=True)
class _ClientTracingState:
    last_prompt: Optional[str] = None
    query_start_time: Optional[float] = None
    conversation_history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


class ToolSpanTracker:
    """Creates and closes tool spans by matching ToolUseBlock / ToolResultBlock pairs."""

    def __init__(self, state: TracingState):
        self._state = state
        self._pending: Dict[str, Tuple[Span, str]] = {}

    def on_assistant_message(self, message: Any) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            if type(block).__name__ != "ToolUseBlock":
                continue
            name = getattr(block, "name", None)
            uid = getattr(block, "id", None)
            if not name or not uid:
                continue
            span = Tracer.start_span(
                str(name),
                context=self._state.parent_context,
                attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "tool"},
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_INPUT,
                safe_serialize(getattr(block, "input", None)),
            )
            self._pending[uid] = (span, name)

    def on_user_message(self, message: Any) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            uid = getattr(block, "tool_use_id", None)
            if not uid or uid not in self._pending:
                continue
            span, _ = self._pending.pop(uid)
            span.set_attribute(
                AttributeKeys.JUDGMENT_OUTPUT,
                safe_serialize(getattr(block, "content", None)),
            )
            if getattr(block, "is_error", None):
                span.set_status(Status(StatusCode.ERROR, "Tool returned an error"))
            span.end()

    def cleanup(self) -> None:
        for span, _ in self._pending.values():
            span.end()
        self._pending.clear()


class LLMSpanTracker:
    """Manages LLM span lifecycle for Claude Agent SDK message streams."""

    def __init__(self, query_start_time: Optional[float] = None):
        self._span: Optional[Span] = None
        self._span_ctx: Optional[Any] = None
        self._next_start: Optional[float] = query_start_time

    def start_llm_span(
        self,
        message: Any,
        prompt: Optional[str],
        history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        start = self._next_start if self._next_start is not None else time.time()

        if self._span_ctx:
            self._span_ctx.__exit__(None, None, None)

        inputs = _build_llm_input(prompt, history)
        model = getattr(message, "model", None)

        self._span_ctx = Tracer.start_as_current_span(
            "anthropic.messages.create",
            attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"},
            start_time=int(start * 1e9),
        )
        self._span = self._span_ctx.__enter__()
        self._next_start = None

        Tracer.recordLLMMetadata({"model": model or "unknown", "provider": "anthropic"})

        if inputs:
            Tracer.set_input(inputs)

        if hasattr(message, "content"):
            content = _serialize_content_blocks(message.content)
            Tracer.set_output([{"content": content, "role": "assistant"}])
            Tracer._emit_partial()
            return {"content": content, "role": "assistant"}

        Tracer._emit_partial()
        return None

    def mark_next_llm_start(self) -> None:
        self._next_start = time.time()

    def log_usage(self, usage: Any) -> None:
        if not self._span or not usage:
            return
        get: Callable[[str], Any] = (
            (lambda k: usage.get(k))
            if isinstance(usage, dict)
            else (lambda k: getattr(usage, k, None))
        )
        metadata: Dict[str, Any] = {}
        for src, dst in (
            ("input_tokens", "non_cached_input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_creation_input_tokens", "cache_creation_input_tokens"),
            ("cache_read_input_tokens", "cache_read_input_tokens"),
        ):
            val = get(src)
            if val is not None:
                metadata[dst] = val
        if metadata:
            Tracer.recordLLMMetadata(metadata)  # type: ignore[arg-type]
        self._span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage)
        )

    def cleanup(self) -> None:
        if self._span_ctx:
            self._span_ctx.__exit__(None, None, None)
        self._span = self._span_ctx = None


def _process_message(ctx: Ctx, message: Any) -> None:
    agent_span: Optional[Span] = ctx.get("agent_span")
    llm: Optional[LLMSpanTracker] = ctx.get("llm_tracker")
    tools: Optional[ToolSpanTracker] = ctx.get("tool_tracker")
    results: Optional[List[Dict[str, Any]]] = ctx.get("final_results")
    if not agent_span or not llm or not tools or results is None:
        return

    kind = type(message).__name__

    if kind == "AssistantMessage":
        conv_history: List[Dict[str, Any]] = ctx.get("conversation_history", [])
        content = llm.start_llm_span(message, ctx.get("prompt"), conv_history + results)
        if content:
            results.append(content)
        tools.on_assistant_message(message)

    elif kind == "UserMessage":
        tools.on_user_message(message)
        if hasattr(message, "content"):
            results.append(
                {
                    "content": _serialize_content_blocks(message.content),
                    "role": "user",
                }
            )
        llm.mark_next_llm_start()

    elif kind == "ResultMessage":
        usage = getattr(message, "usage", None)
        if usage:
            llm.log_usage(usage)
        for attr in ("num_turns", "session_id"):
            val = getattr(message, attr, None)
            if val is not None:
                agent_span.set_attribute(
                    f"agent.{attr}",
                    val
                    if isinstance(val, (str, int, float, bool))
                    else safe_serialize(val),
                )


def _init_agent_span(
    ctx: Ctx,
    ts: TracingState,
    prompt: Optional[str],
    start_time: Optional[float],
    span_name: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> None:
    ctx.update(
        agent_span=None,
        agent_span_ctx=None,
        llm_tracker=None,
        tool_tracker=None,
        final_results=[],
        prompt=prompt,
        conversation_history=conversation_history or [],
    )

    span_ctx = Tracer.start_as_current_span(
        span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "agent"}
    )
    span = span_ctx.__enter__()
    ctx["agent_span"] = span
    ctx["agent_span_ctx"] = span_ctx

    if prompt:
        Tracer.set_input(prompt)
    Tracer._emit_partial()

    ts.parent_context = set_span_in_context(
        span, Tracer._get_proxy_provider().get_current_context()
    )
    ctx["llm_tracker"] = LLMSpanTracker(query_start_time=start_time)
    ctx["tool_tracker"] = ToolSpanTracker(state=ts)


def _yield_hook(ctx: Ctx, message: Any) -> None:
    _process_message(ctx, message)


def _make_post_hook(cs: Optional[_ClientTracingState] = None) -> Callable[[Ctx], None]:
    def hook(ctx: Ctx) -> None:
        results: Optional[List[Dict[str, Any]]] = ctx.get("final_results")
        agent_span: Optional[Span] = ctx.get("agent_span")
        if agent_span and results:
            agent_span.set_attribute(
                AttributeKeys.JUDGMENT_OUTPUT,
                serialize_attribute(results[-1], safe_serialize),
            )
        if cs is not None and results:
            prompt = ctx.get("prompt")
            if prompt:
                cs.conversation_history.append({"content": prompt, "role": "user"})
            cs.conversation_history.extend(results)

    return hook


def _error_hook(ctx: Ctx, error: Exception) -> None:
    span: Optional[Span] = ctx.get("agent_span")
    if span:
        span.record_exception(error)


def _make_finally_hook(ts: TracingState) -> Callable[[Ctx], None]:
    def hook(ctx: Ctx) -> None:
        for key in ("tool_tracker", "llm_tracker"):
            obj = ctx.get(key)
            if obj is not None:
                obj.cleanup()
        span_ctx = ctx.get("agent_span_ctx")
        if span_ctx is not None:
            span_ctx.__exit__(None, None, None)
        ts.parent_context = None

    return hook


def _make_query_pre_hook(cs: _ClientTracingState) -> Callable[[Ctx, Any], None]:
    def hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
        cs.query_start_time = time.time()
        if args:
            cs.last_prompt = str(args[0])
        elif "prompt" in kwargs:
            cs.last_prompt = str(kwargs["prompt"])

    return hook


def _create_client_wrapper_class(
    original_client_class: Any, state: TracingState
) -> Any:
    """Creates a wrapper class that traces ClaudeSDKClient."""
    finally_hook = _make_finally_hook(state)

    class WrappedClaudeSDKClient(original_client_class):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            cs = _ClientTracingState()

            orig_query = super().query
            self.query = immutable_wrap_async(  # type: ignore[assignment]
                orig_query,
                pre_hook=_make_query_pre_hook(cs),
            )

            def response_pre(ctx: Ctx) -> None:
                _init_agent_span(
                    ctx,
                    state,
                    cs.last_prompt,
                    cs.query_start_time,
                    "Claude_Agent",
                    conversation_history=list(cs.conversation_history),
                )

            orig_receive = super().receive_response
            self.receive_response = immutable_wrap_async_iterator(  # type: ignore[assignment]
                orig_receive,
                pre_hook=response_pre,
                yield_hook=_yield_hook,
                post_hook=_make_post_hook(cs),
                error_hook=_error_hook,
                finally_hook=finally_hook,
            )

    return WrappedClaudeSDKClient


def _wrap_query_function(
    original_query_fn: Any, state: TracingState
) -> Callable[..., Any]:
    """Wraps the standalone query() function."""
    finally_hook = _make_finally_hook(state)

    def pre_hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
        prompt = kwargs.get("prompt") or (
            args[0] if args and isinstance(args[0], str) else None
        )
        _init_agent_span(
            ctx,
            state,
            str(prompt) if prompt else None,
            time.time(),
            "Claude_Agent_Query",
        )

    return immutable_wrap_async_iterator(
        original_query_fn,
        pre_hook=pre_hook,
        yield_hook=_yield_hook,
        post_hook=_make_post_hook(),
        error_hook=_error_hook,
        finally_hook=finally_hook,
    )


def _serialize_content_blocks(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    result = []
    for block in content:
        if dataclasses.is_dataclass(block) and not isinstance(block, type):
            s: Dict[str, Any] = dataclasses.asdict(block)  # type: ignore[arg-type]
            name = type(block).__name__
            if name == "TextBlock":
                s["type"] = "text"
            elif name == "ToolUseBlock":
                s["type"] = "tool_use"
            elif name == "ToolResultBlock":
                s["type"] = "tool_result"
                cv = s.get("content")
                if isinstance(cv, list) and len(cv) == 1:
                    item = cv[0]
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and "text" in item
                    ):
                        s["content"] = item["text"]
                if "is_error" in s and s["is_error"] is None:
                    del s["is_error"]
            elif name == "ThinkingBlock":
                s["type"] = "thinking"
            result.append(s)
        else:
            result.append(block)
    return result


def _build_llm_input(
    prompt: Any, history: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    if isinstance(prompt, str):
        msgs = [{"content": prompt, "role": "user"}]
        return msgs + history if history else msgs
    return history or None
