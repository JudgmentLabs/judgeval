"""Claude Agent SDK auto-instrumentation wrapper."""

from __future__ import annotations

import contextvars
import dataclasses
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from opentelemetry.trace import set_span_in_context

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.v1.tracer.tracer import BaseTracer


# =============================================================================
# Tracer Registry
# =============================================================================

_registered_tracers: List["BaseTracer"] = []
_module_patched: bool = False
_parent_context: contextvars.ContextVar = contextvars.ContextVar(
    "parent_context", default=None
)


def register_tracer(tracer: "BaseTracer") -> None:
    """Register a tracer for Claude Agent SDK."""
    if tracer not in _registered_tracers:
        _registered_tracers.append(tracer)


def get_active_tracer() -> Optional["BaseTracer"]:
    """Get active tracer based on current span context.

    Priority:
    1. Tracer with active span (from @tracer.observe() or tracer.span())
    2. Last registered tracer (fallback for single-tracer setups)
    """
    for tracer in _registered_tracers:
        try:
            span = tracer._get_current_span()
            if span and span.is_recording():
                return tracer
        except Exception:
            continue

    return _registered_tracers[-1] if _registered_tracers else None


def is_module_patched() -> bool:
    return _module_patched


def mark_module_patched() -> None:
    global _module_patched
    _module_patched = True


def _reset_registry_state() -> None:
    """Reset tracer registry state. For testing only."""
    global _module_patched
    _registered_tracers.clear()
    _module_patched = False


# =============================================================================
# LLM Span Tracker
# =============================================================================


class LLMSpanTracker:
    """Tracks LLM spans across message stream turns.

    Manages span timing: marks when next LLM call starts (after tool results),
    uses that time when creating the span for accurate duration tracking.
    """

    def __init__(self, tracer: "BaseTracer", start_time: Optional[float] = None):
        self.tracer = tracer
        self.span = None
        self.span_ctx = None
        self.next_start_time = start_time

    def start_span(
        self, message: Any, prompt: Any, history: List[Dict]
    ) -> Optional[Dict]:
        if self.span_ctx:
            self.span_ctx.__exit__(None, None, None)

        start = self.next_start_time if self.next_start_time else time.time()
        content, self.span, self.span_ctx = _create_llm_span(
            self.tracer, message, prompt, history, start
        )
        self.next_start_time = None
        return content

    def mark_next_start(self) -> None:
        """Mark when next LLM call starts (after tool results)."""
        self.next_start_time = time.time()

    def log_usage(self, metrics: Dict) -> None:
        if self.span and metrics:
            for k, v in metrics.items():
                set_span_attribute(self.span, k, v)

    def cleanup(self) -> None:
        if self.span_ctx:
            self.span_ctx.__exit__(None, None, None)
        self.span = self.span_ctx = None


# =============================================================================
# Wrapper Factories
# =============================================================================


def _create_client_wrapper_class(original_class: Any) -> Any:
    """Create wrapped ClaudeSDKClient with auto-tracing."""

    class WrappedClient(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._prompt: Optional[str] = None
            self._query_time: Optional[float] = None

        async def query(self, *args, **kwargs) -> Any:
            self._query_time = time.time()
            self._prompt = str(args[0]) if args else kwargs.get("prompt")
            return await super().query(*args, **kwargs)

        async def receive_response(self) -> AsyncGenerator[Any, None]:
            tracer = get_active_tracer()
            if not tracer:
                async for msg in super().receive_response():
                    yield msg
                return

            async for msg in _traced_response_stream(
                super().receive_response(), tracer, self._prompt, self._query_time
            ):
                yield msg

    return WrappedClient


def _create_tool_wrapper_class(original_class: Any) -> Any:
    """Create wrapped SdkMcpTool with auto-tracing."""

    class WrappedTool(original_class):
        def __init__(self, name, description, input_schema, handler, **kwargs):
            super().__init__(
                name,
                description,
                input_schema,
                _wrap_tool_handler(handler, name),
                **kwargs,
            )

        __class_getitem__ = classmethod(lambda cls, _: cls)

    return WrappedTool


def _wrap_query_function(original_fn: Any) -> Callable:
    """Wrap standalone query() function with auto-tracing."""

    async def wrapped(*args, **kwargs):
        tracer = get_active_tracer()
        if not tracer:
            async for msg in original_fn(*args, **kwargs):
                yield msg
            return

        prompt = kwargs.get("prompt") or (args[0] if args else None)
        async for msg in _traced_response_stream(
            original_fn(*args, **kwargs), tracer, prompt, time.time()
        ):
            yield msg

    return wrapped


def _wrap_tool_factory(original_fn: Any) -> Callable:
    """Wrap tool() decorator factory with auto-tracing."""

    def wrapped(*args, **kwargs):
        result = original_fn(*args, **kwargs)
        if not callable(result):
            return result

        def decorator(handler):
            tool_def = result(handler)
            if tool_def and hasattr(tool_def, "handler"):
                tool_def.handler = _wrap_tool_handler(
                    tool_def.handler, getattr(tool_def, "name", "tool")
                )
            return tool_def

        return decorator

    return wrapped


def _wrap_tool_handler(handler: Any, name: Any) -> Callable:
    """Wrap tool handler with tracing."""
    if getattr(handler, "_judgeval_wrapped", False):
        return handler

    async def wrapped(args: Any) -> Any:
        tracer = get_active_tracer()
        if not tracer:
            return await handler(args)

        ctx = _parent_context.get()
        span = tracer.get_tracer().start_span(
            str(name),
            context=ctx,
            attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "tool"},
        )

        with tracer.use_span(span, end_on_exit=True):
            set_span_attribute(span, AttributeKeys.JUDGMENT_INPUT, safe_serialize(args))
            try:
                result = await handler(args)
                set_span_attribute(
                    span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result)
                )
                return result
            except Exception as e:
                span.record_exception(e)
                raise

    setattr(wrapped, "_judgeval_wrapped", True)
    return wrapped


# =============================================================================
# Tracing Helpers
# =============================================================================


async def _traced_response_stream(
    generator: AsyncGenerator,
    tracer: "BaseTracer",
    prompt: Optional[str],
    start_time: Optional[float] = None,
) -> AsyncGenerator[Any, None]:
    """Wrap response stream with agent span and LLM tracking."""
    span_ctx = tracer.get_tracer().start_as_current_span(
        "Claude_Agent", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "agent"}
    )
    span = span_ctx.__enter__()

    if prompt:
        set_span_attribute(span, AttributeKeys.JUDGMENT_INPUT, safe_serialize(prompt))

    token = _parent_context.set(set_span_in_context(span, tracer.get_context()))

    results: List[Dict] = []
    tracker = LLMSpanTracker(tracer, start_time)

    try:
        async for msg in generator:
            msg_type = type(msg).__name__

            if msg_type == "AssistantMessage":
                if content := tracker.start_span(msg, prompt, results):
                    results.append(content)

            elif msg_type == "UserMessage":
                if hasattr(msg, "content"):
                    results.append(
                        {"content": _serialize_content(msg.content), "role": "user"}
                    )
                tracker.mark_next_start()

            elif msg_type == "ResultMessage":
                if hasattr(msg, "usage"):
                    tracker.log_usage(_extract_usage(msg))
                for key in ("num_turns", "session_id"):
                    if (val := getattr(msg, key, None)) is not None:
                        set_span_attribute(span, f"agent.{key}", val)

            yield msg

        if results:
            set_span_attribute(
                span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(results[-1])
            )

    except Exception as e:
        span.record_exception(e)
        raise
    finally:
        tracker.cleanup()
        span_ctx.__exit__(None, None, None)
        _parent_context.reset(token)


def _create_llm_span(
    tracer: "BaseTracer",
    message: Any,
    prompt: Any,
    history: List[Dict],
    start_time: Optional[float] = None,
) -> Tuple[Optional[Dict], Optional[Any], Optional[Any]]:
    """Create LLM span for an AssistantMessage."""
    if type(message).__name__ != "AssistantMessage":
        return None, None, None

    model = getattr(message, "model", None)
    content = _serialize_content(getattr(message, "content", None))

    input_msgs = None
    if isinstance(prompt, str):
        input_msgs = [{"content": prompt, "role": "user"}] + (history or [])
    elif history:
        input_msgs = history

    span_ctx = tracer.get_tracer().start_as_current_span(
        "anthropic.messages.create",
        attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"},
    )
    span = span_ctx.__enter__()

    if model:
        set_span_attribute(span, AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model)
        set_span_attribute(span, AttributeKeys.JUDGMENT_LLM_PROVIDER, "anthropic")
    if input_msgs:
        set_span_attribute(
            span, AttributeKeys.JUDGMENT_INPUT, safe_serialize(input_msgs)
        )
    if content:
        set_span_attribute(
            span,
            AttributeKeys.JUDGMENT_OUTPUT,
            safe_serialize([{"content": content, "role": "assistant"}]),
        )

    return (
        {"content": content, "role": "assistant"} if content else None,
        span,
        span_ctx,
    )


def _serialize_content(content: Any) -> Any:
    """Serialize content blocks to JSON-friendly format."""
    if not isinstance(content, list):
        return content

    result = []
    for block in content:
        if dataclasses.is_dataclass(block) and not isinstance(block, type):
            data = dataclasses.asdict(block)

            type_map = {
                "TextBlock": "text",
                "ToolUseBlock": "tool_use",
                "ToolResultBlock": "tool_result",
                "ThinkingBlock": "thinking",
            }
            if block_type := type_map.get(type(block).__name__):
                data["type"] = block_type

            if data.get("type") == "tool_result":
                if isinstance(data.get("content"), list) and len(data["content"]) == 1:
                    item = data["content"][0]
                    if isinstance(item, dict) and item.get("type") == "text":
                        data["content"] = item.get("text", "")
                if data.get("is_error") is None:
                    data.pop("is_error", None)

            result.append(data)
        else:
            result.append(block)

    return result


def _extract_usage(msg: Any) -> Dict[str, Any]:
    """Extract usage metrics from ResultMessage."""
    usage = getattr(msg, "usage", None)
    if not usage:
        return {}

    get = (
        (lambda k: usage.get(k))
        if isinstance(usage, dict)
        else (lambda k: getattr(usage, k, None))
    )

    metrics: Dict[str, Any] = {}
    key_map = {
        "input_tokens": AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
        "output_tokens": AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
        "cache_creation_input_tokens": AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
        "cache_read_input_tokens": AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS,
    }

    for src, dst in key_map.items():
        if (val := get(src)) is not None:
            metrics[str(dst)] = val

    metrics[str(AttributeKeys.JUDGMENT_USAGE_METADATA)] = safe_serialize(usage)
    return metrics
