"""Internal wrapper that patches the Claude Agent SDK for automatic tracing."""

from __future__ import annotations

import dataclasses
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize, serialize_attribute
from judgeval.trace.tracer import Tracer
from judgeval.utils.wrappers import immutable_wrap_async, immutable_wrap_async_iterator


Ctx = Dict[str, Any]


@dataclasses.dataclass(slots=True)
class TracingState:
    """Shared mutable state carrying the parent span context across turns."""

    parent_context: Optional[Context] = None


@dataclasses.dataclass(slots=True)
class _ClientTracingState:
    """Per-client state bridging query() -> receive_response()."""

    last_prompt: Optional[str] = None
    query_start_time: Optional[float] = None
    conversation_history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


class ToolSpanTracker:
    """Creates and closes tool spans by matching ToolUseBlock / ToolResultBlock pairs."""

    def __init__(self, state: TracingState):
        self.state = state
        self._pending_spans: Dict[str, Tuple[Span, str]] = {}

    def on_assistant_message(self, message: Any) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return

        for block in content:
            if type(block).__name__ != "ToolUseBlock":
                continue

            tool_name = getattr(block, "name", None)
            tool_use_id = getattr(block, "id", None)
            tool_input = getattr(block, "input", None)

            if not tool_name or not tool_use_id:
                continue

            span = Tracer._get_otel_tracer().start_span(
                str(tool_name),
                context=self.state.parent_context,
                attributes={
                    AttributeKeys.JUDGMENT_SPAN_KIND: "tool",
                },
            )

            span.set_attribute(AttributeKeys.JUDGMENT_INPUT, safe_serialize(tool_input))

            self._pending_spans[tool_use_id] = (span, tool_name)

        if self._pending_spans:
            Tracer._emit_partial()

    def on_user_message(self, message: Any) -> None:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return

        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue

            tool_use_id = getattr(block, "tool_use_id", None)
            if not tool_use_id or tool_use_id not in self._pending_spans:
                continue

            span, _ = self._pending_spans.pop(tool_use_id)

            result_content = getattr(block, "content", None)
            is_error = getattr(block, "is_error", None)

            span.set_attribute(
                AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result_content)
            )

            if is_error:
                span.set_status(Status(StatusCode.ERROR, "Tool returned an error"))

            span.end()

    def cleanup(self) -> None:
        for span, _ in self._pending_spans.values():
            span.end()
        self._pending_spans.clear()


class LLMSpanTracker:
    """Manages LLM span lifecycle for Claude Agent SDK message streams.

    Message flow per turn:
    1. UserMessage (tool results) -> mark the time when next LLM will start
    2. AssistantMessage - LLM response arrives -> create span with the marked start time, ending previous span
    3. ResultMessage - usage metrics -> log to span
    """

    def __init__(self, query_start_time: Optional[float] = None):
        self.current_span: Optional[Span] = None
        self.current_span_context: Optional[Any] = None
        self.next_start_time: Optional[float] = query_start_time

    def start_llm_span(
        self,
        message: Any,
        prompt: Optional[str],
        conversation_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Start a new LLM span, ending the previous one if it exists."""
        start_time = (
            self.next_start_time if self.next_start_time is not None else time.time()
        )

        if self.current_span_context:
            self.current_span_context.__exit__(None, None, None)

        model = getattr(message, "model", None)
        input_messages = _build_llm_input(prompt, conversation_history)

        outputs: List[Dict[str, Any]] = []
        if hasattr(message, "content"):
            content = _serialize_content_blocks(message.content)
            outputs.append({"content": content, "role": "assistant"})

        self.current_span_context = Tracer._get_otel_tracer().start_as_current_span(
            "anthropic.messages.create",
            attributes={
                AttributeKeys.JUDGMENT_SPAN_KIND: "llm",
            },
            start_time=int(start_time * 1e9),
        )
        self.current_span = self.current_span_context.__enter__()
        self.next_start_time = None

        if model:
            self.current_span.set_attribute(
                AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
                serialize_attribute(model, safe_serialize),
            )
            self.current_span.set_attribute(
                AttributeKeys.JUDGMENT_LLM_PROVIDER,
                serialize_attribute("anthropic", safe_serialize),
            )

        if input_messages:
            self.current_span.set_attribute(
                AttributeKeys.JUDGMENT_INPUT,
                serialize_attribute(input_messages, safe_serialize),
            )

        if outputs:
            self.current_span.set_attribute(
                AttributeKeys.JUDGMENT_OUTPUT,
                serialize_attribute(outputs, safe_serialize),
            )

        Tracer._emit_partial()

        return outputs[0] if outputs else None

    def mark_next_llm_start(self) -> None:
        """Mark when the next LLM call will start (after tool results)."""
        self.next_start_time = time.time()

    def log_usage(self, usage_metrics: Dict[str, Any]) -> None:
        """Log usage metrics to the current LLM span."""
        if self.current_span and usage_metrics:
            for key, value in usage_metrics.items():
                self.current_span.set_attribute(
                    key, serialize_attribute(value, safe_serialize)
                )

    def cleanup(self) -> None:
        """End any unclosed spans."""
        if self.current_span_context:
            self.current_span_context.__exit__(None, None, None)
        self.current_span = None
        self.current_span_context = None


def _process_message(ctx: Ctx, message: Any) -> None:
    agent_span: Optional[Span] = ctx.get("agent_span")
    llm_tracker: Optional[LLMSpanTracker] = ctx.get("llm_tracker")
    tool_tracker: Optional[ToolSpanTracker] = ctx.get("tool_tracker")
    final_results: Optional[List[Dict[str, Any]]] = ctx.get("final_results")
    if not agent_span or not llm_tracker or not tool_tracker or final_results is None:
        return

    message_type = type(message).__name__

    if message_type == "AssistantMessage":
        final_content = llm_tracker.start_llm_span(
            message, ctx.get("prompt"), final_results
        )
        if final_content:
            final_results.append(final_content)
        tool_tracker.on_assistant_message(message)

    elif message_type == "UserMessage":
        tool_tracker.on_user_message(message)
        if hasattr(message, "content"):
            content = _serialize_content_blocks(message.content)
            final_results.append({"content": content, "role": "user"})
        llm_tracker.mark_next_llm_start()

    elif message_type == "ResultMessage":
        if hasattr(message, "usage"):
            usage_metrics = _extract_usage_from_result_message(message)
            llm_tracker.log_usage(usage_metrics)
        result_metadata = {
            k: v
            for k, v in {
                "num_turns": getattr(message, "num_turns", None),
                "session_id": getattr(message, "session_id", None),
            }.items()
            if v is not None
        }
        if result_metadata:
            for key, value in result_metadata.items():
                agent_span.set_attribute(
                    f"agent.{key}",
                    serialize_attribute(value, safe_serialize),
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

    agent_span_context = Tracer.start_as_current_span(
        span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "agent"}
    )
    agent_span = agent_span_context.__enter__()
    ctx["agent_span"] = agent_span
    ctx["agent_span_ctx"] = agent_span_context

    if prompt:
        agent_span.set_attribute(
            AttributeKeys.JUDGMENT_INPUT,
            serialize_attribute(prompt, safe_serialize),
        )
    Tracer._emit_partial()

    ts.parent_context = set_span_in_context(
        agent_span, Tracer._get_proxy_provider().get_current_context()
    )
    ctx["llm_tracker"] = LLMSpanTracker(query_start_time=start_time)
    ctx["tool_tracker"] = ToolSpanTracker(state=ts)


def _yield_hook(ctx: Ctx, message: Any) -> None:
    _process_message(ctx, message)


def _make_post_hook(cs: Optional[_ClientTracingState] = None) -> Callable[[Ctx], None]:
    def hook(ctx: Ctx) -> None:
        final_results: Optional[List[Dict[str, Any]]] = ctx.get("final_results")
        agent_span: Optional[Span] = ctx.get("agent_span")
        if agent_span and final_results:
            agent_span.set_attribute(
                AttributeKeys.JUDGMENT_OUTPUT,
                serialize_attribute(final_results[-1], safe_serialize),
            )
        if cs is not None and final_results:
            prompt = ctx.get("prompt")
            if prompt:
                cs.conversation_history.append({"content": prompt, "role": "user"})
            cs.conversation_history.extend(final_results)

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
        agent_span_context = ctx.get("agent_span_ctx")
        if agent_span_context is not None:
            agent_span_context.__exit__(None, None, None)
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
    """Creates a wrapper class for ClaudeSDKClient that wraps query and receive_response."""
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
    """Wraps the standalone query() function to add tracing."""
    finally_hook = _make_finally_hook(state)

    def pre_hook(ctx: Ctx, *args: Any, **kwargs: Any) -> None:
        prompt = kwargs.get("prompt") or (args[0] if args else None)
        _init_agent_span(
            ctx,
            state,
            prompt if isinstance(prompt, str) else None,
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
    """Converts content blocks to a serializable format with proper type fields."""
    if isinstance(content, list):
        result = []
        for block in content:
            if dataclasses.is_dataclass(block) and not isinstance(block, type):
                serialized = dataclasses.asdict(block)  # type: ignore

                block_type = type(block).__name__
                if block_type == "TextBlock":
                    serialized["type"] = "text"
                elif block_type == "ToolUseBlock":
                    serialized["type"] = "tool_use"
                elif block_type == "ToolResultBlock":
                    serialized["type"] = "tool_result"

                    content_value = serialized.get("content")
                    if isinstance(content_value, list) and len(content_value) == 1:
                        item = content_value[0]
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and "text" in item
                        ):
                            serialized["content"] = item["text"]

                    if "is_error" in serialized and serialized["is_error"] is None:
                        del serialized["is_error"]
                elif block_type == "ThinkingBlock":
                    serialized["type"] = "thinking"
            else:
                serialized = block

            result.append(serialized)
        return result
    return content


def _build_llm_input(
    prompt: Any, conversation_history: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """Builds the input array for an LLM span from the initial prompt and conversation history."""
    if isinstance(prompt, str):
        if len(conversation_history) == 0:
            return [{"content": prompt, "role": "user"}]
        else:
            return [{"content": prompt, "role": "user"}] + conversation_history

    return conversation_history if conversation_history else None


def _extract_usage_from_result_message(result_message: Any) -> Dict[str, Any]:
    """Extracts and normalizes usage metrics from a ResultMessage."""
    if not hasattr(result_message, "usage"):
        return {}

    usage = result_message.usage
    if not usage:
        return {}

    metrics: Dict[str, Any] = {}

    # Handle both dict and object with attributes
    def get_value(key: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(key)
        return getattr(usage, key, None)

    input_tokens = get_value("input_tokens")
    if input_tokens is not None:
        metrics[AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS] = input_tokens

    output_tokens = get_value("output_tokens")
    if output_tokens is not None:
        metrics[AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS] = output_tokens

    cache_creation_input_tokens = get_value("cache_creation_input_tokens")
    if cache_creation_input_tokens is not None:
        metrics[AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
            cache_creation_input_tokens
        )

    cache_read_input_tokens = get_value("cache_read_input_tokens")
    if cache_read_input_tokens is not None:
        metrics[AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS] = (
            cache_read_input_tokens
        )

    metrics[AttributeKeys.JUDGMENT_USAGE_METADATA] = safe_serialize(usage)

    return metrics
