"""Wrapper implementation for Claude Agent SDK."""

from __future__ import annotations
import dataclasses
import threading
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

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize, serialize_attribute
from judgeval.v1.trace.proxy_tracer_provider import ProxyTracerProvider

if TYPE_CHECKING:
    from judgeval.v1.trace import BaseTracer
# Thread-local storage to propagate parent span context to tool handlers
# Claude Agent SDK breaks OpenTelemetry's automatic context propagation
# when executing tools, so we need to explicitly store and pass the context
_thread_local = threading.local()


class LLMSpanTracker:
    """Manages LLM span lifecycle for Claude Agent SDK message streams.

    Message flow per turn:
    1. UserMessage (tool results) → mark the time when next LLM will start
    2. AssistantMessage - LLM response arrives → create span with the marked start time, ending previous span
    3. ResultMessage - usage metrics → log to span

    We end the previous span when the next AssistantMessage arrives, using the marked
    start time to ensure sequential timing (no overlapping LLM spans).
    """

    def __init__(self, tracer: BaseTracer, query_start_time: Optional[float] = None):
        self.tracer = tracer
        self.current_span: Optional[Any] = None
        self.current_span_context: Optional[Any] = None
        self.next_start_time: Optional[float] = query_start_time

    def start_llm_span(
        self, message: Any, prompt: Any, conversation_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Start a new LLM span, ending the previous one if it exists."""
        # Use the marked start time, or current time as fallback
        start_time = (
            self.next_start_time if self.next_start_time is not None else time.time()
        )

        # End the previous span - only use __exit__ as it calls end() internally
        if self.current_span_context:
            self.current_span_context.__exit__(None, None, None)

        final_content, span, span_context = _create_llm_span_for_messages(
            self.tracer,
            [message],
            prompt,
            conversation_history,
            start_time=start_time,
        )
        self.current_span = span
        self.current_span_context = span_context
        self.next_start_time = None  # Reset for next span
        return final_content

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


def _create_client_wrapper_class(
    original_client_class: Any, tracer: "BaseTracer"
) -> Any:
    """Creates a wrapper class for ClaudeSDKClient that wraps query and receive_response."""

    class WrappedClaudeSDKClient(original_client_class):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.__last_prompt: Optional[str] = None
            self.__query_start_time: Optional[float] = None

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            """Wrap query to capture the prompt and start time for tracing."""
            # Capture the time when query is called (when LLM call starts)
            self.__query_start_time = time.time()

            # Capture the prompt for use in receive_response
            if args:
                self.__last_prompt = str(args[0])
            elif "prompt" in kwargs:
                self.__last_prompt = str(kwargs["prompt"])

            return await super().query(*args, **kwargs)

        async def receive_response(self) -> AsyncGenerator[Any, None]:
            """Wrap receive_response to add tracing with proper span hierarchy."""
            generator = super().receive_response()

            # Create TASK span for the entire agent conversation
            agent_span_context = (
                ProxyTracerProvider.get_instance()
                .get_tracer(__name__)
                .start_as_current_span(
                    "Claude_Agent",
                    attributes={
                        AttributeKeys.JUDGMENT_SPAN_KIND: "agent",
                    },
                )
            )
            agent_span = agent_span_context.__enter__()

            if self.__last_prompt:
                agent_span.set_attribute(
                    AttributeKeys.JUDGMENT_INPUT,
                    serialize_attribute(self.__last_prompt, safe_serialize),
                )

            parent_context = set_span_in_context(
                agent_span, ProxyTracerProvider.get_instance().get_current_context()
            )
            _thread_local.parent_context = parent_context

            final_results: List[Dict[str, Any]] = []
            llm_tracker = LLMSpanTracker(
                tracer, query_start_time=self.__query_start_time
            )

            try:
                async for message in generator:
                    message_type = type(message).__name__

                    if message_type == "AssistantMessage":
                        final_content = llm_tracker.start_llm_span(
                            message, self.__last_prompt, final_results
                        )
                        if final_content:
                            final_results.append(final_content)

                    elif message_type == "UserMessage":
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

                    yield message

                if final_results:
                    agent_span.set_attribute(
                        AttributeKeys.JUDGMENT_OUTPUT,
                        serialize_attribute(
                            final_results[-1] if final_results else None, safe_serialize
                        ),
                    )

            except Exception as e:
                agent_span.record_exception(e)
                raise
            finally:
                llm_tracker.cleanup()
                agent_span_context.__exit__(None, None, None)
                if hasattr(_thread_local, "parent_context"):
                    delattr(_thread_local, "parent_context")

    return WrappedClaudeSDKClient


def _create_tool_wrapper_class(original_tool_class: Any, tracer: "BaseTracer") -> Any:
    """Creates a wrapper class for SdkMcpTool that wraps handlers."""

    class WrappedSdkMcpTool(original_tool_class):  # type: ignore
        def __init__(
            self,
            name: Any,
            description: Any,
            input_schema: Any,
            handler: Any,
            **kwargs: Any,
        ):
            wrapped_handler = _wrap_tool_handler(tracer, handler, name)
            super().__init__(name, description, input_schema, wrapped_handler, **kwargs)

        # Preserve generic typing support
        __class_getitem__ = classmethod(lambda cls, params: cls)  # type: ignore

    return WrappedSdkMcpTool


def _wrap_query_function(
    original_query_fn: Any, tracer: "BaseTracer"
) -> Callable[..., Any]:
    """Wraps the standalone query() function to add tracing."""

    async def wrapped_query(*args: Any, **kwargs: Any) -> Any:
        """Wrapped query function with automatic tracing."""
        agent_span_context = (
            ProxyTracerProvider.get_instance()
            .get_tracer(__name__)
            .start_as_current_span(
                "Claude_Agent_Query",
                attributes={
                    AttributeKeys.JUDGMENT_SPAN_KIND: "agent",
                },
            )
        )
        agent_span = agent_span_context.__enter__()

        prompt = kwargs.get("prompt") or (args[0] if args else None)
        if prompt and isinstance(prompt, str):
            agent_span.set_attribute(
                AttributeKeys.JUDGMENT_INPUT,
                serialize_attribute(prompt, safe_serialize),
            )

        parent_context = set_span_in_context(
            agent_span, ProxyTracerProvider.get_instance().get_current_context()
        )
        _thread_local.parent_context = parent_context

        final_results: List[Dict[str, Any]] = []
        llm_tracker = LLMSpanTracker(tracer, query_start_time=time.time())

        try:
            # Call original query function
            async for message in original_query_fn(*args, **kwargs):
                message_type = type(message).__name__

                if message_type == "AssistantMessage":
                    final_content = llm_tracker.start_llm_span(
                        message,
                        prompt if isinstance(prompt, str) else None,
                        final_results,
                    )
                    if final_content:
                        final_results.append(final_content)

                elif message_type == "UserMessage":
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

                yield message

            if final_results:
                agent_span.set_attribute(
                    AttributeKeys.JUDGMENT_OUTPUT,
                    serialize_attribute(
                        final_results[-1] if final_results else None, safe_serialize
                    ),
                )

        except Exception as e:
            agent_span.record_exception(e)
            raise
        finally:
            llm_tracker.cleanup()
            agent_span_context.__exit__(None, None, None)
            if hasattr(_thread_local, "parent_context"):
                delattr(_thread_local, "parent_context")

    return wrapped_query


def _wrap_tool_factory(tool_fn: Any, tracer: "BaseTracer") -> Callable[..., Any]:
    """Wraps the tool() factory function to return wrapped tools."""

    def wrapped_tool(*args: Any, **kwargs: Any) -> Any:
        result = tool_fn(*args, **kwargs)

        # The tool() function returns a decorator, not a tool definition
        # We need to wrap the decorator to intercept the final tool definition
        if not callable(result):
            return result

        def wrapped_decorator(handler_fn: Any) -> Any:
            tool_def = result(handler_fn)

            # Now we have the actual tool definition, wrap its handler
            if tool_def and hasattr(tool_def, "handler"):
                tool_name = getattr(tool_def, "name", "unknown")
                original_handler = getattr(tool_def, "handler")
                setattr(
                    tool_def,
                    "handler",
                    _wrap_tool_handler(tracer, original_handler, tool_name),
                )

            return tool_def

        return wrapped_decorator

    return wrapped_tool


def _wrap_tool_handler(
    tracer: "BaseTracer", handler: Callable[..., Any], tool_name: Any
) -> Callable[..., Any]:
    """Wraps a tool handler to add tracing.

    Claude Agent SDK breaks OpenTelemetry's automatic context propagation,
    so we retrieve the parent context from thread-local storage and use it
    explicitly when creating tool spans to ensure proper nesting.
    """
    # Check if already wrapped to prevent double-wrapping
    if hasattr(handler, "_judgeval_wrapped"):
        return handler

    async def wrapped_handler(args: Any) -> Any:
        # Get parent context from thread-local storage
        # Claude Agent SDK breaks context propagation, so we stored it explicitly
        parent_context = getattr(_thread_local, "parent_context", None)

        ctx = parent_context if parent_context is not None else None

        proxy = ProxyTracerProvider.get_instance()
        span = proxy.get_tracer(__name__).start_span(
            str(tool_name),
            context=ctx,
            attributes={
                AttributeKeys.JUDGMENT_SPAN_KIND: "tool",
            },
        )

        with proxy.use_span(span, end_on_exit=True):
            span.set_attribute(
                AttributeKeys.JUDGMENT_INPUT, serialize_attribute(args, safe_serialize)
            )

            try:
                result = await handler(args)

                span.set_attribute(
                    AttributeKeys.JUDGMENT_OUTPUT,
                    serialize_attribute(result, safe_serialize),
                )

                return result
            except Exception as e:
                span.record_exception(e)
                raise

    # Mark as wrapped to prevent double-wrapping
    wrapped_handler._judgeval_wrapped = True  # type: ignore
    return wrapped_handler


def _create_llm_span_for_messages(
    tracer: "BaseTracer",
    messages: List[Any],  # List of AssistantMessage objects
    prompt: Any,
    conversation_history: List[Dict[str, Any]],
    start_time: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[Any]]:
    """Creates an LLM span for a group of AssistantMessage objects.

    Returns a tuple of (final_content, span, span_context):
    - final_content: The final message content to add to conversation history
    - span: The LLM span object (for logging metrics later)
    - span_context: The span context manager
    """
    if not messages:
        return None, None, None

    last_message = messages[-1]
    if type(last_message).__name__ != "AssistantMessage":
        return None, None, None

    model = getattr(last_message, "model", None)
    input_messages = _build_llm_input(prompt, conversation_history)

    outputs: List[Dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "content"):
            content = _serialize_content_blocks(msg.content)
            outputs.append({"content": content, "role": "assistant"})

    llm_span_context = (
        ProxyTracerProvider.get_instance()
        .get_tracer(__name__)
        .start_as_current_span(
            "anthropic.messages.create",
            attributes={
                AttributeKeys.JUDGMENT_SPAN_KIND: "llm",
            },
        )
    )
    llm_span = llm_span_context.__enter__()

    if model:
        llm_span.set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
            serialize_attribute(model, safe_serialize),
        )
        llm_span.set_attribute(
            AttributeKeys.JUDGMENT_LLM_PROVIDER,
            serialize_attribute("anthropic", safe_serialize),
        )

    if input_messages:
        llm_span.set_attribute(
            AttributeKeys.JUDGMENT_INPUT,
            serialize_attribute(input_messages, safe_serialize),
        )

    if outputs:
        llm_span.set_attribute(
            AttributeKeys.JUDGMENT_OUTPUT, serialize_attribute(outputs, safe_serialize)
        )

    # Return final message content for conversation history and the span
    if hasattr(last_message, "content"):
        content = _serialize_content_blocks(last_message.content)
        return {"content": content, "role": "assistant"}, llm_span, llm_span_context

    return None, llm_span, llm_span_context


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

                    # Simplify content if it's a single text block
                    content_value = serialized.get("content")
                    if isinstance(content_value, list) and len(content_value) == 1:
                        item = content_value[0]
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and "text" in item
                        ):
                            serialized["content"] = item["text"]

                    # Remove None is_error
                    if "is_error" in serialized and serialized["is_error"] is None:
                        del serialized["is_error"]
                elif block_type == "ThinkingBlock":
                    serialized["type"] = "thinking"
            else:
                serialized = block

            result.append(serialized)
        return result
    return content


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
