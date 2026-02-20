"""Wrapper implementation for Claude Agent SDK."""

from __future__ import annotations
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

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.v1.tracer.tracer import BaseTracer


class TracingState:
    """Shared mutable state for a single setup_claude_agent_sdk() call.

    Passed through closures to all wrapper functions, providing:
    - parent_context: The current agent span context for tool span nesting.
      Set by receive_response/wrapped_query, read by tool handlers and
      BuiltInToolSpanTracker. This works even when tool handlers run in
      a separate asyncio Task (as the Claude Agent SDK's MCP server does)
      because all closures share the same TracingState instance.
    - sdk_tool_names: Set of SDK-defined tool names to avoid duplicate spans
      between _wrap_tool_handler and BuiltInToolSpanTracker.
    """

    def __init__(self) -> None:
        self.parent_context: Any = None
        self.sdk_tool_names: set = set()


class BuiltInToolSpanTracker:
    """Captures tool call spans from the message stream for built-in tools.

    Built-in tools (file edit, bash, etc.) are handled by the Claude Code CLI
    and appear in the message stream as ToolUseBlock in AssistantMessage content
    followed by ToolResultBlock in UserMessage content. SDK-defined tools already
    get spans from _wrap_tool_handler, so this tracker skips tools it knows are
    handled by the SDK to avoid duplicate spans.

    Flow:
    1. AssistantMessage contains ToolUseBlock → open a tool span
    2. UserMessage contains ToolResultBlock → match by tool_use_id, record output, close span
    """

    def __init__(self, tracer: "BaseTracer", state: TracingState):
        self.tracer = tracer
        self.state = state
        self._pending_spans: Dict[str, Tuple[Any, Any]] = {}

    def on_assistant_message(self, message: Any) -> None:
        """Extract ToolUseBlock entries and open tool spans."""
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

            # SDK tools are registered with their base name (e.g. "get_weather"),
            # but appear in the message stream with MCP prefix
            # (e.g. "mcp__server__get_weather"). Check both forms.
            base_name = _extract_base_tool_name(tool_name)
            if (
                tool_name in self.state.sdk_tool_names
                or base_name in self.state.sdk_tool_names
            ):
                continue

            ctx = self.state.parent_context

            span = self.tracer.get_tracer().start_span(
                str(tool_name),
                context=ctx,
                attributes={
                    AttributeKeys.JUDGMENT_SPAN_KIND: "tool",
                },
            )

            self.tracer.set_span_attribute(
                span, AttributeKeys.JUDGMENT_INPUT, safe_serialize(tool_input)
            )

            self._pending_spans[tool_use_id] = (span, tool_name)

    def on_user_message(self, message: Any) -> None:
        """Match ToolResultBlock entries to pending spans and close them."""
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

            self.tracer.set_span_attribute(
                span, AttributeKeys.JUDGMENT_OUTPUT, safe_serialize(result_content)
            )

            if is_error:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, "Tool returned an error"))

            span.end()

    def cleanup(self) -> None:
        """End any unclosed tool spans."""
        for span, _ in self._pending_spans.values():
            span.end()
        self._pending_spans.clear()


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
                self.tracer.set_span_attribute(self.current_span, key, value)

    def cleanup(self) -> None:
        """End any unclosed spans."""
        if self.current_span_context:
            self.current_span_context.__exit__(None, None, None)
        self.current_span = None
        self.current_span_context = None


def _create_client_wrapper_class(
    original_client_class: Any, tracer: "BaseTracer", state: TracingState
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
            agent_span_context = tracer.get_tracer().start_as_current_span(
                "Claude_Agent",
                attributes={
                    AttributeKeys.JUDGMENT_SPAN_KIND: "agent",
                },
            )
            agent_span = agent_span_context.__enter__()

            # Record input
            if self.__last_prompt:
                tracer.set_span_attribute(
                    agent_span,
                    AttributeKeys.JUDGMENT_INPUT,
                    self.__last_prompt,
                )

            tracer.emit_partial()

            # Store the parent span context on the shared state so tool handlers
            # can access it — even when they run in a separate asyncio Task
            # (as the Claude Agent SDK's MCP server does).
            state.parent_context = set_span_in_context(agent_span, tracer.get_context())

            final_results: List[Dict[str, Any]] = []
            llm_tracker = LLMSpanTracker(
                tracer, query_start_time=self.__query_start_time
            )
            tool_tracker = BuiltInToolSpanTracker(tracer, state=state)

            try:
                async for message in generator:
                    message_type = type(message).__name__

                    if message_type == "AssistantMessage":
                        final_content = llm_tracker.start_llm_span(
                            message, self.__last_prompt, final_results
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
                                tracer.set_span_attribute(
                                    agent_span, f"agent.{key}", value
                                )

                    yield message

                # Record output
                if final_results:
                    tracer.set_span_attribute(
                        agent_span,
                        AttributeKeys.JUDGMENT_OUTPUT,
                        final_results[-1] if final_results else None,
                    )

            except Exception as e:
                agent_span.record_exception(e)
                raise
            finally:
                tool_tracker.cleanup()
                llm_tracker.cleanup()
                agent_span_context.__exit__(None, None, None)
                state.parent_context = None

    return WrappedClaudeSDKClient


def _create_tool_wrapper_class(
    original_tool_class: Any, tracer: "BaseTracer", state: TracingState
) -> Any:
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
            wrapped_handler = _wrap_tool_handler(tracer, handler, name, state)
            super().__init__(name, description, input_schema, wrapped_handler, **kwargs)

        # Preserve generic typing support
        __class_getitem__ = classmethod(lambda cls, params: cls)  # type: ignore

    return WrappedSdkMcpTool


def _wrap_query_function(
    original_query_fn: Any, tracer: "BaseTracer", state: TracingState
) -> Callable[..., Any]:
    """Wraps the standalone query() function to add tracing."""

    async def wrapped_query(*args: Any, **kwargs: Any) -> Any:
        """Wrapped query function with automatic tracing."""
        # Create agent span for the query
        agent_span_context = tracer.get_tracer().start_as_current_span(
            "Claude_Agent_Query",
            attributes={
                AttributeKeys.JUDGMENT_SPAN_KIND: "agent",
            },
        )
        agent_span = agent_span_context.__enter__()

        # Capture prompt if available
        prompt = kwargs.get("prompt") or (args[0] if args else None)
        if prompt and isinstance(prompt, str):
            tracer.set_span_attribute(agent_span, AttributeKeys.JUDGMENT_INPUT, prompt)

        tracer.emit_partial()

        # Store parent context on shared state for tool handlers
        state.parent_context = set_span_in_context(agent_span, tracer.get_context())

        final_results: List[Dict[str, Any]] = []
        llm_tracker = LLMSpanTracker(tracer, query_start_time=time.time())
        tool_tracker = BuiltInToolSpanTracker(tracer, state=state)

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
                            tracer.set_span_attribute(agent_span, f"agent.{key}", value)

                yield message

            # Record output
            if final_results:
                tracer.set_span_attribute(
                    agent_span,
                    AttributeKeys.JUDGMENT_OUTPUT,
                    final_results[-1] if final_results else None,
                )

        except Exception as e:
            agent_span.record_exception(e)
            raise
        finally:
            tool_tracker.cleanup()
            llm_tracker.cleanup()
            agent_span_context.__exit__(None, None, None)
            state.parent_context = None

    return wrapped_query


def _wrap_tool_factory(
    tool_fn: Any, tracer: "BaseTracer", state: TracingState
) -> Callable[..., Any]:
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
                    _wrap_tool_handler(tracer, original_handler, tool_name, state),
                )

            return tool_def

        return wrapped_decorator

    return wrapped_tool


def _wrap_create_sdk_mcp_server(
    original_fn: Callable[..., Any], tracer: "BaseTracer", state: TracingState
) -> Callable[..., Any]:
    """Wraps create_sdk_mcp_server to ensure all tools get traced handlers.

    Tools created before setup_claude_agent_sdk() won't have wrapped handlers.
    This wrapper retroactively wraps any unwrapped tool handlers so that every
    SDK-registered tool gets proper span tracking regardless of creation order.
    """

    def wrapped_create_sdk_mcp_server(*args: Any, **kwargs: Any) -> Any:
        tools = kwargs.get("tools") or (args[2] if len(args) > 2 else None)
        if tools:
            for tool_def in tools:
                handler = getattr(tool_def, "handler", None)
                if handler and not getattr(handler, "_judgeval_wrapped", False):
                    tool_name = getattr(tool_def, "name", "unknown")
                    tool_def.handler = _wrap_tool_handler(
                        tracer, handler, tool_name, state
                    )
        return original_fn(*args, **kwargs)

    return wrapped_create_sdk_mcp_server


def _wrap_tool_handler(
    tracer: "BaseTracer",
    handler: Callable[..., Any],
    tool_name: Any,
    state: TracingState,
) -> Callable[..., Any]:
    """Wraps a tool handler to add tracing.

    Reads the parent span context from the shared TracingState instance.
    This works across asyncio Task boundaries because all closures from the
    same setup_claude_agent_sdk() call share the same TracingState reference.
    """
    # Check if already wrapped to prevent double-wrapping
    if hasattr(handler, "_judgeval_wrapped"):
        return handler

    state.sdk_tool_names.add(str(tool_name))

    async def wrapped_handler(args: Any) -> Any:
        # Read parent context from shared state — works even when this handler
        # runs in a separate asyncio Task (MCP server) because all wrappers
        # share the same TracingState instance via closure.
        ctx = state.parent_context

        # Create tool span with explicit parent context to ensure proper nesting
        tracer_obj = tracer.get_tracer()
        span = tracer_obj.start_span(
            str(tool_name),
            context=ctx,
            attributes={
                AttributeKeys.JUDGMENT_SPAN_KIND: "tool",
            },
        )

        # Set this span as active in the context
        with tracer.use_span(span, end_on_exit=True):
            # Record input
            tracer.set_span_attribute(span, AttributeKeys.JUDGMENT_INPUT, args)
            tracer.emit_partial()

            try:
                result = await handler(args)

                # Record output
                tracer.set_span_attribute(span, AttributeKeys.JUDGMENT_OUTPUT, result)

                return result
            except Exception as e:
                span.record_exception(e)
                raise

    # Mark as wrapped to prevent double-wrapping
    wrapped_handler._judgeval_wrapped = True  # type: ignore
    return wrapped_handler


def _extract_base_tool_name(full_name: str) -> str:
    """Extract the base tool name from an MCP-prefixed name.

    MCP tools appear in the message stream as ``mcp__<server>__<tool_name>``.
    Returns the ``<tool_name>`` portion, or the original name if it's not
    MCP-prefixed.
    """
    if full_name.startswith("mcp__"):
        parts = full_name.split("__")
        if len(parts) >= 3:
            return parts[-1]
    return full_name


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

    # Create LLM span
    span_start_time = int(start_time * 1e9) if start_time is not None else None
    llm_span_context = tracer.get_tracer().start_as_current_span(
        "anthropic.messages.create",
        attributes={
            AttributeKeys.JUDGMENT_SPAN_KIND: "llm",
        },
        start_time=span_start_time,
    )
    llm_span = llm_span_context.__enter__()

    # Record attributes
    if model:
        tracer.set_span_attribute(
            llm_span, AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model
        )
        # Set provider to anthropic for cost calculation
        tracer.set_span_attribute(
            llm_span, AttributeKeys.JUDGMENT_LLM_PROVIDER, "anthropic"
        )

    if input_messages:
        tracer.set_span_attribute(
            llm_span, AttributeKeys.JUDGMENT_INPUT, input_messages
        )

    if outputs:
        tracer.set_span_attribute(llm_span, AttributeKeys.JUDGMENT_OUTPUT, outputs)

    tracer.emit_partial()

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
