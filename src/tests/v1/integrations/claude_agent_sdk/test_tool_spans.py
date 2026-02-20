"""Tests for Claude Agent SDK tool call span capture.

Verifies that the JudgeVal integration correctly captures tool call spans
when using SdkMcpTool, @tool decorator, and built-in tools from the message stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import set_span_in_context

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
    TracingState,
    _wrap_tool_handler,
    _create_tool_wrapper_class,
    _wrap_tool_factory,
    _wrap_create_sdk_mcp_server,
    _extract_base_tool_name,
    LLMSpanTracker,
    BuiltInToolSpanTracker,
)


# ---------------------------------------------------------------------------
# Fake Claude Agent SDK types for unit testing (avoids needing the real SDK)
# ---------------------------------------------------------------------------


@dataclass
class ToolUseBlock:
    """Named to match type().__name__ check in _serialize_content_blocks."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class TextBlock:
    """Named to match type().__name__ check in _serialize_content_blocks."""

    text: str


@dataclass
class ToolResultBlock:
    """Named to match type().__name__ check in _serialize_content_blocks."""

    tool_use_id: str
    content: Any = None
    is_error: Optional[bool] = None


@dataclass
class AssistantMessage:
    """Named to match type().__name__ check in _create_llm_span_for_messages."""

    content: list
    model: str = "claude-sonnet-4-20250514"
    parent_tool_use_id: Optional[str] = None


@dataclass
class FakeUserMessage:
    content: Any = None
    uuid: Optional[str] = None
    parent_tool_use_id: Optional[str] = None
    tool_use_result: Optional[Dict[str, Any]] = None


@dataclass
class FakeResultMessage:
    subtype: str = "result"
    duration_ms: int = 100
    duration_api_ms: int = 90
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "test-session"
    total_cost_usd: Optional[float] = 0.01
    usage: Optional[Dict[str, Any]] = None
    result: Optional[str] = None


@dataclass
class FakeSdkMcpTool:
    """Simulates the real SdkMcpTool dataclass."""

    name: str
    description: str
    input_schema: Any
    handler: Any
    annotations: Any = None


# ---------------------------------------------------------------------------
# Helpers to create a real tracer with in-memory span collection
# ---------------------------------------------------------------------------


class InMemoryCollector:
    """Collects exported spans for assertions."""

    def __init__(self):
        self.spans = []

    def get_exporter(self):
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

        collector = self

        class _Exporter(SpanExporter):
            def export(self, spans):
                collector.spans.extend(spans)
                return SpanExportResult.SUCCESS

            def shutdown(self):
                pass

        return _Exporter()


def _make_tracer(collector: InMemoryCollector):
    """Create a real BaseTracer backed by in-memory span collection."""
    from judgeval.v1.tracer.tracer import Tracer

    mock_client = MagicMock()
    mock_client.api_key = "test_key"
    mock_client.organization_id = "test_org"
    mock_client.base_url = "http://test.com"

    tracer = Tracer(
        project_name="test-claude-tool-spans",
        project_id="test-project-id",
        enable_evaluation=False,
        enable_monitoring=True,
        api_client=mock_client,
        serializer=lambda x: str(x),
        isolated=True,
        use_default_span_processor=False,
    )

    processor = SimpleSpanProcessor(collector.get_exporter())
    tracer._tracer_provider.add_span_processor(processor)
    return tracer


def _make_state_with_parent(tracer, parent_span) -> TracingState:
    """Create a TracingState with parent_context set from the given span."""
    state = TracingState()
    state.parent_context = set_span_in_context(parent_span, tracer.get_context())
    return state


# ---------------------------------------------------------------------------
# Tests for _wrap_tool_handler
# ---------------------------------------------------------------------------


class TestWrapToolHandler:
    """Test that _wrap_tool_handler creates proper tool spans."""

    @pytest.mark.asyncio
    async def test_tool_handler_creates_span(self):
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"content": [{"type": "text", "text": f"Result: {args['x']}"}]}

        with tracer.get_tracer().start_as_current_span("test_parent") as parent_span:
            state = _make_state_with_parent(tracer, parent_span)
            wrapped = _wrap_tool_handler(tracer, my_tool, "calculator", state)
            result = await wrapped({"x": 42})

        assert result == {"content": [{"type": "text", "text": "Result: 42"}]}

        tool_spans = [s for s in collector.spans if s.name == "calculator"]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        attrs = dict(tool_span.attributes)
        assert attrs[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"
        assert AttributeKeys.JUDGMENT_INPUT in attrs
        assert AttributeKeys.JUDGMENT_OUTPUT in attrs

    @pytest.mark.asyncio
    async def test_tool_handler_records_exception(self):
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def failing_tool(args):
            raise ValueError("tool failed")

        with tracer.get_tracer().start_as_current_span("test_parent") as parent_span:
            state = _make_state_with_parent(tracer, parent_span)
            wrapped = _wrap_tool_handler(tracer, failing_tool, "broken_tool", state)

            with pytest.raises(ValueError, match="tool failed"):
                await wrapped({"input": "data"})

        tool_spans = [s for s in collector.spans if s.name == "broken_tool"]
        assert len(tool_spans) == 1
        assert len(tool_spans[0].events) > 0

    @pytest.mark.asyncio
    async def test_tool_handler_prevents_double_wrapping(self):
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"result": "ok"}

        state = TracingState()
        wrapped_once = _wrap_tool_handler(tracer, my_tool, "tool1", state)
        wrapped_twice = _wrap_tool_handler(tracer, wrapped_once, "tool1", state)

        assert wrapped_once is wrapped_twice

    @pytest.mark.asyncio
    async def test_tool_handler_nests_under_parent(self):
        """Tool spans should be children of the agent span via shared state."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"result": "ok"}

        with tracer.get_tracer().start_as_current_span("agent_span") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            wrapped = _wrap_tool_handler(tracer, my_tool, "nested_tool", state)
            await wrapped({"a": 1})

        agent_spans = [s for s in collector.spans if s.name == "agent_span"]
        tool_spans = [s for s in collector.spans if s.name == "nested_tool"]

        assert len(agent_spans) == 1
        assert len(tool_spans) == 1

        agent_span_ctx = agent_spans[0].context
        tool_parent = tool_spans[0].parent

        assert tool_parent is not None
        assert tool_parent.span_id == agent_span_ctx.span_id
        assert tool_parent.trace_id == agent_span_ctx.trace_id


# ---------------------------------------------------------------------------
# Tests for _create_tool_wrapper_class (SdkMcpTool wrapping)
# ---------------------------------------------------------------------------


class TestCreateToolWrapperClass:
    """Test that wrapping SdkMcpTool properly wraps the handler."""

    @pytest.mark.asyncio
    async def test_wrapped_sdk_mcp_tool_captures_span(self):
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("parent") as parent_span:
            state = _make_state_with_parent(tracer, parent_span)
            WrappedTool = _create_tool_wrapper_class(FakeSdkMcpTool, tracer, state)

            async def my_handler(args):
                return {"content": [{"type": "text", "text": "hello"}]}

            tool = WrappedTool(
                name="greet",
                description="Greet someone",
                input_schema={"name": str},
                handler=my_handler,
            )

            assert tool.name == "greet"
            assert tool.description == "Greet someone"
            assert hasattr(tool.handler, "_judgeval_wrapped")

            result = await tool.handler({"name": "World"})

        assert result == {"content": [{"type": "text", "text": "hello"}]}

        tool_spans = [s for s in collector.spans if s.name == "greet"]
        assert len(tool_spans) == 1
        assert (
            dict(tool_spans[0].attributes)[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"
        )


# ---------------------------------------------------------------------------
# Tests for _wrap_tool_factory (@tool decorator wrapping)
# ---------------------------------------------------------------------------


class TestWrapToolFactory:
    """Test that wrapping the tool() factory function captures tool spans."""

    @pytest.mark.asyncio
    async def test_wrapped_tool_decorator_captures_span(self):
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        def fake_tool_fn(name, description, input_schema, annotations=None):
            def decorator(handler):
                return FakeSdkMcpTool(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    handler=handler,
                    annotations=annotations,
                )

            return decorator

        state = TracingState()
        wrapped_tool_fn = _wrap_tool_factory(fake_tool_fn, tracer, state)

        @wrapped_tool_fn("add", "Add numbers", {"a": float, "b": float})
        async def add(args):
            return {
                "content": [{"type": "text", "text": f"Sum: {args['a'] + args['b']}"}]
            }

        assert add.name == "add"
        assert hasattr(add.handler, "_judgeval_wrapped")

        with tracer.get_tracer().start_as_current_span("parent") as parent_span:
            state.parent_context = set_span_in_context(
                parent_span, tracer.get_context()
            )
            result = await add.handler({"a": 3.0, "b": 4.0})

        assert result == {"content": [{"type": "text", "text": "Sum: 7.0"}]}

        tool_spans = [s for s in collector.spans if s.name == "add"]
        assert len(tool_spans) == 1
        assert (
            dict(tool_spans[0].attributes)[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"
        )


# ---------------------------------------------------------------------------
# Tests for the full message stream (receive_response integration)
# ---------------------------------------------------------------------------


class TestMessageStreamToolSpans:
    """Test that tool calls visible in the message stream are properly handled."""

    @pytest.mark.asyncio
    async def test_tool_use_in_assistant_message_is_serialized(self):
        """When AssistantMessage contains ToolUseBlock, it should be captured in LLM span output."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        tool_use_block = ToolUseBlock(
            id="tool_123",
            name="calculator",
            input={"expression": "2+2"},
        )
        text_block = TextBlock(text="Let me calculate that for you.")

        assistant_msg = AssistantMessage(
            content=[text_block, tool_use_block],
            model="claude-sonnet-4-20250514",
        )

        tracker = LLMSpanTracker(tracer)
        with tracer.get_tracer().start_as_current_span("test_agent"):
            final_content = tracker.start_llm_span(assistant_msg, "what is 2+2?", [])
            tracker.cleanup()

        assert final_content is not None
        content_list = final_content.get("content", [])

        # Verify ToolUseBlock was serialized with proper type
        has_tool_use = (
            any(
                isinstance(item, dict) and item.get("type") == "tool_use"
                for item in content_list
            )
            if isinstance(content_list, list)
            else False
        )
        assert has_tool_use, (
            f"Expected tool_use block in serialized content, got: {content_list}"
        )

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_create_separate_spans(self):
        """Multiple tool handler invocations should each get their own span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def tool_a(args):
            return {"result": "a"}

        async def tool_b(args):
            return {"result": "b"}

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            wrapped_a = _wrap_tool_handler(tracer, tool_a, "tool_a", state)
            wrapped_b = _wrap_tool_handler(tracer, tool_b, "tool_b", state)

            await wrapped_a({"input": "1"})
            await wrapped_b({"input": "2"})

        tool_a_spans = [s for s in collector.spans if s.name == "tool_a"]
        tool_b_spans = [s for s in collector.spans if s.name == "tool_b"]

        assert len(tool_a_spans) == 1
        assert len(tool_b_spans) == 1

        # Both should be under the same agent span
        agent_spans = [s for s in collector.spans if s.name == "agent"]
        assert len(agent_spans) == 1
        agent_ctx = agent_spans[0].context

        for ts in [tool_a_spans[0], tool_b_spans[0]]:
            assert ts.parent is not None
            assert ts.parent.trace_id == agent_ctx.trace_id

    @pytest.mark.asyncio
    async def test_tool_span_has_input_output_attributes(self):
        """Tool spans should record both input and output."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def echo_tool(args):
            return {"echo": args["message"]}

        with tracer.get_tracer().start_as_current_span("parent") as parent_span:
            state = _make_state_with_parent(tracer, parent_span)
            wrapped = _wrap_tool_handler(tracer, echo_tool, "echo", state)
            result = await wrapped({"message": "hello world"})

        assert result == {"echo": "hello world"}

        echo_spans = [s for s in collector.spans if s.name == "echo"]
        assert len(echo_spans) == 1

        attrs = dict(echo_spans[0].attributes)
        input_val = attrs[AttributeKeys.JUDGMENT_INPUT]
        output_val = attrs[AttributeKeys.JUDGMENT_OUTPUT]

        assert "hello world" in str(input_val)
        assert "hello world" in str(output_val)


# ---------------------------------------------------------------------------
# Tests for BuiltInToolSpanTracker (message-stream-based tool spans)
# ---------------------------------------------------------------------------


class TestBuiltInToolSpanTracker:
    """Test that built-in tool calls from the message stream are captured as spans."""

    @pytest.mark.asyncio
    async def test_builtin_tool_creates_span_from_message_stream(self):
        """A ToolUseBlock in AssistantMessage + ToolResultBlock in UserMessage = tool span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            # Simulate AssistantMessage with a ToolUseBlock
            assistant_msg = AssistantMessage(
                content=[
                    TextBlock(text="I'll edit that file for you."),
                    ToolUseBlock(
                        id="tool_abc",
                        name="file_edit",
                        input={"path": "/tmp/test.py", "content": "print('hi')"},
                    ),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            # Simulate UserMessage with matching ToolResultBlock
            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="tool_abc",
                        content="File written successfully",
                    ),
                ]
            )
            tracker.on_user_message(user_msg)

        tool_spans = [s for s in collector.spans if s.name == "file_edit"]
        assert len(tool_spans) == 1

        attrs = dict(tool_spans[0].attributes)
        assert attrs[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"
        assert AttributeKeys.JUDGMENT_INPUT in attrs
        assert AttributeKeys.JUDGMENT_OUTPUT in attrs
        assert "/tmp/test.py" in str(attrs[AttributeKeys.JUDGMENT_INPUT])
        assert "File written" in str(attrs[AttributeKeys.JUDGMENT_OUTPUT])

    @pytest.mark.asyncio
    async def test_builtin_tool_error_sets_error_status(self):
        """Tool results with is_error=True should set error status on the span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="tool_err",
                        name="bash",
                        input={"command": "rm -rf /"},
                    ),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="tool_err",
                        content="Permission denied",
                        is_error=True,
                    ),
                ]
            )
            tracker.on_user_message(user_msg)

        tool_spans = [s for s in collector.spans if s.name == "bash"]
        assert len(tool_spans) == 1

        from opentelemetry.trace import StatusCode

        assert tool_spans[0].status.status_code == StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_sdk_tools_are_skipped(self):
        """SDK-defined tools should not get duplicate spans from the message stream."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            # "calculator" is an SDK-defined tool
            state.sdk_tool_names.add("calculator")
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            # SDK tool: should be skipped
            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="tool_sdk",
                        name="calculator",
                        input={"expression": "2+2"},
                    ),
                    # Built-in tool: should create a span
                    ToolUseBlock(
                        id="tool_builtin",
                        name="bash",
                        input={"command": "echo hello"},
                    ),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(tool_use_id="tool_sdk", content="4"),
                    ToolResultBlock(tool_use_id="tool_builtin", content="hello"),
                ]
            )
            tracker.on_user_message(user_msg)

        calc_spans = [s for s in collector.spans if s.name == "calculator"]
        bash_spans = [s for s in collector.spans if s.name == "bash"]

        assert len(calc_spans) == 0, "SDK tool should not create message-stream span"
        assert len(bash_spans) == 1, "Built-in tool should create message-stream span"

    @pytest.mark.asyncio
    async def test_mcp_prefixed_sdk_tools_are_skipped(self):
        """SDK tools appearing with MCP prefix (mcp__server__tool) should not get duplicate spans."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            # "get_weather" is the base name registered by _wrap_tool_handler
            state.sdk_tool_names.add("get_weather")
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            # Message stream uses MCP-prefixed name
            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="tool_mcp",
                        name="mcp__weather__get_weather",
                        input={"city": "SF"},
                    ),
                    # Built-in tool: should still create a span
                    ToolUseBlock(
                        id="tool_bash",
                        name="bash",
                        input={"command": "echo hi"},
                    ),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(tool_use_id="tool_mcp", content="62F"),
                    ToolResultBlock(tool_use_id="tool_bash", content="hi"),
                ]
            )
            tracker.on_user_message(user_msg)

        weather_spans = [s for s in collector.spans if "weather" in s.name.lower()]
        bash_spans = [s for s in collector.spans if s.name == "bash"]

        assert len(weather_spans) == 0, (
            "MCP-prefixed SDK tool should not create message-stream span"
        )
        assert len(bash_spans) == 1, "Built-in tool should still create a span"

    @pytest.mark.asyncio
    async def test_multiple_builtin_tools_in_same_message(self):
        """Multiple ToolUseBlocks in a single AssistantMessage."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(id="t1", name="read_file", input={"path": "a.py"}),
                    ToolUseBlock(id="t2", name="write_file", input={"path": "b.py"}),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t1", content="file contents"),
                    ToolResultBlock(tool_use_id="t2", content="written"),
                ]
            )
            tracker.on_user_message(user_msg)

        read_spans = [s for s in collector.spans if s.name == "read_file"]
        write_spans = [s for s in collector.spans if s.name == "write_file"]

        assert len(read_spans) == 1
        assert len(write_spans) == 1

    @pytest.mark.asyncio
    async def test_cleanup_ends_unclosed_spans(self):
        """Cleanup should end spans that never got a matching ToolResultBlock."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(id="orphan", name="abandoned_tool", input={"x": 1}),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            # No matching UserMessage - force cleanup
            tracker.cleanup()

        orphan_spans = [s for s in collector.spans if s.name == "abandoned_tool"]
        assert len(orphan_spans) == 1

    @pytest.mark.asyncio
    async def test_builtin_tool_nests_under_agent(self):
        """Built-in tool spans should be children of the agent span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"}),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t1", content="file1\nfile2"),
                ]
            )
            tracker.on_user_message(user_msg)

        agent_spans = [s for s in collector.spans if s.name == "agent"]
        tool_spans = [s for s in collector.spans if s.name == "bash"]

        assert len(agent_spans) == 1
        assert len(tool_spans) == 1

        assert tool_spans[0].parent is not None
        assert tool_spans[0].parent.span_id == agent_spans[0].context.span_id


# ---------------------------------------------------------------------------
# Tests for emit_partial integration
# ---------------------------------------------------------------------------


class TestEmitPartial:
    """Test that emit_partial() is called at key points for real-time UI updates."""

    @pytest.mark.asyncio
    async def test_sdk_tool_handler_calls_emit_partial(self):
        """SDK tool handler should call emit_partial after recording input."""
        from unittest.mock import patch
        from judgeval.v1.tracer.base_tracer import BaseTracer

        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"result": "done"}

        with tracer.get_tracer().start_as_current_span("parent") as parent_span:
            state = _make_state_with_parent(tracer, parent_span)
            wrapped = _wrap_tool_handler(tracer, my_tool, "traced_tool", state)

            with patch.object(
                BaseTracer, "emit_partial", wraps=tracer.emit_partial
            ) as mock_emit:
                await wrapped({"x": 1})

        assert mock_emit.call_count >= 1, "emit_partial should be called for tool spans"

    @pytest.mark.asyncio
    async def test_builtin_tool_tracker_does_not_call_emit_partial(self):
        """BuiltInToolSpanTracker should NOT call emit_partial when opening a tool span.

        The span is emitted when it ends (in on_user_message), not when opened.
        """
        from unittest.mock import patch
        from judgeval.v1.tracer.base_tracer import BaseTracer

        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = BuiltInToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"}),
                ]
            )
            with patch.object(
                BaseTracer, "emit_partial", wraps=tracer.emit_partial
            ) as mock_emit:
                tracker.on_assistant_message(assistant_msg)

            tracker.cleanup()

        assert mock_emit.call_count == 0, (
            "emit_partial should not be called when opening built-in tool span"
        )

    @pytest.mark.asyncio
    async def test_llm_span_calls_emit_partial(self):
        """LLM span creation should call emit_partial after setting attributes."""
        from unittest.mock import patch
        from judgeval.v1.tracer.base_tracer import BaseTracer

        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        tracker = LLMSpanTracker(tracer)
        with tracer.get_tracer().start_as_current_span("agent"):
            with patch.object(
                BaseTracer, "emit_partial", wraps=tracer.emit_partial
            ) as mock_emit:
                tracker.start_llm_span(
                    AssistantMessage(
                        content=[TextBlock(text="Hello")],
                        model="claude-sonnet-4-20250514",
                    ),
                    "test prompt",
                    [],
                )
            tracker.cleanup()

        assert mock_emit.call_count >= 1, "emit_partial should be called for LLM spans"


# ---------------------------------------------------------------------------
# Tests for _wrap_create_sdk_mcp_server
# ---------------------------------------------------------------------------


class TestWrapCreateSdkMcpServer:
    """Tests for _wrap_create_sdk_mcp_server."""

    @pytest.mark.asyncio
    async def test_wraps_unwrapped_tool_handlers(self):
        """Tools created before patching get their handlers wrapped."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_handler(args):
            return {"content": [{"type": "text", "text": "ok"}]}

        tool_def = FakeSdkMcpTool(
            name="pre_patch_tool",
            description="created before setup",
            input_schema={"q": str},
            handler=my_handler,
        )

        assert not getattr(tool_def.handler, "_judgeval_wrapped", False)

        def fake_create(name, version="1.0.0", tools=None):
            return {"name": name, "tools": tools}

        state = TracingState()
        wrapped_create = _wrap_create_sdk_mcp_server(fake_create, tracer, state)
        result = wrapped_create("my_server", tools=[tool_def])

        assert getattr(tool_def.handler, "_judgeval_wrapped", False)
        assert result["name"] == "my_server"
        assert "pre_patch_tool" in state.sdk_tool_names

    @pytest.mark.asyncio
    async def test_does_not_double_wrap(self):
        """Already-wrapped handlers are left untouched."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_handler(args):
            return {"content": [{"type": "text", "text": "ok"}]}

        state = TracingState()
        wrapped = _wrap_tool_handler(tracer, my_handler, "already_wrapped", state)
        tool_def = FakeSdkMcpTool(
            name="already_wrapped",
            description="already wrapped",
            input_schema={},
            handler=wrapped,
        )

        original_handler = tool_def.handler

        def fake_create(name, version="1.0.0", tools=None):
            return {"name": name}

        wrapped_create = _wrap_create_sdk_mcp_server(fake_create, tracer, state)
        wrapped_create("srv", tools=[tool_def])

        assert tool_def.handler is original_handler

    @pytest.mark.asyncio
    async def test_no_tools_passes_through(self):
        """Calling without tools still works."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        def fake_create(name, version="1.0.0", tools=None):
            return {"name": name}

        state = TracingState()
        wrapped_create = _wrap_create_sdk_mcp_server(fake_create, tracer, state)
        result = wrapped_create("empty_server")
        assert result["name"] == "empty_server"


# ---------------------------------------------------------------------------
# Tests for shared TracingState context propagation
# ---------------------------------------------------------------------------


class TestTracingStateContextPropagation:
    """Test that tool spans use the shared TracingState for parent context."""

    @pytest.mark.asyncio
    async def test_tool_handler_uses_shared_state(self):
        """Tool handlers read parent context from TracingState, not ContextVar."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"content": [{"type": "text", "text": "ok"}]}

        state = TracingState()
        wrapped = _wrap_tool_handler(tracer, my_tool, "mcp_tool", state)

        # Set parent context on state (simulates receive_response setting it)
        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state.parent_context = set_span_in_context(agent_span, tracer.get_context())
            result = await wrapped({"x": 1})

        assert result == {"content": [{"type": "text", "text": "ok"}]}

        agent_spans = [s for s in collector.spans if s.name == "agent"]
        tool_spans = [s for s in collector.spans if s.name == "mcp_tool"]

        assert len(agent_spans) == 1
        assert len(tool_spans) == 1

        # Tool span should be a child of the agent span
        assert tool_spans[0].parent is not None
        assert tool_spans[0].parent.span_id == agent_spans[0].context.span_id

    @pytest.mark.asyncio
    async def test_separate_states_are_independent(self):
        """Two TracingState instances don't interfere with each other."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"result": "ok"}

        state_a = TracingState()
        state_b = TracingState()

        wrapped_a = _wrap_tool_handler(tracer, my_tool, "tool_a", state_a)

        async def other_tool(args):
            return {"result": "ok"}

        wrapped_b = _wrap_tool_handler(tracer, other_tool, "tool_b", state_b)

        with tracer.get_tracer().start_as_current_span("agent_a") as span_a:
            state_a.parent_context = set_span_in_context(span_a, tracer.get_context())
            await wrapped_a({"x": 1})

        # state_b has no parent context set — tool_b should still work
        await wrapped_b({"x": 2})

        tool_a_spans = [s for s in collector.spans if s.name == "tool_a"]
        tool_b_spans = [s for s in collector.spans if s.name == "tool_b"]

        assert len(tool_a_spans) == 1
        assert len(tool_b_spans) == 1

        # tool_a should be parented under agent_a
        assert tool_a_spans[0].parent is not None
        # tool_b has no parent context — should still be created (no crash)


# ---------------------------------------------------------------------------
# Tests for _extract_base_tool_name
# ---------------------------------------------------------------------------


class TestExtractBaseToolName:
    """Test MCP-prefixed tool name extraction."""

    def test_mcp_prefixed_name(self):
        assert _extract_base_tool_name("mcp__weather__get_weather") == "get_weather"

    def test_mcp_prefixed_with_multiple_parts(self):
        assert _extract_base_tool_name("mcp__my_server__my_tool") == "my_tool"

    def test_plain_name_unchanged(self):
        assert _extract_base_tool_name("calculator") == "calculator"

    def test_non_mcp_prefix_unchanged(self):
        assert _extract_base_tool_name("other__prefix__name") == "other__prefix__name"

    def test_mcp_with_only_two_parts(self):
        # "mcp__something" has only 2 parts when split by __, not enough
        assert _extract_base_tool_name("mcp__something") == "mcp__something"
