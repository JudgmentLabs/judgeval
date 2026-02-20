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

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
    _wrap_tool_handler,
    _create_tool_wrapper_class,
    _wrap_tool_factory,
    _thread_local,
    _sdk_tool_names,
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

        wrapped = _wrap_tool_handler(tracer, my_tool, "calculator")

        # Create a parent span context to simulate the agent span
        with tracer.get_tracer().start_as_current_span("test_parent") as parent_span:
            from opentelemetry.trace import set_span_in_context

            parent_ctx = set_span_in_context(parent_span, tracer.get_context())
            _thread_local.parent_context = parent_ctx

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

        wrapped = _wrap_tool_handler(tracer, failing_tool, "broken_tool")

        with tracer.get_tracer().start_as_current_span("test_parent") as parent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                parent_span, tracer.get_context()
            )

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

        wrapped_once = _wrap_tool_handler(tracer, my_tool, "tool1")
        wrapped_twice = _wrap_tool_handler(tracer, wrapped_once, "tool1")

        assert wrapped_once is wrapped_twice

    @pytest.mark.asyncio
    async def test_tool_handler_nests_under_parent(self):
        """Tool spans should be children of the agent span via thread-local context."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        async def my_tool(args):
            return {"result": "ok"}

        wrapped = _wrap_tool_handler(tracer, my_tool, "nested_tool")

        with tracer.get_tracer().start_as_current_span("agent_span") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )
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

        WrappedTool = _create_tool_wrapper_class(FakeSdkMcpTool, tracer)

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

        # Call the wrapped handler
        with tracer.get_tracer().start_as_current_span("parent") as parent:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                parent, tracer.get_context()
            )
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

        wrapped_tool_fn = _wrap_tool_factory(fake_tool_fn, tracer)

        @wrapped_tool_fn("add", "Add numbers", {"a": float, "b": float})
        async def add(args):
            return {
                "content": [{"type": "text", "text": f"Sum: {args['a'] + args['b']}"}]
            }

        assert add.name == "add"
        assert hasattr(add.handler, "_judgeval_wrapped")

        with tracer.get_tracer().start_as_current_span("parent") as parent:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                parent, tracer.get_context()
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

        wrapped_a = _wrap_tool_handler(tracer, tool_a, "tool_a")
        wrapped_b = _wrap_tool_handler(tracer, tool_b, "tool_b")

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )
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

        wrapped = _wrap_tool_handler(tracer, echo_tool, "echo")

        with tracer.get_tracer().start_as_current_span("parent") as parent:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                parent, tracer.get_context()
            )
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

        tracker = BuiltInToolSpanTracker(tracer)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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

        tracker = BuiltInToolSpanTracker(tracer)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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

        # "calculator" is an SDK-defined tool
        tracker = BuiltInToolSpanTracker(tracer, sdk_tool_names={"calculator"})

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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
    async def test_multiple_builtin_tools_in_same_message(self):
        """Multiple ToolUseBlocks in a single AssistantMessage."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        tracker = BuiltInToolSpanTracker(tracer)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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

        tracker = BuiltInToolSpanTracker(tracer)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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

        tracker = BuiltInToolSpanTracker(tracer)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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

        wrapped = _wrap_tool_handler(tracer, my_tool, "traced_tool")

        with tracer.get_tracer().start_as_current_span("parent") as parent:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                parent, tracer.get_context()
            )
            with patch.object(
                BaseTracer, "emit_partial", wraps=tracer.emit_partial
            ) as mock_emit:
                await wrapped({"x": 1})

        assert mock_emit.call_count >= 1, "emit_partial should be called for tool spans"

    @pytest.mark.asyncio
    async def test_builtin_tool_tracker_calls_emit_partial(self):
        """BuiltInToolSpanTracker should call emit_partial after opening a tool span."""
        from unittest.mock import patch
        from judgeval.v1.tracer.base_tracer import BaseTracer

        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        tracker = BuiltInToolSpanTracker(tracer)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            from opentelemetry.trace import set_span_in_context

            _thread_local.parent_context = set_span_in_context(
                agent_span, tracer.get_context()
            )

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

        assert mock_emit.call_count >= 1, (
            "emit_partial should be called when opening built-in tool span"
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
# Cleanup thread-local storage and SDK tool names after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_thread_local():
    yield
    if hasattr(_thread_local, "parent_context"):
        delattr(_thread_local, "parent_context")
    _sdk_tool_names.clear()
