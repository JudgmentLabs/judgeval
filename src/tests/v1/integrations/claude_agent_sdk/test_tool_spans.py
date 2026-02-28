"""Tests for Claude Agent SDK tool call span capture.

Verifies that the JudgeVal integration correctly captures tool call spans
from the message stream for all tools (built-in and SDK-defined MCP tools).
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
    LLMSpanTracker,
    ToolSpanTracker,
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
# Tests for ToolSpanTracker (message-stream-based tool spans for ALL tools)
# ---------------------------------------------------------------------------


class TestToolSpanTracker:
    """Test that tool calls from the message stream are captured as spans."""

    @pytest.mark.asyncio
    async def test_builtin_tool_creates_span(self):
        """A ToolUseBlock + ToolResultBlock = tool span for built-in tools."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

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
    async def test_mcp_tool_creates_span(self):
        """MCP tools (mcp__server__tool_name) also get spans from the message stream."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="tool_mcp",
                        name="mcp__weather__get_weather",
                        input={"city": "San Francisco"},
                    ),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="tool_mcp",
                        content="62F, Foggy",
                    ),
                ]
            )
            tracker.on_user_message(user_msg)

        weather_spans = [
            s for s in collector.spans if s.name == "mcp__weather__get_weather"
        ]
        assert len(weather_spans) == 1

        attrs = dict(weather_spans[0].attributes)
        assert attrs[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"
        assert "San Francisco" in str(attrs[AttributeKeys.JUDGMENT_INPUT])
        assert "62F" in str(attrs[AttributeKeys.JUDGMENT_OUTPUT])

    @pytest.mark.asyncio
    async def test_tool_error_sets_error_status(self):
        """Tool results with is_error=True should set error status on the span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

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
    async def test_multiple_tools_in_same_message(self):
        """Multiple ToolUseBlocks in a single AssistantMessage each get a span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

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
    async def test_mixed_builtin_and_mcp_tools(self):
        """Both built-in and MCP tools in the same message get separate spans."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

            assistant_msg = AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="t_bash", name="bash", input={"command": "echo hi"}
                    ),
                    ToolUseBlock(
                        id="t_weather",
                        name="mcp__weather__get_weather",
                        input={"city": "NYC"},
                    ),
                ]
            )
            tracker.on_assistant_message(assistant_msg)

            user_msg = FakeUserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t_bash", content="hi"),
                    ToolResultBlock(tool_use_id="t_weather", content="45F, Clear"),
                ]
            )
            tracker.on_user_message(user_msg)

        bash_spans = [s for s in collector.spans if s.name == "bash"]
        weather_spans = [
            s for s in collector.spans if s.name == "mcp__weather__get_weather"
        ]

        assert len(bash_spans) == 1
        assert len(weather_spans) == 1

    @pytest.mark.asyncio
    async def test_cleanup_ends_unclosed_spans(self):
        """Cleanup should end spans that never got a matching ToolResultBlock."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

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
    async def test_tool_nests_under_agent(self):
        """Tool spans should be children of the agent span."""
        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

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
    async def test_tool_tracker_calls_emit_partial(self):
        """ToolSpanTracker should call emit_partial after opening tool spans."""
        from unittest.mock import patch
        from judgeval.v1.tracer.base_tracer import BaseTracer

        collector = InMemoryCollector()
        tracer = _make_tracer(collector)

        with tracer.get_tracer().start_as_current_span("agent") as agent_span:
            state = _make_state_with_parent(tracer, agent_span)
            tracker = ToolSpanTracker(tracer, state=state)

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
            "emit_partial should be called when opening tool spans"
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
# Tests for the LLM span content serialization
# ---------------------------------------------------------------------------


class TestMessageStreamLLMSpans:
    """Test that LLM spans correctly serialize tool use blocks."""

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


# ---------------------------------------------------------------------------
# Tests for TracingState
# ---------------------------------------------------------------------------


class TestTracingState:
    """Test TracingState shared context behavior."""

    @pytest.mark.asyncio
    async def test_separate_states_are_independent(self):
        """Two TracingState instances don't interfere with each other."""
        state_a = TracingState()
        state_b = TracingState()

        state_a.parent_context = "context_a"
        assert state_b.parent_context is None

        state_b.parent_context = "context_b"
        assert state_a.parent_context == "context_a"
        assert state_b.parent_context == "context_b"
