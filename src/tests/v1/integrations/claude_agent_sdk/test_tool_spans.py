from __future__ import annotations

from unittest.mock import MagicMock


from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
    ToolSpanTracker,
    LLMSpanTracker,
    TracingState,
)


def _make_tool_use_block(name="my_tool", tool_use_id="use-1", input_data=None):
    block = MagicMock()
    block.__class__.__name__ = "ToolUseBlock"
    block.name = name
    block.id = tool_use_id
    block.input = input_data or {"arg": "val"}
    return block


def _make_tool_result_block(tool_use_id="use-1", content="result", is_error=False):
    block = MagicMock()
    block.__class__.__name__ = "ToolResultBlock"
    block.tool_use_id = tool_use_id
    block.content = content
    block.is_error = is_error
    return block


class TestToolSpanTracker:
    def test_on_assistant_message_creates_pending_span(self, tracer):
        state = TracingState()
        tracker = ToolSpanTracker(state)

        tool_block = _make_tool_use_block()
        msg = MagicMock()
        msg.content = [tool_block]

        tracker.on_assistant_message(msg)
        assert "use-1" in tracker._pending_spans

    def test_on_user_message_ends_span(self, tracer, collecting_exporter):
        state = TracingState()
        tracker = ToolSpanTracker(state)

        tool_block = _make_tool_use_block()
        assistant_msg = MagicMock()
        assistant_msg.content = [tool_block]
        tracker.on_assistant_message(assistant_msg)

        result_block = _make_tool_result_block()
        user_msg = MagicMock()
        user_msg.content = [result_block]
        tracker.on_user_message(user_msg)

        assert "use-1" not in tracker._pending_spans

    def test_cleanup_ends_all_pending_spans(self, tracer):
        state = TracingState()
        tracker = ToolSpanTracker(state)

        for i in range(3):
            block = _make_tool_use_block(tool_use_id=f"use-{i}")
            msg = MagicMock()
            msg.content = [block]
            tracker.on_assistant_message(msg)

        tracker.cleanup()
        assert len(tracker._pending_spans) == 0

    def test_on_assistant_message_no_content_noop(self, tracer):
        state = TracingState()
        tracker = ToolSpanTracker(state)
        msg = MagicMock()
        msg.content = None
        tracker.on_assistant_message(msg)
        assert len(tracker._pending_spans) == 0

    def test_on_user_message_unknown_id_noop(self, tracer):
        state = TracingState()
        tracker = ToolSpanTracker(state)
        result_block = _make_tool_result_block(tool_use_id="unknown")
        msg = MagicMock()
        msg.content = [result_block]
        tracker.on_user_message(msg)


class TestLLMSpanTracker:
    def test_cleanup_noop_when_no_span(self, tracer):
        tracker = LLMSpanTracker()
        tracker.cleanup()
        assert tracker.current_span is None

    def test_mark_next_llm_start_sets_time(self, tracer):
        tracker = LLMSpanTracker()
        tracker.mark_next_llm_start()
        assert tracker.next_start_time is not None

    def test_log_usage_noop_when_no_span(self, tracer):
        tracker = LLMSpanTracker()
        tracker.log_usage({"key": "val"})
