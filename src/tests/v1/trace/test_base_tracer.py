from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.trace.base_tracer import BaseTracer


class TestSpanCreation:
    def test_span_context_manager_creates_span(self, tracer, collecting_exporter):
        with BaseTracer.span("my-span"):
            pass
        assert any(s.name == "my-span" for s in collecting_exporter.spans)

    def test_span_records_exception_and_reraises(self, tracer, collecting_exporter):
        with pytest.raises(RuntimeError, match="boom"):
            with BaseTracer.span("err-span"):
                raise RuntimeError("boom")
        span = next(s for s in collecting_exporter.spans if s.name == "err-span")
        assert span.status.status_code.name == "ERROR"
        assert any(e.name == "exception" for e in span.events)

    def test_nested_spans_share_trace_id(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("parent"):
            with BaseTracer.start_as_current_span("child"):
                pass
        parent = next(s for s in collecting_exporter.spans if s.name == "parent")
        child = next(s for s in collecting_exporter.spans if s.name == "child")
        assert parent.context.trace_id == child.context.trace_id
        assert child.parent.span_id == parent.context.span_id

    def test_start_span_returns_usable_span(self, tracer, collecting_exporter):
        span = BaseTracer.start_span("manual")
        span.end()
        assert any(s.name == "manual" for s in collecting_exporter.spans)

    def test_get_current_span_inside_context(self, tracer):
        with BaseTracer.start_as_current_span("active"):
            span = BaseTracer.get_current_span()
            assert span.is_recording()


class TestAttributes:
    def test_set_attribute(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("attr"):
            BaseTracer.set_attribute("my.key", "value")
        span = next(s for s in collecting_exporter.spans if s.name == "attr")
        assert span.attributes["my.key"] == "value"

    def test_set_attributes_multiple(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("multi"):
            BaseTracer.set_attributes({"k1": "v1", "k2": "v2"})
        span = next(s for s in collecting_exporter.spans if s.name == "multi")
        assert span.attributes["k1"] == "v1"
        assert span.attributes["k2"] == "v2"

    def test_set_input_and_output(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("io"):
            BaseTracer.set_input({"q": "what?"})
            BaseTracer.set_output("answer")
        span = next(s for s in collecting_exporter.spans if s.name == "io")
        assert "q" in span.attributes[AttributeKeys.JUDGMENT_INPUT]
        assert "answer" in span.attributes[AttributeKeys.JUDGMENT_OUTPUT]

    def test_set_span_kind_llm(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("llm"):
            BaseTracer.set_llm_span()
        span = next(s for s in collecting_exporter.spans if s.name == "llm")
        assert span.attributes[AttributeKeys.JUDGMENT_SPAN_KIND] == "llm"

    def test_set_span_kind_tool(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("tool"):
            BaseTracer.set_tool_span()
        span = next(s for s in collecting_exporter.spans if s.name == "tool")
        assert span.attributes[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"

    def test_set_span_kind_general(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("general"):
            BaseTracer.set_general_span()
        span = next(s for s in collecting_exporter.spans if s.name == "general")
        assert span.attributes[AttributeKeys.JUDGMENT_SPAN_KIND] == "span"

    def test_record_llm_metadata(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("llm-meta"):
            BaseTracer.recordLLMMetadata(
                {
                    "model": "gpt-4",
                    "provider": "openai",
                    "non_cached_input_tokens": 10,
                    "output_tokens": 20,
                    "total_cost_usd": 0.001,
                }
            )
        span = next(s for s in collecting_exporter.spans if s.name == "llm-meta")
        assert span.attributes[AttributeKeys.JUDGMENT_LLM_MODEL_NAME] == "gpt-4"
        assert span.attributes[AttributeKeys.JUDGMENT_LLM_PROVIDER] == "openai"
        assert (
            span.attributes[AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS] == 10
        )
        assert span.attributes[AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS] == 20


class TestContextPropagation:
    def test_set_customer_id_on_span(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("cust"):
            BaseTracer.set_customer_id("cust-001")
        span = next(s for s in collecting_exporter.spans if s.name == "cust")
        assert span.attributes[AttributeKeys.JUDGMENT_CUSTOMER_ID] == "cust-001"

    def test_set_session_id_on_span(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("sess"):
            BaseTracer.set_session_id("sess-abc")
        span = next(s for s in collecting_exporter.spans if s.name == "sess")
        assert span.attributes[AttributeKeys.JUDGMENT_SESSION_ID] == "sess-abc"

    def test_set_customer_user_id_on_span(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("cuid"):
            BaseTracer.set_customer_user_id("user-xyz")
        span = next(s for s in collecting_exporter.spans if s.name == "cuid")
        assert span.attributes[AttributeKeys.JUDGMENT_CUSTOMER_USER_ID] == "user-xyz"

    def test_customer_id_propagates_to_child(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("root"):
            BaseTracer.set_customer_id("cid")
            with BaseTracer.start_as_current_span("child"):
                pass
        child = next(s for s in collecting_exporter.spans if s.name == "child")
        assert child.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cid"

    def test_set_propagating_baggage_key_noop_outside_span(self, tracer):
        BaseTracer._set_propagating_baggage_key("some.key", "val")


class TestGuardBranches:
    def test_set_attribute_noop_outside_span(self, tracer):
        BaseTracer.set_attribute("key", "value")

    def test_set_attribute_noop_empty_key(self, tracer):
        with BaseTracer.start_as_current_span("s"):
            BaseTracer.set_attribute("", "value")

    def test_set_attribute_noop_none_value(self, tracer):
        with BaseTracer.start_as_current_span("s"):
            BaseTracer.set_attribute("key", None)

    def test_set_attributes_noop_when_none(self, tracer):
        BaseTracer.set_attributes(None)

    def test_set_span_kind_noop_when_none(self, tracer):
        with BaseTracer.start_as_current_span("s"):
            BaseTracer.set_span_kind(None)

    def test_record_llm_metadata_noop_outside_span(self, tracer):
        BaseTracer.recordLLMMetadata({"model": "gpt-4"})

    def test_record_llm_metadata_cache_fields(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("llm-cache"):
            BaseTracer.recordLLMMetadata(
                {
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 3,
                }
            )
        span = next(s for s in collecting_exporter.spans if s.name == "llm-cache")
        assert (
            span.attributes[AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS] == 5
        )
        assert (
            span.attributes[AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS]
            == 3
        )


class TestLifecycle:
    def test_force_flush(self, tracer):
        result = BaseTracer.force_flush()
        assert isinstance(result, bool)

    def test_shutdown(self, tracer):
        BaseTracer.shutdown()

    def test_register_otel_instrumentation(self, tracer):
        from unittest.mock import MagicMock

        instrumentor = MagicMock()
        instrumentor.instrument = MagicMock()
        BaseTracer.registerOTELInstrumentation(instrumentor)
        instrumentor.instrument.assert_called_once()


class TestGetCurrentTraceAndSpanId:
    def test_returns_none_outside_span(self, tracer):
        result = BaseTracer._get_current_trace_and_span_id()
        assert result is None

    def test_returns_ids_inside_span(self, tracer):
        with BaseTracer.start_as_current_span("s"):
            result = BaseTracer._get_current_trace_and_span_id()
        assert result is not None
        trace_id, span_id = result
        assert len(trace_id) == 32
        assert len(span_id) == 16

    def test_returns_none_when_context_not_valid(self, tracer):
        from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan

        invalid_ctx = SpanContext(
            trace_id=0x1234567890ABCDEF1234567890ABCDEF,
            span_id=0x1234567890ABCDEF,
            is_remote=False,
            trace_flags=TraceFlags(0),
        )
        mock_span = NonRecordingSpan(invalid_ctx)
        with patch.object(
            BaseTracer._get_proxy_provider(),
            "get_current_span",
            return_value=mock_span,
        ):
            pass

        with patch(
            "judgeval.v1.trace.base_tracer.BaseTracer._get_proxy_provider"
        ) as mock_proxy:
            mock_instance = MagicMock()
            mock_proxy.return_value = mock_instance
            mock_recording_span = MagicMock()
            mock_recording_span.is_recording.return_value = True
            mock_recording_span.get_span_context.return_value = invalid_ctx
            mock_instance.get_current_span.return_value = mock_recording_span
            result = BaseTracer._get_current_trace_and_span_id()
        assert result is None


class TestEmitPartial:
    def test_noop_without_active_tracer(self, _reset_provider):
        BaseTracer._emit_partial()


class TestTag:
    def test_tag_enqueues_with_client(self, tracer, collecting_exporter):
        from unittest.mock import MagicMock, patch

        tracer._client = MagicMock()
        tracer._client.post_projects_traces_by_trace_id_tags = MagicMock()
        tracer.project_id = "proj-123"

        with patch("judgeval.v1.trace.base_tracer.bg_enqueue") as mock_enqueue:
            with BaseTracer.start_as_current_span("tag-span"):
                BaseTracer.tag("release-v1")
            mock_enqueue.assert_called_once()
            fn = mock_enqueue.call_args[0][0]
            fn()

        tracer._client.post_projects_traces_by_trace_id_tags.assert_called_once()
        _, kwargs = tracer._client.post_projects_traces_by_trace_id_tags.call_args
        assert "release-v1" in kwargs["payload"]["tags"]

    def test_tag_list_of_strings(self, tracer):
        from unittest.mock import MagicMock, patch

        tracer._client = MagicMock()
        tracer.project_id = "proj-123"

        with patch("judgeval.v1.trace.base_tracer.bg_enqueue") as mock_enqueue:
            with BaseTracer.start_as_current_span("s"):
                BaseTracer.tag(["a", "b"])
            mock_enqueue.assert_called_once()

    def test_tag_noop_without_tracer(self):
        BaseTracer.tag("my-tag")

    def test_tag_noop_empty_list(self, tracer):
        with patch("judgeval.v1.trace.base_tracer.bg_enqueue") as mock_enqueue:
            with BaseTracer.start_as_current_span("s"):
                BaseTracer.tag([])
            mock_enqueue.assert_not_called()

    def test_tag_noop_when_no_client(self, tracer):
        tracer.project_id = "proj-123"
        tracer._client = None

        with patch("judgeval.v1.trace.base_tracer.bg_enqueue") as mock_enqueue:
            with BaseTracer.start_as_current_span("s"):
                BaseTracer.tag("my-tag")
            mock_enqueue.assert_not_called()

    def test_tag_noop_when_ids_none(self, tracer):
        tracer.project_id = "proj-123"
        tracer._client = MagicMock()

        with patch(
            "judgeval.v1.trace.base_tracer.BaseTracer._get_current_trace_and_span_id",
            return_value=None,
        ):
            with patch("judgeval.v1.trace.base_tracer.bg_enqueue") as mock_enqueue:
                with BaseTracer.start_as_current_span("s"):
                    BaseTracer.tag("my-tag")
                mock_enqueue.assert_not_called()


class TestAsyncEvaluate:
    def test_async_evaluate_noop_without_tracer(self):
        BaseTracer.async_evaluate("my-judge")

    def test_async_evaluate_noop_without_active_span(self, tracer):
        tracer.project_id = "proj-123"
        BaseTracer.async_evaluate("my-judge")

    def test_async_evaluate_sets_pending_attribute(self, tracer, collecting_exporter):
        import json

        tracer.project_id = "proj-123"

        with BaseTracer.start_as_current_span("eval-span"):
            BaseTracer.async_evaluate("my-judge", example={"input": "q", "output": "a"})

        span = next(s for s in collecting_exporter.spans if s.name == "eval-span")
        raw = span.attributes.get(AttributeKeys.JUDGMENT_PENDING_TRACE_EVAL)
        assert raw is not None
        evals = json.loads(raw)
        assert len(evals) == 1
        assert evals[0]["judges"][0]["name"] == "my-judge"
