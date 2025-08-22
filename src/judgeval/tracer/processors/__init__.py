from typing import Optional
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.trace import get_current_span
from judgeval.tracer.exporters import JudgmentSpanExporter
from judgeval.tracer.keys import AttributeKeys


class NoOpSpanProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class JudgmentSpanProcessor(BatchSpanProcessor):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        organization_id: str,
        /,
        *,
        max_queue_size: int = 2**18,
        export_timeout_millis: int = 30000,
    ):
        super().__init__(
            JudgmentSpanExporter(
                endpoint=endpoint,
                api_key=api_key,
                organization_id=organization_id,
            ),
            max_queue_size=max_queue_size,
            export_timeout_millis=export_timeout_millis,
        )
        self._span_update_ids: dict[tuple[int, int], int] = {}

    def emit_partial(self) -> None:
        current_span = get_current_span()
        if not current_span or not current_span.is_recording():
            return

        if not isinstance(current_span, ReadableSpan):
            return

        span_context = current_span.get_span_context()
        span_key = (span_context.trace_id, span_context.span_id)

        current_update_id = self._span_update_ids.get(span_key, 0)
        self._span_update_ids[span_key] = current_update_id + 1

        attributes = dict(current_span.attributes or {})
        attributes[AttributeKeys.JUDGMENT_UPDATE_ID] = current_update_id

        partial_span = ReadableSpan(
            name=current_span.name,
            context=span_context,
            parent=current_span.parent,
            resource=current_span.resource,
            attributes=attributes,
            events=current_span.events,
            links=current_span.links,
            status=current_span.status,
            kind=current_span.kind,
            start_time=current_span.start_time,
            end_time=None,
            instrumentation_scope=current_span.instrumentation_scope,
        )

        super().on_end(partial_span)

    def on_end(self, span: ReadableSpan) -> None:
        if span.end_time is not None and span.context:
            span_key = (span.context.trace_id, span.context.span_id)
            self._span_update_ids.pop(span_key, None)
        super().on_end(span)


class NoOpJudgmentSpanProcessor(JudgmentSpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int | None = 30000) -> bool:
        return True

    def emit_partial(self) -> None:
        pass


__all__ = ("NoOpSpanProcessor", "JudgmentSpanProcessor", "NoOpJudgmentSpanProcessor")
