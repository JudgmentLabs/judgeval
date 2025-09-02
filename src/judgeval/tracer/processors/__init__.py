from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Any
from collections import defaultdict
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, SpanContext
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from judgeval.tracer.exporters import JudgmentSpanExporter
from judgeval.tracer.keys import AttributeKeys, InternalAttributeKeys

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


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
        tracer: Tracer,
        endpoint: str,
        api_key: str,
        organization_id: str,
        /,
        *,
        max_queue_size: int = 2**18,
        export_timeout_millis: int = 30000,
    ):
        self.tracer = tracer
        super().__init__(
            JudgmentSpanExporter(
                endpoint=endpoint,
                api_key=api_key,
                organization_id=organization_id,
            ),
            max_queue_size=max_queue_size,
            export_timeout_millis=export_timeout_millis,
        )
        self._internal_attributes: defaultdict[tuple[int, int], dict[str, Any]] = (
            defaultdict(dict)
        )

    def _get_span_key(self, span_context: SpanContext) -> tuple[int, int]:
        return (span_context.trace_id, span_context.span_id)

    def set_internal_attribute(
        self, span_context: SpanContext, key: str, value: Any
    ) -> None:
        span_key = self._get_span_key(span_context)
        self._internal_attributes[span_key][key] = value

    def get_internal_attribute(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        span_key = self._get_span_key(span_context)
        return self._internal_attributes[span_key].get(key, default)

    def increment_update_id(self, span_context: SpanContext) -> int:
        current_id = self.get_internal_attribute(
            span_context=span_context, key=AttributeKeys.JUDGMENT_UPDATE_ID, default=0
        )
        new_id = current_id + 1
        self.set_internal_attribute(
            span_context=span_context,
            key=AttributeKeys.JUDGMENT_UPDATE_ID,
            value=new_id,
        )
        return current_id

    def _cleanup_span_state(self, span_key: tuple[int, int]) -> None:
        self._internal_attributes.pop(span_key, None)

    def emit_partial(self) -> None:
        current_span = self.tracer.get_current_span()
        if not current_span or not current_span.is_recording():
            return

        if not isinstance(current_span, ReadableSpan):
            return

        span_context = current_span.get_span_context()
        if self.get_internal_attribute(
            span_context=span_context,
            key=InternalAttributeKeys.DISABLE_PARTIAL_EMIT,
            default=False,
        ):
            return

        current_update_id = self.increment_update_id(span_context=span_context)

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
        if not span.context:
            super().on_end(span)
            return

        span_key = self._get_span_key(span.context)

        if self.get_internal_attribute(
            span.context, InternalAttributeKeys.CANCELLED, False
        ):
            self._cleanup_span_state(span_key)
            return

        if span.end_time is not None:
            attributes = dict(span.attributes or {})
            attributes[AttributeKeys.JUDGMENT_UPDATE_ID] = 20

            final_span = ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes=attributes,
                events=span.events,
                links=span.links,
                status=span.status,
                kind=span.kind,
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )

            self._cleanup_span_state(span_key)
            super().on_end(final_span)
        else:
            super().on_end(span)


class NoOpJudgmentSpanProcessor(JudgmentSpanProcessor):
    def __init__(self):
        super().__init__(None, "", "", "")  # type: ignore[arg-type]

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
