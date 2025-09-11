from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Any
from collections import defaultdict
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, SpanContext
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from judgeval.tracer.exporters import JudgmentSpanExporter
from judgeval.tracer.keys import AttributeKeys, InternalAttributeKeys, ResourceKeys
from judgeval.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger
from judgeval.utils.url import url_for
from judgeval.version import get_version

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
        project_name: str,
        api_key: str,
        organization_id: str,
        /,
        *,
        max_queue_size: int = 2**18,
        export_timeout_millis: int = 30000,
        resource_attributes: Optional[dict[str, Any]] = None,
    ):
        self.tracer = tracer
        self.project_name = project_name
        self.api_key = api_key
        self.organization_id = organization_id

        # Resolve project_id
        self.project_id = self._resolve_project_id()

        # Set up resource attributes with project_id
        self._setup_resource_attributes(resource_attributes or {})

        endpoint = url_for("/otel/v1/traces")
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

    def _resolve_project_id(self) -> str | None:
        """Resolve project_id from project_name using the API."""
        try:
            client = JudgmentSyncClient(
                api_key=self.api_key,
                organization_id=self.organization_id,
            )
            return client.projects_resolve({"project_name": self.project_name})[
                "project_id"
            ]
        except Exception:
            return None

    def _setup_resource_attributes(self, resource_attributes: dict[str, Any]) -> None:
        """Set up resource attributes including project_id."""
        resource_attributes.update(
            {
                ResourceKeys.SERVICE_NAME: self.project_name,
                ResourceKeys.TELEMETRY_SDK_NAME: "judgeval",
                ResourceKeys.TELEMETRY_SDK_VERSION: get_version(),
            }
        )

        if self.project_id is not None:
            resource_attributes[ResourceKeys.JUDGMENT_PROJECT_ID] = self.project_id
        else:
            judgeval_logger.error(
                f"Failed to resolve project {self.project_name}, please create it first at https://app.judgmentlabs.ai/org/{self.organization_id}/projects. Skipping Judgment export."
            )

        self.resource_attributes = resource_attributes

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

        existing_resource_attrs = (
            dict(current_span.resource.attributes) if current_span.resource else {}
        )
        merged_resource_attrs = {**existing_resource_attrs, **self.resource_attributes}
        merged_resource = Resource.create(merged_resource_attrs)

        partial_span = ReadableSpan(
            name=current_span.name,
            context=span_context,
            parent=current_span.parent,
            resource=merged_resource,
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

            existing_resource_attrs = (
                dict(span.resource.attributes) if span.resource else {}
            )
            merged_resource_attrs = {
                **existing_resource_attrs,
                **self.resource_attributes,
            }
            merged_resource = Resource.create(merged_resource_attrs)

            final_span = ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=merged_resource,
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
        pass

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

    def set_internal_attribute(
        self, span_context: SpanContext, key: str, value: Any
    ) -> None:
        pass

    def get_internal_attribute(
        self, span_context: SpanContext, key: str, default: Any = None
    ) -> Any:
        return default

    def increment_update_id(self, span_context: SpanContext) -> int:
        return 0


__all__ = ["NoOpSpanProcessor", "JudgmentSpanProcessor", "NoOpJudgmentSpanProcessor"]
