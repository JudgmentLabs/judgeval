from __future__ import annotations

from typing import Iterator, Optional, Sequence

from opentelemetry import trace as trace_api
from opentelemetry.context.context import Context
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.trace import Link, SpanKind, Tracer, Span, Status, StatusCode
from opentelemetry.util.types import Attributes
from opentelemetry.util._decorator import _agnosticcontextmanager

_Links = Optional[Sequence[Link]]


class JudgmentIsolatedTracer(Tracer):
    __slots__ = ("_delegate", "_runtime_context")

    def __init__(self, delegate: Tracer, runtime_context: ContextVarsRuntimeContext):
        self._delegate = delegate
        self._runtime_context = runtime_context

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: _Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        if context is None:
            context = self._runtime_context.get_current()
        return self._delegate.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: _Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        if context is None:
            context = self._runtime_context.get_current()
        span = self._delegate.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )

        with self.use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as s:
            yield s

    def get_current_span(self, context: Optional[Context] = None) -> Span:
        """Get the current span from this tracer's isolated context."""
        if context is None:
            context = self._runtime_context.get_current()
        return trace_api.get_current_span(context)

    @_agnosticcontextmanager
    def use_span(
        self,
        span: Span,
        end_on_exit: bool = False,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Iterator[Span]:
        """Use a span as the current span in this tracer's isolated context."""
        try:
            ctx = trace_api.set_span_in_context(
                span, self._runtime_context.get_current()
            )
            token = self._runtime_context.attach(ctx)
            try:
                yield span
            finally:
                self._runtime_context.detach(token)

        except Exception as exc:
            if isinstance(span, Span) and span.is_recording():
                if record_exception:
                    span.record_exception(exc)

                if set_status_on_exception:
                    span.set_status(
                        Status(
                            status_code=StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )
            raise

        finally:
            if end_on_exit:
                span.end()
