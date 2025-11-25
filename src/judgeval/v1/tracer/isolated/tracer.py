from __future__ import annotations

from typing import Iterator, Optional, Sequence

from opentelemetry.context.context import Context
from opentelemetry.trace import Link, SpanKind, Tracer, Span
from opentelemetry.util.types import Attributes
from opentelemetry.util._decorator import _agnosticcontextmanager

from judgeval.v1.tracer.isolated.context import attach, detach, get_current
from judgeval.v1.tracer.isolated.propagation import set_span_in_context

_Links = Optional[Sequence[Link]]


class JudgmentIsolatedTracer(Tracer):
    __slots__ = ("_delegate",)

    def __init__(self, delegate: Tracer):
        self._delegate = delegate

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
            context = get_current()
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
            context = get_current()
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
        ctx = set_span_in_context(span, context)
        token = attach(ctx)
        try:
            if end_on_exit:
                try:
                    yield span
                finally:
                    span.end()
            else:
                yield span
        finally:
            detach(token)
