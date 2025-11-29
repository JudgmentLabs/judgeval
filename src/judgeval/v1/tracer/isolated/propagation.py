from __future__ import annotations

from typing import Iterator, Optional

from opentelemetry import trace as trace_api
from opentelemetry.context.context import Context
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util._decorator import _agnosticcontextmanager

from judgeval.v1.tracer.isolated.context import attach, detach, get_current


def set_span_in_context(span: Span, context: Optional[Context] = None) -> Context:
    if context is None:
        context = get_current()
    return trace_api.set_span_in_context(span, context)


def get_current_span(context: Optional[Context] = None) -> Span:
    if context is None:
        context = get_current()
    return trace_api.get_current_span(context)


@_agnosticcontextmanager
def use_span(
    span: Span,
    end_on_exit: bool = False,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Iterator[Span]:
    try:
        ctx = set_span_in_context(span, get_current())
        token = attach(ctx)
        try:
            yield span
        finally:
            detach(token)

    # Record only exceptions that inherit Exception class but not BaseException, because
    # classes that directly inherit BaseException are not technically errors, e.g. GeneratorExit.
    # See https://github.com/open-telemetry/opentelemetry-python/issues/4484
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if isinstance(span, Span) and span.is_recording():
            # Record the exception as an event
            if record_exception:
                span.record_exception(exc)

            # Set status in case exception was raised
            if set_status_on_exception:
                span.set_status(
                    Status(
                        status_code=StatusCode.ERROR,
                        description=f"{type(exc).__name__}: {exc}",
                    )
                )

        # This causes parent spans to set their status to ERROR and to record
        # an exception as an event if a child span raises an exception even if
        # such child span was started with both record_exception and
        # set_status_on_exception attributes set to False.
        raise

    finally:
        if end_on_exit:
            span.end()
