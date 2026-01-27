from __future__ import annotations
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar, overload

from opentelemetry.trace import Span

from .llm import *
from .llm.providers import ApiClient

T = TypeVar("T", bound=ApiClient)
C = TypeVar("C", bound=Callable[..., Any])


def _get_tracers():
    from judgeval.v1.tracer.base_tracer import BaseTracer

    return BaseTracer._tracers


def wrap(client: T) -> T:
    for tracer in _get_tracers():
        client = tracer.wrap(client)
    return client


@overload
def observe(
    func: C,
    span_type: Optional[str] = "span",
    span_name: Optional[str] = None,
    record_input: bool = True,
    record_output: bool = True,
    disable_generator_yield_span: bool = False,
) -> C: ...


@overload
def observe(
    func: None = None,
    span_type: Optional[str] = "span",
    span_name: Optional[str] = None,
    record_input: bool = True,
    record_output: bool = True,
    disable_generator_yield_span: bool = False,
) -> Callable[[C], C]: ...


def observe(
    func: Optional[C] = None,
    span_type: Optional[str] = "span",
    span_name: Optional[str] = None,
    record_input: bool = True,
    record_output: bool = True,
    disable_generator_yield_span: bool = False,
) -> C | Callable[[C], C]:
    def decorator(f: C) -> C:
        result = f
        for tracer in _get_tracers():
            result = tracer.observe(
                result,
                span_type=span_type,
                span_name=span_name,
                record_input=record_input,
                record_output=record_output,
                disable_generator_yield_span=disable_generator_yield_span,
            )
        return result

    if func is None:
        return decorator
    return decorator(func)


def set_attribute(key: str, value: Any) -> None:
    for tracer in _get_tracers():
        tracer.set_attribute(key, value)


def set_attributes(attributes: Dict[str, Any]) -> None:
    for tracer in _get_tracers():
        tracer.set_attributes(attributes)


def set_input(input_data: Any) -> None:
    for tracer in _get_tracers():
        tracer.set_input(input_data)


def set_output(output_data: Any) -> None:
    for tracer in _get_tracers():
        tracer.set_output(output_data)


def set_customer_id(customer_id: str) -> None:
    for tracer in _get_tracers():
        tracer.set_customer_id(customer_id)


def set_session_id(session_id: str) -> None:
    for tracer in _get_tracers():
        tracer.set_session_id(session_id)


def tag(tags: str | list[str]) -> None:
    for tracer in _get_tracers():
        tracer.tag(tags)


@contextmanager
def span(span_name: str) -> Iterator[list[Span]]:
    tracers = _get_tracers()
    with ExitStack() as stack:
        spans = [stack.enter_context(tracer.span(span_name)) for tracer in tracers]
        yield spans


__all__ = [
    "wrap_provider",
    "wrap",
    "observe",
    "set_attribute",
    "set_attributes",
    "set_input",
    "set_output",
    "set_customer_id",
    "set_session_id",
    "tag",
    "span",
]
