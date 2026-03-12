from __future__ import annotations

from re import compile
from types import MappingProxyType
from typing import Dict, Mapping, Optional

from opentelemetry.context import Context, create_key, get_value, set_value
from opentelemetry.util.re import (
    _BAGGAGE_PROPERTY_FORMAT,
    _KEY_FORMAT,
    _VALUE_FORMAT,
)

from judgeval.v1.trace.baggage.propagator import JudgmentBaggagePropagator

_BAGGAGE_KEY = create_key("baggage")

_KEY_PATTERN = compile(_KEY_FORMAT)
_VALUE_PATTERN = compile(_VALUE_FORMAT)
_PROPERTY_PATTERN = compile(_BAGGAGE_PROPERTY_FORMAT)


def _resolve_context(context: Optional[Context]) -> Context:
    if context is not None:
        return context
    from judgeval.v1.trace.judgment_tracer_provider import JudgmentTracerProvider

    return JudgmentTracerProvider.get_instance().get_current_context()


def _get_baggage_value(context: Optional[Context] = None) -> Dict[str, object]:
    baggage = get_value(_BAGGAGE_KEY, context=context)
    if isinstance(baggage, dict):
        return baggage
    return {}


def _is_valid_key(name: str) -> bool:
    return _KEY_PATTERN.fullmatch(str(name)) is not None


def _is_valid_value(value: object) -> bool:
    parts = str(value).split(";")
    is_valid = _VALUE_PATTERN.fullmatch(parts[0]) is not None
    if len(parts) > 1:
        for prop in parts[1:]:
            if _PROPERTY_PATTERN.fullmatch(prop) is None:
                return False
    return is_valid


def _is_valid_pair(key: str, value: object) -> bool:
    return _is_valid_key(key) and _is_valid_value(value)


def get_all(context: Optional[Context] = None) -> Mapping[str, object]:
    return MappingProxyType(_get_baggage_value(context=_resolve_context(context)))


def get_baggage(name: str, context: Optional[Context] = None) -> Optional[object]:
    return _get_baggage_value(context=_resolve_context(context)).get(name)


def set_baggage(name: str, value: object, context: Optional[Context] = None) -> Context:
    ctx = _resolve_context(context)
    baggage = _get_baggage_value(context=ctx).copy()
    baggage[name] = value
    return set_value(_BAGGAGE_KEY, baggage, context=ctx)


def remove_baggage(name: str, context: Optional[Context] = None) -> Context:
    ctx = _resolve_context(context)
    baggage = _get_baggage_value(context=ctx).copy()
    baggage.pop(name, None)
    return set_value(_BAGGAGE_KEY, baggage, context=ctx)


def clear(context: Optional[Context] = None) -> Context:
    return set_value(_BAGGAGE_KEY, {}, context=_resolve_context(context))


__all__ = [
    "JudgmentBaggagePropagator",
    "clear",
    "get_all",
    "get_baggage",
    "remove_baggage",
    "set_baggage",
]
