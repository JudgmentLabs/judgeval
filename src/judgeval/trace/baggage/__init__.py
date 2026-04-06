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

from judgeval.trace.baggage.propagator import JudgmentBaggagePropagator

_BAGGAGE_KEY = create_key("baggage")

_KEY_PATTERN = compile(_KEY_FORMAT)
_VALUE_PATTERN = compile(_VALUE_FORMAT)
_PROPERTY_PATTERN = compile(_BAGGAGE_PROPERTY_FORMAT)


def _resolve_context(context: Optional[Context]) -> Context:
    if context is not None:
        return context
    from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

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
    """Return all baggage entries as a read-only mapping.

    Args:
        context: OTel context to read from. Defaults to the current
            Judgment context.

    Returns:
        An immutable mapping of baggage key-value pairs.
    """
    return MappingProxyType(_get_baggage_value(context=_resolve_context(context)))


def get_baggage(name: str, context: Optional[Context] = None) -> Optional[object]:
    """Retrieve a single baggage value by key.

    Args:
        name: The baggage key to look up.
        context: OTel context to read from. Defaults to the current
            Judgment context.

    Returns:
        The baggage value, or None if the key is not set.
    """
    return _get_baggage_value(context=_resolve_context(context)).get(name)


def set_baggage(name: str, value: object, context: Optional[Context] = None) -> Context:
    """Set a baggage key-value pair, returning a new context with the entry.

    Args:
        name: The baggage key.
        value: The baggage value (will be stringified for propagation).
        context: Base context. Defaults to the current Judgment context.

    Returns:
        A new ``Context`` containing the updated baggage.
    """
    ctx = _resolve_context(context)
    baggage = _get_baggage_value(context=ctx).copy()
    baggage[name] = value
    return set_value(_BAGGAGE_KEY, baggage, context=ctx)


def remove_baggage(name: str, context: Optional[Context] = None) -> Context:
    """Remove a baggage entry by key, returning a new context without it.

    Args:
        name: The baggage key to remove.
        context: Base context. Defaults to the current Judgment context.

    Returns:
        A new ``Context`` with the entry removed.
    """
    ctx = _resolve_context(context)
    baggage = _get_baggage_value(context=ctx).copy()
    baggage.pop(name, None)
    return set_value(_BAGGAGE_KEY, baggage, context=ctx)


def clear(context: Optional[Context] = None) -> Context:
    """Remove all baggage entries, returning a clean context.

    Args:
        context: Base context. Defaults to the current Judgment context.

    Returns:
        A new ``Context`` with an empty baggage map.
    """
    return set_value(_BAGGAGE_KEY, {}, context=_resolve_context(context))


__all__ = [
    "JudgmentBaggagePropagator",
    "clear",
    "get_all",
    "get_baggage",
    "remove_baggage",
    "set_baggage",
]
