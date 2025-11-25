from __future__ import annotations

import typing
from contextvars import Token

from opentelemetry.context.context import Context
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext

_JUDGMENT_RUNTIME_CONTEXT = ContextVarsRuntimeContext()


def get_value(key: str, context: typing.Optional[Context] = None) -> object:
    return context.get(key) if context is not None else get_current().get(key)


def set_value(
    key: str, value: object, context: typing.Optional[Context] = None
) -> Context:
    if context is None:
        context = get_current()
    new_values = context.copy()
    new_values[key] = value
    return Context(new_values)


def get_current() -> Context:
    return _JUDGMENT_RUNTIME_CONTEXT.get_current()


def attach(context: Context) -> Token[Context]:
    return _JUDGMENT_RUNTIME_CONTEXT.attach(context)


def detach(token: Token[Context]) -> None:
    _JUDGMENT_RUNTIME_CONTEXT.detach(token)
