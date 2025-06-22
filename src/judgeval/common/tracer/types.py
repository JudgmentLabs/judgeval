from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, Optional, TypeAlias, Union


ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]


@dataclass(slots=True, frozen=True, repr=True)
class CurrentSpanEntry:
    """
    Represents the current span entry in the context.
    This exists in a partial state before the span is fully created.
    """

    name: str
    span_id: str
    span_type: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    inputs: Optional[Dict[str, Any]] = None
    output: Any = None
    error: Optional[Dict[str, Any]] = None


@dataclass(slots=True, frozen=True, repr=True)
class CurrentSpanExit:
    """
    Represents the current span in the context.
    This is the fully created span that can be used for tracing.
    """

    name: str
    span_id: str
    span_type: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: float
    inputs: Optional[Dict[str, Any]] = None
    output: Any = None
    error: Optional[Dict[str, Any]] = None


CurrentSpanType = CurrentSpanEntry | CurrentSpanExit


TSpanResetTokens: TypeAlias = dict[str, Token[CurrentSpanEntry]]
