from __future__ import annotations

import logging
from re import split
from typing import Iterable, List, Mapping, Optional, Set
from urllib.parse import quote_plus, unquote_plus

from opentelemetry.context import Context
from opentelemetry.propagators.textmap import (
    CarrierT,
    DefaultGetter,
    DefaultSetter,
    Getter,
    Setter,
    TextMapPropagator,
)
from opentelemetry.util.re import _DELIMITER_PATTERN

_logger = logging.getLogger(__name__)

_BAGGAGE_HEADER = "baggage"
_default_getter = DefaultGetter()
_default_setter = DefaultSetter()


def _resolve_context(context: Optional[Context]) -> Context:
    if context is not None:
        return context
    from judgeval.v1.trace.judgment_tracer_provider import JudgmentTracerProvider

    return JudgmentTracerProvider.get_instance().get_current_context()


def _format_baggage(baggage_entries: Mapping[str, object]) -> str:
    return ",".join(
        quote_plus(str(k)) + "=" + quote_plus(str(v))
        for k, v in baggage_entries.items()
    )


def _extract_first_element(
    items: Optional[Iterable[CarrierT]],
) -> Optional[CarrierT]:
    if items is None:
        return None
    return next(iter(items), None)


class JudgmentBaggagePropagator(TextMapPropagator):
    _MAX_HEADER_LENGTH = 8192
    _MAX_PAIR_LENGTH = 4096
    _MAX_PAIRS = 180

    def inject(
        self,
        carrier: object,
        context: Optional[Context] = None,
        setter: Setter = _default_setter,
    ) -> None:
        from judgeval.v1.trace.baggage import get_all

        entries = get_all(_resolve_context(context))
        if not entries:
            return
        setter.set(carrier, _BAGGAGE_HEADER, _format_baggage(entries))

    def extract(
        self,
        carrier: object,
        context: Optional[Context] = None,
        getter: Getter = _default_getter,
    ) -> Context:
        from judgeval.v1.trace.baggage import _is_valid_pair, set_baggage

        ctx = _resolve_context(context)
        header = _extract_first_element(getter.get(carrier, _BAGGAGE_HEADER))

        if not header:
            return ctx

        if len(header) > self._MAX_HEADER_LENGTH:
            _logger.warning(
                "Baggage header `%s` exceeded the maximum number of bytes per baggage-string",
                header,
            )
            return ctx

        baggage_entries: List[str] = split(_DELIMITER_PATTERN, header)
        total_baggage_entries = self._MAX_PAIRS

        if len(baggage_entries) > self._MAX_PAIRS:
            _logger.warning(
                "Baggage header `%s` exceeded the maximum number of list-members",
                header,
            )

        for entry in baggage_entries:
            if len(entry) > self._MAX_PAIR_LENGTH:
                _logger.warning(
                    "Baggage entry `%s` exceeded the maximum number of bytes per list-member",
                    entry,
                )
                continue
            if not entry:
                continue
            try:
                name, value = entry.split("=", 1)
            except Exception:
                _logger.warning(
                    "Baggage list-member `%s` doesn't match the format", entry
                )
                continue

            if not _is_valid_pair(name, value):
                _logger.warning("Invalid baggage entry: `%s`", entry)
                continue

            ctx = set_baggage(
                unquote_plus(name).strip(),
                unquote_plus(value).strip(),
                ctx,
            )
            total_baggage_entries -= 1
            if total_baggage_entries == 0:
                break

        return ctx

    @property
    def fields(self) -> Set[str]:
        return {_BAGGAGE_HEADER}
