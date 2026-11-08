"""Removes personal identifiers and echoed argument values from recorded text."""

from __future__ import annotations

import json
import re
from itertools import islice
from typing import Any, List, Mapping, Sequence, Set

ARGUMENT_PLACEHOLDER = "[argument]"
# Bounds the pattern matching below. Callers keep far less text than this.
_MAX_SCANNED_LENGTH = 4_000
# An error can echo a long argument, so argument values are removed from more.
_MAX_ECHO_SCANNED_LENGTH = 100_000
# Shorter values are left: replacing them would garble more than it hides.
_MIN_ARGUMENT_LENGTH = 4
_MAX_ARGUMENT_VALUES = 100
_MAX_ARGUMENT_NODES = 1_000

# Each pattern can start only where its identifier can, so a long run of
# characters is scanned once, not once per position.
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 13 to 19 digits, optionally grouped, without taking the space after them.
_CARD = re.compile(r"\d(?:[ -]?\d){12,18}")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Full, or compressed such as ``2001:db8::1`` and ``::1``. Not ``10:30:00``.
_IPV6 = re.compile(
    r"(?<![\w:])(?:"
    r"(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}"
    r"|(?=[0-9a-f:]*[0-9a-f])"
    r"(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,6})?"
    r"::(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4}){0,6})?"
    r")(?![\w:])",
    re.IGNORECASE,
)
# Nine or more digits and separators, except a date such as 2026-11-08.
_PHONE = re.compile(r"(?<![\w.-])(?!\d{4}-\d{2}-\d{2}(?!\d))\+?\d[\d ().-]{7,}\d\b")
# pydantic prints the input it rejected, possibly shortened, as ``input_value=``.
_PYDANTIC_INPUT = re.compile(r"input_value=[^\n]*?(?=, input_type=|\n|$)")


def scrub_identifiers(text: str) -> str:
    """Replaces emails, SSNs, card numbers, IP addresses and phone numbers.

    It can't recognize free-form data such as names.
    """
    text = text[:_MAX_SCANNED_LENGTH]
    text = _EMAIL.sub("[email]", text)
    text = _SSN.sub("[ssn]", text)
    text = _CARD.sub(
        lambda match: "[card]"
        if _is_luhn_valid(re.sub(r"[ -]", "", match.group(0)))
        else match.group(0),
        text,
    )
    text = _IPV4.sub("[ip]", text)
    text = _IPV6.sub("[ip]", text)
    return _PHONE.sub("[phone]", text)


def argument_values(params: Mapping[str, Any]) -> List[str]:
    """How a request's text arguments could appear in an error, longest first.

    Covers the tool or prompt arguments, nested ones included, and a resource
    URI, each as sent and as Python and JSON would quote it.
    """
    found: Set[str] = set()
    pending: List[Any] = [params.get("arguments"), params.get("uri")]
    visited = 0
    while pending and visited < _MAX_ARGUMENT_NODES:
        value = pending.pop()
        visited += 1
        if isinstance(value, str):
            for form in (value, repr(value)[1:-1], json.dumps(value)[1:-1]):
                if len(form) >= _MIN_ARGUMENT_LENGTH:
                    found.add(form)
            if len(found) >= _MAX_ARGUMENT_VALUES:
                break
        elif isinstance(value, (Mapping, list, tuple)):
            children = value.values() if isinstance(value, Mapping) else value
            pending.extend(islice(children, _MAX_ARGUMENT_NODES - len(pending)))
    return sorted(found, key=len, reverse=True)


def scrub_error_message(message: str, values: Sequence[str]) -> str:
    """An error message without echoed argument values or personal identifiers.

    Tools and SDKs often repeat the input they rejected, which can be a token
    or someone's data, so each value becomes ``[argument]``. That also lets the
    same failure group together across calls.
    """
    text = message[:_MAX_ECHO_SCANNED_LENGTH]
    for value in values:
        text = text.replace(value, ARGUMENT_PLACEHOLDER)
    text = _PYDANTIC_INPUT.sub(
        f"input_value={ARGUMENT_PLACEHOLDER}", text[:_MAX_SCANNED_LENGTH]
    )
    return scrub_identifiers(text)


def _is_luhn_valid(digits: str) -> bool:
    total = 0
    for index, character in enumerate(reversed(digits)):
        digit = int(character)
        if index % 2 == 0:
            total += digit
        else:
            total += digit * 2 - 9 if digit > 4 else digit * 2
    return total % 10 == 0
