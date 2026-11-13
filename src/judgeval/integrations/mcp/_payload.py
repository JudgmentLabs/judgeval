"""Detached, bounded payload capture shared by MCP adapters."""

import json
from typing import Any, Callable, Literal, Mapping, Optional

from pydantic import BaseModel

PayloadField = Literal["input", "output"]
RedactPayload = Callable[[Any, PayloadField], Any]
OMITTED = object()
_MAX_PAYLOAD_BYTES = 128 * 1024


def result_payload(value: Any) -> Any:
    """Check result contents before allocating their JSON representation."""
    try:
        _remaining_budget(value, _MAX_PAYLOAD_BYTES)
        if isinstance(value, Mapping):
            return dict(value)
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    except Exception:
        return OMITTED


def capture_payload(
    value: Any, field: PayloadField, redact: Optional[RedactPayload]
) -> Any:
    """Redact a detached JSON snapshot, omitting unsafe/oversized payloads."""
    try:
        snapshot = json.loads(_bounded_json(value))
        redacted = snapshot if redact is None else redact(snapshot, field)
        if redacted is OMITTED:
            return OMITTED
        return json.loads(_bounded_json(redacted))
    except Exception:
        return OMITTED


def _bounded_json(value: Any) -> str:
    # iterencode can emit an entire string as one chunk. Bound source strings
    # and traversal first, then enforce the exact encoded UTF-8 byte limit.
    _remaining_budget(value, _MAX_PAYLOAD_BYTES)
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False)
    remaining = _MAX_PAYLOAD_BYTES
    chunks = []
    for chunk in encoder.iterencode(value):
        remaining -= len(chunk.encode("utf-8"))
        if remaining < 0:
            raise ValueError("MCP payload exceeds capture limit")
        chunks.append(chunk)
    return "".join(chunks)


def _remaining_budget(value: Any, remaining: int) -> int:
    remaining -= 1
    if remaining < 0:
        raise ValueError("MCP payload exceeds capture limit")
    if isinstance(value, BaseModel):
        remaining = _remaining_budget(value.__dict__, remaining)
        if value.__pydantic_extra__:
            remaining = _remaining_budget(value.__pydantic_extra__, remaining)
    elif isinstance(value, (str, bytes)):
        remaining -= len(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            remaining = _remaining_budget(key, remaining)
            remaining = _remaining_budget(item, remaining)
    elif isinstance(value, (list, tuple)):
        for item in value:
            remaining = _remaining_budget(item, remaining)
    if remaining < 0:
        raise ValueError("MCP payload exceeds capture limit")
    return remaining
