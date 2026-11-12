"""Detached, bounded payload capture shared by MCP adapters."""

import json
from typing import Any, Callable, Literal, Optional

PayloadField = Literal["input", "output"]
RedactPayload = Callable[[Any, PayloadField], Any]
OMITTED = object()


def capture_payload(
    value: Any, field: PayloadField, redact: Optional[RedactPayload]
) -> Any:
    """Redact a detached JSON snapshot, omitting unsafe/oversized payloads."""
    try:
        snapshot = json.loads(json.dumps(value, allow_nan=False))
        redacted = snapshot if redact is None else redact(snapshot, field)
        if redacted is OMITTED:
            return OMITTED
        serialized = json.dumps(redacted, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > 128 * 1024:
            return OMITTED
        return json.loads(serialized)
    except Exception:
        return OMITTED
