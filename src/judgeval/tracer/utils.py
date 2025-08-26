from typing import Any
from opentelemetry.trace import Span


def set_span_attribute(span: Span, name: str, value: Any):
    if value is None or value == "":
        return

    span.set_attribute(name, value)
