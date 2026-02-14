from __future__ import annotations

from typing import Callable, List

from opentelemetry.sdk.trace import SpanProcessor


ProcessorFactory = Callable[[], SpanProcessor]

_lifecycle_processors: List[ProcessorFactory] = []


def register(processor_class: ProcessorFactory) -> None:
    _lifecycle_processors.append(processor_class)


def get_all() -> List[SpanProcessor]:
    return [factory() for factory in _lifecycle_processors]
