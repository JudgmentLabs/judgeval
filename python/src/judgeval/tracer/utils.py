from typing import Any
from opentelemetry.trace import Span
from pydantic import BaseModel
from typing import Callable, Optional
from judgeval.scorers.api_scorer import TraceAPIScorerConfig


def set_span_attribute(span: Span, name: str, value: Any):
    if value is None or value == "":
        return

    span.set_attribute(name, value)


class TraceScorerConfig(BaseModel):
    scorer: TraceAPIScorerConfig | None
    model: Optional[str] = None
    sampling_rate: float = 1.0
    run_condition: Optional[Callable[..., bool]] = None
