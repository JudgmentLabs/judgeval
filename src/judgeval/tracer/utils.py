from typing import Any
from opentelemetry.trace import Span
from pydantic import BaseModel
from typing import Callable, Optional
from judgeval.scorers.trace_api_scorer import TraceAPIScorerConfig


def set_span_attribute(span: Span, name: str, value: Any):
    if value is None or value == "":
        return

    span.set_attribute(name, value)


class TraceScorerConfig(BaseModel):
    scorer: TraceAPIScorerConfig
    sampling_rate: float
    model: str
    run_condition: Optional[Callable[..., bool]]
