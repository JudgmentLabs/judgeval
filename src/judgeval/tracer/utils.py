from typing import Any
from opentelemetry.trace import Span
from pydantic import BaseModel
from typing import Callable, Optional
from judgeval.scorers.api_scorer import TraceAPIScorerConfig
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL


def set_span_attribute(span: Span, name: str, value: Any):
    if value is None or value == "":
        return

    span.set_attribute(name, value)


class TraceScorerConfig(BaseModel):
    scorer: TraceAPIScorerConfig
    model: str = JUDGMENT_DEFAULT_GPT_MODEL
    sampling_rate: float = 1.0
    run_condition: Optional[Callable[..., bool]] = None
