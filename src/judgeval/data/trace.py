from __future__ import annotations

from dataclasses import dataclass
from typing import List

from judgeval.internal.api.models import TraceSpan


@dataclass(slots=True)
class Trace:
    """A recorded execution trace consisting of one or more spans.

    Represents a complete request lifecycle as captured by `Tracer`. Each
    span in the trace corresponds to a function call, LLM request, or
    tool invocation.

    Attributes:
        spans: The spans in this trace, ordered by start time.
    """

    spans: List[TraceSpan]
