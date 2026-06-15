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


@dataclass(slots=True)
class TraceRef:
    """A pointer to a trace by id, for use as a dataset example field value.

    A dataset column can be declared with `{"type": "trace"}`; its value is
    not literal data but a reference to a stored trace. Wrap the trace id in
    `TraceRef` so the SDK records the column as trace-typed when inferring a
    schema, and serializes the value as the bare trace id on upload:

        Example.create(question="...", transcript=TraceRef("<trace_id>"))

    Attributes:
        trace_id: The id of the trace this field points at.
    """

    trace_id: str

    def __str__(self) -> str:
        return self.trace_id
