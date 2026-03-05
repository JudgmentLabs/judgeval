from __future__ import annotations

from dataclasses import dataclass
from typing import List

from judgeval.v1.internal.api.models import TraceSpan


@dataclass(slots=True)
class Trace:
    spans: List[TraceSpan]
