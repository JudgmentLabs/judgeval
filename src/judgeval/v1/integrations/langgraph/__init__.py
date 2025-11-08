from __future__ import annotations

from abc import ABC
import os


class Langgraph(ABC):
    @staticmethod
    def initialize(otel_only: bool = True):
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        if otel_only:
            os.environ["LANGSMITH_OTEL_ONLY"] = "true"
