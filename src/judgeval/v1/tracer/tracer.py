from __future__ import annotations

from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger
from judgeval.v1.tracer.judgment_tracer_provider import JudgmentTracerProvider
from judgeval.version import get_version
from judgeval.v1.tracer.base_tracer import BaseTracer
from judgeval.v1.tracer.judgment_tracer_provider import FilterTracerCallback


class Tracer(BaseTracer):
    __slots__ = ("_tracer_provider", "_filter_tracer")

    def __init__(
        self,
        project_name: str,
        enable_evaluation: bool,
        api_client: JudgmentSyncClient,
        serializer: Callable[[Any], str],
        initialize: bool,
        filter_tracer: Optional[FilterTracerCallback] = None,
    ):
        super().__init__(
            project_name=project_name,
            enable_evaluation=enable_evaluation,
            api_client=api_client,
            serializer=serializer,
        )
        self._tracer_provider: Optional[TracerProvider] = None
        self._filter_tracer = filter_tracer

        if initialize:
            self.initialize()

    def initialize(self) -> None:
        resource = Resource.create(
            {
                "service.name": self.project_name,
                "telemetry.sdk.name": self.TRACER_NAME,
                "telemetry.sdk.version": get_version(),
            }
        )

        self._tracer_provider = JudgmentTracerProvider(
            resource=resource, filter_tracer=self._filter_tracer
        )
        self._tracer_provider.add_span_processor(self.get_span_processor())

        trace.set_tracer_provider(self._tracer_provider)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if self._tracer_provider is None:
            judgeval_logger.error("Cannot forceFlush: tracer not initialized")
            return False
        return self._tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: int = 30000) -> None:
        if self._tracer_provider is None:
            judgeval_logger.error("Cannot shutdown: tracer not initialized")
            return
        self._tracer_provider.shutdown()
