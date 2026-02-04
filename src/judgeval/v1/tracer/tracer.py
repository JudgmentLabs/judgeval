from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.tracer.judgment_tracer_provider import JudgmentTracerProvider
from judgeval.version import get_version
from judgeval.v1.tracer.base_tracer import BaseTracer
from judgeval.v1.tracer.judgment_tracer_provider import FilterTracerCallback


class Tracer(BaseTracer):
    __slots__ = ("_filter_tracer",)

    def __init__(
        self,
        project_name: str,
        project_id: Optional[str],
        enable_evaluation: bool,
        enable_monitoring: bool,
        api_client: JudgmentSyncClient,
        serializer: Callable[[Any], str],
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: bool = False,
        resource_attributes: Optional[Dict[str, Any]] = None,
        initialize: bool = True,
        use_default_span_processor: bool = True,
    ):
        self._filter_tracer = filter_tracer

        resource_attrs = {
            "service.name": project_name,
            "telemetry.sdk.name": self.TRACER_NAME,
            "telemetry.sdk.version": get_version(),
        }
        if resource_attributes:
            resource_attrs.update(resource_attributes)

        resource = Resource.create(resource_attrs)

        tracer_provider = JudgmentTracerProvider(
            resource=resource,
            filter_tracer=self._filter_tracer,
            isolated=isolated,
        )

        if enable_monitoring and not project_id:
            judgeval_logger.warning(
                "Monitoring disabled: project_id is not set. "
                "Spans will not be exported."
            )
            enable_monitoring = False

        super().__init__(
            project_name=project_name,
            project_id=project_id,
            enable_evaluation=enable_evaluation,
            enable_monitoring=enable_monitoring,
            api_client=api_client,
            serializer=serializer,
            tracer_provider=tracer_provider,
        )

        if enable_monitoring and use_default_span_processor:
            judgeval_logger.info("Adding JudgmentSpanProcessor for monitoring.")
            tracer_provider.add_span_processor(self.get_span_processor())

        if initialize and enable_monitoring and not isolated:
            judgeval_logger.info("Setting global tracer provider for monitoring.")
            trace.set_tracer_provider(tracer_provider)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: int = 30000) -> None:
        self._tracer_provider.shutdown()
