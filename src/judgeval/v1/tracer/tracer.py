from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional

from opentelemetry.sdk.resources import Resource

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.tracer.judgment_tracer_provider import (
    JudgmentTracerProvider,
    FilterTracerCallback,
)
from judgeval.version import get_version
from judgeval.v1.tracer.base_tracer import BaseTracer


class Tracer(BaseTracer):
    __slots__ = ()

    def __init__(
        self,
        project_name: str,
        enable_evaluation: bool,
        enable_monitoring: bool,
        api_client: JudgmentSyncClient,
        serializer: Callable[[Any], str],
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: bool = True,
        resource_attributes: Optional[Dict[str, Any]] = None,
        initialize: bool = True,
    ):
        if filter_tracer is not None:
            warnings.warn(
                "filter_tracer parameter is deprecated and will be ignored",
                DeprecationWarning,
                stacklevel=2,
            )
        if isolated is not True:
            warnings.warn(
                "isolated parameter is deprecated, tracers are always isolated",
                DeprecationWarning,
                stacklevel=2,
            )
        if initialize is not True:
            warnings.warn(
                "initialize parameter is deprecated and will be ignored",
                DeprecationWarning,
                stacklevel=2,
            )

        resource_attrs = {
            "service.name": project_name,
            "telemetry.sdk.name": self.TRACER_NAME,
            "telemetry.sdk.version": get_version(),
        }
        if resource_attributes:
            resource_attrs.update(resource_attributes)

        resource = Resource.create(resource_attrs)

        tracer_provider = JudgmentTracerProvider(resource=resource)

        super().__init__(
            project_name=project_name,
            enable_evaluation=enable_evaluation,
            enable_monitoring=enable_monitoring,
            api_client=api_client,
            serializer=serializer,
            tracer_provider=tracer_provider,
        )

        if enable_monitoring:
            judgeval_logger.info("Adding JudgmentSpanProcessor for monitoring.")
            tracer_provider.add_span_processor(self.get_span_processor())

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: int = 30000) -> None:
        self._tracer_provider.shutdown()
