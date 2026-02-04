from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from judgeval.utils.serialize import safe_serialize
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.tracer.judgment_tracer_provider import FilterTracerCallback
from judgeval.v1.tracer.tracer import Tracer


class TracerFactory:
    __slots__ = ("_client", "_project_name", "_project_id")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_name: str,
        project_id: Optional[str],
    ):
        self._client = client
        self._project_name = project_name
        self._project_id = project_id

    def create(
        self,
        enable_evaluation: bool = True,
        enable_monitoring: bool = True,
        serializer: Callable[[Any], str] = safe_serialize,
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: bool = False,
        resource_attributes: Optional[Dict[str, Any]] = None,
        initialize: bool = True,
        use_default_span_processor: bool = True,
    ) -> Tracer:
        return Tracer(
            project_name=self._project_name,
            project_id=self._project_id,
            enable_evaluation=enable_evaluation,
            enable_monitoring=enable_monitoring,
            api_client=self._client,
            serializer=serializer,
            filter_tracer=filter_tracer,
            isolated=isolated,
            resource_attributes=resource_attributes,
            initialize=initialize,
            use_default_span_processor=use_default_span_processor,
        )
