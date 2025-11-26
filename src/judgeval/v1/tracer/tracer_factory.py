from __future__ import annotations

from typing import Any, Callable, Optional

from judgeval.utils.serialize import safe_serialize
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.tracer.judgment_tracer_provider import FilterTracerCallback
from judgeval.v1.tracer.tracer import Tracer


class TracerFactory:
    __slots__ = "_client"

    def __init__(
        self,
        client: JudgmentSyncClient,
    ):
        self._client = client

    def create(
        self,
        project_name: str,
        enable_evaluation: bool = True,
        enable_monitoring: bool = True,
        serializer: Callable[[Any], str] = safe_serialize,
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: bool = False,
    ) -> Tracer:
        return Tracer(
            project_name=project_name,
            enable_evaluation=enable_evaluation,
            enable_monitoring=enable_monitoring,
            api_client=self._client,
            serializer=serializer,
            filter_tracer=filter_tracer,
            isolated=isolated,
        )
