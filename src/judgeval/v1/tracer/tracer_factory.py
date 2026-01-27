from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional

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
        isolated: bool = True,
        resource_attributes: Optional[Dict[str, Any]] = None,
        initialize: bool = True,
    ) -> Tracer:
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
        return Tracer(
            project_name=project_name,
            enable_evaluation=enable_evaluation,
            enable_monitoring=enable_monitoring,
            api_client=self._client,
            serializer=serializer,
            resource_attributes=resource_attributes,
        )
