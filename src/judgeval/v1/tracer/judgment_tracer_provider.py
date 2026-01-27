from __future__ import annotations

import warnings
from typing import Callable, Optional

from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from opentelemetry.util.types import Attributes

from judgeval.v1.tracer.id_generator import IsolatedRandomIdGenerator
from judgeval.v1.tracer.isolated import JudgmentIsolatedTracer

FilterTracerCallback = Callable[[str, Optional[str], Optional[str], Attributes], bool]


class JudgmentTracerProvider(TracerProvider):
    __slots__ = ("_runtime_context",)

    def __init__(
        self,
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: Optional[bool] = None,
        **kwargs,
    ):
        if filter_tracer is not None:
            warnings.warn(
                "filter_tracer parameter is deprecated and will be ignored",
                DeprecationWarning,
                stacklevel=2,
            )
        if isolated is not None:
            warnings.warn(
                "isolated parameter is deprecated, tracers are always isolated",
                DeprecationWarning,
                stacklevel=2,
            )
        if "id_generator" not in kwargs:
            kwargs["id_generator"] = IsolatedRandomIdGenerator()
        super().__init__(**kwargs)
        self._runtime_context = ContextVarsRuntimeContext()

    def get_isolated_current_context(self):
        return self._runtime_context.get_current()

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Attributes = None,
    ) -> Tracer:
        tracer = super().get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            schema_url,
            attributes,
        )
        return JudgmentIsolatedTracer(tracer, self._runtime_context)
