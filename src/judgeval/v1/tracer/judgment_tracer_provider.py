from __future__ import annotations

from typing import Callable, Optional

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer, NoOpTracer
from opentelemetry.util.types import Attributes

from judgeval.logger import judgeval_logger
from judgeval.v1.tracer.base_tracer import BaseTracer

FilterTracerCallback = Callable[[str, Optional[str], Optional[str], Attributes], bool]


class JudgmentTracerProvider(TracerProvider):
    __slots__ = ("_filter_tracer",)

    def __init__(
        self,
        filter_tracer: Optional[FilterTracerCallback] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._filter_tracer = (
            filter_tracer if filter_tracer is not None else lambda *_: True
        )

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Attributes = None,
    ) -> Tracer:
        if instrumenting_module_name == BaseTracer.TRACER_NAME:
            return super().get_tracer(
                instrumenting_module_name,
                instrumenting_library_version,
                schema_url,
                attributes,
            )

        try:
            if self._filter_tracer(
                instrumenting_module_name,
                instrumenting_library_version,
                schema_url,
                attributes,
            ):
                return super().get_tracer(
                    instrumenting_module_name,
                    instrumenting_library_version,
                    schema_url,
                    attributes,
                )
            else:
                judgeval_logger.debug(
                    f"[JudgmentTracerProvider] Returning NoOpTracer for tracer {instrumenting_module_name} as it is disallowed by the filterTracer callback."
                )
                return NoOpTracer()
        except Exception as error:
            judgeval_logger.error(
                f"[JudgmentTracerProvider] Failed to filter tracer {instrumenting_module_name}: {error}."
            )
            return super().get_tracer(
                instrumenting_module_name,
                instrumenting_library_version,
                schema_url,
                attributes,
            )
