from __future__ import annotations

from typing import Callable, Iterable, Optional

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.trace import Tracer, NoOpTracer
from opentelemetry.util.types import Attributes

from judgeval.logger import judgeval_logger
from judgeval.v1.tracer.base_tracer import BaseTracer
from judgeval.v1.tracer.isolated import JudgmentIsolatedTracer

FilterTracerCallback = Callable[[str, Optional[str], Optional[str], Attributes], bool]
SpanFilterCallback = Callable[[ReadableSpan], bool]


class JudgmentTracerProvider(TracerProvider):
    __slots__ = (
        "_filter_tracer",
        "_isolated",
        "_isolated_allowed_modules",
    )

    _filter_tracer: FilterTracerCallback
    _isolated: bool
    _isolated_allowed_modules: Iterable[str]

    def __init__(
        self,
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: bool = False,
        isolated_allowed_modules: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._filter_tracer = filter_tracer or (lambda *_: True)
        self._isolated = isolated
        self._isolated_allowed_modules = isolated_allowed_modules or []

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Attributes = None,
    ) -> Tracer:
        try:
            if (
                instrumenting_module_name != BaseTracer.TRACER_NAME
                and not self._filter_tracer(
                    instrumenting_module_name,
                    instrumenting_library_version,
                    schema_url,
                    attributes,
                )
            ):
                judgeval_logger.debug(
                    f"Returning NoOpTracer for {instrumenting_module_name} (filtered)"
                )
                return NoOpTracer()
        except Exception as error:
            judgeval_logger.error(
                f"Failed to filter tracer {instrumenting_module_name}: {error}"
            )

        tracer = self._get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            schema_url,
            attributes,
        )

        if self._isolated and instrumenting_module_name == BaseTracer.TRACER_NAME:
            return JudgmentIsolatedTracer(tracer)
        return tracer

    def _get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str],
        schema_url: Optional[str],
        attributes: Attributes,
    ) -> Tracer:
        return super().get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            schema_url,
            attributes,
        )
