from __future__ import annotations

from typing import Callable, Optional

from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer, NoOpTracer
from opentelemetry.util.types import Attributes

from judgeval.logger import judgeval_logger
from judgeval.v1.tracer.base_tracer import BaseTracer
from judgeval.v1.tracer.id_generator import IsolatedRandomIdGenerator
from judgeval.v1.tracer.isolated import JudgmentIsolatedTracer

FilterTracerCallback = Callable[[str, Optional[str], Optional[str], Attributes], bool]


class JudgmentTracerProvider(TracerProvider):
    __slots__ = ("_filter_tracer", "_isolated", "_runtime_context")

    def __init__(
        self,
        filter_tracer: Optional[FilterTracerCallback] = None,
        isolated: bool = False,
        **kwargs,
    ):
        # Use our isolated ID generator to be immune to user random.seed() calls
        if "id_generator" not in kwargs:
            kwargs["id_generator"] = IsolatedRandomIdGenerator()
        super().__init__(**kwargs)
        self._filter_tracer = filter_tracer or (lambda *_: True)
        self._isolated = isolated
        self._runtime_context = (
            ContextVarsRuntimeContext()
        )  # Only used for isolated tracers

    def get_isolated_current_context(self):
        return self._runtime_context.get_current()

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Attributes = None,
    ) -> Tracer:
        is_judgment_module = instrumenting_module_name == BaseTracer.TRACER_NAME

        if self._isolated and not is_judgment_module:
            judgeval_logger.debug(
                f"Returning NoOpTracer for {instrumenting_module_name} (isolated mode)"
            )
            return NoOpTracer()

        if not is_judgment_module:
            try:
                if not self._filter_tracer(
                    instrumenting_module_name,
                    instrumenting_library_version,
                    schema_url,
                    attributes,
                ):
                    judgeval_logger.debug(
                        f"Returning NoOpTracer for {instrumenting_module_name} (filtered)"
                    )
                    return NoOpTracer()
            except Exception as error:
                judgeval_logger.error(
                    f"Failed to filter tracer {instrumenting_module_name}: {error}"
                )

        tracer = super().get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            schema_url,
            attributes,
        )

        if self._isolated and is_judgment_module:
            return JudgmentIsolatedTracer(tracer, self._runtime_context)

        return tracer
