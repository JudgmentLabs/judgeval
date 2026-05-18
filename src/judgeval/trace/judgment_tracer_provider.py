from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, ClassVar, Optional, Sequence
from weakref import WeakSet

from opentelemetry import trace as trace_api
from opentelemetry.context.context import Context
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.trace import Link, NoOpTracer, Span, SpanKind, Tracer
from opentelemetry.util.types import Attributes
from opentelemetry.util._decorator import _agnosticcontextmanager

from judgeval.logger import judgeval_logger
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME

if TYPE_CHECKING:
    from judgeval.trace.tracer import Tracer as JudgmentTracer

_Links = Optional[Sequence[Link]]
_active_tracer_var: ContextVar[Optional[JudgmentTracer]] = ContextVar(
    "active_tracer", default=None
)


class ProxyTracer(Tracer):
    """Internal tracer that delegates to the currently active ``JudgmentTracer``.

    All span creation goes through this proxy so that instrumentation
    libraries always target whichever tracer is active in the current
    context, even when multiple tracers exist.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: JudgmentTracerProvider):
        self._provider = provider

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: _Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        if context is None:
            context = self._provider.get_current_context()
        delegate = self._provider._get_delegate_tracer()
        return delegate.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: _Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ):
        if context is None:
            context = self._provider.get_current_context()
        span = self.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )
        with self._provider.use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as s:
            yield s


class JudgmentTracerProvider(TracerProvider):
    """Global singleton that manages Judgment tracers and context propagation.

    Acts as the OpenTelemetry ``TracerProvider`` for the Judgment SDK.
    It maintains a ``WeakSet`` of registered tracers and routes all span
    creation through a ``ProxyTracer`` that delegates to the currently
    active tracer.

    You don't create this directly -- ``Tracer.init()`` registers with
    the singleton automatically. Access it via ``get_instance()`` when
    building custom integrations.

    Examples:
        ```python
        from judgeval.trace import JudgmentTracerProvider

        provider = JudgmentTracerProvider.get_instance()
        span = provider.get_current_span()
        ```
    """

    _instance: ClassVar[Optional[JudgmentTracerProvider]] = None

    __slots__ = (
        "_runtime_context",
        "_instrumentations",
        "_proxy_tracer",
        "_tracers",
        "_external_span_processors",
    )

    def __init__(self):
        super().__init__(shutdown_on_exit=False)
        self._runtime_context = ContextVarsRuntimeContext()
        self._instrumentations: list = []
        self._proxy_tracer = ProxyTracer(self)
        self._tracers: WeakSet[JudgmentTracer] = WeakSet()
        self._external_span_processors: list[SpanProcessor] = []

    @classmethod
    def get_instance(cls) -> JudgmentTracerProvider:
        """Return the global singleton, creating it on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def install_as_global_tracer_provider(cls) -> bool:
        """Install this provider as the OpenTelemetry global tracer provider.

        Returns True if the provider was successfully installed, False if
        another provider was already set (OpenTelemetry enforces
        first-writer-wins semantics).
        """
        instance = cls.get_instance()
        trace_api.set_tracer_provider(instance)
        installed = trace_api.get_tracer_provider() is instance
        if not installed:
            judgeval_logger.warning(
                "Failed to install JudgmentTracerProvider as the global "
                "tracer provider. Another TracerProvider was already "
                "installed. Spans created by external instrumentation "
                "may not be captured by Judgment."
            )
        return installed

    def register(self, tracer: JudgmentTracer) -> None:
        """Add a tracer to the tracked set (weak reference).

        Any span processors previously added via ``add_span_processor``
        are automatically forwarded to the tracer's underlying provider.
        """
        self._tracers.add(tracer)
        for processor in self._external_span_processors:
            tracer._tracer_provider.add_span_processor(processor)

    def deregister(self, tracer: JudgmentTracer) -> None:
        """Remove a tracer from the tracked set."""
        self._tracers.discard(tracer)

    def set_active(self, tracer: JudgmentTracer) -> bool:
        """Set a tracer as the active tracer for the current context.

        Fails if a root span is currently recording to prevent mid-trace
        provider switches.

        Returns:
            True if the tracer was activated, False if blocked by an
            active root span.
        """
        current_span = self.get_current_span()
        if current_span is not None and current_span.is_recording():
            parent = getattr(current_span, "parent", None)
            if parent is None:
                judgeval_logger.error(
                    "Cannot set_active() while a root span is active. "
                    "Keeping existing tracer provider."
                )
                return False
        self.register(tracer)
        _active_tracer_var.set(tracer)
        return True

    def get_active_tracer(self) -> Optional[JudgmentTracer]:
        """Return the tracer active in the current async context, or None."""
        return _active_tracer_var.get()

    def get_current_context(self) -> Context:
        """Return the current OpenTelemetry context."""
        return self._runtime_context.get_current()

    def get_current_span(self) -> Span:
        """Return the span that is active in the current context."""
        ctx = self.get_current_context()
        return trace_api.get_current_span(ctx)

    def has_active_root_span(self) -> bool:
        """Check whether a root span (no parent) is currently recording."""
        current_span = self.get_current_span()
        if current_span is None or not current_span.is_recording():
            return False
        return getattr(current_span, "parent", None) is None

    def _get_delegate_tracer(self) -> Tracer:
        tracer = _active_tracer_var.get()
        if tracer is None:
            judgeval_logger.debug("No active tracer, returning NoOpTracer")
            return NoOpTracer()
        return tracer._tracer_provider.get_tracer(
            JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME
        )

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Attributes = None,
    ) -> Tracer:
        return self._proxy_tracer

    def add_span_processor(self, span_processor: SpanProcessor) -> None:
        """Register a span processor with all managed tracers.

        Processors are forwarded to the underlying ``TracerProvider`` of
        every currently registered ``JudgmentTracer``, and will be
        automatically added to any tracer registered in the future via
        ``register()``.
        """
        self._external_span_processors.append(span_processor)
        for tracer in self._tracers:
            tracer._tracer_provider.add_span_processor(span_processor)

    def add_instrumentation(self, instrumentor) -> None:
        """Register and activate a third-party OTel instrumentor."""
        try:
            instrumentor.instrument(tracer_provider=self)
            self._instrumentations.append(instrumentor)
        except Exception as e:
            judgeval_logger.error(f"Failed to add instrumentation: {e}")

    @_agnosticcontextmanager
    def use_span(
        self,
        span: Span,
        end_on_exit: bool = False,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ):
        from opentelemetry.trace import Status, StatusCode

        try:
            ctx = trace_api.set_span_in_context(span, self.get_current_context())
            token = self._runtime_context.attach(ctx)
            try:
                yield span
            finally:
                self._runtime_context.detach(token)
        except Exception as exc:
            if isinstance(span, Span) and span.is_recording():
                if record_exception:
                    span.record_exception(exc)
                if set_status_on_exception:
                    span.set_status(
                        Status(
                            status_code=StatusCode.ERROR,
                            description=f"{type(exc).__name__}: {exc}",
                        )
                    )
            raise
        finally:
            if end_on_exit:
                span.end()

    def attach_context(self, ctx: Context) -> object:
        return self._runtime_context.attach(ctx)

    def detach_context(self, token) -> None:
        self._runtime_context.detach(token)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush pending spans from all registered tracers."""
        results = [
            t._tracer_provider.force_flush(timeout_millis) for t in self._tracers
        ]
        results.append(self._active_span_processor.force_flush(timeout_millis))
        return all(results)

    def shutdown(self) -> None:
        """Shut down all registered tracers and clear the tracked set."""
        for t in self._tracers:
            t._tracer_provider.shutdown()
        self._tracers.clear()
        self._active_span_processor.shutdown()
