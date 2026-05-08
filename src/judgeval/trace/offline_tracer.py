from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler

from judgeval.data.example import Example
from judgeval.internal.api import JudgmentSyncClient
from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.trace.processors.offline_judgment_span_processor import (
    OfflineJudgmentSpanProcessor,
)
from judgeval.trace.tracer import Tracer
from judgeval.utils.serialize import safe_serialize


OFFLINE_TRACES_PATH = "otel/v1/offline-traces"


class OfflineTracer(Tracer):
    """Tracer for offline / experiment-style runs.

    Behaves like ``Tracer`` for span creation and ``@Tracer.observe``,
    with two differences:

    * Spans are pushed to the project's *offline* OTLP endpoint and stored
      in the ``offline_otel_traces`` ClickHouse table. They do **not**
      appear on the live monitoring page.
    * Each completed root span produces a new ``Example`` that is
      appended to the caller-supplied ``dataset`` list. The example
      carries the ``offline_trace_id`` of the offline trace plus any
      static ``example_fields`` configured at init time.

    OfflineTracer is **not** constructed directly by user code. Use
    ``Judgeval.offline_tracer`` instead, which fills in project /
    credentials from the active ``Judgeval`` client:

    ```python
    client = Judgeval(project_name="default_project")
    results: list[Example] = []
    tracer = client.offline_tracer.init(
        dataset=results,
        example_fields={"input": "...", "golden_output": "..."},
    )
    ```
    """

    __slots__ = (
        "_dataset",
        "_example_fields",
    )

    # The offline processor must run even when credentials are missing so
    # the dataset list still gets populated.
    _ALWAYS_ATTACH_PROCESSOR: ClassVar[bool] = True

    def __init__(
        self,
        project_name: Optional[str],
        project_id: Optional[str],
        api_key: Optional[str],
        organization_id: Optional[str],
        api_url: Optional[str],
        environment: Optional[str],
        serializer: Callable[[Any], str],
        tracer_provider: TracerProvider,
        enable_monitoring: bool,
        client: Optional[JudgmentSyncClient],
        dataset: List[Example],
        example_fields: Optional[Dict[str, Any]],
    ):
        super().__init__(
            project_name=project_name,
            project_id=project_id,
            api_key=api_key,
            organization_id=organization_id,
            api_url=api_url,
            environment=environment,
            serializer=serializer,
            tracer_provider=tracer_provider,
            enable_monitoring=enable_monitoring,
            client=client,
        )
        self._dataset = dataset
        self._example_fields: Dict[str, Any] = dict(example_fields or {})

    @classmethod
    def init(  # type: ignore[override]
        cls,
        project_name: Optional[str] = None,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        api_url: Optional[str] = None,
        environment: Optional[str] = None,
        set_active: bool = True,
        serializer: Callable[[Any], str] = safe_serialize,
        resource_attributes: Optional[Dict[str, Any]] = None,
        sampler: Optional[Sampler] = None,
        span_limits: Optional[SpanLimits] = None,
        span_processors: Optional[Sequence[SpanProcessor]] = None,
        *,
        dataset: List[Example],
        example_fields: Optional[Dict[str, Any]] = None,
    ) -> "OfflineTracer":
        """Create and activate a new OfflineTracer.

        Args mirror ``Tracer.init``, plus:
            dataset: Caller-owned list. Each completed root span appends a
                new ``Example`` with the offline trace id and any
                ``example_fields`` configured here.
            example_fields: Static fields copied onto every emitted
                example (e.g. ``{"input": ..., "golden_output": ...}``).

        Prefer ``Judgeval.offline_tracer`` over calling this directly so
        credentials are reused from the ``Judgeval`` client.
        """
        return super().init(  # type: ignore[return-value]
            project_name=project_name,
            api_key=api_key,
            organization_id=organization_id,
            api_url=api_url,
            environment=environment,
            set_active=set_active,
            serializer=serializer,
            resource_attributes={
                "judgment.offline": "true",
                **(resource_attributes or {}),
            },
            sampler=sampler,
            span_limits=span_limits,
            span_processors=span_processors,
            _extra_init_kwargs={
                "dataset": dataset,
                "example_fields": example_fields,
            },
        )

    # ------------------------------------------------------------------ #
    # Exporter / processor overrides — the only behavioral differences
    # vs. plain Tracer.
    # ------------------------------------------------------------------ #

    def get_span_exporter(self) -> JudgmentSpanExporter:
        """Return the offline span exporter for this tracer.

        Falls back to a no-op exporter when monitoring is disabled.
        """
        if self._span_exporter is not None:
            return self._span_exporter

        if (
            not self._enable_monitoring
            or not self.project_id
            or not self.api_key
            or not self.organization_id
            or not self.api_url
        ):
            self._span_exporter = NoOpJudgmentSpanExporter()
        else:
            base = self.api_url.rstrip("/")
            endpoint = f"{base}/{OFFLINE_TRACES_PATH}"
            self._span_exporter = JudgmentSpanExporter(
                endpoint=endpoint,
                api_key=self.api_key,
                organization_id=self.organization_id,
                project_id=self.project_id,
            )
        return self._span_exporter

    def get_span_processor(self) -> OfflineJudgmentSpanProcessor:  # type: ignore[override]
        """Return the offline span processor for this tracer.

        Unlike the online processor, this one is wired up regardless of
        ``enable_monitoring`` so the dataset list is still populated when
        spans only route to a no-op exporter (e.g. credentials missing).
        """
        if self._span_processor is not None:
            assert isinstance(self._span_processor, OfflineJudgmentSpanProcessor)
            return self._span_processor

        processor = OfflineJudgmentSpanProcessor(
            self,
            self.get_span_exporter(),
            dataset=self._dataset,
            example_fields=self._example_fields,
        )
        self._span_processor = processor
        return processor
