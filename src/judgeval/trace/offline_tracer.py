from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler

from judgeval.data.example import Example
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.internal.api import JudgmentSyncClient
from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.trace.id_generator import IsolatedRandomIdGenerator
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
from judgeval.trace.processors.offline_judgment_span_processor import (
    OfflineJudgmentSpanProcessor,
)
from judgeval.trace.tracer import Tracer
from judgeval.utils import resolve_project_id
from judgeval.utils.serialize import safe_serialize
from judgeval.version import get_version


OFFLINE_TRACES_PATH = "otel/v1/offline-traces"


class OfflineTracer(Tracer):
    """Tracer for offline / experiment-style runs.

    Behaves like `Tracer` for span creation and `@Tracer.observe`, with
    two differences:

    * Spans are pushed to the project's *offline* OTLP endpoint and stored
      in the `offline_otel_traces` ClickHouse table. They do **not**
      appear on the live monitoring page.
    * Each completed root span produces a new `Example` that is appended
      to the caller-supplied `dataset` list. The example carries the
      `offline_trace_id` of the offline trace plus any static
      `example_fields` configured at init time.

    Unlike `Tracer`, `OfflineTracer` requires all credentials upfront and
    raises `ValueError` if any are missing — there is no no-op fallback.
    Prefer `Judgeval.offline_tracer(...)` over calling `OfflineTracer.create`
    directly so credentials are reused from the active `Judgeval` client.
    """

    __slots__ = (
        "_dataset",
        "_example_fields",
    )

    SUPPORTS_LIVE_INSTRUMENTATION: ClassVar[bool] = False

    def __init__(
        self,
        project_name: str,
        project_id: str,
        api_key: str,
        organization_id: str,
        api_url: str,
        environment: Optional[str],
        serializer: Callable[[Any], str],
        tracer_provider: TracerProvider,
        client: JudgmentSyncClient,
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
            enable_monitoring=True,
            client=client,
        )
        self._dataset = dataset
        self._example_fields: Dict[str, Any] = dict(example_fields or {})

    @classmethod
    def create(
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
        """Create and activate a new `OfflineTracer`.

        Args mirror `Tracer.init` plus:
            dataset: Caller-owned list. Each completed root span appends a
                new `Example` carrying the `offline_trace_id` of the trace
                and the static `example_fields`.
            example_fields: Static fields copied onto every emitted example
                (e.g. `{"input": ..., "golden_output": ...}`).

        Raises:
            ValueError: If `project_name`, `api_key`, `organization_id`, or
                `api_url` cannot be resolved (explicit arg or env var), or
                if the project cannot be found on the backend.
        """
        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID
        api_url = api_url or JUDGMENT_API_URL

        if not project_name:
            raise ValueError("project_name is required for OfflineTracer")
        if not api_key:
            raise ValueError("api_key is required for OfflineTracer")
        if not organization_id:
            raise ValueError("organization_id is required for OfflineTracer")
        if not api_url:
            raise ValueError("api_url is required for OfflineTracer")

        client = JudgmentSyncClient(api_url, api_key, organization_id)
        project_id = resolve_project_id(client, project_name)
        if not project_id:
            raise ValueError(
                f"Project '{project_name}' not found; cannot start OfflineTracer"
            )

        resource_attrs: Dict[str, Any] = {
            "service.name": project_name,
            "telemetry.sdk.name": cls.TRACER_NAME,
            "telemetry.sdk.version": get_version(),
            "judgment.offline": "true",
        }
        if environment:
            resource_attrs["deployment.environment"] = environment
        if resource_attributes:
            resource_attrs.update(resource_attributes)

        tracer_provider = TracerProvider(
            resource=Resource.create(resource_attrs),
            id_generator=IsolatedRandomIdGenerator(),
            sampler=sampler,
            span_limits=span_limits,
        )

        tracer = cls(
            project_name=project_name,
            project_id=project_id,
            api_key=api_key,
            organization_id=organization_id,
            api_url=api_url,
            environment=environment,
            serializer=serializer,
            tracer_provider=tracer_provider,
            client=client,
            dataset=dataset,
            example_fields=example_fields,
        )

        tracer_provider.add_span_processor(tracer.get_span_processor())
        for processor in span_processors or []:
            tracer_provider.add_span_processor(processor)

        proxy = JudgmentTracerProvider.get_instance()
        proxy.register(tracer)

        if set_active:
            tracer.set_active()

        return tracer

    def get_span_exporter(self) -> JudgmentSpanExporter:
        """Return the offline span exporter for this tracer.

        Targets the project's offline OTLP endpoint. Credentials are
        guaranteed present (validated in `create`).
        """
        if self._span_exporter is None:
            assert self.api_url is not None
            assert self.api_key is not None
            assert self.organization_id is not None
            assert self.project_id is not None
            self._span_exporter = JudgmentSpanExporter(
                endpoint=f"{self.api_url.rstrip('/')}/{OFFLINE_TRACES_PATH}",
                api_key=self.api_key,
                organization_id=self.organization_id,
                project_id=self.project_id,
            )
        return self._span_exporter

    def get_span_processor(self) -> OfflineJudgmentSpanProcessor:
        """Return the offline span processor for this tracer."""
        if self._span_processor is None:
            self._span_processor = OfflineJudgmentSpanProcessor(
                self,
                self.get_span_exporter(),
                dataset=self._dataset,
                example_fields=self._example_fields,
            )
        assert isinstance(self._span_processor, OfflineJudgmentSpanProcessor)
        return self._span_processor
