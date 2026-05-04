from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler

from judgeval.data.example import Example
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.logger import judgeval_logger
from judgeval.utils.serialize import safe_serialize
from judgeval.trace.base_tracer import BaseTracer
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.trace.processors.offline_judgment_span_processor import (
    OfflineJudgmentSpanProcessor,
)
from judgeval.trace.id_generator import IsolatedRandomIdGenerator
from judgeval.internal.api import JudgmentSyncClient
from judgeval.utils import resolve_project_id
from judgeval.version import get_version
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME


OFFLINE_TRACES_PATH = "otel/v1/offline-traces"


class OfflineTracer(BaseTracer):
    """A Tracer for offline / experiment-style runs.

    ``OfflineTracer`` behaves like :class:`judgeval.trace.Tracer` for span
    creation and export, with two important differences:

    * Spans are pushed to the project's *offline* OTLP endpoint and stored
      in the ``offline_otel_traces`` ClickHouse table. They do **not**
      appear on the live monitoring page.
    * Each completed root span produces a new
      :class:`judgeval.data.example.Example` that is appended to the
      caller-supplied ``dataset`` list. The example carries the
      ``offline_trace_id`` of the offline trace plus any static ``example_fields``
      configured at init time (e.g. golden inputs / outputs).

    Note that ``dataset`` is not a Judgment ``Dataset`` -- it is a plain
    Python list of :class:`Example` objects. Callers can later upload it
    using ``client.datasets`` or pass it directly to an experiment.

    Examples:
        ```python
        from judgeval import OfflineTracer

        results: list[Example] = []

        for dataset_item in eval_dataset:
            tracer = OfflineTracer.init(
                project_name="production",
                dataset=results,
                example_fields={
                    "input": dataset_item.input,
                    "golden_output": dataset_item.golden_output,
                },
            )
            my_agent(dataset_item.input)
            tracer.shutdown()
        ```
    """

    __slots__ = (
        "__weakref__",
        "_span_exporter",
        "_span_processor",
        "_enable_monitoring",
        "_dataset",
        "_example_fields",
    )

    TRACER_NAME = JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME

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
            client=client,
        )
        self._enable_monitoring = enable_monitoring
        self._span_exporter: Optional[JudgmentSpanExporter] = None
        self._span_processor: Optional[OfflineJudgmentSpanProcessor] = None
        self._dataset = dataset
        self._example_fields: Dict[str, Any] = dict(example_fields or {})

    @classmethod
    def init(
        cls,
        project_name: Optional[str] = None,
        dataset: Optional[List[Example]] = None,
        example_fields: Optional[Dict[str, Any]] = None,
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
    ) -> OfflineTracer:
        """Create and activate a new ``OfflineTracer``.

        Args:
            project_name: Your Judgment project name. Required for span export.
            dataset: A list that newly created :class:`Example` objects will
                be appended to. Required. This is a plain Python list of
                Examples, not a Judgment ``Dataset``.
            example_fields: Static key/value pairs added to every example
                created by this tracer (e.g. golden inputs / outputs).
            api_key: Judgment API key. Defaults to ``JUDGMENT_API_KEY`` env var.
            organization_id: Organization ID. Defaults to ``JUDGMENT_ORG_ID``
                env var.
            api_url: API endpoint URL. Defaults to ``JUDGMENT_API_URL`` env
                var.
            environment: Label for this deployment.
            set_active: If True, sets this as the global tracer.
            serializer: Custom serializer for span inputs/outputs.
            resource_attributes: Extra OpenTelemetry resource attributes.
            sampler: Custom OpenTelemetry sampler.
            span_limits: OpenTelemetry span limits.
            span_processors: Additional span processors.

        Returns:
            A configured and active ``OfflineTracer`` instance.
        """
        if dataset is None:
            raise ValueError(
                "OfflineTracer.init requires a `dataset` argument "
                "(a list of Example objects to append created examples to)."
            )

        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID
        api_url = api_url or JUDGMENT_API_URL

        enable_monitoring = True

        if not project_name:
            judgeval_logger.warning(
                "project_name not provided. OfflineTracer will not export spans."
            )
            enable_monitoring = False

        if not api_key:
            judgeval_logger.warning(
                "api_key not provided. OfflineTracer will not export spans."
            )
            enable_monitoring = False

        if not organization_id:
            judgeval_logger.warning(
                "organization_id not provided. OfflineTracer will not export spans."
            )
            enable_monitoring = False

        if not api_url:
            judgeval_logger.warning(
                "api_url not provided. OfflineTracer will not export spans."
            )
            enable_monitoring = False

        client: Optional[JudgmentSyncClient] = None
        project_id: Optional[str] = None
        if (
            enable_monitoring
            and project_name
            and api_key
            and organization_id
            and api_url
        ):
            client = JudgmentSyncClient(api_url, api_key, organization_id)
            project_id = resolve_project_id(client, project_name)
            if not project_id:
                judgeval_logger.warning(
                    f"Project '{project_name}' not found. OfflineTracer will not export spans."
                )
                enable_monitoring = False

        resource_attrs = {
            "service.name": project_name or "unknown",
            "telemetry.sdk.name": cls.TRACER_NAME,
            "telemetry.sdk.version": get_version(),
            "judgment.offline": "true",
        }
        if environment:
            resource_attrs["deployment.environment"] = environment
        if resource_attributes:
            resource_attrs.update(resource_attributes)

        resource = Resource.create(resource_attrs)
        tracer_provider = TracerProvider(
            resource=resource,
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
            enable_monitoring=enable_monitoring,
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

    def set_active(self) -> bool:
        """Set this tracer as the globally active tracer."""
        proxy = JudgmentTracerProvider.get_instance()
        return proxy.set_active(self)

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

    def get_span_processor(self) -> OfflineJudgmentSpanProcessor:
        """Return the offline span processor for this tracer.

        The processor is responsible for exporting spans *and* appending
        new :class:`Example` objects to the configured ``dataset`` list
        whenever a root span ends. Example creation runs even when
        monitoring is disabled (the spans simply route to a no-op
        exporter in that case) so that ``dataset`` is always populated.
        """
        if self._span_processor is not None:
            return self._span_processor

        self._span_processor = OfflineJudgmentSpanProcessor(
            self,
            self.get_span_exporter(),
            dataset=self._dataset,
            example_fields=self._example_fields,
        )
        return self._span_processor
