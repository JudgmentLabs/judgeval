from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.sdk.trace import SpanProcessor

from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.logger import judgeval_logger
from judgeval.utils.serialize import safe_serialize
from judgeval.trace.base_tracer import BaseTracer
from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider
from judgeval.trace.exporters.judgment_span_exporter import JudgmentSpanExporter
from judgeval.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.trace.processors.judgment_span_processor import JudgmentSpanProcessor
from judgeval.trace.processors.noop_judgment_span_processor import (
    NoOpJudgmentSpanProcessor,
)
from judgeval.trace.id_generator import IsolatedRandomIdGenerator
from judgeval.internal.api import JudgmentSyncClient
from judgeval.utils import resolve_project_id
from judgeval.version import get_version
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME


class Tracer(BaseTracer):
    """Capture execution traces and LLM performance metrics for your application.

    `Tracer` is the primary way to add observability to AI agents and LLM
    pipelines. It records spans (units of work), automatically captures
    inputs/outputs, and exports everything to the Judgment dashboard.

    **Getting started:**

    1. Call `Tracer.init()` to create and activate a tracer.
    2. Decorate your functions with `@Tracer.observe()` to trace them.
    3. Optionally wrap LLM clients with `Tracer.wrap()` for automatic
       token/cost tracking.

    Args:
        project_name: Your Judgment project name.
        project_id: Resolved project ID (set automatically by `init`).
        api_key: Judgment API key.
        organization_id: Organization ID.
        api_url: Judgment API endpoint URL.
        environment: Deployment environment label (e.g. `"production"`).
        serializer: Function used to serialize span inputs/outputs.
        tracer_provider: The OpenTelemetry TracerProvider backing this tracer.
        enable_monitoring: Whether span export is enabled.
        client: Internal API client for server-side operations.

    Examples:
        Basic setup and usage:

        ```python
        from judgeval import Tracer

        tracer = Tracer.init(project_name="search-assistant")

        @Tracer.observe(span_type="tool")
        def search(query: str) -> str:
            return vector_db.search(query)

        @Tracer.observe(span_type="agent")
        async def answer(question: str) -> str:
            context = search(question)
            return await llm.generate(question, context)
        ```

        Wrap an LLM client for automatic instrumentation:

        ```python
        from openai import OpenAI

        openai = Tracer.wrap(OpenAI())
        ```
    """

    __slots__ = (
        "__weakref__",
        "_span_exporter",
        "_span_processor",
        "_enable_monitoring",
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
        self._span_processor: Optional[JudgmentSpanProcessor] = None

    @classmethod
    def init(
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
    ) -> Tracer:
        """Create and activate a new Tracer.

        This is the recommended way to initialize tracing. Credentials are
        read from environment variables (`JUDGMENT_API_KEY`, `JUDGMENT_ORG_ID`,
        `JUDGMENT_API_URL`) when not passed explicitly. If credentials are
        missing, the tracer still works but spans won't be exported.

        Args:
            project_name: Your Judgment project name. Required for span export.
            api_key: Judgment API key. Defaults to `JUDGMENT_API_KEY` env var.
            organization_id: Organization ID. Defaults to `JUDGMENT_ORG_ID` env var.
            api_url: API endpoint URL. Defaults to `JUDGMENT_API_URL` env var.
            environment: Label for this deployment (e.g. `"staging"`,
                `"production"`). Shows up in the Judgment dashboard.
            set_active: If True (default), sets this as the global tracer so
                `@Tracer.observe()` and other static methods use it.
            serializer: Custom serializer for span inputs/outputs.
            resource_attributes: Extra OpenTelemetry resource attributes.
            sampler: Custom OpenTelemetry sampler.
            span_limits: OpenTelemetry span limits.
            span_processors: Additional span processors appended after the
                default Judgment processor.

        Returns:
            A configured and active `Tracer` instance.

        Examples:
            ```python
            tracer = Tracer.init(
                project_name="search-assistant",
                environment="production",
            )
            ```
        """
        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID
        api_url = api_url or JUDGMENT_API_URL

        enable_monitoring = True

        if not project_name:
            judgeval_logger.warning(
                "project_name not provided. Tracer will not export spans."
            )
            enable_monitoring = False

        if not api_key:
            judgeval_logger.warning(
                "api_key not provided. Tracer will not export spans."
            )
            enable_monitoring = False

        if not organization_id:
            judgeval_logger.warning(
                "organization_id not provided. Tracer will not export spans."
            )
            enable_monitoring = False

        if not api_url:
            judgeval_logger.warning(
                "api_url not provided. Tracer will not export spans."
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
                    f"Project '{project_name}' not found. Tracer will not export spans."
                )
                enable_monitoring = False

        resource_attrs = {
            "service.name": project_name or "unknown",
            "telemetry.sdk.name": cls.TRACER_NAME,
            "telemetry.sdk.version": get_version(),
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
        )

        if enable_monitoring:
            tracer_provider.add_span_processor(tracer.get_span_processor())

        for processor in span_processors or []:
            tracer_provider.add_span_processor(processor)

        proxy = JudgmentTracerProvider.get_instance()
        proxy.register(tracer)

        if set_active:
            tracer.set_active()

        return tracer

    def set_active(self) -> bool:
        """Set this tracer as the globally active tracer.

        Returns:
            True if the tracer was successfully activated.
        """
        proxy = JudgmentTracerProvider.get_instance()
        return proxy.set_active(self)

    def get_span_exporter(self) -> JudgmentSpanExporter:
        """Return the span exporter for this tracer.

        Returns a no-op exporter when monitoring is disabled.

        Returns:
            The ``JudgmentSpanExporter`` (or no-op variant) for this tracer.
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
            endpoint = (
                self.api_url + "otel/v1/traces"
                if self.api_url.endswith("/")
                else self.api_url + "/otel/v1/traces"
            )
            self._span_exporter = JudgmentSpanExporter(
                endpoint=endpoint,
                api_key=self.api_key,
                organization_id=self.organization_id,
                project_id=self.project_id,
            )
        return self._span_exporter

    def get_span_processor(self) -> JudgmentSpanProcessor:
        """Return the span processor for this tracer.

        Returns a no-op processor when monitoring is disabled.

        Returns:
            The ``JudgmentSpanProcessor`` (or no-op variant) for this tracer.
        """
        if self._span_processor is not None:
            return self._span_processor

        if not self._enable_monitoring:
            self._span_processor = NoOpJudgmentSpanProcessor()
        else:
            self._span_processor = JudgmentSpanProcessor(
                self,
                self.get_span_exporter(),
            )
        return self._span_processor
