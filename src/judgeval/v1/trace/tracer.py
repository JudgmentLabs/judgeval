from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor as BaseSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter as BaseSpanExporter

from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.logger import judgeval_logger
from judgeval.utils.serialize import safe_serialize
from judgeval.v1.trace.base_tracer import BaseTracer
from judgeval.v1.trace.proxy_tracer_provider import ProxyTracerProvider
from judgeval.v1.trace.exporters.span_exporter import SpanExporter
from judgeval.v1.trace.exporters.noop_span_exporter import NoOpSpanExporter
from judgeval.v1.trace.processors.span_processor import SpanProcessor
from judgeval.v1.trace.processors.noop_span_processor import NoOpSpanProcessor
from judgeval.v1.trace.id_generator import IsolatedRandomIdGenerator
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.utils import resolve_project_id
from judgeval.version import get_version
from judgeval.constants import JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME


class Tracer(BaseTracer):
    __slots__ = (
        "_client",
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
        )
        self._enable_monitoring = enable_monitoring
        self._client = client
        self._span_exporter: Optional[BaseSpanExporter] = None
        self._span_processor: Optional[BaseSpanProcessor] = None

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
    ) -> Tracer:
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

        if set_active:
            tracer.set_active()

        return tracer

    def set_active(self) -> bool:
        proxy = ProxyTracerProvider.get_instance()
        return proxy.set_active(self)

    def get_span_exporter(self) -> BaseSpanExporter:
        if self._span_exporter is not None:
            return self._span_exporter

        if (
            not self._enable_monitoring
            or not self.project_id
            or not self.api_key
            or not self.organization_id
            or not self.api_url
        ):
            self._span_exporter = NoOpSpanExporter()
        else:
            endpoint = (
                self.api_url + "otel/v1/traces"
                if self.api_url.endswith("/")
                else self.api_url + "/otel/v1/traces"
            )
            self._span_exporter = SpanExporter(
                endpoint=endpoint,
                api_key=self.api_key,
                organization_id=self.organization_id,
                project_id=self.project_id,
            )
        return self._span_exporter

    def get_span_processor(self) -> BaseSpanProcessor:
        if self._span_processor is not None:
            return self._span_processor

        if not self._enable_monitoring:
            self._span_processor = NoOpSpanProcessor()
        else:
            self._span_processor = SpanProcessor(
                self,
                self.get_span_exporter(),
            )
        return self._span_processor

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: int = 30000) -> None:
        self._tracer_provider.shutdown()
