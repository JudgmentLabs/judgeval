from __future__ import annotations

from typing import Sequence

import requests

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter as BaseSpanExporter,
    SpanExportResult,
)

from judgeval.logger import judgeval_logger


class JudgmentSpanExporter(BaseSpanExporter):
    """Exports completed spans to the Judgment platform over OTLP/HTTP.

    This is the default exporter created by ``Tracer.init()``. It wraps the
    OpenTelemetry ``OTLPSpanExporter`` and injects Judgment authentication
    headers automatically.

    You rarely need to instantiate this directly -- ``Tracer.init()`` wires
    it up for you. Use it when building a custom ``TracerProvider``.

    Args:
        endpoint: The Judgment OTLP ingest URL.
        api_key: Judgment API key for authentication.
        organization_id: Your Judgment organization ID.
        project_id: The resolved Judgment project ID.

    Examples:
        ```python
        from judgeval.trace import JudgmentSpanExporter

        exporter = JudgmentSpanExporter(
            endpoint="https://api.judgmentlabs.ai/otel/v1/traces",
            api_key="jdg_...",
            organization_id="org_123",
            project_id="proj_456",
        )
        ```
    """

    __slots__ = ("_delegate", "_billing_error_reported")

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        organization_id: str,
        project_id: str,
    ):
        self._billing_error_reported = False
        session = requests.Session()
        session.hooks["response"].append(self._report_billing_error)
        self._delegate = OTLPSpanExporter(
            endpoint=endpoint,
            session=session,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": organization_id,
                "X-Project-Id": project_id,
            },
        )

    def _report_billing_error(self, response: requests.Response, **kwargs) -> None:
        # Some OTel versions log only the HTTP reason, omitting the response body.
        try:
            if 200 <= response.status_code < 300:
                self._billing_error_reported = False
                return
            if response.status_code != 402 or self._billing_error_reported:
                return
            message = (
                "Payment required. Review your organization's billing and usage limits."
            )
            try:
                body = response.json()
                detail = body.get("message") if isinstance(body, dict) else None
                if isinstance(detail, str) and detail.strip():
                    message = detail.strip()
            except ValueError:
                pass
            judgeval_logger.error("Trace export rejected (HTTP 402): %s", message)
            self._billing_error_reported = True
        except Exception:
            # Diagnostics must not interrupt OTel's export or retry handling.
            pass

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Send a batch of spans to Judgment."""
        judgeval_logger.info(f"Exported {len(spans)} spans")
        return self._delegate.export(spans)

    def shutdown(self) -> None:
        """Shut down the underlying OTLP exporter and release resources."""
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush any queued spans within the given timeout."""
        return self._delegate.force_flush(timeout_millis)
