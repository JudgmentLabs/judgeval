from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import SpanLimits, SpanProcessor
from opentelemetry.sdk.trace.sampling import Sampler

from judgeval.data.example import Example
from judgeval.trace.offline_tracer import OfflineTracer
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.judgeval import Judgeval


class JudgevalOfflineTracer:
    """Bound factory exposed as ``Judgeval().offline_tracer``.

    Reuses the parent ``Judgeval`` client's project name and credentials
    so callers don't have to repeat them when spinning up an
    ``OfflineTracer``.
    """

    __slots__ = ("_client",)

    def __init__(self, client: "Judgeval"):
        self._client = client

    def init(
        self,
        *,
        dataset: List[Example],
        example_fields: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        set_active: bool = True,
        serializer: Callable[[Any], str] = safe_serialize,
        resource_attributes: Optional[Dict[str, Any]] = None,
        sampler: Optional[Sampler] = None,
        span_limits: Optional[SpanLimits] = None,
        span_processors: Optional[Sequence[SpanProcessor]] = None,
    ) -> OfflineTracer:
        """Create and activate an ``OfflineTracer`` for this project.

        Args:
            dataset: Caller-owned list. Each completed root span appends a
                new ``Example`` carrying the offline trace id and the
                static ``example_fields``.
            example_fields: Static fields copied onto every emitted
                example (e.g. ``{"input": ..., "golden_output": ...}``).
            environment: Deployment environment label.
            set_active: If True, register this as the active tracer.
            serializer: Custom serializer for span inputs/outputs.
            resource_attributes: Extra OTel resource attributes.
            sampler: Custom OTel sampler.
            span_limits: OTel span limits.
            span_processors: Additional span processors appended after the
                default offline processor.

        Examples:
            ```python
            client = Judgeval(project_name="default_project")
            results: list[Example] = []
            tracer = client.offline_tracer.init(
                dataset=results,
                example_fields={
                    "input": item.input,
                    "golden_output": item.golden_output,
                },
            )
            ```
        """
        return OfflineTracer.init(
            project_name=self._client._project_name,
            api_key=self._client._api_key,
            organization_id=self._client._organization_id,
            api_url=self._client._api_url,
            environment=environment,
            set_active=set_active,
            serializer=serializer,
            resource_attributes=resource_attributes,
            sampler=sampler,
            span_limits=span_limits,
            span_processors=span_processors,
            dataset=dataset,
            example_fields=example_fields,
        )
