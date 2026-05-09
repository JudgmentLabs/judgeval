from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from judgeval.internal.api import JudgmentSyncClient
from judgeval.utils import resolve_project_id
from judgeval.utils.serialize import safe_serialize
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.logger import judgeval_logger

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanLimits, SpanProcessor
    from opentelemetry.sdk.trace.sampling import Sampler

    from judgeval.data.example import Example
    from judgeval.trace.offline_tracer import OfflineTracer


class Judgeval:
    """The main entry point for interacting with the Judgment platform.

    `Judgeval` connects to your Judgment project and gives you access to
    **evaluations**, **datasets**, and **prompt versioning** through
    convenient properties.

    Credentials are resolved in order: explicit arguments first, then
    environment variables `JUDGMENT_API_KEY`, `JUDGMENT_ORG_ID`, and
    `JUDGMENT_API_URL`.

    Args:
        project_name: The name of your Judgment project.
        api_key: Your Judgment API key. If omitted, reads from the
            `JUDGMENT_API_KEY` environment variable.
        organization_id: Your organization ID. If omitted, reads from the
            `JUDGMENT_ORG_ID` environment variable.
        api_url: Override the API endpoint URL. If omitted, reads from the
            `JUDGMENT_API_URL` environment variable.

    Raises:
        ValueError: If any required credential or `project_name` is missing.

    Examples:
        Minimal setup (credentials from environment variables):

        ```python
        from judgeval import Judgeval

        client = Judgeval(project_name="search-assistant")
        ```

        Explicit credentials:

        ```python
        client = Judgeval(
            project_name="search-assistant",
            api_key="jdg_...",
            organization_id="org_...",
        )
        ```

        Once initialized, use the `evaluation`, `datasets`, and `prompts`
        properties:

        ```python
        eval_runner = client.evaluation.create()
        dataset = client.datasets.get(name="golden-set")
        prompt = client.prompts.get(name="system-prompt", tag="production")
        ```
    """

    __slots__ = (
        "_api_key",
        "_organization_id",
        "_api_url",
        "_internal_client",
        "_project_name",
        "_project_id",
    )

    def __init__(
        self,
        project_name: str,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID
        api_url = api_url or JUDGMENT_API_URL

        if not api_key:
            raise ValueError("api_key is required")
        if not organization_id:
            raise ValueError("organization_id is required")
        if not api_url:
            raise ValueError("api_url is required")
        if not project_name:
            raise ValueError("project_name is required")

        self._api_key = api_key
        self._organization_id = organization_id
        self._api_url = api_url
        self._project_name = project_name

        self._internal_client = JudgmentSyncClient(
            self._api_url,
            self._api_key,
            self._organization_id,
        )

        self._project_id: Optional[str] = resolve_project_id(
            self._internal_client, project_name
        )
        if not self._project_id:
            judgeval_logger.warning(
                f"Project '{project_name}' not found. "
                "Some operations requiring project_id will be skipped."
            )

    def offline_tracer(
        self,
        *,
        dataset: List["Example"],
        example_fields: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        set_active: bool = True,
        serializer: Callable[[Any], str] = safe_serialize,
        resource_attributes: Optional[Dict[str, Any]] = None,
        sampler: Optional["Sampler"] = None,
        span_limits: Optional["SpanLimits"] = None,
        span_processors: Optional[Sequence["SpanProcessor"]] = None,
    ) -> "OfflineTracer":
        """Create and activate an ``OfflineTracer`` for this project.

        Reuses the credentials supplied to this ``Judgeval`` instance. Each
        completed root span appends an ``Example`` to ``dataset``, carrying
        the offline trace id and the static ``example_fields``.

        Args:
            dataset: Caller-owned list. Each completed root span appends a
                new ``Example`` carrying the ``offline_trace_id`` of the
                trace and the static ``example_fields``.
            example_fields: Static fields copied onto every emitted example
                (e.g. ``{"input": ..., "golden_output": ...}``).
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
            tracer = client.offline_tracer(
                dataset=results,
                example_fields={
                    "input": item.input,
                    "golden_output": item.golden_output,
                },
            )
            ```
        """
        from judgeval.trace.offline_tracer import OfflineTracer

        return OfflineTracer.create(
            project_name=self._project_name,
            api_key=self._api_key,
            organization_id=self._organization_id,
            api_url=self._api_url,
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

    @property
    def evaluation(self):
        """Access evaluations for scoring examples with hosted or custom judges.

        Returns:
            EvaluationFactory: Use `.create()` to get an `Evaluation` you
                can call `.run()` on.

        Examples:
            ```python
            eval_runner = client.evaluation.create()
            results = eval_runner.run(
                examples=examples,
                scorers=["faithfulness", "answer_relevancy"],
                eval_run_name="nightly-eval",
            )
            ```
        """
        from judgeval.evaluation.evaluation_factory import EvaluationFactory

        return EvaluationFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )

    @property
    def datasets(self):
        """Manage datasets of evaluation examples.

        Returns:
            DatasetFactory: Use `.create()`, `.get()`, or `.list()` to work
                with datasets.

        Examples:
            ```python
            dataset = client.datasets.create(
                name="golden-set",
                examples=[
                    Example.create(input="What is 2+2?", expected_output="4"),
                ],
            )
            ```
        """
        from judgeval.datasets.dataset_factory import DatasetFactory

        return DatasetFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )

    @property
    def prompts(self):
        """Manage versioned prompt templates with tagging support.

        Returns:
            PromptFactory: Use `.create()`, `.get()`, `.tag()`, or `.list()`
                to work with prompts.

        Examples:
            ```python
            prompt = client.prompts.create(
                name="system-prompt",
                prompt="You are a helpful assistant for {{product}}.",
                tags=["v1"],
            )
            compiled = prompt.compile(product="Acme Search")
            ```
        """
        from judgeval.prompts.prompt_factory import PromptFactory

        return PromptFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )
