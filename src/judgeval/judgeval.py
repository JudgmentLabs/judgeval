from __future__ import annotations
from typing import Optional

from judgeval.internal.api import JudgmentSyncClient
from judgeval.utils import resolve_project_id
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.logger import judgeval_logger


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
