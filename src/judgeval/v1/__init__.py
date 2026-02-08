from __future__ import annotations

from typing import Optional

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.utils import resolve_project_id
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.logger import judgeval_logger


class Judgeval:
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
    def tracer(self):
        from judgeval.v1.tracer.tracer_factory import TracerFactory

        return TracerFactory(
            client=self._internal_client,
            project_name=self._project_name,
            project_id=self._project_id,
        )

    @property
    def scorers(self):
        from judgeval.v1.scorers.scorers_factory import ScorersFactory

        return ScorersFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )

    @property
    def evaluation(self):
        from judgeval.v1.evaluation.evaluation_factory import EvaluationFactory

        return EvaluationFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )

    @property
    def trainers(self):
        from judgeval.v1.trainers.trainers_factory import TrainersFactory

        return TrainersFactory(
            client=self._internal_client,
        )

    @property
    def datasets(self):
        from judgeval.v1.datasets.dataset_factory import DatasetFactory

        return DatasetFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )

    @property
    def prompts(self):
        from judgeval.v1.prompts.prompt_factory import PromptFactory

        return PromptFactory(
            client=self._internal_client,
            project_id=self._project_id,
            project_name=self._project_name,
        )


__all__ = ["Judgeval"]
