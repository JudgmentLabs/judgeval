from __future__ import annotations

from typing import Optional

from functools import cached_property
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID


class JudgmentClient:
    def __init__(
        self,
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

        self._api_key = api_key
        self._organization_id = organization_id
        self._api_url = api_url

        self._internal_client = JudgmentSyncClient(
            self._api_url,
            self._api_key,
            self._organization_id,
        )

    @cached_property
    def Tracer(self):
        from judgeval.v1.tracer.tracer_factory import TracerFactory

        return TracerFactory(
            client=self._internal_client,
        )

    @cached_property
    def Scorers(self):
        from judgeval.v1.scorers.scorers_factory import ScorersFactory

        return ScorersFactory(
            client=self._internal_client,
        )

    @cached_property
    def Evaluation(self):
        from judgeval.v1.evaluation.evaluation_factory import EvaluationFactory

        return EvaluationFactory(
            client=self._internal_client,
        )

    @cached_property
    def Trainers(self):
        from judgeval.v1.trainers.trainers_factory import TrainersFactory

        return TrainersFactory(
            client=self._internal_client,
        )

    @cached_property
    def Datasets(self):
        from judgeval.v1.datasets.dataset_factory import DatasetFactory

        return DatasetFactory(
            client=self._internal_client,
        )


__all__ = ["JudgmentClient"]
