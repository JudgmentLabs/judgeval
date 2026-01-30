from __future__ import annotations

from typing import TYPE_CHECKING
from judgeval.constants import APIScorerType

if TYPE_CHECKING:
    from judgeval.v1.internal.api.api_types import (
        BaseScorer as BaseScorerDict,
        ScorerConfig,
    )

from judgeval.v1.scorers.base_scorer import BaseScorer


class CustomScorer(BaseScorer):
    __slots__ = (
        "_name",
        "_project_id",
    )

    def __init__(
        self,
        name: str,
        class_name: str = "",
        server_hosted: bool = True,
        *,
        project_id: str,
    ):
        self._name = name
        self._project_id = project_id

    def get_name(self) -> str:
        return self._name

    def get_scorer_config(self) -> ScorerConfig:
        raise NotImplementedError("CustomScorer does not use get_scorer_config")

    def to_dict(self) -> BaseScorerDict:
        return {"score_type": APIScorerType.CUSTOM.value, "name": self._name}
