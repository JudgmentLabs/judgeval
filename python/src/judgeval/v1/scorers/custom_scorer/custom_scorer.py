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
        "_class_name",
        "_server_hosted",
    )

    def __init__(
        self,
        name: str,
        class_name: str = "",
        server_hosted: bool = True,
    ):
        self._name = name
        self._class_name = class_name or name
        self._server_hosted = server_hosted

    def get_name(self) -> str:
        return self._name

    def get_class_name(self) -> str:
        return self._class_name

    def is_server_hosted(self) -> bool:
        return self._server_hosted

    def get_scorer_config(self) -> ScorerConfig:
        raise NotImplementedError("CustomScorer does not use get_scorer_config")

    def to_dict(self) -> BaseScorerDict:
        return {
            "score_type": APIScorerType.CUSTOM.value,
            "name": self._name,
            "class_name": self._class_name,
            "server_hosted": self._server_hosted,
        }
