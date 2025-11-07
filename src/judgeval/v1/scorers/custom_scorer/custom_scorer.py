from __future__ import annotations

from judgeval.v1.internal.api.api_types import ScorerConfig
from judgeval.v1.scorers.base_scorer import BaseScorer


class CustomScorer(BaseScorer):
    __slots__ = ("_name", "_class_name", "_server_hosted")

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
