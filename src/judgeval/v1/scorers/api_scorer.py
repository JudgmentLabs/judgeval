from __future__ import annotations

from typing import Any, Dict, List, Optional

from judgeval.v1.internal.api.api_types import ScorerConfig
from judgeval.v1.scorers.base_scorer import BaseScorer


class APIScorer(BaseScorer):
    __slots__ = (
        "_score_type",
        "_required_params",
        "_threshold",
        "_name",
        "_model",
        "_additional_properties",
    )

    def __init__(
        self,
        score_type: str,
        required_params: Optional[List[str]] = None,
        threshold: float = 0.5,
        name: Optional[str] = None,
        model: Optional[str] = None,
        **additional_properties: Any,
    ):
        self._score_type = score_type
        self._required_params = required_params or []
        self._threshold = threshold
        self._name = name or score_type
        self._model = model
        self._additional_properties = additional_properties

    def get_name(self) -> str:
        return self._name

    def get_score_type(self) -> str:
        return self._score_type

    def get_threshold(self) -> float:
        return self._threshold

    def get_model(self) -> Optional[str]:
        return self._model

    def get_required_params(self) -> List[str]:
        return self._required_params.copy()

    def set_threshold(self, threshold: float) -> None:
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, got: {threshold}")
        self._threshold = threshold

    def set_name(self, name: str) -> None:
        self._name = name

    def set_model(self, model: str) -> None:
        self._model = model

    def get_scorer_config(self) -> ScorerConfig:
        kwargs: Dict[str, Any] = dict(self._additional_properties)
        if self._model:
            kwargs["model"] = self._model

        return ScorerConfig(
            score_type=self._score_type,
            threshold=self._threshold,
            name=self._name,
            required_params=self._required_params,
            kwargs=kwargs,
        )
