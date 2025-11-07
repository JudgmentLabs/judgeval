from __future__ import annotations

from typing import Any, Dict, Optional

from judgeval.v1.internal.api.api_types import ScorerConfig
from judgeval.v1.scorers.base_scorer import BaseScorer


class PromptScorer(BaseScorer):
    __slots__ = (
        "_name",
        "_prompt",
        "_threshold",
        "_options",
        "_model",
        "_description",
        "_judgment_api_key",
        "_organization_id",
        "_is_trace",
    )

    def __init__(
        self,
        name: str,
        prompt: str,
        threshold: float = 0.5,
        options: Optional[Dict[str, float]] = None,
        model: Optional[str] = None,
        description: Optional[str] = None,
        judgment_api_key: str = "",
        organization_id: str = "",
        is_trace: bool = False,
    ):
        self._name = name
        self._prompt = prompt
        self._threshold = threshold
        self._options = options.copy() if options else None
        self._model = model
        self._description = description
        self._judgment_api_key = judgment_api_key
        self._organization_id = organization_id
        self._is_trace = is_trace

    def get_name(self) -> str:
        return self._name

    def get_prompt(self) -> str:
        return self._prompt

    def get_threshold(self) -> float:
        return self._threshold

    def get_options(self) -> Optional[Dict[str, float]]:
        return self._options.copy() if self._options else None

    def get_model(self) -> Optional[str]:
        return self._model

    def get_description(self) -> Optional[str]:
        return self._description

    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def set_model(self, model: str) -> None:
        self._model = model

    def set_options(self, options: Dict[str, float]) -> None:
        self._options = options.copy()

    def set_description(self, description: str) -> None:
        self._description = description

    def append_to_prompt(self, addition: str) -> None:
        self._prompt = self._prompt + addition

    def get_scorer_config(self) -> ScorerConfig:
        score_type = "trace_prompt_scorer" if self._is_trace else "prompt_scorer"
        kwargs: Dict[str, Any] = {"prompt": self._prompt}

        if self._options:
            kwargs["options"] = self._options
        if self._model:
            kwargs["model"] = self._model
        if self._description:
            kwargs["description"] = self._description

        return ScorerConfig(
            score_type=score_type,
            threshold=self._threshold,
            name=self._name,
            kwargs=kwargs,
        )
