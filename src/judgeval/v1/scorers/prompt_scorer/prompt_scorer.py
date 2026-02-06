from __future__ import annotations

from typing import Any, Dict, Optional

from judgeval.constants import APIScorerType
from judgeval.v1.internal.api.api_types import ScorerConfig
from judgeval.v1.scorers.api_scorer import APIScorer


class PromptScorer(APIScorer):
    __slots__ = (
        "_prompt",
        "_options",
        "_description",
        "_judgment_api_key",
        "_organization_id",
        "_project_id",
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
        project_id: Optional[str] = None,
        is_trace: bool = False,
    ):
        score_type = (
            APIScorerType.TRACE_PROMPT_SCORER
            if is_trace
            else APIScorerType.PROMPT_SCORER
        )
        super().__init__(
            score_type=score_type,
            threshold=threshold,
            name=name,
            model=model,
        )
        self._prompt = prompt
        self._options = options.copy() if options else None
        self._description = description
        self._judgment_api_key = judgment_api_key
        self._organization_id = organization_id
        self._project_id = project_id
        self._is_trace = is_trace

    def get_prompt(self) -> str:
        return self._prompt

    def get_options(self) -> Optional[Dict[str, float]]:
        return self._options.copy() if self._options else None

    def get_description(self) -> Optional[str]:
        return self._description

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def set_options(self, options: Dict[str, float]) -> None:
        self._options = options.copy()

    def set_description(self, description: str) -> None:
        self._description = description

    def get_scorer_config(self) -> ScorerConfig:
        kwargs: Dict[str, Any] = {"prompt": self._prompt}

        if self._options:
            kwargs["options"] = self._options
        if self._model:
            kwargs["model"] = self._model
        if self._description:
            kwargs["description"] = self._description

        return ScorerConfig(
            score_type=self._score_type,
            threshold=self._threshold,
            name=self._name,
            kwargs=kwargs,
            result_type="numeric",
        )
