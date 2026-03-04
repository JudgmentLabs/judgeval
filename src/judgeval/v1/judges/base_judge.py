from __future__ import annotations

from typing import Any, Dict, Optional

from judgeval.constants import APIScorerType
from judgeval.v1.internal.api.models import ScorerConfig


class BaseJudge:
    __slots__ = (
        "_name",
        "_threshold",
        "_model",
        "_prompt",
        "_options",
        "_description",
        "_project_id",
    )

    def __init__(
        self,
        *,
        name: str,
        prompt: str,
        threshold: float = 0.5,
        model: Optional[str] = None,
        options: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self._name = name
        self._prompt = prompt
        self._threshold = threshold
        self._model = model
        self._options = options.copy() if options else None
        self._description = description
        self._project_id = project_id

    def get_name(self) -> str:
        return self._name

    def get_scorer_config(self) -> ScorerConfig:
        kwargs: Dict[str, Any] = {"prompt": self._prompt}
        if self._options:
            kwargs["options"] = self._options
        if self._model:
            kwargs["model"] = self._model
        if self._description:
            kwargs["description"] = self._description

        return ScorerConfig(
            score_type=APIScorerType.PROMPT_SCORER,
            threshold=self._threshold,
            name=self._name,
            kwargs=kwargs,
            result_type="numeric",
        )
