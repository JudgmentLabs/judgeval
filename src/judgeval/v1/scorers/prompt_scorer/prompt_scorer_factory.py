from __future__ import annotations

from typing import Dict, Optional, Tuple

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    FetchPromptScorersResponse,
    PromptScorer as APIPromptScorer,
)
from judgeval.exceptions import JudgmentAPIError
from judgeval.v1.scorers.prompt_scorer.prompt_scorer import PromptScorer
from judgeval.logger import judgeval_logger
from judgeval.utils.guards import expect_project_id


class PromptScorerFactory:
    __slots__ = ("_client", "_is_trace", "_project_id")
    _cache: Dict[Tuple[str, str, str, str, bool], APIPromptScorer] = {}

    def __init__(
        self,
        client: JudgmentSyncClient,
        is_trace: bool,
        project_id: Optional[str],
    ):
        self._client = client
        self._is_trace = is_trace
        self._project_id = project_id

    def get(
        self,
        name: str,
    ) -> PromptScorer | None:
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        cache_key = (
            name,
            self._client.organization_id,
            self._client.api_key,
            project_id,
            self._is_trace,
        )
        cached = self._cache.get(cache_key)

        if cached is None:
            try:
                response: FetchPromptScorersResponse = (
                    self._client.get_projects_scorers(
                        project_id=project_id,
                        names=name,
                        is_trace=str(self._is_trace).lower()
                        if self._is_trace is not None
                        else None,
                    )
                )
                scorers = response.get("scorers", [])

                if not scorers:
                    raise JudgmentAPIError(
                        404, f"Failed to fetch prompt scorer '{name}': not found", None
                    )

                scorer = scorers[0]
                scorer_is_trace = scorer.get("is_trace", False)

                if scorer_is_trace != self._is_trace:
                    expected_type = (
                        "TracePromptScorer" if self._is_trace else "PromptScorer"
                    )
                    actual_type = (
                        "TracePromptScorer" if scorer_is_trace else "PromptScorer"
                    )
                    raise JudgmentAPIError(
                        400,
                        f"Scorer with name {name} is a {actual_type}, not a {expected_type}",
                        None,
                    )

                self._cache[cache_key] = scorer
                cached = scorer
            except JudgmentAPIError:
                judgeval_logger.error(
                    f"Failed to fetch prompt scorer '{name}' : prompt scorer '{name}' not found in the organization."
                )
                return None
            except Exception:
                judgeval_logger.error(f"Failed to fetch prompt scorer '{name}'.")
                return None

        return PromptScorer(
            name=name,
            prompt=cached.get("prompt", ""),
            threshold=cached.get("threshold", 0.5),
            options=cached.get("options"),
            model=cached.get("model"),
            description=cached.get("description"),
            is_trace=self._is_trace,
            project_id=project_id,
        )
