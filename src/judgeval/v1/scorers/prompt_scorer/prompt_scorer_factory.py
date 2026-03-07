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
    __slots__ = ("_client", "_project_id")
    _cache: Dict[Tuple[str, str, str, str], APIPromptScorer] = {}

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
    ):
        self._client = client
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
        )
        cached = self._cache.get(cache_key)

        if cached is None:
            try:
                response: FetchPromptScorersResponse = (
                    self._client.get_projects_scorers(
                        project_id=project_id,
                        names=name,
                    )
                )
                scorers = response.get("scorers", [])

                if not scorers:
                    raise JudgmentAPIError(
                        404, f"Failed to fetch prompt scorer '{name}': not found", None
                    )

                self._cache[cache_key] = scorers[0]
                cached = scorers[0]
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
            project_id=project_id,
        )
