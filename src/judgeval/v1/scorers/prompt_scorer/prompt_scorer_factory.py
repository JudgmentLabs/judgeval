from __future__ import annotations

from typing import Dict, Tuple

from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.internal.api.api_types import (
    FetchPromptScorersRequest,
    FetchPromptScorersResponse,
    PromptScorer as APIPromptScorer,
)
from judgeval.exceptions import JudgmentAPIError
from judgeval.v1.scorers.prompt_scorer.prompt_scorer import PromptScorer
from judgeval.logger import judgeval_logger


class PromptScorerFactory:
    __slots__ = ("_client", "_is_trace")
    _cache: Dict[Tuple[str, str, str, bool], APIPromptScorer] = {}

    def __init__(
        self,
        client: JudgmentSyncClient,
        is_trace: bool,
    ):
        self._client = client
        self._is_trace = is_trace

    def get(self, name: str) -> PromptScorer | None:
        cache_key = (
            name,
            self._client.organization_id,
            self._client.api_key,
            self._is_trace,
        )
        cached = self._cache.get(cache_key)

        if cached is None:
            request: FetchPromptScorersRequest = {"names": [name]}
            if self._is_trace is not None:
                request["is_trace"] = self._is_trace

            try:
                response: FetchPromptScorersResponse = self._client.fetch_scorers(
                    request
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
        )
