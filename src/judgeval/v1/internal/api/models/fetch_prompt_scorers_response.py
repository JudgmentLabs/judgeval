from __future__ import annotations

from typing import TypedDict, List

from .prompt_scorer import PromptScorer


class FetchPromptScorersResponse(TypedDict):
    scorers: List[PromptScorer]
