from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data.judgment_types import Trace as JudgmentTrace
from typing import List, Optional
from abc import abstractmethod


class AgentScorer(BaseScorer):
    @abstractmethod
    async def a_score_trace(
        self, trace: JudgmentTrace, tools: Optional[List] = None, *args, **kwargs
    ) -> float:
        """
        Asynchronously measures the score on a trace
        """
        raise NotImplementedError(
            "You must implement the `a_score_trace` method in your custom scorer"
        )
