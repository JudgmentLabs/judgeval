from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data import Trace
from typing import List, Optional
from abc import abstractmethod

from judgeval.common.logger import warning, error


class AgentScorer(BaseScorer):
    @abstractmethod
    async def a_score_trace(
        self, trace: Trace, tools: Optional[List] = None, *args, **kwargs
    ) -> float:
        """
        Asynchronously measures the score on a trace
        """
        warning("Attempting to call unimplemented a_score_trace method")
        error("a_score_trace method not implemented")
        raise NotImplementedError(
            "You must implement the `a_score_trace` method in your custom scorer"
        )
