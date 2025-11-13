from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data import TraceData


class TraceScorer(BaseScorer):
    score_type: str = "Custom Trace"

    async def a_score_trace(self, trace: TraceData, *args, **kwargs) -> float:
        """
        Asynchronously measures the score on a single trace
        """
        raise NotImplementedError(
            "You must implement the `a_score_trace` method in your custom scorer"
        )
