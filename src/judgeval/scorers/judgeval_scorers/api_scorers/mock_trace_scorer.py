from judgeval.scorers.trace_api_scorer import TraceAPIScorerConfig
from judgeval.constants import APIScorerType


class MockTraceScorer(TraceAPIScorerConfig):
    score_type: APIScorerType = APIScorerType.MOCK_TRACE_SCORER
    threshold: float = 0.5
