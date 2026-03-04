from judgeval.v1.scorers.base_custom_scorer.base_custom_scorer import BaseCustomScorer
from judgeval.v1.internal.api.models import TraceSpan
from typing import List

TraceScorer = BaseCustomScorer[List[TraceSpan]]
