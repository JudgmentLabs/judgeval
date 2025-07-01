"""
`judgeval` tool dependency scorer
"""

# Internal imports
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from typing import Optional, Dict


class ToolDependencyScorer(APIScorerConfig):
    kwargs: Optional[Dict] = None

    def __init__(self, threshold: float = 1.0, enable_param_checking: bool = True):
        super().__init__(threshold=threshold, score_type=APIScorerType.TOOL_DEPENDENCY)
        self.kwargs = {"enable_param_checking": enable_param_checking}

    @property
    def __name__(self):
        return "Tool Dependency"
