from typing import Type, Optional, Any
from functools import wraps

# Import implementations
from judgeval.scorers.judgeval_scorers.api_scorers import (
    ToolCorrectnessScorer as APIToolCorrectnessScorer,
    JSONCorrectnessScorer as APIJSONCorrectnessScorer,
    SummarizationScorer as APISummarizationScorer,
    HallucinationScorer as APIHallucinationScorer,
    FaithfulnessScorer as APIFaithfulnessScorer,
    ContextualRelevancyScorer as APIContextualRelevancyScorer,
    ContextualPrecisionScorer as APIContextualPrecisionScorer,
    ContextualRecallScorer as APIContextualRecallScorer,
    AnswerRelevancyScorer as APIAnswerRelevancyScorer,
    AnswerCorrectnessScorer as APIAnswerCorrectnessScorer,  
    ComparisonScorer as APIComparisonScorer,
)

from judgeval.scorers.judgeval_scorers.local_implementations import (
    AnswerRelevancyScorer as LocalAnswerRelevancyScorer,
    ContextualPrecisionScorer as LocalContextualPrecisionScorer,
    ContextualRecallScorer as LocalContextualRecallScorer,
    ContextualRelevancyScorer as LocalContextualRelevancyScorer,
    FaithfulnessScorer as LocalFaithfulnessScorer,
    JsonCorrectnessScorer as LocalJsonCorrectnessScorer,
    ToolCorrectnessScorer as LocalToolCorrectnessScorer,
    HallucinationScorer as LocalHallucinationScorer,
    SummarizationScorer as LocalSummarizationScorer,
    AnswerCorrectnessScorer as LocalAnswerCorrectnessScorer,
    ComparisonScorer as LocalComparisonScorer
)

from judgeval.scorers.judgeval_scorers.classifiers import Text2SQLScorer


class ScorerWrapper:
    """
    Wrapper class that can dynamically load either API or local implementation of a scorer.
    """
    def __init__(self, api_implementation: Type, local_implementation: Optional[Type] = None):
        self.api_implementation = api_implementation
        self.local_implementation = local_implementation
        self._instance = None
        self._init_args = None
        self._init_kwargs = None
        
    def __call__(self, *args, **kwargs):
        """Store initialization arguments for later use when implementation is loaded"""
        self._init_args = args
        self._init_kwargs = kwargs
        return self
    
    def load_implementation(self, use_judgment: bool = True) -> Any:
        """
        Load the appropriate implementation based on the use_judgment flag.
        
        Args:
            use_judgment (bool): If True, use API implementation. If False, use local implementation.
        
        Returns:
            Instance of the appropriate implementation
        
        Raises:
            ValueError: If local implementation is requested but not available
        """
        if self._instance is not None:
            return self._instance
            
        if use_judgment:
            implementation = self.api_implementation
        else:
            if self.local_implementation is None:
                raise ValueError("No local implementation available for this scorer")
            implementation = self.local_implementation
            
        args = self._init_args or ()
        kwargs = self._init_kwargs or {}
        self._instance = implementation(*args, **kwargs)
        return self._instance
    
    def __getattr__(self, name):
        """Defer all attribute access to the loaded implementation"""
        if self._instance is None:
            raise RuntimeError("Implementation not loaded. Call load_implementation() first")
        return getattr(self._instance, name)

# Create wrapped versions of all scorers

AnswerCorrectnessScorer = ScorerWrapper(
    api_implementation=APIAnswerCorrectnessScorer,
    local_implementation=LocalAnswerCorrectnessScorer
)

AnswerRelevancyScorer = ScorerWrapper(
    api_implementation=APIAnswerRelevancyScorer,
    local_implementation=LocalAnswerRelevancyScorer
)

ToolCorrectnessScorer = ScorerWrapper(
    api_implementation=APIToolCorrectnessScorer,
    local_implementation=LocalToolCorrectnessScorer
)

JSONCorrectnessScorer = ScorerWrapper(
    api_implementation=APIJSONCorrectnessScorer,
    local_implementation=LocalJsonCorrectnessScorer
)

SummarizationScorer = ScorerWrapper(
    api_implementation=APISummarizationScorer,
    local_implementation=LocalSummarizationScorer
)

HallucinationScorer = ScorerWrapper(
    api_implementation=APIHallucinationScorer,
    local_implementation=LocalHallucinationScorer
)

FaithfulnessScorer = ScorerWrapper(
    api_implementation=APIFaithfulnessScorer,
    local_implementation=LocalFaithfulnessScorer
)

ContextualRelevancyScorer = ScorerWrapper(
    api_implementation=APIContextualRelevancyScorer,
    local_implementation=LocalContextualRelevancyScorer
)

ContextualPrecisionScorer = ScorerWrapper(
    api_implementation=APIContextualPrecisionScorer,
    local_implementation=LocalContextualPrecisionScorer
)

ContextualRecallScorer = ScorerWrapper(
    api_implementation=APIContextualRecallScorer,
    local_implementation=LocalContextualRecallScorer
)

def ComparisonScorer(threshold: float, criteria: str, description: str):
    return ScorerWrapper(
        api_implementation=APIComparisonScorer,
        local_implementation=LocalComparisonScorer
    )(threshold=threshold, criteria=criteria, description=description)

__all__ = [
    "ToolCorrectnessScorer",
    "JSONCorrectnessScorer",
    "SummarizationScorer",
    "HallucinationScorer",
    "FaithfulnessScorer",
    "ContextualRelevancyScorer",
    "ContextualPrecisionScorer",
    "ContextualRecallScorer",
    "AnswerRelevancyScorer",
    "Text2SQLScorer",
    "ComparisonScorer",
]
