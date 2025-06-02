from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from typing import List, Mapping, Optional, Dict
from pydantic import model_serializer

class ClassifierScorer(APIJudgmentScorer):
    """
    In the Judgment backend, this scorer is implemented as a PromptScorer that takes 
    1. a system role that may involve the Example object
    2. options for scores on the example

    and uses a judge to execute the evaluation from the system role and classify into one of the options

    ex:
    system_role = "You are a judge that evaluates whether the response is positive or negative. The response is: {example.actual_output}"
    options = {"positive": 1, "negative": 0}
    
    Args:
        name (str): The name of the scorer
        slug (str): A unique identifier for the scorer
        conversation (List[dict]): The conversation template with placeholders (e.g., {{actual_output}})
        options (Mapping[str, float]): A mapping of classification options to their corresponding scores
        threshold (float): The threshold for determining success (default: 0.5)
        include_reason (bool): Whether to include reasoning in the response (default: True)
        strict_mode (bool): Whether to use strict mode (default: False)
        verbose_mode (bool): Whether to include verbose logging (default: False)
    """
    name: Optional[str] = None
    slug: Optional[str] = None
    conversation: Optional[List[dict]] = None
    options: Optional[Mapping[str, float]] = None
    verbose_mode: bool = False
    strict_mode: bool = False
    include_reason: bool = True,
    async_mode: bool = True,
    threshold: float = 0.5

    def __init__(
        self,
        name: str,
        slug: str,
        conversation: List[dict],
        options: Mapping[str, float],
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        async_mode: bool = True,
    ):
        super().__init__(
            threshold=threshold,
            score_type=APIScorer.CLASSIFIER,
        )
        self.name = name
        self.verbose_mode = verbose_mode
        self.strict_mode = strict_mode
        self.include_reason = include_reason
        self.slug = slug
        self.conversation = conversation
        self.options = options
        self.async_mode = async_mode

    def update_name(self, name: str):
        """
        Updates the name of the scorer.
        """
        self.name = name
        
    def update_threshold(self, threshold: float):
        """
        Updates the threshold of the scorer.
        """
        self.threshold = threshold
    
    def update_conversation(self, conversation: List[dict]):
        """
        Updates the conversation with the new conversation.
        
        Sample conversation:
        [{'role': 'system', 'content': "Did the chatbot answer the user's question in a kind way?: {{actual_output}}."}]
        """
        self.conversation = conversation
        
    def update_options(self, options: Mapping[str, float]):
        """
        Updates the options with the new options.
        
        Sample options:
        {"yes": 1, "no": 0}
        """
        self.options = options

    def __str__(self):
        return f"ClassifierScorer(name={self.name}, slug={self.slug}, conversation={self.conversation}, threshold={self.threshold}, options={self.options})"

    # @model_serializer
    # def serialize_model(self) -> dict:
    #     """
    #     Defines how the ClassifierScorer should be serialized when model_dump() is called.
    #     """
    #     return {
    #         "name": self.name,
    #         "score_type": self.name,
    #         "conversation": self.conversation,
    #         "options": self.options,
    #         "threshold": self.threshold,
    #         "include_reason": self.include_reason,
    #         "async_mode": self.async_mode,
    #         "strict_mode": self.strict_mode,
    #         "verbose_mode": self.verbose_mode,
    #     }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score_type": self.name,
            "conversation": self.conversation,
            "options": self.options,
            "threshold": self.threshold,
            "include_reason": self.include_reason,
            "async_mode": self.async_mode,
            "strict_mode": self.strict_mode,
            "verbose_mode": self.verbose_mode,
        }
