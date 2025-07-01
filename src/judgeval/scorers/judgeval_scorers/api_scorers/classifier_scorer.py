from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from typing import List, Mapping, Optional, Dict, Any


class ClassifierScorer(APIScorerConfig):
    """
    In the Judgment backend, this scorer is implemented as a PromptScorer that takes
    1. a system role that may involve the Example object
    2. options for scores on the example

    and uses a judge to execute the evaluation from the system role and classify into one of the options

    ex:
    system_role = "You are a judge that evaluates whether the response is positive or negative. The response is: {example.actual_output}"
    options = {"positive": 1, "negative": 0}

    Args:
        slug (str): A unique identifier for the scorer
        conversation (List[dict]): The conversation template with placeholders (e.g., {{actual_output}})
        options (Mapping[str, float]): A mapping of classification options to their corresponding scores
    """

    slug: Optional[str] = None
    conversation: Optional[List[dict]] = None
    options: Optional[Mapping[str, float]] = None
    score_type: APIScorerType = APIScorerType.PROMPT_SCORER

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

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        base = super().model_dump(*args, **kwargs)
        base_fields = set(APIScorerConfig.model_fields.keys())
        all_fields = set(self.__class__.model_fields.keys())

        extra_fields = all_fields - base_fields - {"kwargs"}

        base["kwargs"] = {
            k: getattr(self, k) for k in extra_fields if getattr(self, k) is not None
        }

        return base
