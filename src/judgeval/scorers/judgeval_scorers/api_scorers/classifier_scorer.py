from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from typing import List, Mapping, Optional
import requests
from judgeval.constants import ROOT_API
from judgeval.common.exceptions import JudgmentAPIError
import os
from slugify import slugify
import random
import string


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
    include_reason: bool = True
    async_mode: bool = True
    threshold: float = 0.5

    # Constructor. Sets the variables and pushes the scorer to the DB.
    def __init__(
        self,
        name: Optional[str] = None,
        conversation: Optional[List[dict]] = None,
        options: Optional[Mapping[str, float]] = None,
        threshold: float = 0.5,
        slug: Optional[str] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        async_mode: bool = True,
    ):
        super().__init__(
            threshold=threshold,
            score_type=APIScorer.CLASSIFIER,
        )

        if slug and not (name is None and conversation is None and options is None):
            raise ValueError(
                "Only provide the slug if you are fetching an existing scorer (and don't set the name, conversation or options). If you are creating a new scorer, pass in values for name, conversation, and options."
            )

        if slug:
            self.slug = slug
            scorer_config = self.fetch_classifier_scorer(slug)
            self.name = scorer_config["name"]
            self.conversation = scorer_config["conversation"]
            self.options = scorer_config["options"]
            self.threshold = threshold
            self.verbose_mode = verbose_mode
            self.strict_mode = strict_mode
            self.include_reason = include_reason
            self.async_mode = async_mode
        elif name and conversation and options:
            self.name = name
            self.verbose_mode = verbose_mode
            self.strict_mode = strict_mode
            self.include_reason = include_reason
            self.slug = slugify(name) + "-" + self._generate_suffix()
            self.conversation = conversation
            self.options = options
            self.async_mode = async_mode
            self.push_classifier_scorer()
        else:
            raise ValueError(
                "You must provide the name, conversation, and options to create a new scorer. If you are fetching an existing scorer, pass in the slug. The conversation and options variables must be non-empty."
            )

    # Setter functions. Each setter function pushes the scorer to the DB.
    def update_name(self, name: str):
        """
        Updates the name of the scorer.
        """
        self.name = name
        self.push_classifier_scorer()

    def update_threshold(self, threshold: float):
        """
        Updates the threshold of the scorer.
        """
        self.threshold = threshold
        self.push_classifier_scorer()

    def update_conversation(self, conversation: List[dict]):
        """
        Updates the conversation with the new conversation.

        Sample conversation:
        [{'role': 'system', 'content': "Did the chatbot answer the user's question in a kind way?: {{actual_output}}."}]
        """
        self.conversation = conversation
        self.push_classifier_scorer()

    def update_options(self, options: Mapping[str, float]):
        """
        Updates the options with the new options.

        Sample options:
        {"yes": 1, "no": 0}
        """
        self.options = options
        self.push_classifier_scorer()

    # Getters
    def get_conversation(self) -> List[dict] | None:
        """
        Returns the conversation of the scorer.
        """
        return self.conversation

    def get_options(self) -> Mapping[str, float] | None:
        """
        Returns the options of the scorer.
        """
        return self.options

    def get_name(self) -> str | None:
        """
        Returns the name of the scorer.
        """
        return self.name

    def get_slug(self) -> str | None:
        """
        Returns the slug of the scorer.
        """
        return self.slug

    def get_config(self) -> dict:
        """
        Returns a dictionary with all the fields in the scorer.
        """
        return {
            "name": self.name,
            "slug": self.slug,
            "conversation": self.conversation,
            "threshold": self.threshold,
            "options": self.options,
        }

    def push_classifier_scorer(self) -> str:
        """
        Pushes a classifier scorer configuration to the Judgment API.

        Args:
            slug (str): Slug identifier for the scorer. If it exists, the scorer will be updated.
            scorer (ClassifierScorer): The classifier scorer to save

        Returns:
            str: The slug identifier of the saved scorer

        Raises:
            JudgmentAPIError: If there's an error saving the scorer
        """
        request_body = {
            "name": self.name,
            "conversation": self.conversation,
            "options": self.options,
            "slug": self.slug,
        }

        response = requests.post(
            f"{ROOT_API}/save_scorer/",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('JUDGMENT_API_KEY')}",
                "X-Organization-Id": os.getenv("JUDGMENT_ORG_ID"),
            },
            verify=True,
        )

        if response.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. \
                                    Please try your request again in a few moments. \
                                    Error details: {response.json().get('detail', '')}"
            )
        elif response.status_code != 200:
            raise JudgmentAPIError(
                f"Failed to save classifier scorer: {response.json().get('detail', '')}"
            )

        return response.json()["slug"]

    def fetch_classifier_scorer(self, slug: str):
        """
        Fetches a classifier scorer configuration from the Judgment API.

        Args:
            slug (str): Slug identifier of the custom scorer to fetch

        Returns:
            ClassifierScorer: The configured classifier scorer object

        Raises:
            JudgmentAPIError: If the scorer cannot be fetched or doesn't exist
        """
        request_body = {
            "slug": slug,
        }

        response = requests.post(
            f"{ROOT_API}/fetch_scorer/",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('JUDGMENT_API_KEY')}",
                "X-Organization-Id": os.getenv("JUDGMENT_ORG_ID"),
            },
            verify=True,
        )

        if response.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {response.json().get('detail', '')}"
            )
        elif response.status_code != 200:
            raise JudgmentAPIError(
                f"Failed to fetch classifier scorer '{slug}': {response.json().get('detail', '')}"
            )

        scorer_config = response.json()
        scorer_config.pop("created_at")
        scorer_config.pop("updated_at")

        try:
            return scorer_config
        except Exception as e:
            raise JudgmentAPIError(
                f"Failed to create classifier scorer '{slug}' with config {scorer_config}: {str(e)}"
            )

    def _generate_suffix(self):
        """
        Generates a suffix for the scorer.
        """
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))

    def __str__(self):
        return f"ClassifierScorer(name={self.name}, slug={self.slug}, conversation={self.conversation}, threshold={self.threshold}, options={self.options})"

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
