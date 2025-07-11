from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from typing import Mapping, Optional, Dict, Any
import requests
from judgeval.constants import ROOT_API
from judgeval.common.exceptions import JudgmentAPIError
import os


def push_prompt_scorer(
    name: str,
    prompt: str,
    options: Mapping[str, float],
    judgment_api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY"),
    organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID"),
) -> str:
    """
    Pushes a classifier scorer configuration to the Judgment API.

    Returns:
        str: The slug identifier of the saved scorer

    Raises:
        JudgmentAPIError: If there's an error saving the scorer
    """
    request_body = {
        "name": name,
        "prompt": prompt,
        "options": options,
    }

    response = requests.post(
        f"{ROOT_API}/save_scorer/",
        json=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judgment_api_key}",
            "X-Organization-Id": organization_id,
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
    return response.json()["name"]


def fetch_prompt_scorer(
    name: str,
    judgment_api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY"),
    organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID"),
):
    """
    Fetches a classifier scorer configuration from the Judgment API.

    Args:
        slug (str): Slug identifier of the custom scorer to fetch

    Returns:
        dict: The configured classifier scorer object as a dictionary

    Raises:
        JudgmentAPIError: If the scorer cannot be fetched or doesn't exist
    """
    request_body = {
        "name": name,
    }

    response = requests.post(
        f"{ROOT_API}/fetch_scorer/",
        json=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judgment_api_key}",
            "X-Organization-Id": organization_id,
        },
        verify=True,
    )

    if response.status_code == 500:
        raise JudgmentAPIError(
            f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {response.json().get('detail', '')}"
        )
    elif response.status_code != 200:
        raise JudgmentAPIError(
            f"Failed to fetch classifier scorer '{name}': {response.json().get('detail', '')}"
        )

    scorer_config = response.json()
    scorer_config.pop("created_at")
    scorer_config.pop("updated_at")

    return scorer_config


def scorer_exists(
    name: str,
    judgment_api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY"),
    organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID"),
):
    """
    Checks if a scorer exists in the DB.
    """
    request_body = {
        "name": name,
    }

    response = requests.post(
        f"{ROOT_API}/scorer_exists/",
        json=request_body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judgment_api_key}",
            "X-Organization-Id": organization_id,
        },
        verify=True,
    )

    if response.status_code == 500:
        raise JudgmentAPIError(
            f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {response.json().get('detail', '')}"
        )
    elif response.status_code != 200:
        raise JudgmentAPIError(
            f"Failed to check if scorer exists: {response.json().get('detail', '')}"
        )
    return response.json()["exists"]


class PromptScorer(APIScorerConfig):
    """
    In the Judgment backend, this scorer is implemented as a PromptScorer that takes
    1. a system role that may involve the Example object
    2. options for scores on the example

    and uses a judge to execute the evaluation from the system role and classify into one of the options
    """

    prompt: str
    options: Mapping[str, float]
    score_type: APIScorerType = APIScorerType.PROMPT_SCORER
    judgment_api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY")
    organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID")

    @classmethod
    def get(
        cls,
        name: str,
        judgment_api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY"),
        organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID"),
    ):
        scorer_config = fetch_prompt_scorer(name, judgment_api_key, organization_id)
        return cls(
            name=name,
            prompt=scorer_config["prompt"],
            options=scorer_config["options"],
            judgment_api_key=judgment_api_key,
            organization_id=organization_id,
        )

    @classmethod
    def create(
        cls,
        name: str,
        prompt: str,
        options: Mapping[str, float],
        judgment_api_key: Optional[str] = os.getenv("JUDGMENT_API_KEY"),
        organization_id: Optional[str] = os.getenv("JUDGMENT_ORG_ID"),
    ):
        if not scorer_exists(name, judgment_api_key, organization_id):
            push_prompt_scorer(name, prompt, options, judgment_api_key, organization_id)
            return cls(
                name=name,
                prompt=prompt,
                options=options,
                judgment_api_key=judgment_api_key,
                organization_id=organization_id,
            )
        else:
            raise JudgmentAPIError(
                f"Scorer with name {name} already exists. Either use the existing scorer with the get() method or use a new name."
            )

    # Setter functions. Each setter function pushes the scorer to the DB.
    def update_name(self, name: str):
        """
        Updates the name of the scorer.
        """
        self.name = name
        self.push_prompt_scorer()

    def update_threshold(self, threshold: float):
        """
        Updates the threshold of the scorer.
        """
        self.threshold = threshold
        self.push_prompt_scorer()

    def update_prompt(self, prompt: str):
        """
        Updates the prompt with the new prompt.

        Sample prompt:
        "Did the chatbot answer the user's question in a kind way?"
        """
        self.prompt = prompt
        self.push_prompt_scorer()

    def update_options(self, options: Mapping[str, float]):
        """
        Updates the options with the new options.

        Sample options:
        {"yes": 1, "no": 0}
        """
        self.options = options
        self.push_prompt_scorer()

    # Getters
    def get_prompt(self) -> str | None:
        """
        Returns the prompt of the scorer.
        """
        return self.prompt

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

    def get_config(self) -> dict:
        """
        Returns a dictionary with all the fields in the scorer.
        """
        return {
            "name": self.name,
            "prompt": self.prompt,
            "options": self.options,
        }

    def push_prompt_scorer(self):
        """
        Pushes the scorer to the DB.
        """
        push_prompt_scorer(
            self.name,
            self.prompt,
            self.options,
            self.judgment_api_key,
            self.organization_id,
        )

    def __str__(self):
        return f"PromptScorer(name={self.name}, prompt={self.prompt}, options={self.options})"

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        base = super().model_dump(*args, **kwargs)
        base_fields = set(APIScorerConfig.model_fields.keys())
        all_fields = set(self.__class__.model_fields.keys())

        extra_fields = all_fields - base_fields - {"kwargs"}

        base["kwargs"] = {
            k: getattr(self, k) for k in extra_fields if getattr(self, k) is not None
        }
        return base
