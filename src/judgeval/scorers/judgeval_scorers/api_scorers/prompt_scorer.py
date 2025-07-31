from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from typing import Dict, Any, Optional
from judgeval.common.api import JudgmentApiClient, JudgmentAPIException
import os
from judgeval.common.exceptions import JudgmentAPIError
from copy import copy


def push_prompt_scorer(
    name: str,
    prompt: str,
    options: Optional[Dict[str, float]] = None,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
) -> str:
    client = JudgmentApiClient(judgment_api_key, organization_id)
    try:
        r = client.save_scorer(name, prompt, options)
    except JudgmentAPIException as e:
        if e.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.error_detail}"
            )
        raise JudgmentAPIError(f"Failed to save classifier scorer: {e.error_detail}")
    return r["name"]


def fetch_prompt_scorer(
    name: str,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
):
    client = JudgmentApiClient(judgment_api_key, organization_id)
    try:
        scorer_config = client.fetch_scorer(name)["scorer"]
        scorer_config.pop("created_at")
        scorer_config.pop("updated_at")
        return scorer_config
    except JudgmentAPIException as e:
        if e.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.error_detail}"
            )
        raise JudgmentAPIError(
            f"Failed to fetch classifier scorer '{name}': {e.error_detail}"
        )


def scorer_exists(
    name: str,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
):
    client = JudgmentApiClient(judgment_api_key, organization_id)
    try:
        return client.scorer_exists(name)["exists"]
    except JudgmentAPIException as e:
        if e.status_code == 500:
            raise JudgmentAPIError(
                f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.error_detail}"
            )
        raise JudgmentAPIError(f"Failed to check if scorer exists: {e.error_detail}")


class PromptScorer(APIScorerConfig):
    """
    In the Judgment backend, this scorer is implemented as a PromptScorer that takes
    1. a system role that may involve the Example object
    2. options for scores on the example

    and uses a judge to execute the evaluation from the system role and classify into one of the options
    """

    prompt: str
    options: Optional[Dict[str, float]] = None
    score_type: APIScorerType = APIScorerType.PROMPT_SCORER
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or ""
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or ""

    @classmethod
    def get(
        cls,
        name: str,
        judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
        organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
    ):
        scorer_config = fetch_prompt_scorer(name, judgment_api_key, organization_id)
        return cls(
            name=name,
            prompt=scorer_config["prompt"],
            options=scorer_config.get("options"),
            judgment_api_key=judgment_api_key,
            organization_id=organization_id,
        )

    @classmethod
    def create(
        cls,
        name: str,
        prompt: str,
        options: Optional[Dict[str, float]] = None,
        judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
        organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
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
    def set_threshold(self, threshold: float):
        """
        Updates the threshold of the scorer.
        """
        self.threshold = threshold
        self.push_prompt_scorer()

    def set_prompt(self, prompt: str):
        """
        Updates the prompt with the new prompt.

        Sample prompt:
        "Did the chatbot answer the user's question in a kind way?"
        """
        self.prompt = prompt
        self.push_prompt_scorer()

    def set_options(self, options: Dict[str, float]):
        """
        Updates the options with the new options.

        Sample options:
        {"yes": 1, "no": 0}
        """
        self.options = options
        self.push_prompt_scorer()

    def append_to_prompt(self, prompt_addition: str):
        """
        Appends a string to the prompt.
        """
        self.prompt += prompt_addition
        self.push_prompt_scorer()

    # Getters
    def get_prompt(self) -> str | None:
        """
        Returns the prompt of the scorer.
        """
        return self.prompt

    def get_options(self) -> Dict[str, float] | None:
        """
        Returns the options of the scorer.
        """
        return copy(self.options) if self.options is not None else None

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
