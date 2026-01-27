from judgeval.scorers.api_scorer import (
    APIScorerConfig,
    ExampleAPIScorerConfig,
    TraceAPIScorerConfig,
)
from judgeval.constants import APIScorerType
from typing import Dict, Any, Optional
from judgeval.api import JudgmentSyncClient
from judgeval.exceptions import JudgmentAPIError
import os
from judgeval.logger import judgeval_logger
from abc import ABC
from judgeval.env import JUDGMENT_DEFAULT_GPT_MODEL
from copy import copy
from judgeval.utils.decorators.dont_throw import dont_throw


def push_prompt_scorer(
    name: str,
    prompt: str,
    threshold: float,
    options: Optional[Dict[str, float]] = None,
    model: str = JUDGMENT_DEFAULT_GPT_MODEL,
    description: Optional[str] = None,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
    is_trace: bool = False,
) -> str:
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        r = client.save_scorer(
            payload={
                "name": name,
                "prompt": prompt,
                "threshold": threshold,
                "options": options,
                "model": model,
                "description": description,
                "is_trace": is_trace,
            }
        )
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to save prompt scorer: {e.detail}",
            response=e.response,
        )
    return r["scorer_response"]["name"]


def fetch_prompt_scorer(
    name: str,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
):
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        fetched_scorers = client.fetch_scorers({"names": [name]})
        if len(fetched_scorers["scorers"]) == 0:
            judgeval_logger.error(f"Prompt scorer '{name}' not found")
            raise JudgmentAPIError(
                status_code=404,
                detail=f"Prompt scorer '{name}' not found",
                response=None,  # type: ignore
            )
        else:
            scorer_config = fetched_scorers["scorers"][0]
            scorer_config.pop("created_at")
            scorer_config.pop("updated_at")
            return scorer_config
    except JudgmentAPIError as e:
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to fetch prompt scorer '{name}': {e.detail}",
            response=e.response,
        )


def scorer_exists(
    name: str,
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
):
    client = JudgmentSyncClient(judgment_api_key, organization_id)
    try:
        return client.scorer_exists({"name": name})["exists"]
    except JudgmentAPIError as e:
        if e.status_code == 500:
            raise JudgmentAPIError(
                status_code=e.status_code,
                detail=f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.detail}",
                response=e.response,
            )
        raise JudgmentAPIError(
            status_code=e.status_code,
            detail=f"Failed to check if scorer exists: {e.detail}",
            response=e.response,
        )


class BasePromptScorer(ABC, APIScorerConfig):
    score_type: APIScorerType
    prompt: str
    options: Optional[Dict[str, float]] = None
    description: Optional[str] = None
    judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or ""
    organization_id: str = os.getenv("JUDGMENT_ORG_ID") or ""

    @classmethod
    @dont_throw
    def get(
        cls,
        name: str,
        judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
        organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
    ):
        scorer_config = fetch_prompt_scorer(name, judgment_api_key, organization_id)
        if scorer_config["is_trace"] != issubclass(cls, TracePromptScorer):
            raise JudgmentAPIError(
                status_code=400,
                detail=f"Scorer with name {name} is not a {cls.__name__}",
                response=None,  # type: ignore
            )
        if issubclass(cls, TracePromptScorer):
            score_type = APIScorerType.TRACE_PROMPT_SCORER
        else:
            score_type = APIScorerType.PROMPT_SCORER
        return cls(
            score_type=score_type,
            name=name,
            prompt=scorer_config["prompt"],
            threshold=scorer_config["threshold"],
            options=scorer_config.get("options"),
            model=scorer_config.get("model"),
            description=scorer_config.get("description"),
            judgment_api_key=judgment_api_key,
            organization_id=organization_id,
        )

    @classmethod
    def create(
        cls,
        name: str,
        prompt: str,
        threshold: float = 0.5,
        options: Optional[Dict[str, float]] = None,
        model: str = JUDGMENT_DEFAULT_GPT_MODEL,
        description: Optional[str] = None,
        judgment_api_key: str = os.getenv("JUDGMENT_API_KEY") or "",
        organization_id: str = os.getenv("JUDGMENT_ORG_ID") or "",
    ):
        if not scorer_exists(name, judgment_api_key, organization_id):
            if issubclass(cls, TracePromptScorer):
                is_trace = True
                score_type = APIScorerType.TRACE_PROMPT_SCORER
            else:
                is_trace = False
                score_type = APIScorerType.PROMPT_SCORER
            push_prompt_scorer(
                name,
                prompt,
                threshold,
                options,
                model,
                description,
                judgment_api_key,
                organization_id,
                is_trace,
            )
            judgeval_logger.info(f"Successfully created PromptScorer: {name}")
            return cls(
                score_type=score_type,
                name=name,
                prompt=prompt,
                threshold=threshold,
                options=options,
                model=model,
                description=description,
                judgment_api_key=judgment_api_key,
                organization_id=organization_id,
            )
        else:
            raise JudgmentAPIError(
                status_code=400,
                detail=f"Scorer with name {name} already exists. Either use the existing scorer with the get() method or use a new name.",
                response=None,  # type: ignore
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
        judgeval_logger.info(f"Successfully updated prompt for {self.name}")

    def set_model(self, model: str):
        """
        Updates the model of the scorer.
        """
        self.model = model
        self.push_prompt_scorer()
        judgeval_logger.info(f"Successfully updated model for {self.name}")

    def set_options(self, options: Optional[Dict[str, float]]):
        """
        Updates the options of the scorer.
        """
        self.options = options
        self.push_prompt_scorer()
        judgeval_logger.info(f"Successfully updated options for {self.name}")

    def set_description(self, description: Optional[str]):
        """
        Updates the description of the scorer.
        """
        self.description = description
        self.push_prompt_scorer()
        judgeval_logger.info(f"Successfully updated description for {self.name}")

    def append_to_prompt(self, prompt_addition: str):
        """
        Appends a string to the prompt.
        """
        self.prompt += prompt_addition
        self.push_prompt_scorer()
        judgeval_logger.info(f"Successfully appended to prompt for {self.name}")

    # Getters
    def get_threshold(self) -> float:
        """
        Returns the threshold of the scorer.
        """
        return self.threshold

    def get_prompt(self) -> str:
        """
        Returns the prompt of the scorer.
        """
        return self.prompt

    def get_model(self) -> str:
        """
        Returns the model of the scorer.
        """
        return self.model

    def get_options(self) -> Dict[str, float] | None:
        """
        Returns the options of the scorer.
        """
        return copy(self.options) if self.options is not None else None

    def get_description(self) -> str | None:
        """
        Returns the description of the scorer.
        """
        return self.description

    def get_name(self) -> str:
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
            "model": self.model,
            "prompt": self.prompt,
            "threshold": self.threshold,
            "options": self.options,
            "description": self.description,
        }

    def push_prompt_scorer(self):
        """
        Pushes the scorer to the DB.
        """
        push_prompt_scorer(
            self.name,
            self.prompt,
            self.threshold,
            self.options,
            self.model,
            self.description,
            self.judgment_api_key,
            self.organization_id,
            isinstance(self, TracePromptScorer),
        )

    def __str__(self):
        return f"PromptScorer(name={self.name}, model={self.model}, prompt={self.prompt}, threshold={self.threshold}, options={self.options}, description={self.description})"

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        base = super().model_dump(*args, **kwargs)
        base_fields = set(APIScorerConfig.model_fields.keys())
        all_fields = set(self.__class__.model_fields.keys())

        extra_fields = all_fields - base_fields - {"kwargs"}

        base["kwargs"] = {
            k: getattr(self, k) for k in extra_fields if getattr(self, k) is not None
        }
        return base


class PromptScorer(BasePromptScorer, ExampleAPIScorerConfig):
    pass


class TracePromptScorer(BasePromptScorer, TraceAPIScorerConfig):
    pass
