from typing import List, Optional, Union
from pydantic import BaseModel, field_validator, Field

from judgeval.data import Example
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.constants import ACCEPTABLE_MODELS


class EvaluationRun(BaseModel):
    """
    Stores example and evaluation scorers together for running an eval task

    Args:
        project_name (str): The name of the project the evaluation results belong to
        eval_name (str): A name for this evaluation run
        examples (List[Example]): The examples to evaluate
        scorers (List[Union[JudgmentScorer, BaseScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
        judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API
    """

    organization_id: Optional[str] = None
    project_name: Optional[str] = Field(default=None, validate_default=True)
    eval_name: Optional[str] = Field(default=None, validate_default=True)
    examples: List[Example]
    scorers: List[Union[APIScorerConfig, BaseScorer]]
    model: Optional[str] = "gpt-4.1"
    trace_span_id: Optional[str] = None
    # API Key will be "" until user calls client.run_eval(), then API Key will be set
    judgment_api_key: Optional[str] = ""
    override: Optional[bool] = False
    append: Optional[bool] = False

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)

        data["scorers"] = [
            scorer.model_dump() for scorer in self.scorers
        ]  # Pydantic has problems with properly calling model_dump() on the scorers, so we need to do it manually

        return data

    @field_validator("examples")
    def validate_examples(cls, v):
        if not v:
            raise ValueError("Examples cannot be empty.")
        return v

    @field_validator("scorers", mode="before")
    def validate_scorers(cls, v):
        if not v:
            raise ValueError("Scorers cannot be empty.")
        if not all(
            isinstance(scorer, BaseScorer) or isinstance(scorer, APIScorerConfig)
            for scorer in v
        ):
            raise ValueError(
                "All scorers must be of type BaseScorer or APIScorerConfig."
            )
        return v

    @field_validator("model")
    def validate_model(cls, v, values):
        if not v:
            raise ValueError("Model cannot be empty.")

        # Check if model is string or list of strings
        if isinstance(v, str):
            if v not in ACCEPTABLE_MODELS:
                raise ValueError(
                    f"Model name {v} not recognized. Please select a valid model name.)"
                )
            return v

    class Config:
        arbitrary_types_allowed = True
