from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, field_validator, Field

from judgeval.data import Example, CustomExample
from judgeval.scorers import JudgevalScorer, APIJudgmentScorer
from judgeval.constants import ACCEPTABLE_MODELS
from judgeval.common.logger import debug, error
from judgeval.judges import JudgevalJudge
from judgeval.rules import Rule

class EvaluationRun(BaseModel):
    """
    Stores example and evaluation scorers together for running an eval task
    
    Args: 
        project_name (str): The name of the project the evaluation results belong to
        eval_name (str): A name for this evaluation run
        examples (Union[List[Example], List[CustomExample]]): The examples to evaluate
        scorers (List[Union[JudgmentScorer, JudgevalScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
        judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API
        rules (Optional[List[Rule]]): Rules to evaluate against scoring results
    """

    # The user will specify whether they want log_results when they call run_eval
    log_results: bool = False  # NOTE: log_results has to be set first because it is used to validate project_name and eval_name
    organization_id: Optional[str] = None
    project_name: Optional[str] = Field(default=None, validate_default=True)
    eval_name: Optional[str] = Field(default=None, validate_default=True)
    examples: Union[List[Example], List[CustomExample]]
    scorers: List[Union[APIJudgmentScorer, JudgevalScorer]]
    model: Optional[Union[str, List[str], JudgevalJudge]] = "gpt-4.1"
    aggregator: Optional[str] = Field(default=None, validate_default=True)
    metadata: Optional[Dict[str, Any]] = None
    trace_span_id: Optional[str] = None
    # API Key will be "" until user calls client.run_eval(), then API Key will be set
    judgment_api_key: Optional[str] = ""
    override: Optional[bool] = False
    append: Optional[bool] = False
    rules: Optional[List[Rule]] = None
    
    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)

        data["scorers"] = [
            scorer.to_dict() if hasattr(scorer, "to_dict")
            else scorer.model_dump() if hasattr(scorer, "model_dump")
            else {"score_type": scorer.score_type, "threshold": scorer.threshold}
            for scorer in self.scorers
        ]

        if self.rules:
            # Process rules to ensure proper serialization
              data["rules"] = [rule.model_dump() for rule in self.rules]
            
        return data

    @field_validator('log_results', mode='before')
    def validate_log_results(cls, v):
        if not isinstance(v, bool):
            raise ValueError(f"log_results must be a boolean. Received {v} of type {type(v)}")
        return v

    @field_validator('project_name')
    def validate_project_name(cls, v, values):
        if values.data.get('log_results', False) and not v:
            debug("No project name provided when log_results is True")
            error("Validation failed: Project name required when logging results")
            raise ValueError("Project name is required when log_results is True. Please include the project_name argument.")
        return v

    @field_validator('eval_name')
    def validate_eval_name(cls, v, values):
        if values.data.get('log_results', False) and not v:
            debug("No eval name provided when log_results is True") 
            error("Validation failed: Eval name required when logging results")
            raise ValueError("Eval name is required when log_results is True. Please include the eval_run_name argument.")
        return v

    @field_validator('examples')
    def validate_examples(cls, v):
        if not v:
            raise ValueError("Examples cannot be empty.")
        
        first_type = type(v[0])
        if first_type not in (Example, CustomExample):
            raise ValueError(f"Invalid type for Example/CustomExample: {first_type}")
        if not all(isinstance(ex, first_type) for ex in v):
            raise ValueError("All examples must be of the same type, either all Example or all CustomExample.")
        
        return v

    @field_validator('scorers')
    def validate_scorers(cls, v):
        if not v:
            raise ValueError("Scorers cannot be empty.")
        return v

    @field_validator('model')
    def validate_model(cls, v, values):
        if not v:
            raise ValueError("Model cannot be empty.")
        
        # Check if model is a judgevalJudge
        if isinstance(v, JudgevalJudge):
            # Verify all scorers are JudgevalScorer when using judgevalJudge
            scorers = values.data.get('scorers', [])
            if not all(isinstance(s, JudgevalScorer) for s in scorers):
                raise ValueError("When using a judgevalJudge model, all scorers must be JudgevalScorer type")
            return v
            
        # Check if model is string or list of strings
        if isinstance(v, str):
            if v not in ACCEPTABLE_MODELS:
                raise ValueError(f"Model name {v} not recognized. Please select a valid model name.)")
            return v
            
        if isinstance(v, list):
            if not all(isinstance(m, str) for m in v):
                raise ValueError("When providing a list of models, all elements must be strings")
            for m in v:
                if m not in ACCEPTABLE_MODELS:
                    raise ValueError(f"Model name {m} not recognized. Please select a valid model name.")
            return v
        raise ValueError(f"Model must be one of: string, list of strings, or JudgevalJudge instance. Received type {type(v)}.")

    @field_validator('aggregator', mode='before')
    def validate_aggregator(cls, v, values):
        model = values.data.get('model')
        if isinstance(model, list) and v is None:
            raise ValueError("Aggregator cannot be empty.")
            
        if isinstance(model, list) and not isinstance(v, str):
            raise ValueError("Aggregator must be a string if provided.")
            
        if v is not None and v not in ACCEPTABLE_MODELS:
            raise ValueError(f"Model name {v} not recognized.")
            
        return v
    
    class Config:
        arbitrary_types_allowed = True
