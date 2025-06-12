from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Callable
from judgeval.data import Trace
from judgeval.scorers import APIJudgmentScorer, JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.rules import Rule


class TraceRun(BaseModel):
    """
    Stores example and evaluation scorers together for running an eval task
    
    Args: 
        project_name (str): The name of the project the evaluation results belong to
        eval_name (str): A name for this evaluation run
        traces (List[Trace]): The traces to evaluate
        scorers (List[Union[JudgmentScorer, JudgevalScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
        judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API
        rules (Optional[List[Rule]]): Rules to evaluate against scoring results
        append (Optional[bool]): Whether to append to existing evaluation results
        tools (Optional[List[Dict[str, Any]]]): List of tools to use for evaluation
    """

    # The user will specify whether they want log_results when they call run_eval
    log_results: bool = False  # NOTE: log_results has to be set first because it is used to validate project_name and eval_name
    organization_id: Optional[str] = None
    project_name: Optional[str] = None
    eval_name: Optional[str] = None
    traces: Optional[List[Trace]] = None
    scorers: List[Union[APIJudgmentScorer, JudgevalScorer]]
    model: Optional[Union[str, List[str], JudgevalJudge]] = "gpt-4.1"
    aggregator: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    trace_span_id: Optional[str] = None
    append: Optional[bool] = False
    # API Key will be "" until user calls client.run_eval(), then API Key will be set
    judgment_api_key: Optional[str] = ""
    override: Optional[bool] = False
    rules: Optional[List[Rule]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed = True