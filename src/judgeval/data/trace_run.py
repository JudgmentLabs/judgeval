from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from judgeval.data import Trace
from judgeval.scorers import APIScorerConfig, BaseScorer
from judgeval.rules import Rule


class TraceRun(BaseModel):
    """
    Stores example and evaluation scorers together for running an eval task

    Args:
        project_name (str): The name of the project the evaluation results belong to
        eval_name (str): A name for this evaluation run
        traces (List[Trace]): The traces to evaluate
        scorers (List[Union[JudgmentScorer, BaseScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
        judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API
        rules (Optional[List[Rule]]): Rules to evaluate against scoring results
        append (Optional[bool]): Whether to append to existing evaluation results
        tools (Optional[List[Dict[str, Any]]]): List of tools to use for evaluation
    """

    organization_id: Optional[str] = None
    project_name: Optional[str] = None
    eval_name: Optional[str] = None
    traces: Optional[List[Trace]] = None
    scorers: List[Union[APIScorerConfig, BaseScorer]]
    model: Optional[str] = "gpt-4.1"
    trace_span_id: Optional[str] = None
    append: Optional[bool] = False
    # API Key will be "" until user calls client.run_eval(), then API Key will be set
    judgment_api_key: Optional[str] = ""
    override: Optional[bool] = False
    rules: Optional[List[Rule]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed = True
