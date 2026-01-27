from __future__ import annotations

from typing import TypedDict, Optional, List, Union, Any, Dict
from typing_extensions import NotRequired


class AddProjectRequest(TypedDict):
    project_name: str


class AddProjectResponse(TypedDict):
    project_id: str


class AddToRunEvalQueueExamplesResponse(TypedDict):
    success: bool
    status: str
    message: str


class AddToRunEvalQueueRequest(TypedDict):
    pass


class AddToRunEvalQueueTracesResponse(TypedDict):
    success: bool
    status: str
    message: str


class AddTraceTagsRequest(TypedDict):
    project_name: str
    trace_id: str
    tags: List[str]


class AddTraceTagsResponse(TypedDict):
    success: bool


class BaseScorer(TypedDict):
    score_type: str
    name: NotRequired[Optional[str]]
    class_name: NotRequired[Optional[str]]
    score: NotRequired[Optional[float]]
    minimum_score_range: NotRequired[Optional[float]]
    maximum_score_range: NotRequired[Optional[float]]
    score_breakdown: NotRequired[Optional[Dict[str, Any]]]
    reason: NotRequired[Optional[Union[str, Dict[str, Any]]]]
    success: NotRequired[Optional[bool]]
    model: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    additional_metadata: NotRequired[Optional[Dict[str, Any]]]
    user: NotRequired[Optional[str]]
    server_hosted: bool
    using_native_model: NotRequired[Optional[bool]]
    required_params: NotRequired[Optional[List[str]]]
    strict_mode: NotRequired[Optional[bool]]


class CreateDatasetRequest(TypedDict):
    name: str
    dataset_kind: str
    project_name: str
    examples: List[Example]
    overwrite: bool


class CreateDatasetResponse(TypedDict):
    detail: str


class DatasetInfo(TypedDict):
    dataset_id: str
    name: str
    created_at: str
    kind: str
    entries: float
    creator: str


class DeleteProjectRequest(TypedDict):
    project_name: str


class DeleteProjectResponse(TypedDict):
    status: str
    message: str


class ErrorResponse(TypedDict):
    error: str
    message: NotRequired[Optional[str]]


class Example(TypedDict):
    example_id: str
    created_at: str
    name: NotRequired[Optional[str]]


class ExampleEvaluationRun(TypedDict):
    id: NotRequired[Optional[str]]
    project_name: str
    eval_name: str
    model: NotRequired[Optional[str]]
    created_at: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    scorers: NotRequired[Optional[List[Any]]]
    custom_scorers: List[BaseScorer]
    judgment_scorers: List[ScorerConfig]
    examples: List[Example]
    trace_span_id: NotRequired[Optional[str]]
    trace_id: NotRequired[Optional[str]]


class ExampleScoringResult(TypedDict):
    success: bool
    scorers_data: List[ScorerData]
    name: NotRequired[Optional[str]]
    data_object: Example
    trace_id: NotRequired[Optional[str]]
    run_duration: NotRequired[Optional[float]]
    evaluation_cost: NotRequired[Optional[float]]


class ExperimentRunItem(TypedDict):
    organization_id: str
    experiment_run_id: str
    example_id: str
    data: Dict[str, Any]
    name: NotRequired[Optional[str]]
    created_at: str
    scorers: List[ExperimentScorer]


class ExperimentScorer(TypedDict):
    scorer_data_id: str
    name: str
    score: float
    success: float
    reason: NotRequired[Optional[str]]
    evaluation_model: NotRequired[Optional[str]]
    threshold: float
    created_at: str
    error: NotRequired[Optional[str]]
    additional_metadata: NotRequired[Optional[Dict[str, Any]]]
    minimum_score_range: float
    maximum_score_range: float


class FetchExperimentRunRequest(TypedDict):
    experiment_run_id: str
    project_name: str


class FetchExperimentRunResponse(TypedDict):
    results: NotRequired[Optional[List[ExperimentRunItem]]]
    ui_results_url: NotRequired[Optional[str]]


class FetchPromptResponse(TypedDict):
    commit: NotRequired[Optional[PromptCommitInfo]]


class FetchPromptScorersRequest(TypedDict):
    names: NotRequired[Optional[List[str]]]
    is_trace: NotRequired[Optional[bool]]


class FetchPromptScorersResponse(TypedDict):
    scorers: List[PromptScorer]


class GetPromptVersionsResponse(TypedDict):
    versions: List[PromptCommitInfo]


class HealthResponse(TypedDict):
    status: str
    timestamp: str


class InsertExamplesRequest(TypedDict):
    dataset_name: str
    examples: List[Example]
    project_name: str


class InsertExamplesResponse(TypedDict):
    detail: str


class InsertPromptRequest(TypedDict):
    project_id: str
    name: str
    prompt: str
    tags: NotRequired[Optional[List[str]]]


class InsertPromptResponse(TypedDict):
    commit_id: str
    parent_commit_id: NotRequired[Optional[str]]
    created_at: str


class LogEvalResultsRequest(TypedDict):
    results: List[ScoringResult]
    run: ExampleEvaluationRun


class LogEvalResultsResponse(TypedDict):
    ui_results_url: str


class PromptCommitInfo(TypedDict):
    name: str
    prompt: str
    tags: List[str]
    commit_id: str
    parent_commit_id: NotRequired[Optional[str]]
    created_at: str
    first_name: str
    last_name: str
    user_email: str


class PromptScorer(TypedDict):
    id: str
    user_id: str
    organization_id: str
    name: str
    prompt: str
    threshold: float
    model: str
    options: NotRequired[Optional[Dict[str, Any]]]
    description: NotRequired[Optional[str]]
    created_at: NotRequired[Optional[str]]
    updated_at: NotRequired[Optional[str]]
    is_trace: NotRequired[Optional[bool]]
    is_bucket_rubric: NotRequired[Optional[bool]]


class PullAllDatasetsRequest(TypedDict):
    project_name: str


PullAllDatasetsResponse = List[DatasetInfo]


class PullDatasetRequest(TypedDict):
    dataset_name: str
    project_name: str


class PullDatasetResponse(TypedDict):
    name: str
    project_name: str
    dataset_kind: str
    examples: List[Example]


class ResolveProjectRequest(TypedDict):
    project_name: str


class ResolveProjectResponse(TypedDict):
    project_id: str


class SavePromptScorerRequest(TypedDict):
    name: str
    prompt: str
    threshold: float
    model: str
    is_trace: bool
    options: NotRequired[Optional[Dict[str, Any]]]
    description: NotRequired[Optional[str]]


class SavePromptScorerResponse(TypedDict):
    scorer_response: PromptScorer


class ScorerConfig(TypedDict):
    score_type: str
    name: NotRequired[Optional[str]]
    threshold: float
    model: NotRequired[Optional[str]]
    required_params: NotRequired[Optional[List[str]]]
    kwargs: NotRequired[Optional[Dict[str, Any]]]


class ScorerData(TypedDict):
    id: NotRequired[Optional[str]]
    name: str
    threshold: float
    success: bool
    score: NotRequired[Optional[float]]
    minimum_score_range: NotRequired[Optional[Union[str, float]]]
    maximum_score_range: NotRequired[Optional[Union[str, float]]]
    reason: NotRequired[Optional[Union[str, Dict[str, Any]]]]
    strict_mode: NotRequired[Optional[bool]]
    evaluation_model: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    additional_metadata: NotRequired[Optional[Dict[str, Any]]]


class ScorerExistsRequest(TypedDict):
    name: str


class ScorerExistsResponse(TypedDict):
    exists: bool


class ScoringResult(TypedDict):
    pass


class TagPromptRequest(TypedDict):
    project_id: str
    name: str
    commit_id: str
    tags: List[str]


class TagPromptResponse(TypedDict):
    commit_id: str


class TraceEvaluationRun(TypedDict):
    id: NotRequired[Optional[str]]
    project_name: str
    eval_name: str
    model: NotRequired[Optional[str]]
    created_at: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    scorers: NotRequired[Optional[List[Any]]]
    custom_scorers: List[BaseScorer]
    judgment_scorers: List[ScorerConfig]
    trace_and_span_ids: List[List[str]]
    is_offline: bool
    is_behavior: bool


class TraceInfo(TypedDict):
    trace_id: str
    span_id: str


class TraceScoringResult(TypedDict):
    success: bool
    scorers_data: List[ScorerData]
    name: NotRequired[Optional[str]]
    data_object: TraceSpan
    trace_id: NotRequired[Optional[str]]
    run_duration: NotRequired[Optional[float]]
    evaluation_cost: NotRequired[Optional[float]]


class TraceSpan(TypedDict):
    organization_id: str
    project_id: str
    user_id: str
    timestamp: str
    trace_id: str
    span_id: str
    parent_span_id: NotRequired[Optional[str]]
    trace_state: NotRequired[Optional[str]]
    span_name: NotRequired[Optional[str]]
    span_kind: NotRequired[Optional[str]]
    service_name: NotRequired[Optional[str]]
    resource_attributes: Dict[str, Any]
    span_attributes: Dict[str, Any]
    duration: str
    status_code: float
    status_message: NotRequired[Optional[str]]
    events: List[Dict[str, Any]]
    links: NotRequired[Optional[str]]


class TriggerRootSpanRulesRequest(TypedDict):
    traces: List[TraceInfo]


class TriggerRootSpanRulesResponse(TypedDict):
    success: bool
    queued_traces: float


class UntagPromptRequest(TypedDict):
    project_id: str
    name: str
    tags: List[str]


class UntagPromptResponse(TypedDict):
    commit_ids: List[str]


class UploadCustomScorerRequest(TypedDict):
    scorer_name: str
    scorer_code: str
    requirements_text: str
    class_name: str
    overwrite: bool
    scorer_type: NotRequired[Optional[str]]
    version: NotRequired[Optional[float]]


class UploadCustomScorerResponse(TypedDict):
    scorer_name: str
    status: str
    message: str


class WelcomeResponse(TypedDict):
    pass
