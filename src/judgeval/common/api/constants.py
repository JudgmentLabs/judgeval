import os
from typing import Optional, TypedDict, List, Dict, Any

ROOT_API = os.getenv("JUDGMENT_API_URL", "https://api.judgmentlabs.ai")

# Traces API
JUDGMENT_TRACES_FETCH_API_URL = f"{ROOT_API}/traces/fetch/"
JUDGMENT_TRACES_SAVE_API_URL = f"{ROOT_API}/traces/save/"
JUDGMENT_TRACES_UPSERT_API_URL = f"{ROOT_API}/traces/upsert/"
JUDGMENT_TRACES_DELETE_API_URL = f"{ROOT_API}/traces/delete/"
JUDGMENT_TRACES_SPANS_BATCH_API_URL = f"{ROOT_API}/traces/spans/batch/"
JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL = (
    f"{ROOT_API}/traces/evaluation_runs/batch/"
)


class TraceFetchPayload(TypedDict):
    trace_id: str


class TraceDeletePayload(TypedDict):
    trace_ids: List[str]


class SpansBatchPayload(TypedDict):
    spans: List[Dict[str, Any]]
    organization_id: str


class EvaluationEntryResponse(TypedDict):
    evaluation_run: Dict[str, Any]
    associated_span: Dict[str, Any]
    queued_at: Optional[float]


class EvaluationRunsBatchPayload(TypedDict):
    organization_id: str
    evaluation_entries: List[EvaluationEntryResponse]


# Evaluation API
JUDGMENT_EVAL_API_URL = f"{ROOT_API}/evaluate/"
JUDGMENT_TRACE_EVAL_API_URL = f"{ROOT_API}/evaluate_trace/"
JUDGMENT_EVAL_LOG_API_URL = f"{ROOT_API}/log_eval_results/"
JUDGMENT_EVAL_FETCH_API_URL = f"{ROOT_API}/fetch_experiment_run/"
JUDGMENT_EVAL_DELETE_API_URL = (
    f"{ROOT_API}/delete_eval_results_by_project_and_run_names/"
)
JUDGMENT_EVAL_DELETE_PROJECT_API_URL = f"{ROOT_API}/delete_eval_results_by_project/"
JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL = f"{ROOT_API}/add_to_run_eval_queue/"
JUDGMENT_GET_EVAL_STATUS_API_URL = f"{ROOT_API}/get_evaluation_status/"
JUDGMENT_CHECK_EXPERIMENT_TYPE_API_URL = f"{ROOT_API}/check_experiment_type/"
JUDGMENT_EVAL_RUN_NAME_EXISTS_API_URL = f"{ROOT_API}/eval-run-name-exists/"
JUDGMENT_CHECK_EXAMPLE_KEYS_API_URL = f"{ROOT_API}/check_example_keys/"


# Evaluation API Payloads
class EvalRunRequestBody(TypedDict):
    eval_name: str
    project_name: str
    judgment_api_key: str


class DeleteEvalRunRequestBody(TypedDict):
    eval_names: List[str]
    project_name: str
    judgment_api_key: str


class EvalLogPayload(TypedDict):
    results: List[Dict[str, Any]]
    run: Dict[str, Any]


class EvalStatusPayload(TypedDict):
    eval_name: str
    project_name: str
    judgment_api_key: str


class CheckExperimentTypePayload(TypedDict):
    eval_name: str
    project_name: str
    judgment_api_key: str
    is_trace: bool


class EvalRunNameExistsPayload(TypedDict):
    eval_name: str
    project_name: str
    judgment_api_key: str


class CheckExampleKeysPayload(TypedDict):
    keys: List[str]
    eval_name: str
    project_name: str


# Datasets API
JUDGMENT_DATASETS_PUSH_API_URL = f"{ROOT_API}/datasets/push/"
JUDGMENT_DATASETS_APPEND_EXAMPLES_API_URL = f"{ROOT_API}/datasets/insert_examples/"
JUDGMENT_DATASETS_APPEND_TRACES_API_URL = f"{ROOT_API}/traces/add_to_dataset/"
JUDGMENT_DATASETS_PULL_API_URL = f"{ROOT_API}/datasets/pull_for_judgeval/"
JUDGMENT_DATASETS_DELETE_API_URL = f"{ROOT_API}/datasets/delete/"
JUDGMENT_DATASETS_EXPORT_JSONL_API_URL = f"{ROOT_API}/datasets/export_jsonl/"
JUDGMENT_DATASETS_PROJECT_STATS_API_URL = f"{ROOT_API}/datasets/fetch_stats_by_project/"
JUDGMENT_DATASETS_INSERT_API_URL = f"{ROOT_API}/datasets/insert_examples/"


class DatasetPushPayload(TypedDict):
    dataset_alias: str
    project_name: str
    examples: List[Dict[str, Any]]
    traces: List[Dict[str, Any]]
    overwrite: bool


class DatasetAppendPayload(TypedDict):
    dataset_alias: str
    project_name: str
    examples: List[Dict[str, Any]]


class DatasetPullPayload(TypedDict):
    dataset_alias: str
    project_name: str


class DatasetDeletePayload(TypedDict):
    dataset_alias: str
    project_name: str


class DatasetExportPayload(TypedDict):
    dataset_alias: str
    project_name: str


class DatasetStatsPayload(TypedDict):
    project_name: str


# Projects API
JUDGMENT_PROJECT_DELETE_API_URL = f"{ROOT_API}/projects/delete_from_judgeval/"
JUDGMENT_PROJECT_CREATE_API_URL = f"{ROOT_API}/projects/add/"


class ProjectDeletePayload(TypedDict):
    project_list: List[str]


class ProjectCreatePayload(TypedDict):
    project_name: str


JUDGMENT_SCORER_SAVE_API_URL = f"{ROOT_API}/save_scorer/"
JUDGMENT_SCORER_FETCH_API_URL = f"{ROOT_API}/fetch_scorer/"
JUDGMENT_SCORER_EXISTS_API_URL = f"{ROOT_API}/scorer_exists/"


class ScorerSavePayload(TypedDict):
    name: str
    prompt: str
    threshold: float
    options: Optional[dict]


class ScorerFetchPayload(TypedDict):
    name: str


class ScorerExistsPayload(TypedDict):
    name: str
