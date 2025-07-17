from typing import Literal, List, Dict, Any
from requests import exceptions
from judgeval.common.api.constants import (
    JUDGMENT_TRACES_FETCH_API_URL,
    JUDGMENT_TRACES_UPSERT_API_URL,
    JUDGMENT_TRACES_DELETE_API_URL,
    JUDGMENT_TRACES_SPANS_BATCH_API_URL,
    JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
    JUDGMENT_DATASETS_PUSH_API_URL,
    JUDGMENT_DATASETS_APPEND_EXAMPLES_API_URL,
    JUDGMENT_DATASETS_PULL_API_URL,
    JUDGMENT_DATASETS_DELETE_API_URL,
    JUDGMENT_DATASETS_PROJECT_STATS_API_URL,
    JUDGMENT_PROJECT_DELETE_API_URL,
    JUDGMENT_PROJECT_CREATE_API_URL,
    JUDGMENT_EVAL_API_URL,
    JUDGMENT_TRACE_EVAL_API_URL,
    JUDGMENT_EVAL_LOG_API_URL,
    JUDGMENT_EVAL_FETCH_API_URL,
    JUDGMENT_EVAL_DELETE_API_URL,
    JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL,
    JUDGMENT_GET_EVAL_STATUS_API_URL,
    JUDGMENT_CHECK_EXPERIMENT_TYPE_API_URL,
    JUDGMENT_EVAL_RUN_NAME_EXISTS_API_URL,
    JUDGMENT_SCORER_SAVE_API_URL,
    JUDGMENT_SCORER_FETCH_API_URL,
    JUDGMENT_SCORER_EXISTS_API_URL,
)
from judgeval.common.api.constants import (
    TraceFetchPayload,
    TraceDeletePayload,
    SpansBatchPayload,
    EvaluationEntryResponse,
    EvaluationRunsBatchPayload,
    DatasetPushPayload,
    DatasetAppendPayload,
    DatasetPullPayload,
    DatasetDeletePayload,
    DatasetStatsPayload,
    ProjectCreatePayload,
    ProjectDeletePayload,
    EvalRunRequestBody,
    DeleteEvalRunRequestBody,
    EvalLogPayload,
    EvalStatusPayload,
    CheckExperimentTypePayload,
    EvalRunNameExistsPayload,
    ScorerSavePayload,
    ScorerFetchPayload,
    ScorerExistsPayload,
)
from judgeval.utils.requests import requests


class JudgmentAPIException(exceptions.HTTPError):
    """
    Exception raised when an error occurs while executing a Judgment API request.
    Extends requests.exceptions.HTTPError to provide access to the response object.
    """

    def __init__(self, message: str, response=None, request=None):
        super().__init__(message, response=response, request=request)
        self.message = message
        self.response = response
        self.request = request

    @property
    def status_code(self) -> int | None:
        """Get the HTTP status code from the response."""
        return self.response.status_code if self.response else None

    @property
    def response_json(self) -> Dict[str, Any]:
        """Get the JSON response body."""
        try:
            return self.response.json() if self.response else {}
        except (ValueError, AttributeError):
            return {}

    @property
    def error_detail(self) -> str:
        """Get the error detail from the response JSON."""
        return self.response_json.get("detail", "An unknown error occurred.")


class JudgmentApiClient:
    def __init__(self, api_key: str, organization_id: str):
        self.api_key = api_key
        self.organization_id = organization_id

    def _do_request(
        self,
        method: Literal["POST", "PATCH", "GET", "DELETE"],
        url: str,
        payload: Any,
    ) -> Any:
        if method == "GET":
            r = requests.request(
                method,
                url,
                params=payload,
                headers=self._headers(),
                **self._request_kwargs(),
            )
        else:
            r = requests.request(
                method,
                url,
                data=self._serialize(payload),
                headers=self._headers(),
                **self._request_kwargs(),
            )

        try:
            r.raise_for_status()
        except exceptions.HTTPError as e:
            raise JudgmentAPIException(
                f"HTTP {r.status_code}: {r.reason}", response=r, request=e.request
            )

        return r.json()

    def send_spans_batch(self, spans: List[Dict[str, Any]]):
        payload: SpansBatchPayload = {
            "spans": spans,
            "organization_id": self.organization_id,
        }

        return self._do_request("POST", JUDGMENT_TRACES_SPANS_BATCH_API_URL, payload)

    def send_evaluation_runs_batch(
        self, evaluation_entries: List[EvaluationEntryResponse]
    ):
        payload: EvaluationRunsBatchPayload = {
            "organization_id": self.organization_id,
            "evaluation_entries": evaluation_entries,
        }

        return self._do_request(
            "POST", JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL, payload
        )

    def fetch_trace(self, trace_id: str):
        payload: TraceFetchPayload = {"trace_id": trace_id}
        return self._do_request("POST", JUDGMENT_TRACES_FETCH_API_URL, payload)

    def upsert_trace(self, trace_data: Dict[str, Any]):
        return self._do_request("POST", JUDGMENT_TRACES_UPSERT_API_URL, trace_data)

    def delete_trace(self, trace_id: str):
        payload: TraceDeletePayload = {"trace_ids": [trace_id]}
        return self._do_request("DELETE", JUDGMENT_TRACES_DELETE_API_URL, payload)

    def delete_traces(self, trace_ids: List[str]):
        payload: TraceDeletePayload = {"trace_ids": trace_ids}
        return self._do_request("DELETE", JUDGMENT_TRACES_DELETE_API_URL, payload)

    def delete_project(self, project_name: str):
        payload: ProjectDeletePayload = {"project_name": project_name}
        return self._do_request("DELETE", JUDGMENT_PROJECT_DELETE_API_URL, payload)

    def create_project(self, project_name: str):
        payload: ProjectCreatePayload = {"project_name": project_name}
        return self._do_request("POST", JUDGMENT_PROJECT_CREATE_API_URL, payload)

    def run_evaluation(self, evaluation_run: Dict[str, Any]):
        return self._do_request("POST", JUDGMENT_EVAL_API_URL, evaluation_run)

    def run_trace_evaluation(self, trace_run: Dict[str, Any]):
        return self._do_request("POST", JUDGMENT_TRACE_EVAL_API_URL, trace_run)

    def log_evaluation_results(
        self, results: List[Dict[str, Any]], run: Dict[str, Any]
    ):
        payload: EvalLogPayload = {"results": results, "run": run}
        return self._do_request("POST", JUDGMENT_EVAL_LOG_API_URL, payload)

    def fetch_evaluation_results(self, project_name: str, eval_name: str):
        payload: EvalRunRequestBody = {
            "project_name": project_name,
            "eval_name": eval_name,
            "judgment_api_key": self.api_key,
        }
        return self._do_request("POST", JUDGMENT_EVAL_FETCH_API_URL, payload)

    def delete_evaluation_results(self, project_name: str, eval_names: List[str]):
        payload: DeleteEvalRunRequestBody = {
            "project_name": project_name,
            "eval_names": eval_names,
            "judgment_api_key": self.api_key,
        }
        return self._do_request("POST", JUDGMENT_EVAL_DELETE_API_URL, payload)

    def add_to_evaluation_queue(self, payload: Dict[str, Any]):
        return self._do_request("POST", JUDGMENT_ADD_TO_RUN_EVAL_QUEUE_API_URL, payload)

    def get_evaluation_status(self, eval_name: str, project_name: str):
        payload: EvalStatusPayload = {
            "eval_name": eval_name,
            "project_name": project_name,
            "judgment_api_key": self.api_key,
        }
        return self._do_request("GET", JUDGMENT_GET_EVAL_STATUS_API_URL, payload)

    def check_experiment_type(self, eval_name: str, project_name: str, is_trace: bool):
        payload: CheckExperimentTypePayload = {
            "eval_name": eval_name,
            "project_name": project_name,
            "judgment_api_key": self.api_key,
            "is_trace": is_trace,
        }
        return self._do_request("POST", JUDGMENT_CHECK_EXPERIMENT_TYPE_API_URL, payload)

    def check_eval_run_name_exists(self, eval_name: str, project_name: str):
        payload: EvalRunNameExistsPayload = {
            "eval_name": eval_name,
            "project_name": project_name,
            "judgment_api_key": self.api_key,
        }
        return self._do_request("POST", JUDGMENT_EVAL_RUN_NAME_EXISTS_API_URL, payload)

    def save_scorer(self, name: str, prompt: str, options: dict):
        payload: ScorerSavePayload = {
            "name": name,
            "prompt": prompt,
            "options": options,
        }
        try:
            return self._do_request("POST", JUDGMENT_SCORER_SAVE_API_URL, payload)
        except JudgmentAPIException as e:
            if e.status_code == 500:
                raise JudgmentAPIException(
                    f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.error_detail}",
                    response=e.response,
                    request=e.request,
                )
            raise JudgmentAPIException(
                f"Failed to save classifier scorer: {e.error_detail}",
                response=e.response,
                request=e.request,
            )

    def fetch_scorer(self, name: str):
        payload: ScorerFetchPayload = {"name": name}
        try:
            return self._do_request("POST", JUDGMENT_SCORER_FETCH_API_URL, payload)
        except JudgmentAPIException as e:
            if e.status_code == 500:
                raise JudgmentAPIException(
                    f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.error_detail}",
                    response=e.response,
                    request=e.request,
                )
            raise JudgmentAPIException(
                f"Failed to fetch classifier scorer '{name}': {e.error_detail}",
                response=e.response,
                request=e.request,
            )

    def scorer_exists(self, name: str):
        payload: ScorerExistsPayload = {"name": name}
        try:
            return self._do_request("POST", JUDGMENT_SCORER_EXISTS_API_URL, payload)
        except JudgmentAPIException as e:
            if e.status_code == 500:
                raise JudgmentAPIException(
                    f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {e.error_detail}",
                    response=e.response,
                    request=e.request,
                )
            raise JudgmentAPIException(
                f"Failed to check if scorer exists: {e.error_detail}",
                response=e.response,
                request=e.request,
            )

    def push_dataset(
        self,
        dataset_alias: str,
        project_name: str,
        examples: List[Dict[str, Any]],
        traces: List[Dict[str, Any]],
        overwrite: bool,
    ):
        payload: DatasetPushPayload = {
            "dataset_alias": dataset_alias,
            "project_name": project_name,
            "examples": examples,
            "traces": traces,
            "overwrite": overwrite,
        }
        return self._do_request("POST", JUDGMENT_DATASETS_PUSH_API_URL, payload)

    def append_examples(
        self, dataset_alias: str, project_name: str, examples: List[Dict[str, Any]]
    ):
        payload: DatasetAppendPayload = {
            "dataset_alias": dataset_alias,
            "project_name": project_name,
            "examples": examples,
        }
        return self._do_request(
            "POST", JUDGMENT_DATASETS_APPEND_EXAMPLES_API_URL, payload
        )

    def pull_dataset(self, dataset_alias: str, project_name: str):
        payload: DatasetPullPayload = {
            "dataset_alias": dataset_alias,
            "project_name": project_name,
        }
        return self._do_request("POST", JUDGMENT_DATASETS_PULL_API_URL, payload)

    def delete_dataset(self, dataset_alias: str, project_name: str):
        payload: DatasetDeletePayload = {
            "dataset_alias": dataset_alias,
            "project_name": project_name,
        }
        return self._do_request("POST", JUDGMENT_DATASETS_DELETE_API_URL, payload)

    def get_project_dataset_stats(self, project_name: str):
        payload: DatasetStatsPayload = {"project_name": project_name}
        return self._do_request(
            "POST", JUDGMENT_DATASETS_PROJECT_STATS_API_URL, payload
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Organization-Id": self.organization_id,
        }

    def _request_kwargs(self):
        # NOTE: We may want to configure custom kwargs that different requests may need.
        # For this purpose we can store that as a property of self, and return the appropriate kwargs from this method.
        return {
            "verify": True,
            "timeout": 30,
        }

    def _serialize(self, data: Any) -> str:
        def fallback_encoder(obj):
            try:
                return repr(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception as e:
                    return f"<Unserializable object of type {type(obj).__name__}: {e}>"

        import json

        return json.dumps(data, default=fallback_encoder)
