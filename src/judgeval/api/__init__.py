from typing import Dict, Any, Mapping, Literal, Optional
import httpx
from httpx import Response
from judgeval.exceptions import JudgmentAPIError
from judgeval.utils.url import url_for
from judgeval.utils.serialize import json_encoder
from judgeval.api.api_types import *


def _headers(api_key: str, organization_id: str) -> Mapping[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Organization-Id": organization_id,
    }


def _handle_response(r: Response) -> Any:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text
        raise JudgmentAPIError(r.status_code, detail, r)
    return r.json()


class JudgmentSyncClient:
    __slots__ = ("api_key", "organization_id", "client")

    def __init__(self, api_key: str, organization_id: str):
        self.api_key = api_key
        self.organization_id = organization_id
        self.client = httpx.Client(timeout=30)

    def _request(
        self,
        method: Literal["POST", "PATCH", "GET", "DELETE"],
        url: str,
        payload: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if method == "GET":
            r = self.client.request(
                method,
                url,
                params=payload if params is None else params,
                headers=_headers(self.api_key, self.organization_id),
            )
        else:
            r = self.client.request(
                method,
                url,
                json=json_encoder(payload),
                params=params,
                headers=_headers(self.api_key, self.organization_id),
            )
        return _handle_response(r)

    def add_to_run_eval_queue_examples(self, payload: ExampleEvaluationRun) -> Any:
        return self._request(
            "POST",
            url_for("/add_to_run_eval_queue/examples"),
            payload,
        )

    def add_to_run_eval_queue_traces(self, payload: TraceEvaluationRun) -> Any:
        return self._request(
            "POST",
            url_for("/add_to_run_eval_queue/traces"),
            payload,
        )

    def evaluate_examples(
        self, payload: ExampleEvaluationRun, stream: Optional[str] = None
    ) -> Any:
        query_params = {}
        if stream is not None:
            query_params["stream"] = stream
        return self._request(
            "POST",
            url_for("/evaluate/examples"),
            payload,
            params=query_params,
        )

    def evaluate_traces(
        self, payload: TraceEvaluationRun, stream: Optional[str] = None
    ) -> Any:
        query_params = {}
        if stream is not None:
            query_params["stream"] = stream
        return self._request(
            "POST",
            url_for("/evaluate/traces"),
            payload,
            params=query_params,
        )

    def log_eval_results(self, payload: EvalResults) -> Any:
        return self._request(
            "POST",
            url_for("/log_eval_results/"),
            payload,
        )

    def fetch_experiment_run(self, payload: EvalResultsFetch) -> Any:
        return self._request(
            "POST",
            url_for("/fetch_experiment_run/"),
            payload,
        )

    def get_evaluation_status(self, experiment_run_id: str, project_name: str) -> Any:
        query_params = {}
        query_params["experiment_run_id"] = experiment_run_id
        query_params["project_name"] = project_name
        return self._request(
            "GET",
            url_for("/get_evaluation_status/"),
            query_params,
        )

    def datasets_insert_examples_for_judgeval(
        self, payload: DatasetInsertExamples
    ) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/insert_examples_for_judgeval/"),
            payload,
        )

    def datasets_pull_for_judgeval(self, payload: DatasetFetch) -> DatasetReturn:
        return self._request(
            "POST",
            url_for("/datasets/pull_for_judgeval/"),
            payload,
        )

    def datasets_create_for_judgeval(self, payload: DatasetCreate) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/create_for_judgeval/"),
            payload,
        )

    def projects_add(self, payload: ProjectAdd) -> ProjectAddResponse:
        return self._request(
            "POST",
            url_for("/projects/add/"),
            payload,
        )

    def projects_delete_from_judgeval(
        self, payload: ProjectDeleteFromJudgevalResponse
    ) -> ProjectDeleteResponse:
        return self._request(
            "DELETE",
            url_for("/projects/delete_from_judgeval/"),
            payload,
        )

    def scorer_exists(self, payload: ScorerExistsRequest) -> ScorerExistsResponse:
        return self._request(
            "POST",
            url_for("/scorer_exists/"),
            payload,
        )

    def save_scorer(self, payload: SavePromptScorerRequest) -> SavePromptScorerResponse:
        return self._request(
            "POST",
            url_for("/save_scorer/"),
            payload,
        )

    def fetch_scorer(
        self, payload: FetchPromptScorerRequest
    ) -> FetchPromptScorerResponse:
        return self._request(
            "POST",
            url_for("/fetch_scorer/"),
            payload,
        )

    def upload_custom_scorer(
        self, payload: CustomScorerUploadPayload
    ) -> CustomScorerTemplateResponse:
        return self._request(
            "POST",
            url_for("/upload_custom_scorer/"),
            payload,
        )

    def projects_resolve(
        self, payload: ResolveProjectNameRequest
    ) -> ResolveProjectNameResponse:
        return self._request(
            "POST",
            url_for("/projects/resolve/"),
            payload,
        )

    def e2e_fetch_trace(self, payload: TraceIdRequest) -> Any:
        return self._request(
            "POST",
            url_for("/e2e_fetch_trace/"),
            payload,
        )

    def e2e_fetch_span_score(self, payload: SpanScoreRequest) -> Any:
        return self._request(
            "POST",
            url_for("/e2e_fetch_span_score/"),
            payload,
        )

    def e2e_fetch_trace_scorer_span_score(self, payload: SpanScoreRequest) -> Any:
        return self._request(
            "POST",
            url_for("/e2e_fetch_trace_scorer_span_score/"),
            payload,
        )


class JudgmentAsyncClient:
    __slots__ = ("api_key", "organization_id", "client")

    def __init__(self, api_key: str, organization_id: str):
        self.api_key = api_key
        self.organization_id = organization_id
        self.client = httpx.AsyncClient(timeout=30)

    async def _request(
        self,
        method: Literal["POST", "PATCH", "GET", "DELETE"],
        url: str,
        payload: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if method == "GET":
            r = self.client.request(
                method,
                url,
                params=payload if params is None else params,
                headers=_headers(self.api_key, self.organization_id),
            )
        else:
            r = self.client.request(
                method,
                url,
                json=json_encoder(payload),
                params=params,
                headers=_headers(self.api_key, self.organization_id),
            )
        return _handle_response(await r)

    async def add_to_run_eval_queue_examples(
        self, payload: ExampleEvaluationRun
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/add_to_run_eval_queue/examples"),
            payload,
        )

    async def add_to_run_eval_queue_traces(self, payload: TraceEvaluationRun) -> Any:
        return await self._request(
            "POST",
            url_for("/add_to_run_eval_queue/traces"),
            payload,
        )

    async def evaluate_examples(
        self, payload: ExampleEvaluationRun, stream: Optional[str] = None
    ) -> Any:
        query_params = {}
        if stream is not None:
            query_params["stream"] = stream
        return await self._request(
            "POST",
            url_for("/evaluate/examples"),
            payload,
            params=query_params,
        )

    async def evaluate_traces(
        self, payload: TraceEvaluationRun, stream: Optional[str] = None
    ) -> Any:
        query_params = {}
        if stream is not None:
            query_params["stream"] = stream
        return await self._request(
            "POST",
            url_for("/evaluate/traces"),
            payload,
            params=query_params,
        )

    async def log_eval_results(self, payload: EvalResults) -> Any:
        return await self._request(
            "POST",
            url_for("/log_eval_results/"),
            payload,
        )

    async def fetch_experiment_run(self, payload: EvalResultsFetch) -> Any:
        return await self._request(
            "POST",
            url_for("/fetch_experiment_run/"),
            payload,
        )

    async def get_evaluation_status(
        self, experiment_run_id: str, project_name: str
    ) -> Any:
        query_params = {}
        query_params["experiment_run_id"] = experiment_run_id
        query_params["project_name"] = project_name
        return await self._request(
            "GET",
            url_for("/get_evaluation_status/"),
            query_params,
        )

    async def datasets_insert_examples_for_judgeval(
        self, payload: DatasetInsertExamples
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/insert_examples_for_judgeval/"),
            payload,
        )

    async def datasets_pull_for_judgeval(self, payload: DatasetFetch) -> DatasetReturn:
        return await self._request(
            "POST",
            url_for("/datasets/pull_for_judgeval/"),
            payload,
        )

    async def datasets_create_for_judgeval(self, payload: DatasetCreate) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/create_for_judgeval/"),
            payload,
        )

    async def projects_add(self, payload: ProjectAdd) -> ProjectAddResponse:
        return await self._request(
            "POST",
            url_for("/projects/add/"),
            payload,
        )

    async def projects_delete_from_judgeval(
        self, payload: ProjectDeleteFromJudgevalResponse
    ) -> ProjectDeleteResponse:
        return await self._request(
            "DELETE",
            url_for("/projects/delete_from_judgeval/"),
            payload,
        )

    async def scorer_exists(self, payload: ScorerExistsRequest) -> ScorerExistsResponse:
        return await self._request(
            "POST",
            url_for("/scorer_exists/"),
            payload,
        )

    async def save_scorer(
        self, payload: SavePromptScorerRequest
    ) -> SavePromptScorerResponse:
        return await self._request(
            "POST",
            url_for("/save_scorer/"),
            payload,
        )

    async def fetch_scorer(
        self, payload: FetchPromptScorerRequest
    ) -> FetchPromptScorerResponse:
        return await self._request(
            "POST",
            url_for("/fetch_scorer/"),
            payload,
        )

    async def upload_custom_scorer(
        self, payload: CustomScorerUploadPayload
    ) -> CustomScorerTemplateResponse:
        return await self._request(
            "POST",
            url_for("/upload_custom_scorer/"),
            payload,
        )

    async def projects_resolve(
        self, payload: ResolveProjectNameRequest
    ) -> ResolveProjectNameResponse:
        return await self._request(
            "POST",
            url_for("/projects/resolve/"),
            payload,
        )

    async def e2e_fetch_trace(self, payload: TraceIdRequest) -> Any:
        return await self._request(
            "POST",
            url_for("/e2e_fetch_trace/"),
            payload,
        )

    async def e2e_fetch_span_score(self, payload: SpanScoreRequest) -> Any:
        return await self._request(
            "POST",
            url_for("/e2e_fetch_span_score/"),
            payload,
        )

    async def e2e_fetch_trace_scorer_span_score(self, payload: SpanScoreRequest) -> Any:
        return await self._request(
            "POST",
            url_for("/e2e_fetch_trace_scorer_span_score/"),
            payload,
        )


__all__ = [
    "JudgmentSyncClient",
    "JudgmentAsyncClient",
]
