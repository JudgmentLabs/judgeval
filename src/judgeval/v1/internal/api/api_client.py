from typing import Dict, Any, Mapping, Literal, Optional
import httpx
from httpx import Response
from judgeval.exceptions import JudgmentAPIError
from judgeval.utils.url import url_for
from judgeval.utils.serialize import json_encoder
from judgeval.v1.internal.api.api_types import *


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
    __slots__ = ("base_url", "api_key", "organization_id", "client")

    def __init__(self, base_url: str, api_key: str, organization_id: str):
        self.base_url = base_url
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

    def post_otel_v1_traces(self) -> Any:
        return self._request(
            "POST",
            url_for("/otel/v1/traces", self.base_url),
            {},
        )

    def post_otel_trigger_root_span_rules(
        self, payload: TriggerRootSpanRulesRequest
    ) -> TriggerRootSpanRulesResponse:
        return self._request(
            "POST",
            url_for("/otel/trigger_root_span_rules", self.base_url),
            payload,
        )

    def post_projects_resolve(
        self, payload: ResolveProjectRequest
    ) -> ResolveProjectResponse:
        return self._request(
            "POST",
            url_for("/v1/projects/resolve/", self.base_url),
            payload,
        )

    def post_projects(self, payload: AddProjectRequest) -> AddProjectResponse:
        return self._request(
            "POST",
            url_for("/v1/projects", self.base_url),
            payload,
        )

    def delete_projects(self, project_id: str) -> DeleteProjectResponse:
        return self._request(
            "DELETE",
            url_for(f"/v1/projects/{project_id}", self.base_url),
            {},
        )

    def post_projects_datasets(
        self, project_id: str, payload: CreateDatasetRequest
    ) -> CreateDatasetResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/datasets", self.base_url),
            payload,
        )

    def get_projects_datasets(self, project_id: str) -> PullAllDatasetsResponse:
        return self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/datasets", self.base_url),
            {},
        )

    def post_projects_datasets_by_dataset_name_examples(
        self, project_id: str, dataset_name: str, payload: InsertExamplesRequest
    ) -> InsertExamplesResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{project_id}/datasets/{dataset_name}/examples",
                self.base_url,
            ),
            payload,
        )

    def get_projects_datasets_by_dataset_name(
        self, project_id: str, dataset_name: str
    ) -> PullDatasetResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{project_id}/datasets/{dataset_name}", self.base_url
            ),
            {},
        )

    def post_projects_evaluate_examples(
        self, project_id: str, payload: ExampleEvaluationRun
    ) -> Any:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/evaluate/examples", self.base_url),
            payload,
        )

    def post_projects_evaluate_traces(
        self, project_id: str, payload: TraceEvaluationRun
    ) -> Any:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/evaluate/traces", self.base_url),
            payload,
        )

    def post_projects_eval_results(
        self, project_id: str, payload: LogEvalResultsRequest
    ) -> LogEvalResultsResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/eval-results", self.base_url),
            payload,
        )

    def get_projects_experiments_by_run_id(
        self, project_id: str, run_id: str
    ) -> FetchExperimentRunResponse:
        return self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/experiments/{run_id}", self.base_url),
            {},
        )

    def post_projects_eval_queue_examples(
        self, project_id: str, payload: ExampleEvaluationRun
    ) -> AddToRunEvalQueueExamplesResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/eval-queue/examples", self.base_url),
            payload,
        )

    def post_projects_eval_queue_traces(
        self, project_id: str, payload: TraceEvaluationRun
    ) -> AddToRunEvalQueueTracesResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/eval-queue/traces", self.base_url),
            payload,
        )

    def get_projects_prompts_by_name(
        self,
        project_id: str,
        name: str,
        commit_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> FetchPromptResponse:
        query_params = {}
        if commit_id is not None:
            query_params["commit_id"] = commit_id
        if tag is not None:
            query_params["tag"] = tag
        return self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/prompts/{name}", self.base_url),
            query_params,
        )

    def post_projects_prompts(
        self, project_id: str, payload: InsertPromptRequest
    ) -> InsertPromptResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/prompts", self.base_url),
            payload,
        )

    def post_projects_prompts_by_name_tags(
        self, project_id: str, name: str, payload: TagPromptRequest
    ) -> TagPromptResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/prompts/{name}/tags", self.base_url),
            payload,
        )

    def delete_projects_prompts_by_name_tags(
        self, project_id: str, name: str, payload: UntagPromptRequest
    ) -> UntagPromptResponse:
        return self._request(
            "DELETE",
            url_for(f"/v1/projects/{project_id}/prompts/{name}/tags", self.base_url),
            payload,
        )

    def get_projects_prompts_by_name_versions(
        self, project_id: str, name: str
    ) -> GetPromptVersionsResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{project_id}/prompts/{name}/versions", self.base_url
            ),
            {},
        )

    def get_projects_scorers(
        self,
        project_id: str,
        names: Optional[str] = None,
        is_trace: Optional[str] = None,
    ) -> FetchPromptScorersResponse:
        query_params = {}
        if names is not None:
            query_params["names"] = names
        if is_trace is not None:
            query_params["is_trace"] = is_trace
        return self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/scorers", self.base_url),
            query_params,
        )

    def post_projects_scorers(
        self, project_id: str, payload: SavePromptScorerRequest
    ) -> SavePromptScorerResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/scorers", self.base_url),
            payload,
        )

    def get_projects_scorers_by_name_exists(
        self, project_id: str, name: str
    ) -> ScorerExistsResponse:
        return self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/scorers/{name}/exists", self.base_url),
            {},
        )

    def post_projects_scorers_custom(
        self, project_id: str, payload: UploadCustomScorerRequest
    ) -> UploadCustomScorerResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/scorers/custom", self.base_url),
            payload,
        )

    def get_projects_scorers_custom_by_name_exists(
        self, project_id: str, name: str
    ) -> CustomScorerExistsResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{project_id}/scorers/custom/{name}/exists", self.base_url
            ),
            {},
        )

    def post_projects_traces_by_trace_id_tags(
        self, project_id: str, trace_id: str, payload: AddTraceTagsRequest
    ) -> AddTraceTagsResponse:
        return self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/traces/{trace_id}/tags", self.base_url),
            payload,
        )

    def post_e2e_fetch_trace(
        self, payload: E2EFetchTraceRequest
    ) -> E2EFetchTraceResponse:
        return self._request(
            "POST",
            url_for("/v1/e2e_fetch_trace/", self.base_url),
            payload,
        )

    def post_e2e_fetch_span_score(
        self, payload: E2EFetchSpanScoreRequest
    ) -> E2EFetchSpanScoreResponse:
        return self._request(
            "POST",
            url_for("/v1/e2e_fetch_span_score/", self.base_url),
            payload,
        )


class JudgmentAsyncClient:
    __slots__ = ("base_url", "api_key", "organization_id", "client")

    def __init__(self, base_url: str, api_key: str, organization_id: str):
        self.base_url = base_url
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

    async def post_otel_v1_traces(self) -> Any:
        return await self._request(
            "POST",
            url_for("/otel/v1/traces", self.base_url),
            {},
        )

    async def post_otel_trigger_root_span_rules(
        self, payload: TriggerRootSpanRulesRequest
    ) -> TriggerRootSpanRulesResponse:
        return await self._request(
            "POST",
            url_for("/otel/trigger_root_span_rules", self.base_url),
            payload,
        )

    async def post_projects_resolve(
        self, payload: ResolveProjectRequest
    ) -> ResolveProjectResponse:
        return await self._request(
            "POST",
            url_for("/v1/projects/resolve/", self.base_url),
            payload,
        )

    async def post_projects(self, payload: AddProjectRequest) -> AddProjectResponse:
        return await self._request(
            "POST",
            url_for("/v1/projects", self.base_url),
            payload,
        )

    async def delete_projects(self, project_id: str) -> DeleteProjectResponse:
        return await self._request(
            "DELETE",
            url_for(f"/v1/projects/{project_id}", self.base_url),
            {},
        )

    async def post_projects_datasets(
        self, project_id: str, payload: CreateDatasetRequest
    ) -> CreateDatasetResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/datasets", self.base_url),
            payload,
        )

    async def get_projects_datasets(self, project_id: str) -> PullAllDatasetsResponse:
        return await self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/datasets", self.base_url),
            {},
        )

    async def post_projects_datasets_by_dataset_name_examples(
        self, project_id: str, dataset_name: str, payload: InsertExamplesRequest
    ) -> InsertExamplesResponse:
        return await self._request(
            "POST",
            url_for(
                f"/v1/projects/{project_id}/datasets/{dataset_name}/examples",
                self.base_url,
            ),
            payload,
        )

    async def get_projects_datasets_by_dataset_name(
        self, project_id: str, dataset_name: str
    ) -> PullDatasetResponse:
        return await self._request(
            "GET",
            url_for(
                f"/v1/projects/{project_id}/datasets/{dataset_name}", self.base_url
            ),
            {},
        )

    async def post_projects_evaluate_examples(
        self, project_id: str, payload: ExampleEvaluationRun
    ) -> Any:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/evaluate/examples", self.base_url),
            payload,
        )

    async def post_projects_evaluate_traces(
        self, project_id: str, payload: TraceEvaluationRun
    ) -> Any:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/evaluate/traces", self.base_url),
            payload,
        )

    async def post_projects_eval_results(
        self, project_id: str, payload: LogEvalResultsRequest
    ) -> LogEvalResultsResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/eval-results", self.base_url),
            payload,
        )

    async def get_projects_experiments_by_run_id(
        self, project_id: str, run_id: str
    ) -> FetchExperimentRunResponse:
        return await self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/experiments/{run_id}", self.base_url),
            {},
        )

    async def post_projects_eval_queue_examples(
        self, project_id: str, payload: ExampleEvaluationRun
    ) -> AddToRunEvalQueueExamplesResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/eval-queue/examples", self.base_url),
            payload,
        )

    async def post_projects_eval_queue_traces(
        self, project_id: str, payload: TraceEvaluationRun
    ) -> AddToRunEvalQueueTracesResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/eval-queue/traces", self.base_url),
            payload,
        )

    async def get_projects_prompts_by_name(
        self,
        project_id: str,
        name: str,
        commit_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> FetchPromptResponse:
        query_params = {}
        if commit_id is not None:
            query_params["commit_id"] = commit_id
        if tag is not None:
            query_params["tag"] = tag
        return await self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/prompts/{name}", self.base_url),
            query_params,
        )

    async def post_projects_prompts(
        self, project_id: str, payload: InsertPromptRequest
    ) -> InsertPromptResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/prompts", self.base_url),
            payload,
        )

    async def post_projects_prompts_by_name_tags(
        self, project_id: str, name: str, payload: TagPromptRequest
    ) -> TagPromptResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/prompts/{name}/tags", self.base_url),
            payload,
        )

    async def delete_projects_prompts_by_name_tags(
        self, project_id: str, name: str, payload: UntagPromptRequest
    ) -> UntagPromptResponse:
        return await self._request(
            "DELETE",
            url_for(f"/v1/projects/{project_id}/prompts/{name}/tags", self.base_url),
            payload,
        )

    async def get_projects_prompts_by_name_versions(
        self, project_id: str, name: str
    ) -> GetPromptVersionsResponse:
        return await self._request(
            "GET",
            url_for(
                f"/v1/projects/{project_id}/prompts/{name}/versions", self.base_url
            ),
            {},
        )

    async def get_projects_scorers(
        self,
        project_id: str,
        names: Optional[str] = None,
        is_trace: Optional[str] = None,
    ) -> FetchPromptScorersResponse:
        query_params = {}
        if names is not None:
            query_params["names"] = names
        if is_trace is not None:
            query_params["is_trace"] = is_trace
        return await self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/scorers", self.base_url),
            query_params,
        )

    async def post_projects_scorers(
        self, project_id: str, payload: SavePromptScorerRequest
    ) -> SavePromptScorerResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/scorers", self.base_url),
            payload,
        )

    async def get_projects_scorers_by_name_exists(
        self, project_id: str, name: str
    ) -> ScorerExistsResponse:
        return await self._request(
            "GET",
            url_for(f"/v1/projects/{project_id}/scorers/{name}/exists", self.base_url),
            {},
        )

    async def post_projects_scorers_custom(
        self, project_id: str, payload: UploadCustomScorerRequest
    ) -> UploadCustomScorerResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/scorers/custom", self.base_url),
            payload,
        )

    async def get_projects_scorers_custom_by_name_exists(
        self, project_id: str, name: str
    ) -> CustomScorerExistsResponse:
        return await self._request(
            "GET",
            url_for(
                f"/v1/projects/{project_id}/scorers/custom/{name}/exists", self.base_url
            ),
            {},
        )

    async def post_projects_traces_by_trace_id_tags(
        self, project_id: str, trace_id: str, payload: AddTraceTagsRequest
    ) -> AddTraceTagsResponse:
        return await self._request(
            "POST",
            url_for(f"/v1/projects/{project_id}/traces/{trace_id}/tags", self.base_url),
            payload,
        )

    async def post_e2e_fetch_trace(
        self, payload: E2EFetchTraceRequest
    ) -> E2EFetchTraceResponse:
        return await self._request(
            "POST",
            url_for("/v1/e2e_fetch_trace/", self.base_url),
            payload,
        )

    async def post_e2e_fetch_span_score(
        self, payload: E2EFetchSpanScoreRequest
    ) -> E2EFetchSpanScoreResponse:
        return await self._request(
            "POST",
            url_for("/v1/e2e_fetch_span_score/", self.base_url),
            payload,
        )


__all__ = [
    "JudgmentSyncClient",
    "JudgmentAsyncClient",
]
