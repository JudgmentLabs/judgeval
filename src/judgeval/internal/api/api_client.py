# Client for the /v1 routes the SDK calls. Models are generated from the
# committed spec (scripts/generate_api_models.py); tests/test_api_routes.py
# checks every route here exists in it.
# mypy: ignore-errors
from typing import Dict, Any, Mapping, Literal, Optional
import json
import httpx
from httpx import Response
from judgeval.exceptions import JudgmentAPIError
from judgeval.utils.url import encode_path_segment, url_for
from judgeval.utils.serialize import json_encoder
from judgeval.internal.api.models import *
from judgeval.logger import judgeval_logger as logger


def _headers(
    api_key: str, organization_id: str, is_multipart: bool = False
) -> Mapping[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Organization-Id": organization_id,
    }
    if not is_multipart:
        headers["Content-Type"] = "application/json"
    return headers


def _handle_response(r: Response) -> Any:
    if r.status_code >= 400:
        try:
            body = r.json()
            detail = (
                body.get("detail") or body.get("message") or body.get("error") or ""
            )
            # The error code is also the readable fallback when detail/message are absent.
            code = body.get("error")
            hint = body.get("hint")
        except Exception:
            detail = r.text
            code = None
            hint = None
        retry_after = r.headers.get("Retry-After")
        try:
            retry_after_seconds = int(retry_after) if retry_after else None
        except ValueError:
            retry_after_seconds = None
        raise JudgmentAPIError(
            r.status_code,
            detail,
            r,
            code=code,
            hint=hint,
            retry_after_seconds=retry_after_seconds,
        )
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
        method: Literal["POST", "PATCH", "PUT", "GET", "DELETE"],
        url: str,
        payload: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        logger.debug(f"HTTP {method} {url}")
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
        logger.debug(f"HTTP {method} {url} -> {r.status_code}")
        return _handle_response(r)

    def _multipart_request(
        self,
        method: Literal["POST", "PATCH", "PUT"],
        url: str,
        payload: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        logger.debug(f"HTTP {method} (multipart) {url}")
        data = {}
        files = {}
        for key, value in payload.items():
            if isinstance(value, (bytes, bytearray)):
                files[key] = (key, value, "application/octet-stream")
            else:
                if isinstance(value, dict):
                    data[key] = json.dumps(value)
                else:
                    data[key] = value
        r = self.client.request(
            method,
            url,
            data=data,
            files=files,
            params=params,
            headers=_headers(self.api_key, self.organization_id, True),
        )
        logger.debug(f"HTTP {method} (multipart) {url} -> {r.status_code}")
        return _handle_response(r)

    def post_projects(self, payload: AddProjectRequest) -> AddProjectResponse:
        return self._request(
            "POST",
            url_for("/v1/projects", self.base_url),
            payload,
        )

    def delete_projects(self, project_id: str) -> DeleteProjectResponse:
        return self._request(
            "DELETE",
            url_for(f"/v1/projects/{encode_path_segment(project_id)}", self.base_url),
            {},
        )

    def post_projects_resolve(
        self, payload: ResolveProjectRequest
    ) -> ResolveProjectResponse:
        return self._request(
            "POST",
            url_for("/v1/projects/resolve/", self.base_url),
            payload,
        )

    def post_projects_datasets(
        self, project_id: str, payload: CreateOfflineDatasetRequest
    ) -> CreateOfflineDatasetResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets",
                self.base_url,
            ),
            payload,
        )

    def get_projects_datasets(self, project_id: str) -> PullAllOfflineDatasetsResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets",
                self.base_url,
            ),
            {},
        )

    def get_projects_datasets_by_dataset_identifier_versions(
        self, project_id: str, dataset_identifier: str
    ) -> OfflineDatasetVersionsResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets/{encode_path_segment(dataset_identifier)}/versions",
                self.base_url,
            ),
            {},
        )

    def get_projects_datasets_by_dataset_identifier_page(
        self,
        project_id: str,
        dataset_identifier: str,
        version: Optional[str] = None,
        limit: Optional[str] = None,
        cursor_created_at: Optional[str] = None,
        cursor_example_id: Optional[str] = None,
    ) -> OfflineDatasetPageResponse:
        query_params = {}
        if version is not None:
            query_params["version"] = version
        if limit is not None:
            query_params["limit"] = limit
        if cursor_created_at is not None:
            query_params["cursor_created_at"] = cursor_created_at
        if cursor_example_id is not None:
            query_params["cursor_example_id"] = cursor_example_id
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets/{encode_path_segment(dataset_identifier)}/page",
                self.base_url,
            ),
            query_params,
        )

    def post_projects_datasets_by_dataset_identifier_examples(
        self,
        project_id: str,
        dataset_identifier: str,
        payload: InsertOfflineDatasetExamplesRequest,
    ) -> AddDatasetItemsResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets/{encode_path_segment(dataset_identifier)}/examples",
                self.base_url,
            ),
            payload,
        )

    def delete_projects_datasets_by_dataset_identifier(
        self, project_id: str, dataset_identifier: str
    ) -> DeleteOfflineDatasetResponse:
        return self._request(
            "DELETE",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets/{encode_path_segment(dataset_identifier)}",
                self.base_url,
            ),
            {},
        )

    def get_projects_datasets_by_dataset_identifier(
        self, project_id: str, dataset_identifier: str
    ) -> PullOfflineDatasetResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/datasets/{encode_path_segment(dataset_identifier)}",
                self.base_url,
            ),
            {},
        )

    def get_projects_test_configs(
        self, project_id: str, dataset_id: Optional[str] = None
    ) -> TestConfigsResponse:
        query_params = {}
        if dataset_id is not None:
            query_params["dataset_id"] = dataset_id
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/test-configs",
                self.base_url,
            ),
            query_params,
        )

    def post_projects_test_configs(
        self, project_id: str, payload: CreateTestConfigRequest
    ) -> TestConfigResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/test-configs",
                self.base_url,
            ),
            payload,
        )

    def get_projects_test_configs_by_test_config_id(
        self, project_id: str, test_config_id: str
    ) -> TestConfigResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/test-configs/{encode_path_segment(test_config_id)}",
                self.base_url,
            ),
            {},
        )

    def delete_projects_test_configs_by_test_config_id(
        self, project_id: str, test_config_id: str
    ) -> DeleteTestConfigResponse:
        return self._request(
            "DELETE",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/test-configs/{encode_path_segment(test_config_id)}",
                self.base_url,
            ),
            {},
        )

    def post_projects_test_runs(
        self, project_id: str, payload: CreateTestRunRequest
    ) -> PreparedTestRunResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/test-runs",
                self.base_url,
            ),
            payload,
        )

    def get_projects_test_runs_by_test_run_id(
        self, project_id: str, test_run_id: str
    ) -> TestRunResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/test-runs/{encode_path_segment(test_run_id)}",
                self.base_url,
            ),
            {},
        )

    def get_projects_experiments_by_run_id(
        self,
        project_id: str,
        run_id: str,
        limit: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> FetchTestRunViaExperimentAliasResponse:
        query_params = {}
        if limit is not None:
            query_params["limit"] = limit
        if cursor is not None:
            query_params["cursor"] = cursor
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/experiments/{encode_path_segment(run_id)}",
                self.base_url,
            ),
            query_params,
        )

    def post_projects_eval_results_examples(
        self, project_id: str, payload: OfflineTestsLogEvalResultsExamplesRequest
    ) -> OfflineTestsLogEvalResultsExamplesResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/eval-results/examples",
                self.base_url,
            ),
            payload,
        )

    def post_projects_eval_queue_examples(
        self, project_id: str, payload: ExampleEvaluationRun
    ) -> AddToRunEvalQueueExamplesResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/eval-queue/examples",
                self.base_url,
            ),
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
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/prompts/{encode_path_segment(name)}",
                self.base_url,
            ),
            query_params,
        )

    def post_projects_prompts(
        self, project_id: str, payload: InsertPromptRequest
    ) -> InsertPromptResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/prompts", self.base_url
            ),
            payload,
        )

    def post_projects_prompts_by_name_tags(
        self, project_id: str, name: str, payload: TagPromptRequest
    ) -> TagPromptResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/prompts/{encode_path_segment(name)}/tags",
                self.base_url,
            ),
            payload,
        )

    def delete_projects_prompts_by_name_tags(
        self, project_id: str, name: str, payload: UntagPromptRequest
    ) -> UntagPromptResponse:
        return self._request(
            "DELETE",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/prompts/{encode_path_segment(name)}/tags",
                self.base_url,
            ),
            payload,
        )

    def get_projects_prompts_by_name_versions(
        self, project_id: str, name: str
    ) -> GetPromptVersionsResponse:
        return self._request(
            "GET",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/prompts/{encode_path_segment(name)}/versions",
                self.base_url,
            ),
            {},
        )

    def post_projects_scorers_custom_bundle(
        self, project_id: str, payload: UploadCustomScorerBundleRequest
    ) -> UploadCustomScorerResponse:
        return self._multipart_request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/scorers/custom/bundle",
                self.base_url,
            ),
            payload,
        )

    def post_projects_traces_by_trace_id_tags(
        self, project_id: str, trace_id: str, payload: AddTraceTagsRequest
    ) -> AddTraceTagsResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/traces/{encode_path_segment(trace_id)}/tags",
                self.base_url,
            ),
            payload,
        )

    def post_projects_judges(
        self, project_id: str, payload: CreateJudgeRequest
    ) -> CreateJudgeResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/judges", self.base_url
            ),
            payload,
        )

    def patch_projects_judges_by_judge_id(
        self, project_id: str, judge_id: str, payload: UpdateJudgeRequest
    ) -> UpdateJudgeResponse:
        return self._request(
            "PATCH",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/judges/{encode_path_segment(judge_id)}",
                self.base_url,
            ),
            payload,
        )

    def post_projects_judges_by_judge_id_lookup(
        self, project_id: str, judge_id: str, payload: GetJudgeRequest
    ) -> GetJudgeResponse:
        return self._request(
            "POST",
            url_for(
                f"/v1/projects/{encode_path_segment(project_id)}/judges/{encode_path_segment(judge_id)}/lookup",
                self.base_url,
            ),
            payload,
        )


__all__ = [
    "JudgmentSyncClient",
]
