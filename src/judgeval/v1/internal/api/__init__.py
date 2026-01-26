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

    def index(self) -> Any:
        return self._request(
            "GET",
            url_for("/", self.base_url),
            {},
        )

    def health(self) -> Any:
        return self._request(
            "GET",
            url_for("/health", self.base_url),
            {},
        )

    def otel_v1_traces(self) -> Any:
        return self._request(
            "POST",
            url_for("/otel/v1/traces", self.base_url),
            {},
        )

    def otel_trigger_root_span_rules(self) -> Any:
        return self._request(
            "POST",
            url_for("/otel/trigger_root_span_rules", self.base_url),
            {},
        )

    def projects_resolve(self) -> Any:
        return self._request(
            "POST",
            url_for("/projects/resolve/", self.base_url),
            {},
        )

    def projects_add(self) -> Any:
        return self._request(
            "POST",
            url_for("/projects/add/", self.base_url),
            {},
        )

    def projects_delete_from_judgeval(self) -> Any:
        return self._request(
            "DELETE",
            url_for("/projects/delete_from_judgeval/", self.base_url),
            {},
        )

    def datasets_create_for_judgeval(self) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/create_for_judgeval/", self.base_url),
            {},
        )

    def datasets_insert_examples_for_judgeval(self) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/insert_examples_for_judgeval/", self.base_url),
            {},
        )

    def datasets_pull_for_judgeval(self) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/pull_for_judgeval/", self.base_url),
            {},
        )

    def datasets_pull_all_for_judgeval(self) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/pull_all_for_judgeval/", self.base_url),
            {},
        )

    def evaluate_examples(self) -> Any:
        return self._request(
            "POST",
            url_for("/evaluate/examples", self.base_url),
            {},
        )

    def evaluate_traces(self) -> Any:
        return self._request(
            "POST",
            url_for("/evaluate/traces", self.base_url),
            {},
        )

    def log_eval_results(self) -> Any:
        return self._request(
            "POST",
            url_for("/log_eval_results/", self.base_url),
            {},
        )

    def fetch_experiment_run(self) -> Any:
        return self._request(
            "POST",
            url_for("/fetch_experiment_run/", self.base_url),
            {},
        )

    def add_to_run_eval_queue(self) -> Any:
        return self._request(
            "POST",
            url_for("/add_to_run_eval_queue/", self.base_url),
            {},
        )

    def add_to_run_eval_queue_examples(self) -> Any:
        return self._request(
            "POST",
            url_for("/add_to_run_eval_queue/examples", self.base_url),
            {},
        )

    def add_to_run_eval_queue_traces(self) -> Any:
        return self._request(
            "POST",
            url_for("/add_to_run_eval_queue/traces", self.base_url),
            {},
        )

    def prompts_fetch(
        self,
        project_id: str,
        name: str,
        commit_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Any:
        query_params = {}
        query_params["project_id"] = project_id
        query_params["name"] = name
        if commit_id is not None:
            query_params["commit_id"] = commit_id
        if tag is not None:
            query_params["tag"] = tag
        return self._request(
            "GET",
            url_for("/prompts/fetch", self.base_url),
            query_params,
        )

    def prompts_insert(self) -> Any:
        return self._request(
            "POST",
            url_for("/prompts/insert", self.base_url),
            {},
        )

    def prompts_tag(self) -> Any:
        return self._request(
            "POST",
            url_for("/prompts/tag", self.base_url),
            {},
        )

    def prompts_untag(self) -> Any:
        return self._request(
            "POST",
            url_for("/prompts/untag", self.base_url),
            {},
        )

    def prompts_get_prompt_versions(self, project_id: str, name: str) -> Any:
        query_params = {}
        query_params["project_id"] = project_id
        query_params["name"] = name
        return self._request(
            "GET",
            url_for("/prompts/get_prompt_versions", self.base_url),
            query_params,
        )

    def fetch_scorers(self) -> Any:
        return self._request(
            "POST",
            url_for("/fetch_scorers", self.base_url),
            {},
        )

    def save_scorer(self) -> Any:
        return self._request(
            "POST",
            url_for("/save_scorer", self.base_url),
            {},
        )

    def scorer_exists(self) -> Any:
        return self._request(
            "POST",
            url_for("/scorer_exists", self.base_url),
            {},
        )

    def upload_custom_scorer(self) -> Any:
        return self._request(
            "POST",
            url_for("/upload_custom_scorer/", self.base_url),
            {},
        )

    def traces_tags_add(self) -> Any:
        return self._request(
            "POST",
            url_for("/traces/tags/add", self.base_url),
            {},
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

    async def index(self) -> Any:
        return await self._request(
            "GET",
            url_for("/", self.base_url),
            {},
        )

    async def health(self) -> Any:
        return await self._request(
            "GET",
            url_for("/health", self.base_url),
            {},
        )

    async def otel_v1_traces(self) -> Any:
        return await self._request(
            "POST",
            url_for("/otel/v1/traces", self.base_url),
            {},
        )

    async def otel_trigger_root_span_rules(self) -> Any:
        return await self._request(
            "POST",
            url_for("/otel/trigger_root_span_rules", self.base_url),
            {},
        )

    async def projects_resolve(self) -> Any:
        return await self._request(
            "POST",
            url_for("/projects/resolve/", self.base_url),
            {},
        )

    async def projects_add(self) -> Any:
        return await self._request(
            "POST",
            url_for("/projects/add/", self.base_url),
            {},
        )

    async def projects_delete_from_judgeval(self) -> Any:
        return await self._request(
            "DELETE",
            url_for("/projects/delete_from_judgeval/", self.base_url),
            {},
        )

    async def datasets_create_for_judgeval(self) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/create_for_judgeval/", self.base_url),
            {},
        )

    async def datasets_insert_examples_for_judgeval(self) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/insert_examples_for_judgeval/", self.base_url),
            {},
        )

    async def datasets_pull_for_judgeval(self) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/pull_for_judgeval/", self.base_url),
            {},
        )

    async def datasets_pull_all_for_judgeval(self) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/pull_all_for_judgeval/", self.base_url),
            {},
        )

    async def evaluate_examples(self) -> Any:
        return await self._request(
            "POST",
            url_for("/evaluate/examples", self.base_url),
            {},
        )

    async def evaluate_traces(self) -> Any:
        return await self._request(
            "POST",
            url_for("/evaluate/traces", self.base_url),
            {},
        )

    async def log_eval_results(self) -> Any:
        return await self._request(
            "POST",
            url_for("/log_eval_results/", self.base_url),
            {},
        )

    async def fetch_experiment_run(self) -> Any:
        return await self._request(
            "POST",
            url_for("/fetch_experiment_run/", self.base_url),
            {},
        )

    async def add_to_run_eval_queue(self) -> Any:
        return await self._request(
            "POST",
            url_for("/add_to_run_eval_queue/", self.base_url),
            {},
        )

    async def add_to_run_eval_queue_examples(self) -> Any:
        return await self._request(
            "POST",
            url_for("/add_to_run_eval_queue/examples", self.base_url),
            {},
        )

    async def add_to_run_eval_queue_traces(self) -> Any:
        return await self._request(
            "POST",
            url_for("/add_to_run_eval_queue/traces", self.base_url),
            {},
        )

    async def prompts_fetch(
        self,
        project_id: str,
        name: str,
        commit_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Any:
        query_params = {}
        query_params["project_id"] = project_id
        query_params["name"] = name
        if commit_id is not None:
            query_params["commit_id"] = commit_id
        if tag is not None:
            query_params["tag"] = tag
        return await self._request(
            "GET",
            url_for("/prompts/fetch", self.base_url),
            query_params,
        )

    async def prompts_insert(self) -> Any:
        return await self._request(
            "POST",
            url_for("/prompts/insert", self.base_url),
            {},
        )

    async def prompts_tag(self) -> Any:
        return await self._request(
            "POST",
            url_for("/prompts/tag", self.base_url),
            {},
        )

    async def prompts_untag(self) -> Any:
        return await self._request(
            "POST",
            url_for("/prompts/untag", self.base_url),
            {},
        )

    async def prompts_get_prompt_versions(self, project_id: str, name: str) -> Any:
        query_params = {}
        query_params["project_id"] = project_id
        query_params["name"] = name
        return await self._request(
            "GET",
            url_for("/prompts/get_prompt_versions", self.base_url),
            query_params,
        )

    async def fetch_scorers(self) -> Any:
        return await self._request(
            "POST",
            url_for("/fetch_scorers", self.base_url),
            {},
        )

    async def save_scorer(self) -> Any:
        return await self._request(
            "POST",
            url_for("/save_scorer", self.base_url),
            {},
        )

    async def scorer_exists(self) -> Any:
        return await self._request(
            "POST",
            url_for("/scorer_exists", self.base_url),
            {},
        )

    async def upload_custom_scorer(self) -> Any:
        return await self._request(
            "POST",
            url_for("/upload_custom_scorer/", self.base_url),
            {},
        )

    async def traces_tags_add(self) -> Any:
        return await self._request(
            "POST",
            url_for("/traces/tags/add", self.base_url),
            {},
        )


__all__ = [
    "JudgmentSyncClient",
    "JudgmentAsyncClient",
]
