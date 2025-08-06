from typing import List, Dict, Any, Mapping, Literal, Optional
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
        self, method: Literal["POST", "PATCH", "GET", "DELETE"], url: str, payload: Any
    ) -> Any:
        if method == "GET":
            r = self.client.request(
                method,
                url,
                params=payload,
                headers=_headers(self.api_key, self.organization_id),
            )
        else:
            r = self.client.request(
                method,
                url,
                json=json_encoder(payload),
                headers=_headers(self.api_key, self.organization_id),
            )
        return _handle_response(r)

    def check_experiment_type(self, payload: CheckExperimentTypeJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/check_experiment_type/"),
            payload,
        )

    def eval_run_name_exists(self, payload: EvalRunNameCheckJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/eval-run-name-exists/"),
            payload,
        )

    def add_to_run_eval_queue(self, payload: JudgmentEvalJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/add_to_run_eval_queue/"),
            payload,
        )

    def evaluate_trace(self, payload: TraceRunJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/evaluate_trace/"),
            payload,
        )

    def evaluate(self, payload: JudgmentEvalJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/evaluate/"),
            payload,
        )

    def log_eval_results(self, payload: EvalResultsJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/log_eval_results/"),
            payload,
        )

    def fetch_experiment_run(self, payload: EvalResultsFetchJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/fetch_experiment_run/"),
            payload,
        )

    def get_evaluation_status(self) -> Any:
        return self._request(
            "GET",
            url_for("/get_evaluation_status/"),
            {},
        )

    def datasets_insert_examples(
        self, payload: DatasetInsertExamplesJudgmentType
    ) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/insert_examples/"),
            payload,
        )

    def datasets_pull_for_judgeval(self, payload: DatasetFetchJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/pull_for_judgeval/"),
            payload,
        )

    def datasets_fetch_stats_by_project(
        self, payload: DatasetFetchStatsByProjectJudgmentType
    ) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/fetch_stats_by_project/"),
            payload,
        )

    def datasets_push(self, payload: DatasetPushJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/datasets/push/"),
            payload,
        )

    def traces_upsert(self, payload: TraceSaveJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/traces/upsert/"),
            payload,
        )

    def traces_fetch(self, payload: TraceFetchJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/traces/fetch/"),
            payload,
        )

    def traces_spans_batch(self, payload: SpansBatchRequestJudgmentType) -> Any:
        return self._request(
            "POST",
            url_for("/traces/spans/batch/"),
            payload,
        )

    def traces_evaluation_runs_batch(
        self, payload: EvaluationRunsBatchRequestJudgmentType
    ) -> Any:
        return self._request(
            "POST",
            url_for("/traces/evaluation_runs/batch/"),
            payload,
        )

    def projects_add(
        self, payload: ProjectAddJudgmentType
    ) -> ProjectAddResponseJudgmentType:
        return self._request(
            "POST",
            url_for("/projects/add/"),
            payload,
        )

    def scorer_exists(
        self, payload: ScorerExistsRequestJudgmentType
    ) -> ScorerExistsResponseJudgmentType:
        return self._request(
            "POST",
            url_for("/scorer_exists/"),
            payload,
        )

    def save_scorer(
        self, payload: SavePromptScorerRequestJudgmentType
    ) -> SavePromptScorerResponseJudgmentType:
        return self._request(
            "POST",
            url_for("/save_scorer/"),
            payload,
        )

    def fetch_scorer(
        self, payload: FetchPromptScorerRequestJudgmentType
    ) -> FetchPromptScorerResponseJudgmentType:
        return self._request(
            "POST",
            url_for("/fetch_scorer/"),
            payload,
        )

    def projects_resolve(
        self, payload: ResolveProjectNameRequestJudgmentType
    ) -> ResolveProjectNameResponseJudgmentType:
        return self._request(
            "POST",
            url_for("/projects/resolve/"),
            payload,
        )


class JudgmentAsyncClient:
    __slots__ = ("api_key", "organization_id", "client")

    def __init__(self, api_key: str, organization_id: str):
        self.api_key = api_key
        self.organization_id = organization_id
        self.client = httpx.AsyncClient(timeout=30)

    async def _request(
        self, method: Literal["POST", "PATCH", "GET", "DELETE"], url: str, payload: Any
    ) -> Any:
        if method == "GET":
            r = self.client.request(
                method,
                url,
                params=payload,
                headers=_headers(self.api_key, self.organization_id),
            )
        else:
            r = self.client.request(
                method,
                url,
                json=json_encoder(payload),
                headers=_headers(self.api_key, self.organization_id),
            )
        return _handle_response(await r)

    async def check_experiment_type(
        self, payload: CheckExperimentTypeJudgmentType
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/check_experiment_type/"),
            payload,
        )

    async def eval_run_name_exists(self, payload: EvalRunNameCheckJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/eval-run-name-exists/"),
            payload,
        )

    async def add_to_run_eval_queue(self, payload: JudgmentEvalJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/add_to_run_eval_queue/"),
            payload,
        )

    async def evaluate_trace(self, payload: TraceRunJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/evaluate_trace/"),
            payload,
        )

    async def evaluate(self, payload: JudgmentEvalJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/evaluate/"),
            payload,
        )

    async def log_eval_results(self, payload: EvalResultsJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/log_eval_results/"),
            payload,
        )

    async def fetch_experiment_run(self, payload: EvalResultsFetchJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/fetch_experiment_run/"),
            payload,
        )

    async def get_evaluation_status(self) -> Any:
        return await self._request(
            "GET",
            url_for("/get_evaluation_status/"),
            {},
        )

    async def datasets_insert_examples(
        self, payload: DatasetInsertExamplesJudgmentType
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/insert_examples/"),
            payload,
        )

    async def datasets_pull_for_judgeval(
        self, payload: DatasetFetchJudgmentType
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/pull_for_judgeval/"),
            payload,
        )

    async def datasets_fetch_stats_by_project(
        self, payload: DatasetFetchStatsByProjectJudgmentType
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/fetch_stats_by_project/"),
            payload,
        )

    async def datasets_push(self, payload: DatasetPushJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/datasets/push/"),
            payload,
        )

    async def traces_upsert(self, payload: TraceSaveJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/traces/upsert/"),
            payload,
        )

    async def traces_fetch(self, payload: TraceFetchJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/traces/fetch/"),
            payload,
        )

    async def traces_spans_batch(self, payload: SpansBatchRequestJudgmentType) -> Any:
        return await self._request(
            "POST",
            url_for("/traces/spans/batch/"),
            payload,
        )

    async def traces_evaluation_runs_batch(
        self, payload: EvaluationRunsBatchRequestJudgmentType
    ) -> Any:
        return await self._request(
            "POST",
            url_for("/traces/evaluation_runs/batch/"),
            payload,
        )

    async def projects_add(
        self, payload: ProjectAddJudgmentType
    ) -> ProjectAddResponseJudgmentType:
        return await self._request(
            "POST",
            url_for("/projects/add/"),
            payload,
        )

    async def scorer_exists(
        self, payload: ScorerExistsRequestJudgmentType
    ) -> ScorerExistsResponseJudgmentType:
        return await self._request(
            "POST",
            url_for("/scorer_exists/"),
            payload,
        )

    async def save_scorer(
        self, payload: SavePromptScorerRequestJudgmentType
    ) -> SavePromptScorerResponseJudgmentType:
        return await self._request(
            "POST",
            url_for("/save_scorer/"),
            payload,
        )

    async def fetch_scorer(
        self, payload: FetchPromptScorerRequestJudgmentType
    ) -> FetchPromptScorerResponseJudgmentType:
        return await self._request(
            "POST",
            url_for("/fetch_scorer/"),
            payload,
        )

    async def projects_resolve(
        self, payload: ResolveProjectNameRequestJudgmentType
    ) -> ResolveProjectNameResponseJudgmentType:
        return await self._request(
            "POST",
            url_for("/projects/resolve/"),
            payload,
        )


__all__ = [
    "JudgmentSyncClient",
    "JudgmentAsyncClient",
]
