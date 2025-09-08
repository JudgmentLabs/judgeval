from judgeval.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID

client = JudgmentSyncClient(api_key=JUDGMENT_API_KEY, organization_id=JUDGMENT_ORG_ID)


def delete_project(project_name: str):
    client.projects_delete_from_judgeval(payload={"project_name": project_name})


def create_project(project_name: str):
    client.projects_add(payload={"project_name": project_name})


def retrieve_trace(trace_id: str):
    return client.e2e_fetch_trace(payload={"trace_id": trace_id})


def retrieve_score(span_id: str, trace_id: str):
    return client.e2e_fetch_span_score(
        payload={"span_id": span_id, "trace_id": trace_id}
    )


def retrieve_trace_score(span_id: str, trace_id: str):
    return client.e2e_fetch_trace_scorer_span_score(
        payload={"span_id": span_id, "trace_id": trace_id}
    )
