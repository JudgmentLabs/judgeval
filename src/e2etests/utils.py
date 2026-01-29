from judgeval.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID, JUDGMENT_API_URL
from judgeval.v1.internal.api import JudgmentSyncClient as JudgmentSyncClientV1

client = JudgmentSyncClient(api_key=JUDGMENT_API_KEY, organization_id=JUDGMENT_ORG_ID)
client_v1 = JudgmentSyncClientV1(
    base_url=JUDGMENT_API_URL, api_key=JUDGMENT_API_KEY, organization_id=JUDGMENT_ORG_ID
)


def delete_project(project_name: str):
    client.projects_delete_from_judgeval(payload={"project_name": project_name})


def create_project(project_name: str):
    client.projects_add(payload={"project_name": project_name})


def retrieve_trace(project_name: str, trace_id: str):
    return client_v1.post_e2e_fetch_trace(
        payload={"project_name": project_name, "trace_id": trace_id}
    )


def retrieve_score(project_name: str, span_id: str, trace_id: str):
    return client_v1.post_e2e_fetch_span_score(
        payload={"project_name": project_name, "span_id": span_id, "trace_id": trace_id}
    )
