from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID, JUDGMENT_API_URL

client = JudgmentSyncClient(
    base_url=JUDGMENT_API_URL,
    api_key=JUDGMENT_API_KEY,
    organization_id=JUDGMENT_ORG_ID,
)


def delete_project(project_name: str):
    response = client.post_projects_resolve(payload={"project_name": project_name})
    project_id = response.get("project_id")
    if project_id:
        client.delete_projects(project_id=str(project_id))


def create_project(project_name: str):
    client.post_projects(payload={"project_name": project_name})


def retrieve_trace(project_name: str, trace_id: str):
    return client.post_e2e_fetch_trace(
        payload={"project_name": project_name, "trace_id": trace_id}
    )


def retrieve_score(project_name: str, span_id: str, trace_id: str):
    return client.post_e2e_fetch_span_score(
        payload={
            "project_name": project_name,
            "span_id": span_id,
            "trace_id": trace_id,
        }
    )
