from judgeval.internal.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID, JUDGMENT_API_URL
from judgeval.utils.url import url_for

assert JUDGMENT_API_KEY, "JUDGMENT_API_KEY must be set"
assert JUDGMENT_ORG_ID, "JUDGMENT_ORG_ID must be set"
assert JUDGMENT_API_URL, "JUDGMENT_API_URL must be set"

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


# The e2e fetch routes are hidden from the public spec, so they are not on the
# client; call them through its request helper.
def retrieve_trace(project_name: str, trace_id: str):
    return client._request(
        "GET",
        url_for(f"/v1/e2e_fetch_trace/{project_name}/{trace_id}", client.base_url),
        {},
    )


def retrieve_score(project_name: str, span_id: str, trace_id: str):
    return client._request(
        "POST",
        url_for("/v1/e2e_fetch_span_score/", client.base_url),
        {"project_name": project_name, "span_id": span_id, "trace_id": trace_id},
    )
