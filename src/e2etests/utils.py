from judgeval.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_ORG_ID


def delete_project(project_name: str):
    client = JudgmentSyncClient(
        api_key=JUDGMENT_API_KEY, organization_id=JUDGMENT_ORG_ID
    )
    client.projects_delete(payload={"project_name": project_name})


def create_project(project_name: str):
    client = JudgmentSyncClient(
        api_key=JUDGMENT_API_KEY, organization_id=JUDGMENT_ORG_ID
    )
    client.projects_add(payload={"project_name": project_name})
