from judgeval.utils.decorators.dont_throw import dont_throw
import functools
from judgeval.api import JudgmentSyncClient


@dont_throw
@functools.lru_cache(maxsize=64)
def _resolve_project_id(project_name: str, api_key: str, organization_id: str) -> str:
    """Resolve project_id from project_name using the API."""
    client = JudgmentSyncClient(
        api_key=api_key,
        organization_id=organization_id,
    )
    response = client.projects_resolve({"project_name": project_name})
    return response["project_id"]
