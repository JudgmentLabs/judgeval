from typing import Optional

import functools

from judgeval.api import JudgmentSyncClient
from judgeval.exceptions import JudgmentAPIError
from judgeval.utils.decorators.dont_throw import dont_throw


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


def resolve_project_id_or_none(
    project_name: Optional[str],
    api_key: str,
    organization_id: str,
) -> Optional[str]:
    """Resolve project_name to project_id, raising JudgmentAPIError if not found. Returns None if project_name is None."""
    if not project_name:
        return None
    return resolve_project_id_required(project_name, api_key, organization_id)


def resolve_project_id_required(
    project_name: str,
    api_key: str,
    organization_id: str,
) -> str:
    """Resolve project_name to project_id, raising JudgmentAPIError if not found."""
    project_id = _resolve_project_id(project_name, api_key, organization_id)
    if not project_id:
        raise JudgmentAPIError(
            status_code=404,
            detail=f"Project '{project_name}' not found",
            response=None,  # type: ignore
        )
    return project_id
