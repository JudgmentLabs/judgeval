from __future__ import annotations

from functools import lru_cache
from typing import Optional

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient


@lru_cache(maxsize=128)
def resolve_project_id(client: JudgmentSyncClient, project_name: str) -> Optional[str]:
    try:
        response = client.projects_resolve({"project_name": project_name})
        project_id = response.get("project_id")
        return str(project_id) if project_id else None
    except Exception as e:
        judgeval_logger.error(f"Failed to resolve project '{project_name}': {str(e)}")
        return None


def require_project_id(
    client: JudgmentSyncClient,
    project_name: Optional[str],
    default_project_id: Optional[str],
) -> str:
    """Resolve project_id from name or use default. Raises ValueError if neither available."""
    if project_name:
        project_id = resolve_project_id(client, project_name)
        if not project_id:
            raise ValueError(
                f"Project '{project_name}' not found. Please create it first."
            )
        return project_id
    if default_project_id:
        return default_project_id
    raise ValueError(
        "project_name is required. Either pass it to this method or set it in Judgeval(project_name=...)"
    )
