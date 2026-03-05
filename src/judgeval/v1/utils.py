from __future__ import annotations

from typing import Optional

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient

_project_id_cache: dict[tuple[str, str], str] = {}


def resolve_project_id(client: JudgmentSyncClient, project_name: str) -> Optional[str]:
    key = (client.organization_id, project_name)
    if key in _project_id_cache:
        return _project_id_cache[key]
    try:
        response = client.post_projects_resolve(payload={"project_name": project_name})
        project_id = response.get("project_id")
        if project_id:
            _project_id_cache[key] = project_id
    except Exception as e:
        judgeval_logger.error(f"Failed to resolve project '{project_name}': {str(e)}")
        return None
    return _project_id_cache.get(key, None)
