from __future__ import annotations

from functools import lru_cache
from typing import Optional

from judgeval.logger import judgeval_logger
from judgeval.v1.internal.api import JudgmentSyncClient


@lru_cache(maxsize=128)
def resolve_project_id(client: JudgmentSyncClient, project_name: str) -> Optional[str]:
    try:
        response = client.post_projects_resolve(payload={"project_name": project_name})
        project_id = response.get("project_id")
        return str(project_id) if project_id else None
    except Exception as e:
        judgeval_logger.error(f"Failed to resolve project '{project_name}': {str(e)}")
        return None
