"""Generated from Judgeval public JQL OpenAPI; do not edit."""

from typing import Any, Dict, List, Optional, TypedDict


class JqlQueryResponse(TypedDict):
    query_id: str
    rows: Optional[List[Dict[str, Any]]]
    row_count: Optional[int]
    elapsed_ms: float


class JqlPresentationResponse(TypedDict):
    query_id: str
    presentation: Any
    frame: Optional[Any]
    elapsed_ms: float
