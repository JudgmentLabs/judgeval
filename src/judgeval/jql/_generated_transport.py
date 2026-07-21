"""Generated from Judgeval public JQL OpenAPI; do not edit."""

from typing import Any, Dict, List, Optional, TypedDict

PUBLIC_JQL_OPENAPI_SHA256 = (
    "e889654bbd14fe91ddf3662dc62aa415b264a137787960d20f4ec2e27d123a18"
)


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
