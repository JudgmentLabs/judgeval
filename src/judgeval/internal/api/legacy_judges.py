# Request/response shapes of the flat agent-judge routes removed from /v1 in
# judgment-mono#4302. The new /judges routes take a different body (JUD-10954).
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from typing_extensions import NotRequired


class SDKCreateAgentJudgeRequest(TypedDict):
    name: str
    judge_description: NotRequired[Optional[str]]
    description: NotRequired[Optional[str]]
    model: str
    prompt: str
    score_type: str
    categories: NotRequired[Optional[List[Dict[str, Any]]]]
    min_score: NotRequired[Optional[float]]
    max_score: NotRequired[Optional[float]]


class SDKCreateAgentJudgeResponse(TypedDict):
    judge_id: str


class SDKUpdateAgentJudgeRequest(TypedDict):
    judge_description: NotRequired[Optional[str]]
    score_type: NotRequired[Optional[str]]
    description: NotRequired[Optional[str]]
    model: NotRequired[Optional[str]]
    prompt: NotRequired[Optional[str]]
    categories: NotRequired[Optional[List[Dict[str, Any]]]]
    min_score: NotRequired[Optional[float]]
    max_score: NotRequired[Optional[float]]
    target_major_version: NotRequired[Optional[float]]
    target_minor_version: NotRequired[Optional[float]]
    source_major_version: NotRequired[Optional[float]]
    source_minor_version: NotRequired[Optional[float]]


class SDKUpdateAgentJudgeResponse(TypedDict):
    judge: Dict[str, Any]
