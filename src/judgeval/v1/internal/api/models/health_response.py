from __future__ import annotations

from typing import TypedDict


class HealthResponse(TypedDict):
    status: str
    timestamp: str
