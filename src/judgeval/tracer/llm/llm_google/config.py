from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.client import AsyncClient

try:
    from google.genai import Client
    from google.genai.client import AsyncClient

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    Client = AsyncClient = None  # type: ignore[misc,assignment]

google_genai_Client = Client
google_genai_AsyncClient = AsyncClient

__all__ = [
    "HAS_GOOGLE_GENAI",
    "google_genai_Client",
    "google_genai_AsyncClient",
]
