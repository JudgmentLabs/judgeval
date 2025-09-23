from __future__ import annotations

HAS_GOOGLE_GENAI = False
google_genai_Client = None
google_genai_AsyncClient = None

try:
    from google.genai import Client  # type: ignore[import-untyped]
    from google.genai.client import AsyncClient  # type: ignore[import-untyped]

    google_genai_Client = Client
    google_genai_AsyncClient = AsyncClient
    HAS_GOOGLE_GENAI = True
except ImportError:
    pass

__all__ = [
    "HAS_GOOGLE_GENAI",
    "google_genai_Client",
    "google_genai_AsyncClient",
]
