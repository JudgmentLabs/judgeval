from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai import Client

try:
    from google.genai import Client

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    Client = None  # type: ignore[misc,assignment]

google_genai_Client = Client

__all__ = [
    "HAS_GOOGLE_GENAI",
    "google_genai_Client",
]
