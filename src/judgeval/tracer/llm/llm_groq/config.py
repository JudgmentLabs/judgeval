from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groq import Groq, AsyncGroq

try:
    from groq import Groq, AsyncGroq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    Groq = AsyncGroq = None  # type: ignore[misc,assignment]

# Export the classes for runtime use
groq_Groq = Groq
groq_AsyncGroq = AsyncGroq

__all__ = [
    "HAS_GROQ",
    "groq_Groq",
    "groq_AsyncGroq",
]
