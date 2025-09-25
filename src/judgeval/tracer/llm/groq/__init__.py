from __future__ import annotations

HAS_GROQ = False
groq_Groq = None
groq_AsyncGroq = None

try:
    from groq import Groq, AsyncGroq  # type: ignore[import-untyped]

    groq_Groq = Groq
    groq_AsyncGroq = AsyncGroq
    HAS_GROQ = True
except ImportError:
    pass

__all__ = [
    "HAS_GROQ",
    "groq_Groq",
    "groq_AsyncGroq",
]
