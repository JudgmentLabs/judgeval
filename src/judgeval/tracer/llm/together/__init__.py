from __future__ import annotations

HAS_TOGETHER = False
together_Together = None
together_AsyncTogether = None

try:
    from together import Together, AsyncTogether  # type: ignore[import-untyped]

    together_Together = Together
    together_AsyncTogether = AsyncTogether
    HAS_TOGETHER = True
except ImportError:
    pass

__all__ = [
    "HAS_TOGETHER",
    "together_Together",
    "together_AsyncTogether",
]
