from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from together import Together, AsyncTogether  # type: ignore[import-untyped]

try:
    from together import Together, AsyncTogether  # type: ignore[import-untyped]

    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False
    Together = AsyncTogether = None  # type: ignore[misc,assignment]

# Export the classes for runtime use
together_Together = Together
together_AsyncTogether = AsyncTogether

__all__ = [
    "HAS_TOGETHER",
    "together_Together",
    "together_AsyncTogether",
]
