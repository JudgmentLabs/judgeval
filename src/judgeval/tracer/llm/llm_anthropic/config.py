from __future__ import annotations

HAS_ANTHROPIC = False
anthropic_Anthropic = None
anthropic_AsyncAnthropic = None

try:
    from anthropic import Anthropic, AsyncAnthropic  # type: ignore[import-untyped]

    anthropic_Anthropic = Anthropic
    anthropic_AsyncAnthropic = AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    pass

__all__ = [
    "HAS_ANTHROPIC",
    "anthropic_Anthropic",
    "anthropic_AsyncAnthropic",
]
