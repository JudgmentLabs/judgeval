from __future__ import annotations
import importlib.util

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None

__all__ = ["HAS_ANTHROPIC"]
