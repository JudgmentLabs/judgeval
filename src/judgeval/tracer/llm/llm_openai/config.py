from __future__ import annotations
import importlib.util

HAS_OPENAI = importlib.util.find_spec("openai") is not None

__all__ = ["HAS_OPENAI"]
