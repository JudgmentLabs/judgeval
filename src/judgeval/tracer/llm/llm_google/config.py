from __future__ import annotations
import importlib.util

HAS_GOOGLE_GENAI = importlib.util.find_spec("google.genai") is not None

__all__ = ["HAS_GOOGLE_GENAI"]
