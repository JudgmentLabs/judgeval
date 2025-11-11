from __future__ import annotations
import importlib.util

HAS_TOGETHER = importlib.util.find_spec("together") is not None

__all__ = ["HAS_TOGETHER"]
