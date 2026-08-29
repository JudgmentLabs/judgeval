from __future__ import annotations
import importlib.util

HAS_MINIMAX = importlib.util.find_spec("openai") is not None

MINIMAX_BASE_URL_PATTERN = "api.minimax.io"


def is_minimax_client(client: object) -> bool:
    """Return True if the client is an OpenAI-compatible client pointed at MiniMax."""
    if not HAS_MINIMAX:
        return False
    try:
        base_url = str(getattr(client, "base_url", ""))
        return MINIMAX_BASE_URL_PATTERN in base_url
    except Exception:
        return False


__all__ = ["HAS_MINIMAX", "is_minimax_client"]
