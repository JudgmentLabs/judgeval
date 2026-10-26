from urllib.parse import quote, urljoin

from judgeval.env import JUDGMENT_API_URL


def url_for(path: str, base: str = JUDGMENT_API_URL) -> str:
    return urljoin(base, path)


def encode_path_segment(value: str) -> str:
    """Encodes a path parameter so an ID can't change the route or host."""
    if value in (".", ".."):
        raise ValueError(f"Invalid path parameter: {value!r}")
    return quote(value, safe="")


__all__ = ("url_for", "encode_path_segment")
