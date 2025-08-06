from urllib.parse import urljoin

from judgeval.env import JUDGMENT_API_URL


def url_for(path: str, base: str = JUDGMENT_API_URL) -> str:
    return urljoin(base, path)


__all__ = ("url_for",)
