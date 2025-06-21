from typing import Any, TypeVar, Union
from judgeval.constants import ROOT_API
import requests


class APIClient:
    __slots__ = ("base_url", "api_key", "org_id", "headers")

    base_url: str
    api_key: str
    org_id: str
    headers: dict[str, str]

    def __init__(
        self,
        api_key: str,
        org_id: str,
    ):
        self.base_url = ROOT_API
        self.api_key = api_key
        self.org_id = org_id
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer %s" % self.api_key,
            "X-Orgaization-ID": self.org_id,
        }

    def _do_get(
        self,
        endpoint: str,
        params: Union[dict[str, str], None] = None,
    ) -> requests.Response:
        url = "%s/%s" % (self.base_url, endpoint)
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response

    def _do_post(
        self,
        endpoint: str,
        data: Any = None,
    ) -> requests.Response:
        url = "%s/%s" % (self.base_url, endpoint)
        response = requests.post(url, json=data, headers=self.headers)
        response.raise_for_status()
        return response
