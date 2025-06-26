import requests as requests_original
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http import HTTPStatus


class RetrySession(requests_original.Session):
    def __init__(
        self,
        retries=3,
        backoff_factor=0.5,
        status_forcelist=[HTTPStatus.BAD_GATEWAY, HTTPStatus.SERVICE_UNAVAILABLE],
    ):
        super().__init__()

        retry_strategy = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.mount("http://", adapter)
        self.mount("https://", adapter)


requests = RetrySession()
