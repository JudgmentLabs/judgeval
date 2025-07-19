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
        default_timeout=(10, 60),  # (connect_timeout, read_timeout)
    ):
        super().__init__()

        # Store default timeout
        self.default_timeout = default_timeout

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

    def request(self, method, url, timeout=None, **kwargs):
        """
        Override request method to add default timeout if not specified.

        Args:
            method: HTTP method
            url: Request URL
            timeout: Timeout value. If None, uses default_timeout.
                    Can be a float (total timeout) or tuple (connect, read).
            **kwargs: Other request arguments
        """
        # Use default timeout if none specified
        if timeout is None:
            timeout = self.default_timeout

        return super().request(method, url, timeout=timeout, **kwargs)


requests = RetrySession()
