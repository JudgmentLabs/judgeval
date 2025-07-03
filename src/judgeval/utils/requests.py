import requests as requests_original
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http import HTTPStatus
import httpx
import asyncio
import random


class RetrySession(requests_original.Session):
    def __init__(
        self,
        retries=3,
        backoff_factor=1,
        backoff_jitter=0.1,
        status_forcelist=[
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ],
    ):
        super().__init__()

        retry_strategy = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            backoff_jitter=backoff_jitter,
            status_forcelist=status_forcelist,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.mount("http://", adapter)
        self.mount("https://", adapter)


requests = RetrySession()


class AsyncRetryClient:
    def __init__(
        self,
        retries=3,
        backoff_factor=1,
        jitter=True,
        status_forcelist=None,
    ):
        if status_forcelist is None:
            status_forcelist = {
                HTTPStatus.BAD_GATEWAY,
                HTTPStatus.SERVICE_UNAVAILABLE,
                HTTPStatus.INTERNAL_SERVER_ERROR,
            }
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.status_forcelist = status_forcelist

    async def _request_with_retry(self, method: str, *args, **kwargs):
        async with httpx.AsyncClient() as client:
            for attempt in range(self.retries):
                try:
                    response = await client.request(method, *args, **kwargs)
                    if response.status_code in self.status_forcelist:
                        response.raise_for_status()
                    return response
                except (httpx.RequestError, httpx.HTTPStatusError):
                    if attempt < self.retries - 1:
                        sleep_duration = self.backoff_factor * (2**attempt)
                        if self.jitter:
                            # Apply full jitter: random value between 0 and calculated delay
                            sleep_duration = random.uniform(0, sleep_duration)
                        await asyncio.sleep(sleep_duration)
                    else:
                        print(f"Request failed after {self.retries} attempts.")

    async def post(self, *args, **kwargs):
        return await self._request_with_retry("POST", *args, **kwargs)

    async def delete(self, *args, **kwargs):
        return await self._request_with_retry("DELETE", *args, **kwargs)


async_requests = AsyncRetryClient()
