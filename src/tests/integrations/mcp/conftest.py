import json

import httpx
import pytest

from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider


@pytest.fixture(autouse=True)
def native_requests(monkeypatch):
    requests = []
    original = httpx.Client.request

    def request(client, method, url, **kwargs):
        if not str(url).endswith("/mcp/calls"):
            return original(client, method, url, **kwargs)
        requests.append(json.loads(kwargs["content"]))
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx.Client, "request", request)
    yield requests
    JudgmentTracerProvider.get_instance().shutdown()
