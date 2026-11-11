import json

import httpx
import pytest

from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider


@pytest.fixture(autouse=True)
def native_requests(monkeypatch):
    requests = []
    original = httpx.Client.post

    def post(client, url, **kwargs):
        if not str(url).endswith("/mcp/calls"):
            return original(client, url, **kwargs)
        requests.append(json.loads(kwargs["content"]))
        return httpx.Response(200)

    monkeypatch.setattr(httpx.Client, "post", post)
    yield requests
    JudgmentTracerProvider.get_instance().shutdown()
