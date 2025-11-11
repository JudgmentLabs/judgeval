"""Google-specific fixtures for tests."""

import pytest
import os

pytest.importorskip("google.genai")

from google.genai import Client
from judgeval.tracer.llm.llm_google.wrapper import wrap_google_client


@pytest.fixture
def google_api_key():
    """Google API key from environment"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def client(google_api_key):
    """Unwrapped Google client"""
    return Client(api_key=google_api_key)


@pytest.fixture
def client_wrapped(tracer, client):
    """Wrapped Google client with tracer"""
    return wrap_google_client(tracer, client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def client_maybe_wrapped(request, tracer, client):
    """Parametrized fixture that returns wrapped or unwrapped client"""
    if request.param == "wrapped":
        return wrap_google_client(tracer, client)
    return client
