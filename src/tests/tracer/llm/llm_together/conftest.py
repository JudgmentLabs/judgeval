"""Together-specific fixtures for tests."""

import pytest
import os

pytest.importorskip("together")

from together import Together, AsyncTogether
from judgeval.tracer.llm.llm_together.wrapper import wrap_together_client


@pytest.fixture
def together_api_key():
    """Together API key from environment"""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(together_api_key):
    """Unwrapped sync Together client"""
    return Together(api_key=together_api_key)


@pytest.fixture
def async_client(together_api_key):
    """Unwrapped async Together client"""
    return AsyncTogether(api_key=together_api_key)


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    """Wrapped sync Together client with tracer"""
    return wrap_together_client(tracer, sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    """Wrapped async Together client with tracer"""
    return wrap_together_client(tracer, async_client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    """Parametrized fixture that yields both wrapped and unwrapped sync clients"""
    if request.param == "wrapped":
        return wrap_together_client(tracer, sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    """Parametrized fixture that yields both wrapped and unwrapped async clients"""
    if request.param == "wrapped":
        return wrap_together_client(tracer, async_client)
    return async_client
