"""OpenAI-specific fixtures for tests."""

import pytest
import os

pytest.importorskip("openai")

from openai import OpenAI, AsyncOpenAI
from judgeval.tracer.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
    wrap_openai_client_async,
)


@pytest.fixture
def openai_api_key():
    """OpenAI API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(openai_api_key):
    """Unwrapped sync OpenAI client"""
    return OpenAI(api_key=openai_api_key)


@pytest.fixture
def async_client(openai_api_key):
    """Unwrapped async OpenAI client"""
    return AsyncOpenAI(api_key=openai_api_key)


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    """Wrapped sync OpenAI client with tracer"""
    return wrap_openai_client_sync(tracer, sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    """Wrapped async OpenAI client with tracer"""
    return wrap_openai_client_async(tracer, async_client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    """Parametrized fixture that yields both wrapped and unwrapped sync clients"""
    if request.param == "wrapped":
        return wrap_openai_client_sync(tracer, sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    """Parametrized fixture that yields both wrapped and unwrapped async clients"""
    if request.param == "wrapped":
        return wrap_openai_client_async(tracer, async_client)
    return async_client
