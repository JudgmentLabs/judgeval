"""Anthropic-specific fixtures for tests."""

import pytest
import os

pytest.importorskip("anthropic")

from anthropic import Anthropic, AsyncAnthropic
from judgeval.tracer.llm.llm_anthropic.wrapper import (
    wrap_anthropic_client_sync,
    wrap_anthropic_client_async,
)


@pytest.fixture
def anthropic_api_key():
    """Anthropic API key from environment"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sync_client(anthropic_api_key):
    """Unwrapped sync Anthropic client"""
    return Anthropic(api_key=anthropic_api_key)


@pytest.fixture
def async_client(anthropic_api_key):
    """Unwrapped async Anthropic client"""
    return AsyncAnthropic(api_key=anthropic_api_key)


@pytest.fixture
def wrapped_sync_client(tracer, sync_client):
    """Wrapped sync Anthropic client with tracer"""
    return wrap_anthropic_client_sync(tracer, sync_client)


@pytest.fixture
def wrapped_async_client(tracer, async_client):
    """Wrapped async Anthropic client with tracer"""
    return wrap_anthropic_client_async(tracer, async_client)


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def sync_client_maybe_wrapped(request, tracer, sync_client):
    """Parametrized fixture that yields both wrapped and unwrapped sync clients"""
    if request.param == "wrapped":
        return wrap_anthropic_client_sync(tracer, sync_client)
    return sync_client


@pytest.fixture(params=["wrapped", "unwrapped"], ids=["with_tracer", "without_tracer"])
def async_client_maybe_wrapped(request, tracer, async_client):
    """Parametrized fixture that yields both wrapped and unwrapped async clients"""
    if request.param == "wrapped":
        return wrap_anthropic_client_async(tracer, async_client)
    return async_client
