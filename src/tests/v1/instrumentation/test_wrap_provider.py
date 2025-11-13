import pytest
from unittest.mock import MagicMock, patch
from judgeval.v1.instrumentation.llm.config import wrap_provider, _detect_provider
from judgeval.v1.instrumentation.llm.constants import ProviderType
from judgeval.v1.instrumentation.llm.providers import (
    HAS_OPENAI,
    HAS_ANTHROPIC,
    HAS_TOGETHER,
    HAS_GOOGLE_GENAI,
)


@pytest.fixture
def mock_tracer():
    return MagicMock()


@pytest.mark.skipif(not HAS_OPENAI, reason="OpenAI not installed")
def test_detect_openai_provider():
    from openai import OpenAI

    client = OpenAI(api_key="test")
    provider = _detect_provider(client)
    assert provider == ProviderType.OPENAI


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic not installed")
def test_detect_anthropic_provider():
    from anthropic import Anthropic

    client = Anthropic(api_key="test")
    provider = _detect_provider(client)
    assert provider == ProviderType.ANTHROPIC


@pytest.mark.skipif(not HAS_TOGETHER, reason="Together not installed")
def test_detect_together_provider():
    from together import Together

    client = Together(api_key="test")
    provider = _detect_provider(client)
    assert provider == ProviderType.TOGETHER


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="Google GenAI not installed")
def test_detect_google_provider():
    from google.genai import Client as GoogleClient

    client = GoogleClient(api_key="test")
    provider = _detect_provider(client)
    assert provider == ProviderType.GOOGLE


def test_detect_unknown_provider():
    client = MagicMock()
    provider = _detect_provider(client)
    assert provider == ProviderType.DEFAULT


def test_wrap_provider_with_mock_detect(mock_tracer):
    with patch(
        "judgeval.v1.instrumentation.llm.config._detect_provider"
    ) as mock_detect:
        with patch(
            "judgeval.v1.instrumentation.llm.llm_openai.wrapper.wrap_openai_client"
        ) as mock_wrap:
            mock_detect.return_value = ProviderType.OPENAI
            client = MagicMock()

            wrap_provider(mock_tracer, client)
            mock_wrap.assert_called_once_with(mock_tracer, client)
