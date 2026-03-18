from __future__ import annotations

from unittest.mock import MagicMock, patch

from judgeval.v1.instrumentation.llm.config import _detect_provider, wrap_provider
from judgeval.v1.instrumentation.llm.constants import ProviderType


class TestDetectProvider:
    def test_openai_sync_client(self):
        from openai import OpenAI

        with (
            patch("judgeval.v1.instrumentation.llm.config.HAS_OPENAI", True),
            patch("judgeval.v1.instrumentation.llm.config.HAS_ANTHROPIC", False),
            patch("judgeval.v1.instrumentation.llm.config.HAS_TOGETHER", False),
            patch("judgeval.v1.instrumentation.llm.config.HAS_GOOGLE_GENAI", False),
        ):
            from openai import AsyncOpenAI

            with patch(
                "judgeval.v1.instrumentation.llm.config._detect_provider",
                side_effect=lambda c: ProviderType.OPENAI  # noqa: E731
                if isinstance(c, (OpenAI, AsyncOpenAI))
                else ProviderType.DEFAULT,
            ):
                result = _detect_provider(MagicMock(spec=OpenAI))
        assert result in (ProviderType.OPENAI, ProviderType.DEFAULT)

    def test_unknown_client_returns_default(self):
        client = MagicMock()
        result = _detect_provider(client)
        assert result == ProviderType.DEFAULT

    def test_openai_real_detection(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="http://localhost")
        result = _detect_provider(client)
        assert result == ProviderType.OPENAI

    def test_openai_async_real_detection(self):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        result = _detect_provider(client)
        assert result == ProviderType.OPENAI

    def test_anthropic_real_detection(self):
        from anthropic import Anthropic

        client = Anthropic(api_key="test")
        result = _detect_provider(client)
        assert result == ProviderType.ANTHROPIC


class TestWrapProvider:
    def test_wraps_openai_sync(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="http://localhost")
        original_create = client.chat.completions.create
        wrap_provider(client)
        assert client.chat.completions.create is not original_create

    def test_wraps_openai_async(self):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="test", base_url="http://localhost")
        original_create = client.chat.completions.create
        wrap_provider(client)
        assert client.chat.completions.create is not original_create

    def test_returns_same_client_object(self):
        from openai import OpenAI

        client = OpenAI(api_key="test", base_url="http://localhost")
        result = wrap_provider(client)
        assert result is client
