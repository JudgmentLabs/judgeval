from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from google.genai import Client


@pytest.fixture
def google_client():
    return Client(api_key="test-key")


def make_google_response(text="Hello", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.text = text
    response.usage_metadata = MagicMock(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        cached_content_token_count=None,
    )
    return response
