"""Tests for Forge chat.completions wrapper (via OpenAI SDK)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("openai")

from openai import OpenAI

from judgeval.v1.instrumentation.llm.llm_openai.wrapper import (
    wrap_openai_client_sync,
)
from ..utils import verify_span_attributes_comprehensive


def test_forge_chat_completions_create(tracer, mock_processor):
    api_key = os.getenv("FORGE_API_KEY")
    if not api_key:
        pytest.skip("FORGE_API_KEY environment variable not set")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("FORGE_API_BASE", "https://api.forge.tensorblock.co/v1"),
    )
    wrapped = wrap_openai_client_sync(tracer, client)

    wrapped.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
        max_tokens=32,
    )

    span = mock_processor.get_last_ended_span()
    attrs = mock_processor.get_span_attributes(span)
    verify_span_attributes_comprehensive(
        span=span,
        attrs=attrs,
        expected_span_name="OPENAI_API_CALL",
        expected_model_name="openai/gpt-4o-mini",
    )
