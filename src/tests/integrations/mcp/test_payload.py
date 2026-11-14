"""MCP-specific payload safety; delivery uses the shared OTel pipeline."""

import json

import pytest

from judgeval.integrations.mcp._payload import OMITTED, capture_payload


@pytest.mark.parametrize(
    "value", ["x" * (128 * 1024), ["x" * 1024] * 1024], ids=["string", "nested"]
)
def test_oversized_sources_skip_serialization_and_redaction(value, monkeypatch):
    operations = []

    def serialize(*args, **kwargs):
        operations.append("serialize")
        raise AssertionError("Oversized source must not be serialized")

    def redact(value, field):
        operations.append("redact")
        return "small"

    monkeypatch.setattr(json.JSONEncoder, "encode", serialize)
    monkeypatch.setattr(json.JSONEncoder, "iterencode", serialize)
    assert (capture_payload(value, "output", redact), operations) == (OMITTED, [])


@pytest.mark.parametrize("adapter_version", [1, 2])
def test_oversized_model_results_skip_conversion(adapter_version, monkeypatch):
    from types import SimpleNamespace

    from pydantic import BaseModel

    from judgeval.integrations.mcp._v1 import ModelResults
    from judgeval.integrations.mcp._v2 import WireResults

    class Result(BaseModel):
        content: str

    conversions = []

    def dump(*args, **kwargs):
        conversions.append("dump")
        raise AssertionError("Oversized model must not be converted")

    monkeypatch.setattr(Result, "model_dump", dump)
    result = Result(content="x" * (128 * 1024))
    adapter = ModelResults() if adapter_version == 1 else WireResults()
    wrapped = SimpleNamespace(root=result) if adapter_version == 1 else result
    assert (adapter.payload(wrapped), conversions) == (OMITTED, [])


def test_capture_detaches_and_omits_unsafe_payloads():
    original = {"token": "secret"}

    def redact(value, field):
        value["token"] = "safe"
        return value

    def fail(value, field):
        raise ValueError("secret")

    assert {
        "captured": capture_payload(original, "input", redact),
        "original": original,
        "failed": capture_payload(original, "output", fail),
        "oversized": capture_payload("x" * (128 * 1024), "output", None),
        "null": capture_payload(None, "output", None),
    } == {
        "captured": {"token": "safe"},
        "original": {"token": "secret"},
        "failed": OMITTED,
        "oversized": OMITTED,
        "null": None,
    }


def test_v2_capture_preserves_existing_middleware():
    from types import SimpleNamespace
    from judgeval.integrations.mcp._observer import McpObserver, McpOptions
    from judgeval.integrations.mcp._v2 import instrument, JudgmentMcpMiddleware

    upstream = object()
    server = SimpleNamespace(middleware=[upstream], get_request_handler=lambda _: None)
    observer = McpObserver(McpOptions(), server_name="test", server_version="1")
    instrument(observer, server)
    instrument(observer, server)
    assert {
        "instrumented": isinstance(server.middleware[0], JudgmentMcpMiddleware),
        "remaining": server.middleware[1:],
    } == {"instrumented": True, "remaining": [upstream]}
