"""MCP-specific payload safety; delivery uses the shared OTel pipeline."""

from judgeval.integrations.mcp._payload import OMITTED, capture_payload


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
