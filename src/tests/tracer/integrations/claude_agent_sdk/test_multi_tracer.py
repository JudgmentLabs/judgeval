"""Tests for multi-tracer Claude Agent SDK registration."""

import pytest
from judgeval.v1.integrations.claude_agent_sdk import setup_claude_agent_sdk
from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
    _registered_tracers,
    get_active_tracer,
)


@pytest.fixture(autouse=True)
def reset_wrapper_state():
    """Reset wrapper state between tests."""
    import judgeval.v1.integrations.claude_agent_sdk.wrapper as w

    w._registered_tracers.clear()
    w._module_patched = False
    yield
    w._registered_tracers.clear()
    w._module_patched = False


def test_multiple_tracers_register(tracer, mock_processor):
    """Multiple tracers can be registered."""
    # Create second tracer using same pattern as conftest
    from opentelemetry.trace import get_tracer_provider
    from tests.tracer.integrations.claude_agent_sdk.conftest import MockTracer

    tracer2 = MockTracer(get_tracer_provider().get_tracer("tracer2"))

    setup_claude_agent_sdk(tracer)
    setup_claude_agent_sdk(tracer2)

    assert len(_registered_tracers) == 2


def test_fallback_to_last_registered(tracer, mock_processor):
    """Falls back to last registered tracer when no active span."""
    from opentelemetry.trace import get_tracer_provider
    from tests.tracer.integrations.claude_agent_sdk.conftest import MockTracer

    tracer2 = MockTracer(get_tracer_provider().get_tracer("tracer2"))

    setup_claude_agent_sdk(tracer)
    setup_claude_agent_sdk(tracer2)

    assert get_active_tracer() is tracer2


def test_returns_tracer_with_active_span(tracer, mock_processor):
    """Returns registered tracer when span is active."""
    setup_claude_agent_sdk(tracer)

    with tracer.get_tracer().start_as_current_span("test_op"):
        active = get_active_tracer()
        assert active is tracer
