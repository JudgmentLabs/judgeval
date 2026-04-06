from __future__ import annotations

from judgeval.logger import judgeval_logger

__all__ = ["setup_claude_agent_sdk"]

try:
    import claude_agent_sdk  # type: ignore
except ImportError:
    raise ImportError(
        "Claude Agent SDK is not installed and required for the claude agent sdk integration. Please install it with `pip install claude-agent-sdk`."
    )


def setup_claude_agent_sdk() -> bool:
    """Patch the Claude Agent SDK for automatic tracing.

    Monkey-patches ``ClaudeSDKClient`` and the standalone ``query()``
    function so every agent turn, LLM call, and tool invocation is
    recorded as a span. Call this once at startup, before any SDK usage.

    Returns:
        True if patching succeeded, False on error.

    Examples:
        ```python
        from judgeval import Tracer
        from judgeval.integrations import setup_claude_agent_sdk

        Tracer.init(project_name="my-agent")
        setup_claude_agent_sdk()

        # All Claude Agent SDK calls are now traced automatically
        ```
    """
    from judgeval.integrations.claude_agent_sdk.wrapper import (
        TracingState,
        _create_client_wrapper_class,
        _wrap_query_function,
    )

    try:
        state = TracingState()

        # Store original classes before patching
        original_client = (
            claude_agent_sdk.ClaudeSDKClient
            if hasattr(claude_agent_sdk, "ClaudeSDKClient")
            else None
        )
        original_query_fn = (
            claude_agent_sdk.query if hasattr(claude_agent_sdk, "query") else None
        )

        # Patch ClaudeSDKClient
        if original_client:
            wrapped_client = _create_client_wrapper_class(original_client, state)
            claude_agent_sdk.ClaudeSDKClient = wrapped_client  # type: ignore

        # Patch standalone query() function if it exists
        # Note: The standalone query() uses InternalClient, not ClaudeSDKClient,
        # so we need to wrap it separately to add tracing
        if original_query_fn:
            wrapped_query_fn = _wrap_query_function(original_query_fn, state)
            claude_agent_sdk.query = wrapped_query_fn  # type: ignore

        judgeval_logger.info("Claude Agent SDK integration setup successful")
        return True

    except Exception as e:
        judgeval_logger.error(f"Failed to setup Claude Agent SDK integration: {e}")
        return False
