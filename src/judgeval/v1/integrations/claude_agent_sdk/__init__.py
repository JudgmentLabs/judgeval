from __future__ import annotations
from typing import TYPE_CHECKING
import sys
from judgeval.logger import judgeval_logger

if TYPE_CHECKING:
    from judgeval.v1.tracer.base_tracer import BaseTracer

__all__ = ["setup_claude_agent_sdk"]

try:
    import claude_agent_sdk  # type: ignore
except ImportError:
    raise ImportError(
        "Claude Agent SDK is not installed and required for the claude agent sdk integration. Please install it with `pip install claude-agent-sdk`."
    )


def setup_claude_agent_sdk(
    tracer: "BaseTracer",
) -> bool:
    """
    Setup Judgeval integration with Claude Agent SDK. Will automatically patch the SDK for automatic tracing.

    Args:
        tracer: Judgeval Tracer instance

    Returns:
        bool: True if setup was successful, False otherwise.

    Example:
        ```python
        import claude_agent_sdk
        from judgeval.v1.integrations.claude_agent_sdk import setup_claude_agent_sdk

        tracer = Tracer(project_name="my-project")
        setup_claude_agent_sdk(tracer=tracer)

        # Now use claude_agent_sdk normally - all calls automatically traced
        ```
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        _create_client_wrapper_class,
        _create_tool_wrapper_class,
        _wrap_tool_factory,
        _wrap_query_function,
    )

    try:
        # Store original classes before patching
        original_client = (
            claude_agent_sdk.ClaudeSDKClient
            if hasattr(claude_agent_sdk, "ClaudeSDKClient")
            else None
        )
        original_tool_class = (
            claude_agent_sdk.SdkMcpTool
            if hasattr(claude_agent_sdk, "SdkMcpTool")
            else None
        )
        original_tool_fn = (
            claude_agent_sdk.tool if hasattr(claude_agent_sdk, "tool") else None
        )
        original_query_fn = (
            claude_agent_sdk.query if hasattr(claude_agent_sdk, "query") else None
        )

        # Patch ClaudeSDKClient
        if original_client:
            wrapped_client = _create_client_wrapper_class(original_client, tracer)
            claude_agent_sdk.ClaudeSDKClient = wrapped_client  # type: ignore

            # Update all modules that already imported ClaudeSDKClient
            for module in list(sys.modules.values()):
                if module and hasattr(module, "ClaudeSDKClient"):
                    if getattr(module, "ClaudeSDKClient", None) is original_client:
                        setattr(module, "ClaudeSDKClient", wrapped_client)

        # Patch SdkMcpTool
        if original_tool_class:
            wrapped_tool_class = _create_tool_wrapper_class(original_tool_class, tracer)
            claude_agent_sdk.SdkMcpTool = wrapped_tool_class  # type: ignore

            # Update all modules that already imported SdkMcpTool
            for module in list(sys.modules.values()):
                if module and hasattr(module, "SdkMcpTool"):
                    if getattr(module, "SdkMcpTool", None) is original_tool_class:
                        setattr(module, "SdkMcpTool", wrapped_tool_class)

        # Patch tool() decorator
        if original_tool_fn:
            wrapped_tool_fn = _wrap_tool_factory(original_tool_fn, tracer)
            claude_agent_sdk.tool = wrapped_tool_fn  # type: ignore

            # Update all modules that already imported tool
            for module in list(sys.modules.values()):
                if module and hasattr(module, "tool"):
                    if getattr(module, "tool", None) is original_tool_fn:
                        setattr(module, "tool", wrapped_tool_fn)

        # Patch standalone query() function if it exists
        # Note: The standalone query() uses InternalClient, not ClaudeSDKClient,
        # so we need to wrap it separately to add tracing
        if original_query_fn:
            wrapped_query_fn = _wrap_query_function(original_query_fn, tracer)
            claude_agent_sdk.query = wrapped_query_fn  # type: ignore

            # Update all modules that already imported query
            for module in list(sys.modules.values()):
                if module and hasattr(module, "query"):
                    if getattr(module, "query", None) is original_query_fn:
                        setattr(module, "query", wrapped_query_fn)

        judgeval_logger.info("Claude Agent SDK integration setup successful")
        return True

    except Exception as e:
        judgeval_logger.error(f"Failed to setup Claude Agent SDK integration: {e}")
        return False
