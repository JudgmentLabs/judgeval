"""Claude Agent SDK auto-instrumentation for Judgeval."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from judgeval.logger import judgeval_logger

if TYPE_CHECKING:
    from judgeval.v1.tracer.base_tracer import BaseTracer

__all__ = ["setup_claude_agent_sdk"]

try:
    import claude_agent_sdk
except ImportError:
    raise ImportError(
        "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk"
    )


def setup_claude_agent_sdk(tracer: "BaseTracer") -> bool:
    """Setup auto-tracing for Claude Agent SDK.

    Args:
        tracer: Judgeval tracer (use isolated=True for multi-tracer setups)

    Returns:
        True if successful

    Examples:
        # Single tracer - just works
        tracer = Judgeval().tracer.create(project_name="my-project")
        setup_claude_agent_sdk(tracer)

        async with ClaudeSDKClient(...) as client:
            await client.query("Hello")
            async for msg in client.receive_response():
                ...  # Auto-traced!

        # Multi-tracer - use @tracer.observe() to scope
        vrm = Judgeval().tracer.create(project_name="VRM", isolated=True)
        copilot = Judgeval().tracer.create(project_name="Copilot", isolated=True)

        setup_claude_agent_sdk(vrm)
        setup_claude_agent_sdk(copilot)

        @vrm.observe()
        async def vrm_agent():
            async with ClaudeSDKClient(...) as client:
                ...  # → VRM project

        @copilot.observe()
        async def copilot_agent():
            async with ClaudeSDKClient(...) as client:
                ...  # → Copilot project
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        register_tracer,
        is_module_patched,
        mark_module_patched,
        _create_client_wrapper_class,
        _create_tool_wrapper_class,
        _wrap_tool_factory,
        _wrap_query_function,
    )

    try:
        register_tracer(tracer)

        if is_module_patched():
            judgeval_logger.debug(f"Registered tracer for '{tracer.project_name}'")
            return True

        _patch_attr(claude_agent_sdk, "ClaudeSDKClient", _create_client_wrapper_class)
        _patch_attr(claude_agent_sdk, "SdkMcpTool", _create_tool_wrapper_class)
        _patch_attr(claude_agent_sdk, "tool", _wrap_tool_factory)
        _patch_attr(claude_agent_sdk, "query", _wrap_query_function)

        mark_module_patched()
        judgeval_logger.info(f"Claude Agent SDK setup for '{tracer.project_name}'")
        return True

    except Exception as e:
        judgeval_logger.error(f"Claude Agent SDK setup failed: {e}")
        return False


def _patch_attr(module, attr_name: str, wrapper_fn) -> None:
    """Patch module attribute and propagate to already-imported modules."""
    original = getattr(module, attr_name, None)
    if not original:
        return

    wrapped = wrapper_fn(original)
    setattr(module, attr_name, wrapped)

    for mod in list(sys.modules.values()):
        if mod and getattr(mod, attr_name, None) is original:
            setattr(mod, attr_name, wrapped)
