import pytest
import os
from typing import Any, Dict

from judgeval.v1.integrations.claude_agent_sdk import setup_claude_agent_sdk
from tests.tracer.integrations.claude_agent_sdk.utils import (
    find_spans_by_kind,
    verify_agent_span,
    verify_llm_span,
    verify_tool_span,
)

# Skip these tests if no API key is set
# Note: Claude Code CLI must be installed (npm install -g @anthropic-ai/claude-code)
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping real API tests",
)


@pytest.mark.asyncio
async def test_simple_question_with_real_api(tracer, mock_processor):
    """Test simple question with real Claude Agent SDK API."""
    # Setup integration FIRST
    setup_claude_agent_sdk(tracer=tracer)

    # Import AFTER patching
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    # Configure options
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-20250514",
    )

    # Make real API call
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 2 + 2? Just give me the number.")

        response_text = []
        async for message in client.receive_response():
            # Collect text from assistant messages
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        response_text.append(block.text)

        # Verify we got a response
        full_response = " ".join(response_text)
        assert "4" in full_response, f"Expected '4' in response, got: {full_response}"

    # Force flush to ensure all spans are processed
    mock_processor.force_flush()

    # Verify spans were created
    spans = mock_processor.ended_spans
    assert len(spans) > 0, "No spans were created"

    # Find and verify agent span
    agent_spans = find_spans_by_kind(spans, "agent")
    assert len(agent_spans) >= 1, "No agent span found"

    verify_agent_span(agent_spans[0], expected_input="2 + 2", check_output=True)

    # Find and verify LLM spans
    llm_spans = find_spans_by_kind(spans, "llm")
    assert len(llm_spans) >= 1, "No LLM spans found"

    verify_llm_span(
        llm_spans[0],
        expected_model="claude-sonnet",
        check_provider=True,
        check_usage=True,
    )

    print(f"\n✅ Test passed! Created {len(spans)} spans:")
    print(f"   - {len(agent_spans)} agent span(s)")
    print(f"   - {len(llm_spans)} LLM span(s)")


@pytest.mark.asyncio
async def test_with_tool_execution_real_api(tracer, mock_processor):
    """Test with tool execution using real Claude Agent SDK API."""
    # Setup integration FIRST
    setup_claude_agent_sdk(tracer=tracer)

    # Import AFTER patching
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        tool,
        create_sdk_mcp_server,
    )

    # Define a simple calculator tool
    @tool(
        "calculator",
        "Calculates simple math expressions. Input should be a string like '2+2' or '10*5'.",
        {"expression": str},
    )
    async def calculator(args: Dict[str, Any]) -> Dict[str, Any]:
        """Simple calculator tool."""
        try:
            expression = args.get("expression", "")
            # Security: only allow basic math
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: Invalid characters in expression",
                        }
                    ],
                    "is_error": True,
                }

            result = eval(expression, {"__builtins__": {}}, {})
            return {"content": [{"type": "text", "text": f"{expression} = {result}"}]}
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "is_error": True,
            }

    # Create MCP server with the tool
    calc_server = create_sdk_mcp_server(
        name="math_tools", version="1.0.0", tools=[calculator]
    )

    # Configure options with tool
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant. Use the calculator tool for any math.",
        mcp_servers={"math": calc_server},
        allowed_tools=["mcp__math__calculator"],
        permission_mode="acceptEdits",
    )

    # Make real API call with tool
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 15 multiplied by 23? Use the calculator tool.")

        tool_used = False
        response_text = []

        async for message in client.receive_response():
            # Check for tool use
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "name") and "calculator" in block.name:
                        tool_used = True
                    if hasattr(block, "text"):
                        response_text.append(block.text)

        # Verify tool was used
        assert tool_used, "Calculator tool was not used"

        # Verify we got the right answer
        full_response = " ".join(response_text)
        assert "345" in full_response, (
            f"Expected '345' in response, got: {full_response}"
        )

    # Force flush to ensure all spans are processed
    mock_processor.force_flush()

    # Verify spans were created
    spans = mock_processor.ended_spans
    assert len(spans) > 0, "No spans were created"

    # Find spans by kind
    agent_spans = find_spans_by_kind(spans, "agent")
    llm_spans = find_spans_by_kind(spans, "llm")
    tool_spans = find_spans_by_kind(spans, "tool")

    # Verify we have all expected spans
    assert len(agent_spans) >= 1, "No agent span found"
    assert len(llm_spans) >= 1, "No LLM spans found"
    assert len(tool_spans) >= 1, "No tool spans found"

    # Verify tool span
    verify_tool_span(
        tool_spans[0],
        expected_tool_name="calculator",
        check_input=True,
        check_output=True,
    )

    # Verify span hierarchy - tool should be in the same trace as agent
    # (tool can be child of agent OR child of LLM which is child of agent)
    tool_span = tool_spans[0]
    agent_span = agent_spans[0]

    # Both should be in the same trace
    assert tool_span.context.trace_id == agent_span.context.trace_id, (
        "Tool span is not in the same trace as agent span"
    )
    # Tool should have a parent (either agent or LLM span)
    assert tool_span.parent is not None, "Tool span has no parent"

    print(f"\n✅ Test passed! Created {len(spans)} spans:")
    print(f"   - {len(agent_spans)} agent span(s)")
    print(f"   - {len(llm_spans)} LLM span(s)")
    print(f"   - {len(tool_spans)} tool span(s)")
    print("   - All spans in same trace with proper hierarchy ✅")


@pytest.mark.asyncio
async def test_query_method_tracing(tracer, mock_processor):
    """Test that query method itself is properly traced."""
    # Setup integration FIRST
    setup_claude_agent_sdk(tracer=tracer)

    # Import AFTER patching
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-20250514",
    )

    # Test multiple queries in sequence
    async with ClaudeSDKClient(options=options) as client:
        # First query
        await client.query("What is the capital of France?")
        response_count_1 = 0
        async for message in client.receive_response():
            response_count_1 += 1

        # Second query in the same session
        await client.query("What is 10 + 20?")
        response_count_2 = 0
        async for message in client.receive_response():
            response_count_2 += 1

    # Force flush to ensure all spans are processed
    mock_processor.force_flush()

    # Verify spans were created for both queries
    spans = mock_processor.ended_spans
    assert len(spans) > 0, "No spans were created"

    # Should have agent spans for both queries
    agent_spans = find_spans_by_kind(spans, "agent")
    assert len(agent_spans) >= 2, (
        f"Expected at least 2 agent spans, got {len(agent_spans)}"
    )

    # Should have LLM spans for both queries
    llm_spans = find_spans_by_kind(spans, "llm")
    assert len(llm_spans) >= 2, f"Expected at least 2 LLM spans, got {len(llm_spans)}"

    # Verify each agent span has input
    for agent_span in agent_spans:
        attrs = dict(agent_span.attributes or {})
        from judgeval.tracer.keys import AttributeKeys

        assert AttributeKeys.JUDGMENT_INPUT in attrs, "Agent span missing input"

    print(f"\n✅ Test passed! Created {len(spans)} spans for 2 queries:")
    print(f"   - {len(agent_spans)} agent span(s)")
    print(f"   - {len(llm_spans)} LLM span(s)")
    print(f"   - First query had {response_count_1} message(s)")
    print(f"   - Second query had {response_count_2} message(s)")


@pytest.mark.asyncio
async def test_standalone_query_function(tracer, mock_processor):
    """Test standalone query() function API."""
    # Setup integration FIRST
    setup_claude_agent_sdk(tracer=tracer)

    # Import AFTER patching
    from claude_agent_sdk import query, ClaudeAgentOptions

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant",
        permission_mode="acceptEdits",
    )

    # Use the standalone query function (simpler API)
    response_text = []
    async for message in query(
        prompt="What is the square root of 144?",
        options=options,
    ):
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text.append(block.text)

    # Verify we got a response
    full_response = " ".join(response_text)
    assert "12" in full_response, f"Expected '12' in response, got: {full_response}"

    # Force flush to ensure all spans are processed
    mock_processor.force_flush()

    # Verify spans were created
    spans = mock_processor.ended_spans
    assert len(spans) > 0, "No spans were created"

    # Find and verify agent span
    agent_spans = find_spans_by_kind(spans, "agent")
    assert len(agent_spans) >= 1, "No agent span found"

    verify_agent_span(
        agent_spans[0], expected_input="square root of 144", check_output=True
    )

    # Find and verify LLM spans
    llm_spans = find_spans_by_kind(spans, "llm")
    assert len(llm_spans) >= 1, "No LLM spans found"

    verify_llm_span(
        llm_spans[0],
        expected_model="claude-sonnet",
        check_provider=True,
        check_usage=True,
    )

    print(f"\n✅ Test passed! Standalone query() created {len(spans)} spans:")
    print(f"   - {len(agent_spans)} agent span(s)")
    print(f"   - {len(llm_spans)} LLM span(s)")


@pytest.mark.asyncio
async def test_multiple_turns_real_api(tracer, mock_processor):
    """Test multi-turn conversation with real API."""
    # Setup integration FIRST
    setup_claude_agent_sdk(tracer=tracer)

    # Import AFTER patching
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-20250514",
    )

    # Make API call that should trigger multiple LLM calls
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "First tell me what 5+5 is, then tell me what 10+10 is. "
            "Give me each answer on a separate line."
        )

        async for message in client.receive_response():
            pass  # Just consume all messages

    # Force flush to ensure all spans are processed
    mock_processor.force_flush()

    # Verify multiple LLM spans were created
    llm_spans = find_spans_by_kind(mock_processor.ended_spans, "llm")
    assert len(llm_spans) >= 1, "Expected at least one LLM span"

    # All LLM spans should have usage metrics
    for llm_span in llm_spans:
        verify_llm_span(llm_span, check_usage=True)

    print(f"\n✅ Test passed! Created {len(llm_spans)} LLM span(s)")
