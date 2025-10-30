# LangGraph + OpenTelemetry Integration Guide

## Overview

This integration enables proper OpenTelemetry (OTel) context propagation between Judgeval's tracer and LangGraph. It solves the common issue where LangGraph's internal tracing doesn't correctly parent spans with OTel spans, leading to "SpanContext" warnings and disconnected traces.

## The Problem

LangGraph and LangChain use LangSmith's internal tracing system, which doesn't natively propagate OpenTelemetry context. This causes:

1. **Disconnected Traces**: LangGraph spans don't appear as children of your OTel spans
2. **SpanContext Warnings**: Warnings about invalid or missing span contexts
3. **Lost Context**: Custom tools decorated with `@tracer.observe()` aren't properly nested

Reference: [LangSmith SDK Issue #1866](https://github.com/langchain-ai/langsmith-sdk/issues/1866)

## The Solution

Our `LangGraphCallbackHandler` creates OTel spans for each LangGraph event and ensures proper context propagation by:

1. Using OpenTelemetry's context API to maintain parent-child relationships
2. Creating spans within the current OTel context
3. Properly managing span lifecycle and cleanup

## Installation

The integration is included in Judgeval. Ensure you have the required dependencies:

```bash
pip install judgeval langgraph langchain-core langchain-openai
```

## Quick Start

```python
from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import Langgraph
from langgraph.prebuilt import create_react_agent

# 1. Initialize tracer
tracer = Tracer(project_name="my-project")

# 2. Get the callback handler
handler = Langgraph.get_callback_handler(tracer, verbose=True)

# 3. Create your agent
agent = create_react_agent(model, tools)

# 4. Wrap your agent call with @tracer.observe
@tracer.observe(span_type="agent")
def run_agent(query: str):
    return agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [handler]}  # <-- Pass handler here!
    )
```

## Complete Example

See `example_langgraph_usage.py` for a complete working example with:
- Tool definitions with `@tracer.observe()` decorators
- LangGraph agent setup
- Proper callback handler usage
- Evaluation integration

## Key Integration Points

### 1. Get the Callback Handler

```python
from judgeval.integrations.langgraph import Langgraph

handler = Langgraph.get_callback_handler(tracer, verbose=False)
```

### 2. Pass Handler to LangGraph

Always pass the handler in the `config` parameter:

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"callbacks": [handler]}
)
```

### 3. Wrap Agent Calls

Wrap your agent execution with `@tracer.observe()`:

```python
@tracer.observe(span_type="agent")
def run_agent(query: str):
    agent = create_agent()
    return agent.invoke(
        {"messages": [...]},
        config={"callbacks": [handler]}
    )
```

This creates a parent span that all LangGraph operations will be children of.

### 4. Decorate Custom Tools

Your custom tools should use `@tracer.observe()`:

```python
from langchain_core.tools import tool

@tool
@tracer.observe(span_type="tool")
def get_stock_price(symbol: str) -> str:
    """Get stock price."""
    # Your implementation
    return f"Price: ${price}"
```

## Span Hierarchy

The integration creates the following span hierarchy:

```
root_span (@tracer.observe)
├── langgraph.chain.AgentExecutor
│   ├── langgraph.llm.gpt-4o-mini
│   ├── langgraph.tool.get_stock_price
│   │   └── your_tool_implementation (@tracer.observe)
│   └── langgraph.llm.gpt-4o-mini
```

## Evaluations

Evaluations work seamlessly with the integration:

```python
@tracer.observe(span_type="agent")
def run_agent(query: str):
    result = agent.invoke(...)
    
    # Run evaluation on the span
    tracer.async_evaluate(
        example=Example(
            input=query,
            actual_output=response,
        ),
        scorer=AnswerRelevancyScorer(),
        model="gpt-4o-mini",
    )
    
    return response
```

The evaluation will be attached to the current span created by `@tracer.observe()`.

## Callback Handler API

### `Langgraph.get_callback_handler(tracer, verbose=False)`

Creates a callback handler for LangGraph integration.

**Parameters:**
- `tracer` (Tracer): Your Judgeval Tracer instance
- `verbose` (bool): Whether to log verbose information about events

**Returns:**
- `LangGraphCallbackHandler`: A configured callback handler

### Supported Events

The handler captures:
- **Chain events**: `on_chain_start`, `on_chain_end`, `on_chain_error`
- **LLM events**: `on_llm_start`, `on_llm_end`, `on_llm_error`
- **Tool events**: `on_tool_start`, `on_tool_end`, `on_tool_error`
- **Agent events**: `on_agent_action`, `on_agent_finish`

Each event creates an appropriate OTel span with relevant attributes.

## Troubleshooting

### Still seeing SpanContext warnings?

1. **Verify handler is passed**: Ensure `config={"callbacks": [handler]}` is set
2. **Check decorator**: Wrap agent calls with `@tracer.observe()`
3. **Enable verbose mode**: Set `verbose=True` to see event logs
4. **Check initialization**: Ensure tracer is initialized before creating handler

### Spans not appearing?

1. **Flush traces**: Call `tracer.force_flush()` at the end of your program
2. **Check monitoring**: Ensure `enable_monitoring=True` in Tracer initialization
3. **Verify API keys**: Check `JUDGMENT_API_KEY` and `JUDGMENT_ORG_ID` are set

### Disconnected traces?

1. **Missing handler**: Handler must be passed to every `invoke()` call
2. **Missing decorator**: Agent calls need `@tracer.observe()` wrapper
3. **Context issues**: Avoid mixing sync/async contexts incorrectly

## Migration from Old Integration

If you were using the old `Langgraph.initialize()` method:

**Before:**
```python
from judgeval.integrations.langgraph import Langgraph

Langgraph.initialize(otel_only=True)
agent.invoke(...)  # Traces were disconnected
```

**After:**
```python
from judgeval.integrations.langgraph import Langgraph

handler = Langgraph.get_callback_handler(tracer)

@tracer.observe()
def run_agent(query):
    return agent.invoke(
        {...},
        config={"callbacks": [handler]}
    )
```

## Testing

Run the test suite to verify the integration:

```bash
python3 test_langgraph_integration_simple.py
```

This verifies:
- ✓ Callback handler methods work
- ✓ Spans are tracked and cleaned up
- ✓ Context is propagated correctly
- ✓ Nested operations maintain trace context
- ✓ No SpanContext warnings appear

## Architecture

The integration works by:

1. **Span Creation**: For each LangGraph event (chain, tool, LLM), create an OTel span using `tracer.start_span()`

2. **Context Management**: Use OpenTelemetry's `context.attach()` to set the span as current, ensuring child operations use it as parent

3. **Lifecycle Management**: Track spans by `run_id` and properly end/cleanup when events complete

4. **Attribute Capture**: Extract and store relevant metadata (inputs, outputs, model info) as span attributes

## Best Practices

1. **Always use the handler**: Pass it to every `invoke()` call
2. **Wrap entry points**: Use `@tracer.observe()` on agent entry functions
3. **Decorate tools**: Add `@tracer.observe()` to custom tools
4. **Flush on exit**: Call `tracer.force_flush()` before program exit
5. **Enable verbose for debugging**: Use `verbose=True` when troubleshooting
6. **One handler per tracer**: Reuse the same handler instance

## Examples

### Basic Agent
See `example_langgraph_usage.py`

### Multi-Agent System
```python
handler = Langgraph.get_callback_handler(tracer)

@tracer.observe(span_type="orchestrator")
def run_multi_agent(query: str):
    # Agent 1
    result1 = agent1.invoke(..., config={"callbacks": [handler]})
    
    # Agent 2 uses result from Agent 1
    result2 = agent2.invoke(..., config={"callbacks": [handler]})
    
    return combine_results(result1, result2)
```

### Streaming
```python
@tracer.observe(span_type="agent")
async def stream_agent(query: str):
    async for chunk in agent.astream(
        {...},
        config={"callbacks": [handler]}
    ):
        yield chunk
```

## Support

- **Documentation**: https://docs.judgmentlabs.ai
- **Issues**: https://github.com/JudgmentLabs/judgeval/issues
- **LangSmith Context Issue**: https://github.com/langchain-ai/langsmith-sdk/issues/1866

## Related

- OpenTelemetry Python: https://opentelemetry-python.readthedocs.io
- LangGraph: https://github.com/langchain-ai/langgraph
- LangChain Callbacks: https://python.langchain.com/docs/modules/callbacks/
