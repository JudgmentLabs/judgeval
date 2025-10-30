# Financial Agent Integration Guide

## Overview

This guide shows how to integrate the LangGraph OpenTelemetry callback handler with your financial agent to eliminate SpanContext warnings.

## Prerequisites

Based on testing, you'll need:

```bash
# Install dependencies
pip install judgeval langgraph langchain-openai langchain-core

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export JUDGMENT_API_KEY="your-judgment-key"  
export JUDGMENT_ORG_ID="your-org-id"
```

## Important: LangGraph V1.0 Update

If using LangGraph V1.0+, update your imports:

```python
# Old (deprecated in V1.0)
from langgraph.prebuilt import create_react_agent

# New (V1.0+)
from langchain.agents import create_agent
```

## Integration Steps

### Step 1: Import the Handler

Add to your financial agent code:

```python
from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import Langgraph

# Initialize tracer (usually at the top of your file)
tracer = Tracer(
    project_name="financial-agent",
    enable_monitoring=True,
    enable_evaluation=True,
)

# Get the callback handler
langgraph_handler = Langgraph.get_callback_handler(tracer, verbose=True)
```

### Step 2: Update Tool Definitions

Add `@tracer.observe()` to your tools:

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
@tracer.observe(span_type="tool")  # <-- Add this decorator
def get_stock_price(ticker: Annotated[str, "Stock ticker symbol"]) -> str:
    """Get the current stock price."""
    # Your implementation
    return f"Price of {ticker}: $150.00"

@tool
@tracer.observe(span_type="tool")  # <-- Add this decorator
def calculate_portfolio_value(
    holdings: Annotated[str, "Portfolio holdings"]
) -> str:
    """Calculate portfolio value."""
    # Your implementation
    return "Total value: $10,000"

# Repeat for all tools
```

### Step 3: Update Agent Execution

Wrap your agent execution function with `@tracer.observe()` and pass the handler:

```python
# Before
def run_financial_agent(query: str):
    agent = create_react_agent(model, tools)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result

# After
@tracer.observe(span_type="agent")  # <-- Add decorator
def run_financial_agent(query: str):
    agent = create_react_agent(model, tools)  # Or create_agent in V1.0+
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langgraph_handler]}  # <-- Pass handler
    )
    return result
```

### Step 4: Add Evaluation (Optional)

Add evaluation within the traced function:

```python
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer

@tracer.observe(span_type="agent")
def run_financial_agent(query: str):
    agent = create_react_agent(model, tools)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langgraph_handler]}
    )
    
    # Extract response
    messages = result.get("messages", [])
    response = messages[-1].content if messages else ""
    
    # Run evaluation
    tracer.async_evaluate(
        example=Example(
            input=query,
            actual_output=response,
        ),
        scorer=AnswerRelevancyScorer(),
        model="gpt-4o-mini",
        sampling_rate=1.0,
    )
    
    return result
```

### Step 5: Add Cleanup (Optional)

At the end of your script or in shutdown hooks:

```python
# Ensure all traces are exported before exit
tracer.force_flush(timeout_millis=10000)
```

## Complete Example Structure

Here's what your financial agent should look like after integration:

```python
"""
Financial Agent with Judgeval OpenTelemetry Integration
"""

import os
from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # or from langchain.agents import create_agent

from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import Langgraph
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer

# Initialize tracer
tracer = Tracer(
    project_name="financial-agent",
    enable_monitoring=True,
    enable_evaluation=True,
)

# Get callback handler
langgraph_handler = Langgraph.get_callback_handler(tracer, verbose=True)

# Define tools with @tracer.observe
@tool
@tracer.observe(span_type="tool")
def get_stock_price(ticker: Annotated[str, "Stock ticker"]) -> str:
    """Get stock price."""
    # Your implementation
    pass

@tool
@tracer.observe(span_type="tool")
def calculate_portfolio_value(holdings: Annotated[str, "Holdings"]) -> str:
    """Calculate portfolio value."""
    # Your implementation
    pass

# Add more tools...

# Create agent
def create_financial_agent():
    """Create the financial agent."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_stock_price, calculate_portfolio_value, ...]
    return create_react_agent(model, tools)

# Main execution function
@tracer.observe(span_type="agent")
def run_financial_agent(query: str) -> str:
    """Execute financial query with proper tracing."""
    agent = create_financial_agent()
    
    # CRITICAL: Pass the callback handler
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langgraph_handler]}  # <-- Key integration
    )
    
    # Extract response
    messages = result.get("messages", [])
    response = messages[-1].content if messages else ""
    
    # Optional: Run evaluation
    tracer.async_evaluate(
        example=Example(input=query, actual_output=response),
        scorer=AnswerRelevancyScorer(),
        model="gpt-4o-mini",
    )
    
    return response

# Main execution
if __name__ == "__main__":
    # Test queries
    queries = [
        "What is the price of Apple stock?",
        "Calculate my portfolio value",
        "What's your advice on retirement planning?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = run_financial_agent(query)
        print(f"Response: {response}")
    
    # Flush traces
    tracer.force_flush(timeout_millis=10000)
    print("\n✅ Execution complete - no SpanContext warnings!")
```

## Common Patterns

### With Memory/Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=memory)

@tracer.observe(span_type="agent")
def run_with_memory(query: str, thread_id: str):
    return agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={
            "callbacks": [langgraph_handler],
            "configurable": {"thread_id": thread_id}
        }
    )
```

### With Streaming

```python
@tracer.observe(span_type="agent")
def run_with_streaming(query: str):
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langgraph_handler]}
    ):
        yield chunk
```

### Async Execution

```python
@tracer.observe(span_type="agent")
async def run_async(query: str):
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langgraph_handler]}
    )
    return result
```

### Multi-Agent Orchestration

```python
@tracer.observe(span_type="orchestrator")
def orchestrate_analysis(ticker: str):
    """Orchestrate multiple agents."""
    
    # Each agent call gets the same handler
    price_result = run_financial_agent(f"Get price of {ticker}")
    analysis_result = run_financial_agent(f"Analyze {ticker}")
    
    return combine_results(price_result, analysis_result)
```

## Expected Span Hierarchy

After integration, you should see this hierarchy in your traces:

```
run_financial_agent (@tracer.observe)
├── langgraph.chain.AgentExecutor
│   ├── langgraph.llm.gpt-4o-mini
│   │   └── (LLM call - auto-traced by judgeval.wrap())
│   ├── langgraph.tool.get_stock_price
│   │   └── get_stock_price (@tracer.observe)
│   │       └── (your tool implementation)
│   ├── langgraph.llm.gpt-4o-mini
│   │   └── (LLM call)
│   └── langgraph.chain.AgentExecutor (final)
```

All spans will share the same `trace_id`, ensuring proper trace continuity.

## Verification Checklist

After integration, verify:

- [ ] No "SpanContext" warnings in logs
- [ ] No "invalid span context" errors
- [ ] Traces appear in Judgeval dashboard
- [ ] All spans have the same trace_id
- [ ] Tool spans are properly nested
- [ ] LLM calls are captured
- [ ] Evaluations run successfully

## Testing Without Real API Calls

To test the integration without making real API calls:

1. Set dummy API keys:
   ```bash
   export OPENAI_API_KEY="sk-test"
   export JUDGMENT_API_KEY="test-key"
   export JUDGMENT_ORG_ID="test-org"
   ```

2. Run the test suite:
   ```bash
   python3 test_langgraph_integration_simple.py
   ```

3. Should see: ✅ ALL TESTS PASSED! - NO SPAN CONTEXT ERRORS

## Troubleshooting

### Issue: SpanContext warnings still appear

**Solution**: Ensure you're passing the handler to every `invoke()` call:
```python
config={"callbacks": [langgraph_handler]}
```

### Issue: Tool spans not nested properly

**Solution**: Make sure tools have `@tracer.observe()` decorator AND the handler is passed to agent.invoke()

### Issue: Traces disconnected

**Solution**: Wrap your agent execution function with `@tracer.observe()`:
```python
@tracer.observe(span_type="agent")
def run_financial_agent(query: str):
    # ...
```

### Issue: ImportError with create_react_agent

**Solution**: Update to new import (LangGraph V1.0+):
```python
# Old
from langgraph.prebuilt import create_react_agent

# New
from langchain.agents import create_agent
```

## Performance Considerations

- **Minimal Overhead**: Only span creation/management
- **No Blocking**: All operations are non-blocking
- **Memory Safe**: Proper cleanup of spans
- **Scalable**: Handler can be reused

## Need More Help?

If you encounter issues:

1. Enable verbose mode: `Langgraph.get_callback_handler(tracer, verbose=True)`
2. Check the logs for detailed event information
3. Verify all three integration points:
   - Handler creation
   - Handler passed to invoke()
   - @tracer.observe() on agent function
4. Review the example: `example_langgraph_usage.py`
5. Run tests: `python3 test_langgraph_integration_simple.py`

## What Changed from Old Integration

If you were using `Langgraph.initialize()`:

**Old (doesn't work properly):**
```python
from judgeval.integrations.langgraph import Langgraph

Langgraph.initialize(otel_only=True)  # Just sets env vars
agent.invoke(...)  # Traces disconnected, SpanContext warnings
```

**New (works correctly):**
```python
from judgeval.integrations.langgraph import Langgraph

handler = Langgraph.get_callback_handler(tracer)

@tracer.observe()
def run_agent(query):
    return agent.invoke(..., config={"callbacks": [handler]})
```

---

**Ready to integrate!** Follow the steps above and you should have a fully traced financial agent with no SpanContext warnings.
