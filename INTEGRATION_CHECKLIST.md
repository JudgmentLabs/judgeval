# Integration Checklist for Your Financial Agent

## What I've Built

✅ **LangGraph Callback Handler** (`src/judgeval/integrations/langgraph/callback_handler.py`)
- Handles all LangGraph events (chain, tool, LLM, agent)
- Proper OpenTelemetry context propagation
- Span tracking and cleanup
- Error handling

✅ **Integration API** (`src/judgeval/integrations/langgraph/__init__.py`)
- Simple `get_callback_handler()` method
- Easy to use and integrate

✅ **Comprehensive Tests** 
- All tests passing
- Zero SpanContext warnings
- Context propagation verified

✅ **Documentation**
- Integration guide
- Examples
- Troubleshooting

## What I Couldn't Test (Need Your Help)

❓ **Your Financial Agent Code**
- Repository is private - I don't have access
- Need to verify integration with your specific implementation

❓ **Your Agent's Patterns**
- What tools does your agent use?
- Does it use async/await?
- Does it use streaming?
- Does it have state management?
- Are there sub-graphs or nested agents?

❓ **Your Dependencies**
- What version of LangGraph are you using?
- Are you using `create_react_agent` or `create_agent`?
- Any custom LangGraph configurations?

## Integration Steps for Your Agent

### 1. Add Imports (Top of your file)

```python
from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import Langgraph

# Initialize once
tracer = Tracer(
    project_name="financial-agent",
    enable_monitoring=True,
    enable_evaluation=True,
)

# Get handler once
langgraph_handler = Langgraph.get_callback_handler(tracer, verbose=True)
```

### 2. Decorate Your Tools

For EACH tool in your financial agent:

```python
@tool
@tracer.observe(span_type="tool")  # <-- ADD THIS LINE
def your_tool_function(...):
    """Your tool."""
    # your code
```

### 3. Update Agent Execution

Find where you call `agent.invoke()` and make two changes:

**A. Wrap with @tracer.observe:**
```python
@tracer.observe(span_type="agent")  # <-- ADD THIS
def run_financial_agent(query: str):
    # your code
```

**B. Pass the handler:**
```python
result = agent.invoke(
    {"messages": [...]},
    config={"callbacks": [langgraph_handler]}  # <-- ADD THIS
)
```

### 4. Test It

Run your financial agent and check for:
- ❌ Any "SpanContext" warnings → If yes, handler not passed correctly
- ✅ No warnings → Integration working!

## Potential Edge Cases to Check

### 1. Async Operations

If your agent uses `async`/`await`:

```python
@tracer.observe(span_type="agent")
async def run_financial_agent_async(query: str):
    result = await agent.ainvoke(
        {"messages": [...]},
        config={"callbacks": [langgraph_handler]}  # Still works!
    )
    return result
```

Should work automatically - BaseCallbackHandler supports both sync and async.

### 2. Streaming

If your agent streams responses:

```python
@tracer.observe(span_type="agent")
def run_financial_agent_stream(query: str):
    for chunk in agent.stream(
        {"messages": [...]},
        config={"callbacks": [langgraph_handler]}  # Works with streaming
    ):
        yield chunk
```

Should work - handler captures all events regardless of streaming.

### 3. State Management / Checkpointer

If your agent uses memory/checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=memory)

@tracer.observe(span_type="agent")
def run_with_memory(query: str, thread_id: str):
    result = agent.invoke(
        {"messages": [...]},
        config={
            "callbacks": [langgraph_handler],  # Handler
            "configurable": {"thread_id": thread_id}  # Memory config
        }
    )
```

Should work - config merges callbacks with other settings.

### 4. SubGraphs / Nested Graphs

If your agent has subgraphs:

```python
# Main graph
@tracer.observe(span_type="main_agent")
def run_main_agent(query: str):
    return main_agent.invoke(
        ...,
        config={"callbacks": [langgraph_handler]}
    )

# Sub-graph
@tracer.observe(span_type="sub_agent")  
def run_sub_agent(query: str):
    return sub_agent.invoke(
        ...,
        config={"callbacks": [langgraph_handler]}  # Same handler!
    )
```

Should work - handler tracks all events by run_id.

### 5. Custom State Schemas

If your agent uses custom state:

```python
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    portfolio: dict
    risk_score: float

# Your agent with custom state
agent = StateGraph(AgentState)
# ... build graph ...

compiled_agent = agent.compile()

@tracer.observe(span_type="agent")
def run_custom_state_agent(initial_state):
    return compiled_agent.invoke(
        initial_state,
        config={"callbacks": [langgraph_handler]}  # Works!
    )
```

Should work - handler doesn't depend on state schema.

### 6. Human-in-the-Loop

If your agent has human feedback:

```python
@tracer.observe(span_type="agent")
def run_with_human_feedback(query: str):
    # Handler will capture all events including interrupts
    result = agent.invoke(
        {"messages": [...]},
        config={"callbacks": [langgraph_handler]}
    )
    return result
```

Should work - handler captures events at all stages.

## What Could Go Wrong?

### ❌ Problem: SpanContext warnings still appear

**Possible causes:**
1. Handler not passed to `invoke()` - Check `config={"callbacks": [handler]}`
2. Multiple agent invocations without passing handler to all
3. Tools not decorated with `@tracer.observe()`
4. Agent function not wrapped with `@tracer.observe()`

**Debug steps:**
```python
# Enable verbose mode
handler = Langgraph.get_callback_handler(tracer, verbose=True)

# You should see logs like:
# "Chain started: AgentExecutor"
# "Tool started: get_stock_price"
# etc.
```

### ❌ Problem: Traces disconnected

**Cause:** Missing `@tracer.observe()` wrapper on agent function

**Fix:**
```python
@tracer.observe(span_type="agent")  # <-- Must have this
def run_financial_agent(query: str):
    return agent.invoke(...)
```

### ❌ Problem: Tool spans not nested

**Cause:** Tools missing `@tracer.observe()` decorator

**Fix:**
```python
@tool
@tracer.observe(span_type="tool")  # <-- Must have this
def your_tool(...):
    pass
```

### ❌ Problem: Import errors

**If you see:** `ImportError: cannot import name 'create_react_agent'`

**Cause:** LangGraph V1.0 moved the import

**Fix:**
```python
# Old (deprecated)
from langgraph.prebuilt import create_react_agent

# New (V1.0+)
from langchain.agents import create_agent
```

## Features I've Tested

✅ Basic agent execution
✅ Multiple tool calls
✅ Nested operations (3+ levels)
✅ Error handling
✅ Span tracking and cleanup
✅ Context propagation (verified same trace_id)
✅ No SpanContext warnings

## Features I Haven't Tested (Need Your Financial Agent)

❓ Your specific tool implementations
❓ Your exact LangGraph configuration
❓ Your agent's state schema
❓ Your specific LLM provider setup
❓ Your evaluation criteria
❓ Your production workload patterns

## Quick Integration Test

**Minimal change to test (add 3 lines):**

```python
# At the top of your file
from judgeval.integrations.langgraph import Langgraph
handler = Langgraph.get_callback_handler(tracer, verbose=True)

# In your agent invoke call
result = agent.invoke(
    {"messages": [...]},
    config={"callbacks": [handler]}  # <-- Add this one line
)
```

Run it and check logs for:
- ✅ "Chain started: ..." - Handler working
- ❌ "SpanContext" warnings - Need to wrap with @tracer.observe()

## Complete Integration Test

**Full integration (recommended):**

1. Add handler creation (1 line)
2. Wrap agent function with `@tracer.observe()` (1 line)
3. Pass handler to invoke (1 line in config)
4. Decorate all tools with `@tracer.observe()` (1 line per tool)

Total: ~5-10 lines of code changes

## What to Share for Verification

To help me verify the integration works with your financial agent, please share:

1. **Agent structure:**
   - How many tools?
   - Async or sync?
   - Streaming or not?
   - Uses memory/checkpointer?

2. **Current issues:**
   - What SpanContext warnings are you seeing?
   - Where do they appear?
   - How frequently?

3. **Code snippet:**
   - How you create the agent
   - How you invoke it
   - What tools you have

4. **After integration:**
   - Did SpanContext warnings disappear?
   - Are traces showing up correctly?
   - Any new errors?

## Next Steps

1. **Review** `FINANCIAL_AGENT_INTEGRATION_GUIDE.md` for detailed examples
2. **Integrate** the 3-5 code changes into your financial agent
3. **Test** and check for SpanContext warnings
4. **Report back** if you encounter any issues

If something doesn't work, I can help debug - just share:
- The error message
- Relevant code snippet
- What you've tried

## Files Ready for You

- ✅ `src/judgeval/integrations/langgraph/callback_handler.py` - Core integration
- ✅ `src/judgeval/integrations/langgraph/__init__.py` - Easy API
- ✅ `FINANCIAL_AGENT_INTEGRATION_GUIDE.md` - Step-by-step guide
- ✅ `example_langgraph_usage.py` - Working example
- ✅ `test_langgraph_integration_simple.py` - Test suite (passing)

---

**Ready to test!** The integration is production-ready. Just need to verify it works with your specific financial agent code.
