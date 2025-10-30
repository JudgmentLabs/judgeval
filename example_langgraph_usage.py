"""
Example: Using LangGraph with Judgeval OpenTelemetry Integration

This example demonstrates how to properly integrate LangGraph with
Judgeval's OpenTelemetry tracing to ensure proper context propagation
and avoid SpanContext warnings.

Key Points:
1. Use Langgraph.get_callback_handler() to get the callback handler
2. Pass the handler to LangGraph's config parameter
3. Wrap your agent calls with @tracer.observe() for parent span
4. All LangGraph events will be children of your parent span
"""

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import Langgraph
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer


# Initialize the Judgeval tracer
tracer = Tracer(
    project_name="financial-agent-demo",
    enable_monitoring=True,
    enable_evaluation=True,
)

# Get the LangGraph callback handler for OpenTelemetry integration
# This ensures proper context propagation
langgraph_handler = Langgraph.get_callback_handler(tracer, verbose=True)


# Define your tools with @tracer.observe decorator
# These will automatically be traced and nested properly
@tool
@tracer.observe(span_type="tool")
def get_stock_price(symbol: Annotated[str, "Stock ticker symbol"]) -> str:
    """Get the current stock price for a given ticker symbol."""
    # Your implementation here
    mock_prices = {"AAPL": "$175.50", "GOOGL": "$140.25", "MSFT": "$380.00"}
    return f"Current price of {symbol}: {mock_prices.get(symbol.upper(), '$100.00')}"


@tool
@tracer.observe(span_type="tool")
def calculate_portfolio_value(
    holdings: Annotated[str, "Comma-separated list of ticker:quantity pairs"]
) -> str:
    """Calculate total portfolio value (e.g., 'AAPL:10,GOOGL:5')."""
    # Your implementation here
    return "Total portfolio value: $2,456.25"


@tool
@tracer.observe(span_type="tool") 
def get_financial_advice(
    goal: Annotated[str, "Financial goal"]
) -> str:
    """Get financial advice based on a specific goal."""
    # Your implementation here
    return "Consider diversifying your portfolio with index funds."


# Create your LangGraph agent
def create_agent():
    """Create a financial agent using LangGraph."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_stock_price, calculate_portfolio_value, get_financial_advice]
    return create_react_agent(llm, tools)


# IMPORTANT: Wrap your agent execution with @tracer.observe
# This creates a parent span for all LangGraph operations
@tracer.observe(span_type="agent")
def run_financial_agent(query: str) -> str:
    """
    Execute a financial query through the agent.
    
    The @tracer.observe decorator creates a parent span.
    All LangGraph operations will be children of this span
    thanks to the callback handler.
    
    Args:
        query: The user's financial question
        
    Returns:
        The agent's response
    """
    agent = create_agent()
    
    # CRITICAL: Pass the callback handler in the config
    # This ensures LangGraph events create OTel spans as children
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [langgraph_handler]}  # <-- Key integration point!
    )
    
    # Extract response
    messages = result.get("messages", [])
    response = messages[-1].content if messages else "No response"
    
    # Optional: Run evaluation
    tracer.async_evaluate(
        example=Example(
            input=query,
            actual_output=response,
        ),
        scorer=AnswerRelevancyScorer(),
        model="gpt-4o-mini",
        sampling_rate=1.0,
    )
    
    return response


def main():
    """Example usage."""
    # Example queries
    queries = [
        "What is the current price of Apple stock?",
        "Calculate my portfolio value if I have 10 AAPL and 5 GOOGL",
        "What advice do you have for retirement planning?",
    ]
    
    print("="*80)
    print("FINANCIAL AGENT WITH LANGGRAPH + OTEL INTEGRATION")
    print("="*80)
    
    for query in queries:
        print(f"\n📊 Query: {query}")
        print("-"*80)
        
        try:
            response = run_financial_agent(query)
            print(f"💡 Response: {response}")
            print("✓ No SpanContext warnings!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Flush to ensure all traces are exported
    tracer.force_flush(timeout_millis=10000)
    
    print("\n" + "="*80)
    print("✅ DONE - Check your Judgeval dashboard for traces!")
    print("="*80)


if __name__ == "__main__":
    # Set your API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    if not os.getenv("JUDGMENT_API_KEY"):
        print("⚠️  Set JUDGMENT_API_KEY environment variable")
        print("Example: export JUDGMENT_API_KEY='your-key-here'")
        exit(1)
    
    main()
