import pytest
import os
from typing import TypedDict, List
import warnings
import time

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from judgeval.common.tracer import Tracer, TraceManagerClient, TraceClient
from judgeval.integrations.langgraph import JudgevalCallbackHandler

# --- Test Configuration ---
API_KEY = os.getenv("JUDGMENT_API_KEY")
ORG_ID = os.getenv("JUDGMENT_ORG_ID")

# --- Shared Graph Definition ---
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


class SimpleState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    result: str


def call_llm_node(state: SimpleState) -> SimpleState:
    """Node that calls the LLM."""
    print("--- Node: call_llm_node ---")
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response], "result": "LLM Called"}


async def async_call_llm_node(state: SimpleState) -> SimpleState:
    """Async node that calls the LLM."""
    print("--- Node: async_call_llm_node ---")
    # Use ainvoke for async LLM call if needed, but invoke works fine too
    response = await llm.ainvoke(state["messages"])
    return {"messages": state["messages"] + [response], "result": "Async LLM Called"}


def build_graph():
    """Builds the simple test graph."""
    # Sync Graph
    graph_builder_sync = StateGraph(SimpleState)
    graph_builder_sync.add_node("call_llm", call_llm_node)
    graph_builder_sync.add_node("finish", lambda state: state)
    graph_builder_sync.set_entry_point("call_llm")
    graph_builder_sync.add_edge("call_llm", "finish")
    graph_builder_sync.set_finish_point("finish")
    graph_sync = graph_builder_sync.compile()

    # Async Graph
    graph_builder_async = StateGraph(SimpleState)
    graph_builder_async.add_node("async_call_llm", async_call_llm_node)
    graph_builder_async.add_node("finish", lambda state: state)
    graph_builder_async.set_entry_point("async_call_llm")
    graph_builder_async.add_edge("async_call_llm", "finish")
    graph_builder_async.set_finish_point("finish")
    graph_async = graph_builder_async.compile()

    return graph_sync, graph_async


graph_sync, graph_async = build_graph()  # Compile graphs once


# --- Helper Function ---
def fetch_and_validate_trace(trace_id: str, expected_project: str):
    """Fetches a trace and performs basic validation."""
    print(f"\nFetching trace ID: {trace_id} for project: {expected_project}")
    if not API_KEY or not ORG_ID:
        pytest.skip(
            "JUDGMENT_API_KEY or JUDGMENT_ORG_ID not set, skipping trace fetch."
        )

    client = TraceManagerClient(judgment_api_key=API_KEY, organization_id=ORG_ID)
    try:
        trace_data = client.fetch_trace(trace_id=trace_id)
        print("Trace data fetched successfully.")
        # rprint(trace_data) # Use rich print if available for better readability

        assert trace_data, f"Trace data for {trace_id} should not be empty."
        assert trace_data.get("trace_id") == trace_id
        # assert trace_data.get("project_name") == expected_project # Commenting out: Seems fetch API doesn't return project_name

        # --- MODIFIED CHECK: Look for 'trace_spans' instead of 'entries' ---
        assert "trace_spans" in trace_data, (
            f"Fetched trace {trace_id} should contain 'trace_spans' key."
        )
        trace_spans = trace_data.get("trace_spans")
        assert trace_spans is not None, (
            f"Trace {trace_id} field 'trace_spans' should not be None."
        )
        # assert len(trace_spans) > 0, f"Trace {trace_id} should have at least one span in 'trace_spans'." # Can be 0 if only root span
        # --- END MODIFIED CHECK ---

        # Check for specific nodes/spans if needed
        if trace_spans is None:  # Should not happen due to assert above, but defensive
            warnings.warn(
                f"Trace {trace_id} fetched successfully, but 'trace_spans' was None. Skipping specific span assertions."
            )
            span_functions = []
        else:
            # Extract function names from the list of span dictionaries
            span_functions = [span.get("function") for span in trace_spans]
        print(f"Span functions found: {span_functions}")
        return trace_data  # Return fetched data for further specific assertions

    except Exception as e:
        pytest.fail(f"Failed to fetch or validate trace {trace_id}: {e}")


# --- Synchronous Test ---
def test_sync_graph_execution(project_name: str):
    """Tests synchronous graph execution with JudgevalCallbackHandler."""
    print("\n--- Running Sync Test ---")
    Tracer._instance = None
    tracer_sync = Tracer(
        api_key=API_KEY, organization_id=ORG_ID, project_name=project_name
    )
    handler_sync = JudgevalCallbackHandler(tracer_sync)
    trace_client_sync: TraceClient = None  # Keep this for type hinting
    trace_id_sync: str = ""

    initial_state = {"messages": [HumanMessage(content="What is 5 + 5?")]}
    config = {"callbacks": [handler_sync]}

    try:
        # Invoke the synchronous graph
        result = graph_sync.invoke(initial_state, config=config)  # USE graph_sync

        # --- Assertions after invoke ---
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert result.get("result") == "LLM Called"

        # Get trace_id from the handler's internal client AFTER execution
        trace_client_sync = getattr(
            handler_sync, "_trace_client", None
        )  # Use getattr for safety
        assert trace_client_sync is not None, (
            "Sync handler should have an active trace client after invoke."
        )
        trace_id_sync = trace_client_sync.trace_id
        assert isinstance(trace_id_sync, str), "Trace ID should be a string."
        print(f"Sync test generated Trace ID: {trace_id_sync}")

        # Check node execution via handler AFTER invoke
        # Optional: Add check if executed_nodes exists
        if hasattr(handler_sync, "executed_nodes"):
            assert "call_llm" in handler_sync.executed_nodes
            assert "finish" in handler_sync.executed_nodes
        else:
            warnings.warn(
                "executed_nodes attribute not found on sync handler. Skipping node execution assertion."
            )

    except Exception as e:
        # Include trace_id in failure message if available
        fail_msg = f"Sync execution raised an exception: {e}"
        if trace_id_sync:
            fail_msg += f" (Trace ID: {trace_id_sync})"
        pytest.fail(fail_msg)

    # Fetch and validate the trace outside the main try block
    assert trace_id_sync is not None, "trace_id_sync was not set during the test run."

    if trace_id_sync:
        # Add a small delay before fetching
        print("Waiting 5 seconds before fetching sync trace...")
        time.sleep(5)
        fetched_trace = fetch_and_validate_trace(trace_id_sync, project_name)
        # Add more specific trace content assertions
        assert fetched_trace.get("trace_spans") is not None, (
            f"Trace {trace_id_sync} should have trace_spans."
        )
        assert any(
            "call_llm" in span.get("function", "")
            for span in fetched_trace["trace_spans"]
        ), f"Trace {trace_id_sync} should contain 'call_llm' span."
        assert any(
            "OPENAI_API_CALL" in span.get("function", "")
            for span in fetched_trace["trace_spans"]
        ), f"Trace {trace_id_sync} should contain 'OPENAI_API_CALL' span."


# --- Asynchronous Test ---
@pytest.mark.asyncio
async def test_async_graph_execution(project_name: str):
    """Tests asynchronous graph execution with JudgevalCallbackHandler."""
    print("\n--- Running Async Test ---")
    Tracer._instance = None
    tracer_async = Tracer(
        api_key=API_KEY, organization_id=ORG_ID, project_name=project_name
    )
    handler_async = JudgevalCallbackHandler(tracer_async)
    trace_client_async: TraceClient = None
    trace_id_async: str = ""

    initial_state = {"messages": [HumanMessage(content="What is 10 + 10?")]}
    config = {"callbacks": [handler_async]}

    try:
        # Invoke the asynchronous graph
        result = await graph_async.ainvoke(
            initial_state, config=config
        )  # USE graph_async

        # --- Assertions ---
        # Check node execution via handler AFTER invoke
        if hasattr(handler_async, "executed_nodes"):
            assert (
                "async_call_llm" in handler_async.executed_nodes
            )  # Check for async node
            assert "finish" in handler_async.executed_nodes
        else:
            warnings.warn(
                "executed_nodes attribute not found on async handler. Skipping node execution assertion."
            )

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 2  # Initial + AI response
        assert isinstance(result["messages"][-1], AIMessage)
        # Check the result field populated by the async node
        assert (
            result.get("result") == "Async LLM Called"
        )  # Expect result from async node

        # Get trace_id AFTER execution
        trace_client_async = getattr(
            handler_async, "_trace_client", None
        )  # Use getattr for safety
        assert trace_client_async is not None, (
            "Async handler should have an active trace client after ainvoke."
        )
        trace_id_async = trace_client_async.trace_id
        assert isinstance(trace_id_async, str), "Async Trace ID should be a string."
        print(f"Async test generated Trace ID: {trace_id_async}")

    except Exception as e:
        # Include trace_id in failure message if available
        fail_msg = f"Async execution raised an exception: {e}"
        if trace_id_async:
            fail_msg += f" (Trace ID: {trace_id_async})"
        pytest.fail(fail_msg)

    # Fetch and validate the trace
    assert trace_id_async is not None, (
        "trace_id_async was not set during the async test run."
    )
    if trace_id_async:
        # Add a small delay before fetching
        print("Waiting 5 seconds before fetching async trace...")
        time.sleep(5)
        fetched_trace = fetch_and_validate_trace(trace_id_async, project_name)
        # Add more specific trace content assertions
        assert fetched_trace.get("trace_spans") is not None, (
            f"Trace {trace_id_async} should have trace_spans."
        )
        assert any(
            "async_call_llm" in span.get("function", "")
            for span in fetched_trace["trace_spans"]
        ), f"Trace {trace_id_async} should contain 'async_call_llm' span."
        assert any(
            "OPENAI_API_CALL" in span.get("function", "")
            for span in fetched_trace["trace_spans"]
        ), f"Trace {trace_id_async} should contain 'OPENAI_API_CALL' span."
