import pytest
from judgeval.common.tracer import Tracer, TraceManagerClient
from judgeval.integrations.langgraph import JudgevalCallbackHandler, set_global_handler
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os

class State(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]

PROJECT_NAME = "test-langgraph-project"

judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"), 
    project_name=PROJECT_NAME
)

llm = ChatOpenAI()

def process_message(state: State) -> State:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

graph_builder = StateGraph(State)

graph_builder.add_node("process", process_message)
graph_builder.set_entry_point("process")

def finish_node(state: State) -> State:
    return state

graph_builder.add_node("finish_node", finish_node)
graph_builder.add_edge("process", "finish_node")
graph_builder.set_finish_point("finish_node")

graph = graph_builder.compile()

handler = JudgevalCallbackHandler(judgment)
set_global_handler(handler)  # This will automatically trace your entire workflow

@pytest.fixture
def setup_graph():
    # Setup code for the graph
    return graph, handler

def test_graph_execution(setup_graph):
    graph, handler = setup_graph

    # Execute your workflow
    try:    
        result = graph.invoke({
            "messages": [HumanMessage(content="What is 5 + 5?")]
        })
    except Exception as e:
        pytest.fail(f"Execution raised an exception: {e}")

    # Assertions to verify the expected behavior
    assert "process" in handler.executed_nodes
    assert "finish_node" in handler.executed_nodes
    assert len(handler.executed_nodes) == 2
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)
    

    client = TraceManagerClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"), organization_id=os.getenv("JUDGMENT_ORG_ID"))
    trace = client.fetch_trace(handler._trace_id)

    assert trace
    assert trace["trace_id"] == handler._trace_id