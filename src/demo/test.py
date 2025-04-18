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

@judgment.observe(span_type="graph")
def main():
    result = graph.invoke({
        "messages": [HumanMessage(content="What is 5 + 5?")]
    })

if __name__ == "__main__":
    main()