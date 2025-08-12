import os


JUDGMENT_ORG_ID = "27c7d380-6cb0-495f-b27c-8c1ff82b72af"
JUDGMENT_API_KEY = "736046e3-b3a2-4a1a-bd3e-af6a6f8a9a5d"
JUDGMENT_API_URL = "http://localhost:8000"
# JUDGMENT_API_URL = "https://staging.api.judgmentlabs.ai"

os.environ["JUDGMENT_API_KEY"] = JUDGMENT_API_KEY
os.environ["JUDGMENT_ORG_ID"] = JUDGMENT_ORG_ID
os.environ["JUDGMENT_API_URL"] = JUDGMENT_API_URL

from typing import TypedDict, Sequence
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)


class State(TypedDict):
    messages: Sequence[HumanMessage]


def node_1(state: State):
    # Simple node that processes messages
    messages = state["messages"]
    new_message = HumanMessage(content=f"Processed by node_1: {messages[0].content}")
    return {"messages": [new_message]}


def node_2(state: State):
    # Simple node that processes messages
    messages = state["messages"]
    new_message = HumanMessage(content=f"Processed by node_2: {messages[0].content}")
    return {"messages": [new_message]}


def run_graph():
    # Create tracer and handler
    judgment = Tracer(
        project_name="ahh",
        processors=[
            SimpleSpanProcessor(ConsoleSpanExporter()),
            BatchSpanProcessor(OTLPSpanExporter("http://localhost:4318/v1/traces")),
        ],
    )
    handler = JudgevalCallbackHandler(judgment)

    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("node_1", node_1)
    graph_builder.add_node("node_2", node_2)
    graph_builder.set_entry_point("node_1")
    graph_builder.add_edge("node_1", "node_2")
    graph_builder.add_edge("node_2", END)
    graph = graph_builder.compile()

    # Run the graph with our callback handler
    initial_state = State(messages=[HumanMessage(content="Hello!")])
    config_with_callbacks = RunnableConfig(callbacks=[handler])

    final_state = graph.invoke(initial_state, config=config_with_callbacks)

    print("Executed Nodes:", handler.executed_nodes)
    print("Executed Tools:", handler.executed_tools)
    print("Final State:", final_state)

    # Force flush the tracer to ensure spans are sent
    judgment.force_flush()


if __name__ == "__main__":
    run_graph()
