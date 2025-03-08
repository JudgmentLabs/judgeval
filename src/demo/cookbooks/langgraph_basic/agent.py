from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from judgeval.common.tracer import Tracer, wrap, JudgevalCallbackHandler
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer


import asyncio
import os
from typing import Any, Optional
from uuid import UUID

class State(TypedDict):
    messages: Annotated[list, add_messages]


judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))


def stream_graph_updates(graph,user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        print(f"Event: {event}")
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)


def tavily_search(query: str) -> str:
    """
    Tool that queries the Tavily Search API and gets back json.
    """
    return TavilySearchResults(max_results=2)


async def multiply(int_list: list[int]) -> int:
    """
    Multiples numbers together
    """
    product = 1
    for i in int_list:
        product *= i

    await judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=0.5)],
        input=str(int_list),
        actual_output=str(product),
        model="gpt-4o-mini",
        log_results=True
    )
    return product

async def main():
    with judgment.trace(
        "langgraph_run1",
        project_name="langgraph_basic",
        overwrite=True
    ) as trace:
        graph_builder = StateGraph(State)

        tavily_search = TavilySearchResults(max_results=2)
        tools = [tavily_search, multiply]
        llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
        llm_with_tools = llm.bind_tools(tools)

        def chatbot(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}


        graph_builder.add_node("chatbot", chatbot)

        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        # Any time a tool is called, we return to the chatbot to decide the next step

        graph_builder.add_edge("tools", "chatbot")
        graph_builder.set_entry_point("chatbot")
        graph = graph_builder.compile()

        handler = JudgevalCallbackHandler(trace)


        res = graph.invoke(
            {"messages": [{"role": "user", "content": "What is the amount of people in the US times 5 times number of people in China?"}]},
            config=dict(callbacks=[handler])
            )
        
        print(f"Result: {res}")
            

        # # stream_graph_updates(graph, "What is the amount of people in the US times 5 times number of people in China?")

        trace.save()
        # trace.print()



if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())