from typing import Annotated, List
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from judgeval.common.tracer import Tracer, wrap
from judgeval.integrations.langgraph import JudgevalCallbackHandler
import os
from judgeval.data import Example
from judgeval.data.datasets import EvalDataset
from judgeval.scorers import AnswerRelevancyScorer, ExecutionOrderScorer, AnswerCorrectnessScorer
from judgeval import JudgmentClient
from pydantic import BaseModel
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

PROJECT_NAME = "JNPR_MIST_LANGGRAPH"

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name=PROJECT_NAME)


# @judgment.observe(name="search_restaurants", span_type="tool")
def search_restaurants(location: str, cuisine: str) -> str:
    """Search for restaurants in a location with specific cuisine"""
    ans = f"Top 3 {cuisine} restaurants in {location}: 1. Le Gourmet 2. Spice Palace 3. Carbones"
    example = Example(
        input="Search for restaurants in a location with specific cuisine",
        actual_output=ans
    )
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        example=example,
        model="gpt-4.1-mini"
    )
    return ans

# @judgment.observe(name="check_opening_hours", span_type="tool")
def check_opening_hours(restaurant: str) -> str:
    """Check opening hours for a specific restaurant"""
    ans = f"{restaurant} hours: Mon-Sun 11AM-10PM"
    example = Example(
        input="Check opening hours for a specific restaurant",
        actual_output=ans,
        expected_output=ans
    )
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1)],
        example=example,
        model="gpt-4.1-mini"
    )
    return ans

# @judgment.observe(name="get_menu_items", span_type="tool")
def get_menu_items(restaurant: str) -> str:
    """Get popular menu items for a restaurant"""
    ans = f"{restaurant} popular dishes: 1. Chef's Special 2. Seafood Platter 3. Vegan Delight"
    example = Example(
        input="Get popular menu items for a restaurant",
        actual_output=ans
    )
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        example=example,
        model="gpt-4.1-mini"
    )
    return ans 



# @judgment.observe(name="ask_human", span_type="tool")
def ask_human(state):
    """Ask the human a question about location"""
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    location = interrupt("Please provide your location:")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    elif last_message.tool_calls[0]["name"] == "ask_human":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "tools"



# @judgment.observe(span_type="Run Agent", overwrite=True)
def run_agent(prompt: str, follow_up_inputs: dict):
    tools = [
        TavilySearchResults(max_results=2),
        check_opening_hours,
        get_menu_items,
        search_restaurants,
    ]


    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    llm_with_tools = llm.bind_tools(tools + [ask_human])

    graph_builder = StateGraph(State)

    def assistant(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("ask_human", ask_human)

    graph_builder.set_entry_point("assistant")
    graph_builder.add_conditional_edges(
        "assistant",
        should_continue
    )
    graph_builder.add_edge("tools", "assistant")
    graph_builder.add_edge("ask_human", "assistant")
    
    checkpointer = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    handler = JudgevalCallbackHandler(judgment)
    config = {"configurable": {"thread_id": "001"}, "callbacks": [handler]}

    for event in graph.stream(
        {
            "messages": [
                (
                    "user",
                    prompt,
                )
            ]
        },
        config,
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()

    next_node = graph.get_state(config).next
    if next_node:
        print("Resuming from checkpoint")
        print(next_node)
        node_name_to_resume = next_node[0]
        # Check if the required key exists in follow_up_inputs
        if node_name_to_resume in follow_up_inputs:
            input_value = follow_up_inputs[node_name_to_resume]
            print(f"Resuming with input for '{node_name_to_resume}': {input_value}")
            for event in graph.stream(Command(resume=f"{input_value}"), config, stream_mode="values"):
                event["messages"][-1].pretty_print()
        else:
            print(f"Warning: Required follow-up input for node '{node_name_to_resume}' not found in follow_up_inputs. Skipping resume.")
            # Optionally, handle the missing input case differently, e.g., raise an error

    return handler
    

def test_eval_dataset():
    dataset = EvalDataset()

    # Helper to configure tests with YAML
    dataset.add_from_yaml(os.path.join(os.path.dirname(__file__), "test.yaml"))
    
    for example in dataset.examples:
        # Run your agent here
        follow_up = getattr(example, 'follow_up_inputs', {})
        handler = run_agent(example.input, follow_up)
        example.actual_output = handler.executed_node_tools

    client = JudgmentClient()
    client.run_evaluation(
        examples=dataset.examples,
        scorers=[ExecutionOrderScorer(threshold=1, should_consider_ordering=True)],
        model="gpt-4.1-mini",
        project_name=PROJECT_NAME,
        eval_run_name="mist-demo-examples",
        override=True
    )


if __name__ == "__main__":
    test_eval_dataset()