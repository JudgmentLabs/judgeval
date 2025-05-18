from uuid import uuid4
import openai
import os
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer
from judgeval.data import Example

client = wrap(openai.Client(api_key=os.getenv("OPENAI_API_KEY")))
tracer = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="travel_agent_demo")


@tracer.observe(span_type="tool")
def search_tavily(query):
    """Fetch travel data using Tavily API."""
    return "The weather in Tokyo is sunny with a high of 75°F."

# @judgment.observe(span_type="tool")
def get_attractions(destination):
    """Search for top attractions in the destination."""
    prompt = f"Best tourist attractions in {destination}"
    attractions_search = search_tavily(prompt)
    return attractions_search

# @judgment.observe(span_type="tool")
def get_hotels(destination):
    """Search for hotels in the destination."""
    prompt = f"Best hotels in {destination}"
    hotels_search = search_tavily(prompt)
    return hotels_search

# @judgment.observe(span_type="tool")
def get_flights(destination):
    """Search for flights to the destination."""
    prompt = f"Flights to {destination} from major cities"
    flights_search = search_tavily(prompt)
    return flights_search

# @judgment.observe(span_type="tool")
def get_weather(destination, start_date, end_date):
    """Search for weather information."""
    prompt = f"Weather forecast for {destination} from {start_date} to {end_date}"
    weather_search = search_tavily(prompt)
    example = Example(
        input="What is the weather in Tokyo?",
        actual_output=weather_search
    )
    tracer.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        example=example,
        model="gpt-4o-mini",
    )
    return weather_search

def research_destination(destination, start_date, end_date):
    """Gather all necessary travel information for a destination."""
    # First, check the vector database

    # Get real-time information from Tavily
    tavily_data = {
        "attractions": get_attractions(destination),
        "hotels": get_hotels(destination),
        "flights": get_flights(destination),
        "weather": get_weather(destination, start_date, end_date)
    }
    
    return {
        **tavily_data
    }

def create_travel_plan(destination, start_date, end_date, research_data):
    """Generate a travel itinerary using the researched data."""
    vector_db_context = "No pre-stored information available."
    
    prompt = f"""
    Create a structured travel itinerary for a trip to {destination} from {start_date} to {end_date}.
    
    Pre-stored destination information:
    {vector_db_context}
    
    Current travel data:
    - Attractions: {research_data['attractions']}
    - Hotels: {research_data['hotels']}
    - Flights: {research_data['flights']}
    - Weather: {research_data['weather']}
    """
    
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "You are an expert travel planner. Combine both historical and current information to create the best possible itinerary."},
    #         {"role": "user", "content": prompt}
    #     ]
    # ).choices[0].message.content
    
    return "Here is travel plan"

@tracer.observe(span_type="function")
def generate_itinerary(destination, start_date, end_date):
    """Main function to generate a travel itinerary."""
    research_data = research_destination(destination, start_date, end_date)
    res = create_travel_plan(destination, start_date, end_date, research_data)

from judgeval.scorers import ToolOrderScorer    
from judgeval import JudgmentClient

if __name__ == "__main__":
    judgment = JudgmentClient()
    example = Example(   
        input={"destination": "Paris", "start_date": "2025-06-01", "end_date": "2025-06-02"},
        expected_tools=[
            {
                "tool_name": "search_tavily",
                "parameters": {
                    "query": "Best tourist attractions in Paris"
                }
            },
            {
                "tool_name": "search_tavily",
                "parameters": {
                    "query": "Best hotels in Paris"
                }
            },
            {
                "tool_name": "search_tavily",
                "parameters": {
                    "query": "Flights to Paris from major cities"
                }
            },
            {
                "tool_name": "search_tavily",
                "parameters": {
                    "query": "Weather forecast for Paris from 2025-06-01 to 2025-06-02"
                }
            }
        ]
    )
    example2 = Example(
        input={"destination": "Tokyo", "start_date": "2025-06-01", "end_date": "2025-06-02"},
        expected_tools=[
            {"tool_name": "search_tavily", "parameters": {"query": "Best tourist attractions in Tokyo"}},
            {"tool_name": "search_tavily", "parameters": {"query": "Best hotels in Tokyo"}},
            {"tool_name": "search_tavily", "parameters": {"query": "Flights to Tokyo from major cities"}},
            {"tool_name": "search_tavily", "parameters": {"query": "Weather forecast for Tokyo from 2025-06-01 to 2025-06-03"}}
        ]
    )

    judgment.assert_test(
        project_name="travel_agent_demo",
        examples=[example, example2],
        scorers=[ToolOrderScorer(threshold=0.5)],
        model="gpt-4.1-mini",
        function=generate_itinerary,
        tracer=tracer,
        override=True
    )






