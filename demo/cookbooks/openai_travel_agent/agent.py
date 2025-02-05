import openai
import requests
import os
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import json

from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer
from demo.cookbooks.openai_travel_agent.populate_db import destinations_data

judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))


def populate_vector_db(collection, destinations_data):
    """
    Populate the vector DB with travel information.
    destinations_data should be a list of dictionaries with 'destination' and 'information' keys
    """
    for data in destinations_data:
        collection.add(
            documents=[data['information']],
            metadatas=[{"destination": data['destination']}],
            ids=[f"destination_{data['destination'].lower().replace(' ', '_')}"]
        )

@judgment.observe(span_type="search tool")
def search_tavily(query):
    """Fetch travel data using Tavily API."""
    API_KEY = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=API_KEY)
    results = client.search(query, num_results=3)
    return results

@judgment.observe(span_type="tool")
async def get_attractions(destination):
    """Search for top attractions in the destination."""
    prompt = f"Best tourist attractions in {destination}"
    attractions_search = search_tavily(prompt)
    await judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=prompt,
        actual_output=str(attractions_search),
        model="gpt-4o-mini",
        log_results=True
    )
    return attractions_search

@judgment.observe(span_type="tool")
async def get_hotels(destination):
    """Search for hotels in the destination."""
    prompt = f"Best hotels in {destination}"
    hotels_search = search_tavily(prompt)
    await judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=prompt,
        actual_output=str(hotels_search),
        model="gpt-4o-mini",
        log_results=True
    )
    return hotels_search

@judgment.observe(span_type="tool")
async def get_flights(destination):
    """Search for flights to the destination."""
    prompt = f"Flights to {destination} from major cities"
    flights_search = search_tavily(prompt)
    await judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=prompt,
        actual_output=str(flights_search),
        model="gpt-4o-mini",
        log_results=True
    )
    return flights_search

@judgment.observe(span_type="tool")
async def get_weather(destination, start_date, end_date):
    """Search for weather information."""
    prompt = f"Weather forecast for {destination} from {start_date} to {end_date}"
    weather_search = search_tavily(prompt)
    # await judgment.get_current_trace().async_evaluate(
    #     scorers=[AnswerRelevancyScorer(threshold=0.5)],
    #     input=prompt,
    #     actual_output=str(weather_search),
    #     model="gpt-4o-mini",
    #     log_results=True
    # )
    return weather_search

@judgment.observe(span_type="Retriever")
def initialize_vector_db():
    """Initialize ChromaDB with OpenAI embeddings."""
    client = chromadb.Client()
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    res = client.get_or_create_collection(
        "travel_information",
        embedding_function=embedding_fn
    )
    populate_vector_db(res, destinations_data)
    return res

@judgment.observe(span_type="Retriever")
def query_vector_db(collection, destination, k=3):
    """Query the vector database for existing travel information."""
    try:
        results = collection.query(
            query_texts=[destination],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []
    except Exception:
        return []

@judgment.observe(span_type="function")
async def research_destination(destination, start_date, end_date):
    """Gather all necessary travel information for a destination."""
    # First, check the vector database
    collection = initialize_vector_db()
    existing_info = query_vector_db(collection, destination)
    
    # Get real-time information from Tavily
    tavily_data = {
        "attractions": await get_attractions(destination),
        "hotels": await get_hotels(destination),
        "flights": await get_flights(destination),
        "weather": await get_weather(destination, start_date, end_date)
    }
    
    return {
        "vector_db_results": existing_info,
        **tavily_data
    }

@judgment.observe(span_type="function")
async def create_travel_plan(destination, start_date, end_date, research_data):
    """Generate a travel itinerary using the researched data."""
    vector_db_context = "\n".join(research_data['vector_db_results']) if research_data['vector_db_results'] else "No pre-stored information available."
    
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
    
    client = wrap(openai.Client(api_key=os.getenv("OPENAI_API_KEY")))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert travel planner. Combine both historical and current information to create the best possible itinerary."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content
    

    await judgment.get_current_trace().async_evaluate(
        scorers=[
            FaithfulnessScorer(threshold=0.5)
        ],
        input="",
        actual_output=response,
        retrieval_context=[
            vector_db_context, 
            str(research_data['attractions']), 
            str(research_data['hotels']), 
            str(research_data['flights']), 
            str(research_data['weather'])
        ],
        model="gpt-4o-mini",
        log_results=True
    )
    return response


async def generate_itinerary(destination, start_date, end_date):
    """Main function to generate a travel itinerary."""
    with judgment.trace(
        "generate_itinerary_demo",
        project_name="travel_agent_demo",
        overwrite=True
    ) as trace:    
        research_data = await research_destination(destination, start_date, end_date)
        res = await create_travel_plan(destination, start_date, end_date, research_data)
        trace.save()
        trace.print()
        return res


if __name__ == "__main__":
    load_dotenv()
    destination = input("Enter your travel destination: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    itinerary = asyncio.run(generate_itinerary(destination, start_date, end_date))
    print("\nGenerated Itinerary:\n", itinerary)