from uuid import uuid4
import openai
import os
from dotenv import load_dotenv
import time
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer

# Initialize clients
load_dotenv()
client = wrap(openai.Client(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"), 
    project_name="simple_trace_demo", 
)

async def get_weather(city: str):
    """Simulated weather tool call."""
    judgment.log(f"Fetching weather data for {city}")
    weather_data = f"It is sunny and 72°F in {city}."
    judgment.log(f"Weather data retrieved: {weather_data}")
    return weather_data

async def get_attractions(city: str):
    """Simulated attractions tool call."""
    judgment.log(f"Fetching attractions for {city}")
    attractions = [
        "Eiffel Tower",
        "Louvre Museum",
        "Notre-Dame Cathedral",
        "Arc de Triomphe"
    ]
    judgment.log(f"Found {len(attractions)} attractions")
    return attractions

async def gather_information(city: str):
    """Gather all necessary travel information."""
    judgment.log(f"Starting information gathering for {city}")
    
    weather = await get_weather(city)
    judgment.log("Weather information retrieved")
    
    attractions = await get_attractions(city)
    judgment.log("Attractions information retrieved")

    # judgment.async_evaluate(
    #     scorers=[AnswerRelevancyScorer(threshold=0.5)],
    #     input="What is the weather in Paris?",
    #     actual_output=weather,
    #     model="gpt-4",
    # )
    
    judgment.log("Information gathering complete")
    return {
        "weather": weather,
        "attractions": attractions
    }

async def create_travel_plan(research_data):
    """Generate a travel itinerary using the researched data."""
    judgment.log("Starting travel plan creation")
    
    prompt = f"""
    Create a simple travel itinerary for Paris using this information:
    
    Weather: {research_data['weather']}
    Attractions: {research_data['attractions']}
    """
    
    judgment.log("Sending prompt to GPT-4")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a travel planner. Create a simple itinerary."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content

    judgment.log("Received response from GPT-4")
    
    # judgment.async_evaluate(
    #     scorers=[FaithfulnessScorer(threshold=0.5)],
    #     input=prompt,
    #     actual_output=response,
    #     retrieval_context=[str(research_data)],
    #     model="gpt-4",
    # )
    
    return response

@judgment.observe(span_type="function")
async def generate_simple_itinerary(query: str = "I want to plan a trip to Paris."):
    """Main function to generate a travel itinerary."""
    judgment.log(f"Starting itinerary generation for query: {query}")
    
    research_data = await gather_information(city="Paris")
    judgment.log("Research data gathered successfully")
    
    itinerary = await create_travel_plan(research_data)
    judgment.log("Travel plan created successfully")
    
    return itinerary

if __name__ == "__main__":
    import asyncio
    judgment.log("Starting main execution")
    itinerary = asyncio.run(generate_simple_itinerary("I want to plan a trip to Paris."))
    judgment.log("Execution completed")