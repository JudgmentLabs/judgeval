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

@judgment.observe(span_type="tool")
async def get_weather(city: str):
    """Simulated weather tool call."""
    weather_data = f"It is sunny and 72°F in {city}."
    return weather_data

@judgment.observe(span_type="tool")
async def get_attractions(city: str):
    """Simulated attractions tool call."""
    attractions = [
        "Eiffel Tower",
        "Louvre Museum",
        "Notre-Dame Cathedral",
        "Arc de Triomphe"
    ]
    return attractions

@judgment.observe(span_type="Research")
async def gather_information(city: str):
    """Gather all necessary travel information."""
    weather = await get_weather(city)
    attractions = await get_attractions(city)

    # judgment.async_evaluate(
    #     scorers=[AnswerRelevancyScorer(threshold=0.5)],
    #     input="What is the weather in Paris?",
    #     actual_output=weather,
    #     model="gpt-4",
    # )
    
    return {
        "weather": weather,
        "attractions": attractions
    }

@judgment.observe(span_type="function")
async def create_travel_plan(research_data):
    """Generate a travel itinerary using the researched data."""
    prompt = f"""
    Create a simple travel itinerary for Paris using this information:
    
    Weather: {research_data['weather']}
    Attractions: {research_data['attractions']}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a travel planner. Create a simple itinerary."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content

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
    research_data = await gather_information(city="Paris")
    itinerary = await create_travel_plan(research_data)
    return itinerary

if __name__ == "__main__":
    import asyncio
    itinerary = asyncio.run(generate_simple_itinerary("I want to plan a trip to Paris."))