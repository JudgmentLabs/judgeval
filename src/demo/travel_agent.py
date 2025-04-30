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

destinations_data = [
    {
        "destination": "Paris, France",
        "information": """
Paris is the capital city of France and a global center for art, fashion, and culture.
Key Information:
- Best visited during spring (March-May) or fall (September-November)
- Famous landmarks: Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, Arc de Triomphe
- Known for: French cuisine, café culture, fashion, art galleries
- Local transportation: Metro system is extensive and efficient
- Popular neighborhoods: Le Marais, Montmartre, Latin Quarter
- Cultural tips: Basic French phrases are appreciated; many restaurants close between lunch and dinner
- Must-try experiences: Seine River cruise, visiting local bakeries, Luxembourg Gardens
"""
    },
    {
        "destination": "Tokyo, Japan",
        "information": """
Tokyo is Japan's bustling capital, blending ultramodern and traditional elements.
Key Information:
- Best visited during spring (cherry blossoms) or fall (autumn colors)
- Famous areas: Shibuya, Shinjuku, Harajuku, Akihabara
- Known for: Technology, anime culture, sushi, efficient public transport
- Local transportation: Extensive train and subway network
- Cultural tips: Bow when greeting, remove shoes indoors, no tipping
- Must-try experiences: Robot Restaurant, teamLab Borderless, Tsukiji Outer Market
- Popular day trips: Mount Fuji, Kamakura, Nikko
"""
    },
    {
        "destination": "New York City, USA",
        "information": """
New York City is a global metropolis known for its diversity, culture, and iconic skyline.
Key Information:
- Best visited during spring (April-June) or fall (September-November)
- Famous landmarks: Statue of Liberty, Times Square, Central Park, Empire State Building
- Known for: Broadway shows, diverse cuisine, shopping, museums
- Local transportation: Extensive subway system, yellow cabs, ride-sharing
- Popular areas: Manhattan, Brooklyn, Queens
- Cultural tips: Fast-paced environment, tipping expected (15-20%)
- Must-try experiences: Broadway show, High Line walk, food tours
"""
    },
    {
        "destination": "Barcelona, Spain",
        "information": """
Barcelona is a vibrant city known for its art, architecture, and Mediterranean culture.
Key Information:
- Best visited during spring and fall for mild weather
- Famous landmarks: Sagrada Familia, Park Güell, Casa Batlló
- Known for: Gaudi architecture, tapas, beach culture, FC Barcelona
- Local transportation: Metro, buses, and walkable city center
- Popular areas: Gothic Quarter, Eixample, La Barceloneta
- Cultural tips: Late dinner times (after 8 PM), siesta tradition
- Must-try experiences: La Rambla walk, tapas crawl, local markets
"""
    },
    {
        "destination": "Bangkok, Thailand",
        "information": """
Bangkok is Thailand's capital city, famous for its temples, street food, and vibrant culture.
Key Information:
- Best visited during November to February (cool and dry season)
- Famous sites: Grand Palace, Wat Phra Kaew, Wat Arun
- Known for: Street food, temples, markets, nightlife
- Local transportation: BTS Skytrain, MRT, tuk-tuks, river boats
- Popular areas: Sukhumvit, Old City, Chinatown
- Cultural tips: Dress modestly at temples, respect royal family
- Must-try experiences: Street food tours, river cruises, floating markets
"""
    }
]

client = wrap(openai.Client(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="travel_agent_demo")

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

@judgment.observe(span_type="search_tool")
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
    return attractions_search

@judgment.observe(span_type="tool")
async def get_hotels(destination):
    """Search for hotels in the destination."""
    prompt = f"Best hotels in {destination}"
    hotels_search = search_tavily(prompt)
    return hotels_search

@judgment.observe(span_type="tool")
async def get_flights(destination):
    """Search for flights to the destination."""
    prompt = f"Flights to {destination} from major cities"
    flights_search = search_tavily(prompt)
    example = Example(
        input=prompt,
        actual_output=str(flights_search["results"])
    )
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        example=example,
        model="gpt-4o"
    )
    return flights_search

@judgment.observe(span_type="tool")
async def get_weather(destination, start_date, end_date):
    """Search for weather information."""
    prompt = f"Weather forecast for {destination} from {start_date} to {end_date}"
    weather_search = search_tavily(prompt)
    example = Example(
        input=prompt,
        actual_output=str(weather_search["results"])
    )
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        example=example,
        model="gpt-4o"
    )
    return weather_search

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

@judgment.observe(span_type="retriever")
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

@judgment.observe(span_type="Research")
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
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert travel planner. Combine both historical and current information to create the best possible itinerary."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content

    example = Example(
        input=prompt,
        actual_output=str(response),
        retrieval_context=[str(vector_db_context), str(research_data)]
    )
    judgment.async_evaluate(
        scorers=[FaithfulnessScorer(threshold=0.5)],
        example=example,
        model="gpt-4o"
    )
    
    return response

@judgment.observe(span_type="function")
async def generate_itinerary(destination, start_date, end_date):
    """Main function to generate a travel itinerary."""
    research_data = await research_destination(destination, start_date, end_date)
    res = await create_travel_plan(destination, start_date, end_date, research_data)
    return res


if __name__ == "__main__":
    load_dotenv()
    destination = input("Enter your travel destination: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    itinerary = asyncio.run(generate_itinerary(destination, start_date, end_date))
    print("\nGenerated Itinerary:\n", itinerary)