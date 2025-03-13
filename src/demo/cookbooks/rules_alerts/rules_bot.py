import os
import asyncio
from typing import Dict, List
from openai import OpenAI
from uuid import uuid4
from dotenv import load_dotenv

from judgeval.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer, AnswerCorrectnessScorer
from judgeval.rules import Rule, Condition

# Initialize clients
load_dotenv()
rules = [
        Rule(
            name="All Conditions Check",
            description="Check if all conditions are met",
            conditions=[
                # Use scorer objects instead of strings
                Condition(metric=FaithfulnessScorer(threshold=0.7)),
                Condition(metric=AnswerRelevancyScorer(threshold=0.8)),
                Condition(metric=AnswerCorrectnessScorer(threshold=0.9))
            ],
            combine_type="all"  # Require all conditions to trigger
        ),
        Rule(
            name="Any Condition Check",
            description="Check if any condition is met",
            conditions=[
                Condition(metric=FaithfulnessScorer(threshold=0.7)),
                Condition(metric=AnswerRelevancyScorer(threshold=0.8)),
                Condition(metric=AnswerCorrectnessScorer(threshold=0.9))
            ],
            combine_type="any"  # Require any condition to trigger
        )
        # Removed rules that used SimpleKeywordScorer
    ]

judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="restaurant_bot", rules=rules)
client = wrap(OpenAI())

@judgment.observe(span_type="Research")
async def search_restaurants(cuisine: str, location: str = "nearby") -> List[Dict]:
    """Search for restaurants matching the cuisine type."""
    # Simulate API call to restaurant database
    prompt = f"Find 3 popular {cuisine} restaurants {location}. Return ONLY a JSON array of objects with 'name', 'rating', and 'price_range' fields. No other text."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """You are a restaurant search expert. 
             Return ONLY valid JSON arrays containing restaurant objects.
             Example format: [{"name": "Restaurant Name", "rating": 4.5, "price_range": "$$"}]
             Do not include any other text or explanations."""},
            {"role": "user", "content": prompt}
        ]
    )
    
    try:
        import json
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {response.choices[0].message.content}")
        return [{"name": "Error fetching restaurants", "rating": 0, "price_range": "N/A"}]

@judgment.observe(span_type="Research") 
async def get_menu_highlights(restaurant_name: str) -> List[str]:
    """Get popular menu items for a restaurant."""
    prompt = f"What are 3 must-try dishes at {restaurant_name}?"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a food critic. List only the dish names."},
            {"role": "user", "content": prompt}
        ]
    )

    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=prompt,
        actual_output=response.choices[0].message.content,
        model="gpt-4",
    )

    return response.choices[0].message.content.split("\n")

@judgment.observe(span_type="function")
async def generate_recommendation(cuisine: str, restaurants: List[Dict], menu_items: Dict[str, List[str]]) -> str:
    """Generate a natural language recommendation."""
    context = f"""
    Cuisine: {cuisine}
    Restaurants: {restaurants}
    Popular Items: {menu_items}
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful food recommendation bot. Provide a natural recommendation based on the data."},
            {"role": "user", "content": context}
        ]
    )
    return response.choices[0].message.content

@judgment.observe(span_type="Research")
async def get_food_recommendations(cuisine: str) -> str:
    """Main function to get restaurant recommendations."""
    # Search for restaurants
    restaurants = await search_restaurants(cuisine)
    
    # Get menu highlights for each restaurant
    menu_items = {}
    for restaurant in restaurants:
        menu_items[restaurant['name']] = await get_menu_highlights(restaurant['name'])
        
    # Generate final recommendation
    recommendation = await generate_recommendation(cuisine, restaurants, menu_items)
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5), FaithfulnessScorer(threshold=1.0)],
        input=f"Create a recommendation for a restaurant and dishes based on the desired cuisine: {cuisine}",
        actual_output=recommendation,
        retrieval_context=[str(restaurants), str(menu_items)],
        model="gpt-4",
    )
    return recommendation

if __name__ == "__main__":
    cuisine = input("What kind of food would you like to eat? ")
    recommendation = asyncio.run(get_food_recommendations(cuisine))
    print("\nHere are my recommendations:\n")
    print(recommendation)
