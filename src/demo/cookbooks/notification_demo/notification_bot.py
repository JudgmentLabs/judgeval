import os
import asyncio
import random
from typing import Dict, List, Optional, Union
from openai import OpenAI
from uuid import uuid4
from dotenv import load_dotenv

from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer, AnswerCorrectnessScorer
from judgeval.rules import Rule, Condition, NotificationConfig

# Initialize environment variables and clients
load_dotenv()

# Initialize the rules with notification config
# This demo only supports slack and email notification methods
notification_config = NotificationConfig(
    enabled=True,
    communication_methods=["slack", "email"],  # Only using slack and email
    email_addresses=["minh@judgmentlabs.ai"],  # Replace with your email
    send_at=None  # Send immediately
)

# Create rules with the notification config
rules = [
    Rule(
        name="All Conditions Check",
        description="Check if all conditions are met",
        conditions=[
            Condition(metric=FaithfulnessScorer(threshold=0.7)),
            Condition(metric=AnswerRelevancyScorer(threshold=0.8)),
            Condition(metric=AnswerCorrectnessScorer(threshold=0.9))
        ],
        combine_type="all",  # Require all conditions to trigger
        notification=notification_config
    ),
    Rule(
        name="Any Condition Check",
        description="Check if any condition is met",
        conditions=[
            Condition(metric=FaithfulnessScorer(threshold=0.7)),
            Condition(metric=AnswerRelevancyScorer(threshold=0.8)),
            Condition(metric=AnswerCorrectnessScorer(threshold=0.9))
        ],
        combine_type="any",  # Require any condition to trigger
        notification=notification_config
    )
]

# Initialize tracer with rules for notifications
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"), 
    project_name="notification_demo", 
    rules=rules
)

# Wrap OpenAI client for tracing
client = wrap(OpenAI())

judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="restaurant_bot", rules=rules)
client = wrap(OpenAI())

# Test mode types
class TestMode:
    NORMAL = "normal"
    UNFAITHFUL = "unfaithful"
    IRRELEVANT = "irrelevant"
    BOTH = "both"

@judgment.observe(span_type="Research")
async def search_restaurants(cuisine: str, location: str = "nearby", test_mode: str = TestMode.NORMAL) -> List[Dict]:
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
        restaurants = json.loads(response.choices[0].message.content)
        
        # Intentionally create incorrect data if in test mode
        if test_mode == TestMode.UNFAITHFUL or test_mode == TestMode.BOTH:
            # Change restaurant names to completely different cuisine types
            cuisines = ["Italian", "Mexican", "Indian", "Chinese", "Japanese", "Thai", "Greek", "French"]
            for restaurant in restaurants:
                if random.random() < 0.7:  # 70% chance to modify each restaurant
                    wrong_cuisine = random.choice([c for c in cuisines if c.lower() != cuisine.lower()])
                    restaurant["name"] = f"{wrong_cuisine} Palace"  # Clearly wrong name
        
        return restaurants
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {response.choices[0].message.content}")
        return [{"name": "Error fetching restaurants", "rating": 0, "price_range": "N/A"}]

@judgment.observe(span_type="Research") 
async def get_menu_highlights(restaurant_name: str, test_mode: str = TestMode.NORMAL) -> List[str]:
    """Get popular menu items for a restaurant."""
    prompt = f"What are 3 must-try dishes at {restaurant_name}?"
    
    # If in irrelevant test mode, modify the prompt to get irrelevant results
    if test_mode == TestMode.IRRELEVANT or test_mode == TestMode.BOTH:
        # 50% chance to ask a completely unrelated question instead
        if random.random() < 0.5:
            unrelated_prompts = [
                "List 3 popular tourist attractions in Paris.",
                "What are 3 benefits of regular exercise?",
                "Name 3 bestselling books from 2023."
            ]
            prompt = random.choice(unrelated_prompts)
    
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
async def generate_recommendation(cuisine: str, restaurants: List[Dict], menu_items: Dict[str, List[str]], test_mode: str = TestMode.NORMAL) -> str:
    """Generate a natural language recommendation."""
    context = f"""
    Cuisine: {cuisine}
    Restaurants: {restaurants}
    Popular Items: {menu_items}
    """
    
    # Modify the system prompt based on test mode
    system_prompt = "You are a helpful food recommendation bot. Provide a natural recommendation based on the data."
    
    if test_mode == TestMode.UNFAITHFUL:
        system_prompt = "You are a food recommendation bot. Provide recommendations but INTENTIONALLY include false information that contradicts the data provided. Make up fake restaurant names and dishes that weren't mentioned."
    elif test_mode == TestMode.IRRELEVANT:
        system_prompt = "You are a bot that gets easily distracted. Start with a brief food recommendation then go completely off-topic and talk about something unrelated like space travel, politics, or sports."
    elif test_mode == TestMode.BOTH:
        system_prompt = "You are a problematic recommendation bot. Provide recommendations with false information that contradicts the data AND go completely off-topic midway through your response to talk about unrelated subjects."
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
    )
    return response.choices[0].message.content

@judgment.observe(span_type="Research")
async def get_food_recommendations(cuisine: str, test_mode: str = TestMode.NORMAL) -> str:
    """Main function to get restaurant recommendations."""
    print(f"Running in test mode: {test_mode}")
    
    # Search for restaurants
    restaurants = await search_restaurants(cuisine, test_mode=test_mode)
    
    # Get menu highlights for each restaurant
    menu_items = {}
    for restaurant in restaurants:
        menu_items[restaurant['name']] = await get_menu_highlights(restaurant['name'], test_mode=test_mode)
        
    # Generate final recommendation
    recommendation = await generate_recommendation(cuisine, restaurants, menu_items, test_mode=test_mode)
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
    
    print("\nSelect test mode:")
    print("1. Normal (accurate recommendations)")
    print("2. Unfaithful (includes false information)")
    print("3. Irrelevant (goes off-topic)")
    print("4. Both (unfaithful and irrelevant)")
    
    mode_choice = input("Enter your choice (1-4): ")
    
    test_mode = TestMode.NORMAL
    if mode_choice == "2":
        test_mode = TestMode.UNFAITHFUL
    elif mode_choice == "3":
        test_mode = TestMode.IRRELEVANT
    elif mode_choice == "4":
        test_mode = TestMode.BOTH
    
    recommendation = asyncio.run(get_food_recommendations(cuisine, test_mode=test_mode))
    print("\nHere are my recommendations:\n")
    print(recommendation)
    
    print("\nThis recommendation was generated in test mode:", test_mode)
