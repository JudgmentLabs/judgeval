import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer, GroundednessScorer

# Load environment variables
load_dotenv()

# Initialize OpenAI client and Judgment tracer
client = wrap(OpenAI())
judgment = Tracer(project_name="music_suggestion_bot")

@judgment.observe(span_type="tool")
async def search_tavily(query):
    """Search for information using Tavily."""
    from tavily import TavilyClient
    
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    search_result = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )
    
    return search_result

@judgment.observe(span_type="function")
async def ask_user_preferences():
    """Ask the user a series of questions about their music preferences."""
    questions = [
        "What are some of your favorite artists or bands?",
        "What genres of music do you enjoy the most?",
        "Do you have any favorite songs currently?",
        "Are there any moods or themes you're looking for in new music?",
        "Do you prefer newer releases or classic songs?"
    ]
    
    preferences = {}
    for question in questions:
        print(f"\n{question}")
        answer = input("> ")
        preferences[question] = answer
    
    return preferences

@judgment.observe(span_type="function")
async def search_music_recommendations(preferences):
    """Search for music recommendations based on user preferences."""
    # Construct search queries based on preferences
    search_results = {}
    
    # Search for artist recommendations
    if preferences.get("What are some of your favorite artists or bands?"):
        artists_query = f"Music similar to {preferences['What are some of your favorite artists or bands?']}"
        search_results["artist_based"] = await search_tavily(artists_query)
    
    # Search for genre recommendations
    if preferences.get("What genres of music do you enjoy the most?"):
        genre_query = f"Best {preferences['What genres of music do you enjoy the most?']} songs"
        search_results["genre_based"] = await search_tavily(genre_query)
    
    # Search for mood-based recommendations
    if preferences.get("Are there any moods or themes you're looking for in new music?"):
        mood_query = f"""{preferences["Are there any moods or themes you're looking for in new music?"]} music recommendations"""
        search_results["mood_based"] = await search_tavily(mood_query)
    
    return search_results

@judgment.observe(span_type="function")
async def generate_recommendations(preferences, search_results):
    """Generate personalized music recommendations using the search results."""
    # Prepare context from search results
    context = ""
    for category, results in search_results.items():
        context += f"\n{category.replace('_', ' ').title()} Search Results:\n"
        for result in results.get("results", []):
            context += f"- {result.get('title')}: {result.get('content')[:200]}...\n"
    
    # Create a prompt for the LLM
    prompt = f"""
    Suggest 5-7 songs they could enjoy. Be creative and suggest whatever feels right. You should only recommend songs that are from the user's favorite artists/bands.
    For each song, include the artist name, song title, and a brief explanation of why they might like it.
    
    User Preferences:
    {preferences}
    
    Search Results:
    {context}
    
    Provide recommendations in a clear, organized format. Focus on specific songs rather than just artists.
    """

#     better_prompt = f"""
# Suggest 5-7 songs they could enjoy. Be creative and suggest whatever feels right. You should only recommend songs that are from the user's favorite artists/bands.
# For each song, include the artist name, song title, and a brief explanation of why they might like it.

# User Preferences:
# {preferences}

# Search Results:
# {context}

# Provide recommendations in a clear, organized format. Focus on specific songs rather than just artists.

# **IMPORTANT**: You should only recommend songs that are from the user's favorite artists/bands. Even if the search results contain other songs by those artists, you should only recommend the ones that are listed in the user's favorite artists/bands. 
# Try to recommend songs that are similar to the user's favorite songs, but in the context of the user's favorite artists/bands.
# """
    
    # Generate recommendations using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a music recommendation expert with deep knowledge of various genres, artists, and songs. Your goal is to suggest songs that match the user's preferences; recommend songs from their favorite artists/bands."},
            {"role": "user", "content": prompt}
        ]
    )
    
    recommendations = response.choices[0].message.content
    
    # Evaluate the recommendations
    judgment.get_current_trace().async_evaluate(
        scorers=[
            AnswerRelevancyScorer(threshold=1.0),
            GroundednessScorer(threshold=1.0)
        ],
        input=prompt,
        actual_output=recommendations,
        retrieval_context=[str(search_results)],
        model="gpt-4o"
    )
    
    return recommendations

@judgment.observe(span_type="Main Function")
async def music_recommendation_bot():
    """Main function to run the music recommendation bot."""
    print("🎵 Welcome to the Music Recommendation Bot! 🎵")
    print("I'll ask you a few questions to understand your music taste, then suggest some songs you might enjoy.")
    
    # Get user preferences
    preferences = await ask_user_preferences()
    
    print("\nSearching for music recommendations based on your preferences...")
    search_results = await search_music_recommendations(preferences)
    
    print("\nGenerating personalized recommendations...")
    recommendations = await generate_recommendations(preferences, search_results)
    
    print("\n🎧 Your Personalized Music Recommendations 🎧")
    print(recommendations)
    
    return recommendations

if __name__ == "__main__":
    asyncio.run(music_recommendation_bot())
