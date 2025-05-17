import os
import asyncio
from typing import TypedDict, Sequence, Dict, Any, Optional, List, Union
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler
from judgeval.scorers import AnswerRelevancyScorer

# Load environment variables
load_dotenv()

# Initialize clients
# openai_client = OpenAI()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize LangChain Chat Model
chat_model = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize Judgment tracer
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="music-recommendation-bot",
    enable_monitoring=True,  # Explicitly enable monitoring
    deep_tracing=False # Disable deep tracing when using LangGraph handler
)

# Define the state type
class State(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    preferences: Dict[str, str]
    search_results: Dict[str, Any]
    recommendations: str
    current_question_idx: int
    questions: Sequence[str]

# Node functions
def initialize_state() -> State:
    """Initialize the state with questions and predefined answers."""
    questions = [
        "What are some of your favorite artists or bands?",
        "What genres of music do you enjoy the most?",
        "Do you have any favorite songs currently?",
        "Are there any moods or themes you're looking for in new music?",
        "Do you prefer newer releases or classic songs?"
    ]
    
    # Predefined answers for testing
    answers = [
        "Taylor Swift, The Beatles, and Ed Sheeran",
        "Pop, Rock, and Folk",
        "Anti-Hero, Hey Jude, and Perfect",
        "Upbeat and energetic music for workouts",
        "I enjoy both new and classic songs"
    ]
    
    # Initialize messages with questions and answers alternating
    messages = []
    for question, answer in zip(questions, answers):
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    
    return {
        "messages": messages,
        "preferences": {},
        "search_results": {},
        "recommendations": "",
        "current_question_idx": 0,
        "questions": questions
    }

def ask_question(state: State) -> State:
    """Process the next question-answer pair."""
    if state["current_question_idx"] >= len(state["questions"]):
        return state
    
    # The question is already in messages, just return the state
    return state

def process_answer(state: State) -> State:
    """Process the predefined answer and store it in preferences."""
    messages = state["messages"]
    
    # Ensure we have both a question and an answer
    if len(messages) < 2 or state["current_question_idx"] >= len(state["questions"]):
        return state
    
    try:
        last_question = state["questions"][state["current_question_idx"]]
        # Get the answer from messages - it will be after the question
        answer_idx = (state["current_question_idx"] * 2) + 1  # Calculate the index of the answer
        last_answer = messages[answer_idx].content
        
        state["preferences"][last_question] = last_answer
        state["current_question_idx"] += 1
        
        # Print the Q&A for visibility
        print(f"\nQ: {last_question}")
        print(f"A: {last_answer}\n")
        
    except IndexError:
        return state
    
    return state

def search_music_info(state: State) -> State:
    """Search for music recommendations based on preferences."""
    preferences = state["preferences"]
    search_results = {}
    
    # Search for artist recommendations
    if preferences.get("What are some of your favorite artists or bands?"):
        artists_query = f"Music similar to {preferences['What are some of your favorite artists or bands?']}"
        search_results["artist_based"] = tavily_client.search(
            query=artists_query,
            search_depth="advanced",
            max_results=5
        )
    
    # Search for genre recommendations
    if preferences.get("What genres of music do you enjoy the most?"):
        genre_query = f"Best {preferences['What genres of music do you enjoy the most?']} songs"
        search_results["genre_based"] = tavily_client.search(
            query=genre_query,
            search_depth="advanced",
            max_results=5
        )
    
    # Search for mood-based recommendations
    mood_question = "Are there any moods or themes you're looking for in new music?"  # Fixed apostrophe
    if preferences.get(mood_question):
        mood_query = f"{preferences[mood_question]} music recommendations"
        search_results["mood_based"] = tavily_client.search(
            query=mood_query,
            search_depth="advanced",
            max_results=5
        )
    
    state["search_results"] = search_results
    return state

def generate_recommendations(state: State) -> State:
    """Generate personalized music recommendations using ChatOpenAI."""
    preferences = state["preferences"]
    search_results = state["search_results"]
    
    # Prepare context from search results
    context = ""
    for category, results in search_results.items():
        if results and results.get("results"):
            context += f"\n{category.replace('_', ' ').title()} Search Results:\n"
            for result in results.get("results", []):
                content_preview = result.get('content', '')[:200]
                context += f"- {result.get('title')}: {content_preview}...\n"
        else:
            context += f"\nNo search results found for {category.replace('_', ' ').title()}\n"
    
    # Create messages for the Chat Model
    system_message = SystemMessage(content="""
    You are a music recommendation expert. Your primary rule is to ONLY suggest songs by artists that the user explicitly listed as their favorite artists in response to 'What are some of your favorite artists or bands?'. Never recommend songs by other artists, even if mentioned elsewhere in their preferences or search results.
    """)

    user_prompt = f"""
    Based ONLY on the user's stated favorite artists/bands and considering their other preferences, suggest 5-7 songs. For each song, include:
    1. Artist name (must be one of their explicitly stated favorite artists)
    2. Song title
    3. A brief explanation of why they might like it, considering their genre and mood preferences.

    User Preferences:
    {preferences}

    Potentially Relevant Search Results (for context, NOT necessarily for artists):
    {context}

    Remember: STRICTLY recommend songs ONLY by artists listed in response to 'What are some of your favorite artists or bands?'.
    """
    user_message = HumanMessage(content=user_prompt)

    # Use the LangChain ChatOpenAI instance with ainvoke
    response = chat_model.invoke([system_message, user_message])
    recommendations = response.content
    state["recommendations"] = recommendations

    # --- Prepare and Add Evaluation to State using the new helper ---
    # add_evaluation_to_state(
    #     state=state, # Pass the current state dictionary
    #     scorers=[AnswerRelevancyScorer(threshold=0.5)],
    #     input=user_prompt,
    #     actual_output=recommendations,
    #     model="gpt-4"
    # )

    judgment.async_evaluate(
        input=user_prompt,
        actual_output=recommendations,
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        model="gpt-4o"
    )
    # --- End Evaluation Setup ---

    return state

def should_continue_questions(state: State) -> bool:
    """Determine if we should continue asking questions."""
    return state["current_question_idx"] < len(state["questions"])

def router(state: State) -> str:
    """Route to the next node based on state."""
    if should_continue_questions(state):
        return "ask_question"
    return "search_music"

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("ask_question", ask_question)
workflow.add_node("process_answer", process_answer)

workflow.add_node("search_music", search_music_info)
workflow.add_node("generate_recommendations", generate_recommendations)

# Add edges
workflow.add_edge("ask_question", "process_answer")
workflow.add_conditional_edges(
    "process_answer",
    router,
    {
        "ask_question": "ask_question",
        "search_music": "search_music"
    }
)
workflow.add_edge("search_music", "generate_recommendations")
workflow.add_edge("generate_recommendations", END)

# Set entry point
workflow.set_entry_point("ask_question")

# Compile the graph
graph = workflow.compile()

# Main function
def music_recommendation_bot(handler: JudgevalCallbackHandler):
    """Main function to run the music recommendation bot."""
    print("🎵 Welcome to the Music Recommendation Bot! 🎵")
    print("I'll ask you a few questions to understand your music taste, then suggest some songs you might enjoy.")
    print("\nRunning with predefined answers for testing...\n")
    
    # Initialize state with predefined answers
    initial_state = initialize_state()
    
    try:
        # Initialize the Async handler

        
        # Run the entire workflow with graph.ainvoke (asynchronous)
        # Pass handler directly in config
        # The handler instance needs to be accessible inside the node later
        config_with_callbacks = {"callbacks": [handler]}
        final_state = graph.invoke(initial_state, config=config_with_callbacks) # Use ainvoke (async) and the async handler
        
        print("\n🎧 Your Personalized Music Recommendations 🎧")
        print(final_state.get("recommendations", "No recommendations generated."))
        return final_state.get("recommendations", "No recommendations generated.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    handler = JudgevalCallbackHandler(judgment) 
    music_recommendation_bot(handler)
