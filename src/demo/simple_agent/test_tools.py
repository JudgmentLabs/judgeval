from judgment import JudgmentClient
from judgeval.data import Example
from typing import List, Dict, Any, Optional
import random

judgment = JudgmentClient()

example = Example(
    input={"question": "What is the capital of France?"},
    expected_tools_called = [
        {
            "tool_name": "tool_1",
            "tool_arguments": {
                "field1": "value1",
                "field2": 100
            }
        },
        {
            "tool_name": "tool_2",
            "tool_arguments": {
                "field1": "value1",
                "field2": 100
            }
        }
    ]
)

def run_agent(prompt: str) -> Dict[str, Any]:
    """
    A simple agent that asks the user which tool to use and then simulates using it.
    
    Args:
        prompt (str): The user's input prompt
        
    Returns:
        Dict: Contains the response and tools used
    """
    print("\n" + "="*50)
    print(f"USER PROMPT: {prompt}")
    print("="*50)
    
    # Define available tools
    tools = {
        "1": {"name": "calculator", "description": "Performs mathematical calculations"},
        "2": {"name": "weather", "description": "Checks the weather for a location"},
        "3": {"name": "search", "description": "Searches for information on the web"},
        "4": {"name": "translator", "description": "Translates text between languages"}
    }
    
    # Display tool options to the user
    print("\nAvailable tools:")
    for key, tool in tools.items():
        print(f"{key}: {tool['name']} - {tool['description']}")
    print("0: No tool needed")
    
    # Ask the user which tool to use
    choice = input("\nWhich tool would you like to use? Enter the number: ")
    
    tools_used = []
    response = ""
    
    if choice == "0":
        response = f"I'll answer your question directly without using any tools.\n\nRegarding '{prompt}', I can tell you that this is a general question I can answer directly."
    
    elif choice in tools:
        selected_tool = tools[choice]["name"]
        tools_used.append(selected_tool)
        
        if selected_tool == "calculator":
            response = use_calculator_tool()
            
        elif selected_tool == "weather":
            response = use_weather_tool()
            
        elif selected_tool == "search":
            response = use_search_tool(prompt)
            
        elif selected_tool == "translator":
            response = use_translator_tool()
    
    else:
        response = "Invalid choice. I'll answer without using any tools."
    
    print("\n" + "-"*50)
    print("AGENT RESPONSE:")
    print(response)
    print("-"*50 + "\n")
    
    return {
        "response": response,
        "tools_called": tools_used
    } 

@judgment.
def use_calculator_tool() -> str:
    """Simulate using a calculator tool"""
    print("\nPerforming calculation...")
    result = random.randint(1, 100)
    return f"I used the calculator tool to solve your problem.\nThe result of the calculation is {result}."

def use_weather_tool() -> str:
    """Simulate using a weather tool"""
    print("\nChecking weather information...")
    weather_conditions = ["sunny", "rainy", "cloudy", "partly cloudy", "stormy"]
    temperature = random.randint(0, 35)
    return f"I used the weather tool to check the forecast.\nThe weather is {random.choice(weather_conditions)} with a temperature of {temperature}°C."

def use_search_tool(prompt: str) -> str:
    """Simulate using a search tool"""
    print("\nSearching for information...")
    # Generate a random search result string
    search_terms = ["facts", "information", "details", "data", "knowledge"]
    search_actions = ["found", "discovered", "located", "identified", "uncovered"]
    
    random_string = f"I {random.choice(search_actions)} some interesting {random.choice(search_terms)} about this topic."
    print(f"Random search result: {random_string}")
    return f"I used the search tool to find information about '{prompt}'.\nHere are some relevant details I found from my search {random_string}."

def use_translator_tool() -> str:
    """Simulate using a translator tool"""
    print("\nTranslating text...")
    languages = ["French", "Spanish", "German", "Italian", "Portuguese"]
    random_string = f"I translated the text to {random.choice(languages)}." 
    print(f"Random translated string: {random_string}")
    return f"I used the translator tool to translate your text.\nThe translated content is: 'Example translated content' {random_string}."

judgment.assert_test(
    function=run_agent("What is the capital of France?"),
    run_name="tool_correctness",
    examples=[example],
)