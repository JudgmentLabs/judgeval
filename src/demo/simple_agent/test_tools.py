from judgeval import JudgmentClient
from judgeval.data import Example
from typing import List, Dict, Any, Optional, Callable
import random
from pydantic import BaseModel
from judgeval.common.tracer import Tracer, wrap


judgment = Tracer(project_name="my_project")

class Agent(BaseModel):
    name: str = "Generic Agent"
    specialty: str = "General tasks"
    
    @judgment.observe(span_type="tool")
    def use_calculator_tool(self) -> str:
        """Simulate using a calculator tool"""
        print(f"\n[{self.name}] Performing calculation...")
        result = random.randint(1, 100)
        return f"I, {self.name}, used the calculator tool to solve your problem.\nThe result of the calculation is {result}."

    @judgment.observe(span_type="tool")
    def use_weather_tool(self) -> str:
        """Simulate using a weather tool"""
        print(f"\n[{self.name}] Checking weather information...")
        weather_conditions = ["sunny", "rainy", "cloudy", "partly cloudy", "stormy"]
        temperature = random.randint(0, 35)
        return f"I, {self.name}, used the weather tool to check the forecast.\nThe weather is {random.choice(weather_conditions)} with a temperature of {temperature}°C."

    @judgment.observe(span_type="tool")
    def use_search_tool(self, prompt: str) -> str:
        """Simulate using a search tool"""
        print(f"\n[{self.name}] Searching for information...")
        # Generate a random search result string
        search_terms = ["facts", "information", "details", "data", "knowledge"]
        search_actions = ["found", "discovered", "located", "identified", "uncovered"]
        
        random_string = f"I {random.choice(search_actions)} some interesting {random.choice(search_terms)} about this topic."
        print(f"Random search result: {random_string}")
        return f"I, {self.name}, used the search tool to find information about '{prompt}'.\nHere are some relevant details I found from my search: {random_string}"

    @judgment.observe(span_type="tool")
    def use_translator_tool(self) -> str:
        """Simulate using a translator tool"""
        print(f"\n[{self.name}] Translating text...")
        languages = ["French", "Spanish", "German", "Italian", "Portuguese"]
        random_string = f"I translated the text to {random.choice(languages)}." 
        print(f"Random translated string: {random_string}")
        return f"I, {self.name}, used the translator tool to translate your text.\nThe translated content is: 'Example translated content' {random_string}"

    def run_agent(self, prompt: str) -> Dict[str, Any]:
        """
        A simple agent that asks the user which tool to use and then simulates using it.
        
        Args:
            prompt (str): The user's input prompt
            
        Returns:
            Dict: Contains the response and tools used
        """
        print(f"\n{'='*50}")
        print(f"AGENT: {self.name} ({self.specialty})")
        print(f"USER PROMPT: {prompt}")
        print(f"{'='*50}")
        
        # Define available tools based on specialty
        tools = {
            "1": {"name": "calculator", "description": "Performs mathematical calculations"},
            "2": {"name": "weather", "description": "Checks the weather for a location"},
            "3": {"name": "search", "description": "Searches for information on the web"},
            "4": {"name": "translator", "description": "Translates text between languages"}
        }
        
        # Display tool options to the user
        print(f"\n{self.name}'s Available tools:")
        for key, tool in tools.items():
            print(f"{key}: {tool['name']} - {tool['description']}")
        print("0: No tool needed")

        tools_used = []
        
        # Ask the user which tool to use
        choice = random.choice(list(tools.keys()))
        
        response = ""
        
        if choice == "0":
            response = f"I, {self.name}, will answer your question directly without using any tools.\n\nRegarding '{prompt}', I can tell you that this is a general question I can answer directly."
        
        elif choice in tools:
            selected_tool = tools[choice]["name"]
            tools_used.append(selected_tool)
            
            if selected_tool == "calculator":
                response = self.use_calculator_tool()
                
            elif selected_tool == "weather":
                response = self.use_weather_tool()
                
            elif selected_tool == "search":
                response = self.use_search_tool(prompt)
                
            elif selected_tool == "translator":
                response = self.use_translator_tool()
        
        else:
            response = f"Invalid choice. I, {self.name}, will answer without using any tools."
        
        print(f"\n{'-'*50}")
        print(f"{self.name}'s RESPONSE:")
        print(response)
        print(f"{'-'*50}\n")
        
        return {
            "agent": self.name,
            "specialty": self.specialty,
            "response": response,
            "tools_called": tools_used
        }


class MultiAgentSystem:
    def __init__(self):
        self.agents = []
        
    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        
    def list_agents(self):
        print("\nAvailable Agents:")
        for i, agent in enumerate(self.agents):
            print(f"{i+1}. {agent.name} - Specialty: {agent.specialty}")
            
    def run_system(self, prompt: str):
        """Run the multi-agent system"""
        print("\n" + "="*70)
        print(f"MULTI-AGENT SYSTEM - PROCESSING: {prompt}")
        print("="*70)
        
        # 1. List available agents
        self.list_agents()
        
        # 2. Ask which agent to use
        choice = input("\nSelect an agent (number) or '0' for collaborative mode: ")
        
        responses = []
        
        if choice == "0":
            # Collaborative mode - all agents work on the task
            print("\n🔄 Running in collaborative mode - all agents will contribute")
            for agent in self.agents:
                print(f"\n👉 Now consulting {agent.name}...")
                agent_response = agent.run_agent(prompt)
                responses.append(agent_response)
                
            # Combine responses
            combined_response = "\n\n".join([
                f"🤖 {resp['agent']} ({resp['specialty']}):\n{resp['response']}" 
                for resp in responses
            ])
            
            print("\n" + "="*70)
            print("FINAL COLLABORATIVE RESPONSE:")
            print(combined_response)
            print("="*70)
            
        elif choice.isdigit() and 1 <= int(choice) <= len(self.agents):
            # Single agent mode
            agent_idx = int(choice) - 1
            selected_agent = self.agents[agent_idx]
            print(f"\n👉 Selected agent: {selected_agent.name}")
            agent_response = selected_agent.run_agent(prompt)
            responses.append(agent_response)
            
        else:
            print("Invalid selection. Please run again.")
            
        return responses


# Create a multi-agent system with specialized agents
system = MultiAgentSystem()

# Create specialized agents
calculator_agent = Agent(name="CalcBot", specialty="Mathematical calculations")
weather_agent = Agent(name="WeatherWiz", specialty="Weather forecasting")
search_agent = Agent(name="InfoSeeker", specialty="Information retrieval")
translator_agent = Agent(name="LinguaGenius", specialty="Language translation")

# Add agents to the system
system.add_agent(calculator_agent)
system.add_agent(weather_agent)
system.add_agent(search_agent)
system.add_agent(translator_agent)

example = Example(
    input="What is the capital of France?",
    expected_tools = [
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
                "field1": "value100",
                "field2": 100
            }
        }
    ]
)

client = JudgmentClient()

from judgeval.scorers import AnswerCorrectnessScorer
scorer = AnswerCorrectnessScorer(threshold=0.9)

# You can still use your original test with a single agent
client.assert_test(
    function=calculator_agent.run_agent,
    eval_run_name="tool_correctness_1",
    examples=[example],
    scorers=[scorer],
    model="gpt-4o-mini",
    override = True
)