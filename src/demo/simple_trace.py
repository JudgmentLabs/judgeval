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


class TravelAgent:
    @judgment.observe(span_type="tool")
    async def get_weather(self, city: str):
        """Simulated weather tool call."""
        weather_data = f"It is sunny and 72°F in {city}."
        return weather_data

    @judgment.observe(span_type="tool")
    async def get_attractions(self, city: str):
        """Simulated attractions tool call."""
        attractions = [
            "Eiffel Tower",
            "Louvre Museum",
            "Notre-Dame Cathedral",
            "Arc de Triomphe"
        ]
        return attractions

    @judgment.observe(span_type="Research")
    async def gather_information(self, city: str):
        """Gather all necessary travel information."""
        weather = await self.get_weather(city)
        attractions = await self.get_attractions(city)

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
    async def create_travel_plan(self, research_data):
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
    async def generate_simple_itinerary(self, query: str = "I want to plan a trip to Paris."):
        """Main function to generate a travel itinerary."""
        research_data = await self.gather_information(city="Paris")
        itinerary = await self.create_travel_plan(research_data)
        return itinerary
    

if __name__ == "__main__":
    import asyncio
    agent1 = TravelAgent()
    # TODO: Make this clean
    itinerary, trace = asyncio.run(agent1.generate_simple_itinerary("I want to plan a trip to Paris."))
                    
    # Expected tool calls
    correct_expected_tools = [
        {"agent": "TravelAgent", "tool": "get_weather", "params": {"city": "Paris"}},
        {"agent": "TravelAgent", "tool": "get_attractions", "params": {"city": "Paris"}},
    ]
    
    # check tools get called in right order - have an i = 0...
    # also check that tools don't get called in wrong order - check if a tool either: 1. Exists in the rest of tool list or 2. Is any tool that's not current index
    def check_tool_order(trace, expected_tools):
        current_tool_index = 0
        for span in trace['entries']:
            if span['span_type'] == "tool":
                # Check for which object in the "inputs"
                # Ideally this is a separate field in the span
                var = span['inputs']['args']
                import re

                match = re.match(r"\((.*),\s*'(.*)'\)", var)
                if match:
                    obj_repr, city = match.groups()
                    tup = (obj_repr.strip(), city)
                
                obj_name, *args = tup
                
                if span['function'] != expected_tools[current_tool_index]['tool']:
                    raise ValueError(f"Tool {span['function']} called out of order")
                
                if expected_tools[current_tool_index]['agent'] not in obj_name:
                    raise ValueError(f"Agent {obj_name} called out of order")
                
                if args != list(expected_tools[current_tool_index]['params'].values()):
                    raise ValueError(f"Args {args} called out of order")
                
                current_tool_index += 1
        
        if current_tool_index != len(expected_tools):
            raise ValueError(f"Not all tools were called, expected {len(expected_tools)} but got {current_tool_index}")
        
        return True
    
    check_tool_order(trace, correct_expected_tools)
        
    incorrect_expected_tools = [
        {"agent": "TravelAgent", "tool": "get_attractions", "params": {"city": "Paris"}},
        {"agent": "TravelAgent", "tool": "get_weather", "params": {"city": "Paris"}},
    ]
    
    check_tool_order(trace, incorrect_expected_tools)
    
