"""Google ADK agent with Judgment tracing.

When running with `adk web`, ADK sets up its own global TracerProvider
before importing this module. OpenTelemetry enforces first-writer-wins
semantics, so calling install_as_global_tracer_provider() would silently
fail. Instead, we attach Judgment's span processor directly to ADK's
provider so that all spans — both ADK's internal spans and our own
@Tracer.observe spans — flow to the Judgment dashboard.

Usage:
    adk web my_agent
"""

from opentelemetry import trace as trace_api

from google.adk import Agent
from google.adk.tools import ToolContext

from judgeval import Tracer

# 1. Initialize Judgment tracing.
#    This creates an internal TracerProvider with the JudgmentSpanProcessor
#    and registers it with JudgmentTracerProvider. The @Tracer.observe
#    decorator works immediately after this call.
tracer = Tracer.init(project_name="my-adk-agent")

# 2. Attach Judgment's span processor to ADK's global provider.
#    ADK's _setup_telemetry() has already called set_tracer_provider()
#    by the time this module is imported. Adding our processor here
#    means ADK's own spans (agent execution, model calls, tool calls)
#    are also exported to the Judgment dashboard.
global_provider = trace_api.get_tracer_provider()
global_provider.add_span_processor(tracer.get_span_processor())


@Tracer.observe(span_type="tool")
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    data = {
        "san francisco": {"temp_f": 62, "condition": "foggy"},
        "new york": {"temp_f": 45, "condition": "cloudy"},
        "tokyo": {"temp_f": 73, "condition": "sunny"},
    }
    return data.get(city.lower(), {"temp_f": 0, "condition": "unknown"})


@Tracer.observe(span_type="tool")
def search_restaurants(city: str, cuisine: str, tool_context: ToolContext) -> list[str]:
    """Search for restaurants in a city by cuisine type."""
    Tracer.set_attribute("cuisine", cuisine)
    restaurants = {
        "tokyo": {
            "sushi": ["Sukiyabashi Jiro", "Sushi Saito"],
            "ramen": ["Fuunji", "Ichiran Shibuya"],
        },
        "new york": {
            "pizza": ["Di Fara Pizza", "Lucali"],
            "sushi": ["Sushi Nakazawa", "Masa"],
        },
    }
    results = restaurants.get(city.lower(), {}).get(cuisine.lower(), [])
    tool_context.state["last_search_city"] = city
    return results


root_agent = Agent(
    model="gemini-2.5-flash",
    name="travel_assistant",
    description="A travel assistant that provides weather and restaurant recommendations.",
    instruction=(
        "You are a helpful travel assistant. When asked about a destination, "
        "check the weather and suggest restaurants. Be concise."
    ),
    tools=[get_weather, search_restaurants],
)
