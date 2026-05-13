"""Google ADK agent with Judgment tracing.

There are two runtime environments to handle:

- **adk web** (local dev): ADK sets up its own SDK TracerProvider before
  importing this module.  We attach Judgment's span processor to it so
  ADK's spans also flow to the Judgment dashboard.

- **Vertex Agent Engine**: The agent module is imported before the
  runtime installs its SDK TracerProvider.  get_tracer_provider()
  returns OTel's default ProxyTracerProvider (no add_span_processor).
  We install JudgmentTracerProvider as the global provider instead;
  when Vertex's runtime later calls add_span_processor on it, the
  processor is forwarded to all registered tracers.

Usage:
    adk web my_agent
"""

from google.adk import Agent
from google.adk.tools import ToolContext

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from judgeval import Tracer
from judgeval.trace import JudgmentTracerProvider

tracer = Tracer.init(project_name="my-adk-agent")

# If a real SDK TracerProvider is already installed (adk web), attach
# our processor to it.  Otherwise (Vertex Agent Engine), install
# JudgmentTracerProvider as the global provider so the runtime can
# add its own processors to it later.
global_provider = trace_api.get_tracer_provider()
if isinstance(global_provider, SDKTracerProvider):
    global_provider.add_span_processor(tracer.get_span_processor())
else:
    JudgmentTracerProvider.install_as_global_tracer_provider()


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
