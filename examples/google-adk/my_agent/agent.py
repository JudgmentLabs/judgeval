import re

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from judgeval import Tracer

ARITHMETIC_RE = re.compile(r"^[\d\s+\-*/().]+$")


@Tracer.observe(span_type="tool")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result."""
    if not ARITHMETIC_RE.match(expression):
        raise ValueError(f"Invalid expression: {expression!r}")
    return str(eval(expression, {"__builtins__": {}}, {}))


class JudgmentTaggerPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="judgment_tagger")

    async def before_run_callback(self, *, invocation_context) -> None:
        Tracer.set_customer_id(invocation_context.user_id)
        Tracer.set_session_id(invocation_context.session.id)


root_agent = Agent(
    model="gemini-2.5-flash",
    name="calculator_assistant",
    instruction="You are a calculator. Call the calculator tool with a Python arithmetic expression and report the result.",
    tools=[calculator],
)

app = App(name="my_agent", root_agent=root_agent, plugins=[JudgmentTaggerPlugin()])
