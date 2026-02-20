from anthropic.types import TextBlock
from judgeval import Tracer
from judgeval.v1.integrations.openlit import Openlit
from anthropic import Anthropic

tracer = Tracer.init(project_name="openlit")
Openlit.initialize(disabled_instrumentors=["agno"])

client = Anthropic()


@Tracer.observe()
def run():
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    assert isinstance(response.content[0], TextBlock)
    return response.content[0].text


if __name__ == "__main__":
    run()
