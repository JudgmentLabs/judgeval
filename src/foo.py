from openai import OpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

from judgeval.v1 import Judgeval, observe

judgeval = Judgeval()

tracer1 = judgeval.tracer.create("mt-tracer-1")
tracer2 = judgeval.tracer.create("mt-tracer-2")

OpenAIInstrumentor().instrument(tracer_provider=tracer2.tracer_provider)

client = OpenAI()


@tracer1.observe(span_type="tool")
def tool_1(x: int) -> int:
    return x + 1


@tracer2.observe(span_type="tool")
def tool_2(x: int) -> int:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"What is {x} + 2?. Return the answer as an integer. Do not include any other text in your response.",
            }
        ],
        max_tokens=50,
    )
    return int(response.choices[0].message.content)


@observe(span_type="tool")
def tool_3(x: int) -> int:
    return x + 3


@observe()
def main():
    tool_1(1)
    result = tool_2(2)
    print(f"tool_2 result: {result}")
    tool_3(3)


if __name__ == "__main__":
    main()
