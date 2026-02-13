import asyncio
import os
from openai import AsyncOpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from judgeval.v1.tracer_v2 import Tracer

Tracer.registerOTELInstrumentation(OpenAIInstrumentor())
client = AsyncOpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.anthropic.com/v1",
)

# This works too
# fibonacci_tracer = Tracer.create(project_name="fibonacci", set_active=False)
# fizzbuzz_tracer = Tracer.create(project_name="fizzbuzz", set_active=False)
# chat_tracer = Tracer.create(project_name="chat", set_active=False)


@Tracer.observe()
async def fibonacci(n: int) -> int:
    await asyncio.sleep(0.1)
    if n <= 1:
        return n
    a, b = await asyncio.gather(fibonacci(n - 1), fibonacci(n - 2))
    return a + b


@Tracer.observe()
async def fizzbuzz(n: int) -> list[str]:
    await asyncio.sleep(0.1)
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


@Tracer.observe()
async def chat(message: str) -> str:
    response = await client.chat.completions.create(
        model="claude-opus-4-1",
        messages=[{"role": "user", "content": message}],
        max_tokens=50,
    )
    Tracer.asyncTraceEvaluate("Hallucination")
    return response.choices[0].message.content or ""


async def handle_request(name: str, **kwargs):
    if name == "fibonacci":
        Tracer.create(project_name="fibonacci")
        return await fibonacci(kwargs["n"])
    elif name == "fizzbuzz":
        Tracer.create(project_name="fizzbuzz")
        return await fizzbuzz(kwargs["n"])
    elif name == "chat":
        Tracer.create(project_name="chat")
        return await chat(kwargs["message"])
    raise ValueError(f"Unknown: {name}")


async def main():
    results = await asyncio.gather(
        handle_request("fibonacci", n=5),
        handle_request("fizzbuzz", n=15),
        handle_request(
            "chat",
            message="whats the latest anthropic model?",
        ),
    )
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
