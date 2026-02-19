import asyncio
import os
import uuid
from openai import AsyncOpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from judgeval.v1.trace import Tracer

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


@Tracer.observe()
async def long_running_task(duration: float) -> str:
    await asyncio.sleep(duration)
    with Tracer.span("long_running_task_inner"):
        Tracer.set_input("hi")
    return f"Sleeping for {duration} seconds"


async def handle_request(
    name: str,
    customer_id: str,
    session_id: str,
    tags: list[str],
    **kwargs,
):
    Tracer.init(project_name=name)
    with Tracer.span("handle"):
        Tracer.set_customer_id(customer_id)
        Tracer.set_session_id(session_id)
        Tracer.tag(tags)

        if name == "fibonacci":
            return await fibonacci(kwargs["n"])
        elif name == "fizzbuzz":
            return await fizzbuzz(kwargs["n"])
        elif name == "chat":
            return await chat(kwargs["message"])
        elif name == "long_running_task":
            return await long_running_task(kwargs["duration"])

        raise ValueError(f"Unknown: {name}")


async def main():
    session_id = str(uuid.uuid4())

    await handle_request(
        "chat",
        customer_id="cust_001",
        session_id=session_id,
        tags=["llm"],
        message="whats the latest anthropic model?",
    )

    await asyncio.sleep(3)
    await handle_request(
        "chat",
        customer_id="cust_001",
        session_id=session_id,
        tags=["llm"],
        message="explain quantum computing in one sentence",
    )

    await asyncio.sleep(3)
    await handle_request(
        "chat",
        customer_id="cust_001",
        session_id=session_id,
        tags=["llm"],
        message="what is RLHF?",
    )


if __name__ == "__main__":
    asyncio.run(main())
    Tracer.shutdown(timeout_millis=100000)
