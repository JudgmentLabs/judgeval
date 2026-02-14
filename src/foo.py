import asyncio
import os
from openai import AsyncOpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from judgeval.v1.trace import Tracer

Tracer.registerOTELInstrumentation(OpenAIInstrumentor())
client = AsyncOpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.anthropic.com/v1",
)

# This works too
# fibonacci_tracer = Tracer.init(project_name="fibonacci", set_active=False)
# fizzbuzz_tracer = Tracer.init(project_name="fizzbuzz", set_active=False)
# chat_tracer = Tracer.init(project_name="chat", set_active=False)


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
    results = await asyncio.gather(
        handle_request(
            "fibonacci",
            customer_id="cust_001",
            session_id="sess_01",
            tags=["math", "recursive"],
            n=5,
        ),
        handle_request(
            "long_running_task",
            customer_id="cust_001",
            session_id="sess_02",
            tags=["long"],
            duration=10,
        ),
        handle_request(
            "chat",
            customer_id="cust_003",
            session_id="sess_03",
            tags=["llm", "research"],
            message="whats the latest anthropic model?",
        ),
        handle_request(
            "fibonacci",
            customer_id="cust_001",
            session_id="sess_04",
            tags=["math", "recursive"],
            n=8,
        ),
        handle_request(
            "fizzbuzz",
            customer_id="cust_003",
            session_id="sess_05",
            tags=["math"],
            n=30,
        ),
        handle_request(
            "chat",
            customer_id="cust_002",
            session_id="sess_06",
            tags=["llm"],
            message="explain quantum computing in one sentence",
        ),
        handle_request(
            "fibonacci",
            customer_id="cust_002",
            session_id="sess_07",
            tags=["math", "recursive"],
            n=3,
        ),
        handle_request(
            "fizzbuzz",
            customer_id="cust_001",
            session_id="sess_08",
            tags=["math", "iterative"],
            n=20,
        ),
        handle_request(
            "chat",
            customer_id="cust_001",
            session_id="sess_09",
            tags=["llm", "research"],
            message="what is RLHF?",
        ),
        handle_request(
            "fibonacci",
            customer_id="cust_003",
            session_id="sess_10",
            tags=["math"],
            n=6,
        ),
        handle_request(
            "fizzbuzz",
            customer_id="cust_002",
            session_id="sess_11",
            tags=["math"],
            n=10,
        ),
        handle_request(
            "chat",
            customer_id="cust_003",
            session_id="sess_12",
            tags=["llm"],
            message="summarize transformers architecture",
        ),
    )
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
