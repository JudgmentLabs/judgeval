from judgeval import Tracer, wrap
from openai import OpenAI

Tracer.init(project_name="default_project")
client = wrap(OpenAI())


@Tracer.observe()
def add(a: int, b: int) -> int:
    Tracer.async_evaluate("Calculator")
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a calculator. You are given two integers and you need to return the sum of the two integers. Return the answer as an integer no extra text. Answer only the number, no other text, formatting, or markdown.",
            },
            {
                "role": "user",
                "content": f"{a} + {b}",
            },
        ],
        max_completion_tokens=256,
    )

    result = response.choices[0].message.content or ""
    if result and result.isdigit():
        return int(result)
    else:
        raise ValueError(f"Invalid response: {result}")


if __name__ == "__main__":
    result = add(1, 2)
    print(f"Result: {result}")
