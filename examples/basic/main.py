from judgeval import Tracer, wrap
from openai import OpenAI

Tracer.init(project_name="my-project")
client = wrap(OpenAI())


@Tracer.observe()
def add(a: int, b: int) -> int:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"What is the sum of {a} and {b}? Return the answer as an integer no extra text.",
            }
        ],
        max_tokens=64,
    )
    return int(response.choices[0].message.content)


if __name__ == "__main__":
    result = add(1, 2)
    print(f"Result: {result}")
