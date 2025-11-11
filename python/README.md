# Judgeval Python SDK

Python SDK for [Judgeval](https://judgmentlabs.ai/) - Agent Behavior Monitoring framework.

## Installation

```bash
pip install judgeval
```

## Setup

Set environment variables:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

[Create a free account](https://app.judgmentlabs.ai/register) to get your keys.

## Quick Start

```python
from judgeval.v1 import Judgeval
from judgeval.v1.data import Example
from openai import OpenAI

client = Judgeval()
tracer = client.tracer.create(
    project_name="my_project",
    enable_evaluation=True,
    enable_monitoring=True,
)

openai_client = OpenAI()

def chat_with_user(user_message: str) -> str:
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}]
    )

    result = completion.choices[0].message.content

    tracer.async_evaluate(
        client.scorers.built_in.answer_relevancy(),
        Example.create(
            properties={
                "input": user_message,
                "actual_output": result,
            }
        ),
    )

    return result

observed_chat = tracer.observe(chat_with_user)
result = observed_chat("What is the capital of France?")
tracer.shutdown()
```

## Documentation

- [Full Documentation](https://docs.judgmentlabs.ai/documentation)
- [Cookbooks](https://github.com/JudgmentLabs/judgment-cookbook)
- [Video Tutorials](https://www.youtube.com/@Alexshander-JL)

## Development

```bash
pip install uv
uv sync --dev
uv run pytest tests
```

## Links

- [Judgment Cloud](https://app.judgmentlabs.ai/register)
- [Self-Hosting Guide](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)
- [GitHub](https://github.com/JudgmentLabs/judgeval)
