# Judgeval JavaScript/TypeScript SDK

JavaScript/TypeScript SDK for [Judgeval](https://judgmentlabs.ai/) - Agent Behavior Monitoring framework.

## Installation

```bash
npm install judgeval
```

or

```bash
bun add judgeval
```

## Setup

Set environment variables:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

[Create a free account](https://app.judgmentlabs.ai/register) to get your keys.

## Quick Start

```typescript
import { Judgeval, Example } from "judgeval";
import OpenAI from "openai";

const client = Judgeval.create();

const tracer = await client.nodeTracer.create({
  projectName: "my_project",
  enableEvaluation: true,
  enableMonitoring: true,
});

const openai = new OpenAI();

async function chatWithUser(userMessage: string): Promise<string> {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: userMessage }],
  });

  const result = completion.choices[0].message.content || "";

  tracer.asyncEvaluate(
    client.scorers.builtIn.answerRelevancy(),
    Example.create({
      properties: {
        input: userMessage,
        actual_output: result,
      },
    })
  );

  return result;
}

const observedChat = tracer.observe(chatWithUser);
await observedChat("What is the capital of France?");
await tracer.shutdown();
```

## Documentation

- [Full Documentation](https://docs.judgmentlabs.ai/documentation)
- [Cookbooks](https://github.com/JudgmentLabs/judgment-cookbook)
- [Video Tutorials](https://www.youtube.com/@Alexshander-JL)

## Development

```bash
bun install
bun run check
bun test
bun run build
```

## Links

- [Judgment Cloud](https://app.judgmentlabs.ai/register)
- [Self-Hosting Guide](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)
- [GitHub](https://github.com/JudgmentLabs/judgeval)
