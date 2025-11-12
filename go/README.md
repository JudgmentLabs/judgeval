# Judgeval Go SDK

Go SDK for [Judgeval](https://judgmentlabs.ai/) - Agent Behavior Monitoring framework.

## Installation

```bash
go get github.com/JudgmentLabs/judgeval/go
```

## Setup

Set environment variables:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

[Create a free account](https://app.judgmentlabs.ai/register) to get your keys.

## Quick Start

```go
package main

import (
    "context"
    "os"

    judgeval "github.com/JudgmentLabs/judgeval/go"
)

func main() {
    client, err := judgeval.NewJudgeval(
        judgeval.WithAPIKey(os.Getenv("JUDGMENT_API_KEY")),
        judgeval.WithOrganizationID(os.Getenv("JUDGMENT_ORG_ID")),
    )
    if err != nil {
        panic(err)
    }

    ctx := context.Background()
    tracer, err := client.Tracer.Create(ctx, judgeval.TracerCreateParams{
        ProjectName: "my_project",
    })
    if err != nil {
        panic(err)
    }
    defer tracer.Shutdown(ctx)

    ctx, span := tracer.Span(ctx, "agent-run")
    defer span.End()

    result := "The capital is Paris"

    tracer.AsyncEvaluate(
        ctx,
        client.Scorers.BuiltIn.AnswerRelevancy(judgeval.AnswerRelevancyScorerParams{}),
        judgeval.NewExample(judgeval.ExampleParams{
            Properties: map[string]any{
                "input": "What is the capital of France?",
                "actual_output": result,
            },
        }),
    )
}
```

## Documentation

- [Full Documentation](https://docs.judgmentlabs.ai/documentation)
- [Cookbooks](https://github.com/JudgmentLabs/judgment-cookbook)
- [Video Tutorials](https://www.youtube.com/@Alexshander-JL)

## Development

```bash
go mod download
go test ./...
go build ./...
```

## Links

- [Judgment Cloud](https://app.judgmentlabs.ai/register)
- [Self-Hosting Guide](https://docs.judgmentlabs.ai/documentation/self-hosting/get-started)
- [GitHub](https://github.com/JudgmentLabs/judgeval)
