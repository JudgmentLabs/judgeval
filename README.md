<div align="center">

<a href="https://judgmentlabs.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_darkmode.svg">
    <img src="assets/logo_lightmode.svg" alt="Judgment Logo" width="400" />
  </picture>
</a>

<br>

## The Continuous-Improvement Stack for Agents

Detect failures, triage root causes, and ship fixes backed by production data.

[![PyPI](https://img.shields.io/pypi/v/judgeval?color=orange)](https://pypi.org/project/judgeval/)
[![Docs](https://img.shields.io/badge/Documentation-orange)](https://docs.judgmentlabs.ai/documentation)

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)

</div>

## Overview

Judgeval is an open-source Python SDK for agent improvement. It provides tracing and agent-judge evaluation for LLM-powered applications — so you can detect failures, understand what went wrong, and validate fixes against real production cases before shipping.

To get started, dive into the [docs](https://docs.judgmentlabs.ai/documentation).

## Why Judgeval

**OpenTelemetry-based tracing** -- Instrument any function with `@Tracer.observe()`. Automatically captures inputs, outputs, and LLM token usage. Built on OpenTelemetry for full compatibility with existing observability stacks.

**Agent judges** -- Define prompt-based scorers to evaluate agent behaviors at scale. Judges produce structured behaviors — scored, labeled outputs that describe how your agent acted — which accumulate into a searchable record of agent behavior over time. Run judges against live production traffic or replay them on historical traces to validate fixes before shipping.

**Online monitoring** -- Automatically score live production traffic server-side with no latency impact. Detected behaviors surface as structured signals — configure Slack alerts so regressions and recurrences never go unnoticed.

**Broad integrations** -- Auto-instrumentation for OpenAI, Anthropic, Google GenAI, and Together AI. Framework support for LangGraph, OpenLit, and Claude Agent SDK.

## Quickstart

Install the SDK:

```bash
pip install judgeval
```

Set your credentials:

```bash
export JUDGMENT_API_KEY=...
export JUDGMENT_ORG_ID=...
```

Add observability to your agent with two lines of setup:

```python
from judgeval import Tracer, wrap
from openai import OpenAI

Tracer.init(project_name="my-project")
client = wrap(OpenAI())

@Tracer.observe(span_type="tool")
def search(query: str) -> str:
    results = vector_db.search(query)
    return results

@Tracer.observe(span_type="agent")
def run_agent(question: str) -> str:
    context = search(question)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{context}\n\n{question}"}],
    )
    return response.choices[0].message.content

run_agent("What is the capital of the United States?")
```

## Integrations

Supports OpenAI, Anthropic, Google GenAI, Together AI, LangGraph, OpenLit, and Claude Agent SDK. See the full [integrations docs](https://docs.judgmentlabs.ai/documentation/integrations/introduction).

## CLI

Manage agents, traces, judges, behaviors, and evaluations from the terminal. Query trace history, deploy judges, inspect detected behaviors, and run evals against production data — all without leaving your shell. See the [CLI repo](https://github.com/JudgmentLabs/cli/) and [docs](https://docs.judgmentlabs.ai/documentation/cli).

## MCP Server

Connect Judgment to any MCP-compatible AI tool. Query agent traces, invoke judges, browse detected behaviors, and surface failures directly inside your AI assistant or IDE. See the [docs](https://docs.judgmentlabs.ai/documentation/mcp-server).

## Links

- [Documentation](https://docs.judgmentlabs.ai/documentation)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
