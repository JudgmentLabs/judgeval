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

Judgeval is an open-source Python SDK for agent improvement. It provides tracing and Agent judge evaluation for LLM-powered applications — so you can detect failures, understand what went wrong, and validate fixes against real production cases before shipping.

To get started, dive into the [docs](https://docs.judgmentlabs.ai/documentation).

## Why Judgeval

**OpenTelemetry-based tracing** -- Instrument any function with `@Tracer.observe()`. Automatically captures inputs, outputs, and LLM token usage. Built on OpenTelemetry for full compatibility with existing observability stacks.

**Agent judges** -- Define prompt-based scorers to evaluate traces at scale. Judges apply labels to traces — a binary judge applies its label when true, and a classification judge applies the chosen option — and those labels accumulate into a searchable record of how your agent acted over time. Run judges against live production traffic or replay them on historical traces to validate fixes before shipping.

**Online monitoring** -- Automatically score live production traffic server-side with no latency impact. When a judge applies a label, that label surfaces as a structured signal — configure Slack alerts so regressions and recurrences never go unnoticed.

**Broad integrations** -- Auto-instrumentation for OpenAI, Anthropic, Google GenAI, and Together AI. Framework support for LangGraph, and Claude Agent SDK.

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

### SQL

Use `Judgeval.sql(sql_text)` for read-only queries against Judgment's virtual
schema, which abstracts the underlying storage. The server validates incoming
queries, rejects writes, and enforces organization and project scope. The client
uses its existing API key, organization membership, and resolved project.
Viewer access and the public query rate limit apply.

```python
from judgeval import Judgeval

client = Judgeval(project_name="my-project")
print(client.discover_schema())
result = client.sql("SELECT count() AS run_count FROM telemetry.traces")
print(result["rows"])
```

`client.discover_schema()` returns a Markdown string with the server's generated
tables, column types and descriptions, row semantics, examples, and query limits,
using the same reference as MCP `discover_schema`. It contains no project data
and requires organization viewer access, but no resolved project. The HTTP
equivalent is `GET /v1/sql/schema`, which returns
`{"schema": "...Markdown reference..."}`.

For query execution, use `POST /v1/projects/{projectId}/sql` with
`Authorization: Bearer <api-key>`, `X-Organization-Id: <organization-id>`, and
JSON body `{"sql": "SELECT count() AS run_count FROM telemetry.traces"}`.
Organization and project scope are derived by the server. Use SQL predicates on
supported catalog columns and `LIMIT` to narrow results. Physical tables,
writes, multiple statements, and caller-specified execution limits are
unsupported. DAL catalog allowlists, tenant isolation, and result limits of
1,000 rows and 5 MiB apply; over-limit results return an error. SQL text must
contain a non-whitespace character and cannot exceed 50,000 characters.

The response contains `catalog_version`, `columns` (name, type, nullable),
`rows`, `row_count`, and `elapsed_ms`.
SQL integers outside JavaScript's safe range (`-(2**53 - 1)` to `2**53 - 1`)
arrive as exact decimal strings, including inside nested arrays and objects.
For example, `9007199254740993` arrives as `"9007199254740993"`; use `int(value)`
when you need a Python integer. Small integers and floating-point values remain
numbers, and column types retain their original SQL types.
Validation and execution errors use the SDK's existing exception mapping.

## Integrations

Supports OpenAI, Anthropic, Google GenAI, Together AI, LangGraph, and Claude Agent SDK. See the full [integrations docs](https://docs.judgmentlabs.ai/documentation/integrations/introduction).

### Streaming capture

The Anthropic Messages and OpenAI Chat Completions/Responses wrappers record one
structured response after successful stream consumption, including text, tool
call IDs, names, arguments, and provider reasoning fields. Sync and async paths
follow the same rules. Anthropic `messages.stream()` also captures calls consumed
through `text_stream`, event iteration, `until_done()`, or `get_final_message()`;
its response is recorded when the stream context exits normally.

A response requires the provider's completion signal: Anthropic `message_stop`,
a finish reason for every requested Chat Completions choice, or Responses
`response.completed` with completed status. Raw streams must also exhaust
normally. A finish reason such as Chat Completions `length` is retained as sent;
it does not mean that the model finished its answer without a token limit.

Abandoned, truncated, cancelled, and failed streams do not get a completed output.
Responses `response.incomplete` likewise produces no completed output. Available
model and usage metadata is retained; missing final usage is not estimated.
Instrumentation never drains a stream on close or context exit. Fully consume
streams to capture output. Close partially consumed iterators (`close()` or
`aclose()`) and use the Anthropic helper's context manager to finalize spans;
simply breaking iteration does not guarantee immediate span finalization. A raw
stream that is never iterated retains the existing lazy iterator lifecycle and
may leave its span unfinished.

Conversation ingestion of native Anthropic responses and Python Responses text
requires the server changes in
[JudgmentLabs/judgment-mono#6164](https://github.com/JudgmentLabs/judgment-mono/pull/6164).
Capture preserves provider fields even when the conversation mapper does not yet
support them: for example, redacted Anthropic thinking and multiple Chat
Completions choices. Chat Completions `reasoning_content` remains in the trace;
the current mapper does not turn it into a reasoning item.

## CLI

Manage agents, traces, judges, labels, and evaluations from the terminal. Query trace history, deploy judges, review the labels judges applied, and run evals against production data — all without leaving your shell. See the [CLI repo](https://github.com/JudgmentLabs/cli/) and [docs](https://docs.judgmentlabs.ai/documentation/cli).

## MCP Server

Connect Judgment to any MCP-compatible AI tool. Query agent traces, invoke judges, browse the labels judges applied, and surface failures directly inside your AI assistant or IDE. See the [docs](https://docs.judgmentlabs.ai/documentation/mcp-server).

## Links

- [Documentation](https://docs.judgmentlabs.ai/documentation)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
