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

### MCP tool instrumentation

After `Tracer.init()`, instrument your MCP server with optional argument-name overrides:

```python
from judgeval.integrations.mcp import instrument_mcp_server

instrument_mcp_server(server, intent_argument="intent", model_argument="judgment_llm_model",
                      conversation_id_argument="chat_id")
```

Explicit names reuse existing tool arguments without changing their schemas or removing them from handler inputs; when absent, they are injected into compatible schemas and stripped before execution. Defaults are `intent`, `llm_model`, and `conversation_id`. Without explicit configuration, same-name business arguments are preserved but not interpreted as telemetry. Explicit fields can be read with `inject_agent_arguments=False`; missing/non-string values are omitted. Existing conversation fields supply application sessions, while injected ones use generated UUIDv7 handles. Names must be non-empty and distinct.

Injection changes the advertised tool contract, including an optional `_judgment` property on compatible output schemas. During a mixed instrumented/uninstrumented rollout, strict clients holding an older output schema may reject structured results containing this property. Keep replica schemas consistent and refresh client tool discovery after rollout; gateways caching tool lists must refresh too.

Injection is independent of telemetry availability: installing instrumentation before `Tracer.init()`, or disabling export, does not disable schema changes or conversation handoff. Injected fields and descriptions add model-context overhead per tool when clients expose the listing to an agent. Set `inject_agent_arguments=False` to disable automatic argument injection and handoff while retaining tracing, or `inject_conversation_id=False` to disable only conversation-ID injection and handoff.

With conversation injection enabled, returned tool results without an application or echoed conversation session offer a new ID even on stateful HTTP, including `isError: true` results. Session priority is an explicit application session, then an echoed conversation ID, then a newly minted ID, then the transport session. The first call and later calls echoing its ID share the same session; clients that never echo receive fresh IDs on each call. Set `inject_conversation_id=False` to use transport grouping when no explicit session is supplied. The transport ID remains recorded separately as `mcp.session.id`. Handoff preserves existing error content and flags; thrown exceptions and protocol errors propagate unchanged without handoff. Dictionary outputs stay unchanged and use supplemental text for handoff. Native per-tool lookup avoids full discovery where available; custom low-level servers retain caller-scoped listing.

Low-level servers can avoid full discovery by supplying `resolve_original_tool(name)`, a sync or async callback returning an MCP `Tool` object or the original wire tool descriptor (`inputSchema` and optional `outputSchema`):

```python
instrument_mcp_server(server, resolve_original_tool=lambda name: original_tools.get(name))
```

Use the same original schemas as your `tools/list` handler, before Judgment injection, and scope the lookup to the current caller if tools vary by authorization or request. The SDK does not cache resolver results across calls. When configured, the resolver bypasses the listing cache and replaces discovery: returning `None`, raising, timing out, or returning an unsupported schema leaves arguments untouched and omits generated handoff. High-level servers still prefer their native lookup.

Without a resolver, low-level servers cache up to 1,000 original contracts by tool name, refreshing them when listings are served and clearing them on outgoing tool-list-change notifications or tool handler replacement. Concurrent misses share discovery, and missing or failed lookups are cached for one second. Calls reuse known contracts to strip only SDK-owned arguments, even if discovery later fails. This assumes a named tool's schema is consistent across callers and schema changes are announced; use a caller-aware resolver otherwise. Fresh replicas and cache misses fall back to caller-scoped discovery, capped at 100 pages and one second; unknown ownership remains untouched if discovery fails. Async resolver lookup has the same timeout. These limits cannot preempt blocking synchronous callbacks or forcibly stop cancellation-resistant coroutines. No cache is shared across processes.

FastMCP 2 proxy servers use the same contract cache and discovery policy instead of native per-tool lookup, which would list upstream tools on every call. Successful entries have no time-based expiry; schema changes must be announced or supplied by a caller-aware `resolve_original_tool` callback. Ordinary local FastMCP servers keep fresh native lookups.

Repeated instrumentation keeps the original configuration and warns if options differ. Optional input/output capture is limited to 128 KiB; oversized source payloads are omitted before model conversion or the redaction callback.

The SDK can record the original exception type when the exception reaches its wrapper or is supplied through `mark_mcp_call_error` before conversion to a public tool error. MCP frameworks may instead convert exceptions into `isError` results before the wrapper sees them; these generally use `error.type = "tool_error"`, since the original exception type is no longer available. Recognized refusals use `unknown_tool`, `tool_disabled`, or `invalid_input`; a structured public error code may also be used when error-message capture is enabled.

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

## CLI

Manage agents, traces, judges, labels, and evaluations from the terminal. Query trace history, deploy judges, review the labels judges applied, and run evals against production data — all without leaving your shell. See the [CLI repo](https://github.com/JudgmentLabs/cli/) and [docs](https://docs.judgmentlabs.ai/documentation/cli).

## MCP Server

Connect Judgment to any MCP-compatible AI tool. Query agent traces, invoke judges, browse the labels judges applied, and surface failures directly inside your AI assistant or IDE. See the [docs](https://docs.judgmentlabs.ai/documentation/mcp-server).

## Links

- [Documentation](https://docs.judgmentlabs.ai/documentation)

---

Judgeval is created and maintained by [Judgment Labs](https://judgmentlabs.ai/).
