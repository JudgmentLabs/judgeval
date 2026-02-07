# Judgeval Tracing for Claude Code

Automatically trace Claude Code sessions to Judgeval for observability and debugging.

## Features

- **Session Tracing**: Capture complete conversation sessions with start/end times
- **Turn Tracking**: Track each user prompt as a separate turn
- **LLM Spans**: Log every model call with input/output and token usage
- **Tool Spans**: Track tool invocations (file operations, terminal, MCP tools)
- **Cache Metrics**: Track cache creation and read tokens for prompt caching

## Trace Structure

```
Session (task span)
├── Turn 1 (task span)
│   ├── claude-opus-4-5 (llm span)
│   ├── Read: file.py (tool span)
│   ├── Subagent: code-reviewer (task span)  ← Subagent with nested spans
│   │   ├── claude-3-5-haiku (llm span)
│   │   ├── Read (tool span)
│   │   └── claude-3-5-haiku (llm span)
│   └── claude-opus-4-5 (llm span)
└── Turn 2 (task span)
    └── ...
```

## Setup

After installing the plugin, run setup in your project directory:

```bash
cd /path/to/your/project
bash ~/.claude/plugins/marketplaces/judgeval-claude-plugin/skills/trace-claude-code/setup.sh
```

This will prompt you for:
- `JUDGMENT_API_KEY` - Your Judgeval API key
- `JUDGMENT_ORG_ID` - Your organization ID
- `JUDGMENT_API_URL` - API URL (default: https://api.judgmentlabs.ai)
- `JUDGEVAL_CC_PROJECT` - Project name (default: claude-code)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TRACE_TO_JUDGEVAL` | Yes | Set to `true` to enable tracing |
| `JUDGMENT_API_KEY` | Yes | Your Judgeval API key |
| `JUDGMENT_ORG_ID` | Yes | Your organization ID |
| `JUDGMENT_API_URL` | No | API URL (default: https://api.judgmentlabs.ai) |
| `JUDGEVAL_CC_PROJECT` | No | Project name (default: claude-code) |
| `JUDGEVAL_CC_DEBUG` | No | Set to `true` for debug logging |

## Hooks

| Hook | Trigger | Action |
|------|---------|--------|
| `session_start.sh` | Session begins | Creates root trace span |
| `user_prompt_submit.sh` | User sends prompt | Creates Turn span |
| `post_tool_use.sh` | Tool completes | Tracks tool count |
| `stop_hook.sh` | Response complete | Marks turn for finalization |
| `subagent_stop.sh` | Subagent completes | Parses subagent transcript, creates nested spans |
| `session_end.sh` | Session ends | Creates LLM/Tool spans, finalizes session |

## Span Attributes

### Session Span
- `judgment.span_kind`: "task"
- `judgment.input`: Session description
- `judgment.output`: Completion summary
- `turn_count`: Number of turns

### LLM Span
- `judgment.span_kind`: "llm"
- `judgment.input`: Conversation history
- `judgment.output`: Model response
- `judgment.llm.model`: Model name
- `judgment.llm.provider`: "anthropic"
- `judgment.usage.non_cached_input_tokens`: Input tokens
- `judgment.usage.output_tokens`: Output tokens
- `judgment.usage.cache_creation_input_tokens`: Cache write tokens
- `judgment.usage.cache_read_input_tokens`: Cache read tokens

### Tool Span
- `judgment.span_kind`: "tool"
- `judgment.input`: Tool input
- `judgment.output`: Tool output
- `tool_name`: Tool identifier

## Logs

Hook logs are written to: `~/.claude/state/judgeval_hook.log`

Enable debug logging:
```bash
export JUDGEVAL_CC_DEBUG=true
```

## Troubleshooting

**Traces not appearing:**
1. Check `TRACE_TO_JUDGEVAL=true` is set
2. Verify API key and org ID are correct
3. Check logs for errors

**Missing spans:**
1. Ensure all hooks are executable
2. Check for jq/curl availability
3. Review debug logs
