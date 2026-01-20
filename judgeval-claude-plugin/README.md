# Judgeval Claude Code Plugin

Claude Code plugin for automatic tracing and observability with [Judgeval](https://judgmentlabs.ai).

## Install

```bash
claude plugin marketplace add JudgmentLabs/judgeval-claude-plugin
claude plugin install trace-claude-code@judgeval-claude-plugin
```

See [trace-claude-code/SKILL.md](skills/trace-claude-code/SKILL.md) for setup instructions.

## Setup

After installing, run the setup script in your project directory:

```bash
bash ~/.claude/plugins/marketplaces/judgeval-claude-plugin/skills/trace-claude-code/setup.sh
```

You'll need:
- `JUDGMENT_API_KEY` - Get from [Judgeval Settings](https://app.judgmentlabs.ai/settings/api-keys)
- `JUDGMENT_ORG_ID` - Get from [Organization Settings](https://app.judgmentlabs.ai/settings/organization)

## What You Get

```
Claude Code Session (root trace)
├── Turn 1: "Add error handling"
│   ├── LLM: claude-opus-4-5 (3.2s, 1,240 tokens)
│   ├── Read: src/app.ts
│   ├── Edit: src/app.ts
│   └── LLM: claude-opus-4-5 (1.8s, 890 tokens)
├── Turn 2: "Now run the tests"
│   ├── LLM: claude-opus-4-5
│   ├── Terminal: npm test
│   └── LLM: claude-opus-4-5
└── Turn 3: "Commit this"
    └── ...
```

**Captured data:**
- Session start/end times
- Each conversation turn
- All LLM calls with model, tokens, and duration
- Tool invocations (file reads, edits, terminal, MCP)
- Cache metrics (creation + read tokens)

## Development

Test locally without marketplace:

```bash
claude --plugin-dir /path/to/judgeval-claude-plugin
```

## Updating

After plugin updates are released:

```bash
claude plugin marketplace update judgeval-claude-plugin
claude plugin update trace-claude-code@judgeval-claude-plugin
```

## License

MIT
