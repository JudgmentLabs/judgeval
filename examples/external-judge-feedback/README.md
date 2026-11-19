# Feedback after a traced request

Set `JUDGMENT_API_KEY`, `JUDGMENT_ORG_ID`, and `JUDGMENT_PROJECT_NAME` for an existing project. From this directory, run:

```sh
uv run python main.py setup
uv run python main.py request message-123 conversation-456
# Once the trace is visible in Judgment, run in a later process:
uv run python main.py trace-feedback message-123 false
uv run python main.py session-feedback conversation-456 4.5
```

Run `setup` once per project. A request fails before saving a feedback target if tracing cannot resolve the project. `request` captures the active span's 32-character trace ID and retains it in `feedback-targets.json`, keyed by your unique message ID. `trace-feedback` looks up that mapping after the request process has exited; it does not need an active span or a trace search. Keep the same working directory between commands.

Use trace feedback for one response/run. Use session feedback only for an overall conversation rating. Attach your customer-owned session/conversation ID to every traced request with `Tracer.set_session_id`; an arbitrary ID without ingested traces is not a session in Judgment.

The local file is a single-process demo. Use an organization/project-scoped database mapping and unique message IDs in production; authorize feedback against that message and never repoint its saved trace ID. Keep mappings on your server. Trace export completion does not guarantee immediate query visibility: if submission returns target-not-found during ingestion, wait and retry the same target. Do not fall back to a session or a search match.

There is one current customer score per judge and target. Another submission replaces it, so a delayed retry can overwrite a newer correction. Judgment-generated scores remain separate.

Conversation/trajectory feedback uses its recorded `conversation_id` directly:

```python
client.external_judges.submit_result(
    judge_name="Session quality",
    target={"type": "conversation", "conversation_id": "conversation-456"},
    value=4.5,
)
```

This target requires recorded conversation observations (for example, from `Trajectory.record`). It does not require a trace or a traced session. Wait for ingestion visibility before submitting. The same judge may rate a session and a conversation; their results remain independent. The supporting conversation-feedback API must be deployed before using this target.
