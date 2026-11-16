"""Run setup once, capture a request, then submit feedback in a later process.

uv run python main.py setup
uv run python main.py request message-123 conversation-456
uv run python main.py trace-feedback message-123 false
uv run python main.py session-feedback conversation-456 4.5

Set SDK credentials and JUDGMENT_PROJECT_NAME. Replace this demo's local JSON
file with your application database for multi-worker services.
"""

import argparse
import json
import os
from pathlib import Path

from judgeval import Judgeval, Tracer

PROJECT_NAME = os.environ.get("JUDGMENT_PROJECT_NAME", "my-project")
TARGETS_FILE = Path("feedback-targets.json")


def load_targets():
    return json.loads(TARGETS_FILE.read_text()) if TARGETS_FILE.exists() else {}


@Tracer.observe(span_type="agent")
def answer(message_id: str, session_id: str):
    Tracer.set_session_id(session_id)
    span = Tracer.get_current_span()
    if not span.is_recording():
        raise RuntimeError("Enable tracing before retaining a feedback target")
    trace_id = f"{span.get_span_context().trace_id:032x}"
    targets = load_targets()
    if message_id in targets:
        raise ValueError("Use a unique message ID for each response being rated")
    targets[message_id] = {
        "project_name": PROJECT_NAME,
        "trace_id": trace_id,
        "session_id": session_id,
    }
    TARGETS_FILE.write_text(json.dumps(targets, indent=2))
    return {"message_id": message_id, "answer": "Here is the answer."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["setup", "request", "trace-feedback", "session-feedback"]
    )
    parser.add_argument("target_id", nargs="?")
    parser.add_argument("detail", nargs="?")
    args = parser.parse_args()
    if args.command != "setup" and not args.target_id:
        parser.error("Provide the customer's message ID or session ID")

    if args.command == "request":
        if not args.detail:
            parser.error("request requires MESSAGE_ID SESSION_ID")
        Tracer.init(project_name=PROJECT_NAME)
        response = answer(args.target_id, args.detail)
        if not Tracer.force_flush():
            raise RuntimeError(
                "Trace export did not finish; do not submit feedback yet"
            )
        print(json.dumps(response))
        return

    client = Judgeval(project_name=PROJECT_NAME)
    if args.command == "setup":
        client.external_judges.create(name="Response helpful", score_type="binary")
        client.external_judges.create(name="Session quality", score_type="numeric")
    elif args.command == "trace-feedback":
        if args.detail not in (None, "true", "false"):
            parser.error("trace-feedback value must be true or false")
        target = load_targets().get(args.target_id)
        if target is None or target["project_name"] != PROJECT_NAME:
            raise ValueError(
                "No retained trace for this message in the current project"
            )
        # A later worker needs only the customer's message ID, not an active span.
        print(
            client.external_judges.submit_result(
                judge_name="Response helpful",
                target={"type": "trace", "trace_id": target["trace_id"]},
                value=args.detail == "true",
                reason="Feedback on this specific response.",
            )
        )
    else:
        # Use this only when the rating describes the entire session.
        # Its customer-owned ID was attached to each request with set_session_id.
        print(
            client.external_judges.submit_result(
                judge_name="Session quality",
                target={"type": "session", "session_id": args.target_id},
                value=float(args.detail) if args.detail is not None else 4.5,
                reason="Rating of the whole session.",
            )
        )


if __name__ == "__main__":
    main()
