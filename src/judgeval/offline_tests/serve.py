"""Serve an agent as an endpoint the Judgment platform can dispatch runs to.

A platform-started test run with an *endpoint* target sends one webhook per
run (``test_run.agent_requested``). The server here accepts it, replies
``202`` immediately, and attaches the agent to the run in a background
thread, streaming traces back through ``OfflineTestRunner.attach``. Run it
wherever the agent can execute: a laptop, a staging service, or a sandbox.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

from judgeval.logger import judgeval_logger

AGENT_REQUESTED_EVENT = "test_run.agent_requested"
SECRET_ENV_VAR = "JUDGMENT_AGENT_TARGET_SECRET"

AttachFn = Callable[[str], Any]


def _authorized(headers: Any, secret: Optional[str]) -> bool:
    if not secret:
        return True
    return headers.get("Authorization") == f"Bearer {secret}"


def serve_agent(
    attach: AttachFn,
    host: str = "0.0.0.0",
    port: int = 8787,
    path: str = "/judgment/run",
    secret: Optional[str] = None,
    banner: Optional[Callable[[str], None]] = None,
) -> None:
    """Block serving ``path`` until interrupted.

    ``attach`` is called with the ``test_run_id`` from each accepted webhook.
    ``secret`` (or ``JUDGMENT_AGENT_TARGET_SECRET``) must match the target's
    saved secret when one is configured on the platform.
    """
    shared_secret = secret or os.environ.get(SECRET_ENV_VAR) or None
    active: Dict[str, threading.Thread] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            judgeval_logger.debug("serve_agent: " + format % args)

        def _reply(self, status: int, body: Dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._reply(
                200,
                {
                    "ok": True,
                    "path": path,
                    "active_runs": sorted(
                        run_id for run_id, t in active.items() if t.is_alive()
                    ),
                },
            )

        def do_POST(self) -> None:
            if self.path.rstrip("/") != path.rstrip("/"):
                self._reply(404, {"error": "not found"})
                return
            if not _authorized(self.headers, shared_secret):
                self._reply(401, {"error": "invalid agent target secret"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._reply(400, {"error": "invalid JSON"})
                return
            if body.get("event") != AGENT_REQUESTED_EVENT:
                self._reply(400, {"error": f"unsupported event {body.get('event')!r}"})
                return
            test_run_id = str(body.get("test_run_id") or "")
            if not test_run_id:
                self._reply(400, {"error": "test_run_id is required"})
                return
            existing = active.get(test_run_id)
            if existing is not None and existing.is_alive():
                self._reply(202, {"accepted": True, "test_run_id": test_run_id})
                return

            def work() -> None:
                try:
                    attach(test_run_id)
                except Exception as exc:  # pragma: no cover - surfaced in logs
                    judgeval_logger.error(
                        f"Agent attach failed for test run {test_run_id}: {exc}"
                    )

            thread = threading.Thread(target=work, name=f"attach-{test_run_id}")
            thread.daemon = True
            active[test_run_id] = thread
            thread.start()
            self._reply(202, {"accepted": True, "test_run_id": test_run_id})

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'localhost' if host in ('0.0.0.0', '') else host}:{port}{path}"
    if banner is not None:
        banner(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
