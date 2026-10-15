from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from judgeval.trace.tracer import Tracer


def test_project_id_export_bypasses_resolution():
    recorded: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _record(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            recorded.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            self._record()

        def do_GET(self):
            self._record()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    t = None
    try:
        port = server.server_address[1]
        t = Tracer.init(
            project_id="pid-e2e",
            api_key="dummy",
            organization_id="dummy-org",
            api_url=f"http://127.0.0.1:{port}",
        )

        @Tracer.observe()
        def work():
            return "ok"

        work()
        t._tracer_provider.force_flush()

        otel = [
            r
            for r in recorded
            if r["method"] == "POST" and r["path"].endswith("/otel/v1/traces")
        ]
        assert otel
        for r in otel:
            assert r["headers"].get("x-project-id") == "pid-e2e"
            assert r["headers"].get("x-organization-id") == "dummy-org"
            assert r["headers"].get("authorization") == "Bearer dummy"
        assert not any("/projects/resolve" in r["path"] for r in recorded)
    finally:
        if t is not None:
            t._tracer_provider.shutdown()
        server.shutdown()
        server.server_close()
