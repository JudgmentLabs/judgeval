"""A server over stdio, run as its own process the way an MCP client starts it.

Over stdio the server's stdout is the protocol, so nothing else may write there.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, List

import pytest

pytest.importorskip("mcp")

from tests.integrations.mcp.harness import (  # noqa: E402
    CONVERSATION_PROMPT_PREFIX,
    styles,
)

SRC = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

SERVER = """
import sys

import anyio

from judgeval.integrations.mcp import instrument_mcp_server
from judgeval.logger import judgeval_logger
from tests.integrations.mcp.harness import LOW_LEVEL, build_server

style = sys.argv[1]
judgeval_logger.warning("judgeval warning before the server starts")
orders = build_server(style)
instrument_mcp_server(orders.server, record_spans=True)
if style in LOW_LEVEL:
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read, write):
            await orders.server.run(
                read, write, orders.server.create_initialization_options()
            )

    anyio.run(main)
else:
    orders.server.run("stdio")
"""


class StdioServer:
    def __init__(self, style: str, stderr: Any) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", SERVER, style],
            cwd=SRC,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.messages: List[Dict[str, Any]] = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)

    def send(self, message: Dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        self.process.stdin.flush()

    def response(self, request_id: int) -> Dict[str, Any]:
        while True:
            line = self.lines.get(timeout=60)
            # Every line the server writes must be a protocol message.
            message = json.loads(line)
            self.messages.append(message)
            if message.get("id") == request_id:
                return message

    def close(self) -> None:
        # Shut down the way MCP clients do: close stdin, then terminate. Older
        # mcp servers (1.6) don't exit on end of input.
        assert self.process.stdin is not None
        self.process.stdin.close()
        try:
            self.process.wait(5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(10)
        assert self.process.stdout is not None
        self.process.stdout.close()
        while not self.lines.empty():
            self.messages.append(json.loads(self.lines.get()))


@pytest.mark.parametrize("style", styles())
def test_stdio_server_writes_only_protocol_messages(style):
    with tempfile.TemporaryFile("w+") as stderr:
        server = StdioServer(style, stderr)
        try:
            server.send(
                {
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "stdio-test", "version": "1.0.0"},
                    },
                }
            )
            assert "result" in server.response(1)
            server.send({"method": "notifications/initialized"})
            server.send({"id": 2, "method": "tools/list"})
            listed = server.response(2)["result"]["tools"]
            add = next(tool for tool in listed if tool["name"] == "add")
            assert "context" in add["inputSchema"]["properties"]
            server.send(
                {
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "add",
                        "arguments": {"a": 1, "b": 2, "context": "Adding numbers"},
                    },
                }
            )
            result = server.response(3)["result"]
        finally:
            server.close()
        stderr.seek(0)
        logged = stderr.read()

    assert result.get("isError") is not True, result
    texts = [item["text"] for item in result["content"] if item["type"] == "text"]
    assert texts[0] == "3"
    # Over stdio there's no session, so the agent is given a conversation ID.
    assert texts[-1].startswith(CONVERSATION_PROMPT_PREFIX)
    assert "judgeval warning before the server starts" in logged
