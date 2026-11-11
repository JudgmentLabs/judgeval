"""Exercise real subprocess stdio without implementing another MCP client."""

import sys
import tempfile
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402
from tests.integrations.mcp.harness import (  # noqa: E402
    CONVERSATION_PROMPT_PREFIX,
    McpClient,
    styles,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("style", styles())
async def test_stdio_server_writes_only_protocol_messages(style, caplog):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER, style],
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    with tempfile.TemporaryFile("w+") as stderr, anyio.fail_after(30):
        async with stdio_client(parameters, errlog=stderr) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                client = McpClient(session)
                tools = await client.all_tools()
                outcome = await client.call(
                    "add", {"a": 1, "b": 2, "context": "Adding numbers"}
                )
        stderr.seek(0)
        logged = stderr.read()
    assert {
        "injected": "context" in tools["add"].get("properties", {}),
        "result": (outcome.is_error, outcome.texts[0]),
        "handoff": outcome.texts[-1].startswith(CONVERSATION_PROMPT_PREFIX),
        "warning_on_stderr": "judgeval warning before the server starts" in logged,
        "protocol_errors": [
            record.message
            for record in caplog.records
            if record.name.startswith("mcp.client.stdio") and record.levelno >= 40
        ],
    } == {
        "injected": True,
        "result": (False, "3"),
        "handoff": True,
        "warning_on_stderr": True,
        "protocol_errors": [],
    }
