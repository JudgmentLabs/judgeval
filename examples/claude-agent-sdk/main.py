import asyncio
from typing import Any

from judgeval import Tracer
from judgeval.integrations.claude_agent_sdk import setup_claude_agent_sdk

Tracer.init(project_name="claude-agent-sdk")
setup_claude_agent_sdk()

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)


@tool("lookup_weather", "Get weather for a city", {"city": str})
async def lookup_weather(args: dict[str, Any]) -> dict[str, Any]:
    data = {
        "san francisco": "62F, foggy",
        "new york": "45F, cloudy",
        "tokyo": "73F, sunny",
    }
    weather = data.get(args["city"].lower(), "unknown")
    return {"content": [{"type": "text", "text": f"{args['city']}: {weather}"}]}


server = create_sdk_mcp_server("demo", tools=[lookup_weather])

options = ClaudeAgentOptions(
    mcp_servers={"demo": server},
    allowed_tools=["mcp__demo__lookup_weather"],
    permission_mode="bypassPermissions",
    max_turns=5,
)


def print_response(msg):
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text)
            elif isinstance(block, ToolUseBlock):
                print(f"  -> {block.name}({block.input})")


async def main():
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Check weather in Tokyo and San Francisco. Be brief.")
        async for msg in client.receive_response():
            print_response(msg)

        await client.query("Which city was warmer?")
        async for msg in client.receive_response():
            print_response(msg)


if __name__ == "__main__":
    asyncio.run(main())
