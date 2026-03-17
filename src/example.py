import asyncio
from judgeval import Tracer
from judgeval.v1.integrations.claude_agent_sdk import setup_claude_agent_sdk

tracer = Tracer.init(project_name="test")
setup_claude_agent_sdk()


from claude_agent_sdk import query


async def task(input: str) -> str:
    last = ""
    result = None
    async for message in query(prompt=input):
        if hasattr(message, "result"):
            result = getattr(message, "result", "") or last
        for b in getattr(message, "content", None) or []:
            if hasattr(b, "text"):
                last = getattr(b, "text", "") or last
    return result or last


@Tracer.observe()
async def main():
    result = await task("Whats 2 + 2?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
