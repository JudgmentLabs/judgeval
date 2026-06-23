import os

os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_AND_EVENT"

import asyncio
from judgeval import Tracer
from judgeval.trace import JudgmentTracerProvider

Tracer.init(project_name="my-adk-agent")
JudgmentTracerProvider.install_as_global_tracer_provider()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from my_agent.agent import app


runner = Runner(app=app, session_service=InMemorySessionService())


async def main() -> None:
    user_id, session_id = "user-42", "session-1"
    await runner.session_service.create_session(
        app_name="my_agent", user_id=user_id, session_id=session_id
    )
    message = types.Content(
        role="user", parts=[types.Part(text="what is (3 + 4) * 5?")]
    )
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        for part in event.content.parts if event.content else []:
            if part.text:
                print(part.text)


if __name__ == "__main__":
    asyncio.run(main())
