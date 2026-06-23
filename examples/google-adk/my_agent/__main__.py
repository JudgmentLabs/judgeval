"""Run the agent directly from the terminal: `python -m my_agent`.

This drives `root_agent` with ADK's Runner + an in-memory session,
no `adk web` required. Judgment tracing is wired up via agent.py's
import side effects (it installs the span processor on import).

Requires:
  GOOGLE_API_KEY   (for gemini-2.5-flash via the Gemini API)
  JUDGMENT_API_KEY, JUDGMENT_ORG_ID  (for the Judgment dashboard)
"""

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import root_agent

APP_NAME = "my_agent"
USER_ID = "local-user"
SESSION_ID = "local-session"


async def main() -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print(f"Chatting with '{root_agent.name}'. Ctrl-C or empty line to quit.\n")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break

        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
                print(f"agent> {text}\n")


if __name__ == "__main__":
    asyncio.run(main())
