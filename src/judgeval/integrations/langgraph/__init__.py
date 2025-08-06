try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
    from langchain_core.agents import AgentAction, AgentFinish  # type: ignore
    from langchain_core.outputs import LLMResult  # type: ignore
    from langchain_core.messages.base import BaseMessage  # type: ignore
    from langchain_core.documents import Document  # type: ignore
except ImportError:
    raise ImportError(
        "Judgeval's langgraph integration requires langchain to be installed. Please install it with `pip install judgeval[langchain]`"
    )


class JudgevalCallbackHandler(BaseCallbackHandler): ...
