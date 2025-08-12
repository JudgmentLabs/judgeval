try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult
    from langchain_core.messages.base import BaseMessage
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "Judgeval's langgraph integration requires langchain to be installed. Please install it with `pip install judgeval[langchain]`"
    ) from e


class JudgevalCallbackHandler(BaseCallbackHandler): ...
