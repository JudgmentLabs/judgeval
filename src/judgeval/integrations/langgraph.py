from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID
import time
import uuid
from contextvars import ContextVar
from judgeval.common.tracer import TraceClient, TraceEntry, Tracer, SpanType

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.documents import Document

class JudgevalCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.trace_client = tracer.get_current_trace() if tracer.get_current_trace() else None
        self.previous_spans = [] # stack of previous spans
        self.finished = False

        # Attributes for users to access
        self.previous_node = None
        self.executed_node_tools = []
        self.executed_nodes = []
        self.executed_tools = []

    def start_span(self, name: str, span_type: SpanType = "span"):
        start_time = time.time()
        
        # Record span entry
        self.trace_client.add_entry(TraceEntry(
            type="enter",
            function=name,
            depth=self.trace_client.tracer.depth,
            message=name,
            timestamp=start_time,
            span_type=span_type
        ))

        self.trace_client.tracer.depth += 1
        self.previous_spans.append(self.trace_client._current_span)
        self.trace_client._current_span = name
        self._start_time = start_time
    
    def end_span(self, name: str, span_type: SpanType = "span"):
        self.trace_client.tracer.depth -= 1
        duration = time.time() - self._start_time
        
        # Record span exit
        self.trace_client.add_entry(TraceEntry(
            type="exit",
            function=name,
            depth=self.trace_client.tracer.depth,
            message=f"{name}",
            timestamp=time.time(),
            duration=duration,
            span_type=span_type
        ))
        self.trace_client._current_span = self.previous_spans.pop()

        if self.trace_client.tracer.depth == 0:
            # Save the trace if we are the root, this is when users dont use any @observe decorators
            self.trace_client.save(empty_save=False, overwrite=True)
            self.trace_client._current_trace = None
                 
    def on_retriever_start(
        self,
        serialized: Optional[dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = "RETRIEVER_CALL"
        if serialized and "name" in serialized:
            name = f"RETRIEVER_{serialized['name'].upper()}"
        
        self.start_span(name, span_type="retriever")
        self.trace_client.record_input({
            'query': query,
            'tags': tags,
            'metadata': metadata,
            'kwargs': kwargs
        })

    def on_retriever_end(
        self, 
        documents: Sequence[Document], 
        *, 
        run_id: UUID, 
        parent_run_id: Optional[UUID] = None, 
        **kwargs: Any
    ) -> Any:
        # Process the retrieved documents into a format suitable for logging
        doc_summary = []
        for i, doc in enumerate(documents):
            # Extract key information from each document
            doc_data = {
                "index": i,
                "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "metadata": doc.metadata
            }
            doc_summary.append(doc_data)
        
        # Record the document data
        self.trace_client.record_output({
            "document_count": len(documents),
            "documents": doc_summary
        })
        
        # End the retriever span
        self.end_span(self.trace_client._current_span, span_type="retriever")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        # If the user doesnt use any @observe decorators, the first action in LangGraph workflows seems tohave this attribute, so we intialize our trace client here
        if kwargs.get('name') == 'LangGraph':
            if not self.trace_client:
                trace_id = str(uuid.uuid4())
                project = self.tracer.project_name
                trace = TraceClient(self.tracer, trace_id, trace_id, project_name=project, overwrite=False, rules=self.tracer.rules, enable_monitoring=self.tracer.enable_monitoring, enable_evaluations=self.tracer.enable_evaluations)
                self.trace_client = trace
                self.tracer._current_trace = trace # set the trace in the original tracer object
                # Only save empty trace for the root call
                self.trace_client.save(empty_save=True, overwrite=False)
            
            self.start_span("LangGraph", span_type="Main Function")
            
        metadata = kwargs.get("metadata", {})
        if node := metadata.get("langgraph_node"):
            if node != self.previous_node:
                # Track node execution
                self.trace_client.visited_nodes.append(node)
                self.trace_client.executed_node_tools.append(node)
                self.trace_client.record_input({
                    'args': inputs,
                    'kwargs': kwargs
                })
            self.previous_node = node
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:   
        if outputs == "__end__":
            self.finished = True
        if tags is not None and any("graph:step" in tag for tag in tags):
            self.trace_client.record_output(outputs)
            self.end_span(self.trace_client._current_span, span_type="node")

            if self.finished:
                self.end_span(self.trace_client._current_span, span_type="Main Function")
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Chain error: {error}")
        self.trace_client.record_output(error)
        self.end_span(self.trace_client._current_span, span_type="node")

    def on_tool_start(
        self,
        serialized: Optional[dict[str, Any]],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        name = serialized["name"]
        self.start_span(name, span_type="tool")
        if name:
            # Track tool execution
            self.trace_client.executed_tools.append(name)
            node_tool = f"{self.previous_node}:{name}" if self.previous_node else name
            self.trace_client.executed_node_tools.append(node_tool)
        self.trace_client.record_input({
            'args': input_str,
            'kwargs': kwargs
        })

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        self.trace_client.record_output(output)
        self.end_span(self.trace_client._current_span, span_type="tool")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Tool error: {error}")
        self.trace_client.record_output(error)
        self.end_span(self.trace_client._current_span, span_type="tool")

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Agent action: {action}")

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"Agent finish: {finish}")

    def on_llm_start(
        self,
        serialized: Optional[dict[str, Any]],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:    
        name = "LLM call"
        self.start_span(name, span_type="llm")
        self.trace_client.record_input({
            'args': prompts,
            'kwargs': kwargs
        })

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any):
        self.trace_client.record_output(response.generations[0][0].text)
        self.end_span(self.trace_client._current_span, span_type="llm")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"LLM error: {error}")
        self.trace_client.record_output(error)
        self.end_span(self.trace_client._current_span, span_type="llm")

    def on_chat_model_start(
        self,
        serialized: Optional[dict[str, Any]],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:

        if "openai" in serialized["id"]:
            name = f"OPENAI_API_CALL"
        elif "anthropic" in serialized["id"]:
            name = "ANTHROPIC_API_CALL"
        elif "together" in serialized["id"]:
            name = "TOGETHER_API_CALL"
        else:
            name = "LLM call"

        self.start_span(name, span_type="llm")
        self.trace_client.record_input({
            'args': str(messages),
            'kwargs': kwargs
        })

judgeval_callback_handler_var: ContextVar[Optional[JudgevalCallbackHandler]] = ContextVar(
    "judgeval_callback_handler", default=None
)

def set_global_handler(handler: JudgevalCallbackHandler):
    if not handler.tracer.enable_monitoring:
        return
    judgeval_callback_handler_var.set(handler)

def clear_global_handler():
    judgeval_callback_handler_var.set(None)

register_configure_hook(
    context_var=judgeval_callback_handler_var,
    inheritable=True,
)