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
        self.previous_spans = [] # stack of previous spans
        self.created_trace = False

        # Attributes for users to access
        self.previous_node = None
        self.executed_node_tools = []
        self.executed_nodes = []
        self.executed_tools = []

    def start_span(self, name: str, span_type: SpanType = "span"):
        current_trace = self.tracer.get_current_trace()
        start_time = time.time()

        # Generate a unique ID for *this specific span invocation*
        span_id = str(uuid.uuid4())
        
        parent_span_id = current_trace.get_current_span()
        token = current_trace.set_current_span(span_id) # Set *this* span's ID as the current one
        
        current_depth = 0
        if parent_span_id and parent_span_id in current_trace._span_depths:
            current_depth = current_trace._span_depths[parent_span_id] + 1
        
        current_trace._span_depths[span_id] = current_depth # Store depth by span_id
        # Record span entry
        current_trace.add_entry(TraceEntry(
            type="enter",
            span_id=span_id,
            trace_id=current_trace.trace_id,
            parent_span_id=parent_span_id,
            function=name,
            depth=current_depth,
            message=name,
            created_at=start_time,
            span_type=span_type
        ))

        self.previous_spans.append(token)
        self._start_time = start_time
    
    def end_span(self, span_type: SpanType = "span"):
        current_trace = self.tracer.get_current_trace()
        duration = time.time() - self._start_time
        span_id = current_trace.get_current_span()
        exit_depth = current_trace._span_depths.get(span_id, 0) # Get depth using this span's ID
        
        # Record span exit
        current_trace.add_entry(TraceEntry(
            type="exit",
            span_id=span_id,
            trace_id=current_trace.trace_id,
            depth=exit_depth,
            created_at=time.time(),
            duration=duration,
            span_type=span_type
        ))
        current_trace.reset_current_span(self.previous_spans.pop())
        if exit_depth == 0:
            # Save the trace if we are the root, this is when users dont use any @observe decorators
            trace_id, trace_data = current_trace.save(overwrite=True)
            self._trace_id = trace_id
            current_trace = None
                 
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
        current_trace = self.tracer.get_current_trace()
        self.start_span(name, span_type="retriever")
        current_trace.record_input({
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
        current_trace = self.tracer.get_current_trace()
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
        current_trace.record_output({
            "document_count": len(documents),
            "documents": doc_summary
        })
        
        # End the retriever span
        self.end_span(span_type="retriever")

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
        current_trace = self.tracer.get_current_trace()
        if kwargs.get('name') == 'LangGraph':
            if not current_trace:
                self.created_trace = True
                trace_id = str(uuid.uuid4())
                project = self.tracer.project_name
                trace = TraceClient(self.tracer, trace_id, "Langgraph", project_name=project, overwrite=False, rules=self.tracer.rules, enable_monitoring=self.tracer.enable_monitoring, enable_evaluations=self.tracer.enable_evaluations)
                self.tracer.set_current_trace(trace)
                self.start_span("LangGraph", span_type="Main Function")
            
        node = metadata.get("langgraph_node")
        if node != None and node != self.previous_node:
             self.start_span(node, span_type="node")
             self.executed_node_tools.append(node)
             self.executed_nodes.append(node)
             current_trace.record_input({
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
        current_trace = self.tracer.get_current_trace()
        if tags is not None and any("graph:step" in tag for tag in tags):
            current_trace.record_output(outputs)
            self.end_span(span_type="node")

        if self.created_trace and (outputs == "__end__" or (not kwargs and not tags)):
            self.end_span(span_type="Main Function")
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        current_trace = self.tracer.get_current_trace()
        current_trace.record_output(error)
        self.end_span(span_type="node")

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
        current_trace = self.tracer.get_current_trace()
        if name:
            # Track tool execution
            current_trace.executed_tools.append(name)
            node_tool = f"{self.previous_node}:{name}" if self.previous_node else name
            current_trace.executed_node_tools.append(node_tool)
            current_trace.record_input({
                'args': input_str,
                'kwargs': kwargs
            })

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        current_trace = self.tracer.get_current_trace()
        current_trace.record_output(output)
        self.end_span(span_type="tool")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        current_trace = self.tracer.get_current_trace()
        current_trace.record_output(error)
        self.end_span(span_type="tool")

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        
        pass

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
        current_trace = self.tracer.get_current_trace()
        current_trace.record_input({
            'args': prompts,
            'kwargs': kwargs
        })

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any):
        current_trace = self.tracer.get_current_trace()
        current_trace.record_output(response.generations[0][0].text)
        self.end_span(span_type="llm")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        current_trace = self.tracer.get_current_trace()
        current_trace.record_output(error)
        self.end_span(span_type="llm")

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
        current_trace = self.tracer.get_current_trace()
        current_trace.record_input({
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