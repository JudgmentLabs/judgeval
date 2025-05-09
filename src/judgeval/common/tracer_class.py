import functools
import inspect
import asyncio
import uuid
from dataclasses import dataclass
from judgeval.tracer import Tracer, TraceClient

@dataclass
class TraceConfig:
    project_name: str
    enable_monitoring: bool = False
    enable_evaluations: bool = False
    # …any other flags…

class JudgevalTracer:
    # Default config; subclasses will override these
    _trace_config: TraceConfig = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cfg = getattr(cls, "_trace_config", None)
        if not isinstance(cfg, TraceConfig):
            raise TypeError(
                f"{cls.__name__} must define a _trace_config = TraceConfig(...)"
            )

        cls._tracer = Tracer(
            project_name=cfg.project_name,
            enable_monitoring=cfg.enable_monitoring,
            enable_evaluations=cfg.enable_evaluations,
        )

        # 3) Walk through the subclass namespace and wrap methods
        for name, member in list(cls.__dict__.items()):
            if (inspect.isfunction(member) or inspect.iscoroutinefunction(member)) and not name.startswith('_'):
                wrapped = cls._make_wrapper(member, cls._tracer, cfg)
                setattr(cls, name, wrapped)

    @staticmethod
    def _make_wrapper(func, tracer, cfg: TraceConfig):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):

                # Get current trace from context
                current_trace = tracer.get_current_trace()
                
                # If there's no current trace, create a root trace
                if not current_trace:
                    trace_id = str(uuid.uuid4())
                    
                    # Create a new trace client to serve as the root
                    current_trace = TraceClient(
                        tracer,
                        trace_id,
                        func.__name__,
                        project_name=cfg.project_name,
                        enable_monitoring=cfg.enable_monitoring,
                        enable_evaluations=cfg.enable_evaluations
                    )
                    
                    trace_token = tracer.set_current_trace(current_trace)
                    
                    try:
                        # Use span for the function execution within the root trace
                        # This sets the current_span_var
                        print("func.__qualname__: ", func.__qualname__)
                        with current_trace.span(func.__qualname__, span_type='span') as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })
                            
                            
                            # Execute function
                            result = await func(*args, **kwargs)
                            
                            # Record output
                            span.record_output(result)
                            
                        # Save the completed trace
                        current_trace.save()
                        return result
                    finally:
                        tracer.reset_current_trace(trace_token)
                else:
                    try:
                        with current_trace.span(func.__qualname__, span_type='span') as span:
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })

                            # Execute function
                            result = await func(*args, **kwargs)

                            span.record_output(result)
                        
                        return result
                    finally:
                        pass
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                current_trace = tracer.get_current_trace()
                
                # If there's no current trace, create a root trace
                if not current_trace:
                    trace_id = str(uuid.uuid4())

                    # Create a new trace client to serve as the root
                    current_trace = TraceClient(
                        tracer,
                        trace_id,
                        func.__name__,
                        project_name=cfg.project_name,
                        enable_monitoring=cfg.enable_monitoring,
                        enable_evaluations=cfg.enable_evaluations
                    )
                    
                    trace_token = tracer.set_current_trace(current_trace)
                    
                    try:

                        with current_trace.span(func.__qualname__, span_type='span') as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })
                            # Execute function
                            result = func(*args, **kwargs)
                            
                            # Record output
                            span.record_output(result)
                            
                        # Save the completed trace
                        current_trace.save()
                        return result
                    finally:
                        tracer.reset_current_trace(trace_token)
                else:
                    try:
                        with current_trace.span(func.__qualname__, span_type='span') as span: # MODIFIED: Use span_name directly
                            # Record inputs
                            span.record_input({
                                'args': str(args),
                                'kwargs': kwargs
                            })

                            # Execute function
                            result = func(*args, **kwargs)
                            
                            # Record output
                            span.record_output(result)
                        
                        return result
                    finally:
                        pass
                
            return wrapper
