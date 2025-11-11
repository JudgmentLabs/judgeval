from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Dict, Optional, List, Any
from judgeval.tracer.keys import InternalAttributeKeys
import uuid
from judgeval.exceptions import JudgmentRuntimeError

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


@contextmanager
def sync_span_context(
    tracer: Tracer,
    name: str,
    span_attributes: Optional[Dict[str, str]] = None,
    disable_partial_emit: bool = False,
    end_on_exit: bool = False,
):
    if span_attributes is None:
        span_attributes = {}

    with tracer.get_tracer().start_as_current_span(
        name=name,
        attributes=span_attributes,
        end_on_exit=end_on_exit,
    ) as span:
        if disable_partial_emit:
            tracer.judgment_processor.set_internal_attribute(
                span_context=span.get_span_context(),
                key=InternalAttributeKeys.DISABLE_PARTIAL_EMIT,
                value=True,
            )
        yield span


@asynccontextmanager
async def async_span_context(
    tracer: Tracer,
    name: str,
    span_attributes: Optional[Dict[str, str]] = None,
    disable_partial_emit: bool = False,
    end_on_exit: bool = False,
):
    if span_attributes is None:
        span_attributes = {}

    with tracer.get_tracer().start_as_current_span(
        name=name,
        attributes=span_attributes,
        end_on_exit=end_on_exit,
    ) as span:
        if disable_partial_emit:
            tracer.judgment_processor.set_internal_attribute(
                span_context=span.get_span_context(),
                key=InternalAttributeKeys.DISABLE_PARTIAL_EMIT,
                value=True,
            )
        yield span


def create_agent_context(
    tracer: Tracer,
    args: tuple,
    class_name: Optional[str] = None,
    identifier: Optional[str] = None,
    track_state: bool = False,
    track_attributes: Optional[List[str]] = None,
    field_mappings: Optional[Dict[str, str]] = None,
):
    """Create agent context and return token for cleanup"""
    agent_id = str(uuid.uuid4())
    agent_context: Dict[str, Any] = {"agent_id": agent_id}

    if class_name:
        agent_context["class_name"] = class_name
    else:
        agent_context["class_name"] = None

    agent_context["track_state"] = track_state
    agent_context["track_attributes"] = track_attributes or []
    agent_context["field_mappings"] = field_mappings or {}

    instance = args[0] if args else None
    agent_context["instance"] = instance

    if identifier:
        if not class_name or not instance or not isinstance(instance, object):
            raise JudgmentRuntimeError(
                "'identifier' is set but no class name or instance is available. 'identifier' can only be specified when using the agent() decorator on a class method."
            )
        if (
            instance
            and hasattr(instance, identifier)
            and not callable(getattr(instance, identifier))
        ):
            instance_name = str(getattr(instance, identifier))
            agent_context["instance_name"] = instance_name
        else:
            raise JudgmentRuntimeError(
                f"Attribute {identifier} does not exist for {class_name}. Check your agent() decorator."
            )
    else:
        agent_context["instance_name"] = None

    current_agent_context = tracer.get_current_agent_context().get()
    if current_agent_context and "agent_id" in current_agent_context:
        agent_context["parent_agent_id"] = current_agent_context["agent_id"]
    else:
        agent_context["parent_agent_id"] = None

    agent_context["is_agent_entry_point"] = True
    token = tracer.get_current_agent_context().set(agent_context)  # type: ignore
    return token


@contextmanager
def sync_agent_context(
    tracer: Tracer,
    args: tuple,
    class_name: Optional[str] = None,
    identifier: Optional[str] = None,
    track_state: bool = False,
    track_attributes: Optional[List[str]] = None,
    field_mappings: Optional[Dict[str, str]] = None,
):
    """Context manager for synchronous agent context"""
    token = create_agent_context(
        tracer=tracer,
        args=args,
        class_name=class_name,
        identifier=identifier,
        track_state=track_state,
        track_attributes=track_attributes,
        field_mappings=field_mappings,
    )
    try:
        yield
    finally:
        tracer.get_current_agent_context().reset(token)


@asynccontextmanager
async def async_agent_context(
    tracer: Tracer,
    args: tuple,
    class_name: Optional[str] = None,
    identifier: Optional[str] = None,
    track_state: bool = False,
    track_attributes: Optional[List[str]] = None,
    field_mappings: Optional[Dict[str, str]] = None,
):
    """Context manager for asynchronous agent context"""
    token = create_agent_context(
        tracer=tracer,
        args=args,
        class_name=class_name,
        identifier=identifier,
        track_state=track_state,
        track_attributes=track_attributes,
        field_mappings=field_mappings,
    )
    try:
        yield
    finally:
        tracer.get_current_agent_context().reset(token)
