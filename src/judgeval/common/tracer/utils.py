from collections import defaultdict
import functools
import inspect
from types import FrameType
from typing import Any, List, Set, TypeVar
import uuid

from judgeval.common.tracer.types import CurrentSpanExit


def new_id() -> str:
    """
    Generates a new unique span ID.
    """
    return str(uuid.uuid4())


def extract_inputs_from_entry_frame(frame: FrameType) -> dict[str, Any]:
    """
    Extracts the inputs from the entry frame.
    This is used to capture the inputs to the function being traced.
    """
    args, varargs, varkw, values = inspect.getargvalues(frame)
    inputs = {arg: values[arg] for arg in args if arg in values}
    if varargs:
        inputs[varargs] = values[varargs]
    if varkw:
        inputs[varkw] = values[varkw]
    return inputs


def get_span_exit_depth_map(spans: List[CurrentSpanExit]) -> dict[str, int]:
    """
    Builds a mapping of span_id -> depth number of spans.
    """

    # Build a tree: span_id -> list of child span_ids
    children_map: defaultdict[str, list[str]] = defaultdict(list)
    root_ids: Set[str] = set()

    for span in spans:
        if span.parent_span_id:
            children_map[span.span_id].append(span.span_id)
        else:
            root_ids.add(span.span_id)

    depth_map: defaultdict[str, int] = defaultdict(int)

    def traverse(span_id: str, depth: int = 0):
        depth_map[span_id] = depth
        for child_id in children_map.get(span_id, []):
            traverse(child_id, depth + 1)

    for root_id in root_ids:
        traverse(root_id)

    return depth_map


def fallback_encoder(obj: Any) -> str:
    """
    Custom JSON encoder fallback.
    Tries to use obj.__repr__(), then str(obj) if that fails or for a simpler string.
    You can choose which one you prefer or try them in sequence.
    """
    try:
        return repr(obj)
    except Exception:
        try:
            return str(obj)
        except Exception as e:
            return f"<Unserializable object of type {type(obj).__name__}: {e}>"
