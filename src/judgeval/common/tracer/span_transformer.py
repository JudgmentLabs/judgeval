from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from judgeval.data import TraceSpan
from judgeval.evaluation_run import EvaluationRun


class SpanTransformer:
    @staticmethod
    def _needs_json_serialization(value: Any) -> bool:
        if value is None:
            return False

        simple_types = (str, int, float, bool)
        if isinstance(value, simple_types):
            return False

        complex_types = (dict, list, tuple, set, BaseModel)
        if isinstance(value, complex_types):
            return True

        try:
            json.dumps(value)
            return False
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _safe_json_handle(obj: Any, serialize: bool = True) -> Any:
        if serialize:
            if obj is None:
                return None
            try:
                return json.dumps(obj, default=str)
            except Exception:
                return json.dumps(str(obj))
        else:
            if not isinstance(obj, str):
                return obj
            try:
                return json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                return obj

    @staticmethod
    def _format_timestamp(timestamp: Optional[Union[float, int, str]]) -> str:
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()

        if isinstance(timestamp, str):
            return timestamp

        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.isoformat()
        except (ValueError, OSError):
            return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def trace_span_to_otel_attributes(
        trace_span: TraceSpan, span_state: str = "completed"
    ) -> Dict[str, Any]:
        serialized_data = trace_span.model_dump()
        attributes: Dict[str, Any] = {}

        for field_name, value in serialized_data.items():
            if value is None:
                continue

            attr_name = f"judgment.{field_name}"

            if field_name == "created_at":
                attributes[attr_name] = SpanTransformer._format_timestamp(value)
            elif field_name == "expected_tools" and value:
                attributes[attr_name] = SpanTransformer._safe_json_handle(
                    [tool.model_dump() for tool in trace_span.expected_tools]
                )
            elif field_name == "usage" and value:
                attributes[attr_name] = SpanTransformer._safe_json_handle(
                    trace_span.usage.model_dump()
                )
            elif SpanTransformer._needs_json_serialization(value):
                attributes[attr_name] = SpanTransformer._safe_json_handle(value)
            else:
                attributes[attr_name] = value

        attributes["judgment.span_state"] = span_state
        if not attributes.get("judgment.span_type"):
            attributes["judgment.span_type"] = "span"

        return attributes

    @staticmethod
    def otel_attributes_to_judgment_data(attributes: Dict[str, Any]) -> Dict[str, Any]:
        judgment_data: Dict[str, Any] = {}

        for key, value in attributes.items():
            if not key.startswith("judgment."):
                continue

            field_name = key[9:]

            if isinstance(value, str):
                deserialized = SpanTransformer._safe_json_handle(value, serialize=False)
                judgment_data[field_name] = deserialized
            else:
                judgment_data[field_name] = value

        return judgment_data

    @staticmethod
    def otel_span_to_judgment_format(span: ReadableSpan) -> Dict[str, Any]:
        attributes = span.attributes or {}
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        duration = judgment_data.get("duration")
        if duration is None and span.end_time and span.start_time:
            duration = (span.end_time - span.start_time) / 1_000_000_000

        span_id = judgment_data.get("span_id") or str(uuid.uuid4())
        trace_id = judgment_data.get("trace_id") or str(uuid.uuid4())

        created_at = judgment_data.get("created_at")
        if not created_at:
            created_at = (
                span.start_time / 1_000_000_000 if span.start_time else time.time()
            )

        return {
            "type": "span",
            "data": {
                "span_id": span_id,
                "trace_id": trace_id,
                "function": span.name,
                "depth": judgment_data.get("depth", 0),
                "created_at": SpanTransformer._format_timestamp(created_at),
                "parent_span_id": judgment_data.get("parent_span_id"),
                "span_type": judgment_data.get("span_type", "span"),
                "inputs": judgment_data.get("inputs"),
                "error": judgment_data.get("error"),
                "output": judgment_data.get("output"),
                "usage": judgment_data.get("usage"),
                "duration": duration,
                "expected_tools": judgment_data.get("expected_tools"),
                "additional_metadata": judgment_data.get("additional_metadata"),
                "has_evaluation": judgment_data.get("has_evaluation", False),
                "agent_name": judgment_data.get("agent_name"),
                "state_before": judgment_data.get("state_before"),
                "state_after": judgment_data.get("state_after"),
                "update_id": judgment_data.get("update_id", 1),
                "span_state": judgment_data.get("span_state", "completed"),
                "queued_at": time.time(),
            },
        }

    @staticmethod
    def evaluation_run_to_otel_attributes(
        evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> Dict[str, Any]:
        attributes = {
            "judgment.evaluation_run": True,
            "judgment.associated_span_id": span_id,
            "judgment.span_data": SpanTransformer._safe_json_handle(
                span_data.model_dump()
            ),
        }

        eval_data = evaluation_run.model_dump()
        for key, value in eval_data.items():
            if value is None:
                continue

            attr_name = f"judgment.{key}"
            if SpanTransformer._needs_json_serialization(value):
                attributes[attr_name] = SpanTransformer._safe_json_handle(value)
            else:
                attributes[attr_name] = value

        return attributes

    @staticmethod
    def otel_span_to_evaluation_run_format(span: ReadableSpan) -> Dict[str, Any]:
        attributes = span.attributes or {}
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        associated_span_id = judgment_data.get("associated_span_id") or str(
            uuid.uuid4()
        )

        eval_run_data = {
            key: value
            for key, value in judgment_data.items()
            if key not in ["associated_span_id", "span_data", "evaluation_run"]
        }

        eval_run_data["associated_span_id"] = associated_span_id
        eval_run_data["span_data"] = judgment_data.get("span_data")
        eval_run_data["queued_at"] = time.time()

        return {
            "type": "evaluation_run",
            "data": eval_run_data,
        }
