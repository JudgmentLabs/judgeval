from typing import Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Span

from judgeval.tracer.keys import AttributeKeys


def deduplicate_spans(spans: Sequence[ReadableSpan]) -> Sequence[ReadableSpan]:
    spans_by_key: dict[tuple[int, int], ReadableSpan] = {}
    for span in spans:
        if span.attributes and span.context:
            update_id = span.attributes.get(AttributeKeys.JUDGMENT_UPDATE_ID)

            if not isinstance(update_id, int):
                continue

            key = (span.context.trace_id, span.context.span_id)
            if key not in spans_by_key:
                spans_by_key[key] = span
            else:
                existing_attrs = spans_by_key[key].attributes
                existing_update_id = (
                    existing_attrs.get(AttributeKeys.JUDGMENT_UPDATE_ID, 0)
                    if existing_attrs
                    else 0
                )
                if (
                    isinstance(existing_update_id, (int, float))
                    and update_id > existing_update_id
                ):
                    spans_by_key[key] = span

    return list(spans_by_key.values())
