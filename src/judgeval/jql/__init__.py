"""Pure-Python builders for Judgment Query Language (JQL).

The builders emit the canonical, project-free JSON IR. Tenant scope is supplied
by :class:`judgeval.Judgeval` when the query is sent to the Judgment API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from judgeval.jql._generated_contract import DiscoveryKind
from judgeval.jql._generated_transport import (
    JqlPresentationResponse,
    JqlQueryResponse,
)

JsonObject = Dict[str, Any]
Filter = Mapping[str, Any]
Expr = Mapping[str, Any]


def _compact(values: Mapping[str, Any]) -> JsonObject:
    return {key: deepcopy(value) for key, value in values.items() if value is not None}


def status(value: str) -> JsonObject:
    return {"op": "status", "value": value}


def name(value: str) -> JsonObject:
    return {"op": "name", "value": value}


def model(value: str) -> JsonObject:
    return {"op": "model", "value": value}


def cost(**bounds: float) -> JsonObject:
    return {"op": "cost", **bounds}


def duration(**bounds: float) -> JsonObject:
    return {"op": "duration", **bounds}


def judge(name: str, value: Any = None) -> JsonObject:
    return _compact({"op": "judge", "name": name, "value": value})


def judged(
    *,
    name: Optional[str] = None,
    value: Any = None,
    prompt: Optional[str] = None,
    type: Optional[str] = None,
    mode: Optional[str] = None,
) -> JsonObject:
    return _compact(
        {
            "op": "judged",
            "name": name,
            "value": value,
            "prompt": prompt,
            "type": type,
            "mode": mode,
        }
    )


def attr(key: str, value: Any = None, selector: Optional[str] = None) -> JsonObject:
    return _compact({"op": "attr", "key": key, "value": value, "selector": selector})


def grep(field: str, value: str) -> JsonObject:
    return {"op": "grep", "field": field, "value": value}


def rg(field: str, pattern: str, *, ignore_case: Optional[bool] = None) -> JsonObject:
    return _compact(
        {"op": "rg", "field": field, "pattern": pattern, "ignore_case": ignore_case}
    )


def tokens(field: str, words: str) -> JsonObject:
    return {"op": "tokens", "field": field, "words": words}


def _compare(op: str, field: str, value: Any) -> JsonObject:
    return {"op": op, "field": field, "value": deepcopy(value)}


def eq(field: str, value: Any) -> JsonObject:
    return _compare("eq", field, value)


def ne(field: str, value: Any) -> JsonObject:
    return _compare("ne", field, value)


def gt(field: str, value: Any) -> JsonObject:
    return _compare("gt", field, value)


def gte(field: str, value: Any) -> JsonObject:
    return _compare("gte", field, value)


def lt(field: str, value: Any) -> JsonObject:
    return _compare("lt", field, value)


def lte(field: str, value: Any) -> JsonObject:
    return _compare("lte", field, value)


def cited_by(judge: str, value: Any = None) -> JsonObject:
    return _compact({"op": "cited_by", "judge": judge, "value": value})


def all_(first: Filter, *rest: Filter) -> JsonObject:
    return {"op": "all", "filters": [deepcopy(dict(item)) for item in (first, *rest)]}


def any_(first: Filter, *rest: Filter) -> JsonObject:
    return {"op": "any", "filters": [deepcopy(dict(item)) for item in (first, *rest)]}


def not_(filter: Filter) -> JsonObject:
    return {"op": "not", "filter": deepcopy(dict(filter))}


def _quantify(op: str, filter: Filter) -> JsonObject:
    return {"op": op, "filter": deepcopy(dict(filter))}


def any_span(filter: Filter) -> JsonObject:
    return _quantify("any_span", filter)


def every_span(filter: Filter) -> JsonObject:
    return _quantify("every_span", filter)


def no_span(filter: Filter) -> JsonObject:
    return _quantify("no_span", filter)


def any_trace(filter: Filter) -> JsonObject:
    return _quantify("any_trace", filter)


def every_trace(filter: Filter) -> JsonObject:
    return _quantify("every_trace", filter)


def no_trace(filter: Filter) -> JsonObject:
    return _quantify("no_trace", filter)


def descendant_of(filter: Filter, depth: Optional[int] = 1) -> JsonObject:
    return {"op": "descendant_of", "filter": deepcopy(dict(filter)), "depth": depth}


def ancestor_of(filter: Filter, depth: Optional[int] = 1) -> JsonObject:
    return {"op": "ancestor_of", "filter": deepcopy(dict(filter)), "depth": depth}


def _over(
    op: str,
    agg: Mapping[str, Any],
    cmp: Literal["eq", "ne", "gt", "gte", "lt", "lte"],
    value: float,
    where: Optional[Filter] = None,
) -> JsonObject:
    return _compact(
        {
            "op": op,
            "agg": dict(agg),
            "cmp": cmp,
            "value": value,
            "where": dict(where) if where is not None else None,
        }
    )


def over_spans(
    agg: Mapping[str, Any], cmp: str, value: float, where: Optional[Filter] = None
) -> JsonObject:
    return _over("over_spans", agg, cmp, value, where)  # type: ignore[arg-type]


def over_traces(
    agg: Mapping[str, Any], cmp: str, value: float, where: Optional[Filter] = None
) -> JsonObject:
    return _over("over_traces", agg, cmp, value, where)  # type: ignore[arg-type]


def over_scores(agg: Mapping[str, Any], cmp: str, value: float) -> JsonObject:
    return _over("over_scores", agg, cmp, value)  # type: ignore[arg-type]


def at_least(
    k: int, of: Literal["spans", "traces"], where: Optional[Filter] = None
) -> JsonObject:
    return _compact(
        {"op": "at_least", "k": k, "of": of, "where": dict(where) if where else None}
    )


def agg_expr(
    func: str,
    field: Optional[str] = None,
    *,
    q: Optional[float] = None,
    per: Optional[str] = None,
    where: Optional[Filter] = None,
) -> JsonObject:
    return _compact(
        {
            "op": "agg_expr",
            "func": func,
            "field": field,
            "q": q,
            "per": per,
            "where": dict(where) if where else None,
        }
    )


def arith(fn: Literal["div", "mul", "add", "sub"], left: Any, right: Any) -> JsonObject:
    return {"op": "arith", "fn": fn, "left": deepcopy(left), "right": deepcopy(right)}


def bucket(field: str, every: str) -> JsonObject:
    return {"op": "bucket", "field": field, "every": every}


def col(name: str) -> JsonObject:
    return {"op": "col", "name": name}


@dataclass(frozen=True)
class QueryBuilder:
    _spec: JsonObject

    def where(self, filter: Filter) -> "QueryBuilder":
        incoming = dict(filter)
        current = self._spec.get("filter")
        combined = all_(current, incoming) if current is not None else incoming
        return self._replace(filter=combined)

    def last(self, window: str) -> "QueryBuilder":
        return self._replace(time={"last": window})

    def since(self, since: str) -> "QueryBuilder":
        return self._replace(time={"since": since})

    def between(self, start: str, end: str) -> "QueryBuilder":
        return self._replace(time={"between": [start, end]})

    def pipe(self) -> "PipelineBuilder":
        spec = self.to_json()
        spec.pop("select", None)
        spec.pop("pipe", None)
        return PipelineBuilder(spec, ())

    def rows(
        self, *, fields: Optional[Sequence[str]] = None, limit: Optional[int] = None
    ) -> "QueryBuilder":
        return self._select(
            _compact(
                {
                    "op": "rows",
                    "fields": list(fields) if fields else None,
                    "limit": limit,
                }
            )
        )

    def ids(self) -> "QueryBuilder":
        return self._select({"op": "ids"})

    def count(self, by: Optional[str] = None) -> "QueryBuilder":
        return self._select(_compact({"op": "count", "by": by}))

    def recent(self, n: int) -> "QueryBuilder":
        return self._select({"op": "recent", "n": n})

    def top(self, n: int, by: str) -> "QueryBuilder":
        return self._select({"op": "top", "n": n, "by": by})

    def ranked(self, **options: Any) -> "QueryBuilder":
        return self._select({"op": "ranked", **deepcopy(options)})

    def agg(self, func: str, field: str, q: Optional[float] = None) -> "QueryBuilder":
        return self._select(
            _compact({"op": "agg", "func": func, "field": field, "q": q})
        )

    def trend(
        self, *, metric: Optional[str] = None, bucket: Optional[str] = None
    ) -> "QueryBuilder":
        return self._select(
            _compact({"op": "trend", "metric": metric, "bucket": bucket})
        )

    def chart(self, **options: Any) -> JsonObject:
        return {**deepcopy(options), "op": "chart", "query": self.to_json()}

    def table(self, **options: Any) -> JsonObject:
        return {**deepcopy(options), "op": "table", "query": self.to_json()}

    def to_json(self) -> JsonObject:
        return deepcopy(self._spec)

    def _select(self, select: JsonObject) -> "QueryBuilder":
        return self._replace(select=select)

    def _replace(self, **values: Any) -> "QueryBuilder":
        return QueryBuilder({**self.to_json(), **deepcopy(values)})


@dataclass(frozen=True)
class PipelineBuilder:
    _spec: JsonObject
    _stages: tuple[JsonObject, ...]

    def where(self, filter: Filter) -> "PipelineBuilder":
        return self._append({"op": "where", "filter": dict(filter)})

    def pick(self, **options: Any) -> "PipelineBuilder":
        return self._append({"op": "pick", **deepcopy(options)})

    def derive(self, cols: Mapping[str, Any]) -> "PipelineBuilder":
        return self._append({"op": "derive", "cols": deepcopy(dict(cols))})

    def summarize(
        self, aggs: Mapping[str, Any], *, by: Any = None
    ) -> "PipelineBuilder":
        return self._append(_compact({"op": "summarize", "by": by, "aggs": dict(aggs)}))

    def sort(self, by: str) -> "PipelineBuilder":
        return self._append({"op": "sort", "by": by})

    def take(self, n: int, offset: Optional[int] = None) -> "PipelineBuilder":
        return self._append(_compact({"op": "take", "n": n, "offset": offset}))

    def chart(self, **options: Any) -> JsonObject:
        return {**deepcopy(options), "op": "chart", "query": self.to_json()}

    def table(self, **options: Any) -> JsonObject:
        return {**deepcopy(options), "op": "table", "query": self.to_json()}

    def to_json(self) -> JsonObject:
        return {**deepcopy(self._spec), "pipe": deepcopy(list(self._stages))}

    def _append(self, stage: JsonObject) -> "PipelineBuilder":
        return PipelineBuilder(self._spec, (*self._stages, deepcopy(stage)))


def _query(
    source: Literal["traces", "spans", "sessions"], filter: Optional[Filter]
) -> QueryBuilder:
    return QueryBuilder(
        _compact(
            {
                "op": "query",
                "source": source,
                "filter": dict(filter) if filter else None,
            }
        )
    )


def traces(filter: Optional[Filter] = None) -> QueryBuilder:
    return _query("traces", filter)


def spans(filter: Optional[Filter] = None) -> QueryBuilder:
    return _query("spans", filter)


def sessions(filter: Optional[Filter] = None) -> QueryBuilder:
    return _query("sessions", filter)


def discovery(kind: DiscoveryKind, **options: Any) -> JsonObject:
    return _compact({"op": "discovery", "kind": kind, **options})


QueryInput = Union[JsonObject, QueryBuilder, PipelineBuilder]


def to_json(query: QueryInput) -> JsonObject:
    return (
        query.to_json()
        if isinstance(query, (QueryBuilder, PipelineBuilder))
        else deepcopy(query)
    )


# Python-safe names are primary; aliases retain the canonical combinator vocabulary.
all = all_
any = any_

__all__ = [
    "DiscoveryKind",
    "Expr",
    "Filter",
    "JqlPresentationResponse",
    "JqlQueryResponse",
    "JsonObject",
    "PipelineBuilder",
    "QueryBuilder",
    "QueryInput",
    "agg_expr",
    "all",
    "all_",
    "ancestor_of",
    "any",
    "any_",
    "any_span",
    "any_trace",
    "arith",
    "at_least",
    "attr",
    "bucket",
    "cited_by",
    "col",
    "cost",
    "descendant_of",
    "discovery",
    "duration",
    "eq",
    "every_span",
    "every_trace",
    "grep",
    "gt",
    "gte",
    "judge",
    "judged",
    "lt",
    "lte",
    "model",
    "name",
    "ne",
    "no_span",
    "no_trace",
    "not_",
    "over_scores",
    "over_spans",
    "over_traces",
    "rg",
    "sessions",
    "spans",
    "status",
    "to_json",
    "tokens",
    "traces",
]
