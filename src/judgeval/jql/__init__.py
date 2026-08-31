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

from judgeval.jql._generated_contract import DiscoveryKind, QuerySource, SUPPORTED_OPS
from judgeval.jql._generated_roots import (
    offline_spans,
    offline_traces,
    sessions,
    spans,
    traces,
)
from judgeval.jql._generated_transport import (
    JqlPresentationResponse,
    JqlQueryResponse,
)

JsonObject = Dict[str, Any]
Filter = Mapping[str, Any]
Expr = Mapping[str, Any]
Cmp = Literal["eq", "ne", "gt", "gte", "lt", "lte"]
ChartType = Literal["bar", "line", "area", "pie"]
PresentationField = Mapping[str, Any]
# HierarchyFilter.depth is pinned to exactly 1 (or null) in the public IR.
HierarchyDepth = Literal[1]


def _compact(values: Mapping[str, Any]) -> JsonObject:
    return {key: deepcopy(value) for key, value in values.items() if value is not None}


def _node(operation: str, values: Optional[Mapping[str, Any]] = None) -> JsonObject:
    if operation not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported JQL operation: {operation}")
    spec = deepcopy(dict(values or {}))
    supplied = spec.pop("op", operation)
    if supplied != operation:
        raise ValueError(
            f"Conflicting op {supplied!r} for JQL node {operation!r}; "
            "the operation discriminator cannot be overridden"
        )
    spec["op"] = operation
    return spec


def status(value: str) -> JsonObject:
    return _node("status", {"value": value})


def name(value: str) -> JsonObject:
    return _node("name", {"value": value})


def model(value: str) -> JsonObject:
    return _node("model", {"value": value})


def cost(**bounds: float) -> JsonObject:
    return _node("cost", bounds)


def duration(**bounds: float) -> JsonObject:
    return _node("duration", bounds)


def judge(name: str, value: Any = None) -> JsonObject:
    return _compact(_node("judge", {"name": name, "value": value}))


def judged(
    *,
    name: Optional[str] = None,
    value: Any = None,
    prompt: Optional[str] = None,
    type: Optional[str] = None,
    mode: Optional[str] = None,
) -> JsonObject:
    return _compact(
        _node(
            "judged",
            {
                "name": name,
                "value": value,
                "prompt": prompt,
                "type": type,
                "mode": mode,
            },
        )
    )


def attr(key: str, value: Any = None, selector: Optional[str] = None) -> JsonObject:
    return _compact(_node("attr", {"key": key, "value": value, "selector": selector}))


def grep(field: str, value: str) -> JsonObject:
    return _node("grep", {"field": field, "value": value})


def rg(field: str, pattern: str, *, ignore_case: Optional[bool] = None) -> JsonObject:
    return _compact(
        _node("rg", {"field": field, "pattern": pattern, "ignore_case": ignore_case})
    )


def tokens(field: str, words: str) -> JsonObject:
    return _node("tokens", {"field": field, "words": words})


def _compare(op: str, field: str, value: Any) -> JsonObject:
    return _node(op, {"field": field, "value": value})


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
    return _compact(_node("cited_by", {"judge": judge, "value": value}))


def all_(first: Filter, *rest: Filter) -> JsonObject:
    return _node("all", {"filters": [deepcopy(dict(item)) for item in (first, *rest)]})


def any_(first: Filter, *rest: Filter) -> JsonObject:
    return _node("any", {"filters": [deepcopy(dict(item)) for item in (first, *rest)]})


def not_(filter: Filter) -> JsonObject:
    return _node("not", {"filter": filter})


def _quantify(op: str, filter: Filter) -> JsonObject:
    return _node(op, {"filter": filter})


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


def descendant_of(filter: Filter, depth: Optional[HierarchyDepth] = 1) -> JsonObject:
    return _node("descendant_of", {"filter": filter, "depth": depth})


def ancestor_of(filter: Filter, depth: Optional[HierarchyDepth] = 1) -> JsonObject:
    return _node("ancestor_of", {"filter": filter, "depth": depth})


def _over(
    op: str,
    agg: Mapping[str, Any],
    cmp: Cmp,
    value: float,
    where: Optional[Filter] = None,
) -> JsonObject:
    return _compact(
        _node(
            op,
            {
                "agg": dict(agg),
                "cmp": cmp,
                "value": value,
                "where": dict(where) if where is not None else None,
            },
        )
    )


def over_spans(
    agg: Mapping[str, Any], cmp: Cmp, value: float, where: Optional[Filter] = None
) -> JsonObject:
    return _over("over_spans", agg, cmp, value, where)


def over_traces(
    agg: Mapping[str, Any], cmp: Cmp, value: float, where: Optional[Filter] = None
) -> JsonObject:
    return _over("over_traces", agg, cmp, value, where)


def over_scores(agg: Mapping[str, Any], cmp: Cmp, value: float) -> JsonObject:
    return _over("over_scores", agg, cmp, value)


def at_least(
    k: int, of: Literal["spans", "traces"], where: Optional[Filter] = None
) -> JsonObject:
    return _compact(
        _node(
            "at_least",
            {
                "k": k,
                "of": of,
                "where": dict(where) if where is not None else None,
            },
        )
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
        _node(
            "agg_expr",
            {
                "func": func,
                "field": field,
                "q": q,
                "per": per,
                "where": dict(where) if where is not None else None,
            },
        )
    )


def arith(fn: Literal["div", "mul", "add", "sub"], left: Any, right: Any) -> JsonObject:
    return _node("arith", {"fn": fn, "left": left, "right": right})


def bucket(field: str, every: str) -> JsonObject:
    return _node("bucket", {"field": field, "every": every})


def col(name: str) -> JsonObject:
    return _node("col", {"name": name})


def _chart(
    query: JsonObject,
    *,
    chart_type: ChartType,
    title: str,
    x_axis: PresentationField,
    y_axis: PresentationField,
    series_by: Optional[PresentationField] = None,
    description: Optional[str] = None,
) -> JsonObject:
    return _compact(
        _node(
            "chart",
            {
                "chart_type": chart_type,
                "title": title,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "series_by": series_by,
                "description": description,
                "query": query,
            },
        )
    )


def _table(
    query: JsonObject,
    *,
    title: str,
    columns: Sequence[PresentationField],
    description: Optional[str] = None,
) -> JsonObject:
    return _compact(
        _node(
            "table",
            {
                "title": title,
                "columns": list(columns),
                "description": description,
                "query": query,
            },
        )
    )


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
        if "select" in spec:
            raise ValueError(
                "pipe() cannot follow a select; call pipe() before "
                "rows()/ids()/count()/recent()/top()/ranked()/agg()/trend()"
            )
        spec.pop("pipe", None)
        return PipelineBuilder(spec, ())

    def rows(
        self, *, fields: Optional[Sequence[str]] = None, limit: Optional[int] = None
    ) -> "QueryBuilder":
        return self._select(
            _compact(
                _node(
                    "rows",
                    {
                        "fields": list(fields) if fields else None,
                        "limit": limit,
                    },
                )
            )
        )

    def ids(self) -> "QueryBuilder":
        return self._select(_node("ids"))

    def count(self, by: Optional[str] = None) -> "QueryBuilder":
        return self._select(_compact(_node("count", {"by": by})))

    def recent(self, n: int) -> "QueryBuilder":
        return self._select(_node("recent", {"n": n}))

    def top(self, n: int, by: str) -> "QueryBuilder":
        return self._select(_node("top", {"n": n, "by": by}))

    def ranked(
        self,
        *,
        by: Optional[str] = None,
        pick: Optional[Union[int, Sequence[int]]] = None,
        within: Optional[str] = None,
    ) -> "QueryBuilder":
        return self._select(
            _compact(
                _node(
                    "ranked",
                    {
                        "by": by,
                        "pick": pick
                        if pick is None or isinstance(pick, int)
                        else list(pick),
                        "within": within,
                    },
                )
            )
        )

    def agg(self, func: str, field: str, q: Optional[float] = None) -> "QueryBuilder":
        return self._select(
            _compact(_node("agg", {"func": func, "field": field, "q": q}))
        )

    def trend(
        self, *, metric: Optional[str] = None, bucket: Optional[str] = None
    ) -> "QueryBuilder":
        return self._select(
            _compact(_node("trend", {"metric": metric, "bucket": bucket}))
        )

    def chart(
        self,
        *,
        chart_type: ChartType,
        title: str,
        x_axis: PresentationField,
        y_axis: PresentationField,
        series_by: Optional[PresentationField] = None,
        description: Optional[str] = None,
    ) -> JsonObject:
        return _chart(
            self.to_json(),
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            series_by=series_by,
            description=description,
        )

    def table(
        self,
        *,
        title: str,
        columns: Sequence[PresentationField],
        description: Optional[str] = None,
    ) -> JsonObject:
        return _table(
            self.to_json(),
            title=title,
            columns=columns,
            description=description,
        )

    def to_json(self) -> JsonObject:
        return deepcopy(self._spec)

    def _select(self, select: JsonObject) -> "QueryBuilder":
        existing = self._spec.get("select")
        if existing is not None:
            raise ValueError(
                f"select is already set to {existing['op']!r}; a query has exactly "
                f"one select, so start a new query instead of adding {select['op']!r}"
            )
        return self._replace(select=select)

    def _replace(self, **values: Any) -> "QueryBuilder":
        return QueryBuilder({**self.to_json(), **deepcopy(values)})


@dataclass(frozen=True)
class PipelineBuilder:
    _spec: JsonObject
    _stages: tuple[JsonObject, ...]

    def where(self, filter: Filter) -> "PipelineBuilder":
        return self._append(_node("where", {"filter": filter}))

    def pick(
        self,
        *,
        by: Optional[str] = None,
        n: Optional[int] = None,
        per: Optional[str] = None,
        reverse: Optional[bool] = None,
    ) -> "PipelineBuilder":
        return self._append(
            _compact(_node("pick", {"by": by, "n": n, "per": per, "reverse": reverse}))
        )

    def derive(self, cols: Mapping[str, Any]) -> "PipelineBuilder":
        return self._append(_node("derive", {"cols": cols}))

    def summarize(
        self, aggs: Mapping[str, Any], *, by: Any = None
    ) -> "PipelineBuilder":
        return self._append(
            _compact(_node("summarize", {"by": by, "aggs": dict(aggs)}))
        )

    def sort(self, by: str) -> "PipelineBuilder":
        return self._append(_node("sort", {"by": by}))

    def take(self, n: int, offset: Optional[int] = None) -> "PipelineBuilder":
        return self._append(_compact(_node("take", {"n": n, "offset": offset})))

    def chart(
        self,
        *,
        chart_type: ChartType,
        title: str,
        x_axis: PresentationField,
        y_axis: PresentationField,
        series_by: Optional[PresentationField] = None,
        description: Optional[str] = None,
    ) -> JsonObject:
        return _chart(
            self.to_json(),
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            series_by=series_by,
            description=description,
        )

    def table(
        self,
        *,
        title: str,
        columns: Sequence[PresentationField],
        description: Optional[str] = None,
    ) -> JsonObject:
        return _table(
            self.to_json(),
            title=title,
            columns=columns,
            description=description,
        )

    def to_json(self) -> JsonObject:
        spec = deepcopy(self._spec)
        if self._stages:
            spec["pipe"] = deepcopy(list(self._stages))
        else:
            spec.pop("pipe", None)
        return spec

    def _append(self, stage: JsonObject) -> "PipelineBuilder":
        return PipelineBuilder(self._spec, (*self._stages, deepcopy(stage)))


def _query(source: QuerySource, filter: Optional[Filter]) -> QueryBuilder:
    return QueryBuilder(
        _compact(
            _node(
                "query",
                {
                    "source": source,
                    "filter": dict(filter) if filter is not None else None,
                },
            )
        )
    )


def discovery(kind: DiscoveryKind, **options: Any) -> JsonObject:
    return _compact(_node("discovery", {**deepcopy(options), "kind": kind}))


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
    "ChartType",
    "Cmp",
    "Expr",
    "Filter",
    "HierarchyDepth",
    "JqlPresentationResponse",
    "JqlQueryResponse",
    "JsonObject",
    "PipelineBuilder",
    "PresentationField",
    "QueryBuilder",
    "QueryInput",
    "QuerySource",
    "agg_expr",
    "all_",
    "ancestor_of",
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
    "offline_spans",
    "offline_traces",
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
