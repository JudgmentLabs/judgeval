"""Generated from DAL OpenAPI query sources; do not edit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from judgeval.jql import Filter, QueryBuilder


def traces(filter: Optional[Filter] = None) -> QueryBuilder:
    from judgeval.jql import _query

    return _query("traces", filter)


def spans(filter: Optional[Filter] = None) -> QueryBuilder:
    from judgeval.jql import _query

    return _query("spans", filter)


def sessions(filter: Optional[Filter] = None) -> QueryBuilder:
    from judgeval.jql import _query

    return _query("sessions", filter)


def offline_traces(filter: Optional[Filter] = None) -> QueryBuilder:
    from judgeval.jql import _query

    return _query("offline_traces", filter)


def offline_spans(filter: Optional[Filter] = None) -> QueryBuilder:
    from judgeval.jql import _query

    return _query("offline_spans", filter)


__all__ = [
    "traces",
    "spans",
    "sessions",
    "offline_traces",
    "offline_spans",
]
