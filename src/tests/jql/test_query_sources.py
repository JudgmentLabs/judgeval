from judgeval.jql import sessions, spans, traces
from judgeval.jql._generated_contract import QUERY_SOURCES


def test_generated_query_sources_cover_every_public_root() -> None:
    assert QUERY_SOURCES == (
        "traces",
        "spans",
        "sessions",
    )


def test_public_roots_emit_canonical_json() -> None:
    assert [
        traces().rows().to_json(),
        spans().rows().to_json(),
        sessions().rows().to_json(),
    ] == [
        {
            "op": "query",
            "source": "traces",
            "select": {"op": "rows"},
        },
        {
            "op": "query",
            "source": "spans",
            "select": {"op": "rows"},
        },
        {
            "op": "query",
            "source": "sessions",
            "select": {"op": "rows"},
        },
    ]
