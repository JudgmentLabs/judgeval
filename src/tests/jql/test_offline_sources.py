from judgeval.jql import offline_spans, offline_traces
from judgeval.jql._generated_contract import QUERY_SOURCES


def test_generated_query_sources_cover_every_public_root() -> None:
    assert QUERY_SOURCES == (
        "traces",
        "spans",
        "sessions",
        "offline_traces",
        "offline_spans",
    )


def test_offline_roots_emit_canonical_json() -> None:
    assert [
        offline_traces().rows().to_json(),
        offline_spans().rows().to_json(),
    ] == [
        {
            "op": "query",
            "source": "offline_traces",
            "select": {"op": "rows"},
        },
        {
            "op": "query",
            "source": "offline_spans",
            "select": {"op": "rows"},
        },
    ]
