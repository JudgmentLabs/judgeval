from __future__ import annotations

import time

import pytest

from judgeval.integrations.mcp._scrub import (
    argument_values,
    scrub_error_message,
    scrub_identifiers,
)


class TestArgumentValues:
    def test_reads_nested_text_values_longest_first(self):
        values = argument_values(
            {
                "name": "lookup",
                "arguments": {
                    "order_id": "order-1234",
                    "filters": [{"owner": "ana-lee"}, "tag-9"],
                    "limit": 5000,
                    "draft": True,
                },
            }
        )
        assert values == ["order-1234", "ana-lee", "tag-9"]

    def test_includes_quoted_forms(self):
        values = argument_values({"arguments": {"note": 'it\'s "x"\nok'}})
        assert 'it\'s "x"\nok' in values
        # As Python's repr and JSON would print it inside an error.
        assert 'it\\\'s "x"\\nok' in values
        assert 'it\'s \\"x\\"\\nok' in values

    def test_includes_a_resource_uri(self):
        assert argument_values({"uri": "orders://o-1?token=t0k3n"}) == [
            "orders://o-1?token=t0k3n"
        ]

    def test_skips_short_values_and_odd_shapes(self):
        assert argument_values({"arguments": {"a": "abc", "b": None}}) == []
        assert argument_values({"arguments": "not a mapping"}) == ["not a mapping"]
        assert argument_values({}) == []

    def test_is_bounded(self):
        wide = {f"k{index}": f"value-{index}" for index in range(5_000)}
        deep: object = "deep-value"
        for _ in range(5_000):
            deep = [deep]
        started = time.perf_counter()
        assert len(argument_values({"arguments": wide})) == 100
        argument_values({"arguments": {"deep": deep}})
        assert time.perf_counter() - started < 0.5


class TestScrubErrorMessage:
    def test_replaces_echoed_values_longest_first(self):
        values = argument_values(
            {"arguments": {"id": "secret-order-77", "prefix": "secret"}}
        )
        assert (
            scrub_error_message("order secret-order-77 not found (secret)", values)
            == "order [argument] not found ([argument])"
        )

    @pytest.mark.parametrize(
        "rejected",
        [
            "input_value='tok-123', input_type=str",
            # pydantic shortens a long input, so it no longer equals the argument.
            "input_value='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...aaaaaaaaaaaa', input_type=str",
            "input_value={'nested': 'x'}, input_type=dict",
        ],
    )
    def test_replaces_pydantic_inputs(self, rejected):
        message = f"1 validation error for addArguments\na\n  Input should be a valid integer [type=int_type, {rejected}]\n    For further information visit https://errors.pydantic.dev"
        scrubbed = scrub_error_message(message, [])
        assert (
            "[type=int_type, input_value=[argument], input_type=" in scrubbed
            and "For further information" in scrubbed
        )

    def test_unterminated_pydantic_input_is_replaced_to_the_line_end(self):
        assert (
            scrub_error_message("bad input_value='tok-123\nnext line", [])
            == "bad input_value=[argument]\nnext line"
        )

    def test_scrubs_identifiers_but_keeps_times(self):
        assert (
            scrub_error_message(
                "owner ana@example.com from 10.0.0.7 failed at 2026-11-08 10:30:00",
                [],
            )
            == "owner [email] from [ip] failed at 2026-11-08 10:30:00"
        )

    def test_long_echoed_values_are_removed_whole(self):
        document = "x" * 50_000
        message = f"could not parse {document} after 2 tries"
        scrubbed = scrub_error_message(
            message, argument_values({"arguments": {"d": document}})
        )
        assert scrubbed == "could not parse [argument] after 2 tries"

    def test_takes_bounded_time(self):
        values = argument_values(
            {"arguments": {f"k{i}": f"v-{i:04}" for i in range(200)}}
        )
        started = time.perf_counter()
        for message in ("a" * 2_000_000, "input_value=" * 200_000, "1 " * 1_000_000):
            scrub_error_message(message, values)
        assert time.perf_counter() - started < 1


def test_scrub_identifiers_is_bounded_in_length():
    assert len(scrub_identifiers("word " * 10_000)) == 4_000
