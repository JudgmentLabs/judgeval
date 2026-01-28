"""
Edge case tests for the v1 SDK.

These tests verify that the SDK handles unusual but valid inputs gracefully,
including large payloads, special characters, deeply nested data, and
circular references.

Run with: pytest src/tests/reliability/test_edge_cases.py -v
"""

import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import patch

from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.tracer_factory import TracerFactory


@pytest.mark.reliability
class TestLargePayloads:
    """Test handling of large input/output payloads."""

    def test_large_string_input_output(self, tracer: Tracer):
        """
        Functions with 1MB+ string payloads should work correctly.
        """
        PAYLOAD_SIZE = 1_000_000  # 1MB
        large_string = "x" * PAYLOAD_SIZE

        @tracer.observe(span_type="function")
        def process_large_string(data: str) -> str:
            return data[:100] + "...truncated"

        result = process_large_string(large_string)

        assert result == "x" * 100 + "...truncated"

    def test_large_dict_payload(self, tracer: Tracer):
        """
        Functions with large dictionary payloads should work.
        """
        ITEMS = 10_000
        large_dict = {f"key_{i}": f"value_{i}" for i in range(ITEMS)}

        @tracer.observe(span_type="function")
        def process_large_dict(data: dict) -> int:
            return len(data)

        result = process_large_dict(large_dict)

        assert result == ITEMS

    def test_large_list_payload(self, tracer: Tracer):
        """
        Functions with large list payloads should work.
        """
        ITEMS = 100_000
        large_list = list(range(ITEMS))

        @tracer.observe(span_type="function")
        def process_large_list(data: list) -> int:
            return sum(data)

        result = process_large_list(large_list)

        assert result == sum(range(ITEMS))

    def test_large_nested_structure(self, tracer: Tracer):
        """
        Large nested data structures should be handled.
        """
        # Create a large nested structure
        large_nested = {
            "level1": {
                f"key_{i}": {"data": list(range(100)), "metadata": {"index": i}}
                for i in range(100)
            }
        }

        @tracer.observe(span_type="function")
        def process_nested(data: dict) -> int:
            count = 0
            for k, v in data["level1"].items():
                count += len(v["data"])
            return count

        result = process_nested(large_nested)

        assert result == 10_000  # 100 keys * 100 items each


@pytest.mark.reliability
class TestDeeplyNestedData:
    """Test handling of deeply nested data structures."""

    def test_deeply_nested_dict(self, tracer: Tracer):
        """
        100-level nested dicts should not crash.
        """
        DEPTH = 100

        # Create deeply nested dict
        nested = {"value": "leaf"}
        for i in range(DEPTH):
            nested = {"nested": nested, "level": i}

        @tracer.observe(span_type="function")
        def process_deep_dict(data: dict) -> str:
            # Navigate to the bottom
            current = data
            for _ in range(DEPTH):
                current = current["nested"]
            return current["value"]

        result = process_deep_dict(nested)

        assert result == "leaf"

    def test_deeply_nested_list(self, tracer: Tracer):
        """
        Deeply nested lists should not crash.
        """
        DEPTH = 100

        nested: Any = ["leaf"]
        for _ in range(DEPTH):
            nested = [nested]

        @tracer.observe(span_type="function")
        def process_deep_list(data: list) -> str:
            current = data
            for _ in range(DEPTH):
                current = current[0]
            return current[0]

        result = process_deep_list(nested)

        assert result == "leaf"

    def test_deeply_nested_mixed(self, tracer: Tracer):
        """
        Deeply nested mixed dict/list structures should work.
        """
        DEPTH = 50

        nested: Any = {"value": "leaf"}
        for i in range(DEPTH):
            if i % 2 == 0:
                nested = {"data": [nested]}
            else:
                nested = [{"item": nested}]

        @tracer.observe(span_type="function")
        def process_mixed(data) -> str:
            return "processed"

        result = process_mixed(nested)

        assert result == "processed"


@pytest.mark.reliability
class TestSpecialCharacters:
    """Test handling of special characters in strings."""

    def test_unicode_characters(self, tracer: Tracer):
        """
        Unicode characters should be handled correctly.
        """
        unicode_strings = [
            "Hello 世界",  # Chinese
            "مرحبا",  # Arabic
            "שלום",  # Hebrew
            "🎉🚀💻",  # Emoji
            "Ñoño",  # Spanish
            "Привет",  # Russian
            "日本語テスト",  # Japanese
        ]

        @tracer.observe(span_type="function")
        def process_unicode(text: str) -> str:
            return f"Processed: {text}"

        for text in unicode_strings:
            result = process_unicode(text)
            assert text in result

    def test_control_characters(self, tracer: Tracer):
        """
        Control characters should be handled gracefully.
        """
        control_strings = [
            "hello\x00world",  # Null byte
            "tab\there",  # Tab
            "new\nline",  # Newline
            "carriage\rreturn",  # Carriage return
            "bell\x07ring",  # Bell
            "backspace\x08test",  # Backspace
        ]

        @tracer.observe(span_type="function")
        def process_control(text: str) -> int:
            return len(text)

        for text in control_strings:
            result = process_control(text)
            assert result == len(text)

    def test_mixed_encoding_data(self, tracer: Tracer):
        """
        Mixed encoding scenarios should be handled.
        """
        mixed_data = {
            "utf8": "Hello 世界",
            "ascii": "plain text",
            "emoji": "🎉",
            "special": "line1\nline2\ttab",
        }

        @tracer.observe(span_type="function")
        def process_mixed(data: dict) -> dict:
            return {k: len(v) for k, v in data.items()}

        result = process_mixed(mixed_data)

        assert result["ascii"] == len("plain text")


@pytest.mark.reliability
class TestCircularReferences:
    """Test handling of circular references in data."""

    def test_circular_dict_reference(self, tracer: Tracer):
        """
        Circular dict references should not crash (may truncate).
        """
        circular: Dict[str, Any] = {"name": "root"}
        circular["self"] = circular  # Circular reference

        @tracer.observe(span_type="function")
        def process_circular(data: dict) -> str:
            return data["name"]

        # Should not crash
        result = process_circular(circular)

        assert result == "root"

    def test_circular_list_reference(self, tracer: Tracer):
        """
        Circular list references should not crash.
        """
        circular: List[Any] = [1, 2, 3]
        circular.append(circular)  # Circular reference

        @tracer.observe(span_type="function")
        def process_circular(data: list) -> int:
            return data[0]

        result = process_circular(circular)

        assert result == 1

    def test_mutual_circular_reference(self, tracer: Tracer):
        """
        Mutually circular references should not crash.
        """
        obj_a: Dict[str, Any] = {"name": "A"}
        obj_b: Dict[str, Any] = {"name": "B"}
        obj_a["partner"] = obj_b
        obj_b["partner"] = obj_a

        @tracer.observe(span_type="function")
        def process_mutual(a: dict, b: dict) -> str:
            return f"{a['name']}-{b['name']}"

        result = process_mutual(obj_a, obj_b)

        assert result == "A-B"


@pytest.mark.reliability
class TestBoundaryConditions:
    """Test boundary conditions and edge values."""

    def test_empty_inputs(self, tracer: Tracer):
        """
        Empty inputs should be handled correctly.
        """

        @tracer.observe(span_type="function")
        def process_empty(
            empty_str: str,
            empty_list: list,
            empty_dict: dict,
        ) -> str:
            return "processed"

        result = process_empty("", [], {})

        assert result == "processed"

    def test_none_values(self, tracer: Tracer):
        """
        None values should be handled correctly.
        """

        @tracer.observe(span_type="function")
        def process_none(data: Any) -> str:
            return "none" if data is None else "not none"

        assert process_none(None) == "none"
        assert process_none(0) == "not none"
        assert process_none("") == "not none"

    def test_extreme_numeric_values(self, tracer: Tracer):
        """
        Extreme numeric values should be handled.
        """

        @tracer.observe(span_type="function")
        def process_numbers(
            big_int: int,
            small_int: int,
            big_float: float,
            small_float: float,
        ) -> str:
            return "processed"

        result = process_numbers(
            big_int=sys.maxsize,
            small_int=-sys.maxsize,
            big_float=float("inf"),
            small_float=float("-inf"),
        )

        assert result == "processed"

    def test_special_float_values(self, tracer: Tracer):
        """
        Special float values (NaN, Inf) should be handled.
        """

        @tracer.observe(span_type="function")
        def process_special_floats(nan: float, pos_inf: float, neg_inf: float) -> str:
            return "processed"

        result = process_special_floats(
            nan=float("nan"),
            pos_inf=float("inf"),
            neg_inf=float("-inf"),
        )

        assert result == "processed"

    def test_boolean_edge_cases(self, tracer: Tracer):
        """
        Boolean-like values should be handled correctly.
        """

        @tracer.observe(span_type="function")
        def process_boolish(
            true_val: bool,
            false_val: bool,
            zero: int,
            empty_str: str,
        ) -> list:
            return [true_val, false_val, zero, empty_str]

        result = process_boolish(True, False, 0, "")

        assert result == [True, False, 0, ""]


@pytest.mark.reliability
class TestMultipleTracerInstances:
    """Test behavior with multiple tracer instances."""

    def test_multiple_tracers_dont_interfere(self, mock_client):
        """
        Multiple tracer instances should not interfere with each other.
        """
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            factory = TracerFactory(mock_client)

            tracer1 = factory.create(
                project_name="project-1",
                enable_monitoring=True,
                enable_evaluation=False,
                isolated=True,
            )

            tracer2 = factory.create(
                project_name="project-2",
                enable_monitoring=True,
                enable_evaluation=False,
                isolated=True,
            )

        @tracer1.observe(span_type="function")
        def function_for_tracer1():
            return "tracer1-result"

        @tracer2.observe(span_type="function")
        def function_for_tracer2():
            return "tracer2-result"

        # Both should work independently
        result1 = function_for_tracer1()
        result2 = function_for_tracer2()

        assert result1 == "tracer1-result"
        assert result2 == "tracer2-result"

    def test_tracer_with_different_configs(self, mock_client):
        """
        Tracers with different configurations should work independently.
        """
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            factory = TracerFactory(mock_client)

            tracer_enabled = factory.create(
                project_name="enabled",
                enable_monitoring=True,
                enable_evaluation=False,
                isolated=True,
            )

            tracer_disabled = factory.create(
                project_name="disabled",
                enable_monitoring=False,
                enable_evaluation=False,
                isolated=True,
            )

        @tracer_enabled.observe(span_type="function")
        def enabled_function():
            return "enabled"

        @tracer_disabled.observe(span_type="function")
        def disabled_function():
            return "disabled"

        # Both should return correct results
        assert enabled_function() == "enabled"
        assert disabled_function() == "disabled"


@pytest.mark.reliability
class TestUnusualFunctionSignatures:
    """Test tracing functions with unusual signatures."""

    def test_function_with_args_kwargs(self, tracer: Tracer):
        """
        Functions with *args and **kwargs should work.
        """

        @tracer.observe(span_type="function")
        def variadic_function(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

        result = variadic_function(1, 2, 3, a="x", b="y")

        assert result["args"] == (1, 2, 3)
        assert result["kwargs"] == {"a": "x", "b": "y"}

    def test_function_with_default_args(self, tracer: Tracer):
        """
        Functions with default arguments should work.
        """

        @tracer.observe(span_type="function")
        def function_with_defaults(a, b=10, c="default"):
            return f"{a}-{b}-{c}"

        assert function_with_defaults(1) == "1-10-default"
        assert function_with_defaults(1, 20) == "1-20-default"
        assert function_with_defaults(1, 20, "custom") == "1-20-custom"

    def test_lambda_function(self, tracer: Tracer):
        """
        Lambda functions should be traceable.
        """
        traced_lambda = tracer.observe(lambda x: x * 2, span_type="function")

        result = traced_lambda(5)

        assert result == 10

    def test_method_on_class(self, tracer: Tracer):
        """
        Methods on classes should be traceable.
        """

        class MyClass:
            def __init__(self, value):
                self.value = value

            @tracer.observe(span_type="function")
            def process(self):
                return self.value * 2

        obj = MyClass(21)
        result = obj.process()

        assert result == 42

    def test_static_method(self, tracer: Tracer):
        """
        Static methods should be traceable.
        """

        class MyClass:
            @staticmethod
            @tracer.observe(span_type="function")
            def static_process(x):
                return x + 1

        result = MyClass.static_process(41)

        assert result == 42

    def test_class_method(self, tracer: Tracer):
        """
        Class methods should be traceable.
        """

        class MyClass:
            multiplier = 2

            @classmethod
            @tracer.observe(span_type="function")
            def class_process(cls, x):
                return x * cls.multiplier

        result = MyClass.class_process(21)

        assert result == 42
