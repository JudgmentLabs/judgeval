"""Tests for serialization utilities."""

import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List

import pytest

from judgeval.utils.serialize import json_encoder, safe_serialize


class TestColor(Enum):
    RED = "red"
    BLUE = "blue"


class TestDictSerialization:
    """Test dictionary serialization with various key types."""

    def test_dict_with_int_keys(self):
        """Test that dicts with integer keys are properly serialized."""
        obj = {1: "a", 2: "b", 3: "c"}
        result = json_encoder(obj)
        assert result == {"1": "a", "2": "b", "3": "c"}

    def test_dict_with_int_keys_and_list_values(self):
        """Test Dict[int, List[str]] serialization."""
        obj: Dict[int, List[str]] = {
            1: ["hello", "world"],
            2: ["foo", "bar"],
            3: ["baz", "qux"],
        }
        result = json_encoder(obj)
        assert result == {
            "1": ["hello", "world"],
            "2": ["foo", "bar"],
            "3": ["baz", "qux"],
        }

    def test_dict_with_mixed_value_types(self):
        """Test dict with int keys and mixed value types."""
        obj = {
            1: ["hello", "world"],
            2: 42,
            3: {"nested": "dict"},
            4: None,
        }
        result = json_encoder(obj)
        assert result == {
            "1": ["hello", "world"],
            "2": 42,
            "3": {"nested": "dict"},
            "4": None,
        }

    def test_dict_with_float_keys(self):
        """Test that dicts with float keys are properly serialized."""
        obj = {1.5: "a", 2.7: "b"}
        result = json_encoder(obj)
        assert result == {"1.5": "a", "2.7": "b"}

    def test_dict_with_bool_keys(self):
        """Test that dicts with boolean keys are properly serialized."""
        obj = {True: "yes", False: "no"}
        result = json_encoder(obj)
        assert result == {"True": "yes", "False": "no"}

    def test_dict_with_string_keys(self):
        """Test that dicts with string keys work as before."""
        obj = {"a": "value1", "b": "value2"}
        result = json_encoder(obj)
        assert result == {"a": "value1", "b": "value2"}

    def test_nested_dict_with_int_keys(self):
        """Test nested dicts with int keys."""
        obj = {
            1: {
                10: "nested_a",
                20: "nested_b",
            },
            2: {
                30: "nested_c",
            },
        }
        result = json_encoder(obj)
        assert result == {
            "1": {
                "10": "nested_a",
                "20": "nested_b",
            },
            "2": {
                "30": "nested_c",
            },
        }

    def test_safe_serialize_with_int_keys(self):
        """Test that safe_serialize works with int keys."""
        obj: Dict[int, List[str]] = {
            1: ["hello", "world"],
            2: ["foo", "bar"],
        }
        result = safe_serialize(obj)
        # Should not raise an exception and should produce valid JSON
        assert '"1"' in result
        assert '"2"' in result
        assert "hello" in result
        assert "world" in result


class TestGeneralSerialization:
    """Test general serialization functionality."""

    def test_serialize_enum(self):
        """Test enum serialization."""
        obj = {"color": TestColor.RED}
        result = json_encoder(obj)
        assert result == {"color": "red"}

    def test_serialize_datetime(self):
        """Test datetime serialization."""
        dt = datetime.datetime(2023, 1, 1, 12, 0, 0)
        result = json_encoder(dt)
        assert result == "2023-01-01T12:00:00"

    def test_serialize_decimal(self):
        """Test decimal serialization."""
        obj = {"price": Decimal("19.99")}
        result = json_encoder(obj)
        assert result == {"price": 19.99}

    def test_serialize_none(self):
        """Test None serialization."""
        result = json_encoder(None)
        assert result is None

    def test_serialize_list(self):
        """Test list serialization."""
        obj = [1, 2, 3, "a", "b"]
        result = json_encoder(obj)
        assert result == [1, 2, 3, "a", "b"]

    def test_serialize_tuple(self):
        """Test tuple serialization."""
        obj = (1, 2, 3)
        result = json_encoder(obj)
        assert result == [1, 2, 3]

    def test_serialize_set(self):
        """Test set serialization."""
        obj = {1, 2, 3}
        result = json_encoder(obj)
        assert sorted(result) == [1, 2, 3]

    def test_serialize_complex_nested_structure(self):
        """Test complex nested structure with int keys."""
        obj = {
            "users": {
                1: {
                    "name": "Alice",
                    "tags": ["admin", "developer"],
                    "settings": {
                        100: "value1",
                        200: "value2",
                    },
                },
                2: {
                    "name": "Bob",
                    "tags": ["user"],
                    "settings": {
                        300: "value3",
                    },
                },
            },
        }
        result = json_encoder(obj)
        assert result == {
            "users": {
                "1": {
                    "name": "Alice",
                    "tags": ["admin", "developer"],
                    "settings": {
                        "100": "value1",
                        "200": "value2",
                    },
                },
                "2": {
                    "name": "Bob",
                    "tags": ["user"],
                    "settings": {
                        "300": "value3",
                    },
                },
            },
        }
