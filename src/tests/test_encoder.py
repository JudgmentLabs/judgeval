import datetime
import uuid
from collections import defaultdict
from enum import Enum

import orjson
from pydantic import BaseModel

from judgeval.utils.serialize import json_encoder, safe_serialize


class SimpleModel(BaseModel):
    id: int
    name: str


def test_basic_serialization():
    data = {"a": 1, "b": "string", "c": [1, 2, 3]}
    result = json_encoder(data)
    assert result == data


def test_pydantic_model_serialization():
    model = SimpleModel(id=1, name="Test")
    result = json_encoder(model)
    assert result == {"id": 1, "name": "Test"}


def test_unserializable_builtin_function():
    result = json_encoder(print)
    assert isinstance(result, str)
    assert "built-in function print" in result


def test_unserializable_builtin_class():
    result = json_encoder(defaultdict)
    assert isinstance(result, str)
    assert "class" in result and "collections.defaultdict" in result


def test_function_wrapped_in_dict():
    obj = {"key": print}
    result = json_encoder(obj)
    assert isinstance(result["key"], str)
    assert "built-in function print" in result["key"]


# Tests for non-string keys with OPT_NON_STR_KEYS


class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


def test_integer_keys():
    """Test that integer keys are preserved and properly serialized."""
    data = {1: "one", 2: "two", 3: "three"}
    result = json_encoder(data)
    assert result == {1: "one", 2: "two", 3: "three"}
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    # orjson converts int keys to strings in JSON
    assert decoded == {"1": "one", "2": "two", "3": "three"}


def test_float_keys():
    """Test that float keys are preserved and properly serialized."""
    data = {1.5: "one point five", 2.7: "two point seven"}
    result = json_encoder(data)
    assert result == {1.5: "one point five", 2.7: "two point seven"}
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert decoded == {"1.5": "one point five", "2.7": "two point seven"}


def test_boolean_keys():
    """Test that boolean keys are preserved and properly serialized."""
    data = {True: "yes", False: "no"}
    result = json_encoder(data)
    assert result == {True: "yes", False: "no"}
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert decoded == {"true": "yes", "false": "no"}


def test_none_key():
    """Test that None key is preserved and properly serialized."""
    data = {None: "null value"}
    result = json_encoder(data)
    assert result == {None: "null value"}
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert decoded == {"null": "null value"}


def test_datetime_keys():
    """Test that datetime keys are preserved and properly serialized."""
    dt = datetime.datetime(2023, 10, 15, 12, 30, 45)
    date = datetime.date(2023, 10, 15)
    time = datetime.time(12, 30, 45)
    
    data = {
        dt: "datetime value",
        date: "date value",
        time: "time value"
    }
    result = json_encoder(data)
    
    # Keys are converted to ISO format strings by json_encoder
    assert "2023-10-15T12:30:45" in result
    assert "2023-10-15" in result
    assert "12:30:45" in result
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    # orjson converts datetime keys to ISO format strings
    assert "2023-10-15T12:30:45" in decoded
    assert "2023-10-15" in decoded
    assert "12:30:45" in decoded


def test_uuid_keys():
    """Test that UUID keys are preserved and properly serialized."""
    uuid1 = uuid.UUID("7202d115-7ff3-4c81-a7c1-2a1f067b1ece")
    uuid2 = uuid.UUID("12345678-1234-5678-1234-567812345678")
    
    data = {uuid1: "first uuid", uuid2: "second uuid"}
    result = json_encoder(data)
    
    # UUIDs should be converted to strings during json_encoder
    assert result == {str(uuid1): "first uuid", str(uuid2): "second uuid"}
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert decoded == {
        "7202d115-7ff3-4c81-a7c1-2a1f067b1ece": "first uuid",
        "12345678-1234-5678-1234-567812345678": "second uuid"
    }


def test_enum_keys():
    """Test that Enum keys are preserved and properly serialized."""
    data = {Status.ACTIVE: "running", Status.INACTIVE: "stopped"}
    result = json_encoder(data)
    
    # Enums should be converted to their values
    assert result == {"active": "running", "inactive": "stopped"}
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert decoded == {"active": "running", "inactive": "stopped"}


def test_mixed_key_types():
    """Test that mixed key types work together."""
    # Note: In Python, True == 1, so they are the same dict key
    # Using 2 instead of True to avoid key collision
    data = {
        "string": "str value",
        1: "int value",
        2.5: "float value",
        2: "int value 2",
        None: "none value"
    }
    result = json_encoder(data)
    assert "string" in result
    assert 1 in result
    assert 2.5 in result
    assert 2 in result
    assert None in result
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    # JSON will have all keys as strings
    assert "string" in decoded
    assert "1" in decoded
    assert "2.5" in decoded
    assert "2" in decoded
    assert "null" in decoded


def test_nested_dicts_with_non_string_keys():
    """Test that nested dictionaries with non-string keys work properly."""
    data = {
        1: {
            2: "nested int",
            "inner": "value"
        },
        "outer": {
            3.14: "pi",
            True: "truth"
        }
    }
    result = json_encoder(data)
    assert 1 in result
    assert 2 in result[1]
    assert "outer" in result
    assert 3.14 in result["outer"]
    assert True in result["outer"]
    
    # Verify it can be serialized with orjson
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert "1" in decoded
    assert "2" in decoded["1"]
    assert "outer" in decoded
    assert "3.14" in decoded["outer"]


def test_safe_serialize_with_integer_keys():
    """Test safe_serialize function specifically with integer keys."""
    data = {1: "one", 2: "two", 3: {"nested": "value", 4: "four"}}
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert "1" in decoded
    assert "2" in decoded
    assert "3" in decoded
    assert "4" in decoded["3"]


def test_safe_serialize_with_complex_values():
    """Test safe_serialize with non-string keys and complex values."""
    model = SimpleModel(id=1, name="Test")
    data = {
        1: model,
        2: [1, 2, 3],
        3: {"nested": "dict"},
        4: datetime.datetime(2023, 10, 15, 12, 30)
    }
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert "1" in decoded
    assert decoded["1"]["id"] == 1
    assert decoded["2"] == [1, 2, 3]
    assert decoded["3"] == {"nested": "dict"}


def test_safe_serialize_preserves_string_keys():
    """Test that string keys still work as expected."""
    data = {"a": 1, "b": 2, "c": 3}
    json_str = safe_serialize(data)
    assert json_str is not None
    decoded = orjson.loads(json_str)
    assert decoded == {"a": 1, "b": 2, "c": 3}
