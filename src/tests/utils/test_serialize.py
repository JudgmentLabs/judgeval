from __future__ import annotations

import dataclasses
import datetime
import re
from collections import deque
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from uuid import UUID

import pytest
from pydantic import BaseModel
from pydantic.types import SecretBytes, SecretStr

from judgeval.utils.serialize import (
    decimal_encoder,
    generate_encoders_by_class_tuples,
    iso_format,
    json_encoder,
    safe_serialize,
    serialize_attribute,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Color(Enum):
    RED = "red"
    BLUE = 42


class Inner(BaseModel):
    x: int


class Outer(BaseModel):
    inner: Inner
    label: str


@dataclasses.dataclass
class Point:
    x: int
    y: int


# ---------------------------------------------------------------------------
# json_encoder — scalars & primitives
# ---------------------------------------------------------------------------


def test_none_passthrough():
    assert json_encoder(None) is None


def test_string_passthrough():
    assert json_encoder("hello") == "hello"


def test_int_passthrough():
    assert json_encoder(42) == 42


def test_float_passthrough():
    assert json_encoder(3.14) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# json_encoder — BaseModel
# ---------------------------------------------------------------------------


def test_pydantic_model():
    result = json_encoder(Outer(inner=Inner(x=1), label="a"))
    assert result == {"inner": {"x": 1}, "label": "a"}


# ---------------------------------------------------------------------------
# json_encoder — dataclass
# ---------------------------------------------------------------------------


def test_dataclass():
    assert json_encoder(Point(3, 4)) == {"x": 3, "y": 4}


# ---------------------------------------------------------------------------
# json_encoder — Enum
# ---------------------------------------------------------------------------


def test_enum_string_value():
    assert json_encoder(Color.RED) == "red"


def test_enum_int_value():
    assert json_encoder(Color.BLUE) == 42


# ---------------------------------------------------------------------------
# json_encoder — PurePath / Path
# ---------------------------------------------------------------------------


def test_purepath():
    assert json_encoder(PurePath("/a/b")) == "/a/b"


def test_path():
    assert json_encoder(Path("/tmp/x")) == "/tmp/x"


# ---------------------------------------------------------------------------
# json_encoder — dict
# ---------------------------------------------------------------------------


def test_dict_basic():
    assert json_encoder({"a": 1, "b": "two"}) == {"a": 1, "b": "two"}


def test_dict_nested():
    assert json_encoder({"m": Inner(x=7)}) == {"m": {"x": 7}}


# ---------------------------------------------------------------------------
# json_encoder — sequences
# ---------------------------------------------------------------------------


def test_list():
    assert json_encoder([1, "two", None]) == [1, "two", None]


def test_tuple():
    assert json_encoder((1, 2)) == [1, 2]


def test_set_returns_list():
    result = json_encoder({1})
    assert result == [1]


def test_frozenset_returns_list():
    result = json_encoder(frozenset({2}))
    assert result == [2]


def test_deque():
    result = json_encoder(deque([1, 2]))
    assert result == [1, 2]


# ---------------------------------------------------------------------------
# json_encoder — ENCODERS_BY_TYPE entries
# ---------------------------------------------------------------------------


def test_bytes():
    assert json_encoder(b"hello") == "hello"


def test_date():
    d = datetime.date(2024, 1, 15)
    assert json_encoder(d) == "2024-01-15"


def test_datetime():
    dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
    assert json_encoder(dt) == dt.isoformat()


def test_time():
    t = datetime.time(12, 0, 0)
    assert json_encoder(t) == "12:00:00"


def test_timedelta():
    td = datetime.timedelta(seconds=90)
    assert json_encoder(td) == 90.0


def test_decimal_integer():
    assert json_encoder(Decimal("5")) == 5


def test_decimal_float():
    assert json_encoder(Decimal("3.14")) == pytest.approx(3.14)


def test_pattern():
    p = re.compile(r"\d+")
    assert json_encoder(p) == r"\d+"


def test_secret_str():
    result = json_encoder(SecretStr("hidden"))
    assert isinstance(result, str)


def test_secret_bytes():
    result = json_encoder(SecretBytes(b"hidden"))
    assert isinstance(result, str)


def test_uuid():
    u = UUID("12345678-1234-5678-1234-567812345678")
    assert json_encoder(u) == "12345678-1234-5678-1234-567812345678"


def test_generator_type():
    def gen():
        yield 1

    result = json_encoder(gen())
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# json_encoder — custom_serializer fallback
# ---------------------------------------------------------------------------


def test_custom_serializer_called_for_unknown():
    class Unknown:
        pass

    result = json_encoder(Unknown(), custom_serializer=lambda o: "custom")
    assert result == "custom"


# ---------------------------------------------------------------------------
# json_encoder — _dump_other repr fallback
# ---------------------------------------------------------------------------


def test_unknown_type_uses_repr():
    class MyObj:
        def __repr__(self):
            return "MyObj()"

    assert json_encoder(MyObj()) == "MyObj()"


def test_subclass_of_known_type_uses_class_tuple_fallback():
    class MyDatetime(datetime.datetime):
        pass

    obj = MyDatetime(2024, 1, 1)
    result = json_encoder(obj)
    assert isinstance(result, str)


def test_dump_other_falls_back_to_str_when_repr_raises():
    from judgeval.utils.serialize import _dump_other

    class BadRepr:
        def __repr__(self):
            raise RuntimeError("repr failed")

        def __str__(self):
            return "str_fallback"

    assert _dump_other(obj=BadRepr()) == "str_fallback"


def test_safe_serialize_falls_back_to_repr_on_orjson_failure():
    from unittest.mock import patch

    class Obj:
        def __repr__(self):
            return "Obj()"

    with patch("judgeval.utils.serialize.orjson.dumps", side_effect=Exception("fail")):
        result = safe_serialize(Obj())
    assert result == "Obj()"


# ---------------------------------------------------------------------------
# iso_format
# ---------------------------------------------------------------------------


def test_iso_format_date():
    assert iso_format(datetime.date(2024, 6, 1)) == "2024-06-01"


def test_iso_format_time():
    assert iso_format(datetime.time(8, 30)) == "08:30:00"


# ---------------------------------------------------------------------------
# decimal_encoder
# ---------------------------------------------------------------------------


def test_decimal_encoder_integer():
    assert decimal_encoder(Decimal("10")) == 10
    assert isinstance(decimal_encoder(Decimal("10")), int)


def test_decimal_encoder_float():
    result = decimal_encoder(Decimal("10.5"))
    assert isinstance(result, float)
    assert result == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# generate_encoders_by_class_tuples
# ---------------------------------------------------------------------------


def test_generate_encoders_groups_same_encoder():
    enc = lambda o: str(o)  # noqa: E731
    mapping = generate_encoders_by_class_tuples({int: enc, float: enc})
    assert int in mapping[enc]
    assert float in mapping[enc]


def test_generate_encoders_different_encoders():
    enc1 = lambda o: "a"  # noqa: E731
    enc2 = lambda o: "b"  # noqa: E731
    mapping = generate_encoders_by_class_tuples({int: enc1, str: enc2})
    assert int in mapping[enc1]
    assert str in mapping[enc2]


# ---------------------------------------------------------------------------
# safe_serialize
# ---------------------------------------------------------------------------


def test_safe_serialize_basic():
    result = safe_serialize({"key": "value"})
    assert result == '{"key":"value"}'


def test_safe_serialize_pydantic():
    result = safe_serialize(Inner(x=5))
    assert '"x":5' in result


def test_safe_serialize_fallback_on_error():
    class Unserializable:
        def __repr__(self):
            return "Unserializable()"

        def __str__(self):
            return "Unserializable()"

    result = safe_serialize(Unserializable())
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# serialize_attribute
# ---------------------------------------------------------------------------


def test_serialize_attribute_str_passthrough():
    assert serialize_attribute("hello") == "hello"


def test_serialize_attribute_int_passthrough():
    assert serialize_attribute(42) == 42


def test_serialize_attribute_float_passthrough():
    assert serialize_attribute(3.14) == pytest.approx(3.14)


def test_serialize_attribute_bool_passthrough():
    assert serialize_attribute(True) is True


def test_serialize_attribute_non_primitive_calls_serializer():
    sentinel = object()
    called_with = []

    def custom(obj):
        called_with.append(obj)
        return "serialized"

    result = serialize_attribute(sentinel, serializer=custom)
    assert result == "serialized"
    assert called_with[0] is sentinel


def test_serialize_attribute_dict_uses_safe_serialize():
    result = serialize_attribute({"a": 1})
    assert isinstance(result, str)
    assert "a" in result
