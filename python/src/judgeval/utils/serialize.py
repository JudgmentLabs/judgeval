"""

This is a modified version of https://docs.powertools.aws.dev/lambda/python/2.35.1/api/event_handler/openapi/encoders.html

"""

import dataclasses
import datetime
from collections import defaultdict, deque
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from re import Pattern
from types import GeneratorType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import UUID

from pydantic import BaseModel
from pydantic.types import SecretBytes, SecretStr
import orjson

from judgeval.logger import judgeval_logger


"""
This module contains the encoders used by jsonable_encoder to convert Python objects to JSON serializable data types.
"""


def _model_dump(
    model: BaseModel, mode: Literal["json", "python"] = "json", **kwargs: Any
) -> Any:
    return model.model_dump(mode=mode, **kwargs)


def json_encoder(
    obj: Any,
    custom_serializer: Optional[Callable[[Any], str]] = None,
) -> Any:
    """
    JSON encodes an arbitrary Python object into JSON serializable data types.

    This is a modified version of fastapi.encoders.jsonable_encoder that supports
    encoding of pydantic.BaseModel objects.

    Parameters
    ----------
    obj : Any
        The object to encode
    custom_serializer : Callable, optional
        A custom serializer to use for encoding the object, when everything else fails.

    Returns
    -------
    Any
        The JSON serializable data types
    """
    # Pydantic models
    if isinstance(obj, BaseModel):
        return _dump_base_model(
            obj=obj,
        )

    # Dataclasses
    if dataclasses.is_dataclass(obj):
        obj_dict = dataclasses.asdict(obj)  # type: ignore[arg-type]
        return json_encoder(
            obj_dict,
        )

    # Enums
    if isinstance(obj, Enum):
        return obj.value

    # Paths
    if isinstance(obj, PurePath):
        return str(obj)

    # Scalars
    if isinstance(obj, (str, int, float, type(None))):
        return obj

    # Dictionaries
    if isinstance(obj, dict):
        return _dump_dict(
            obj=obj,
        )

    # Sequences
    if isinstance(obj, (list, set, frozenset, tuple, deque)):
        return _dump_sequence(
            obj=obj,
        )

    # Other types
    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)

    for encoder, classes_tuple in encoders_by_class_tuples.items():
        if isinstance(obj, classes_tuple):
            return encoder(obj)

    # Use custom serializer if present
    if custom_serializer:
        return custom_serializer(obj)

    # Default
    return _dump_other(
        obj=obj,
    )


def _dump_base_model(
    *,
    obj: Any,
):
    """
    Dump a BaseModel object to a dict, using the same parameters as jsonable_encoder
    """
    obj_dict = _model_dump(
        obj,
        mode="json",
    )
    if "__root__" in obj_dict:
        obj_dict = obj_dict["__root__"]

    return json_encoder(
        obj_dict,
    )


def _dump_dict(
    *,
    obj: Any,
) -> Dict[str, Any]:
    """
    Dump a dict to a dict, using the same parameters as jsonable_encoder
    """
    encoded_dict = {}
    allowed_keys = set(obj.keys())
    for key, value in obj.items():
        if key in allowed_keys:
            encoded_key = json_encoder(
                key,
            )
            encoded_value = json_encoder(
                value,
            )
            encoded_dict[encoded_key] = encoded_value
    return encoded_dict


def _dump_sequence(
    *,
    obj: Any,
) -> List[Any]:
    """
    Dump a sequence to a list, using the same parameters as jsonable_encoder
    """
    encoded_list = []
    for item in obj:
        encoded_list.append(
            json_encoder(
                item,
            ),
        )
    return encoded_list


def _dump_other(
    *,
    obj: Any,
) -> Any:
    """
    Dump an object to a representation without iterating it.

    Avoids calling dict(obj) which can consume iterators/generators or
    invoke user-defined iteration protocols.
    """
    try:
        return repr(obj)
    except Exception:
        return str(obj)


def iso_format(o: Union[datetime.date, datetime.time]) -> str:
    """
    ISO format for date and time
    """
    return o.isoformat()


def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
    """
    Encodes a Decimal as int of there's no exponent, otherwise float

    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
    where an integer (but not int typed) is used. Encoding this as a float
    results in failed round-tripping between encode and parse.

    >>> decimal_encoder(Decimal("1.0"))
    1.0

    >>> decimal_encoder(Decimal("1"))
    1
    """
    if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]
        return int(dec_value)
    else:
        return float(dec_value)


ENCODERS_BY_TYPE: Dict[Type[Any], Callable[[Any], Any]] = {
    bytes: lambda o: o.decode(),
    datetime.date: iso_format,
    datetime.datetime: iso_format,
    datetime.time: iso_format,
    datetime.timedelta: lambda td: td.total_seconds(),
    Decimal: decimal_encoder,
    Enum: lambda o: o.value,
    frozenset: list,
    deque: list,
    GeneratorType: repr,
    Path: str,
    Pattern: lambda o: o.pattern,
    SecretBytes: str,
    SecretStr: str,
    set: list,
    UUID: str,
}


# Generates a mapping of encoders to a tuple of classes that they can encode
def generate_encoders_by_class_tuples(
    type_encoder_map: Dict[Any, Callable[[Any], Any]],
) -> Dict[Callable[[Any], Any], Tuple[Any, ...]]:
    encoders: Dict[Callable[[Any], Any], Tuple[Any, ...]] = defaultdict(tuple)
    for type_, encoder in type_encoder_map.items():
        encoders[encoder] += (type_,)
    return encoders


# Mapping of encoders to a tuple of classes that they can encode
encoders_by_class_tuples = generate_encoders_by_class_tuples(ENCODERS_BY_TYPE)


# Seralize arbitrary object to a json string
def safe_serialize(obj: Any) -> str:
    try:
        return orjson.dumps(json_encoder(obj), option=orjson.OPT_NON_STR_KEYS).decode()
    except Exception as e:
        judgeval_logger.warning(f"Error serializing object: {e}")
        return repr(obj)
