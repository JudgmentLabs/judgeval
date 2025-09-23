from __future__ import annotations
from typing import TypeVar, Dict, cast, Type

T = TypeVar("T")


class SingletonMeta(type):
    """
    Metaclass for creating singleton classes.
    """

    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs) -> object:
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super(SingletonMeta, cls).__call__(
                *args, **kwargs
            )
        return SingletonMeta._instances[cls]

    def get_instance(cls: Type[T]) -> T | None:
        """Get the singleton instance if it exists, otherwise return None"""
        instance = SingletonMeta._instances.get(cls, None)
        return cast(T, instance) if instance is not None else None


__all__ = ("SingletonMeta",)
