from typing import (
    Callable,
)

class BaseFactory:
    _registry = {}

    @classmethod
    def register(cls, mode: str) -> Callable:
        def decorator(subclass):
            cls._registry[mode] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, mode: str, *args, **kwargs):
        return cls._registry[mode](*args, **kwargs)
