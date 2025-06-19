import json
from abc import ABC
from typing import Any, ClassVar

from pydantic import BaseModel

"""
Motivation for this file: Pydantic2 does not support dynamic discriminated unions.
Use case for ERT: User defines a custom response config, and registers it as a plugin
to ERT. Instances of this response config then needs to be (1) a subclass of
ResponseConfig, and (2) serializable via pydantic .model_validate. This class will
enable such dynamic serialization/deserialization.
"""


class WithDynamicDiscriminator(BaseModel, ABC):
    type: str
    _registry: ClassVar[dict[str, type["WithDynamicDiscriminator"]]] = {}

    @classmethod
    def _dispatch_validate(
        cls, raw: Any, *, is_json: bool = False
    ) -> "WithDynamicDiscriminator":
        # Parse JSON if needed
        obj = json.loads(raw) if is_json else raw

        if not isinstance(obj, dict):
            raise TypeError("Expected dict-like input")

        type_value = obj.get("type")
        if type_value not in cls._registry:
            raise ValueError(f"Unknown type '{type_value}' for {cls.__name__}")

        subclass = cls._registry[type_value]

        if is_json:
            # Call model_validate_json on subclass's base class, skip dispatch
            return super(WithDynamicDiscriminator, subclass).model_validate_json(raw)
        else:
            # Call model_validate on subclass's base class, skip dispatch
            return super(WithDynamicDiscriminator, subclass).model_validate(raw)

    @classmethod
    def model_validate_subclass(cls, obj: Any) -> "WithDynamicDiscriminator":
        return cls._dispatch_validate(obj, is_json=False)

    @classmethod
    def model_validate_subclass_json(cls, json_str: str) -> "WithDynamicDiscriminator":
        return cls._dispatch_validate(json_str, is_json=True)


def make_registry_decorator(base_cls: type["WithDynamicDiscriminator"]):
    """
    Creates a decorator that can be used to annotate subclasses that should be
    registered onto superclass. For example, if adding a ResponseConfig subclass,
    annotating this decorator in the subclass will make it pydantic
    serializable/deserializable.
    """

    def decorator(cls: type["WithDynamicDiscriminator"]):
        type_value = cls.model_fields.get("type") and cls.model_fields["type"].default
        if not type_value:
            raise ValueError(f"{cls.__name__} must define a default 'type' field")
        base_cls._registry[type_value] = cls
        return cls

    return decorator
