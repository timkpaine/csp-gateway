from enum import Enum
from typing import Annotated, Any, get_args, get_origin

from pydantic import BeforeValidator

__all__ = ("ReadWriteMode", "coerce_basket_key", "enum_by_name")


class ReadWriteMode(str, Enum):
    """Enum representing whether a component is set to read, write, or both."""

    READ = "READ"
    WRITE = "WRITE"
    READ_AND_WRITE = "READ_AND_WRITE"


def enum_by_name(key_type: type) -> Any:
    """``key_type``, additionally accepting a member *name* as input.

    Dict-basket keys are published by name over REST and websockets, so they have to be read back by
    name. Pydantic matches an enum on its *value*, which for ``A = 1`` is the int ``1`` -- fine for a
    str-valued enum, wrong for every other kind. Non-enum key types are returned unchanged.
    """
    if not (isinstance(key_type, type) and issubclass(key_type, Enum)):
        return key_type
    by_name = BeforeValidator(lambda v: key_type.__members__.get(v, v) if isinstance(v, str) else v)
    return Annotated[key_type, by_name]


def coerce_basket_key(key_type: type, key: str) -> Any:
    """Turn a dict-basket key from a URL into the value the basket is keyed by.

    Enum keys appear in URLs as member *names*, which is how they are published. Calling the enum
    would look the name up as a value and fail for anything but a str-valued enum.
    """
    # The web layer annotates basket keys with enum_by_name, so unwrap that before inspecting.
    while get_origin(key_type) is Annotated:
        key_type = get_args(key_type)[0]
    if isinstance(key_type, type) and issubclass(key_type, Enum):
        try:
            return key_type[key]
        except KeyError:
            raise ValueError(f"{key!r} is not a valid {key_type.__name__}") from None
    return key_type(key)
