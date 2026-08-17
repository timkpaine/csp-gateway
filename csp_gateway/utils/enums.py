from enum import Enum
from typing import Annotated, Any, get_args, get_origin

from pydantic import BeforeValidator, PlainSerializer

__all__ = ("ReadWriteMode", "coerce_basket_key", "enum_by_name")


class ReadWriteMode(str, Enum):
    """Enum representing whether a component is set to read, write, or both."""

    READ = "READ"
    WRITE = "WRITE"
    READ_AND_WRITE = "READ_AND_WRITE"


def enum_by_name(key_type: type) -> Any:
    """``key_type``, represented on the wire by member *name* in both directions.

    Dict-basket keys are published by name over REST, websockets and JSON snapshots. Pydantic instead
    matches an enum on its *value* -- fine for a str-valued enum, wrong for every other kind, and for
    a dict key it round trips as the stringified value (``A = 1`` writes ``"1"``, which then fails to
    validate). Non-enum key types are returned unchanged.
    """
    if not (isinstance(key_type, type) and issubclass(key_type, Enum)):
        return key_type

    def _from_name(value: Any) -> Any:
        # Fall through on a miss so a raw member value still validates the normal way.
        return key_type.__members__.get(value, value) if isinstance(value, str) else value

    return Annotated[
        key_type,
        BeforeValidator(_from_name),
        PlainSerializer(lambda v: v.name if isinstance(v, Enum) else v, return_type=str),
    ]


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
