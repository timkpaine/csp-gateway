"""Tests for the dict-basket key helpers.

Enum basket keys are published by member *name* -- in REST paths, in websocket messages and in JSON
snapshots -- so they have to be read back by name. Pydantic's own behaviour is to match an enum on its
*value*, which for ``A = 1`` is the int ``1``, and to write a dict key as that value.
"""

from enum import Enum

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from csp_gateway.utils import coerce_basket_key, enum_by_name


class Key(Enum):
    A = 1
    B = 2


def test_validates_by_name():
    assert TypeAdapter(enum_by_name(Key)).validate_python("A") is Key.A


def test_validates_by_value_too():
    # A miss on the name falls through, so a raw member value still works.
    assert TypeAdapter(enum_by_name(Key)).validate_python(1) is Key.A


def test_rejects_an_unknown_name():
    with pytest.raises(ValidationError):
        TypeAdapter(enum_by_name(Key)).validate_python("nope")


def test_serializes_by_name():
    assert TypeAdapter(enum_by_name(Key)).dump_python(Key.B) == "B"


def test_dict_keys_round_trip_through_json():
    class Model(BaseModel):
        basket: dict[enum_by_name(Key), int]

    encoded = Model(basket={Key.A: 7}).model_dump_json()
    assert '"A"' in encoded  # not the stringified value "1"
    assert Model.model_validate_json(encoded).basket == {Key.A: 7}


def test_non_enum_key_type_is_untouched():
    assert enum_by_name(str) is str


def test_coerce_basket_key_resolves_a_name():
    assert coerce_basket_key(Key, "A") is Key.A


def test_coerce_basket_key_accepts_the_annotated_form():
    # The web layer stores the enum_by_name() annotation, not the bare enum.
    assert coerce_basket_key(enum_by_name(Key), "B") is Key.B


def test_coerce_basket_key_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="not a valid Key"):
        coerce_basket_key(Key, "nope")


def test_coerce_basket_key_passes_through_a_plain_type():
    assert coerce_basket_key(str, "key1") == "key1"
